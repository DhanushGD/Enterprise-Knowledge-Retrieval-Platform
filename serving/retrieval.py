"""Step 9: retrieval = permission filter -> hybrid search -> rerank -> diversity cut.

    query ──embed──► dense vector + sparse (BM25) vector
                         │
          Qdrant: dense search  ┐   both restricted to what the caller's ROLE may see,
                  sparse search ┘   fused with Reciprocal Rank Fusion (RRF)
                         │
              top `candidates` (default 30)
                         │
          cross-encoder reranker re-scores (query, passage) pairs
                         │
          keep at most `max_per_doc` chunks per document → top `top_k`

Design rules:
  * Permissions are enforced INSIDE the vector search (a filter on every prefetch), never by
    dropping results afterwards, so restricted text can't leak through a bug in post-filtering
    and top-k isn't wasted on chunks the user can't see.
  * Fail closed: an unknown role, or a role with an empty allow-list, gets nothing.
  * Fail soft on the reranker: if it errors, return the fused order and say so in `degraded`.
  * Heavy dependencies (qdrant, fastembed) are imported lazily so the logic is testable
    with in-memory fakes.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # noisy and harmless on Windows

log = logging.getLogger("rag.retrieval")

MODES = ("dense", "sparse", "hybrid", "hybrid_rerank")


class UnknownRoleError(KeyError):
    """The caller's role isn't defined in config/company.yaml."""


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    source_path: str
    text: str
    access_level: str
    department: str
    product: str
    page: int | None
    score: float                          # retrieval score (RRF, cosine or BM25 by mode)
    rerank_score: float | None = None     # cross-encoder score, when reranked

    @property
    def location(self) -> str:
        return f"{self.doc_title} > {self.section}" if self.section else self.doc_title

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score


@dataclass
class RetrievalResult:
    query: str
    role: str
    mode: str
    hits: list[Hit]
    timings: dict[str, float] = field(default_factory=dict)   # milliseconds per stage
    candidates: int = 0                                         # how many entered reranking
    degraded: str | None = None                                 # set when a stage fell back
    # Best dense cosine similarity among the chunks this role may see. Unlike RRF/BM25/rerank
    # scores it has a stable scale, so it can drive an "is anything relevant?" threshold.
    dense_top: float | None = None


def _hit(point) -> Hit:
    p = point.payload or {}
    return Hit(
        chunk_id=p.get("chunk_id", str(point.id)), doc_id=p.get("doc_id", ""),
        doc_title=p.get("doc_title", ""), section=p.get("section", ""),
        source_path=p.get("source_path", ""), text=p.get("text", ""),
        access_level=p.get("access_level", ""), department=p.get("department", ""),
        product=p.get("product", ""), page=p.get("page"), score=float(point.score),
    )


def passage_for_rerank(hit: Hit) -> str:
    """Same 'Title > Section' framing the chunk was embedded with."""
    return f"{hit.location}\n\n{hit.text}"


class FastEmbedReranker:
    """Cross-encoder via fastembed (ONNX, CPU). It reads query and passage *together*, which is
    far more accurate than comparing two independent vectors - but slower, so it only sees the
    few dozen candidates that survived the cheap first stage."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", cache_dir: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder   # lazy: heavy import

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, passages)]


class Retriever:
    def __init__(
        self,
        embedder,
        index,
        roles: dict,
        reranker=None,
        candidates: int = 30,
        top_k: int = 5,
        max_per_doc: int = 2,
    ):
        self.embedder = embedder
        self.index = index
        self.roles = roles
        self.reranker = reranker
        self.candidates = candidates
        self.top_k = top_k
        self.max_per_doc = max_per_doc

    # ------------------------------------------------------------------ public
    def retrieve(
        self, query: str, role: str, mode: str | None = None, top_k: int | None = None
    ) -> RetrievalResult:
        mode = mode or ("hybrid_rerank" if self.reranker else "hybrid")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode == "hybrid_rerank" and self.reranker is None:
            raise ValueError("mode 'hybrid_rerank' needs a reranker (enable it in config)")

        levels, departments = self._permissions(role)        # raises for unknown roles
        top_k = top_k or self.top_k
        result = RetrievalResult(query=query, role=role, mode=mode, hits=[])
        query = query.strip()
        if not query or not levels or not departments:        # fail closed
            return result

        began = time.perf_counter()
        t = time.perf_counter()
        embedding = self.embedder.embed_query(query)
        result.timings["embed_ms"] = _ms(t)

        t = time.perf_counter()
        pool = [_hit(p) for p in self._search(mode, embedding, levels, departments)]
        result.timings["search_ms"] = _ms(t)
        result.candidates = len(pool)
        result.dense_top = self._dense_top(mode, pool, embedding, levels, departments)

        if mode == "hybrid_rerank" and pool:
            t = time.perf_counter()
            try:
                scores = self.reranker.rerank(query, [passage_for_rerank(h) for h in pool])
                if len(scores) != len(pool):
                    raise RuntimeError(f"reranker returned {len(scores)} scores for {len(pool)} passages")
                for hit, score in zip(pool, scores):
                    hit.rerank_score = score
                pool.sort(key=lambda h: h.rerank_score, reverse=True)   # stable: ties keep RRF order
            except Exception as exc:                          # keep serving with the fused order
                result.degraded = f"reranker failed ({type(exc).__name__}: {exc}); using fused order"
                log.warning("%s", result.degraded)
                for hit in pool:
                    hit.rerank_score = None
            result.timings["rerank_ms"] = _ms(t)

        result.hits = self._diverse(pool, top_k)
        result.timings["total_ms"] = _ms(began)
        return result

    # ----------------------------------------------------------------- internals
    def _permissions(self, role: str) -> tuple[list[str], list[str]]:
        try:
            cfg = self.roles[role]
        except KeyError:
            raise UnknownRoleError(role) from None
        return list(cfg.get("allowed_access_levels") or []), list(cfg.get("allowed_departments") or [])

    def _search(self, mode, embedding, levels, departments):
        kw = dict(limit=self.candidates, access_levels=levels, departments=departments)
        if mode == "dense":
            return self.index.dense_search(embedding, **kw)
        if mode == "sparse":
            return self.index.sparse_search(embedding, **kw)
        return self.index.hybrid_search(embedding, **kw)

    def _dense_top(self, mode, pool, embedding, levels, departments) -> float | None:
        if mode == "dense":
            return pool[0].score if pool else None
        if mode == "sparse":
            return None
        try:                                        # one extra cheap query: top-1 dense hit
            top = self.index.dense_search(embedding, limit=1, access_levels=levels,
                                          departments=departments)
            return float(top[0].score) if top else None
        except Exception as exc:                    # the gate is optional; never fail retrieval
            log.warning("dense_top unavailable: %s", exc)
            return None

    def _diverse(self, pool: list[Hit], top_k: int) -> list[Hit]:
        """At most `max_per_doc` chunks from any one document, so one long page can't fill
        every slot with overlapping text."""
        per_doc: dict[str, int] = {}
        out: list[Hit] = []
        for hit in pool:
            if per_doc.get(hit.doc_id, 0) >= self.max_per_doc:
                continue
            per_doc[hit.doc_id] = per_doc.get(hit.doc_id, 0) + 1
            out.append(hit)
            if len(out) == top_k:
                break
        return out


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)


def build_retriever(rerank: bool | None = None, reranker_model: str | None = None,
                    candidates: int | None = None) -> Retriever:
    """Wire the real thing from config + env (Qdrant, fastembed, optional reranker)."""
    from core.settings import company, settings
    from core.vectorstore import QdrantIndex
    from ingestion.runner import get_embedder

    cfg = company.get("retrieval", {}) or {}
    use_rerank = cfg.get("rerank", True) if rerank is None else rerank
    reranker = (
        FastEmbedReranker(reranker_model or settings.reranker_model, settings.resolved_model_cache)
        if use_rerank else None
    )
    return Retriever(
        get_embedder(), QdrantIndex(), company["roles"], reranker,
        candidates=candidates or cfg.get("candidates", 30), top_k=cfg.get("top_k", 5),
        max_per_doc=cfg.get("max_per_doc", 2),
    )
