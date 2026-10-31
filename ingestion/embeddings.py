"""Local, free embeddings via fastembed (ONNX, no PyTorch needed).

Two vectors per text:
  dense  - BAAI/bge-small-en-v1.5, 384 dims: matches meaning ("get my money back" ~ "refund")
  sparse - BM25 term weights: matches exact words ("X-OpenDesk-Signature", error codes)
Using both is what makes hybrid retrieval (Step 9) beat either alone.
"""
from __future__ import annotations

import os

from ingestion.records import Embedding

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # noisy and harmless on Windows


class FastEmbedEmbedder:
    def __init__(
        self,
        dense_model: str = "BAAI/bge-small-en-v1.5",
        sparse_model: str = "Qdrant/bm25",
        cache_dir: str | None = None,
        batch_size: int = 32,
    ):
        from fastembed import SparseTextEmbedding, TextEmbedding   # lazy: heavy import

        self.batch_size = batch_size
        self._dense = TextEmbedding(model_name=dense_model, cache_dir=cache_dir)
        self._sparse = SparseTextEmbedding(model_name=sparse_model, cache_dir=cache_dir)
        self.dimension = len(next(iter(self._dense.embed(["dimension probe"]))))

    def embed_documents(self, texts: list[str]) -> list[Embedding]:
        dense = [v.tolist() for v in self._dense.embed(texts, batch_size=self.batch_size)]
        sparse = list(self._sparse.embed(texts, batch_size=self.batch_size))
        return [
            Embedding(d, s.indices.tolist(), s.values.tolist()) for d, s in zip(dense, sparse)
        ]

    def embed_query(self, text: str) -> Embedding:
        dense = next(iter(self._dense.query_embed(text))).tolist()
        sparse = next(iter(self._sparse.query_embed(text)))
        return Embedding(dense, sparse.indices.tolist(), sparse.values.tolist())
