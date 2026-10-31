"""Qdrant: collection setup, upserts, stale-point cleanup and filtered search."""
from __future__ import annotations

from qdrant_client import QdrantClient, models

from core.settings import settings
from ingestion.records import Embedding, IndexPoint

DENSE = "dense"
SPARSE = "bm25"
# Fields we filter on at query time (permissions and scoping). Indexing them keeps
# filtered search fast instead of scanning every payload.
KEYWORD_FIELDS = ["doc_id", "source_name", "department", "access_level", "product"]


def get_qdrant() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)


def build_filter(
    access_levels: list[str] | None = None, departments: list[str] | None = None
) -> models.Filter | None:
    """Permission filter. None means "no restriction"; an empty list means "nothing allowed"."""
    conditions = []
    if access_levels is not None:
        conditions.append(models.FieldCondition(
            key="access_level", match=models.MatchAny(any=list(access_levels))))
    if departments is not None:
        conditions.append(models.FieldCondition(
            key="department", match=models.MatchAny(any=list(departments))))
    return models.Filter(must=conditions) if conditions else None


class QdrantIndex:
    def __init__(self, client: QdrantClient | None = None, collection: str | None = None):
        self.client = client or get_qdrant()
        self.collection = collection or settings.qdrant_collection

    def ensure(self, dimension: int) -> None:
        if self.client.collection_exists(self.collection):
            existing = self.client.get_collection(self.collection).config.params.vectors
            size = existing[DENSE].size if isinstance(existing, dict) else existing.size
            if size != dimension:
                raise RuntimeError(
                    f"Collection '{self.collection}' has {size}-dim vectors but the current "
                    f"embedding model produces {dimension}. Re-run with --recreate."
                )
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=dimension, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        for field in KEYWORD_FIELDS:
            self.client.create_payload_index(
                self.collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
            )

    def recreate(self, dimension: int) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.ensure(dimension)

    def upsert(self, points: list[IndexPoint], batch_size: int = 64) -> None:
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=p.point_id,
                        vector={
                            DENSE: p.embedding.dense,
                            SPARSE: models.SparseVector(
                                indices=p.embedding.sparse_indices,
                                values=p.embedding.sparse_values,
                            ),
                        },
                        payload=p.payload,
                    )
                    for p in points[i:i + batch_size]
                ],
            )

    def delete_stale(self, doc_id: str, keep_ids: list[str]) -> None:
        """Remove a document's points that are no longer part of its current chunk set."""
        if not keep_ids:
            return self.delete_document(doc_id)
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))],
                must_not=[models.HasIdCondition(has_id=keep_ids)],
            )),
        )

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            )),
        )

    def delete_source(self, source_name: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(
                    key="source_name", match=models.MatchValue(value=source_name))]
            )),
        )

    def healthy(self) -> None:
        """Raises if Qdrant is unreachable (cheap call, no collection required)."""
        self.client.get_collections()

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    # ------------------------------------------------------------ Step 9 retrieval API
    def dense_search(self, embedding: Embedding, limit: int = 30,
                     access_levels=None, departments=None):
        return self.search(embedding, DENSE, limit, access_levels, departments)

    def sparse_search(self, embedding: Embedding, limit: int = 30,
                      access_levels=None, departments=None):
        if not embedding.sparse_indices:        # e.g. a query made only of stopwords
            return []
        return self.search(embedding, SPARSE, limit, access_levels, departments)

    def hybrid_search(self, embedding: Embedding, limit: int = 30,
                      access_levels=None, departments=None, prefetch_limit: int | None = None):
        """Dense + sparse in ONE Qdrant call, fused with Reciprocal Rank Fusion.

        RRF scores a chunk by sum(1 / (k + rank)) over each list it appears in, so it needs no
        score calibration between cosine similarity and BM25 - only ranks. A chunk that is
        decent in both lists beats one that is great in only one.

        The permission filter goes on EVERY prefetch (and the outer query): Qdrant applies a
        top-level filter only to the final stage, so without this the candidate lists could
        contain chunks the caller may not see.
        """
        flt = build_filter(access_levels, departments)
        depth = prefetch_limit or limit * 2
        prefetch = [models.Prefetch(query=embedding.dense, using=DENSE, limit=depth, filter=flt)]
        if embedding.sparse_indices:
            prefetch.append(models.Prefetch(
                query=models.SparseVector(
                    indices=embedding.sparse_indices, values=embedding.sparse_values),
                using=SPARSE, limit=depth, filter=flt,
            ))
        return self.client.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=flt,
            limit=limit,
            with_payload=True,
        ).points

    def search(
        self,
        embedding: Embedding,
        using: str = DENSE,
        limit: int = 5,
        access_levels: list[str] | None = None,
        departments: list[str] | None = None,
    ):
        if using == DENSE:
            query = embedding.dense
        else:
            query = models.SparseVector(
                indices=embedding.sparse_indices, values=embedding.sparse_values
            )
        result = self.client.query_points(
            collection_name=self.collection,
            query=query,
            using=using,
            limit=limit,
            query_filter=build_filter(access_levels, departments),
            with_payload=True,
        )
        return result.points
