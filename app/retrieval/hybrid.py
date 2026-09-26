"""
Hybrid retriever — fuses dense and BM25 results via Reciprocal Rank Fusion.

RRF score formula (Cormack et al. 2009):
    RRF(d) = sum_{r in rankings} 1 / (k + rank(d, r))

where ``k`` is a smoothing constant (default 60).
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.retrieval.models import RetrievalResult
from app.retrieval.base import BaseRetriever

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievalResult]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists:
        Each element is a ranked list of ``RetrievalResult`` objects.
    k:
        RRF smoothing constant.
    weights:
        Optional per-list weight multipliers.

    Returns
    -------
    list[RetrievalResult]
        Deduplicated results sorted by fused RRF score (descending).
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    rrf_scores: dict[str, float] = {}
    chunk_lookup: dict[str, RetrievalResult] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank_idx, result in enumerate(ranked_list, start=1):
            key = result.chunk_id or result.content[:100]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + weight * (1.0 / (k + rank_idx))
            if key not in chunk_lookup:
                chunk_lookup[key] = result

    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    fused: list[RetrievalResult] = []
    for new_rank, key in enumerate(sorted_keys, start=1):
        result = chunk_lookup[key]
        fused.append(
            RetrievalResult(
                content=result.content,
                chunk_id=result.chunk_id,
                source=result.source,
                score=rrf_scores[key],
                rank=new_rank,
                metadata=result.metadata,
            )
        )
    return fused


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever: dense + BM25 fused via Reciprocal Rank Fusion.

    Parameters
    ----------
    vectorstore_type:
        Vector store backend for dense retrieval.
    collection_name:
        Collection name shared by both retrievers.
    dense_weight:
        Weight for the dense results in RRF (BM25 weight = 1 - dense_weight).
    rrf_k:
        RRF smoothing constant.
    fetch_k:
        Candidates fetched from each sub-retriever before fusion.
    **dense_kwargs:
        Extra kwargs forwarded to ``DenseRetriever``.
    """

    def __init__(
        self,
        vectorstore_type: str = "chroma",
        collection_name: str = "ragforge",
        dense_weight: float = 0.5,
        rrf_k: int = 60,
        fetch_k: int = 20,
        **dense_kwargs,
    ) -> None:
        self.vectorstore_type = vectorstore_type
        self.collection_name = collection_name
        self.dense_weight = dense_weight
        self.sparse_weight = 1.0 - dense_weight
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k
        self.dense_kwargs = dense_kwargs

    def retrieve(self, query: str, k: int = 5, **kwargs) -> list[RetrievalResult]:
        from app.retrieval.bm25 import BM25Retriever
        from app.retrieval.dense import DenseRetriever

        dense = DenseRetriever(
            vectorstore_type=self.vectorstore_type,
            collection_name=self.collection_name,
            **self.dense_kwargs,
        )
        sparse = BM25Retriever(collection_name=self.collection_name)

        dense_results = dense.retrieve(query, k=self.fetch_k)
        sparse_results = sparse.retrieve(query, k=self.fetch_k)

        fused = reciprocal_rank_fusion(
            ranked_lists=[dense_results, sparse_results],
            k=self.rrf_k,
            weights=[self.dense_weight, self.sparse_weight],
        )

        logger.debug(
            "HybridRetriever: dense=%d, bm25=%d, fused=%d, returning top-%d",
            len(dense_results), len(sparse_results), len(fused), k,
        )
        return fused[:k]
