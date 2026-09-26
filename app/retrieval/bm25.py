"""
BM25 keyword retriever using rank-bm25.

Maintains an in-memory BM25 index over the stored chunks.
For production use, the index should be pre-built and cached.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.retrieval.models import RetrievalResult
from app.retrieval.base import BaseRetriever

logger = get_logger(__name__)

# Module-level in-memory BM25 corpus cache.
# Key: collection_name -> (tokenized_corpus, raw_texts, chunk_ids, sources)
_BM25_INDEX_CACHE: dict[str, tuple] = {}


def register_chunks_for_bm25(
    collection_name: str,
    chunk_ids: list[str],
    texts: list[str],
    sources: list[str],
) -> None:
    """
    Register a batch of chunks into the in-memory BM25 index.

    Called by the indexing pipeline after chunks are created.
    """
    tokenized = [t.lower().split() for t in texts]
    existing = _BM25_INDEX_CACHE.get(collection_name)
    if existing:
        ex_tokenized, ex_texts, ex_ids, ex_sources = existing
        _BM25_INDEX_CACHE[collection_name] = (
            ex_tokenized + tokenized,
            ex_texts + texts,
            ex_ids + chunk_ids,
            ex_sources + sources,
        )
    else:
        _BM25_INDEX_CACHE[collection_name] = (tokenized, texts, chunk_ids, sources)

    logger.debug("BM25 index '%s': %d total chunks", collection_name, len(texts))


class BM25Retriever(BaseRetriever):
    """
    Sparse BM25 keyword retriever using the rank-bm25 library.

    Uses the in-memory corpus cache registered via ``register_chunks_for_bm25``.
    Falls back gracefully when no corpus is available.

    Parameters
    ----------
    collection_name:
        Corpus to search.
    k1:
        BM25 term saturation parameter.
    b:
        BM25 document length normalisation.
    """

    def __init__(
        self,
        collection_name: str = "ragforge",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.collection_name = collection_name
        self.k1 = k1
        self.b = b

    def retrieve(self, query: str, k: int = 5, **kwargs) -> list[RetrievalResult]:
        try:
            from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("rank-bm25 is required. Run: pip install rank-bm25") from exc

        corpus_data = _BM25_INDEX_CACHE.get(self.collection_name)
        if not corpus_data:
            logger.warning(
                "BM25: no corpus registered for collection '%s'. Returning empty results.",
                self.collection_name,
            )
            return []

        tokenized_corpus, raw_texts, chunk_ids, sources = corpus_data
        bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        results: list[RetrievalResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            if scores[idx] <= 0:
                continue
            results.append(
                RetrievalResult(
                    content=raw_texts[idx],
                    chunk_id=chunk_ids[idx],
                    source=sources[idx],
                    score=float(scores[idx]),
                    rank=rank,
                )
            )

        logger.debug("BM25Retriever: %d results for query='%s...'", len(results), query[:40])
        return results
