"""
Retrieval manager — factory for all retriever types.
"""

from __future__ import annotations

from app.core.constants import RetrievalStrategy
from app.core.exceptions import RAGForgeError
from app.retrieval.base import BaseRetriever


class RetrievalManager:
    """
    Factory that returns the correct ``BaseRetriever`` subclass by strategy name.

    Usage
    -----
        retriever = RetrievalManager.get(
            "dense",
            vectorstore_type="chroma",
            collection_name="my_rag",
        )
        results = retriever.retrieve("What is RAG?", k=5)
    """

    @staticmethod
    def get(strategy: str, **kwargs) -> BaseRetriever:
        from app.retrieval.bm25 import BM25Retriever
        from app.retrieval.dense import DenseRetriever
        from app.retrieval.hybrid import HybridRetriever
        from app.retrieval.mmr import MMRRetriever
        from app.retrieval.multi_query import MultiQueryRetriever

        registry: dict[str, type[BaseRetriever]] = {
            RetrievalStrategy.DENSE: DenseRetriever,
            RetrievalStrategy.BM25: BM25Retriever,
            RetrievalStrategy.HYBRID: HybridRetriever,
            RetrievalStrategy.MMR: MMRRetriever,
            RetrievalStrategy.MULTI_QUERY: MultiQueryRetriever,
        }

        cls = registry.get(strategy)
        if cls is None:
            available = list(registry.keys())
            raise RAGForgeError(
                f"Unknown retrieval strategy '{strategy}'. Available: {available}",
                details={"strategy": strategy, "available": available},
            )
        return cls(**kwargs)
