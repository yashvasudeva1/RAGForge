"""
Reranking manager — factory for all reranker types.
"""

from __future__ import annotations

from app.core.constants import RerankerType
from app.core.exceptions import RAGForgeError
from app.reranking.base import BaseReranker


class RerankingManager:
    """
    Factory that returns the correct reranker by type name.

    Usage
    -----
        reranker = RerankingManager.get("cross_encoder", model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        results = reranker.rerank(query, candidates, top_n=5)
    """

    @staticmethod
    def get(reranker_type: str, **kwargs) -> BaseReranker:
        from app.reranking.cohere import CohereReranker
        from app.reranking.cross_encoder import CrossEncoderReranker
        from app.reranking.llm_reranker import LLMReranker
        from app.reranking.voyage import VoyageReranker

        registry: dict[str, type[BaseReranker]] = {
            RerankerType.CROSS_ENCODER: CrossEncoderReranker,
            RerankerType.COHERE: CohereReranker,
            RerankerType.VOYAGE: VoyageReranker,
            RerankerType.LLM: LLMReranker,
        }

        cls = registry.get(reranker_type)
        if cls is None:
            available = list(registry.keys())
            raise RAGForgeError(
                f"Unknown reranker type '{reranker_type}'. Available: {available}",
                details={"reranker_type": reranker_type, "available": available},
            )
        return cls(**kwargs)
