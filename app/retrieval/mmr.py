"""
Maximal Marginal Relevance (MMR) retriever.

MMR iteratively selects documents that are both relevant to the query
and maximally different from already-selected documents, trading off
relevance and diversity via the lambda parameter.
"""

from __future__ import annotations

import math

from app.core.logging import get_logger
from app.domain.retrieval.models import RetrievalResult
from app.retrieval.base import BaseRetriever

logger = get_logger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    ma = math.sqrt(sum(x * x for x in a))
    mb = math.sqrt(sum(x * x for x in b))
    return dot / (ma * mb) if (ma > 0 and mb > 0) else 0.0


class MMRRetriever(BaseRetriever):
    """
    MMR retriever — balances relevance and diversity.

    Parameters
    ----------
    vectorstore_type / collection_name:
        Vector store to search.
    lambda_mult:
        1.0 = pure relevance, 0.0 = pure diversity.
    fetch_k:
        Candidates fetched before MMR filtering.
    **dense_kwargs:
        Forwarded to the underlying ``DenseRetriever``.
    """

    def __init__(
        self,
        vectorstore_type: str = "chroma",
        collection_name: str = "ragforge",
        lambda_mult: float = 0.5,
        fetch_k: int = 20,
        **dense_kwargs,
    ) -> None:
        self.vectorstore_type = vectorstore_type
        self.collection_name = collection_name
        self.lambda_mult = lambda_mult
        self.fetch_k = fetch_k
        self.dense_kwargs = dense_kwargs

    def retrieve(self, query: str, k: int = 5, **kwargs) -> list[RetrievalResult]:
        from app.embeddings.manager import EmbeddingManager
        from app.vectorstores.manager import VectorStoreManager

        embedding_provider = self.dense_kwargs.get("embedding_provider", "openai")
        embedding_model = self.dense_kwargs.get("embedding_model", "text-embedding-3-small")
        embedder = EmbeddingManager.get(embedding_provider, model=embedding_model)
        query_vector = embedder.embed_query(query)

        store = VectorStoreManager.get(
            self.vectorstore_type,
            collection_name=self.collection_name,
        )
        candidates = store.search(query_vector, k=self.fetch_k)

        if not candidates:
            return []

        # We need embedding vectors for the candidates to compute inter-doc similarity.
        # Embed all candidate texts.
        candidate_texts = [r.content for r in candidates]
        candidate_vectors = embedder.embed_texts(candidate_texts)

        selected_indices: list[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(min(k, len(candidates))):
            if not remaining:
                break

            if not selected_indices:
                # First pick: highest relevance to query
                best_idx = max(remaining, key=lambda i: candidates[i].score)
            else:
                # MMR score = lambda * relevance - (1-lambda) * max_sim_to_selected
                selected_vecs = [candidate_vectors[i] for i in selected_indices]

                def mmr_score(i: int) -> float:
                    rel = candidates[i].score
                    max_sim = max(_cosine(candidate_vectors[i], sv) for sv in selected_vecs)
                    return self.lambda_mult * rel - (1 - self.lambda_mult) * max_sim

                best_idx = max(remaining, key=mmr_score)

            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        results = [
            RetrievalResult(
                content=candidates[i].content,
                chunk_id=candidates[i].chunk_id,
                source=candidates[i].source,
                score=candidates[i].score,
                rank=new_rank,
                metadata=candidates[i].metadata,
            )
            for new_rank, i in enumerate(selected_indices, start=1)
        ]

        logger.debug("MMRRetriever: selected %d diverse results", len(results))
        return results
