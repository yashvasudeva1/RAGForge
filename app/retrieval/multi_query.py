"""
Multi-query retriever.

Uses an LLM to generate multiple rephrasings of the original query,
retrieves for each variant, then deduplicates and merges results.
Improves recall on ambiguous or multi-faceted questions.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.retrieval.models import RetrievalResult
from app.retrieval.base import BaseRetriever
from app.retrieval.hybrid import reciprocal_rank_fusion

logger = get_logger(__name__)

_QUERY_GEN_PROMPT = """You are an expert at reformulating search queries.
Given the following question, generate {n} alternative versions that capture
different aspects or phrasings of the same information need.
Return ONLY the queries, one per line, with no numbering or extra text.

Original question: {query}

Alternative queries:"""


def _generate_query_variants(
    query: str,
    n: int,
    llm_provider: str,
    llm_model: str,
) -> list[str]:
    """Call an LLM to generate n query variants."""
    prompt = _QUERY_GEN_PROMPT.format(n=n, query=query)

    try:
        if llm_provider == "openai":
            import os
            from openai import OpenAI  # type: ignore[import-untyped]
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
            )
            text = response.choices[0].message.content or ""
        elif llm_provider == "google":
            import os
            import google.generativeai as genai  # type: ignore[import-untyped]
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel(llm_model)
            text = model.generate_content(prompt).text
        elif llm_provider == "groq":
            import os
            from groq import Groq  # type: ignore[import-untyped]
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            text = response.choices[0].message.content or ""
        else:
            # Fallback: simple rule-based variants
            return [query, f"What is {query}?", f"Tell me about {query}"][:n]

        variants = [line.strip() for line in text.strip().splitlines() if line.strip()]
        return variants[:n]

    except Exception as exc:  # noqa: BLE001
        logger.warning("MultiQuery LLM call failed (%s), using original query only", exc)
        return [query]


class MultiQueryRetriever(BaseRetriever):
    """
    Multi-query retriever that generates query variants via an LLM.

    Parameters
    ----------
    vectorstore_type / collection_name:
        Dense retrieval backend.
    num_queries:
        Number of query variants to generate (including the original).
    llm_provider / llm_model:
        LLM used to generate variants.
    """

    def __init__(
        self,
        vectorstore_type: str = "chroma",
        collection_name: str = "ragforge",
        num_queries: int = 3,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        **dense_kwargs,
    ) -> None:
        self.vectorstore_type = vectorstore_type
        self.collection_name = collection_name
        self.num_queries = num_queries
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.dense_kwargs = dense_kwargs

    def retrieve(self, query: str, k: int = 5, **kwargs) -> list[RetrievalResult]:
        from app.retrieval.dense import DenseRetriever

        # Generate variants (include the original)
        variants = _generate_query_variants(
            query=query,
            n=self.num_queries - 1,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
        )
        all_queries = [query] + [v for v in variants if v != query]

        logger.debug("MultiQueryRetriever: %d query variants", len(all_queries))

        retriever = DenseRetriever(
            vectorstore_type=self.vectorstore_type,
            collection_name=self.collection_name,
            **self.dense_kwargs,
        )

        all_result_lists: list[list[RetrievalResult]] = []
        for q in all_queries:
            results = retriever.retrieve(q, k=k)
            all_result_lists.append(results)

        fused = reciprocal_rank_fusion(
            ranked_lists=all_result_lists,
            k=60,
            weights=[1.0] * len(all_result_lists),
        )

        logger.debug("MultiQueryRetriever: fused to %d results, returning top-%d", len(fused), k)
        return fused[:k]
