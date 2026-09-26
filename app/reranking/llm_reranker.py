"""
LLM-based pointwise reranker.

Prompts an LLM to score each (query, chunk) pair on a 1-10 relevance scale.
Flexible but slower than dedicated rerank models.
"""

from __future__ import annotations

import os
import re

from app.core.logging import get_logger
from app.domain.retrieval.models import RetrievalResult
from app.reranking.base import BaseReranker

logger = get_logger(__name__)

_DEFAULT_PROMPT = (
    "On a scale of 1 to 10, how relevant is the following passage to the query?\n"
    "Respond with ONLY a single integer between 1 and 10.\n\n"
    "Query: {query}\n\n"
    "Passage: {chunk}\n\n"
    "Score:"
)


def _parse_score(text: str) -> float:
    """Extract the first integer 1-10 from the LLM response."""
    match = re.search(r"\b([1-9]|10)\b", text.strip())
    return float(match.group(1)) if match else 5.0


class LLMReranker(BaseReranker):
    """
    Pointwise LLM reranker.

    Parameters
    ----------
    llm_provider:
        One of ``openai``, ``anthropic``, ``google``, ``groq``.
    llm_model:
        Model name for the chosen provider.
    scoring_prompt:
        Jinja2-style prompt template with ``{query}`` and ``{chunk}`` placeholders.
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        scoring_prompt: str | None = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.scoring_prompt = scoring_prompt or _DEFAULT_PROMPT

    def _score(self, query: str, chunk: str) -> float:
        prompt = self.scoring_prompt.format(query=query, chunk=chunk[:500])

        try:
            if self.llm_provider == "openai":
                from openai import OpenAI  # type: ignore[import-untyped]
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                resp = client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0.0,
                )
                return _parse_score(resp.choices[0].message.content or "")

            elif self.llm_provider == "google":
                import google.generativeai as genai  # type: ignore[import-untyped]
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                model = genai.GenerativeModel(self.llm_model)
                return _parse_score(model.generate_content(prompt).text)

            elif self.llm_provider == "groq":
                from groq import Groq  # type: ignore[import-untyped]
                client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                resp = client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0.0,
                )
                return _parse_score(resp.choices[0].message.content or "")

            else:
                return 5.0  # neutral fallback

        except Exception as exc:  # noqa: BLE001
            logger.warning("LLMReranker scoring failed: %s", exc)
            return 5.0

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int = 5,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        scored = [(r, self._score(query, r.content)) for r in results]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                content=r.content,
                chunk_id=r.chunk_id,
                source=r.source,
                score=score / 10.0,  # Normalise to 0-1
                rank=rank,
                metadata=r.metadata,
            )
            for rank, (r, score) in enumerate(scored[:top_n], start=1)
        ]
