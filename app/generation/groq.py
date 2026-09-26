"""
Groq ultra-fast inference generator.
"""

from __future__ import annotations

import os

from app.domain.generation.models import GenerationResult, TokenUsage
from app.domain.retrieval.models import RetrievalResult
from app.generation.base import BaseGenerator


class GroqGenerator(BaseGenerator):
    """Generate answers using Groq's inference API."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("GROQ_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        source_chunks: list[RetrievalResult] | None = None,
    ) -> GenerationResult:
        try:
            from groq import Groq  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("groq is required. Run: pip install groq") from exc

        client = Groq(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = response.choices[0].message.content or ""
        usage = response.usage

        return GenerationResult(
            answer=answer,
            model=self.model,
            provider="groq",
            source_chunks=source_chunks or [],
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            finish_reason=response.choices[0].finish_reason or "stop",
        )
