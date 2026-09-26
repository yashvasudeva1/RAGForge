"""
Mistral AI generator.
"""

from __future__ import annotations

import os

from app.domain.generation.models import GenerationResult, TokenUsage
from app.domain.retrieval.models import RetrievalResult
from app.generation.base import BaseGenerator


class MistralGenerator(BaseGenerator):
    """Generate answers using Mistral AI models."""

    def __init__(
        self,
        model: str = "mistral-small-latest",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        source_chunks: list[RetrievalResult] | None = None,
    ) -> GenerationResult:
        try:
            from mistralai import Mistral  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("mistralai is required. Run: pip install mistralai") from exc

        client = Mistral(api_key=self._api_key)
        response = client.chat.complete(
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
            provider="mistral",
            source_chunks=source_chunks or [],
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            finish_reason=response.choices[0].finish_reason or "stop",
        )
