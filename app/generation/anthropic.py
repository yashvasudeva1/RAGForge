"""
Anthropic Claude generator.
"""

from __future__ import annotations

import os

from app.core.logging import get_logger
from app.domain.generation.models import GenerationResult, TokenUsage
from app.domain.retrieval.models import RetrievalResult
from app.generation.base import BaseGenerator

logger = get_logger(__name__)


class AnthropicGenerator(BaseGenerator):
    """Generate answers using Anthropic Claude models."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        source_chunks: list[RetrievalResult] | None = None,
    ) -> GenerationResult:
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("anthropic is required. Run: pip install anthropic") from exc

        client = anthropic.Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = message.content[0].text if message.content else ""
        usage = message.usage

        return GenerationResult(
            answer=answer,
            model=self.model,
            provider="anthropic",
            source_chunks=source_chunks or [],
            usage=TokenUsage(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
            ),
            finish_reason=message.stop_reason or "end_turn",
        )
