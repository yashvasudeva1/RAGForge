"""
OpenAI Chat Completions generator.
"""

from __future__ import annotations

import os

from app.core.logging import get_logger
from app.domain.generation.models import GenerationResult, TokenUsage
from app.domain.retrieval.models import RetrievalResult
from app.generation.base import BaseGenerator

logger = get_logger(__name__)


class OpenAIGenerator(BaseGenerator):
    """
    Generate answers using OpenAI Chat Completions.

    Parameters
    ----------
    model:
        OpenAI model ID (e.g. ``gpt-4o-mini``, ``gpt-4o``, ``o1``).
    api_key:
        OpenAI API key. Falls back to ``OPENAI_API_KEY``.
    temperature / max_tokens:
        Generation parameters.
    system_prompt:
        Optional system message prepended to every request.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def generate(
        self,
        prompt: str,
        source_chunks: list[RetrievalResult] | None = None,
    ) -> GenerationResult:
        try:
            from openai import OpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("openai is required. Run: pip install openai") from exc

        client = OpenAI(api_key=self._api_key)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = response.choices[0].message.content or ""
        usage = response.usage

        result = GenerationResult(
            answer=answer,
            model=self.model,
            provider="openai",
            source_chunks=source_chunks or [],
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            finish_reason=response.choices[0].finish_reason or "stop",
        )
        logger.debug("OpenAI generated %d chars, %d tokens", len(answer), result.usage.total_tokens)
        return result
