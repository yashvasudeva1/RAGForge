"""
Google Gemini generator.
"""

from __future__ import annotations

import os

from app.core.logging import get_logger
from app.domain.generation.models import GenerationResult, TokenUsage
from app.domain.retrieval.models import RetrievalResult
from app.generation.base import BaseGenerator

logger = get_logger(__name__)


class GoogleGenerator(BaseGenerator):
    """Generate answers using Google Gemini models."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        source_chunks: list[RetrievalResult] | None = None,
    ) -> GenerationResult:
        try:
            import google.generativeai as genai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("google-generativeai is required. Run: pip install google-generativeai") from exc

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        response = model.generate_content(prompt)
        answer = response.text or ""

        usage_meta = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0

        return GenerationResult(
            answer=answer,
            model=self.model,
            provider="google",
            source_chunks=source_chunks or [],
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reason="stop",
        )
