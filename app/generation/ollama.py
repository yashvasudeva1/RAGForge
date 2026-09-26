"""
Ollama local LLM generator.
"""

from __future__ import annotations

from app.domain.generation.models import GenerationResult, TokenUsage
from app.domain.retrieval.models import RetrievalResult
from app.generation.base import BaseGenerator


class OllamaGenerator(BaseGenerator):
    """Generate answers using locally-running Ollama models."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        source_chunks: list[RetrievalResult] | None = None,
    ) -> GenerationResult:
        try:
            import ollama  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("ollama is required. Run: pip install ollama") from exc

        client = ollama.Client(host=self.base_url)
        response = client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        )

        answer = response.message.content or ""
        prompt_tokens = getattr(response, "prompt_eval_count", 0) or 0
        completion_tokens = getattr(response, "eval_count", 0) or 0

        return GenerationResult(
            answer=answer,
            model=self.model,
            provider="ollama",
            source_chunks=source_chunks or [],
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reason="stop",
        )
