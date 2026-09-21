"""
Domain models for LLM generation results.

A ``GenerationResult`` is the output of a generator node.  It carries the
generated answer text together with token usage statistics and references
to the source chunks that informed the answer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.retrieval.models import RetrievalResult


class TokenUsage(BaseModel):
    """
    Token consumption reported by an LLM API call.

    Not all providers report all fields.
    """

    prompt_tokens: int | None = Field(default=None, description="Tokens in the prompt.")
    completion_tokens: int | None = Field(default=None, description="Tokens in the completion.")
    total_tokens: int | None = Field(default=None, description="Total tokens consumed.")

    @property
    def cost_estimate_usd(self) -> float | None:
        """
        Placeholder for cost estimation.

        Actual cost calculations depend on provider pricing and should be
        implemented in the generator nodes themselves.
        """
        return None


class GenerationResult(BaseModel):
    """
    The output of a generator node.

    Attributes
    ----------
    answer:
        The generated text answer.
    source_chunks:
        The retrieval results that were included in the context window.
        Used to render source citations in the frontend output panel.
    model:
        Name of the LLM model that produced this answer.
    provider:
        Name of the LLM provider.
    usage:
        Token usage statistics.
    finish_reason:
        The reason the generation stopped (e.g. 'stop', 'length').
    metadata:
        Provider-specific extra information (e.g. model version, latency).
    """

    answer: str = Field(description="The generated answer text.")
    source_chunks: list[RetrievalResult] = Field(
        default_factory=list,
        description="Retrieved chunks that informed the answer.",
    )
    model: str = Field(description="LLM model name.")
    provider: str = Field(description="LLM provider name.")
    usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: str | None = Field(
        default=None,
        description="Stop reason from the provider API (e.g. 'stop', 'length', 'content_filter').",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific metadata.",
    )

    @property
    def unique_sources(self) -> list[str]:
        """Return deduplicated list of source paths/URLs cited."""
        seen: set[str] = set()
        sources: list[str] = []
        for result in self.source_chunks:
            src = result.source
            if src and src not in seen:
                seen.add(src)
                sources.append(src)
        return sources

    def __repr__(self) -> str:
        preview = self.answer[:80].replace("\n", " ")
        if len(self.answer) > 80:
            preview += "..."
        return (
            f"GenerationResult(model={self.model!r}, "
            f"tokens={self.usage.total_tokens}, answer={preview!r})"
        )
