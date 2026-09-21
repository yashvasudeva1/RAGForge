"""
Domain models for embeddings.

An ``EmbeddedChunk`` pairs a ``Chunk`` with its dense vector representation
produced by an embedder node.  This is the unit stored in a vector store.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, model_validator

from app.domain.chunks.models import Chunk


class EmbeddedChunk(BaseModel):
    """
    A chunk paired with its dense vector embedding.

    Attributes
    ----------
    chunk:
        The source ``Chunk`` object.
    embedding:
        The dense embedding vector as a list of floats.
    model_name:
        Name of the embedding model that produced this vector.
    provider:
        Name of the embedding provider (e.g. ``"openai"``).
    dimensions:
        Length of the embedding vector.
    """

    chunk: Chunk = Field(description="The source chunk.")
    embedding: list[float] = Field(description="Dense embedding vector.")
    model_name: str = Field(description="Name of the embedding model used.")
    provider: str = Field(description="Embedding provider (e.g. 'openai', 'cohere').")
    dimensions: int = Field(
        description="Dimensionality (length) of the embedding vector.",
    )

    @model_validator(mode="after")
    def validate_dimensions(self) -> "EmbeddedChunk":
        """Verify that ``dimensions`` matches the actual vector length."""
        actual = len(self.embedding)
        if actual != self.dimensions:
            raise ValueError(
                f"'dimensions' field is {self.dimensions} but embedding has {actual} elements."
            )
        return self

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    @property
    def id(self) -> str:
        """Delegate to the chunk's ID for use as the vector store document ID."""
        return self.chunk.id

    def cosine_similarity(self, other_embedding: list[float]) -> float:
        """
        Compute the cosine similarity between this embedding and another vector.

        Returns a value in [-1.0, 1.0].  Returns 0.0 if either vector is
        the zero vector to avoid division by zero.
        """
        dot = sum(a * b for a, b in zip(self.embedding, other_embedding))
        mag_self = math.sqrt(sum(x * x for x in self.embedding))
        mag_other = math.sqrt(sum(x * x for x in other_embedding))
        if mag_self == 0.0 or mag_other == 0.0:
            return 0.0
        return dot / (mag_self * mag_other)

    def __repr__(self) -> str:
        return (
            f"EmbeddedChunk(chunk_id={self.chunk.id!r}, "
            f"model={self.model_name!r}, dims={self.dimensions})"
        )


class EmbeddingBatch(BaseModel):
    """
    The result of embedding an entire list of chunks in one API call.

    Attributes
    ----------
    embedded_chunks:
        The list of embedded chunks in the same order as the input.
    model_name:
        Embedding model used.
    provider:
        Embedding provider used.
    total_tokens:
        Total tokens consumed by the embedding call, if reported by the API.
    """

    embedded_chunks: list[EmbeddedChunk]
    model_name: str
    provider: str
    total_tokens: int | None = Field(
        default=None,
        description="Total token usage for this batch, if reported by the provider.",
    )

    @property
    def count(self) -> int:
        return len(self.embedded_chunks)
