"""
Embedder nodes for RAGForge.

Nodes registered here:
  - openai_embedder
  - google_embedder
  - huggingface_embedder
  - cohere_embedder
  - voyage_embedder
  - local_embedder
"""

from __future__ import annotations

from typing import Any

from app.core.constants import FieldType, NodeCategory, PortType
from app.core.registry import registry
from app.nodes.base import BaseNode, NodeField, NodeMetadata, NodePortDefinition

_CHUNKS_INPUT = NodePortDefinition(
    name="chunks",
    port_type=PortType.CHUNKS,
    display_name="Chunks",
    description="Chunks to embed.",
    required=True,
)
_EMBEDDED_CHUNKS_OUTPUT = NodePortDefinition(
    name="embedded_chunks",
    port_type=PortType.EMBEDDED_CHUNKS,
    display_name="Embedded Chunks",
    description="Chunks paired with their dense embedding vectors.",
    required=False,
)


async def _embed_chunks(embedder: Any, chunks: list) -> list:
    """Helper: embed a list of chunks and return EmbeddedChunk objects."""
    from app.domain.embeddings.models import EmbeddedChunk

    texts = [c.content for c in chunks]
    embeddings = await embedder.aembed_texts(texts)
    return [
        EmbeddedChunk(
            chunk=chunk,
            embedding=emb,
            model_name=embedder.model_name,
            provider=embedder.provider,
            dimensions=len(emb),
        )
        for chunk, emb in zip(chunks, embeddings)
    ]


# --------------------------------------------------------------------------- #
# OpenAI Embedder
# --------------------------------------------------------------------------- #


@registry.register("openai_embedder")
class OpenAIEmbedderNode(BaseNode):
    """Embed chunks using the OpenAI Embeddings API."""

    node_type = "openai_embedder"
    metadata = NodeMetadata(
        category=NodeCategory.EMBEDDER,
        display_name="OpenAI Embedder",
        description="Embed text chunks using OpenAI's embedding models (text-embedding-3-small, text-embedding-3-large).",
        icon_name="zap",
        tags=["embed", "openai", "embedding", "dense"],
    )
    input_ports = [_CHUNKS_INPUT]
    output_ports = [_EMBEDDED_CHUNKS_OUTPUT]
    fields = [
        NodeField(
            name="model",
            field_type=FieldType.SELECT,
            display_name="Model",
            description="OpenAI embedding model to use.",
            default="text-embedding-3-small",
            options=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
        ),
        NodeField(
            name="api_key",
            field_type=FieldType.PASSWORD,
            display_name="API Key",
            description="OpenAI API key. Falls back to the OPENAI_API_KEY environment variable.",
            default=None,
            advanced=True,
        ),
        NodeField(
            name="batch_size",
            field_type=FieldType.INTEGER,
            display_name="Batch Size",
            description="Number of chunks to embed per API call.",
            default=100,
            min_value=1,
            max_value=2048,
            advanced=True,
        ),
        NodeField(
            name="dimensions",
            field_type=FieldType.INTEGER,
            display_name="Dimensions",
            description="Output embedding dimensions (only supported by text-embedding-3-* models).",
            default=None,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.embeddings.openai import OpenAIEmbedder

        chunks = inputs.get("chunks", [])
        embedder = OpenAIEmbedder(
            model=self.get_config("model", "text-embedding-3-small"),
            api_key=self.get_config("api_key"),
            batch_size=self.get_config("batch_size", 100),
            dimensions=self.get_config("dimensions"),
        )
        embedded = await _embed_chunks(embedder, chunks)
        return {"embedded_chunks": embedded}


# --------------------------------------------------------------------------- #
# Google Embedder
# --------------------------------------------------------------------------- #


@registry.register("google_embedder")
class GoogleEmbedderNode(BaseNode):
    """Embed chunks using Google Generative AI embedding models."""

    node_type = "google_embedder"
    metadata = NodeMetadata(
        category=NodeCategory.EMBEDDER,
        display_name="Google Embedder",
        description="Embed text chunks using Google Generative AI (Gemini) embedding models.",
        icon_name="zap",
        tags=["embed", "google", "gemini", "embedding", "dense"],
    )
    input_ports = [_CHUNKS_INPUT]
    output_ports = [_EMBEDDED_CHUNKS_OUTPUT]
    fields = [
        NodeField(
            name="model",
            field_type=FieldType.SELECT,
            display_name="Model",
            description="Google embedding model.",
            default="models/text-embedding-004",
            options=["models/text-embedding-004", "models/embedding-001"],
        ),
        NodeField(
            name="api_key",
            field_type=FieldType.PASSWORD,
            display_name="API Key",
            description="Google API key. Falls back to GOOGLE_API_KEY environment variable.",
            default=None,
            advanced=True,
        ),
        NodeField(
            name="task_type",
            field_type=FieldType.SELECT,
            display_name="Task Type",
            description="Embedding task type for optimisation.",
            default="retrieval_document",
            options=[
                "retrieval_document",
                "retrieval_query",
                "semantic_similarity",
                "classification",
                "clustering",
            ],
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.embeddings.google import GoogleEmbedder

        chunks = inputs.get("chunks", [])
        embedder = GoogleEmbedder(
            model=self.get_config("model", "models/text-embedding-004"),
            api_key=self.get_config("api_key"),
            task_type=self.get_config("task_type", "retrieval_document"),
        )
        embedded = await _embed_chunks(embedder, chunks)
        return {"embedded_chunks": embedded}


# --------------------------------------------------------------------------- #
# HuggingFace Embedder
# --------------------------------------------------------------------------- #


@registry.register("huggingface_embedder")
class HuggingFaceEmbedderNode(BaseNode):
    """Embed chunks using HuggingFace Hub embedding models."""

    node_type = "huggingface_embedder"
    metadata = NodeMetadata(
        category=NodeCategory.EMBEDDER,
        display_name="HuggingFace Embedder",
        description="Embed text chunks using models hosted on HuggingFace Hub via the Inference API.",
        icon_name="zap",
        tags=["embed", "huggingface", "hf", "embedding", "dense"],
    )
    input_ports = [_CHUNKS_INPUT]
    output_ports = [_EMBEDDED_CHUNKS_OUTPUT]
    fields = [
        NodeField(
            name="model",
            field_type=FieldType.STRING,
            display_name="Model",
            description="HuggingFace model ID (e.g. 'BAAI/bge-large-en-v1.5').",
            default="BAAI/bge-large-en-v1.5",
        ),
        NodeField(
            name="api_key",
            field_type=FieldType.PASSWORD,
            display_name="API Token",
            description="HuggingFace Hub token. Falls back to HUGGINGFACE_API_KEY.",
            default=None,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.embeddings.huggingface import HuggingFaceEmbedder

        chunks = inputs.get("chunks", [])
        embedder = HuggingFaceEmbedder(
            model=self.get_config("model", "BAAI/bge-large-en-v1.5"),
            api_key=self.get_config("api_key"),
        )
        embedded = await _embed_chunks(embedder, chunks)
        return {"embedded_chunks": embedded}


# --------------------------------------------------------------------------- #
# Cohere Embedder
# --------------------------------------------------------------------------- #


@registry.register("cohere_embedder")
class CohereEmbedderNode(BaseNode):
    """Embed chunks using Cohere Embed v3."""

    node_type = "cohere_embedder"
    metadata = NodeMetadata(
        category=NodeCategory.EMBEDDER,
        display_name="Cohere Embedder",
        description="Embed text chunks using Cohere Embed v3 models with separate document/query encodings.",
        icon_name="zap",
        tags=["embed", "cohere", "embedding", "dense"],
    )
    input_ports = [_CHUNKS_INPUT]
    output_ports = [_EMBEDDED_CHUNKS_OUTPUT]
    fields = [
        NodeField(
            name="model",
            field_type=FieldType.SELECT,
            display_name="Model",
            description="Cohere embedding model.",
            default="embed-english-v3.0",
            options=["embed-english-v3.0", "embed-multilingual-v3.0", "embed-english-light-v3.0"],
        ),
        NodeField(
            name="api_key",
            field_type=FieldType.PASSWORD,
            display_name="API Key",
            description="Cohere API key. Falls back to COHERE_API_KEY.",
            default=None,
            advanced=True,
        ),
        NodeField(
            name="input_type",
            field_type=FieldType.SELECT,
            display_name="Input Type",
            description="Whether these embeddings are for documents or queries.",
            default="search_document",
            options=["search_document", "search_query", "classification", "clustering"],
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.embeddings.cohere import CohereEmbedder

        chunks = inputs.get("chunks", [])
        embedder = CohereEmbedder(
            model=self.get_config("model", "embed-english-v3.0"),
            api_key=self.get_config("api_key"),
            input_type=self.get_config("input_type", "search_document"),
        )
        embedded = await _embed_chunks(embedder, chunks)
        return {"embedded_chunks": embedded}


# --------------------------------------------------------------------------- #
# Voyage Embedder
# --------------------------------------------------------------------------- #


@registry.register("voyage_embedder")
class VoyageEmbedderNode(BaseNode):
    """Embed chunks using VoyageAI embedding models."""

    node_type = "voyage_embedder"
    metadata = NodeMetadata(
        category=NodeCategory.EMBEDDER,
        display_name="Voyage Embedder",
        description="Embed text chunks using VoyageAI's high-quality retrieval embedding models.",
        icon_name="zap",
        tags=["embed", "voyage", "voyageai", "embedding", "dense"],
    )
    input_ports = [_CHUNKS_INPUT]
    output_ports = [_EMBEDDED_CHUNKS_OUTPUT]
    fields = [
        NodeField(
            name="model",
            field_type=FieldType.SELECT,
            display_name="Model",
            description="VoyageAI embedding model.",
            default="voyage-3",
            options=["voyage-3", "voyage-3-lite", "voyage-finance-2", "voyage-code-3"],
        ),
        NodeField(
            name="api_key",
            field_type=FieldType.PASSWORD,
            display_name="API Key",
            description="VoyageAI API key. Falls back to VOYAGE_API_KEY.",
            default=None,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.embeddings.voyage import VoyageEmbedder

        chunks = inputs.get("chunks", [])
        embedder = VoyageEmbedder(
            model=self.get_config("model", "voyage-3"),
            api_key=self.get_config("api_key"),
        )
        embedded = await _embed_chunks(embedder, chunks)
        return {"embedded_chunks": embedded}


# --------------------------------------------------------------------------- #
# Local Embedder (sentence-transformers)
# --------------------------------------------------------------------------- #


@registry.register("local_embedder")
class LocalEmbedderNode(BaseNode):
    """
    Embed chunks using a locally-downloaded sentence-transformers model.

    No API key required. Model is downloaded from HuggingFace on first use.
    """

    node_type = "local_embedder"
    metadata = NodeMetadata(
        category=NodeCategory.EMBEDDER,
        display_name="Local Embedder",
        description=(
            "Embed text chunks using a locally-run sentence-transformers model. "
            "No API key required. Ideal for offline or privacy-sensitive use cases."
        ),
        icon_name="cpu",
        tags=["embed", "local", "sentence-transformers", "offline", "embedding"],
    )
    input_ports = [_CHUNKS_INPUT]
    output_ports = [_EMBEDDED_CHUNKS_OUTPUT]
    fields = [
        NodeField(
            name="model",
            field_type=FieldType.STRING,
            display_name="Model",
            description="sentence-transformers model name or local path.",
            default="all-MiniLM-L6-v2",
        ),
        NodeField(
            name="device",
            field_type=FieldType.SELECT,
            display_name="Device",
            description="Compute device for inference.",
            default="cpu",
            options=["cpu", "cuda", "mps"],
            advanced=True,
        ),
        NodeField(
            name="batch_size",
            field_type=FieldType.INTEGER,
            display_name="Batch Size",
            description="Number of chunks to encode per forward pass.",
            default=32,
            min_value=1,
            max_value=512,
            advanced=True,
        ),
        NodeField(
            name="normalize",
            field_type=FieldType.BOOLEAN,
            display_name="Normalize Embeddings",
            description="L2-normalize embeddings. Recommended for cosine similarity.",
            default=True,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.embeddings.local import LocalEmbedder

        chunks = inputs.get("chunks", [])
        embedder = LocalEmbedder(
            model=self.get_config("model", "all-MiniLM-L6-v2"),
            device=self.get_config("device", "cpu"),
            batch_size=self.get_config("batch_size", 32),
            normalize=self.get_config("normalize", True),
        )
        embedded = await _embed_chunks(embedder, chunks)
        return {"embedded_chunks": embedded}
