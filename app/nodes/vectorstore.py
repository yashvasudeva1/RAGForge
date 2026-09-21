"""
Vector store nodes for RAGForge.

Nodes registered here:
  - chroma_vectorstore
  - qdrant_vectorstore
  - faiss_vectorstore
  - pinecone_vectorstore
  - weaviate_vectorstore
  - milvus_vectorstore
  - lancedb_vectorstore

Each node has two modes, controlled by the ``mode`` field:
  - ``index`` — ingests embedded chunks into the store.
  - ``query`` — searches the store and emits retrieval results.

This dual-mode design lets a single VectorStore node serve both
the ingestion side and the retrieval side of the pipeline.
"""

from __future__ import annotations

from typing import Any

from app.core.constants import FieldType, NodeCategory, PortType
from app.core.registry import registry
from app.nodes.base import BaseNode, NodeField, NodeMetadata, NodePortDefinition

# Shared port definitions
_EMBEDDED_CHUNKS_INPUT = NodePortDefinition(
    name="embedded_chunks",
    port_type=PortType.EMBEDDED_CHUNKS,
    display_name="Embedded Chunks",
    description="Embedded chunks to index into the vector store.",
    required=False,
)
_QUERY_INPUT = NodePortDefinition(
    name="query",
    port_type=PortType.QUERY,
    display_name="Query",
    description="Query string to search for.",
    required=False,
)
_RETRIEVAL_RESULTS_OUTPUT = NodePortDefinition(
    name="retrieval_results",
    port_type=PortType.RETRIEVAL_RESULTS,
    display_name="Retrieval Results",
    description="Retrieved chunks with scores.",
    required=False,
)

# Shared fields
_COLLECTION_FIELD = NodeField(
    name="collection_name",
    field_type=FieldType.STRING,
    display_name="Collection Name",
    description="Name of the collection / index / namespace to use.",
    default="ragforge",
    required=True,
)
_MODE_FIELD = NodeField(
    name="mode",
    field_type=FieldType.SELECT,
    display_name="Mode",
    description="'index' to add chunks, 'query' to search for relevant chunks.",
    default="index",
    options=["index", "query"],
    required=True,
)
_TOP_K_FIELD = NodeField(
    name="top_k",
    field_type=FieldType.SLIDER,
    display_name="Top K",
    description="Number of results to return when in query mode.",
    default=5,
    min_value=1,
    max_value=100,
    step=1,
)


def _make_ports() -> tuple[list[NodePortDefinition], list[NodePortDefinition]]:
    return (
        [_EMBEDDED_CHUNKS_INPUT, _QUERY_INPUT],
        [_RETRIEVAL_RESULTS_OUTPUT],
    )


# --------------------------------------------------------------------------- #
# ChromaDB
# --------------------------------------------------------------------------- #


@registry.register("chroma_vectorstore")
class ChromaVectorStoreNode(BaseNode):
    """ChromaDB vector store — works locally with no external service."""

    node_type = "chroma_vectorstore"
    metadata = NodeMetadata(
        category=NodeCategory.VECTOR_STORE,
        display_name="ChromaDB",
        description="Local or server ChromaDB vector store. No external service needed for local mode. Best for experimentation.",
        icon_name="database",
        tags=["vectorstore", "chroma", "chromadb", "local", "index", "retrieval"],
    )
    input_ports, output_ports = _make_ports()
    fields = [
        _MODE_FIELD,
        _COLLECTION_FIELD,
        _TOP_K_FIELD,
        NodeField(
            name="persist_dir",
            field_type=FieldType.STRING,
            display_name="Persist Directory",
            description="Local directory for ChromaDB persistence. Leave empty for in-memory mode.",
            default="./data/chroma",
        ),
        NodeField(
            name="host",
            field_type=FieldType.STRING,
            display_name="Host",
            description="ChromaDB server host (for client-server mode).",
            default=None,
            advanced=True,
        ),
        NodeField(
            name="port",
            field_type=FieldType.INTEGER,
            display_name="Port",
            description="ChromaDB server port.",
            default=8001,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.vectorstores.chroma import ChromaVectorStore

        store = ChromaVectorStore(
            collection_name=self.require_config("collection_name"),
            persist_dir=self.get_config("persist_dir", "./data/chroma"),
            host=self.get_config("host"),
            port=self.get_config("port", 8001),
        )

        if self.get_config("mode", "index") == "index":
            embedded_chunks = inputs.get("embedded_chunks", [])
            await store.aadd(embedded_chunks)
            return {"retrieval_results": []}
        else:
            query: str = inputs.get("query", "")
            results = await store.asearch(query=query, k=self.get_config("top_k", 5))
            return {"retrieval_results": results}


# --------------------------------------------------------------------------- #
# Qdrant
# --------------------------------------------------------------------------- #


@registry.register("qdrant_vectorstore")
class QdrantVectorStoreNode(BaseNode):
    """Qdrant vector store — local or cloud."""

    node_type = "qdrant_vectorstore"
    metadata = NodeMetadata(
        category=NodeCategory.VECTOR_STORE,
        display_name="Qdrant",
        description="Qdrant vector database — run locally via Docker or use Qdrant Cloud.",
        icon_name="database",
        tags=["vectorstore", "qdrant", "index", "retrieval"],
    )
    input_ports, output_ports = _make_ports()
    fields = [
        _MODE_FIELD,
        _COLLECTION_FIELD,
        _TOP_K_FIELD,
        NodeField(
            name="url",
            field_type=FieldType.STRING,
            display_name="URL",
            description="Qdrant server URL (e.g. http://localhost:6333 or https://xyz.qdrant.io).",
            default="http://localhost:6333",
        ),
        NodeField(
            name="api_key",
            field_type=FieldType.PASSWORD,
            display_name="API Key",
            description="Qdrant Cloud API key. Leave empty for local Qdrant.",
            default=None,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.vectorstores.qdrant import QdrantVectorStore

        store = QdrantVectorStore(
            collection_name=self.require_config("collection_name"),
            url=self.get_config("url", "http://localhost:6333"),
            api_key=self.get_config("api_key"),
        )

        if self.get_config("mode", "index") == "index":
            await store.aadd(inputs.get("embedded_chunks", []))
            return {"retrieval_results": []}
        else:
            results = await store.asearch(
                query=inputs.get("query", ""),
                k=self.get_config("top_k", 5),
            )
            return {"retrieval_results": results}


# --------------------------------------------------------------------------- #
# FAISS
# --------------------------------------------------------------------------- #


@registry.register("faiss_vectorstore")
class FAISSVectorStoreNode(BaseNode):
    """FAISS in-memory vector store — fast, no external service required."""

    node_type = "faiss_vectorstore"
    metadata = NodeMetadata(
        category=NodeCategory.VECTOR_STORE,
        display_name="FAISS",
        description="Facebook AI Similarity Search (FAISS) in-memory vector store. Fast exact and approximate nearest-neighbour search.",
        icon_name="database",
        tags=["vectorstore", "faiss", "in-memory", "index", "retrieval"],
    )
    input_ports, output_ports = _make_ports()
    fields = [
        _MODE_FIELD,
        _TOP_K_FIELD,
        NodeField(
            name="index_path",
            field_type=FieldType.STRING,
            display_name="Index Path",
            description="Path to save/load the FAISS index file. Leave empty for ephemeral in-memory index.",
            default=None,
            advanced=True,
        ),
        NodeField(
            name="index_type",
            field_type=FieldType.SELECT,
            display_name="Index Type",
            description="FAISS index type.",
            default="Flat",
            options=["Flat", "IVF", "HNSW"],
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.vectorstores.faiss import FAISSVectorStore

        store = FAISSVectorStore(
            index_path=self.get_config("index_path"),
            index_type=self.get_config("index_type", "Flat"),
        )

        if self.get_config("mode", "index") == "index":
            await store.aadd(inputs.get("embedded_chunks", []))
            return {"retrieval_results": []}
        else:
            results = await store.asearch(
                query=inputs.get("query", ""),
                k=self.get_config("top_k", 5),
            )
            return {"retrieval_results": results}


# --------------------------------------------------------------------------- #
# Pinecone
# --------------------------------------------------------------------------- #


@registry.register("pinecone_vectorstore")
class PineconeVectorStoreNode(BaseNode):
    """Pinecone managed vector database."""

    node_type = "pinecone_vectorstore"
    metadata = NodeMetadata(
        category=NodeCategory.VECTOR_STORE,
        display_name="Pinecone",
        description="Pinecone fully-managed cloud vector database with serverless and pod-based options.",
        icon_name="database",
        tags=["vectorstore", "pinecone", "cloud", "index", "retrieval"],
    )
    input_ports, output_ports = _make_ports()
    fields = [
        _MODE_FIELD,
        _TOP_K_FIELD,
        NodeField(
            name="api_key",
            field_type=FieldType.PASSWORD,
            display_name="API Key",
            description="Pinecone API key. Falls back to PINECONE_API_KEY.",
            default=None,
            required=True,
        ),
        NodeField(
            name="index_name",
            field_type=FieldType.STRING,
            display_name="Index Name",
            description="Pinecone index name.",
            required=True,
            default=None,
        ),
        NodeField(
            name="namespace",
            field_type=FieldType.STRING,
            display_name="Namespace",
            description="Pinecone namespace within the index.",
            default="",
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.vectorstores.pinecone import PineconeVectorStore

        store = PineconeVectorStore(
            api_key=self.get_config("api_key"),
            index_name=self.require_config("index_name"),
            namespace=self.get_config("namespace", ""),
        )

        if self.get_config("mode", "index") == "index":
            await store.aadd(inputs.get("embedded_chunks", []))
            return {"retrieval_results": []}
        else:
            results = await store.asearch(
                query=inputs.get("query", ""),
                k=self.get_config("top_k", 5),
            )
            return {"retrieval_results": results}


# --------------------------------------------------------------------------- #
# LanceDB
# --------------------------------------------------------------------------- #


@registry.register("lancedb_vectorstore")
class LanceDBVectorStoreNode(BaseNode):
    """LanceDB local columnar vector store."""

    node_type = "lancedb_vectorstore"
    metadata = NodeMetadata(
        category=NodeCategory.VECTOR_STORE,
        display_name="LanceDB",
        description="LanceDB — a local columnar vector database stored as files. No server required.",
        icon_name="database",
        tags=["vectorstore", "lancedb", "local", "columnar", "index", "retrieval"],
    )
    input_ports, output_ports = _make_ports()
    fields = [
        _MODE_FIELD,
        _COLLECTION_FIELD,
        _TOP_K_FIELD,
        NodeField(
            name="uri",
            field_type=FieldType.STRING,
            display_name="URI",
            description="Local directory path or LanceDB Cloud URI.",
            default="./data/lancedb",
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.vectorstores.lancedb import LanceDBVectorStore

        store = LanceDBVectorStore(
            uri=self.get_config("uri", "./data/lancedb"),
            table_name=self.get_config("collection_name", "ragforge"),
        )

        if self.get_config("mode", "index") == "index":
            await store.aadd(inputs.get("embedded_chunks", []))
            return {"retrieval_results": []}
        else:
            results = await store.asearch(
                query=inputs.get("query", ""),
                k=self.get_config("top_k", 5),
            )
            return {"retrieval_results": results}
