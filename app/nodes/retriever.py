"""
Retriever nodes for RAGForge.

Nodes registered here:
  - dense_retriever
  - bm25_retriever
  - hybrid_retriever
  - mmr_retriever
  - multi_query_retriever
  - contextual_retriever
"""

from __future__ import annotations

from typing import Any

from app.core.constants import DEFAULT_RETRIEVAL_K, FieldType, NodeCategory, PortType
from app.core.registry import registry
from app.nodes.base import BaseNode, NodeField, NodeMetadata, NodePortDefinition

_QUERY_INPUT = NodePortDefinition(
    name="query",
    port_type=PortType.QUERY,
    display_name="Query",
    description="The user query to retrieve relevant chunks for.",
    required=True,
)
_RETRIEVAL_RESULTS_OUTPUT = NodePortDefinition(
    name="retrieval_results",
    port_type=PortType.RETRIEVAL_RESULTS,
    display_name="Retrieval Results",
    description="Retrieved chunks with relevance scores.",
    required=False,
)
_TOP_K_FIELD = NodeField(
    name="top_k",
    field_type=FieldType.SLIDER,
    display_name="Top K",
    description="Number of chunks to retrieve.",
    default=DEFAULT_RETRIEVAL_K,
    min_value=1,
    max_value=50,
    step=1,
)


# --------------------------------------------------------------------------- #
# Dense Retriever
# --------------------------------------------------------------------------- #


@registry.register("dense_retriever")
class DenseRetrieverNode(BaseNode):
    """Dense vector similarity retrieval from a vector store."""

    node_type = "dense_retriever"
    metadata = NodeMetadata(
        category=NodeCategory.RETRIEVER,
        display_name="Dense Retriever",
        description="Retrieves chunks by embedding the query and computing cosine similarity against stored embeddings.",
        icon_name="search",
        tags=["retriever", "dense", "similarity", "vector", "embedding"],
    )
    input_ports = [_QUERY_INPUT]
    output_ports = [_RETRIEVAL_RESULTS_OUTPUT]
    fields = [
        _TOP_K_FIELD,
        NodeField(
            name="vectorstore_type",
            field_type=FieldType.SELECT,
            display_name="Vector Store",
            description="Which vector store to retrieve from.",
            default="chroma",
            options=["chroma", "qdrant", "faiss", "pinecone", "lancedb"],
            required=True,
        ),
        NodeField(
            name="collection_name",
            field_type=FieldType.STRING,
            display_name="Collection Name",
            description="Collection / index name in the vector store.",
            default="ragforge",
            required=True,
        ),
        NodeField(
            name="score_threshold",
            field_type=FieldType.FLOAT,
            display_name="Score Threshold",
            description="Minimum similarity score. Results below this are discarded. Set to 0 to disable.",
            default=0.0,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.retrieval.dense import DenseRetriever

        query: str = inputs.get("query", "")
        retriever = DenseRetriever(
            vectorstore_type=self.require_config("vectorstore_type"),
            collection_name=self.get_config("collection_name", "ragforge"),
            score_threshold=self.get_config("score_threshold", 0.0),
        )
        results = await retriever.aretrieve(
            query=query,
            k=self.get_config("top_k", DEFAULT_RETRIEVAL_K),
        )
        return {"retrieval_results": results}


# --------------------------------------------------------------------------- #
# BM25 Retriever
# --------------------------------------------------------------------------- #


@registry.register("bm25_retriever")
class BM25RetrieverNode(BaseNode):
    """Sparse BM25 keyword retrieval."""

    node_type = "bm25_retriever"
    metadata = NodeMetadata(
        category=NodeCategory.RETRIEVER,
        display_name="BM25 Retriever",
        description="Classic BM25 keyword-based retrieval. Works well for exact keyword matches and does not require embeddings.",
        icon_name="search",
        tags=["retriever", "bm25", "sparse", "keyword", "lexical"],
    )
    input_ports = [_QUERY_INPUT]
    output_ports = [_RETRIEVAL_RESULTS_OUTPUT]
    fields = [
        _TOP_K_FIELD,
        NodeField(
            name="k1",
            field_type=FieldType.FLOAT,
            display_name="k1",
            description="BM25 term saturation parameter.",
            default=1.5,
            min_value=0.5,
            max_value=3.0,
            advanced=True,
        ),
        NodeField(
            name="b",
            field_type=FieldType.FLOAT,
            display_name="b",
            description="BM25 document length normalisation parameter.",
            default=0.75,
            min_value=0.0,
            max_value=1.0,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.retrieval.bm25 import BM25Retriever

        query: str = inputs.get("query", "")
        retriever = BM25Retriever(
            k1=self.get_config("k1", 1.5),
            b=self.get_config("b", 0.75),
        )
        results = await retriever.aretrieve(
            query=query,
            k=self.get_config("top_k", DEFAULT_RETRIEVAL_K),
        )
        return {"retrieval_results": results}


# --------------------------------------------------------------------------- #
# Hybrid Retriever
# --------------------------------------------------------------------------- #


@registry.register("hybrid_retriever")
class HybridRetrieverNode(BaseNode):
    """
    Hybrid retriever combining dense and BM25 via Reciprocal Rank Fusion (RRF).

    Fuses the ranked lists from both retrievers to produce a single
    result list that benefits from both semantic and keyword matching.
    """

    node_type = "hybrid_retriever"
    metadata = NodeMetadata(
        category=NodeCategory.RETRIEVER,
        display_name="Hybrid Retriever (RRF)",
        description=(
            "Combines dense vector retrieval and BM25 keyword retrieval "
            "via Reciprocal Rank Fusion. Typically outperforms either alone."
        ),
        icon_name="git-merge",
        tags=["retriever", "hybrid", "rrf", "dense", "bm25", "fusion"],
    )
    input_ports = [_QUERY_INPUT]
    output_ports = [_RETRIEVAL_RESULTS_OUTPUT]
    fields = [
        _TOP_K_FIELD,
        NodeField(
            name="dense_weight",
            field_type=FieldType.SLIDER,
            display_name="Dense Weight",
            description="Weight for the dense retriever in the fusion. BM25 weight = 1 - dense_weight.",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
        ),
        NodeField(
            name="rrf_k",
            field_type=FieldType.INTEGER,
            display_name="RRF K",
            description="RRF smoothing constant (default 60, as per the original paper).",
            default=60,
            min_value=1,
            max_value=200,
            advanced=True,
        ),
        NodeField(
            name="fetch_k",
            field_type=FieldType.INTEGER,
            display_name="Fetch K",
            description="Number of candidates to retrieve from each sub-retriever before fusion.",
            default=20,
            min_value=5,
            max_value=200,
            advanced=True,
        ),
        NodeField(
            name="vectorstore_type",
            field_type=FieldType.SELECT,
            display_name="Vector Store",
            description="Vector store used by the dense retriever.",
            default="chroma",
            options=["chroma", "qdrant", "faiss", "pinecone", "lancedb"],
            required=True,
        ),
        NodeField(
            name="collection_name",
            field_type=FieldType.STRING,
            display_name="Collection Name",
            default="ragforge",
            required=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.retrieval.hybrid import HybridRetriever

        query: str = inputs.get("query", "")
        retriever = HybridRetriever(
            vectorstore_type=self.require_config("vectorstore_type"),
            collection_name=self.get_config("collection_name", "ragforge"),
            dense_weight=self.get_config("dense_weight", 0.5),
            rrf_k=self.get_config("rrf_k", 60),
            fetch_k=self.get_config("fetch_k", 20),
        )
        results = await retriever.aretrieve(
            query=query,
            k=self.get_config("top_k", DEFAULT_RETRIEVAL_K),
        )
        return {"retrieval_results": results}


# --------------------------------------------------------------------------- #
# MMR Retriever
# --------------------------------------------------------------------------- #


@registry.register("mmr_retriever")
class MMRRetrieverNode(BaseNode):
    """Maximal Marginal Relevance retriever for diversity-aware retrieval."""

    node_type = "mmr_retriever"
    metadata = NodeMetadata(
        category=NodeCategory.RETRIEVER,
        display_name="MMR Retriever",
        description=(
            "Maximal Marginal Relevance retrieval. "
            "Balances relevance and diversity to avoid returning near-duplicate chunks."
        ),
        icon_name="shuffle",
        tags=["retriever", "mmr", "diversity", "maximal-marginal-relevance"],
    )
    input_ports = [_QUERY_INPUT]
    output_ports = [_RETRIEVAL_RESULTS_OUTPUT]
    fields = [
        _TOP_K_FIELD,
        NodeField(
            name="lambda_mult",
            field_type=FieldType.SLIDER,
            display_name="Lambda (diversity)",
            description="Trade-off between relevance (1.0) and diversity (0.0).",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
        ),
        NodeField(
            name="fetch_k",
            field_type=FieldType.INTEGER,
            display_name="Fetch K",
            description="Initial candidates to retrieve before MMR filtering.",
            default=20,
            min_value=5,
            max_value=200,
            advanced=True,
        ),
        NodeField(
            name="vectorstore_type",
            field_type=FieldType.SELECT,
            display_name="Vector Store",
            default="chroma",
            options=["chroma", "qdrant", "faiss", "pinecone", "lancedb"],
            required=True,
        ),
        NodeField(
            name="collection_name",
            field_type=FieldType.STRING,
            display_name="Collection Name",
            default="ragforge",
            required=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.retrieval.mmr import MMRRetriever

        query: str = inputs.get("query", "")
        retriever = MMRRetriever(
            vectorstore_type=self.require_config("vectorstore_type"),
            collection_name=self.get_config("collection_name", "ragforge"),
            lambda_mult=self.get_config("lambda_mult", 0.5),
            fetch_k=self.get_config("fetch_k", 20),
        )
        results = await retriever.aretrieve(
            query=query,
            k=self.get_config("top_k", DEFAULT_RETRIEVAL_K),
        )
        return {"retrieval_results": results}


# --------------------------------------------------------------------------- #
# Multi-Query Retriever
# --------------------------------------------------------------------------- #


@registry.register("multi_query_retriever")
class MultiQueryRetrieverNode(BaseNode):
    """
    Multi-query retriever.

    Uses an LLM to generate multiple rephrasings of the query,
    runs each through the dense retriever, and merges the results.
    Improves recall for queries that can be expressed in multiple ways.
    """

    node_type = "multi_query_retriever"
    metadata = NodeMetadata(
        category=NodeCategory.RETRIEVER,
        display_name="Multi-Query Retriever",
        description=(
            "Generates multiple query variants via an LLM, retrieves for each, "
            "then deduplicates and merges results. Improves recall."
        ),
        icon_name="layers",
        tags=["retriever", "multi-query", "llm", "recall"],
    )
    input_ports = [_QUERY_INPUT]
    output_ports = [_RETRIEVAL_RESULTS_OUTPUT]
    fields = [
        _TOP_K_FIELD,
        NodeField(
            name="num_queries",
            field_type=FieldType.SLIDER,
            display_name="Query Variants",
            description="Number of query variants to generate.",
            default=3,
            min_value=2,
            max_value=10,
            step=1,
        ),
        NodeField(
            name="llm_provider",
            field_type=FieldType.SELECT,
            display_name="LLM Provider",
            description="LLM to use for query generation.",
            default="openai",
            options=["openai", "anthropic", "google", "groq", "ollama"],
        ),
        NodeField(
            name="llm_model",
            field_type=FieldType.STRING,
            display_name="LLM Model",
            default="gpt-4o-mini",
        ),
        NodeField(
            name="vectorstore_type",
            field_type=FieldType.SELECT,
            display_name="Vector Store",
            default="chroma",
            options=["chroma", "qdrant", "faiss", "pinecone", "lancedb"],
            required=True,
        ),
        NodeField(
            name="collection_name",
            field_type=FieldType.STRING,
            display_name="Collection Name",
            default="ragforge",
            required=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.retrieval.multi_query import MultiQueryRetriever

        query: str = inputs.get("query", "")
        retriever = MultiQueryRetriever(
            vectorstore_type=self.require_config("vectorstore_type"),
            collection_name=self.get_config("collection_name", "ragforge"),
            num_queries=self.get_config("num_queries", 3),
            llm_provider=self.get_config("llm_provider", "openai"),
            llm_model=self.get_config("llm_model", "gpt-4o-mini"),
        )
        results = await retriever.aretrieve(
            query=query,
            k=self.get_config("top_k", DEFAULT_RETRIEVAL_K),
        )
        return {"retrieval_results": results}
