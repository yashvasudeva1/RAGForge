"""
Reranker nodes for RAGForge.

Nodes registered here:
  - cross_encoder_reranker
  - cohere_reranker
  - voyage_reranker
  - llm_reranker
"""

from __future__ import annotations

from typing import Any

from app.core.constants import DEFAULT_RERANK_TOP_N, FieldType, NodeCategory, PortType
from app.core.registry import registry
from app.nodes.base import BaseNode, NodeField, NodeMetadata, NodePortDefinition

_RETRIEVAL_RESULTS_INPUT = NodePortDefinition(
    name="retrieval_results",
    port_type=PortType.RETRIEVAL_RESULTS,
    display_name="Retrieval Results",
    description="Candidate chunks to rerank.",
    required=True,
)
_QUERY_INPUT = NodePortDefinition(
    name="query",
    port_type=PortType.QUERY,
    display_name="Query",
    description="The original user query used to score relevance.",
    required=True,
)
_RERANKED_OUTPUT = NodePortDefinition(
    name="retrieval_results",
    port_type=PortType.RETRIEVAL_RESULTS,
    display_name="Reranked Results",
    description="Reranked chunks — most relevant first.",
    required=False,
)
_TOP_N_FIELD = NodeField(
    name="top_n",
    field_type=FieldType.SLIDER,
    display_name="Top N",
    description="Number of chunks to keep after reranking.",
    default=DEFAULT_RERANK_TOP_N,
    min_value=1,
    max_value=20,
    step=1,
)


# --------------------------------------------------------------------------- #
# Cross-Encoder Reranker
# --------------------------------------------------------------------------- #


@registry.register("cross_encoder_reranker")
class CrossEncoderRerankerNode(BaseNode):
    """
    Cross-encoder reranker using sentence-transformers.

    Runs the query and each candidate chunk through a cross-encoder model
    jointly, producing a fine-grained relevance score.  More accurate than
    bi-encoder retrieval but slower (O(n) model calls per query).
    """

    node_type = "cross_encoder_reranker"
    metadata = NodeMetadata(
        category=NodeCategory.RERANKER,
        display_name="Cross-Encoder Reranker",
        description=(
            "Reranks retrieval results using a local cross-encoder model. "
            "More accurate than bi-encoder retrieval. Runs locally, no API key needed."
        ),
        icon_name="sort-desc",
        tags=["reranker", "cross-encoder", "local", "sentence-transformers"],
    )
    input_ports = [_QUERY_INPUT, _RETRIEVAL_RESULTS_INPUT]
    output_ports = [_RERANKED_OUTPUT]
    fields = [
        _TOP_N_FIELD,
        NodeField(
            name="model",
            field_type=FieldType.STRING,
            display_name="Model",
            description="sentence-transformers cross-encoder model name.",
            default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        ),
        NodeField(
            name="device",
            field_type=FieldType.SELECT,
            display_name="Device",
            description="Compute device.",
            default="cpu",
            options=["cpu", "cuda", "mps"],
            advanced=True,
        ),
        NodeField(
            name="batch_size",
            field_type=FieldType.INTEGER,
            display_name="Batch Size",
            description="Number of (query, chunk) pairs to score per forward pass.",
            default=16,
            min_value=1,
            max_value=128,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.reranking.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker(
            model=self.get_config("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            device=self.get_config("device", "cpu"),
            batch_size=self.get_config("batch_size", 16),
        )
        results = await reranker.arerank(
            query=inputs.get("query", ""),
            results=inputs.get("retrieval_results", []),
            top_n=self.get_config("top_n", DEFAULT_RERANK_TOP_N),
        )
        return {"retrieval_results": results}


# --------------------------------------------------------------------------- #
# Cohere Reranker
# --------------------------------------------------------------------------- #


@registry.register("cohere_reranker")
class CohereRerankerNode(BaseNode):
    """Reranker using Cohere's Rerank v3 API."""

    node_type = "cohere_reranker"
    metadata = NodeMetadata(
        category=NodeCategory.RERANKER,
        display_name="Cohere Reranker",
        description="Reranks retrieval results using Cohere Rerank v3. Highly effective for English and multilingual content.",
        icon_name="sort-desc",
        tags=["reranker", "cohere", "api", "neural"],
    )
    input_ports = [_QUERY_INPUT, _RETRIEVAL_RESULTS_INPUT]
    output_ports = [_RERANKED_OUTPUT]
    fields = [
        _TOP_N_FIELD,
        NodeField(
            name="model",
            field_type=FieldType.SELECT,
            display_name="Model",
            description="Cohere rerank model.",
            default="rerank-english-v3.0",
            options=["rerank-english-v3.0", "rerank-multilingual-v3.0", "rerank-english-v2.0"],
        ),
        NodeField(
            name="api_key",
            field_type=FieldType.PASSWORD,
            display_name="API Key",
            description="Cohere API key. Falls back to COHERE_API_KEY.",
            default=None,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.reranking.cohere import CohereReranker

        reranker = CohereReranker(
            model=self.get_config("model", "rerank-english-v3.0"),
            api_key=self.get_config("api_key"),
        )
        results = await reranker.arerank(
            query=inputs.get("query", ""),
            results=inputs.get("retrieval_results", []),
            top_n=self.get_config("top_n", DEFAULT_RERANK_TOP_N),
        )
        return {"retrieval_results": results}


# --------------------------------------------------------------------------- #
# VoyageAI Reranker
# --------------------------------------------------------------------------- #


@registry.register("voyage_reranker")
class VoyageRerankerNode(BaseNode):
    """Reranker using VoyageAI's rerank API."""

    node_type = "voyage_reranker"
    metadata = NodeMetadata(
        category=NodeCategory.RERANKER,
        display_name="Voyage Reranker",
        description="Reranks retrieval results using VoyageAI's rerank model.",
        icon_name="sort-desc",
        tags=["reranker", "voyage", "voyageai", "api"],
    )
    input_ports = [_QUERY_INPUT, _RETRIEVAL_RESULTS_INPUT]
    output_ports = [_RERANKED_OUTPUT]
    fields = [
        _TOP_N_FIELD,
        NodeField(
            name="model",
            field_type=FieldType.SELECT,
            display_name="Model",
            description="VoyageAI rerank model.",
            default="rerank-2",
            options=["rerank-2", "rerank-2-lite"],
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
        from app.reranking.voyage import VoyageReranker

        reranker = VoyageReranker(
            model=self.get_config("model", "rerank-2"),
            api_key=self.get_config("api_key"),
        )
        results = await reranker.arerank(
            query=inputs.get("query", ""),
            results=inputs.get("retrieval_results", []),
            top_n=self.get_config("top_n", DEFAULT_RERANK_TOP_N),
        )
        return {"retrieval_results": results}


# --------------------------------------------------------------------------- #
# LLM Reranker
# --------------------------------------------------------------------------- #


@registry.register("llm_reranker")
class LLMRerankerNode(BaseNode):
    """
    LLM-based pointwise reranker.

    Prompts an LLM to score each (query, chunk) pair on a relevance scale.
    More flexible than a dedicated rerank model but slower and more expensive.
    """

    node_type = "llm_reranker"
    metadata = NodeMetadata(
        category=NodeCategory.RERANKER,
        display_name="LLM Reranker",
        description=(
            "Uses an LLM to score the relevance of each retrieved chunk against the query. "
            "Flexible and model-agnostic but slower than dedicated rerank models."
        ),
        icon_name="sort-desc",
        tags=["reranker", "llm", "pointwise", "scoring"],
    )
    input_ports = [_QUERY_INPUT, _RETRIEVAL_RESULTS_INPUT]
    output_ports = [_RERANKED_OUTPUT]
    fields = [
        _TOP_N_FIELD,
        NodeField(
            name="llm_provider",
            field_type=FieldType.SELECT,
            display_name="LLM Provider",
            default="openai",
            options=["openai", "anthropic", "google", "groq"],
        ),
        NodeField(
            name="llm_model",
            field_type=FieldType.STRING,
            display_name="LLM Model",
            default="gpt-4o-mini",
        ),
        NodeField(
            name="scoring_prompt",
            field_type=FieldType.CODE,
            display_name="Scoring Prompt",
            description="Prompt template for scoring. Use {query} and {chunk} placeholders.",
            default=(
                "On a scale of 1-10, how relevant is the following text to the query?\n"
                "Query: {query}\nText: {chunk}\nRespond with only a number."
            ),
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.reranking.llm_reranker import LLMReranker

        reranker = LLMReranker(
            llm_provider=self.get_config("llm_provider", "openai"),
            llm_model=self.get_config("llm_model", "gpt-4o-mini"),
            scoring_prompt=self.get_config("scoring_prompt"),
        )
        results = await reranker.arerank(
            query=inputs.get("query", ""),
            results=inputs.get("retrieval_results", []),
            top_n=self.get_config("top_n", DEFAULT_RERANK_TOP_N),
        )
        return {"retrieval_results": results}
