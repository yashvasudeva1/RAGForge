"""
Input and Output nodes for RAGForge pipelines.

QueryInputNode  — the entry point for user queries.
AnswerOutputNode — the final output node that surfaces the generated answer.
"""

from __future__ import annotations

from typing import Any

from app.core.constants import FieldType, NodeCategory, PortType
from app.core.registry import registry
from app.nodes.base import BaseNode, NodeField, NodeMetadata, NodePortDefinition


@registry.register("query_input")
class QueryInputNode(BaseNode):
    """
    Entry point for the pipeline.

    The user types a question here.  The query string is passed downstream
    to retriever nodes, query rewriter nodes, and the prompt template node.

    This node has no input ports — it is always the start of the graph.
    """

    node_type = "query_input"
    metadata = NodeMetadata(
        category=NodeCategory.INPUT,
        display_name="Query Input",
        description=(
            "The starting point of any RAG pipeline. "
            "Accepts a user question and emits it as a query string."
        ),
        icon_name="message-circle",
        tags=["input", "query", "question", "start"],
    )
    input_ports: list[NodePortDefinition] = []
    output_ports = [
        NodePortDefinition(
            name="query",
            port_type=PortType.QUERY,
            display_name="Query",
            description="The user's question passed downstream.",
            required=False,
        ),
    ]
    fields = [
        NodeField(
            name="query",
            field_type=FieldType.TEXT,
            display_name="Query",
            description="The question or instruction to send through the pipeline.",
            required=True,
            placeholder="Ask a question about your documents...",
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        """Emit the configured query string."""
        query: str = self.require_config("query")
        return {"query": query}


@registry.register("answer_output")
class AnswerOutputNode(BaseNode):
    """
    Terminal node that receives and surfaces the final generated answer.

    In the canvas this node is rendered with an expanded output panel
    showing the answer text and the source citations.
    """

    node_type = "answer_output"
    metadata = NodeMetadata(
        category=NodeCategory.OUTPUT,
        display_name="Answer Output",
        description=(
            "The terminal node of a RAG pipeline. "
            "Displays the generated answer and source citations."
        ),
        icon_name="message-square",
        tags=["output", "answer", "result", "end"],
    )
    input_ports = [
        NodePortDefinition(
            name="generation_result",
            port_type=PortType.GENERATION_RESULT,
            display_name="Generation Result",
            description="The full generation result from a generator node.",
            required=False,
        ),
        NodePortDefinition(
            name="answer",
            port_type=PortType.ANSWER,
            display_name="Answer",
            description="A plain answer string (alternative to full generation result).",
            required=False,
        ),
    ]
    output_ports: list[NodePortDefinition] = []
    fields = [
        NodeField(
            name="show_sources",
            field_type=FieldType.BOOLEAN,
            display_name="Show Sources",
            description="Display source document citations below the answer.",
            default=True,
        ),
        NodeField(
            name="max_sources",
            field_type=FieldType.INTEGER,
            display_name="Max Sources Shown",
            description="Maximum number of source citations to display.",
            default=5,
            min_value=1,
            max_value=20,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        """
        Pass through the generation result for the frontend to render.

        Returns the answer string and source list for the output panel.
        """
        generation_result = inputs.get("generation_result")
        answer_str = inputs.get("answer")

        if generation_result is not None:
            answer = generation_result.answer
            sources = generation_result.unique_sources
        elif answer_str is not None:
            answer = answer_str
            sources = []
        else:
            answer = ""
            sources = []

        show_sources = self.get_config("show_sources", True)
        max_sources = self.get_config("max_sources", 5)

        return {
            "answer": answer,
            "sources": sources[: max_sources] if show_sources else [],
        }
