"""
Utility nodes for RAGForge pipelines.

Nodes registered here:
  - prompt_template      (render a Jinja2 prompt with context + query)
  - context_builder      (format retrieval results into a context string)
  - query_rewriter       (LLM-based query expansion / rewriting)
  - text_cleaner         (whitespace normalisation, deduplication)
  - conditional_router   (route flow based on a condition)
  - output_parser        (parse LLM output as JSON/Markdown)
"""

from __future__ import annotations

from typing import Any

from app.core.constants import FieldType, NodeCategory, PortType
from app.core.registry import registry
from app.nodes.base import BaseNode, NodeField, NodeMetadata, NodePortDefinition


# --------------------------------------------------------------------------- #
# Prompt Template
# --------------------------------------------------------------------------- #


@registry.register("prompt_template")
class PromptTemplateNode(BaseNode):
    """
    Renders a Jinja2 prompt template with context and query substituted in.

    The rendered string is passed to a generator node via the ``prompt`` port.
    """

    node_type = "prompt_template"
    metadata = NodeMetadata(
        category=NodeCategory.PROMPT,
        display_name="Prompt Template",
        description="Renders a Jinja2 prompt template, substituting {context} and {query} placeholders.",
        icon_name="file-code",
        tags=["prompt", "template", "jinja2", "context"],
    )
    input_ports = [
        NodePortDefinition(
            name="context",
            port_type=PortType.ANY,
            display_name="Context",
            description="Context string from a Context Builder node.",
            required=True,
        ),
        NodePortDefinition(
            name="query",
            port_type=PortType.QUERY,
            display_name="Query",
            description="The user's query.",
            required=True,
        ),
    ]
    output_ports = [
        NodePortDefinition(
            name="prompt",
            port_type=PortType.PROMPT,
            display_name="Prompt",
            description="The fully rendered prompt string.",
        ),
    ]
    fields = [
        NodeField(
            name="template",
            field_type=FieldType.CODE,
            display_name="Template",
            description="Jinja2 template. Use {{ context }} and {{ query }} as placeholders.",
            required=True,
            default=(
                "You are a helpful assistant. Use the following context to answer the question.\n\n"
                "Context:\n{{ context }}\n\n"
                "Question: {{ query }}\n\n"
                "Answer:"
            ),
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        try:
            from jinja2 import Template
        except ImportError:
            # Fallback: simple str.format substitution
            template_str: str = self.require_config("template")
            prompt = (
                template_str
                .replace("{{ context }}", str(inputs.get("context", "")))
                .replace("{{ query }}", str(inputs.get("query", "")))
            )
            return {"prompt": prompt}

        template = Template(self.require_config("template"))
        rendered = template.render(
            context=inputs.get("context", ""),
            query=inputs.get("query", ""),
        )
        return {"prompt": rendered}


# --------------------------------------------------------------------------- #
# Context Builder
# --------------------------------------------------------------------------- #


@registry.register("context_builder")
class ContextBuilderNode(BaseNode):
    """
    Formats a list of retrieval results into a context string for the prompt.

    Concatenates chunk content with optional source attribution.
    """

    node_type = "context_builder"
    metadata = NodeMetadata(
        category=NodeCategory.PROMPT,
        display_name="Context Builder",
        description="Formats retrieved chunks into a readable context string for the prompt template.",
        icon_name="layout",
        tags=["context", "format", "retrieval", "prompt"],
    )
    input_ports = [
        NodePortDefinition(
            name="retrieval_results",
            port_type=PortType.RETRIEVAL_RESULTS,
            display_name="Retrieval Results",
            required=True,
        ),
    ]
    output_ports = [
        NodePortDefinition(
            name="context",
            port_type=PortType.ANY,
            display_name="Context String",
            description="Formatted context string.",
        ),
    ]
    fields = [
        NodeField(
            name="format",
            field_type=FieldType.SELECT,
            display_name="Format",
            description="How to format each retrieved chunk.",
            default="numbered",
            options=["numbered", "bullet", "plain", "xml_tags"],
        ),
        NodeField(
            name="include_source",
            field_type=FieldType.BOOLEAN,
            display_name="Include Source",
            description="Add source file name or URL below each chunk.",
            default=True,
        ),
        NodeField(
            name="separator",
            field_type=FieldType.STRING,
            display_name="Separator",
            description="String inserted between consecutive chunks.",
            default="\n\n---\n\n",
            advanced=True,
        ),
        NodeField(
            name="max_chars",
            field_type=FieldType.INTEGER,
            display_name="Max Total Characters",
            description="Truncate context to this many characters. 0 = no limit.",
            default=0,
            min_value=0,
            max_value=100000,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        results = inputs.get("retrieval_results", [])
        fmt = self.get_config("format", "numbered")
        include_source = self.get_config("include_source", True)
        separator = self.get_config("separator", "\n\n---\n\n")
        max_chars = self.get_config("max_chars", 0)

        parts: list[str] = []
        for i, result in enumerate(results, 1):
            content = result.content
            source = result.source or ""

            if fmt == "numbered":
                part = f"[{i}] {content}"
            elif fmt == "bullet":
                part = f"- {content}"
            elif fmt == "xml_tags":
                part = f"<chunk id=\"{i}\">\n{content}\n</chunk>"
            else:
                part = content

            if include_source and source:
                part += f"\nSource: {source}"

            parts.append(part)

        context = separator.join(parts)

        if max_chars and len(context) > max_chars:
            context = context[:max_chars] + "\n[... context truncated ...]"

        return {"context": context}


# --------------------------------------------------------------------------- #
# Query Rewriter
# --------------------------------------------------------------------------- #


@registry.register("query_rewriter")
class QueryRewriterNode(BaseNode):
    """
    Rewrites or expands the user query using an LLM before retrieval.

    Useful for fixing poorly phrased queries or making them more specific.
    """

    node_type = "query_rewriter"
    metadata = NodeMetadata(
        category=NodeCategory.UTILITY,
        display_name="Query Rewriter",
        description="Uses an LLM to rephrase, expand, or decompose the user query before retrieval. Improves retrieval quality.",
        icon_name="edit",
        tags=["query", "rewrite", "expand", "llm", "utility"],
    )
    input_ports = [
        NodePortDefinition(
            name="query",
            port_type=PortType.QUERY,
            display_name="Query",
            required=True,
        ),
    ]
    output_ports = [
        NodePortDefinition(
            name="query",
            port_type=PortType.QUERY,
            display_name="Rewritten Query",
        ),
    ]
    fields = [
        NodeField(
            name="strategy",
            field_type=FieldType.SELECT,
            display_name="Strategy",
            description="How to rewrite the query.",
            default="rephrase",
            options=["rephrase", "expand", "step_back", "decompose"],
        ),
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
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.nodes.query_rewriter import _rewrite_query

        rewritten = await _rewrite_query(
            query=inputs.get("query", ""),
            strategy=self.get_config("strategy", "rephrase"),
            llm_provider=self.get_config("llm_provider", "openai"),
            llm_model=self.get_config("llm_model", "gpt-4o-mini"),
        )
        return {"query": rewritten}


# --------------------------------------------------------------------------- #
# Text Cleaner
# --------------------------------------------------------------------------- #


@registry.register("text_cleaner")
class TextCleanerNode(BaseNode):
    """Normalise and clean document text before chunking."""

    node_type = "text_cleaner"
    metadata = NodeMetadata(
        category=NodeCategory.UTILITY,
        display_name="Text Cleaner",
        description="Normalises whitespace, removes boilerplate, and deduplicates repeated lines in document text.",
        icon_name="eraser",
        tags=["clean", "normalise", "preprocess", "utility"],
    )
    input_ports = [
        NodePortDefinition(
            name="documents",
            port_type=PortType.DOCUMENTS,
            display_name="Documents",
            required=True,
        ),
    ]
    output_ports = [
        NodePortDefinition(
            name="documents",
            port_type=PortType.DOCUMENTS,
            display_name="Cleaned Documents",
        ),
    ]
    fields = [
        NodeField(
            name="normalise_whitespace",
            field_type=FieldType.BOOLEAN,
            display_name="Normalise Whitespace",
            description="Collapse multiple whitespace characters into one.",
            default=True,
        ),
        NodeField(
            name="remove_duplicate_lines",
            field_type=FieldType.BOOLEAN,
            display_name="Remove Duplicate Lines",
            description="Remove repeated adjacent lines.",
            default=False,
            advanced=True,
        ),
        NodeField(
            name="min_line_length",
            field_type=FieldType.INTEGER,
            display_name="Min Line Length",
            description="Drop lines shorter than this character count. 0 = keep all.",
            default=0,
            min_value=0,
            max_value=100,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        import re

        documents = inputs.get("documents", [])
        normalise = self.get_config("normalise_whitespace", True)
        remove_dups = self.get_config("remove_duplicate_lines", False)
        min_len = self.get_config("min_line_length", 0)

        cleaned = []
        for doc in documents:
            text = doc.content

            if normalise:
                text = re.sub(r"[ \t]+", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = text.strip()

            if remove_dups or min_len > 0:
                lines = text.splitlines()
                seen: set[str] = set()
                result_lines: list[str] = []
                for line in lines:
                    if min_len > 0 and len(line.strip()) < min_len and line.strip():
                        continue
                    if remove_dups:
                        if line in seen:
                            continue
                        seen.add(line)
                    result_lines.append(line)
                text = "\n".join(result_lines)

            cleaned.append(doc.model_copy(update={"content": text}))

        return {"documents": cleaned}
