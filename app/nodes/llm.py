"""
LLM generator nodes for RAGForge.

Nodes registered here:
  - openai_generator
  - anthropic_generator
  - google_generator
  - groq_generator
  - ollama_generator
  - mistral_generator
"""

from __future__ import annotations

from typing import Any

from app.core.constants import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, FieldType, NodeCategory, PortType
from app.core.registry import registry
from app.nodes.base import BaseNode, NodeField, NodeMetadata, NodePortDefinition

_PROMPT_INPUT = NodePortDefinition(
    name="prompt",
    port_type=PortType.PROMPT,
    display_name="Prompt",
    description="The fully rendered prompt string to send to the LLM.",
    required=True,
)
_RETRIEVAL_RESULTS_INPUT = NodePortDefinition(
    name="retrieval_results",
    port_type=PortType.RETRIEVAL_RESULTS,
    display_name="Retrieval Results",
    description="Source chunks attached to the generation result for citation.",
    required=False,
)
_GENERATION_RESULT_OUTPUT = NodePortDefinition(
    name="generation_result",
    port_type=PortType.GENERATION_RESULT,
    display_name="Generation Result",
    description="The full generation result including answer, sources, and usage.",
    required=False,
)

_TEMPERATURE_FIELD = NodeField(
    name="temperature",
    field_type=FieldType.SLIDER,
    display_name="Temperature",
    description="Sampling temperature. 0 = deterministic, higher = more creative.",
    default=DEFAULT_TEMPERATURE,
    min_value=0.0,
    max_value=2.0,
    step=0.05,
)
_MAX_TOKENS_FIELD = NodeField(
    name="max_tokens",
    field_type=FieldType.INTEGER,
    display_name="Max Tokens",
    description="Maximum number of tokens in the generated response.",
    default=DEFAULT_MAX_TOKENS,
    min_value=64,
    max_value=32000,
)


# --------------------------------------------------------------------------- #
# OpenAI Generator
# --------------------------------------------------------------------------- #


@registry.register("openai_generator")
class OpenAIGeneratorNode(BaseNode):
    """Generate answers using OpenAI Chat Completions (GPT-4o, GPT-4o-mini, etc.)."""

    node_type = "openai_generator"
    metadata = NodeMetadata(
        category=NodeCategory.GENERATOR,
        display_name="OpenAI Generator",
        description="Generate answers using OpenAI GPT models (GPT-4o, GPT-4o-mini, o1, etc.).",
        icon_name="sparkles",
        tags=["generator", "openai", "gpt", "llm", "generation"],
    )
    input_ports = [_PROMPT_INPUT, _RETRIEVAL_RESULTS_INPUT]
    output_ports = [_GENERATION_RESULT_OUTPUT]
    fields = [
        NodeField(
            name="model",
            field_type=FieldType.SELECT,
            display_name="Model",
            default="gpt-4o-mini",
            options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o1-mini", "o3-mini"],
            required=True,
        ),
        _TEMPERATURE_FIELD,
        _MAX_TOKENS_FIELD,
        NodeField(
            name="api_key",
            field_type=FieldType.PASSWORD,
            display_name="API Key",
            description="OpenAI API key. Falls back to OPENAI_API_KEY.",
            default=None,
            advanced=True,
        ),
        NodeField(
            name="system_prompt",
            field_type=FieldType.TEXT,
            display_name="System Prompt",
            description="Optional system message prepended to every request.",
            default="You are a helpful assistant. Answer based on the provided context.",
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.generation.openai import OpenAIGenerator

        generator = OpenAIGenerator(
            model=self.get_config("model", "gpt-4o-mini"),
            api_key=self.get_config("api_key"),
            temperature=self.get_config("temperature", DEFAULT_TEMPERATURE),
            max_tokens=self.get_config("max_tokens", DEFAULT_MAX_TOKENS),
            system_prompt=self.get_config("system_prompt"),
        )
        result = await generator.agenerate(
            prompt=inputs.get("prompt", ""),
            source_chunks=inputs.get("retrieval_results", []),
        )
        return {"generation_result": result}


# --------------------------------------------------------------------------- #
# Anthropic Generator
# --------------------------------------------------------------------------- #


@registry.register("anthropic_generator")
class AnthropicGeneratorNode(BaseNode):
    """Generate answers using Anthropic Claude models."""

    node_type = "anthropic_generator"
    metadata = NodeMetadata(
        category=NodeCategory.GENERATOR,
        display_name="Anthropic Generator",
        description="Generate answers using Anthropic Claude models (Claude 3.5 Sonnet, Claude 3 Haiku, etc.).",
        icon_name="sparkles",
        tags=["generator", "anthropic", "claude", "llm"],
    )
    input_ports = [_PROMPT_INPUT, _RETRIEVAL_RESULTS_INPUT]
    output_ports = [_GENERATION_RESULT_OUTPUT]
    fields = [
        NodeField(
            name="model",
            field_type=FieldType.SELECT,
            display_name="Model",
            default="claude-3-5-sonnet-20241022",
            options=[
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
            ],
            required=True,
        ),
        _TEMPERATURE_FIELD,
        _MAX_TOKENS_FIELD,
        NodeField(
            name="api_key",
            field_type=FieldType.PASSWORD,
            display_name="API Key",
            description="Anthropic API key. Falls back to ANTHROPIC_API_KEY.",
            default=None,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.generation.anthropic import AnthropicGenerator

        generator = AnthropicGenerator(
            model=self.get_config("model", "claude-3-5-sonnet-20241022"),
            api_key=self.get_config("api_key"),
            temperature=self.get_config("temperature", DEFAULT_TEMPERATURE),
            max_tokens=self.get_config("max_tokens", DEFAULT_MAX_TOKENS),
        )
        result = await generator.agenerate(
            prompt=inputs.get("prompt", ""),
            source_chunks=inputs.get("retrieval_results", []),
        )
        return {"generation_result": result}


# --------------------------------------------------------------------------- #
# Google Generator
# --------------------------------------------------------------------------- #


@registry.register("google_generator")
class GoogleGeneratorNode(BaseNode):
    """Generate answers using Google Gemini models."""

    node_type = "google_generator"
    metadata = NodeMetadata(
        category=NodeCategory.GENERATOR,
        display_name="Google Gemini Generator",
        description="Generate answers using Google Gemini models (Gemini 1.5 Pro, Flash, etc.).",
        icon_name="sparkles",
        tags=["generator", "google", "gemini", "llm"],
    )
    input_ports = [_PROMPT_INPUT, _RETRIEVAL_RESULTS_INPUT]
    output_ports = [_GENERATION_RESULT_OUTPUT]
    fields = [
        NodeField(
            name="model",
            field_type=FieldType.SELECT,
            display_name="Model",
            default="gemini-2.0-flash",
            options=["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro", "gemini-1.5-flash"],
            required=True,
        ),
        _TEMPERATURE_FIELD,
        _MAX_TOKENS_FIELD,
        NodeField(
            name="api_key",
            field_type=FieldType.PASSWORD,
            display_name="API Key",
            description="Google API key. Falls back to GOOGLE_API_KEY.",
            default=None,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.generation.google import GoogleGenerator

        generator = GoogleGenerator(
            model=self.get_config("model", "gemini-2.0-flash"),
            api_key=self.get_config("api_key"),
            temperature=self.get_config("temperature", DEFAULT_TEMPERATURE),
            max_tokens=self.get_config("max_tokens", DEFAULT_MAX_TOKENS),
        )
        result = await generator.agenerate(
            prompt=inputs.get("prompt", ""),
            source_chunks=inputs.get("retrieval_results", []),
        )
        return {"generation_result": result}


# --------------------------------------------------------------------------- #
# Groq Generator
# --------------------------------------------------------------------------- #


@registry.register("groq_generator")
class GroqGeneratorNode(BaseNode):
    """Generate answers using Groq's ultra-fast inference API."""

    node_type = "groq_generator"
    metadata = NodeMetadata(
        category=NodeCategory.GENERATOR,
        display_name="Groq Generator",
        description="Generate answers via Groq's ultra-fast inference (Llama, Mixtral, etc.). Extremely low latency.",
        icon_name="zap",
        tags=["generator", "groq", "llama", "mixtral", "fast", "llm"],
    )
    input_ports = [_PROMPT_INPUT, _RETRIEVAL_RESULTS_INPUT]
    output_ports = [_GENERATION_RESULT_OUTPUT]
    fields = [
        NodeField(
            name="model",
            field_type=FieldType.SELECT,
            display_name="Model",
            default="llama-3.3-70b-versatile",
            options=[
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ],
            required=True,
        ),
        _TEMPERATURE_FIELD,
        _MAX_TOKENS_FIELD,
        NodeField(
            name="api_key",
            field_type=FieldType.PASSWORD,
            display_name="API Key",
            description="Groq API key. Falls back to GROQ_API_KEY.",
            default=None,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.generation.groq import GroqGenerator

        generator = GroqGenerator(
            model=self.get_config("model", "llama-3.3-70b-versatile"),
            api_key=self.get_config("api_key"),
            temperature=self.get_config("temperature", DEFAULT_TEMPERATURE),
            max_tokens=self.get_config("max_tokens", DEFAULT_MAX_TOKENS),
        )
        result = await generator.agenerate(
            prompt=inputs.get("prompt", ""),
            source_chunks=inputs.get("retrieval_results", []),
        )
        return {"generation_result": result}


# --------------------------------------------------------------------------- #
# Ollama Generator (local)
# --------------------------------------------------------------------------- #


@registry.register("ollama_generator")
class OllamaGeneratorNode(BaseNode):
    """
    Generate answers using locally-running Ollama models.

    Requires Ollama to be running (http://localhost:11434 by default).
    No API key needed.
    """

    node_type = "ollama_generator"
    metadata = NodeMetadata(
        category=NodeCategory.GENERATOR,
        display_name="Ollama Generator",
        description=(
            "Generate answers using models running locally via Ollama. "
            "No API key needed. Supports Llama, Mistral, Phi, Gemma, and more."
        ),
        icon_name="cpu",
        tags=["generator", "ollama", "local", "llama", "offline", "llm"],
    )
    input_ports = [_PROMPT_INPUT, _RETRIEVAL_RESULTS_INPUT]
    output_ports = [_GENERATION_RESULT_OUTPUT]
    fields = [
        NodeField(
            name="model",
            field_type=FieldType.STRING,
            display_name="Model",
            description="Ollama model name (e.g. 'llama3.2', 'mistral', 'phi3').",
            default="llama3.2",
            required=True,
        ),
        _TEMPERATURE_FIELD,
        _MAX_TOKENS_FIELD,
        NodeField(
            name="base_url",
            field_type=FieldType.STRING,
            display_name="Ollama Base URL",
            description="Ollama server URL.",
            default="http://localhost:11434",
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.generation.ollama import OllamaGenerator

        generator = OllamaGenerator(
            model=self.get_config("model", "llama3.2"),
            base_url=self.get_config("base_url", "http://localhost:11434"),
            temperature=self.get_config("temperature", DEFAULT_TEMPERATURE),
            max_tokens=self.get_config("max_tokens", DEFAULT_MAX_TOKENS),
        )
        result = await generator.agenerate(
            prompt=inputs.get("prompt", ""),
            source_chunks=inputs.get("retrieval_results", []),
        )
        return {"generation_result": result}


# --------------------------------------------------------------------------- #
# Mistral Generator
# --------------------------------------------------------------------------- #


@registry.register("mistral_generator")
class MistralGeneratorNode(BaseNode):
    """Generate answers using Mistral AI models."""

    node_type = "mistral_generator"
    metadata = NodeMetadata(
        category=NodeCategory.GENERATOR,
        display_name="Mistral Generator",
        description="Generate answers using Mistral AI models (Mistral Large, Mistral Small, Codestral, etc.).",
        icon_name="sparkles",
        tags=["generator", "mistral", "llm"],
    )
    input_ports = [_PROMPT_INPUT, _RETRIEVAL_RESULTS_INPUT]
    output_ports = [_GENERATION_RESULT_OUTPUT]
    fields = [
        NodeField(
            name="model",
            field_type=FieldType.SELECT,
            display_name="Model",
            default="mistral-small-latest",
            options=["mistral-large-latest", "mistral-small-latest", "mistral-nemo", "codestral-latest"],
            required=True,
        ),
        _TEMPERATURE_FIELD,
        _MAX_TOKENS_FIELD,
        NodeField(
            name="api_key",
            field_type=FieldType.PASSWORD,
            display_name="API Key",
            description="Mistral API key. Falls back to MISTRAL_API_KEY.",
            default=None,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.generation.mistral import MistralGenerator

        generator = MistralGenerator(
            model=self.get_config("model", "mistral-small-latest"),
            api_key=self.get_config("api_key"),
            temperature=self.get_config("temperature", DEFAULT_TEMPERATURE),
            max_tokens=self.get_config("max_tokens", DEFAULT_MAX_TOKENS),
        )
        result = await generator.agenerate(
            prompt=inputs.get("prompt", ""),
            source_chunks=inputs.get("retrieval_results", []),
        )
        return {"generation_result": result}
