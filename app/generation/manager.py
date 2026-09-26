"""
Generation manager — factory for all LLM generator types.
"""

from __future__ import annotations

from app.core.constants import LLMProvider
from app.core.exceptions import RAGForgeError
from app.generation.base import BaseGenerator


class GenerationManager:
    """
    Factory that returns the correct generator by LLM provider name.

    Usage
    -----
        generator = GenerationManager.get("openai", model="gpt-4o-mini")
        result = generator.generate(prompt, source_chunks=retrieved)
    """

    @staticmethod
    def get(provider: str, **kwargs) -> BaseGenerator:
        from app.generation.anthropic import AnthropicGenerator
        from app.generation.google import GoogleGenerator
        from app.generation.groq import GroqGenerator
        from app.generation.mistral import MistralGenerator
        from app.generation.ollama import OllamaGenerator
        from app.generation.openai import OpenAIGenerator

        registry: dict[str, type[BaseGenerator]] = {
            LLMProvider.OPENAI: OpenAIGenerator,
            LLMProvider.ANTHROPIC: AnthropicGenerator,
            LLMProvider.GOOGLE: GoogleGenerator,
            LLMProvider.GROQ: GroqGenerator,
            LLMProvider.OLLAMA: OllamaGenerator,
            LLMProvider.MISTRAL: MistralGenerator,
        }

        cls = registry.get(provider)
        if cls is None:
            available = list(registry.keys())
            raise RAGForgeError(
                f"Unknown LLM provider '{provider}'. Available: {available}",
                details={"provider": provider, "available": available},
            )
        return cls(**kwargs)
