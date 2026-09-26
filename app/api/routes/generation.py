"""
Standalone generation playground endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.domain.generation.models import GenerationResult
from app.generation.manager import GenerationManager

router = APIRouter(prefix="/generation", tags=["Generation Playground"])


class GenerationTestRequest(BaseModel):
    prompt: str
    provider: str = "openai"
    params: dict[str, Any] = Field(default_factory=dict)


@router.post("/test", summary="Test prompt generation directly with an LLM")
async def test_generation(payload: GenerationTestRequest) -> dict[str, Any]:
    """
    Directly query an LLM provider (OpenAI, Anthropic, Google, Groq, Ollama, Mistral) with a prompt.
    """
    try:
        generator = GenerationManager.get(payload.provider, **payload.params)
        result = await generator.agenerate(payload.prompt)
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Generation failed: {exc}",
        ) from exc
