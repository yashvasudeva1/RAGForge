"""
Standalone retrieval playground endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.domain.retrieval.models import RetrievalResult
from app.retrieval.manager import RetrievalManager

router = APIRouter(prefix="/retrieval", tags=["Retrieval Playground"])


class RetrievalTestRequest(BaseModel):
    query: str
    strategy: str = "dense"
    k: int = 5
    params: dict[str, Any] = Field(default_factory=dict)


@router.post("/test", summary="Test retrieval on a collection")
async def test_retrieval(payload: RetrievalTestRequest) -> list[dict[str, Any]]:
    """
    Test retrieval strategy on a vector store collection or BM25 index.
    """
    try:
        retriever = RetrievalManager.get(payload.strategy, **payload.params)
        results = await retriever.aretrieve(payload.query, k=payload.k)
        return [r.model_dump() for r in results]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Retrieval query failed: {exc}",
        ) from exc
