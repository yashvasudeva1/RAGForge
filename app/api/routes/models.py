"""
Node schemas and component discovery endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.core.constants import NODE_CATEGORY_COLORS, NodeCategory
from app.core.registry import registry

router = APIRouter(prefix="/nodes", tags=["Nodes & Schemas"])


@router.get("", summary="Get all registered node schemas")
async def get_all_node_schemas() -> list[dict[str, Any]]:
    """
    Return JSON schemas for all registered nodes.
    Consumed by the frontend to render the drag-and-drop palette and dynamic config forms.
    """
    return registry.get_all_schemas()


@router.get("/categories", summary="List node categories with UI metadata")
async def get_node_categories() -> list[dict[str, Any]]:
    """
    Return all node categories with display labels and UI color codes.
    """
    schemas = registry.get_all_schemas()
    category_counts: dict[str, int] = {}
    for schema in schemas:
        cat = schema.get("category", "utility")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    result = []
    for cat in NodeCategory:
        result.append(
            {
                "id": cat.value,
                "label": cat.value.replace("_", " ").title(),
                "color": NODE_CATEGORY_COLORS.get(cat, "#64748b"),
                "node_count": category_counts.get(cat.value, 0),
            }
        )
    return result


@router.get("/{node_type}", summary="Get schema for a single node type")
async def get_node_schema(node_type: str) -> dict[str, Any]:
    """
    Return schema definition for a specific node type.
    """
    if not registry.has(node_type):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node type '{node_type}' not found in registry.",
        )
    return registry.get_schema(node_type)
