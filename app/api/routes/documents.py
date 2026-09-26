"""
Document upload, parsing, and chunking test endpoints.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies import get_ingestion_manager
from app.api.schemas.document import (
    ChunkTestRequest,
    ChunkTestResponse,
    DocumentLoadRequest,
    DocumentLoadResponse,
    DocumentUploadResponse,
)
from app.chunking.manager import ChunkingManager
from app.core.config import get_settings
from app.domain.documents.models import Document, DocumentMetadata
from app.ingestion.manager import IngestionManager

router = APIRouter(prefix="/documents", tags=["Documents & Ingestion"])


@router.post("/upload", summary="Upload a file to the server for pipeline ingestion")
async def upload_file(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """
    Accepts file uploads (PDF, DOCX, CSV, MD, TXT, etc.) and saves to the server data/uploads dir.
    Returns the file path for use in loader nodes.
    """
    settings = get_settings()
    upload_dir = Path(settings.pipeline_storage_dir).parent / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    safe_filename = Path(file.filename or "upload.bin").name
    target_path = upload_dir / safe_filename

    content = await file.read()
    target_path.write_bytes(content)

    return DocumentUploadResponse(
        filename=safe_filename,
        file_path=str(target_path.resolve()),
        size_bytes=len(content),
        content_type=file.content_type,
    )


@router.post("/load", summary="Parse a local file or URL into Document objects")
async def load_document(
    payload: DocumentLoadRequest,
    manager: IngestionManager = Depends(get_ingestion_manager),
) -> DocumentLoadResponse:
    """
    Parse a document from a file path or URL and return structured Document models with content.
    """
    try:
        docs = await manager.aload(payload.source)
        return DocumentLoadResponse(
            source=payload.source,
            count=len(docs),
            documents=docs,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load document: {exc}",
        ) from exc


@router.post("/chunk-test", summary="Test chunking on sample text")
async def test_chunking(payload: ChunkTestRequest) -> ChunkTestResponse:
    """
    Simulate chunking on sample text to preview chunks and test split parameters in the UI.
    """
    try:
        chunker = ChunkingManager.get(
            payload.strategy,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            **payload.extra_params,
        )
        dummy_doc = Document(
            content=payload.text,
            metadata=DocumentMetadata(source="playground_test"),
        )
        chunks = chunker.split(dummy_doc)
        return ChunkTestResponse(
            chunk_count=len(chunks),
            total_characters=len(payload.text),
            chunks=[c.model_dump() for c in chunks],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Chunking test failed: {exc}",
        ) from exc
