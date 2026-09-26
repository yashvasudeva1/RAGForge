"""
Parent-child chunker.

Produces small child chunks for precise retrieval and larger parent
chunks for rich generation context.  Child chunks carry a
``parent_chunk_id`` reference so the context builder can retrieve the
full parent when needed.
"""

from __future__ import annotations

from app.chunking.base import BaseChunker
from app.chunking.recursive import RecursiveChunker
from app.domain.chunks.models import Chunk
from app.domain.documents.models import Document


class ParentChildChunker(BaseChunker):
    """
    Hierarchical chunker producing parent and child chunks.

    Parameters
    ----------
    parent_chunk_size:
        Size of the larger parent chunks (characters).
    child_chunk_size:
        Size of the smaller child chunks used for retrieval.
    child_overlap:
        Overlap between consecutive child chunks.
    """

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 400,
        child_overlap: int = 50,
    ) -> None:
        self.parent_chunker = RecursiveChunker(
            chunk_size=parent_chunk_size,
            chunk_overlap=0,
        )
        self.child_chunker = RecursiveChunker(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
        )

    def split(self, document: Document) -> list[Chunk]:
        """
        Return child chunks only (for retrieval).

        Each child chunk has ``metadata.parent_chunk_id`` set to the ID of
        its containing parent chunk.  The parent chunks themselves are NOT
        returned here but can be fetched by ID from the vector store or
        chunk store when building the generation context.
        """
        parent_chunks = self.parent_chunker.split(document)
        all_child_chunks: list[Chunk] = []
        child_idx = 0

        for parent in parent_chunks:
            # Re-use Document wrapper for the parent's content
            from app.domain.documents.models import Document as Doc
            from app.domain.documents.models import DocumentMetadata

            parent_doc = Doc(
                content=parent.content,
                metadata=DocumentMetadata(source=document.metadata.source),
            )
            children = self.child_chunker.split(parent_doc)

            for child in children:
                child.metadata.document_id = document.id
                child.metadata.document_source = document.metadata.source
                child.metadata.parent_chunk_id = parent.id
                child.metadata.chunk_index = child_idx
                child.metadata.chunker_type = "parent_child"
                all_child_chunks.append(child)
                child_idx += 1

        return all_child_chunks
