"""
CSV document loader.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from app.core.constants import FileType
from app.core.exceptions import FileReadError
from app.core.logging import get_logger
from app.domain.documents.models import Document, DocumentMetadata
from app.ingestion.loaders.base import BaseLoader

logger = get_logger(__name__)


class CSVLoader(BaseLoader):
    """
    Load CSV files.

    Parameters
    ----------
    row_per_document:
        When True, each CSV row becomes a separate Document.
        When False, the entire file becomes one Document.
    delimiter:
        Column delimiter character.
    content_columns:
        List of column names to include in the document content.
        When None, all columns are included.
    encoding:
        File encoding.
    """

    def __init__(
        self,
        row_per_document: bool = True,
        delimiter: str = ",",
        content_columns: list[str] | None = None,
        encoding: str = "utf-8",
    ) -> None:
        self.row_per_document = row_per_document
        self.delimiter = delimiter
        self.content_columns = content_columns
        self.encoding = encoding

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise FileReadError(source, FileNotFoundError(f"File not found: {source}"))

        try:
            raw = path.read_text(encoding=self.encoding)
            reader = csv.DictReader(io.StringIO(raw), delimiter=self.delimiter)
            rows: list[dict[str, Any]] = list(reader)

            if not rows:
                return []

            if self.row_per_document:
                documents: list[Document] = []
                for i, row in enumerate(rows):
                    content = self._row_to_text(row)
                    documents.append(
                        Document(
                            content=content,
                            metadata=DocumentMetadata(
                                source=str(path),
                                file_type=FileType.CSV,
                                file_name=path.name,
                                extra={"row_index": i, "row": dict(row)},
                            ),
                        )
                    )
                logger.debug("Loaded CSV: %s (%d rows)", path.name, len(documents))
                return documents
            else:
                # Whole file as one document
                lines = [self._row_to_text(row) for row in rows]
                content = "\n".join(lines)
                return [
                    Document(
                        content=content,
                        metadata=DocumentMetadata(
                            source=str(path),
                            file_type=FileType.CSV,
                            file_name=path.name,
                            file_size_bytes=path.stat().st_size,
                            extra={"row_count": len(rows)},
                        ),
                    )
                ]

        except Exception as exc:
            raise FileReadError(str(path), exc) from exc

    def _row_to_text(self, row: dict[str, Any]) -> str:
        """Convert a CSV row dict to a text string."""
        if self.content_columns:
            parts = [f"{k}: {row.get(k, '')}" for k in self.content_columns if k in row]
        else:
            parts = [f"{k}: {v}" for k, v in row.items()]
        return "\n".join(parts)
