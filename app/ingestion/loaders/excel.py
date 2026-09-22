"""
Excel (.xlsx / .xls) document loader using openpyxl.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.constants import FileType
from app.core.exceptions import FileReadError
from app.core.logging import get_logger
from app.domain.documents.models import Document, DocumentMetadata
from app.ingestion.loaders.base import BaseLoader

logger = get_logger(__name__)


class ExcelLoader(BaseLoader):
    """
    Load Excel spreadsheets (.xlsx, .xls).

    Parameters
    ----------
    sheet_name:
        Load a specific sheet by name. When None, all sheets are loaded.
    row_per_document:
        When True, each non-empty row becomes a separate Document.
    encoding:
        Used when falling back to xlrd for .xls files.
    """

    def __init__(
        self,
        sheet_name: str | None = None,
        row_per_document: bool = True,
    ) -> None:
        self.sheet_name = sheet_name
        self.row_per_document = row_per_document

    def load(self, source: str) -> list[Document]:
        try:
            import openpyxl  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("openpyxl is required. Run: pip install openpyxl") from exc

        path = Path(source)
        if not path.exists():
            raise FileReadError(source, FileNotFoundError(f"File not found: {source}"))

        try:
            wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)

            sheet_names = (
                [self.sheet_name] if self.sheet_name and self.sheet_name in wb.sheetnames
                else wb.sheetnames
            )

            documents: list[Document] = []

            for sname in sheet_names:
                ws = wb[sname]
                rows_data: list[list[Any]] = [
                    [cell.value for cell in row]
                    for row in ws.iter_rows()
                ]

                if not rows_data:
                    continue

                # First row as headers if available
                headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(rows_data[0])]
                data_rows = rows_data[1:]

                if self.row_per_document:
                    for r_idx, row in enumerate(data_rows):
                        parts = [
                            f"{headers[i]}: {str(v)}"
                            for i, v in enumerate(row)
                            if v is not None and str(v).strip()
                        ]
                        if not parts:
                            continue
                        content = "\n".join(parts)
                        documents.append(
                            Document(
                                content=content,
                                metadata=DocumentMetadata(
                                    source=str(path),
                                    file_type=FileType.XLSX,
                                    file_name=path.name,
                                    extra={"sheet": sname, "row_index": r_idx + 1},
                                ),
                            )
                        )
                else:
                    lines = [" | ".join(str(v) if v is not None else "" for v in r) for r in rows_data]
                    content = "\n".join(lines)
                    documents.append(
                        Document(
                            content=content,
                            metadata=DocumentMetadata(
                                source=str(path),
                                file_type=FileType.XLSX,
                                file_name=path.name,
                                file_size_bytes=path.stat().st_size,
                                extra={"sheet": sname},
                            ),
                        )
                    )

            wb.close()
            logger.debug("Loaded Excel: %s (%d documents)", path.name, len(documents))
            return documents

        except Exception as exc:
            raise FileReadError(str(path), exc) from exc
