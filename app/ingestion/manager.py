"""
Ingestion manager — auto-dispatches to the correct loader by file type.
"""

from __future__ import annotations

from pathlib import Path

from app.core.constants import EXTENSION_TO_FILE_TYPE, FileType
from app.core.exceptions import UnsupportedFileTypeError
from app.core.logging import get_logger
from app.domain.documents.models import Document
from app.ingestion.loaders.base import BaseLoader

logger = get_logger(__name__)


class IngestionManager:
    """
    Auto-dispatches load requests to the appropriate loader.

    Detects file type from the file extension (for local files) or
    treats the source as a web URL when it starts with ``http``.

    Usage
    -----
        manager = IngestionManager()
        documents = await manager.aload("report.pdf")
        documents = await manager.aload("https://example.com/article")
    """

    def _get_loader(self, source: str) -> BaseLoader:
        """Return the appropriate loader for ``source``."""
        from app.ingestion.loaders.csv import CSVLoader
        from app.ingestion.loaders.docx import DocxLoader
        from app.ingestion.loaders.excel import ExcelLoader
        from app.ingestion.loaders.html import HTMLLoader
        from app.ingestion.loaders.json import JSONLoader
        from app.ingestion.loaders.markdown import MarkdownLoader
        from app.ingestion.loaders.pdf import PDFLoader
        from app.ingestion.loaders.pptx import PPTXLoader
        from app.ingestion.loaders.web import WebLoader

        if source.startswith("http://") or source.startswith("https://"):
            return WebLoader()

        path = Path(source)
        ext = path.suffix.lower()
        file_type = EXTENSION_TO_FILE_TYPE.get(ext, FileType.UNKNOWN)

        loader_map: dict[FileType, BaseLoader] = {
            FileType.PDF: PDFLoader(),
            FileType.DOCX: DocxLoader(),
            FileType.DOC: DocxLoader(),
            FileType.MARKDOWN: MarkdownLoader(),
            FileType.TXT: MarkdownLoader(strip_front_matter=False),
            FileType.HTML: HTMLLoader(),
            FileType.CSV: CSVLoader(),
            FileType.JSON: JSONLoader(),
            FileType.JSONL: JSONLoader(),
            FileType.PPTX: PPTXLoader(),
            FileType.XLSX: ExcelLoader(),
            FileType.XLS: ExcelLoader(),
            FileType.WEB_URL: WebLoader(),
        }

        if file_type not in loader_map:
            raise UnsupportedFileTypeError(ext or source)

        return loader_map[file_type]

    def load(self, source: str) -> list[Document]:
        """Synchronously load documents from ``source``."""
        loader = self._get_loader(source)
        logger.info("Loading '%s' with %s", source, type(loader).__name__)
        return loader.load(source)

    async def aload(self, source: str) -> list[Document]:
        """Asynchronously load documents from ``source``."""
        loader = self._get_loader(source)
        logger.info("Loading '%s' with %s", source, type(loader).__name__)
        return await loader.aload(source)
