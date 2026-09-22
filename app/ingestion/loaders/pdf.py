"""
PDF document loader.

Uses ``pypdf`` as the primary parser with optional fallback to
``pdfplumber`` for more accurate table and layout extraction.
"""

from __future__ import annotations

from pathlib import Path

from app.core.constants import FileType
from app.core.exceptions import FileReadError, UnsupportedFileTypeError
from app.core.logging import get_logger
from app.domain.documents.models import Document, DocumentMetadata
from app.ingestion.loaders.base import BaseLoader

logger = get_logger(__name__)


class PDFLoader(BaseLoader):
    """
    Load PDF files and extract their text content.

    Parameters
    ----------
    extract_images:
        If True, attempt OCR on embedded images (requires pytesseract).
    password:
        Password for encrypted PDFs.
    use_pdfplumber:
        Force use of pdfplumber instead of pypdf.
    """

    def __init__(
        self,
        extract_images: bool = False,
        password: str | None = None,
        use_pdfplumber: bool = False,
    ) -> None:
        self.extract_images = extract_images
        self.password = password
        self.use_pdfplumber = use_pdfplumber

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise FileReadError(source, FileNotFoundError(f"File not found: {source}"))
        if path.suffix.lower() != ".pdf":
            raise UnsupportedFileTypeError(path.suffix)

        if self.use_pdfplumber:
            return self._load_pdfplumber(path)
        return self._load_pypdf(path)

    # ------------------------------------------------------------------ #
    # pypdf backend
    # ------------------------------------------------------------------ #

    def _load_pypdf(self, path: Path) -> list[Document]:
        try:
            import pypdf  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("pypdf is required for PDF loading. Run: pip install pypdf") from exc

        try:
            reader = pypdf.PdfReader(str(path), password=self.password)
            pages: list[str] = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append(text)

            full_text = "\n\n".join(p for p in pages if p.strip())
            meta = reader.metadata or {}

            document = Document(
                content=full_text,
                metadata=DocumentMetadata(
                    source=str(path),
                    file_type=FileType.PDF,
                    file_name=path.name,
                    file_size_bytes=path.stat().st_size,
                    page_count=len(reader.pages),
                    title=getattr(meta, "title", None),
                    author=getattr(meta, "author", None),
                ),
            )
            logger.debug("Loaded PDF via pypdf: %s (%d pages)", path.name, len(reader.pages))
            return [document]

        except Exception as exc:
            raise FileReadError(str(path), exc) from exc

    # ------------------------------------------------------------------ #
    # pdfplumber backend
    # ------------------------------------------------------------------ #

    def _load_pdfplumber(self, path: Path) -> list[Document]:
        try:
            import pdfplumber  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("pdfplumber is required. Run: pip install pdfplumber") from exc

        try:
            with pdfplumber.open(str(path), password=self.password) as pdf:
                pages_text = [page.extract_text() or "" for page in pdf.pages]
                full_text = "\n\n".join(p for p in pages_text if p.strip())
                page_count = len(pdf.pages)

            document = Document(
                content=full_text,
                metadata=DocumentMetadata(
                    source=str(path),
                    file_type=FileType.PDF,
                    file_name=path.name,
                    file_size_bytes=path.stat().st_size,
                    page_count=page_count,
                ),
            )
            logger.debug("Loaded PDF via pdfplumber: %s (%d pages)", path.name, page_count)
            return [document]

        except Exception as exc:
            raise FileReadError(str(path), exc) from exc
