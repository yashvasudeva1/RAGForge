"""
HTML document loader using BeautifulSoup.
"""

from __future__ import annotations

from pathlib import Path

from app.core.constants import FileType
from app.core.exceptions import FileReadError
from app.core.logging import get_logger
from app.domain.documents.models import Document, DocumentMetadata
from app.ingestion.loaders.base import BaseLoader

logger = get_logger(__name__)


class HTMLLoader(BaseLoader):
    """
    Load HTML files and extract clean readable text.

    Parameters
    ----------
    remove_scripts:
        Strip <script> and <style> tags.
    parser:
        BeautifulSoup HTML parser backend (``lxml``, ``html.parser``, ``html5lib``).
    encoding:
        File encoding.
    """

    def __init__(
        self,
        remove_scripts: bool = True,
        parser: str = "lxml",
        encoding: str = "utf-8",
    ) -> None:
        self.remove_scripts = remove_scripts
        self.parser = parser
        self.encoding = encoding

    def load(self, source: str) -> list[Document]:
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("beautifulsoup4 is required. Run: pip install beautifulsoup4 lxml") from exc

        path = Path(source)
        if not path.exists():
            raise FileReadError(source, FileNotFoundError(f"File not found: {source}"))

        try:
            raw_html = path.read_text(encoding=self.encoding)
            soup = BeautifulSoup(raw_html, self.parser)

            if self.remove_scripts:
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()

            title = soup.title.string.strip() if soup.title and soup.title.string else None
            content = soup.get_text(separator="\n", strip=True)

            document = Document(
                content=content,
                metadata=DocumentMetadata(
                    source=str(path),
                    file_type=FileType.HTML,
                    file_name=path.name,
                    file_size_bytes=path.stat().st_size,
                    title=title,
                ),
            )
            logger.debug("Loaded HTML: %s (%d chars)", path.name, len(content))
            return [document]

        except Exception as exc:
            raise FileReadError(str(path), exc) from exc
