"""
Markdown document loader.

Reads .md files, optionally strips YAML front matter, and returns the
clean document content.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.core.constants import FileType
from app.core.exceptions import FileReadError
from app.core.logging import get_logger
from app.domain.documents.models import Document, DocumentMetadata
from app.ingestion.loaders.base import BaseLoader

logger = get_logger(__name__)

# Regex to match YAML front matter (--- ... ---)
_FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


class MarkdownLoader(BaseLoader):
    """
    Load Markdown (.md) files.

    Parameters
    ----------
    strip_front_matter:
        Remove YAML front matter blocks before returning content.
    encoding:
        File character encoding.
    """

    def __init__(
        self,
        strip_front_matter: bool = True,
        encoding: str = "utf-8",
    ) -> None:
        self.strip_front_matter = strip_front_matter
        self.encoding = encoding

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise FileReadError(source, FileNotFoundError(f"File not found: {source}"))

        try:
            content = path.read_text(encoding=self.encoding)

            # Extract front matter metadata before stripping
            front_matter_meta: dict = {}
            if self.strip_front_matter:
                match = _FRONT_MATTER_RE.match(content)
                if match:
                    fm_text = match.group(0)
                    content = content[match.end():]
                    try:
                        import yaml  # type: ignore[import-untyped]
                        fm_data = yaml.safe_load(fm_text.strip("---\n"))
                        if isinstance(fm_data, dict):
                            front_matter_meta = fm_data
                    except Exception:  # noqa: BLE001
                        pass

            document = Document(
                content=content.strip(),
                metadata=DocumentMetadata(
                    source=str(path),
                    file_type=FileType.MARKDOWN,
                    file_name=path.name,
                    file_size_bytes=path.stat().st_size,
                    title=front_matter_meta.get("title"),
                    author=front_matter_meta.get("author"),
                    extra=front_matter_meta,
                ),
            )
            logger.debug("Loaded Markdown: %s", path.name)
            return [document]

        except Exception as exc:
            raise FileReadError(str(path), exc) from exc
