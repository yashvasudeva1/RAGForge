"""
JSON and JSONL document loader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.constants import FileType
from app.core.exceptions import FileReadError
from app.core.logging import get_logger
from app.domain.documents.models import Document, DocumentMetadata
from app.ingestion.loaders.base import BaseLoader

logger = get_logger(__name__)


class JSONLoader(BaseLoader):
    """
    Load JSON or JSON Lines (.jsonl) files.

    Parameters
    ----------
    content_key:
        Key in each JSON object whose value becomes the document content.
        When None, the whole object is JSON-serialised as the content.
    metadata_keys:
        List of keys to extract as document metadata.
    encoding:
        File encoding.
    """

    def __init__(
        self,
        content_key: str | None = None,
        metadata_keys: list[str] | None = None,
        encoding: str = "utf-8",
    ) -> None:
        self.content_key = content_key
        self.metadata_keys = metadata_keys or []
        self.encoding = encoding

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise FileReadError(source, FileNotFoundError(f"File not found: {source}"))

        try:
            raw = path.read_text(encoding=self.encoding)
            is_jsonl = path.suffix.lower() == ".jsonl"

            records: list[Any]
            if is_jsonl:
                records = [json.loads(line) for line in raw.splitlines() if line.strip()]
            else:
                data = json.loads(raw)
                records = data if isinstance(data, list) else [data]

            documents: list[Document] = []
            for i, record in enumerate(records):
                if isinstance(record, dict):
                    content = (
                        str(record.get(self.content_key, ""))
                        if self.content_key
                        else json.dumps(record, ensure_ascii=False)
                    )
                    extra = {k: record.get(k) for k in self.metadata_keys if k in record}
                else:
                    content = str(record)
                    extra = {}

                documents.append(
                    Document(
                        content=content,
                        metadata=DocumentMetadata(
                            source=str(path),
                            file_type=FileType.JSONL if is_jsonl else FileType.JSON,
                            file_name=path.name,
                            extra={"record_index": i, **extra},
                        ),
                    )
                )

            logger.debug("Loaded JSON: %s (%d records)", path.name, len(documents))
            return documents

        except Exception as exc:
            raise FileReadError(str(path), exc) from exc
