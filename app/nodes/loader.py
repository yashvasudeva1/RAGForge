"""
Document loader nodes for RAGForge.

Each node in this module wraps a backend loader runner and exposes it as a
self-describing pipeline node that can be dragged onto the canvas.

Nodes registered here:
  - pdf_loader
  - docx_loader
  - markdown_loader
  - html_loader
  - csv_loader
  - json_loader
  - web_loader
  - pptx_loader
  - excel_loader
  - text_loader
"""

from __future__ import annotations

from typing import Any

from app.core.constants import FieldType, NodeCategory, PortType
from app.core.exceptions import LoaderError
from app.core.logging import get_logger
from app.core.registry import registry
from app.nodes.base import BaseNode, NodeField, NodeMetadata, NodePortDefinition

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Shared port definitions used by all loaders
# --------------------------------------------------------------------------- #

_DOCUMENTS_OUTPUT = NodePortDefinition(
    name="documents",
    port_type=PortType.DOCUMENTS,
    display_name="Documents",
    description="Loaded documents passed to the next node.",
    required=False,
)


# --------------------------------------------------------------------------- #
# PDF Loader
# --------------------------------------------------------------------------- #


@registry.register("pdf_loader")
class PDFLoaderNode(BaseNode):
    """Load one or more PDF files and emit a list of Documents."""

    node_type = "pdf_loader"
    metadata = NodeMetadata(
        category=NodeCategory.LOADER,
        display_name="PDF Loader",
        description="Load PDF files and extract their text content.",
        icon_name="file-text",
        tags=["pdf", "loader", "document", "file"],
    )
    input_ports: list[NodePortDefinition] = []
    output_ports = [_DOCUMENTS_OUTPUT]
    fields = [
        NodeField(
            name="file_path",
            field_type=FieldType.FILE,
            display_name="File Path",
            description="Path to the PDF file or directory of PDF files.",
            required=True,
        ),
        NodeField(
            name="extract_images",
            field_type=FieldType.BOOLEAN,
            display_name="Extract Images",
            description="Attempt OCR text extraction from embedded images.",
            default=False,
            advanced=True,
        ),
        NodeField(
            name="password",
            field_type=FieldType.PASSWORD,
            display_name="Password",
            description="Password for encrypted PDF files.",
            default=None,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.ingestion.loaders.pdf import PDFLoader

        file_path: str = self.require_config("file_path")
        loader = PDFLoader(
            extract_images=self.get_config("extract_images", False),
            password=self.get_config("password"),
        )
        documents = await loader.aload(file_path)
        logger.info("PDF loader produced %d documents", len(documents))
        return {"documents": documents}


# --------------------------------------------------------------------------- #
# DOCX Loader
# --------------------------------------------------------------------------- #


@registry.register("docx_loader")
class DocxLoaderNode(BaseNode):
    """Load Microsoft Word (.docx) documents."""

    node_type = "docx_loader"
    metadata = NodeMetadata(
        category=NodeCategory.LOADER,
        display_name="DOCX Loader",
        description="Load Microsoft Word (.docx) files and extract their text content.",
        icon_name="file-text",
        tags=["docx", "word", "loader", "document"],
    )
    input_ports: list[NodePortDefinition] = []
    output_ports = [_DOCUMENTS_OUTPUT]
    fields = [
        NodeField(
            name="file_path",
            field_type=FieldType.FILE,
            display_name="File Path",
            description="Path to the .docx file.",
            required=True,
        ),
        NodeField(
            name="include_headers",
            field_type=FieldType.BOOLEAN,
            display_name="Include Headers",
            description="Include document heading text in the extracted content.",
            default=True,
            advanced=True,
        ),
        NodeField(
            name="include_tables",
            field_type=FieldType.BOOLEAN,
            display_name="Include Tables",
            description="Include table content converted to plain text.",
            default=True,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.ingestion.loaders.docx import DocxLoader

        loader = DocxLoader(
            include_headers=self.get_config("include_headers", True),
            include_tables=self.get_config("include_tables", True),
        )
        documents = await loader.aload(self.require_config("file_path"))
        return {"documents": documents}


# --------------------------------------------------------------------------- #
# Markdown Loader
# --------------------------------------------------------------------------- #


@registry.register("markdown_loader")
class MarkdownLoaderNode(BaseNode):
    """Load Markdown (.md) files."""

    node_type = "markdown_loader"
    metadata = NodeMetadata(
        category=NodeCategory.LOADER,
        display_name="Markdown Loader",
        description="Load Markdown files, stripping front-matter and preserving structure.",
        icon_name="hash",
        tags=["markdown", "md", "loader", "document"],
    )
    input_ports: list[NodePortDefinition] = []
    output_ports = [_DOCUMENTS_OUTPUT]
    fields = [
        NodeField(
            name="file_path",
            field_type=FieldType.FILE,
            display_name="File Path",
            description="Path to the .md file.",
            required=True,
        ),
        NodeField(
            name="strip_front_matter",
            field_type=FieldType.BOOLEAN,
            display_name="Strip Front Matter",
            description="Remove YAML front matter from the document content.",
            default=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.ingestion.loaders.markdown import MarkdownLoader

        loader = MarkdownLoader(
            strip_front_matter=self.get_config("strip_front_matter", True),
        )
        documents = await loader.aload(self.require_config("file_path"))
        return {"documents": documents}


# --------------------------------------------------------------------------- #
# HTML Loader
# --------------------------------------------------------------------------- #


@registry.register("html_loader")
class HTMLLoaderNode(BaseNode):
    """Load HTML files and extract clean text using BeautifulSoup."""

    node_type = "html_loader"
    metadata = NodeMetadata(
        category=NodeCategory.LOADER,
        display_name="HTML Loader",
        description="Load HTML files and extract clean readable text using BeautifulSoup.",
        icon_name="code",
        tags=["html", "loader", "web", "document"],
    )
    input_ports: list[NodePortDefinition] = []
    output_ports = [_DOCUMENTS_OUTPUT]
    fields = [
        NodeField(
            name="file_path",
            field_type=FieldType.FILE,
            display_name="File Path",
            description="Path to the .html file.",
            required=True,
        ),
        NodeField(
            name="remove_scripts",
            field_type=FieldType.BOOLEAN,
            display_name="Remove Scripts",
            description="Strip <script> and <style> tags before extracting text.",
            default=True,
        ),
        NodeField(
            name="parser",
            field_type=FieldType.SELECT,
            display_name="HTML Parser",
            description="BeautifulSoup parser to use.",
            default="lxml",
            options=["lxml", "html.parser", "html5lib"],
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.ingestion.loaders.html import HTMLLoader

        loader = HTMLLoader(
            remove_scripts=self.get_config("remove_scripts", True),
            parser=self.get_config("parser", "lxml"),
        )
        documents = await loader.aload(self.require_config("file_path"))
        return {"documents": documents}


# --------------------------------------------------------------------------- #
# CSV Loader
# --------------------------------------------------------------------------- #


@registry.register("csv_loader")
class CSVLoaderNode(BaseNode):
    """Load CSV files — each row becomes a Document or the whole file becomes one."""

    node_type = "csv_loader"
    metadata = NodeMetadata(
        category=NodeCategory.LOADER,
        display_name="CSV Loader",
        description="Load CSV files. Each row can become a separate Document, or the entire file can be one Document.",
        icon_name="table",
        tags=["csv", "tabular", "loader", "spreadsheet"],
    )
    input_ports: list[NodePortDefinition] = []
    output_ports = [_DOCUMENTS_OUTPUT]
    fields = [
        NodeField(
            name="file_path",
            field_type=FieldType.FILE,
            display_name="File Path",
            description="Path to the .csv file.",
            required=True,
        ),
        NodeField(
            name="row_per_document",
            field_type=FieldType.BOOLEAN,
            display_name="Row per Document",
            description="When enabled, each CSV row becomes a separate Document. When disabled, the whole file is one Document.",
            default=True,
        ),
        NodeField(
            name="delimiter",
            field_type=FieldType.STRING,
            display_name="Delimiter",
            description="Column delimiter character.",
            default=",",
            advanced=True,
        ),
        NodeField(
            name="content_columns",
            field_type=FieldType.STRING,
            display_name="Content Columns",
            description="Comma-separated list of column names to include in the document content. Leave empty to include all columns.",
            default=None,
            placeholder="title,body,summary",
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.ingestion.loaders.csv import CSVLoader

        content_cols_raw = self.get_config("content_columns")
        content_columns = (
            [c.strip() for c in content_cols_raw.split(",") if c.strip()]
            if content_cols_raw
            else None
        )
        loader = CSVLoader(
            row_per_document=self.get_config("row_per_document", True),
            delimiter=self.get_config("delimiter", ","),
            content_columns=content_columns,
        )
        documents = await loader.aload(self.require_config("file_path"))
        return {"documents": documents}


# --------------------------------------------------------------------------- #
# JSON Loader
# --------------------------------------------------------------------------- #


@registry.register("json_loader")
class JSONLoaderNode(BaseNode):
    """Load JSON and JSONL files."""

    node_type = "json_loader"
    metadata = NodeMetadata(
        category=NodeCategory.LOADER,
        display_name="JSON / JSONL Loader",
        description="Load JSON or JSON Lines files. Specify a content key to extract text from structured records.",
        icon_name="braces",
        tags=["json", "jsonl", "loader", "structured"],
    )
    input_ports: list[NodePortDefinition] = []
    output_ports = [_DOCUMENTS_OUTPUT]
    fields = [
        NodeField(
            name="file_path",
            field_type=FieldType.FILE,
            display_name="File Path",
            description="Path to the .json or .jsonl file.",
            required=True,
        ),
        NodeField(
            name="content_key",
            field_type=FieldType.STRING,
            display_name="Content Key",
            description="Key in each JSON object whose value becomes the document content. Leave empty to stringify the whole object.",
            default=None,
            placeholder="text",
        ),
        NodeField(
            name="metadata_keys",
            field_type=FieldType.STRING,
            display_name="Metadata Keys",
            description="Comma-separated list of keys to include as document metadata.",
            default=None,
            placeholder="id,title,url",
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.ingestion.loaders.json import JSONLoader

        meta_keys_raw = self.get_config("metadata_keys")
        metadata_keys = (
            [k.strip() for k in meta_keys_raw.split(",") if k.strip()]
            if meta_keys_raw
            else None
        )
        loader = JSONLoader(
            content_key=self.get_config("content_key"),
            metadata_keys=metadata_keys,
        )
        documents = await loader.aload(self.require_config("file_path"))
        return {"documents": documents}


# --------------------------------------------------------------------------- #
# Web / URL Loader
# --------------------------------------------------------------------------- #


@registry.register("web_loader")
class WebLoaderNode(BaseNode):
    """Load clean article text from one or more web URLs using Trafilatura."""

    node_type = "web_loader"
    metadata = NodeMetadata(
        category=NodeCategory.LOADER,
        display_name="Web URL Loader",
        description="Fetch and extract clean readable text from web pages using Trafilatura.",
        icon_name="globe",
        tags=["web", "url", "html", "loader", "scrape", "crawl"],
    )
    input_ports: list[NodePortDefinition] = []
    output_ports = [_DOCUMENTS_OUTPUT]
    fields = [
        NodeField(
            name="urls",
            field_type=FieldType.TEXT,
            display_name="URLs",
            description="One URL per line to load.",
            required=True,
            placeholder="https://example.com/article\nhttps://another.com/page",
        ),
        NodeField(
            name="include_comments",
            field_type=FieldType.BOOLEAN,
            display_name="Include Comments",
            description="Include user comment sections in the extracted text.",
            default=False,
            advanced=True,
        ),
        NodeField(
            name="timeout_seconds",
            field_type=FieldType.INTEGER,
            display_name="Timeout (seconds)",
            description="Per-URL HTTP request timeout.",
            default=30,
            min_value=5,
            max_value=120,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.ingestion.loaders.web import WebLoader

        urls_raw: str = self.require_config("urls")
        urls = [u.strip() for u in urls_raw.splitlines() if u.strip()]
        loader = WebLoader(
            include_comments=self.get_config("include_comments", False),
            timeout_seconds=self.get_config("timeout_seconds", 30),
        )
        documents = []
        for url in urls:
            docs = await loader.aload(url)
            documents.extend(docs)
        return {"documents": documents}


# --------------------------------------------------------------------------- #
# PPTX Loader
# --------------------------------------------------------------------------- #


@registry.register("pptx_loader")
class PPTXLoaderNode(BaseNode):
    """Load PowerPoint (.pptx) presentations."""

    node_type = "pptx_loader"
    metadata = NodeMetadata(
        category=NodeCategory.LOADER,
        display_name="PowerPoint Loader",
        description="Load PowerPoint .pptx files, extracting slide text and speaker notes.",
        icon_name="presentation",
        tags=["pptx", "powerpoint", "slides", "loader"],
    )
    input_ports: list[NodePortDefinition] = []
    output_ports = [_DOCUMENTS_OUTPUT]
    fields = [
        NodeField(
            name="file_path",
            field_type=FieldType.FILE,
            display_name="File Path",
            description="Path to the .pptx file.",
            required=True,
        ),
        NodeField(
            name="include_notes",
            field_type=FieldType.BOOLEAN,
            display_name="Include Speaker Notes",
            description="Include speaker notes text in the extracted content.",
            default=True,
        ),
        NodeField(
            name="slide_per_document",
            field_type=FieldType.BOOLEAN,
            display_name="Slide per Document",
            description="When enabled, each slide becomes a separate Document.",
            default=True,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.ingestion.loaders.pptx import PPTXLoader

        loader = PPTXLoader(
            include_notes=self.get_config("include_notes", True),
            slide_per_document=self.get_config("slide_per_document", True),
        )
        documents = await loader.aload(self.require_config("file_path"))
        return {"documents": documents}


# --------------------------------------------------------------------------- #
# Excel Loader
# --------------------------------------------------------------------------- #


@registry.register("excel_loader")
class ExcelLoaderNode(BaseNode):
    """Load Excel (.xlsx/.xls) spreadsheets."""

    node_type = "excel_loader"
    metadata = NodeMetadata(
        category=NodeCategory.LOADER,
        display_name="Excel Loader",
        description="Load Excel spreadsheets (.xlsx, .xls). Each sheet or each row can become a Document.",
        icon_name="table",
        tags=["excel", "xlsx", "xls", "spreadsheet", "loader"],
    )
    input_ports: list[NodePortDefinition] = []
    output_ports = [_DOCUMENTS_OUTPUT]
    fields = [
        NodeField(
            name="file_path",
            field_type=FieldType.FILE,
            display_name="File Path",
            description="Path to the .xlsx or .xls file.",
            required=True,
        ),
        NodeField(
            name="sheet_name",
            field_type=FieldType.STRING,
            display_name="Sheet Name",
            description="Name of the sheet to load. Leave empty to load all sheets.",
            default=None,
            placeholder="Sheet1",
        ),
        NodeField(
            name="row_per_document",
            field_type=FieldType.BOOLEAN,
            display_name="Row per Document",
            description="When enabled, each row becomes a separate Document.",
            default=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.ingestion.loaders.excel import ExcelLoader

        loader = ExcelLoader(
            sheet_name=self.get_config("sheet_name"),
            row_per_document=self.get_config("row_per_document", True),
        )
        documents = await loader.aload(self.require_config("file_path"))
        return {"documents": documents}


# --------------------------------------------------------------------------- #
# Plain Text Loader
# --------------------------------------------------------------------------- #


@registry.register("text_loader")
class TextLoaderNode(BaseNode):
    """Load plain text files."""

    node_type = "text_loader"
    metadata = NodeMetadata(
        category=NodeCategory.LOADER,
        display_name="Text Loader",
        description="Load plain text (.txt) files.",
        icon_name="file",
        tags=["txt", "text", "plain", "loader"],
    )
    input_ports: list[NodePortDefinition] = []
    output_ports = [_DOCUMENTS_OUTPUT]
    fields = [
        NodeField(
            name="file_path",
            field_type=FieldType.FILE,
            display_name="File Path",
            description="Path to the .txt file.",
            required=True,
        ),
        NodeField(
            name="encoding",
            field_type=FieldType.STRING,
            display_name="Encoding",
            description="File character encoding.",
            default="utf-8",
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        import asyncio
        from pathlib import Path

        from app.core.constants import FileType
        from app.domain.documents.models import Document, DocumentMetadata

        file_path = Path(self.require_config("file_path"))
        encoding = self.get_config("encoding", "utf-8")

        content = await asyncio.to_thread(file_path.read_text, encoding=encoding)
        doc = Document(
            content=content,
            metadata=DocumentMetadata(
                source=str(file_path),
                file_type=FileType.TXT,
                file_name=file_path.name,
                file_size_bytes=file_path.stat().st_size,
            ),
        )
        return {"documents": [doc]}
