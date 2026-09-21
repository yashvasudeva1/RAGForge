"""
RAGForge shared constants and enumerations.

All string identifiers used across the node system, pipeline engine,
API, and frontend are defined here as enums to prevent typos and enable
IDE auto-completion.
"""

from enum import Enum


# --------------------------------------------------------------------------- #
# Node System
# --------------------------------------------------------------------------- #


class NodeCategory(str, Enum):
    """
    High-level category that a node belongs to.

    Used by the frontend to colour-code nodes and group the palette sidebar.
    """

    INPUT = "input"
    LOADER = "loader"
    CHUNKER = "chunker"
    EMBEDDER = "embedder"
    VECTOR_STORE = "vector_store"
    RETRIEVER = "retriever"
    RERANKER = "reranker"
    PROMPT = "prompt"
    GENERATOR = "generator"
    OUTPUT = "output"
    UTILITY = "utility"


class PortType(str, Enum):
    """
    The data type that flows through a connection between two nodes.

    The pipeline validator uses these to enforce type compatibility:
    only ports with matching PortType may be connected.
    """

    # Raw user text query
    QUERY = "query"

    # A single Document (output of a loader)
    DOCUMENT = "document"

    # A list of Documents
    DOCUMENTS = "documents"

    # A single Chunk
    CHUNK = "chunk"

    # A list of Chunks
    CHUNKS = "chunks"

    # An EmbeddedChunk (chunk + dense vector)
    EMBEDDED_CHUNK = "embedded_chunk"

    # A list of EmbeddedChunks
    EMBEDDED_CHUNKS = "embedded_chunks"

    # A list of RetrievalResults (chunk + score)
    RETRIEVAL_RESULTS = "retrieval_results"

    # A rendered prompt string
    PROMPT = "prompt"

    # The final generated answer string
    ANSWER = "answer"

    # A GenerationResult (answer + metadata)
    GENERATION_RESULT = "generation_result"

    # Generic pass-through for utility nodes
    ANY = "any"


class FieldType(str, Enum):
    """
    UI-level field type for auto-rendering config panels in the frontend.

    The frontend maps each FieldType to the appropriate form control.
    """

    STRING = "string"           # Single-line text input
    TEXT = "text"               # Multi-line textarea
    INTEGER = "integer"         # Number input (integer)
    FLOAT = "float"             # Number input (decimal)
    BOOLEAN = "boolean"         # Toggle / checkbox
    SELECT = "select"           # Dropdown with predefined options
    MULTI_SELECT = "multi_select"  # Multi-select dropdown
    PASSWORD = "password"       # Masked text input (for API keys)
    FILE = "file"               # File upload control
    SLIDER = "slider"           # Range slider (min / max / step)
    CODE = "code"               # Code editor (for prompt templates)
    JSON = "json"               # JSON editor


# --------------------------------------------------------------------------- #
# File Types
# --------------------------------------------------------------------------- #


class FileType(str, Enum):
    """Supported input file types for document loaders."""

    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    MARKDOWN = "markdown"
    TXT = "txt"
    HTML = "html"
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    PPTX = "pptx"
    XLSX = "xlsx"
    XLS = "xls"
    IMAGE = "image"
    WEB_URL = "web_url"
    UNKNOWN = "unknown"


# Mapping of file extensions to FileType enum values
EXTENSION_TO_FILE_TYPE: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
    ".doc": FileType.DOC,
    ".md": FileType.MARKDOWN,
    ".markdown": FileType.MARKDOWN,
    ".txt": FileType.TXT,
    ".html": FileType.HTML,
    ".htm": FileType.HTML,
    ".csv": FileType.CSV,
    ".json": FileType.JSON,
    ".jsonl": FileType.JSONL,
    ".pptx": FileType.PPTX,
    ".xlsx": FileType.XLSX,
    ".xls": FileType.XLS,
    ".png": FileType.IMAGE,
    ".jpg": FileType.IMAGE,
    ".jpeg": FileType.IMAGE,
    ".tiff": FileType.IMAGE,
    ".bmp": FileType.IMAGE,
    ".webp": FileType.IMAGE,
}


# --------------------------------------------------------------------------- #
# Embedding Providers
# --------------------------------------------------------------------------- #


class EmbeddingProvider(str, Enum):
    """Supported embedding model providers."""

    OPENAI = "openai"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    VOYAGE = "voyage"
    LOCAL = "local"           # sentence-transformers run locally


# --------------------------------------------------------------------------- #
# Vector Store Backends
# --------------------------------------------------------------------------- #


class VectorStoreBackend(str, Enum):
    """Supported vector store backends."""

    CHROMA = "chroma"
    QDRANT = "qdrant"
    FAISS = "faiss"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    MILVUS = "milvus"
    LANCEDB = "lancedb"


# --------------------------------------------------------------------------- #
# LLM Providers
# --------------------------------------------------------------------------- #


class LLMProvider(str, Enum):
    """Supported large language model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


# --------------------------------------------------------------------------- #
# Retrieval Strategies
# --------------------------------------------------------------------------- #


class RetrievalStrategy(str, Enum):
    """Supported retrieval strategies."""

    DENSE = "dense"
    SPARSE = "sparse"
    BM25 = "bm25"
    HYBRID = "hybrid"
    MMR = "mmr"
    SIMILARITY = "similarity"
    METADATA_FILTER = "metadata_filter"
    MULTI_QUERY = "multi_query"
    CONTEXTUAL = "contextual"


# --------------------------------------------------------------------------- #
# Reranker Types
# --------------------------------------------------------------------------- #


class RerankerType(str, Enum):
    """Supported reranker implementations."""

    CROSS_ENCODER = "cross_encoder"
    COHERE = "cohere"
    VOYAGE = "voyage"
    LLM = "llm"


# --------------------------------------------------------------------------- #
# Chunking Strategies
# --------------------------------------------------------------------------- #


class ChunkingStrategy(str, Enum):
    """Supported text chunking strategies."""

    CHARACTER = "character"
    RECURSIVE = "recursive"
    TOKEN = "token"
    SENTENCE = "sentence"
    MARKDOWN = "markdown"
    SEMANTIC = "semantic"
    PARENT_CHILD = "parent_child"


# --------------------------------------------------------------------------- #
# Pipeline Execution
# --------------------------------------------------------------------------- #


class ExecutionStatus(str, Enum):
    """Lifecycle states of a pipeline execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeExecutionStatus(str, Enum):
    """Lifecycle states of a single node within a pipeline execution."""

    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 200
DEFAULT_RETRIEVAL_K: int = 5
DEFAULT_RERANK_TOP_N: int = 3
DEFAULT_MAX_TOKENS: int = 1024
DEFAULT_TEMPERATURE: float = 0.0

# Node palette colour palette — keyed by NodeCategory
NODE_CATEGORY_COLORS: dict[str, str] = {
    NodeCategory.INPUT: "#6366f1",          # indigo
    NodeCategory.LOADER: "#0ea5e9",         # sky
    NodeCategory.CHUNKER: "#10b981",        # emerald
    NodeCategory.EMBEDDER: "#f59e0b",       # amber
    NodeCategory.VECTOR_STORE: "#8b5cf6",   # violet
    NodeCategory.RETRIEVER: "#ec4899",      # pink
    NodeCategory.RERANKER: "#f97316",       # orange
    NodeCategory.PROMPT: "#64748b",         # slate
    NodeCategory.GENERATOR: "#ef4444",      # red
    NodeCategory.OUTPUT: "#14b8a6",         # teal
    NodeCategory.UTILITY: "#94a3b8",        # light slate
}
