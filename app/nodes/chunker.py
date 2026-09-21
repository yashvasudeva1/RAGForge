"""
Chunker nodes for RAGForge.

Each node wraps a backend chunker runner and exposes it as a pipeline node.

Nodes registered here:
  - recursive_chunker
  - character_chunker
  - token_chunker
  - sentence_chunker
  - markdown_chunker
  - semantic_chunker
  - parent_child_chunker
"""

from __future__ import annotations

from typing import Any

from app.core.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, FieldType, NodeCategory, PortType
from app.core.registry import registry
from app.nodes.base import BaseNode, NodeField, NodeMetadata, NodePortDefinition

# --------------------------------------------------------------------------- #
# Shared port definitions
# --------------------------------------------------------------------------- #

_DOCUMENTS_INPUT = NodePortDefinition(
    name="documents",
    port_type=PortType.DOCUMENTS,
    display_name="Documents",
    description="Documents to chunk.",
    required=True,
)
_CHUNKS_OUTPUT = NodePortDefinition(
    name="chunks",
    port_type=PortType.CHUNKS,
    display_name="Chunks",
    description="Text chunks produced by this chunker.",
    required=False,
)

# Shared size/overlap fields used by most chunkers
_CHUNK_SIZE_FIELD = NodeField(
    name="chunk_size",
    field_type=FieldType.SLIDER,
    display_name="Chunk Size",
    description="Maximum number of characters per chunk.",
    default=DEFAULT_CHUNK_SIZE,
    min_value=100,
    max_value=8000,
    step=100,
)
_CHUNK_OVERLAP_FIELD = NodeField(
    name="chunk_overlap",
    field_type=FieldType.SLIDER,
    display_name="Chunk Overlap",
    description="Number of characters to overlap between consecutive chunks.",
    default=DEFAULT_CHUNK_OVERLAP,
    min_value=0,
    max_value=2000,
    step=50,
)


# --------------------------------------------------------------------------- #
# Recursive Chunker
# --------------------------------------------------------------------------- #


@registry.register("recursive_chunker")
class RecursiveChunkerNode(BaseNode):
    """
    Recursive character text splitter.

    Splits text by trying a hierarchy of separators in order:
    paragraph breaks, newlines, sentences, words.  This is the
    recommended default for most use cases.
    """

    node_type = "recursive_chunker"
    metadata = NodeMetadata(
        category=NodeCategory.CHUNKER,
        display_name="Recursive Chunker",
        description=(
            "Splits text recursively using a hierarchy of separators "
            "(paragraphs, newlines, sentences, words). "
            "Recommended for most document types."
        ),
        icon_name="scissors",
        tags=["chunk", "split", "recursive", "default"],
    )
    input_ports = [_DOCUMENTS_INPUT]
    output_ports = [_CHUNKS_OUTPUT]
    fields = [
        _CHUNK_SIZE_FIELD,
        _CHUNK_OVERLAP_FIELD,
        NodeField(
            name="separators",
            field_type=FieldType.STRING,
            display_name="Separators",
            description='JSON array of separator strings tried in order. Default: ["\\n\\n", "\\n", " ", ""]',
            default=None,
            placeholder='["\\n\\n", "\\n", " ", ""]',
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.chunking.recursive import RecursiveChunker
        import json

        documents = inputs.get("documents", [])
        separators_raw = self.get_config("separators")
        separators = json.loads(separators_raw) if separators_raw else None

        chunker = RecursiveChunker(
            chunk_size=self.get_config("chunk_size", DEFAULT_CHUNK_SIZE),
            chunk_overlap=self.get_config("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
            separators=separators,
        )
        chunks = []
        for doc in documents:
            chunks.extend(chunker.split(doc))
        return {"chunks": chunks}


# --------------------------------------------------------------------------- #
# Character Chunker
# --------------------------------------------------------------------------- #


@registry.register("character_chunker")
class CharacterChunkerNode(BaseNode):
    """Fixed-size character-based text splitter."""

    node_type = "character_chunker"
    metadata = NodeMetadata(
        category=NodeCategory.CHUNKER,
        display_name="Character Chunker",
        description="Splits text into fixed-size character windows with configurable overlap. Simple and predictable.",
        icon_name="scissors",
        tags=["chunk", "split", "character", "fixed"],
    )
    input_ports = [_DOCUMENTS_INPUT]
    output_ports = [_CHUNKS_OUTPUT]
    fields = [
        _CHUNK_SIZE_FIELD,
        _CHUNK_OVERLAP_FIELD,
        NodeField(
            name="separator",
            field_type=FieldType.STRING,
            display_name="Separator",
            description="Preferred split point. The chunker tries to split at this character before falling back to hard truncation.",
            default="\n",
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.chunking.character import CharacterChunker

        documents = inputs.get("documents", [])
        chunker = CharacterChunker(
            chunk_size=self.get_config("chunk_size", DEFAULT_CHUNK_SIZE),
            chunk_overlap=self.get_config("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
            separator=self.get_config("separator", "\n"),
        )
        chunks = []
        for doc in documents:
            chunks.extend(chunker.split(doc))
        return {"chunks": chunks}


# --------------------------------------------------------------------------- #
# Token Chunker
# --------------------------------------------------------------------------- #


@registry.register("token_chunker")
class TokenChunkerNode(BaseNode):
    """
    Token-based chunker using tiktoken.

    Splits at token boundaries rather than character boundaries.
    Ensures chunks fit within LLM context windows exactly.
    """

    node_type = "token_chunker"
    metadata = NodeMetadata(
        category=NodeCategory.CHUNKER,
        display_name="Token Chunker",
        description=(
            "Splits text at token boundaries using tiktoken. "
            "Guarantees chunks fit within a specific token budget."
        ),
        icon_name="hash",
        tags=["chunk", "split", "token", "tiktoken"],
    )
    input_ports = [_DOCUMENTS_INPUT]
    output_ports = [_CHUNKS_OUTPUT]
    fields = [
        NodeField(
            name="chunk_size",
            field_type=FieldType.SLIDER,
            display_name="Chunk Size (tokens)",
            description="Maximum number of tokens per chunk.",
            default=512,
            min_value=64,
            max_value=8192,
            step=64,
        ),
        NodeField(
            name="chunk_overlap",
            field_type=FieldType.SLIDER,
            display_name="Overlap (tokens)",
            description="Number of overlapping tokens between consecutive chunks.",
            default=50,
            min_value=0,
            max_value=512,
            step=10,
        ),
        NodeField(
            name="encoding_name",
            field_type=FieldType.SELECT,
            display_name="Encoding",
            description="Tiktoken encoding to use for tokenisation.",
            default="cl100k_base",
            options=["cl100k_base", "p50k_base", "r50k_base", "o200k_base"],
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.chunking.token import TokenChunker

        documents = inputs.get("documents", [])
        chunker = TokenChunker(
            chunk_size=self.get_config("chunk_size", 512),
            chunk_overlap=self.get_config("chunk_overlap", 50),
            encoding_name=self.get_config("encoding_name", "cl100k_base"),
        )
        chunks = []
        for doc in documents:
            chunks.extend(chunker.split(doc))
        return {"chunks": chunks}


# --------------------------------------------------------------------------- #
# Sentence Chunker
# --------------------------------------------------------------------------- #


@registry.register("sentence_chunker")
class SentenceChunkerNode(BaseNode):
    """Sentence-boundary-aware chunker using NLTK."""

    node_type = "sentence_chunker"
    metadata = NodeMetadata(
        category=NodeCategory.CHUNKER,
        display_name="Sentence Chunker",
        description=(
            "Groups complete sentences into chunks without splitting mid-sentence. "
            "Produces more natural-reading chunks than character splitting."
        ),
        icon_name="align-left",
        tags=["chunk", "split", "sentence", "nltk"],
    )
    input_ports = [_DOCUMENTS_INPUT]
    output_ports = [_CHUNKS_OUTPUT]
    fields = [
        NodeField(
            name="sentences_per_chunk",
            field_type=FieldType.SLIDER,
            display_name="Sentences per Chunk",
            description="Target number of sentences per chunk.",
            default=5,
            min_value=1,
            max_value=30,
            step=1,
        ),
        NodeField(
            name="sentence_overlap",
            field_type=FieldType.SLIDER,
            display_name="Sentence Overlap",
            description="Number of sentences to carry over into the next chunk.",
            default=1,
            min_value=0,
            max_value=10,
            step=1,
        ),
        NodeField(
            name="language",
            field_type=FieldType.SELECT,
            display_name="Language",
            description="NLTK sentence tokenizer language.",
            default="english",
            options=["english", "german", "french", "spanish", "italian", "dutch", "portuguese"],
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.chunking.sentence import SentenceChunker

        documents = inputs.get("documents", [])
        chunker = SentenceChunker(
            sentences_per_chunk=self.get_config("sentences_per_chunk", 5),
            sentence_overlap=self.get_config("sentence_overlap", 1),
            language=self.get_config("language", "english"),
        )
        chunks = []
        for doc in documents:
            chunks.extend(chunker.split(doc))
        return {"chunks": chunks}


# --------------------------------------------------------------------------- #
# Markdown Chunker
# --------------------------------------------------------------------------- #


@registry.register("markdown_chunker")
class MarkdownChunkerNode(BaseNode):
    """Header-aware Markdown chunker that respects document hierarchy."""

    node_type = "markdown_chunker"
    metadata = NodeMetadata(
        category=NodeCategory.CHUNKER,
        display_name="Markdown Chunker",
        description=(
            "Splits Markdown documents at heading boundaries, "
            "preserving the document hierarchy in chunk metadata."
        ),
        icon_name="hash",
        tags=["chunk", "split", "markdown", "headers"],
    )
    input_ports = [_DOCUMENTS_INPUT]
    output_ports = [_CHUNKS_OUTPUT]
    fields = [
        NodeField(
            name="max_chunk_size",
            field_type=FieldType.SLIDER,
            display_name="Max Chunk Size",
            description="Maximum characters per chunk. Sections longer than this are further split.",
            default=DEFAULT_CHUNK_SIZE,
            min_value=200,
            max_value=8000,
            step=100,
        ),
        NodeField(
            name="heading_levels",
            field_type=FieldType.MULTI_SELECT,
            display_name="Split at Heading Levels",
            description="Which Markdown heading levels to use as split boundaries.",
            default=["h1", "h2", "h3"],
            options=["h1", "h2", "h3", "h4", "h5", "h6"],
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.chunking.markdown import MarkdownChunker

        documents = inputs.get("documents", [])
        chunker = MarkdownChunker(
            max_chunk_size=self.get_config("max_chunk_size", DEFAULT_CHUNK_SIZE),
            heading_levels=self.get_config("heading_levels", ["h1", "h2", "h3"]),
        )
        chunks = []
        for doc in documents:
            chunks.extend(chunker.split(doc))
        return {"chunks": chunks}


# --------------------------------------------------------------------------- #
# Semantic Chunker
# --------------------------------------------------------------------------- #


@registry.register("semantic_chunker")
class SemanticChunkerNode(BaseNode):
    """
    Semantic chunker that groups sentences by embedding similarity.

    Uses a lightweight embedding model to compute sentence embeddings,
    then groups consecutive sentences that are semantically similar into
    a single chunk.  Produces topic-coherent chunks at the cost of
    an embedding call during ingestion.
    """

    node_type = "semantic_chunker"
    metadata = NodeMetadata(
        category=NodeCategory.CHUNKER,
        display_name="Semantic Chunker",
        description=(
            "Groups sentences by semantic similarity using embeddings. "
            "Produces topic-coherent chunks. Requires an embedding model."
        ),
        icon_name="brain",
        tags=["chunk", "split", "semantic", "embedding", "similarity"],
    )
    input_ports = [_DOCUMENTS_INPUT]
    output_ports = [_CHUNKS_OUTPUT]
    fields = [
        NodeField(
            name="embedding_model",
            field_type=FieldType.STRING,
            display_name="Embedding Model",
            description="Sentence-transformers model name used to encode sentences.",
            default="all-MiniLM-L6-v2",
        ),
        NodeField(
            name="similarity_threshold",
            field_type=FieldType.SLIDER,
            display_name="Similarity Threshold",
            description="Cosine similarity below which a new chunk is started.",
            default=0.5,
            min_value=0.1,
            max_value=0.99,
            step=0.01,
        ),
        NodeField(
            name="max_chunk_size",
            field_type=FieldType.INTEGER,
            display_name="Max Chunk Size",
            description="Hard upper limit on chunk size in characters.",
            default=2000,
            min_value=200,
            max_value=8000,
            advanced=True,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.chunking.semantic import SemanticChunker

        documents = inputs.get("documents", [])
        chunker = SemanticChunker(
            embedding_model=self.get_config("embedding_model", "all-MiniLM-L6-v2"),
            similarity_threshold=self.get_config("similarity_threshold", 0.5),
            max_chunk_size=self.get_config("max_chunk_size", 2000),
        )
        chunks = []
        for doc in documents:
            chunks.extend(chunker.split(doc))
        return {"chunks": chunks}


# --------------------------------------------------------------------------- #
# Parent-Child Chunker
# --------------------------------------------------------------------------- #


@registry.register("parent_child_chunker")
class ParentChildChunkerNode(BaseNode):
    """
    Parent-child chunker.

    Produces two sets of chunks per document:
    - Small child chunks for precise retrieval.
    - Larger parent chunks for rich context in the generation prompt.

    The child chunks carry a ``parent_chunk_id`` reference so the context
    builder can fetch the full parent context after retrieval.
    """

    node_type = "parent_child_chunker"
    metadata = NodeMetadata(
        category=NodeCategory.CHUNKER,
        display_name="Parent-Child Chunker",
        description=(
            "Produces small child chunks for retrieval and larger parent chunks "
            "for context. Improves answer quality by searching at fine granularity "
            "but generating with broader context."
        ),
        icon_name="layers",
        tags=["chunk", "split", "parent", "child", "hierarchical"],
    )
    input_ports = [_DOCUMENTS_INPUT]
    output_ports = [_CHUNKS_OUTPUT]
    fields = [
        NodeField(
            name="parent_chunk_size",
            field_type=FieldType.SLIDER,
            display_name="Parent Chunk Size",
            description="Size of the larger parent chunks (characters).",
            default=2000,
            min_value=500,
            max_value=8000,
            step=100,
        ),
        NodeField(
            name="child_chunk_size",
            field_type=FieldType.SLIDER,
            display_name="Child Chunk Size",
            description="Size of the smaller child chunks used for retrieval (characters).",
            default=400,
            min_value=50,
            max_value=2000,
            step=50,
        ),
        NodeField(
            name="child_overlap",
            field_type=FieldType.SLIDER,
            display_name="Child Overlap",
            description="Overlap between consecutive child chunks (characters).",
            default=50,
            min_value=0,
            max_value=400,
            step=25,
        ),
    ]

    async def execute(self, **inputs: Any) -> dict[str, Any]:
        from app.chunking.parent_child import ParentChildChunker

        documents = inputs.get("documents", [])
        chunker = ParentChildChunker(
            parent_chunk_size=self.get_config("parent_chunk_size", 2000),
            child_chunk_size=self.get_config("child_chunk_size", 400),
            child_overlap=self.get_config("child_overlap", 50),
        )
        chunks = []
        for doc in documents:
            chunks.extend(chunker.split(doc))
        return {"chunks": chunks}
