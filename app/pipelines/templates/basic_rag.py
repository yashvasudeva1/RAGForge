"""
Basic RAG pipeline template.

Topology
--------
QueryInput --> PDFLoader --> RecursiveChunker --> OpenAIEmbedder
    --> ChromaVectorStore (index) --> ChromaVectorStore (query)
    --> ContextBuilder --> PromptTemplate --> OpenAIGenerator --> AnswerOutput

This is the minimal working RAG pipeline and the recommended starting
point for new users.
"""

from __future__ import annotations

from app.domain.pipelines.models import EdgeConfig, NodeConfig, NodePosition, PipelineConfig

_NODES = [
    NodeConfig(
        id="n-query-input",
        type="query_input",
        position=NodePosition(x=80, y=300),
        config={},
        label="Query",
    ),
    NodeConfig(
        id="n-pdf-loader",
        type="pdf_loader",
        position=NodePosition(x=320, y=100),
        config={"file_path": "", "extract_images": False},
        label="PDF Loader",
    ),
    NodeConfig(
        id="n-chunker",
        type="recursive_chunker",
        position=NodePosition(x=560, y=100),
        config={"chunk_size": 1000, "chunk_overlap": 200},
        label="Recursive Chunker",
    ),
    NodeConfig(
        id="n-embedder",
        type="openai_embedder",
        position=NodePosition(x=800, y=100),
        config={"model": "text-embedding-3-small"},
        label="OpenAI Embedder",
    ),
    NodeConfig(
        id="n-vectorstore-index",
        type="chroma_vectorstore",
        position=NodePosition(x=1040, y=100),
        config={"mode": "index", "collection_name": "basic_rag", "persist_dir": "./data/chroma"},
        label="Chroma (Index)",
    ),
    NodeConfig(
        id="n-vectorstore-query",
        type="chroma_vectorstore",
        position=NodePosition(x=320, y=400),
        config={"mode": "query", "collection_name": "basic_rag", "persist_dir": "./data/chroma", "top_k": 5},
        label="Chroma (Query)",
    ),
    NodeConfig(
        id="n-context",
        type="context_builder",
        position=NodePosition(x=560, y=400),
        config={"format": "numbered", "include_source": True},
        label="Context Builder",
    ),
    NodeConfig(
        id="n-prompt",
        type="prompt_template",
        position=NodePosition(x=800, y=400),
        config={
            "template": (
                "You are a helpful assistant. Use only the provided context to answer the question.\n\n"
                "Context:\n{{ context }}\n\n"
                "Question: {{ query }}\n\n"
                "Answer:"
            )
        },
        label="Prompt Template",
    ),
    NodeConfig(
        id="n-generator",
        type="openai_generator",
        position=NodePosition(x=1040, y=400),
        config={"model": "gpt-4o-mini", "temperature": 0.0, "max_tokens": 1024},
        label="OpenAI Generator",
    ),
    NodeConfig(
        id="n-output",
        type="answer_output",
        position=NodePosition(x=1280, y=400),
        config={"show_sources": True},
        label="Answer",
    ),
]

_EDGES = [
    # Ingestion path
    EdgeConfig(id="e1", source_node_id="n-pdf-loader", source_port="documents",
               target_node_id="n-chunker", target_port="documents"),
    EdgeConfig(id="e2", source_node_id="n-chunker", source_port="chunks",
               target_node_id="n-embedder", target_port="chunks"),
    EdgeConfig(id="e3", source_node_id="n-embedder", source_port="embedded_chunks",
               target_node_id="n-vectorstore-index", target_port="embedded_chunks"),
    # Query path
    EdgeConfig(id="e4", source_node_id="n-query-input", source_port="query",
               target_node_id="n-vectorstore-query", target_port="query"),
    EdgeConfig(id="e5", source_node_id="n-vectorstore-query", source_port="retrieval_results",
               target_node_id="n-context", target_port="retrieval_results"),
    EdgeConfig(id="e6", source_node_id="n-context", source_port="context",
               target_node_id="n-prompt", target_port="context"),
    EdgeConfig(id="e7", source_node_id="n-query-input", source_port="query",
               target_node_id="n-prompt", target_port="query"),
    EdgeConfig(id="e8", source_node_id="n-prompt", source_port="prompt",
               target_node_id="n-generator", target_port="prompt"),
    EdgeConfig(id="e9", source_node_id="n-vectorstore-query", source_port="retrieval_results",
               target_node_id="n-generator", target_port="retrieval_results"),
    EdgeConfig(id="e10", source_node_id="n-generator", source_port="generation_result",
               target_node_id="n-output", target_port="generation_result"),
]

BASIC_RAG_TEMPLATE = PipelineConfig(
    id="template-basic-rag",
    name="Basic RAG",
    description=(
        "The minimal working RAG pipeline. Loads a PDF, chunks it with recursive splitting, "
        "embeds with OpenAI, stores in ChromaDB, retrieves the top 5 chunks, "
        "and generates an answer with GPT-4o-mini."
    ),
    nodes=_NODES,
    edges=_EDGES,
    tags=["template", "basic", "openai", "chroma", "pdf"],
    is_template=True,
)
