"""
Hybrid RAG pipeline template.

Adds BM25 + dense retrieval with RRF fusion and cross-encoder reranking
on top of the basic RAG pipeline. Generally produces better retrieval
quality than pure dense retrieval alone.

Topology
--------
QueryInput --> (ingestion path same as basic_rag)
QueryInput --> HybridRetriever (dense + BM25 fused via RRF)
           --> CrossEncoderReranker
           --> ContextBuilder --> PromptTemplate --> OpenAIGenerator --> AnswerOutput
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
        position=NodePosition(x=320, y=80),
        config={"file_path": "", "extract_images": False},
        label="PDF Loader",
    ),
    NodeConfig(
        id="n-chunker",
        type="recursive_chunker",
        position=NodePosition(x=560, y=80),
        config={"chunk_size": 512, "chunk_overlap": 128},
        label="Chunker",
    ),
    NodeConfig(
        id="n-embedder",
        type="openai_embedder",
        position=NodePosition(x=800, y=80),
        config={"model": "text-embedding-3-small"},
        label="OpenAI Embedder",
    ),
    NodeConfig(
        id="n-vectorstore-index",
        type="chroma_vectorstore",
        position=NodePosition(x=1040, y=80),
        config={"mode": "index", "collection_name": "hybrid_rag", "persist_dir": "./data/chroma"},
        label="Chroma (Index)",
    ),
    NodeConfig(
        id="n-hybrid-retriever",
        type="hybrid_retriever",
        position=NodePosition(x=320, y=400),
        config={
            "vectorstore_type": "chroma",
            "collection_name": "hybrid_rag",
            "top_k": 10,
            "dense_weight": 0.6,
            "rrf_k": 60,
            "fetch_k": 25,
        },
        label="Hybrid Retriever (RRF)",
    ),
    NodeConfig(
        id="n-reranker",
        type="cross_encoder_reranker",
        position=NodePosition(x=560, y=400),
        config={
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_n": 5,
            "device": "cpu",
        },
        label="Cross-Encoder Reranker",
    ),
    NodeConfig(
        id="n-context",
        type="context_builder",
        position=NodePosition(x=800, y=400),
        config={"format": "numbered", "include_source": True},
        label="Context Builder",
    ),
    NodeConfig(
        id="n-prompt",
        type="prompt_template",
        position=NodePosition(x=1040, y=400),
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
        position=NodePosition(x=1280, y=400),
        config={"model": "gpt-4o-mini", "temperature": 0.0, "max_tokens": 1024},
        label="OpenAI Generator",
    ),
    NodeConfig(
        id="n-output",
        type="answer_output",
        position=NodePosition(x=1520, y=400),
        config={"show_sources": True},
        label="Answer",
    ),
]

_EDGES = [
    # Ingestion
    EdgeConfig(id="e1", source_node_id="n-pdf-loader", source_port="documents",
               target_node_id="n-chunker", target_port="documents"),
    EdgeConfig(id="e2", source_node_id="n-chunker", source_port="chunks",
               target_node_id="n-embedder", target_port="chunks"),
    EdgeConfig(id="e3", source_node_id="n-embedder", source_port="embedded_chunks",
               target_node_id="n-vectorstore-index", target_port="embedded_chunks"),
    # Query
    EdgeConfig(id="e4", source_node_id="n-query-input", source_port="query",
               target_node_id="n-hybrid-retriever", target_port="query"),
    EdgeConfig(id="e5", source_node_id="n-hybrid-retriever", source_port="retrieval_results",
               target_node_id="n-reranker", target_port="retrieval_results"),
    EdgeConfig(id="e6", source_node_id="n-query-input", source_port="query",
               target_node_id="n-reranker", target_port="query"),
    EdgeConfig(id="e7", source_node_id="n-reranker", source_port="retrieval_results",
               target_node_id="n-context", target_port="retrieval_results"),
    EdgeConfig(id="e8", source_node_id="n-context", source_port="context",
               target_node_id="n-prompt", target_port="context"),
    EdgeConfig(id="e9", source_node_id="n-query-input", source_port="query",
               target_node_id="n-prompt", target_port="query"),
    EdgeConfig(id="e10", source_node_id="n-prompt", source_port="prompt",
               target_node_id="n-generator", target_port="prompt"),
    EdgeConfig(id="e11", source_node_id="n-reranker", source_port="retrieval_results",
               target_node_id="n-generator", target_port="retrieval_results"),
    EdgeConfig(id="e12", source_node_id="n-generator", source_port="generation_result",
               target_node_id="n-output", target_port="generation_result"),
]

HYBRID_RAG_TEMPLATE = PipelineConfig(
    id="template-hybrid-rag",
    name="Hybrid RAG with Reranking",
    description=(
        "Combines dense vector retrieval and BM25 keyword search via Reciprocal Rank Fusion (RRF), "
        "then reranks with a local cross-encoder. Consistently outperforms pure dense retrieval "
        "especially on keyword-heavy queries."
    ),
    nodes=_NODES,
    edges=_EDGES,
    tags=["template", "hybrid", "rrf", "reranking", "cross-encoder", "openai"],
    is_template=True,
)
