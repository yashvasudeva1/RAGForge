"""
Agentic RAG pipeline template.

Uses query rewriting before retrieval and multi-query retrieval to
significantly improve recall on ambiguous or under-specified questions.

Topology
--------
QueryInput --> QueryRewriter --> MultiQueryRetriever
           --> CohereReranker --> ContextBuilder
           --> PromptTemplate --> OpenAIGenerator --> AnswerOutput
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
        config={"file_path": ""},
        label="PDF Loader",
    ),
    NodeConfig(
        id="n-chunker",
        type="semantic_chunker",
        position=NodePosition(x=560, y=80),
        config={
            "embedding_model": "all-MiniLM-L6-v2",
            "similarity_threshold": 0.5,
            "max_chunk_size": 1500,
        },
        label="Semantic Chunker",
    ),
    NodeConfig(
        id="n-embedder",
        type="openai_embedder",
        position=NodePosition(x=800, y=80),
        config={"model": "text-embedding-3-large"},
        label="OpenAI Embedder (Large)",
    ),
    NodeConfig(
        id="n-vectorstore-index",
        type="qdrant_vectorstore",
        position=NodePosition(x=1040, y=80),
        config={"mode": "index", "collection_name": "agentic_rag", "url": "http://localhost:6333"},
        label="Qdrant (Index)",
    ),
    NodeConfig(
        id="n-query-rewriter",
        type="query_rewriter",
        position=NodePosition(x=320, y=400),
        config={"strategy": "expand", "llm_provider": "openai", "llm_model": "gpt-4o-mini"},
        label="Query Rewriter",
    ),
    NodeConfig(
        id="n-multi-query",
        type="multi_query_retriever",
        position=NodePosition(x=560, y=400),
        config={
            "vectorstore_type": "qdrant",
            "collection_name": "agentic_rag",
            "top_k": 10,
            "num_queries": 4,
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
        },
        label="Multi-Query Retriever",
    ),
    NodeConfig(
        id="n-reranker",
        type="cohere_reranker",
        position=NodePosition(x=800, y=400),
        config={"model": "rerank-english-v3.0", "top_n": 5},
        label="Cohere Reranker",
    ),
    NodeConfig(
        id="n-context",
        type="context_builder",
        position=NodePosition(x=1040, y=400),
        config={"format": "xml_tags", "include_source": True, "max_chars": 8000},
        label="Context Builder",
    ),
    NodeConfig(
        id="n-prompt",
        type="prompt_template",
        position=NodePosition(x=1280, y=400),
        config={
            "template": (
                "You are an expert assistant. Answer the question using ONLY the provided context. "
                "If the answer is not in the context, say 'I don't have enough information to answer this.'\n\n"
                "{{ context }}\n\n"
                "Question: {{ query }}\n\n"
                "Answer:"
            )
        },
        label="Prompt Template",
    ),
    NodeConfig(
        id="n-generator",
        type="openai_generator",
        position=NodePosition(x=1520, y=400),
        config={"model": "gpt-4o", "temperature": 0.0, "max_tokens": 2048},
        label="GPT-4o Generator",
    ),
    NodeConfig(
        id="n-output",
        type="answer_output",
        position=NodePosition(x=1760, y=400),
        config={"show_sources": True, "max_sources": 10},
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
    # Query path
    EdgeConfig(id="e4", source_node_id="n-query-input", source_port="query",
               target_node_id="n-query-rewriter", target_port="query"),
    EdgeConfig(id="e5", source_node_id="n-query-rewriter", source_port="query",
               target_node_id="n-multi-query", target_port="query"),
    EdgeConfig(id="e6", source_node_id="n-multi-query", source_port="retrieval_results",
               target_node_id="n-reranker", target_port="retrieval_results"),
    EdgeConfig(id="e7", source_node_id="n-query-input", source_port="query",
               target_node_id="n-reranker", target_port="query"),
    EdgeConfig(id="e8", source_node_id="n-reranker", source_port="retrieval_results",
               target_node_id="n-context", target_port="retrieval_results"),
    EdgeConfig(id="e9", source_node_id="n-context", source_port="context",
               target_node_id="n-prompt", target_port="context"),
    EdgeConfig(id="e10", source_node_id="n-query-input", source_port="query",
               target_node_id="n-prompt", target_port="query"),
    EdgeConfig(id="e11", source_node_id="n-prompt", source_port="prompt",
               target_node_id="n-generator", target_port="prompt"),
    EdgeConfig(id="e12", source_node_id="n-reranker", source_port="retrieval_results",
               target_node_id="n-generator", target_port="retrieval_results"),
    EdgeConfig(id="e13", source_node_id="n-generator", source_port="generation_result",
               target_node_id="n-output", target_port="generation_result"),
]

AGENTIC_RAG_TEMPLATE = PipelineConfig(
    id="template-agentic-rag",
    name="Agentic RAG",
    description=(
        "Advanced pipeline with query rewriting, multi-query retrieval, "
        "Cohere reranking, semantic chunking, and GPT-4o generation. "
        "Produces high-quality answers on complex or ambiguous questions."
    ),
    nodes=_NODES,
    edges=_EDGES,
    tags=["template", "agentic", "multi-query", "reranking", "cohere", "semantic", "qdrant"],
    is_template=True,
)
