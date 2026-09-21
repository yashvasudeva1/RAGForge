# RAGForge

RAGForge is a visual drag-and-drop builder for Retrieval-Augmented Generation (RAG) pipelines. It lets you assemble, configure, and run RAG pipelines by connecting nodes on a canvas — no code required for experimentation, but fully programmable when you need it.

---

## What It Is

RAG pipelines are notoriously tedious to experiment with. Swapping an embedding model, trying a different chunking strategy, or comparing dense retrieval against hybrid retrieval typically means editing Python code, rerunning scripts, and manually comparing outputs. RAGForge removes that friction.

You drag nodes onto a canvas, connect them with edges, fill in their configuration via a side panel, and click Run. The pipeline executes on the backend and streams results back to the canvas in real time. You can save, share, and version your pipeline configurations as JSON files.

---

## Architecture

```
+---------------------------------------------------------------+
|                      FRONTEND  (React)                        |
|                                                               |
|   Node Palette    |    React Flow Canvas    |  Config Panel  |
|   (sidebar)       |    (drag, connect,      |  (auto-rendered|
|                   |     pan, zoom)          |   from schema) |
|                                                               |
|         Zustand store    REST API    WebSocket stream         |
+---------------------------------------------------------------+
                              |
                         FastAPI Server
                              |
+---------------------------------------------------------------+
|                      BACKEND  (Python)                        |
|                                                               |
|   Node Registry                                               |
|   (maps type string -> class, exposes JSON schemas)           |
|                                                               |
|   Pipeline Engine                                             |
|   +-- Validator  (DAG check, port compatibility)             |
|   +-- Compiler   (topological sort, dependency resolution)   |
|   +-- Executor   (async, per-node streaming via WebSocket)   |
|   +-- Serializer (save/load/export pipeline JSON)            |
|                                                               |
|   Component Runners                                           |
|   +-- Loaders       (PDF, DOCX, Markdown, HTML, CSV, JSON,  |
|   |                  Web, PPTX, Excel)                       |
|   +-- Chunkers      (Recursive, Character, Token, Sentence,  |
|   |                  Markdown, Semantic, Parent-Child)       |
|   +-- Embedders     (OpenAI, Google, HuggingFace, Cohere,   |
|   |                  Voyage, Local)                          |
|   +-- Vector Stores (Chroma, Qdrant, FAISS, Pinecone,       |
|   |                  Weaviate, Milvus, LanceDB)              |
|   +-- Retrievers    (Dense, BM25, Hybrid/RRF, MMR,          |
|   |                  Multi-Query, Contextual)                |
|   +-- Rerankers     (Cross-Encoder, Cohere, Voyage, LLM)    |
|   +-- Generators    (OpenAI, Anthropic, Google, Groq,       |
|                      Mistral, Ollama, Local)                 |
+---------------------------------------------------------------+
```

The defining design principle is **schema-driven nodes**. Every node class declares its input ports, output ports, and configuration fields in a JSON schema. The FastAPI backend serves all schemas via `GET /api/nodes/`. The frontend reads those schemas to:

- Populate the draggable node palette
- Auto-render configuration panels without any frontend-specific node code
- Validate port connections before submitting to the backend

Adding a new component to the Python backend automatically makes it available in the frontend with zero additional frontend work.

---

## Node Types

| Category | Nodes |
|---|---|
| Input | Query Input |
| Loader | PDF, DOCX, Markdown, HTML, CSV, JSON/JSONL, Web URL, PPTX, Excel |
| Chunker | Recursive, Character, Token, Sentence, Markdown, Semantic, Parent-Child |
| Embedder | OpenAI, Google, HuggingFace, Cohere, Voyage, Local (sentence-transformers) |
| Vector Store | ChromaDB, Qdrant, FAISS, Pinecone, Weaviate, Milvus, LanceDB |
| Retriever | Dense, BM25, Hybrid (RRF), MMR, Multi-Query, Contextual Compression |
| Reranker | Cross-Encoder, Cohere Rerank, Voyage Rerank, LLM Reranker |
| Prompt | Prompt Template (Jinja2), Context Builder, Query Rewriter |
| Generator | OpenAI, Anthropic, Google Gemini, Groq, Mistral, Ollama, Local |
| Utility | Text Cleaner, Conditional Router, Output Parser |
| Output | Answer Output |

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Node.js 20 or higher (for the frontend)
- Git

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/yashvasudeva1/RAGForge.git
cd RAGForge
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install the backend**

For minimal installation (core + ChromaDB + OpenAI only):

```bash
pip install -e .
```

For the full installation with all providers and tools:

```bash
pip install -e ".[all]"
```

You can also install specific extras:

```bash
pip install -e ".[loaders,vectorstores,embeddings,llm]"
```

**4. Configure environment variables**

```bash
cp .env.example .env
```

Open `.env` and fill in the API keys for the providers you want to use. At minimum, you need at least one LLM provider key and one embedding provider key. ChromaDB works locally with no external service.

**5. Install the frontend**

```bash
cd frontend
npm install
```

### Running the Application

**Start the backend (from the project root):**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Start the frontend (in a separate terminal):**

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000` in your browser.

### Docker (optional)

```bash
docker compose up
```

This starts the backend on port 8000 and the frontend on port 3000.

---

## Pipeline Configuration Format

A pipeline is stored as a JSON document with this structure:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Basic RAG",
  "description": "Load a PDF, chunk it, embed with OpenAI, store in Chroma, retrieve, and generate.",
  "nodes": [
    {
      "id": "node-1",
      "type": "query_input",
      "position": { "x": 50, "y": 200 },
      "config": {}
    },
    {
      "id": "node-2",
      "type": "pdf_loader",
      "position": { "x": 250, "y": 200 },
      "config": {
        "file_path": "/data/report.pdf",
        "extract_images": false
      }
    },
    {
      "id": "node-3",
      "type": "recursive_chunker",
      "position": { "x": 450, "y": 200 },
      "config": {
        "chunk_size": 1000,
        "chunk_overlap": 200
      }
    }
  ],
  "edges": [
    {
      "id": "edge-1",
      "source_node_id": "node-1",
      "source_port": "query",
      "target_node_id": "node-5",
      "target_port": "query"
    },
    {
      "id": "edge-2",
      "source_node_id": "node-2",
      "source_port": "documents",
      "target_node_id": "node-3",
      "target_port": "documents"
    }
  ]
}
```

Pipelines can be exported from the canvas and re-imported, committed to version control, or shared with teammates.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/nodes/` | List all registered node schemas |
| GET | `/api/nodes/{type}` | Get schema for a single node type |
| GET | `/api/pipelines/` | List all saved pipelines |
| POST | `/api/pipelines/` | Save a new pipeline |
| GET | `/api/pipelines/{id}` | Load a pipeline by ID |
| PUT | `/api/pipelines/{id}` | Update a pipeline |
| DELETE | `/api/pipelines/{id}` | Delete a pipeline |
| POST | `/api/pipelines/{id}/execute` | Execute a pipeline |
| GET | `/api/executions/{job_id}` | Poll execution status |
| WS | `/ws/executions/{job_id}` | Stream real-time execution events |
| GET | `/api/health` | Health check |

Full interactive API documentation is available at `http://localhost:8000/docs` when the server is running.

---

## Adding a Custom Node

To add a new node to the system:

**1. Create a class that inherits from `BaseNode`:**

```python
# app/nodes/my_custom_node.py

from app.core.registry import registry
from app.core.constants import NodeCategory, PortType, FieldType
from app.nodes.base import BaseNode, NodeField, NodePortDefinition, NodeMetadata

@registry.register("my_custom_chunker")
class MyCustomChunkerNode(BaseNode):
    node_type = "my_custom_chunker"
    metadata = NodeMetadata(
        category=NodeCategory.CHUNKER,
        display_name="My Custom Chunker",
        description="Splits text using my custom logic.",
        icon_name="scissors",
    )
    input_ports = [
        NodePortDefinition(name="documents", port_type=PortType.DOCUMENTS, required=True),
    ]
    output_ports = [
        NodePortDefinition(name="chunks", port_type=PortType.CHUNKS),
    ]
    fields = [
        NodeField(
            name="split_token",
            field_type=FieldType.STRING,
            default="---",
            description="Token to split on.",
        ),
    ]

    async def execute(self, documents, **kwargs):
        split_token = self.config.get("split_token", "---")
        chunks = []
        for doc in documents:
            parts = doc.content.split(split_token)
            # ... build Chunk objects ...
        return {"chunks": chunks}
```

**2. Import the module** in `app/nodes/__init__.py` so it self-registers on startup.

That is all. The node will appear in the frontend palette automatically.

---

## Project Structure

```
ragforge/
|-- app/                         Backend Python application
|   |-- core/                    Config, logging, exceptions, registry
|   |-- domain/                  Pydantic data models (Document, Chunk, etc.)
|   |-- nodes/                   Node implementations (BaseNode + all node types)
|   |-- pipelines/               Pipeline compiler, executor, serializer, validator
|   |-- ingestion/               Document loader runners
|   |-- chunking/                Text splitter runners
|   |-- embeddings/              Embedding model runners
|   |-- vectorstores/            Vector store connectors
|   |-- retrieval/               Retrieval strategy runners
|   |-- reranking/               Reranker runners
|   |-- generation/              LLM generation runners
|   |-- api/                     FastAPI routes and schemas
|   `-- main.py                  Application entrypoint
|-- frontend/                    React TypeScript frontend
|   `-- src/
|       |-- canvas/              React Flow canvas component
|       |-- nodes/               Custom React Flow node renderers
|       |-- panels/              Palette, config, and output panels
|       |-- stores/              Zustand state management
|       |-- services/            API and WebSocket clients
|       |-- types/               TypeScript type definitions
|       `-- pages/               Builder and home pages
|-- configs/                     YAML/JSON configuration presets
|-- data/                        Runtime data (pipelines, vector stores)
|-- docs/                        Additional documentation
|-- notebooks/                   Jupyter notebooks for experimentation
|-- plugins/                     Community or third-party node plugins
|-- tests/                       Pytest test suite
|-- .env.example                 Environment variable template
|-- pyproject.toml               Python project metadata and dependencies
|-- docker-compose.yml           Docker Compose for local development
`-- README.md                    This file
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend language | Python 3.11+ |
| Backend framework | FastAPI |
| Data validation | Pydantic v2 |
| Frontend language | TypeScript |
| Frontend framework | React |
| Node canvas | React Flow |
| State management | Zustand |
| UI components | shadcn/ui + Tailwind CSS |
| Default vector store | ChromaDB |
| Default LLM | OpenAI GPT-4o-mini |
| Default embeddings | OpenAI text-embedding-3-small |
| Containerisation | Docker + Docker Compose |

---

## Configuration Reference

All settings are controlled via environment variables (see `.env.example`). The most important ones:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `DEFAULT_LLM_MODEL` | `gpt-4o-mini` | Default generation model |
| `DEFAULT_EMBEDDING_MODEL` | `text-embedding-3-small` | Default embedding model |
| `DEFAULT_VECTORSTORE` | `chroma` | Default vector store backend |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB local storage path |
| `DEFAULT_CHUNK_SIZE` | `1000` | Default chunk size in characters |
| `DEFAULT_CHUNK_OVERLAP` | `200` | Default chunk overlap in characters |
| `DEFAULT_RETRIEVAL_K` | `5` | Default number of chunks to retrieve |
| `LOG_LEVEL` | `INFO` | Log verbosity |
| `DEBUG` | `false` | Enable debug mode |

---

## Contributing

Contributions are welcome. The most impactful things to contribute are:

- New node types (any new loader, chunker, embedder, vectorstore, retriever, reranker, or generator)
- Pipeline templates
- Bug reports with reproduction steps

Please open an issue before submitting a pull request for significant changes.

---

## License

MIT License. See `LICENSE` for details.
