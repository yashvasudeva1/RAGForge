"""
RAGForge nodes package.

Importing this package registers all built-in nodes into the NodeRegistry.
The import order does not matter for registration but is kept logical.

The pipeline engine imports this package on startup so that all nodes
are available before any API request is served.
"""

# ------------------------------------------------------------------ #
# Import all node modules to trigger @registry.register() decorators.
# ------------------------------------------------------------------ #

# Terminal nodes
from app.nodes import input  # noqa: F401
from app.nodes import output  # noqa: F401

# Data ingestion
from app.nodes import loader  # noqa: F401

# Processing
from app.nodes import chunker  # noqa: F401
from app.nodes import embedder  # noqa: F401

# Storage and retrieval
from app.nodes import vectorstore  # noqa: F401
from app.nodes import retriever  # noqa: F401
from app.nodes import reranker  # noqa: F401

# Generation
from app.nodes import llm  # noqa: F401

# Utility
from app.nodes import prompt  # noqa: F401

from app.core.registry import registry

__all__ = ["registry"]
