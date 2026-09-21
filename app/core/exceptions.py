"""
RAGForge exception hierarchy.

All application-specific errors inherit from ``RAGForgeError`` so callers
can catch the full tree with a single ``except RAGForgeError`` clause or
target a specific layer with a more specific type.

Hierarchy
---------
RAGForgeError
├── ConfigurationError
├── NodeError
│   ├── NodeNotFoundError
│   ├── NodeExecutionError
│   └── NodeValidationError
├── PipelineError
│   ├── PipelineValidationError
│   │   ├── PipelineCycleError
│   │   ├── IncompatiblePortError
│   │   └── MissingRequiredPortError
│   ├── PipelineExecutionError
│   └── PipelineNotFoundError
├── LoaderError
│   ├── UnsupportedFileTypeError
│   └── FileReadError
├── ChunkingError
├── EmbeddingError
│   └── EmbeddingProviderError
├── VectorStoreError
│   ├── CollectionNotFoundError
│   └── VectorStorConnectionError
├── RetrievalError
└── GenerationError
    └── LLMProviderError
"""


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #


class RAGForgeError(Exception):
    """
    Base exception for all RAGForge errors.

    Attributes
    ----------
    message:
        Human-readable description of the error.
    details:
        Optional dictionary of additional context (node id, file path, etc.)
        that can be serialised and returned in API error responses.
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict = details or {}

    def to_dict(self) -> dict:
        """Serialise the exception to a JSON-compatible dictionary."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class ConfigurationError(RAGForgeError):
    """Raised when the application is misconfigured (missing API key, etc.)."""


# --------------------------------------------------------------------------- #
# Node Errors
# --------------------------------------------------------------------------- #


class NodeError(RAGForgeError):
    """Base class for errors originating from within a node."""


class NodeNotFoundError(NodeError):
    """Raised when a node type cannot be found in the registry."""

    def __init__(self, node_type: str) -> None:
        super().__init__(
            message=f"No node registered with type '{node_type}'.",
            details={"node_type": node_type},
        )


class NodeExecutionError(NodeError):
    """
    Raised when a node fails during execution.

    Wraps the original exception so it can be reported per-node in the
    pipeline execution log.
    """

    def __init__(self, node_id: str, node_type: str, cause: Exception) -> None:
        super().__init__(
            message=f"Node '{node_id}' (type={node_type}) failed: {cause}",
            details={
                "node_id": node_id,
                "node_type": node_type,
                "cause": str(cause),
                "cause_type": type(cause).__name__,
            },
        )
        self.cause = cause


class NodeValidationError(NodeError):
    """Raised when a node's configuration fails validation before execution."""

    def __init__(self, node_id: str, field: str, reason: str) -> None:
        super().__init__(
            message=f"Node '{node_id}' config field '{field}' is invalid: {reason}",
            details={"node_id": node_id, "field": field, "reason": reason},
        )


# --------------------------------------------------------------------------- #
# Pipeline Errors
# --------------------------------------------------------------------------- #


class PipelineError(RAGForgeError):
    """Base class for pipeline-level errors."""


class PipelineValidationError(PipelineError):
    """Raised when a pipeline config fails structural validation."""


class PipelineCycleError(PipelineValidationError):
    """Raised when the pipeline graph contains a cycle."""

    def __init__(self) -> None:
        super().__init__(
            message="Pipeline graph contains a cycle. Pipelines must be directed acyclic graphs (DAGs).",
        )


class IncompatiblePortError(PipelineValidationError):
    """Raised when two connected ports have incompatible data types."""

    def __init__(
        self,
        source_node: str,
        source_port: str,
        target_node: str,
        target_port: str,
        source_type: str,
        target_type: str,
    ) -> None:
        super().__init__(
            message=(
                f"Incompatible port types: "
                f"'{source_node}.{source_port}' ({source_type}) "
                f"cannot connect to '{target_node}.{target_port}' ({target_type})."
            ),
            details={
                "source_node": source_node,
                "source_port": source_port,
                "target_node": target_node,
                "target_port": target_port,
                "source_type": source_type,
                "target_type": target_type,
            },
        )


class MissingRequiredPortError(PipelineValidationError):
    """Raised when a required input port has no incoming connection."""

    def __init__(self, node_id: str, port_name: str) -> None:
        super().__init__(
            message=f"Required input port '{port_name}' on node '{node_id}' has no connection.",
            details={"node_id": node_id, "port_name": port_name},
        )


class PipelineExecutionError(PipelineError):
    """Raised when a pipeline execution fails at runtime."""


class PipelineNotFoundError(PipelineError):
    """Raised when a pipeline config cannot be found in storage."""

    def __init__(self, pipeline_id: str) -> None:
        super().__init__(
            message=f"Pipeline '{pipeline_id}' not found.",
            details={"pipeline_id": pipeline_id},
        )


# --------------------------------------------------------------------------- #
# Loader Errors
# --------------------------------------------------------------------------- #


class LoaderError(RAGForgeError):
    """Base class for document loading errors."""


class UnsupportedFileTypeError(LoaderError):
    """Raised when no loader is available for a given file type or extension."""

    def __init__(self, file_type: str) -> None:
        super().__init__(
            message=f"No loader available for file type '{file_type}'.",
            details={"file_type": file_type},
        )


class FileReadError(LoaderError):
    """Raised when a file cannot be read or parsed."""

    def __init__(self, path: str, cause: Exception) -> None:
        super().__init__(
            message=f"Failed to read file '{path}': {cause}",
            details={"path": path, "cause": str(cause)},
        )
        self.cause = cause


# --------------------------------------------------------------------------- #
# Chunking Errors
# --------------------------------------------------------------------------- #


class ChunkingError(RAGForgeError):
    """Raised when a chunking operation fails."""


# --------------------------------------------------------------------------- #
# Embedding Errors
# --------------------------------------------------------------------------- #


class EmbeddingError(RAGForgeError):
    """Base class for embedding errors."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when an embedding provider API call fails."""

    def __init__(self, provider: str, cause: Exception) -> None:
        super().__init__(
            message=f"Embedding provider '{provider}' failed: {cause}",
            details={"provider": provider, "cause": str(cause)},
        )
        self.cause = cause


# --------------------------------------------------------------------------- #
# Vector Store Errors
# --------------------------------------------------------------------------- #


class VectorStoreError(RAGForgeError):
    """Base class for vector store errors."""


class CollectionNotFoundError(VectorStoreError):
    """Raised when the target collection/index does not exist in the vector store."""

    def __init__(self, collection: str, backend: str) -> None:
        super().__init__(
            message=f"Collection '{collection}' not found in {backend}.",
            details={"collection": collection, "backend": backend},
        )


class VectorStoreConnectionError(VectorStoreError):
    """Raised when the vector store cannot be reached."""

    def __init__(self, backend: str, cause: Exception) -> None:
        super().__init__(
            message=f"Cannot connect to {backend}: {cause}",
            details={"backend": backend, "cause": str(cause)},
        )
        self.cause = cause


# --------------------------------------------------------------------------- #
# Retrieval Errors
# --------------------------------------------------------------------------- #


class RetrievalError(RAGForgeError):
    """Raised when a retrieval operation fails."""


# --------------------------------------------------------------------------- #
# Generation Errors
# --------------------------------------------------------------------------- #


class GenerationError(RAGForgeError):
    """Base class for generation errors."""


class LLMProviderError(GenerationError):
    """Raised when an LLM provider API call fails."""

    def __init__(self, provider: str, model: str, cause: Exception) -> None:
        super().__init__(
            message=f"LLM provider '{provider}' (model={model}) failed: {cause}",
            details={"provider": provider, "model": model, "cause": str(cause)},
        )
        self.cause = cause
