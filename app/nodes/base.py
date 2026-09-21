"""
RAGForge BaseNode — the foundational abstraction for all pipeline nodes.

Every draggable block in the visual canvas corresponds to a Python class that
inherits from ``BaseNode``.  The base class defines:

  - The schema contract (``get_schema()``) — consumed by ``GET /api/nodes/``
    and used by the frontend to auto-render config panels and validate edges.
  - The execution contract (``execute()``) — the async method called by the
    pipeline executor when a node's turn comes in the execution order.

Design goals
------------
- Adding a new node requires only creating a new subclass + registering it.
  Zero frontend changes are needed for the new node to appear in the palette.
- The schema is the single source of truth for both UI rendering and
  backend validation.
- Nodes are stateless by default; config values are injected before execution.

Usage
-----
    from app.core.registry import registry
    from app.core.constants import NodeCategory, PortType, FieldType
    from app.nodes.base import BaseNode, NodeField, NodePortDefinition, NodeMetadata

    @registry.register("my_loader")
    class MyLoaderNode(BaseNode):
        node_type = "my_loader"
        metadata = NodeMetadata(
            category=NodeCategory.LOADER,
            display_name="My Loader",
            description="Loads data from my custom source.",
            icon_name="file",
        )
        input_ports = []
        output_ports = [
            NodePortDefinition(name="documents", port_type=PortType.DOCUMENTS),
        ]
        fields = [
            NodeField(
                name="source_path",
                field_type=FieldType.STRING,
                required=True,
                description="Path to the source file.",
            ),
        ]

        async def execute(self, **kwargs) -> dict:
            source_path = self.config["source_path"]
            # ... load logic ...
            return {"documents": [...]}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from app.core.constants import FieldType, NodeCategory, PortType
from app.core.exceptions import NodeValidationError


# --------------------------------------------------------------------------- #
# Schema building blocks
# --------------------------------------------------------------------------- #


class NodeField(BaseModel):
    """
    Describes one user-configurable field on a node.

    The frontend maps ``field_type`` to the appropriate form control
    (text input, slider, dropdown, etc.) and uses ``options`` for select
    fields and ``min`` / ``max`` / ``step`` for sliders.

    Attributes
    ----------
    name:
        Internal field identifier.  Used as the key in ``NodeConfig.config``.
    field_type:
        Controls which UI control is rendered.
    display_name:
        Human-readable label shown above the field.  Defaults to a
        title-cased version of ``name`` if not provided.
    description:
        Help text shown below or beside the field.
    required:
        When True the pipeline validator rejects execution if this field
        is absent from ``NodeConfig.config``.
    default:
        Default value pre-filled in the config panel.
    options:
        For ``SELECT`` and ``MULTI_SELECT`` fields — list of allowed values.
    min_value:
        Minimum value for numeric and ``SLIDER`` fields.
    max_value:
        Maximum value for numeric and ``SLIDER`` fields.
    step:
        Step size for ``SLIDER`` and numeric fields.
    placeholder:
        Placeholder text for text inputs.
    group:
        Optional group name for organising fields into collapsible sections.
    advanced:
        When True the field is hidden in a collapsed "Advanced" section.
    """

    name: str
    field_type: FieldType = FieldType.STRING
    display_name: str | None = None
    description: str | None = None
    required: bool = False
    default: Any = None
    options: list[Any] | None = None
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    placeholder: str | None = None
    group: str | None = None
    advanced: bool = False

    def resolved_display_name(self) -> str:
        """Return display_name, falling back to a title-cased version of name."""
        return self.display_name or self.name.replace("_", " ").title()

    def to_schema_dict(self) -> dict:
        """Serialise this field to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "field_type": self.field_type.value,
            "display_name": self.resolved_display_name(),
            "description": self.description,
            "required": self.required,
            "default": self.default,
            "options": self.options,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step": self.step,
            "placeholder": self.placeholder,
            "group": self.group,
            "advanced": self.advanced,
        }


class NodePortDefinition(BaseModel):
    """
    Describes one data port on a node (either input or output).

    The pipeline validator checks that connected ports share the same
    ``port_type`` (or that one of them is ``PortType.ANY``).

    Attributes
    ----------
    name:
        Port identifier.  Used as the key name when the executor passes
        data between nodes.
    port_type:
        The data type flowing through this port.
    display_name:
        Human-readable port label shown on the node card.
    description:
        Tooltip text for the port handle in the canvas.
    required:
        For input ports — whether this connection is mandatory for the
        pipeline to be valid.
    multiple:
        For input ports — whether multiple edges may connect to this port
        (e.g. a merge node accepting several document streams).
    """

    name: str
    port_type: PortType = PortType.ANY
    display_name: str | None = None
    description: str | None = None
    required: bool = True
    multiple: bool = False

    def resolved_display_name(self) -> str:
        return self.display_name or self.name.replace("_", " ").title()

    def to_schema_dict(self) -> dict:
        return {
            "name": self.name,
            "port_type": self.port_type.value,
            "display_name": self.resolved_display_name(),
            "description": self.description,
            "required": self.required,
            "multiple": self.multiple,
        }


class NodeMetadata(BaseModel):
    """
    Display-level metadata for a node type.

    Used by the frontend to render the node card, palette entry, and
    documentation tooltip.

    Attributes
    ----------
    category:
        The node category (affects palette grouping and node colour).
    display_name:
        Human-readable name shown in the palette and on the canvas.
    description:
        Short (one to two sentence) description of what the node does.
    icon_name:
        Lucide icon name used on the node card.
    docs_url:
        Optional link to external documentation.
    tags:
        Searchable tags for the palette search bar.
    """

    category: NodeCategory
    display_name: str
    description: str
    icon_name: str = "box"
    docs_url: str | None = None
    tags: list[str] = Field(default_factory=list)

    def to_schema_dict(self) -> dict:
        return {
            "category": self.category.value,
            "display_name": self.display_name,
            "description": self.description,
            "icon_name": self.icon_name,
            "docs_url": self.docs_url,
            "tags": self.tags,
        }


# --------------------------------------------------------------------------- #
# BaseNode
# --------------------------------------------------------------------------- #


class BaseNode(ABC):
    """
    Abstract base class for all RAGForge pipeline nodes.

    Every node that appears in the drag-and-drop canvas must subclass this
    and implement ``execute()``.

    Class-level attributes (define on subclasses)
    ---------------------------------------------
    node_type:
        Unique string identifier, e.g. ``"pdf_loader"``.  Must match the
        key passed to ``@registry.register()``.
    metadata:
        A ``NodeMetadata`` instance describing the node for the UI.
    input_ports:
        List of ``NodePortDefinition`` objects for input data connections.
    output_ports:
        List of ``NodePortDefinition`` objects for output data connections.
    fields:
        List of ``NodeField`` objects for user-configurable parameters.

    Instance attributes
    -------------------
    config:
        Dictionary of user-supplied configuration values (set by the
        pipeline executor before calling ``execute()``).
    """

    # Subclasses MUST override these.
    node_type: str = ""
    metadata: NodeMetadata
    input_ports: list[NodePortDefinition] = []
    output_ports: list[NodePortDefinition] = []
    fields: list[NodeField] = []

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Parameters
        ----------
        config:
            Pre-loaded configuration dict.  Normally injected by the
            pipeline executor rather than passed directly.
        """
        self.config: dict[str, Any] = config or {}
        self._apply_field_defaults()

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #

    def _apply_field_defaults(self) -> None:
        """
        Populate ``self.config`` with field defaults for any keys that are
        not already present.  Called once in ``__init__``.
        """
        for field in self.fields:
            if field.name not in self.config and field.default is not None:
                self.config[field.name] = field.default

    def set_config(self, config: dict[str, Any]) -> "BaseNode":
        """
        Merge ``config`` into the node's current configuration.

        Returns ``self`` to allow chaining.
        """
        self.config.update(config)
        return self

    def validate_config(self) -> None:
        """
        Validate that all required fields are present in ``self.config``.

        Called by the pipeline executor before ``execute()``.

        Raises
        ------
        NodeValidationError
            For the first missing required field found.
        """
        for field in self.fields:
            if field.required and field.name not in self.config:
                raise NodeValidationError(
                    node_id=self.node_type,
                    field=field.name,
                    reason="Required field is missing from node configuration.",
                )

    # ------------------------------------------------------------------ #
    # Schema (consumed by the API)
    # ------------------------------------------------------------------ #

    def get_schema(self) -> dict:
        """
        Return the complete JSON-serialisable schema for this node type.

        This is the payload returned by ``GET /api/nodes/{type}`` and
        collected in ``GET /api/nodes/`` for the full palette.

        The schema drives:
        - Palette rendering (icon, colour, display name, description)
        - Config panel auto-generation (fields with types, defaults, etc.)
        - Edge validation (port types and required flags)
        """
        return {
            "node_type": self.node_type,
            "metadata": self.metadata.to_schema_dict(),
            "input_ports": [p.to_schema_dict() for p in self.input_ports],
            "output_ports": [p.to_schema_dict() for p in self.output_ports],
            "fields": [f.to_schema_dict() for f in self.fields],
        }

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def execute(self, **inputs: Any) -> dict[str, Any]:
        """
        Run the node's logic.

        Parameters
        ----------
        **inputs:
            Keyword arguments keyed by input port name, containing the data
            passed from upstream nodes.  For example, a chunker node receives
            ``documents=[Document(...), ...]`` from a loader node.

        Returns
        -------
        dict[str, Any]
            A dictionary keyed by output port name.  The pipeline executor
            uses these values to feed downstream nodes.

        Notes
        -----
        - This method must be awaitable (``async def``).
        - Raise ``NodeExecutionError`` for recoverable errors that should
          be reported per-node in the execution log.
        - Raise any ``RAGForgeError`` subclass for domain-level errors.
        - Let unexpected exceptions bubble up; the executor will catch and
          wrap them.
        """

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def get_config(self, key: str, default: Any = None) -> Any:
        """Retrieve a config value with an optional fallback."""
        return self.config.get(key, default)

    def require_config(self, key: str) -> Any:
        """
        Retrieve a required config value.

        Raises
        ------
        NodeValidationError
            If the key is absent from ``self.config``.
        """
        if key not in self.config:
            raise NodeValidationError(
                node_id=self.node_type,
                field=key,
                reason="Required config field is missing.",
            )
        return self.config[key]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.node_type!r}, config={self.config})"
