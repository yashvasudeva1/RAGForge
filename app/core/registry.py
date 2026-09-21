"""
RAGForge node registry.

The registry is a singleton that maps node type strings to their
corresponding ``BaseNode`` subclasses.  Nodes self-register via the
``@registry.register`` decorator defined here.

Usage — registering a node
--------------------------
    from app.core.registry import registry

    @registry.register("pdf_loader")
    class PDFLoaderNode(BaseNode):
        ...

Usage — looking up a node
--------------------------
    from app.core.registry import registry

    node_cls = registry.get("pdf_loader")   # raises NodeNotFoundError if missing
    node = node_cls()

Usage — listing all schemas (consumed by GET /api/nodes/)
---------------------------------------------------------
    schemas = registry.get_all_schemas()
    # Returns a list of dicts, one per registered node type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Type

from app.core.exceptions import NodeNotFoundError
from app.core.logging import get_logger

if TYPE_CHECKING:
    # Avoid a circular import at runtime — BaseNode imports from core too.
    from app.nodes.base import BaseNode

logger = get_logger(__name__)

# Type alias for node classes
NodeClass = Type["BaseNode"]


class NodeRegistry:
    """
    Central registry of all available node types.

    This is implemented as a plain class (not a module-level singleton via
    ``__new__``) so it can be easily replaced with a test double.  The
    module-level ``registry`` object below is the canonical singleton used
    everywhere in the application.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, NodeClass] = {}

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register(self, node_type: str) -> Callable[[NodeClass], NodeClass]:
        """
        Class decorator that registers a ``BaseNode`` subclass under the
        given ``node_type`` identifier.

        Parameters
        ----------
        node_type:
            The unique string key for this node (e.g. ``"pdf_loader"``).
            Must be lowercase with underscores.  This is the value stored
            in ``NodeConfig.type`` inside pipeline JSON configs and is
            what the frontend uses to reference node types.

        Returns
        -------
        The original class, unmodified.

        Raises
        ------
        ValueError
            If a different class is already registered under ``node_type``.
        """

        def decorator(cls: NodeClass) -> NodeClass:
            if node_type in self._nodes and self._nodes[node_type] is not cls:
                raise ValueError(
                    f"Cannot register '{cls.__name__}' as '{node_type}': "
                    f"'{self._nodes[node_type].__name__}' is already registered "
                    f"under that key."
                )
            self._nodes[node_type] = cls
            logger.debug("Registered node", extra={"node_type": node_type, "class": cls.__name__})
            return cls

        return decorator

    def register_class(self, node_type: str, cls: NodeClass) -> None:
        """
        Programmatically register a node class without using the decorator.

        Useful in tests or plugin systems that cannot use the decorator syntax.
        """
        self.register(node_type)(cls)

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #

    def get(self, node_type: str) -> NodeClass:
        """
        Return the node class registered under ``node_type``.

        Raises
        ------
        NodeNotFoundError
            If no node is registered under that key.
        """
        if node_type not in self._nodes:
            raise NodeNotFoundError(node_type)
        return self._nodes[node_type]

    def has(self, node_type: str) -> bool:
        """Return ``True`` if a node is registered under ``node_type``."""
        return node_type in self._nodes

    def all_types(self) -> list[str]:
        """Return a sorted list of all registered node type strings."""
        return sorted(self._nodes.keys())

    # ------------------------------------------------------------------ #
    # Schema export (consumed by the API)
    # ------------------------------------------------------------------ #

    def get_all_schemas(self) -> list[dict]:
        """
        Return a list of JSON-serialisable schema dicts — one for every
        registered node type.

        The frontend calls ``GET /api/nodes/`` to fetch this list and uses
        it to:
          - Populate the node palette sidebar
          - Auto-render config panels for each node
          - Validate port connections before sending to the backend

        Returns
        -------
        list[dict]
            Each dict is the output of ``node_instance.get_schema()``.
        """
        schemas: list[dict] = []
        for node_type, node_cls in sorted(self._nodes.items()):
            try:
                instance = node_cls()
                schemas.append(instance.get_schema())
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not generate schema for node",
                    extra={"node_type": node_type, "error": str(exc)},
                )
        return schemas

    def get_schema(self, node_type: str) -> dict:
        """
        Return the schema dict for a single node type.

        Raises
        ------
        NodeNotFoundError
            If the node type is not registered.
        """
        node_cls = self.get(node_type)
        return node_cls().get_schema()

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_type: str) -> bool:
        return node_type in self._nodes

    def __repr__(self) -> str:
        return f"NodeRegistry({len(self._nodes)} nodes: {self.all_types()})"


# --------------------------------------------------------------------------- #
# Canonical singleton
# --------------------------------------------------------------------------- #

#: The application-wide node registry.
#: Import this object in every node module to self-register.
registry: NodeRegistry = NodeRegistry()
