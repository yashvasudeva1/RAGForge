"""
Pipeline compiler for RAGForge.

The compiler takes a validated ``PipelineConfig`` and produces a
``CompiledPipeline`` — an execution plan with nodes sorted in topological
order and data-flow dependencies explicitly resolved.

The executor consumes a ``CompiledPipeline`` without needing to re-parse
the graph.

Compilation steps
-----------------
1. Build an adjacency list from edges.
2. Topologically sort nodes (Kahn's algorithm).
3. For each node, record which upstream node + port feeds each input port.
4. Instantiate each node class from the registry with its config.
5. Return a ``CompiledPipeline`` ready for the executor.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.core.registry import registry
from app.domain.pipelines.models import EdgeConfig, NodeConfig, PipelineConfig
from app.nodes.base import BaseNode

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Data structures produced by the compiler
# --------------------------------------------------------------------------- #


@dataclass
class PortBinding:
    """
    Describes where a node's input port gets its data from.

    Attributes
    ----------
    source_node_id:
        ID of the upstream node.
    source_port:
        Name of the output port on the upstream node.
    target_port:
        Name of the input port on this node.
    """

    source_node_id: str
    source_port: str
    target_port: str


@dataclass
class CompiledNode:
    """
    A node that has been instantiated and is ready for execution.

    Attributes
    ----------
    node_id:
        Pipeline-unique node instance ID.
    node_type:
        Registry key string.
    instance:
        Instantiated ``BaseNode`` with config pre-loaded.
    input_bindings:
        List of ``PortBinding`` objects — describes which upstream outputs
        feed which input ports on this node.
    static_inputs:
        Pipeline-level inputs (e.g. query) injected before execution begins.
    """

    node_id: str
    node_type: str
    instance: BaseNode
    input_bindings: list[PortBinding] = field(default_factory=list)
    static_inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledPipeline:
    """
    The output of the compiler — a fully resolved, ordered execution plan.

    Attributes
    ----------
    pipeline_id:
        ID of the source ``PipelineConfig``.
    pipeline_name:
        Display name.
    execution_order:
        Node IDs in topological order (safe execution sequence).
    nodes:
        Map of node_id -> ``CompiledNode``.
    """

    pipeline_id: str
    pipeline_name: str
    execution_order: list[str]
    nodes: dict[str, CompiledNode]

    @property
    def ordered_nodes(self) -> list[CompiledNode]:
        """Return compiled nodes in safe execution order."""
        return [self.nodes[nid] for nid in self.execution_order if nid in self.nodes]


# --------------------------------------------------------------------------- #
# Compiler
# --------------------------------------------------------------------------- #


class PipelineCompiler:
    """
    Compiles a validated ``PipelineConfig`` into a ``CompiledPipeline``.

    Usage
    -----
        compiler = PipelineCompiler()
        compiled = compiler.compile(pipeline_config)
    """

    def compile(self, pipeline: PipelineConfig) -> CompiledPipeline:
        """
        Compile ``pipeline`` into an executable plan.

        Parameters
        ----------
        pipeline:
            A ``PipelineConfig`` that has already been validated.

        Returns
        -------
        CompiledPipeline
        """
        logger.info(
            "Compiling pipeline '%s' (id=%s)", pipeline.name, pipeline.id,
        )

        # Build lookup maps
        node_map: dict[str, NodeConfig] = {n.id: n for n in pipeline.nodes}
        adjacency: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {n.id: 0 for n in pipeline.nodes}
        edges_by_target: dict[str, list[EdgeConfig]] = defaultdict(list)

        for edge in pipeline.edges:
            if edge.source_node_id in node_map and edge.target_node_id in node_map:
                adjacency[edge.source_node_id].append(edge.target_node_id)
                in_degree[edge.target_node_id] += 1
                edges_by_target[edge.target_node_id].append(edge)

        # Topological sort (Kahn's)
        queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
        execution_order: list[str] = []

        while queue:
            nid = queue.popleft()
            execution_order.append(nid)
            for neighbour in sorted(adjacency[nid]):  # sort for determinism
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        # Instantiate nodes and resolve bindings
        compiled_nodes: dict[str, CompiledNode] = {}

        for node_id in execution_order:
            node_cfg = node_map[node_id]
            node_cls = registry.get(node_cfg.type)
            node_instance = node_cls(config=dict(node_cfg.config))

            bindings: list[PortBinding] = [
                PortBinding(
                    source_node_id=edge.source_node_id,
                    source_port=edge.source_port,
                    target_port=edge.target_port,
                )
                for edge in edges_by_target.get(node_id, [])
            ]

            compiled_nodes[node_id] = CompiledNode(
                node_id=node_id,
                node_type=node_cfg.type,
                instance=node_instance,
                input_bindings=bindings,
            )

        logger.info(
            "Pipeline compiled successfully",
            extra={
                "pipeline_id": pipeline.id,
                "node_count": len(compiled_nodes),
                "execution_order": execution_order,
            },
        )

        return CompiledPipeline(
            pipeline_id=pipeline.id,
            pipeline_name=pipeline.name,
            execution_order=execution_order,
            nodes=compiled_nodes,
        )
