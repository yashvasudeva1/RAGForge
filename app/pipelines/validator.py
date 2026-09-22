"""
Pipeline validator for RAGForge.

The validator takes a ``PipelineConfig`` and checks that it is structurally
correct before handing it to the compiler.  It catches problems early and
returns clear, actionable error messages to the frontend.

Checks performed
----------------
1. Node types exist in the registry.
2. The graph has no cycles (must be a DAG).
3. All required input ports have an incoming edge.
4. Connected ports share compatible data types.
5. All required node config fields are present.
6. There are no disconnected node islands (optional, soft warning).
7. At least one QueryInputNode is present.
8. At least one AnswerOutputNode is present.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from app.core.constants import PortType
from app.core.exceptions import (
    IncompatiblePortError,
    MissingRequiredPortError,
    NodeNotFoundError,
    PipelineCycleError,
    PipelineValidationError,
)
from app.core.logging import get_logger
from app.core.registry import registry
from app.domain.pipelines.models import EdgeConfig, NodeConfig, PipelineConfig

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Validation result
# --------------------------------------------------------------------------- #


@dataclass
class ValidationWarning:
    """A non-fatal issue found during validation."""

    code: str
    message: str
    node_id: str | None = None


@dataclass
class ValidationResult:
    """
    The outcome of a validation run.

    ``is_valid`` is True only when there are zero errors.
    Warnings do not prevent execution but are surfaced in the UI.
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.is_valid = False
        self.errors.append(message)

    def add_warning(self, code: str, message: str, node_id: str | None = None) -> None:
        self.warnings.append(ValidationWarning(code=code, message=message, node_id=node_id))

    def raise_if_invalid(self) -> None:
        """Raise ``PipelineValidationError`` aggregating all errors if not valid."""
        if not self.is_valid:
            combined = "; ".join(self.errors)
            raise PipelineValidationError(
                message=f"Pipeline validation failed with {len(self.errors)} error(s): {combined}",
                details={"errors": self.errors, "warnings": [w.__dict__ for w in self.warnings]},
            )


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


class PipelineValidator:
    """
    Validates a ``PipelineConfig`` before compilation and execution.

    Usage
    -----
        validator = PipelineValidator()
        result = validator.validate(pipeline_config)
        result.raise_if_invalid()
    """

    def validate(self, pipeline: PipelineConfig) -> ValidationResult:
        """
        Run all validation checks on ``pipeline``.

        Returns a ``ValidationResult`` that callers can inspect or raise.
        """
        result = ValidationResult()

        if not pipeline.nodes:
            result.add_error("Pipeline has no nodes.")
            return result  # Nothing further to check

        # Build lookup structures
        node_map: dict[str, NodeConfig] = {n.id: n for n in pipeline.nodes}
        edges_by_target: dict[str, list[EdgeConfig]] = defaultdict(list)
        edges_by_source: dict[str, list[EdgeConfig]] = defaultdict(list)
        for edge in pipeline.edges:
            edges_by_target[edge.target_node_id].append(edge)
            edges_by_source[edge.source_node_id].append(edge)

        # Run individual checks
        self._check_node_types(pipeline, result)
        self._check_cycles(pipeline, result)
        self._check_required_ports(pipeline, node_map, edges_by_target, result)
        self._check_port_compatibility(pipeline, node_map, result)
        self._check_required_config_fields(pipeline, result)
        self._check_has_input_node(pipeline, result)
        self._check_has_output_node(pipeline, result)
        self._check_orphan_nodes(pipeline, edges_by_source, edges_by_target, result)

        logger.debug(
            "Pipeline validation complete",
            extra={
                "pipeline_id": pipeline.id,
                "is_valid": result.is_valid,
                "errors": len(result.errors),
                "warnings": len(result.warnings),
            },
        )
        return result

    # ------------------------------------------------------------------ #
    # Individual checks
    # ------------------------------------------------------------------ #

    def _check_node_types(self, pipeline: PipelineConfig, result: ValidationResult) -> None:
        """Verify every node type is in the registry."""
        for node in pipeline.nodes:
            if not registry.has(node.type):
                result.add_error(
                    f"Node '{node.id}' has unknown type '{node.type}'. "
                    f"Available types: {registry.all_types()[:10]}..."
                )

    def _check_cycles(self, pipeline: PipelineConfig, result: ValidationResult) -> None:
        """
        Detect cycles using Kahn's topological sort algorithm.

        If the algorithm cannot consume all nodes (because some are stuck in
        a cycle), it reports a cycle error.
        """
        in_degree: dict[str, int] = {n.id: 0 for n in pipeline.nodes}
        adjacency: dict[str, list[str]] = defaultdict(list)

        for edge in pipeline.edges:
            if edge.source_node_id in in_degree and edge.target_node_id in in_degree:
                adjacency[edge.source_node_id].append(edge.target_node_id)
                in_degree[edge.target_node_id] += 1

        queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
        visited = 0

        while queue:
            nid = queue.popleft()
            visited += 1
            for neighbour in adjacency[nid]:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        if visited < len(pipeline.nodes):
            result.add_error(
                "Pipeline graph contains a cycle. "
                "All pipelines must be directed acyclic graphs (DAGs). "
                "Check for circular connections between nodes."
            )

    def _check_required_ports(
        self,
        pipeline: PipelineConfig,
        node_map: dict[str, NodeConfig],
        edges_by_target: dict[str, list[EdgeConfig]],
        result: ValidationResult,
    ) -> None:
        """Verify all required input ports have an incoming connection."""
        for node_cfg in pipeline.nodes:
            if not registry.has(node_cfg.type):
                continue  # Already reported above

            node_cls = registry.get(node_cfg.type)
            node_instance = node_cls()

            connected_ports: set[str] = {
                edge.target_port for edge in edges_by_target.get(node_cfg.id, [])
            }

            for port in node_instance.input_ports:
                if port.required and port.name not in connected_ports:
                    result.add_error(
                        f"Node '{node_cfg.id}' (type={node_cfg.type}): "
                        f"required input port '{port.name}' has no incoming connection."
                    )

    def _check_port_compatibility(
        self,
        pipeline: PipelineConfig,
        node_map: dict[str, NodeConfig],
        result: ValidationResult,
    ) -> None:
        """
        Verify connected ports share compatible PortTypes.

        ANY is compatible with everything.
        Other port types must match exactly.
        """
        for edge in pipeline.edges:
            src_cfg = node_map.get(edge.source_node_id)
            tgt_cfg = node_map.get(edge.target_node_id)

            if not src_cfg or not tgt_cfg:
                result.add_error(
                    f"Edge '{edge.id}' references a node that does not exist in the pipeline."
                )
                continue

            if not (registry.has(src_cfg.type) and registry.has(tgt_cfg.type)):
                continue  # Type errors already reported

            src_node = registry.get(src_cfg.type)()
            tgt_node = registry.get(tgt_cfg.type)()

            src_port = next(
                (p for p in src_node.output_ports if p.name == edge.source_port), None
            )
            tgt_port = next(
                (p for p in tgt_node.input_ports if p.name == edge.target_port), None
            )

            if src_port is None:
                result.add_error(
                    f"Node '{edge.source_node_id}' (type={src_cfg.type}) has no output "
                    f"port named '{edge.source_port}'."
                )
                continue

            if tgt_port is None:
                result.add_error(
                    f"Node '{edge.target_node_id}' (type={tgt_cfg.type}) has no input "
                    f"port named '{edge.target_port}'."
                )
                continue

            # Check type compatibility (ANY is wildcard)
            if (
                src_port.port_type != PortType.ANY
                and tgt_port.port_type != PortType.ANY
                and src_port.port_type != tgt_port.port_type
            ):
                result.add_error(
                    f"Incompatible port types on edge from "
                    f"'{edge.source_node_id}.{edge.source_port}' "
                    f"({src_port.port_type.value}) to "
                    f"'{edge.target_node_id}.{edge.target_port}' "
                    f"({tgt_port.port_type.value})."
                )

    def _check_required_config_fields(
        self, pipeline: PipelineConfig, result: ValidationResult
    ) -> None:
        """Verify required node config fields are present."""
        for node_cfg in pipeline.nodes:
            if not registry.has(node_cfg.type):
                continue

            node_cls = registry.get(node_cfg.type)
            node_instance = node_cls(config=node_cfg.config)

            for field_def in node_instance.fields:
                if field_def.required and field_def.name not in node_cfg.config:
                    result.add_error(
                        f"Node '{node_cfg.id}' (type={node_cfg.type}): "
                        f"required config field '{field_def.name}' is missing."
                    )

    def _check_has_input_node(self, pipeline: PipelineConfig, result: ValidationResult) -> None:
        """Warn if no QueryInputNode is present."""
        has_input = any(n.type == "query_input" for n in pipeline.nodes)
        if not has_input:
            result.add_warning(
                code="NO_INPUT_NODE",
                message=(
                    "Pipeline has no Query Input node. "
                    "Without one, the pipeline cannot receive a user query."
                ),
            )

    def _check_has_output_node(self, pipeline: PipelineConfig, result: ValidationResult) -> None:
        """Warn if no AnswerOutputNode is present."""
        has_output = any(n.type == "answer_output" for n in pipeline.nodes)
        if not has_output:
            result.add_warning(
                code="NO_OUTPUT_NODE",
                message=(
                    "Pipeline has no Answer Output node. "
                    "Without one, the final answer will not be surfaced to the user."
                ),
            )

    def _check_orphan_nodes(
        self,
        pipeline: PipelineConfig,
        edges_by_source: dict[str, list[EdgeConfig]],
        edges_by_target: dict[str, list[EdgeConfig]],
        result: ValidationResult,
    ) -> None:
        """Warn about nodes with no connections at all."""
        for node in pipeline.nodes:
            has_out = bool(edges_by_source.get(node.id))
            has_in = bool(edges_by_target.get(node.id))
            if not has_out and not has_in:
                result.add_warning(
                    code="ORPHAN_NODE",
                    message=(
                        f"Node '{node.id}' (type={node.type}) has no connections. "
                        "It will be skipped during execution."
                    ),
                    node_id=node.id,
                )
