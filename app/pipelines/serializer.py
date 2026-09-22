"""
Pipeline serializer for RAGForge.

Handles saving, loading, and exporting ``PipelineConfig`` objects to/from
JSON files on disk.  Also generates a human-readable YAML export for
version control and sharing.

Storage layout
--------------
    data/pipelines/
    ├── <pipeline_id>.json          # full pipeline config
    └── templates/
        ├── basic_rag.json
        ├── hybrid_rag.json
        └── advanced_rag.json

Usage
-----
    serializer = PipelineSerializer()

    # Save
    serializer.save(pipeline)

    # Load by ID
    pipeline = serializer.load("abc123")

    # List all saved pipelines
    summaries = serializer.list_all()

    # Export as YAML string (for sharing / git)
    yaml_str = serializer.to_yaml(pipeline)

    # Import from JSON string (from frontend paste or upload)
    pipeline = serializer.from_json_string(json_str)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import PipelineNotFoundError
from app.core.logging import get_logger
from app.domain.pipelines.models import PipelineConfig

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Pipeline summary (used in list views)
# --------------------------------------------------------------------------- #


class PipelineSummary:
    """Lightweight view of a pipeline for listing in the gallery."""

    def __init__(self, pipeline: PipelineConfig) -> None:
        self.id = pipeline.id
        self.name = pipeline.name
        self.description = pipeline.description
        self.node_count = len(pipeline.nodes)
        self.edge_count = len(pipeline.edges)
        self.tags = pipeline.tags
        self.is_template = pipeline.is_template
        self.created_at = pipeline.created_at.isoformat()
        self.updated_at = pipeline.updated_at.isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "tags": self.tags,
            "is_template": self.is_template,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# --------------------------------------------------------------------------- #
# Serializer
# --------------------------------------------------------------------------- #


class PipelineSerializer:
    """
    Persists ``PipelineConfig`` objects as JSON files on disk.

    Parameters
    ----------
    storage_dir:
        Root directory for pipeline storage.
        Defaults to ``settings.pipeline_storage_dir``.
    """

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        settings = get_settings()
        self._root = Path(storage_dir or settings.pipeline_storage_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #

    def save(self, pipeline: PipelineConfig) -> Path:
        """
        Save ``pipeline`` to disk as JSON.

        If a file already exists for this pipeline ID it will be overwritten.
        ``pipeline.updated_at`` is refreshed before saving.

        Returns the path of the saved file.
        """
        pipeline.touch()
        path = self._path_for(pipeline.id)
        path.write_text(
            pipeline.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Pipeline saved",
            extra={"pipeline_id": pipeline.id, "path": str(path)},
        )
        return path

    def load(self, pipeline_id: str) -> PipelineConfig:
        """
        Load a pipeline by ID from disk.

        Raises
        ------
        PipelineNotFoundError
            If no file exists for ``pipeline_id``.
        """
        path = self._path_for(pipeline_id)
        if not path.exists():
            raise PipelineNotFoundError(pipeline_id)

        raw = path.read_text(encoding="utf-8")
        pipeline = PipelineConfig.model_validate_json(raw)
        logger.debug("Pipeline loaded", extra={"pipeline_id": pipeline_id})
        return pipeline

    def delete(self, pipeline_id: str) -> None:
        """
        Delete a pipeline from disk.

        Raises
        ------
        PipelineNotFoundError
            If no file exists for ``pipeline_id``.
        """
        path = self._path_for(pipeline_id)
        if not path.exists():
            raise PipelineNotFoundError(pipeline_id)
        path.unlink()
        logger.info("Pipeline deleted", extra={"pipeline_id": pipeline_id})

    def exists(self, pipeline_id: str) -> bool:
        """Return True if a pipeline with this ID exists on disk."""
        return self._path_for(pipeline_id).exists()

    def list_all(self, include_templates: bool = True) -> list[PipelineSummary]:
        """
        Return a list of summaries for every pipeline stored on disk.

        Parameters
        ----------
        include_templates:
            When False, template pipelines are excluded from the list.
        """
        summaries: list[PipelineSummary] = []
        for path in sorted(self._root.glob("*.json")):
            try:
                raw = path.read_text(encoding="utf-8")
                pipeline = PipelineConfig.model_validate_json(raw)
                if not include_templates and pipeline.is_template:
                    continue
                summaries.append(PipelineSummary(pipeline))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not parse pipeline file",
                    extra={"path": str(path), "error": str(exc)},
                )
        return summaries

    def list_templates(self) -> list[PipelineSummary]:
        """Return only template pipelines."""
        return [s for s in self.list_all(include_templates=True) if s.is_template]

    # ------------------------------------------------------------------ #
    # Import / export
    # ------------------------------------------------------------------ #

    def to_json_string(self, pipeline: PipelineConfig, indent: int = 2) -> str:
        """Serialise ``pipeline`` to a JSON string."""
        return pipeline.model_dump_json(indent=indent)

    def from_json_string(self, json_str: str) -> PipelineConfig:
        """
        Deserialise a ``PipelineConfig`` from a JSON string.

        Used when the frontend pastes a JSON config or uploads a file.
        """
        return PipelineConfig.model_validate_json(json_str)

    def from_dict(self, data: dict[str, Any]) -> PipelineConfig:
        """Deserialise a ``PipelineConfig`` from a plain dictionary."""
        return PipelineConfig.model_validate(data)

    def to_yaml(self, pipeline: PipelineConfig) -> str:
        """
        Export ``pipeline`` as a YAML string for human-readable sharing
        and version control.

        Requires PyYAML (included in the default dependencies).
        """
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("PyYAML is required for YAML export. Run: pip install pyyaml") from exc

        data = json.loads(pipeline.model_dump_json())
        return yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)

    def from_yaml(self, yaml_str: str) -> PipelineConfig:
        """Deserialise a ``PipelineConfig`` from a YAML string."""
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("PyYAML is required for YAML import.") from exc

        data = yaml.safe_load(yaml_str)
        return PipelineConfig.model_validate(data)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _path_for(self, pipeline_id: str) -> Path:
        """Return the JSON file path for a given pipeline ID."""
        # Sanitise to prevent path traversal
        safe_id = "".join(c for c in pipeline_id if c.isalnum() or c in "-_")
        return self._root / f"{safe_id}.json"
