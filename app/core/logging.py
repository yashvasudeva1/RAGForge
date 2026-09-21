"""
RAGForge structured logging.

Provides a factory function ``get_logger`` that returns a named
``logging.Logger`` with consistent formatting across the application.

Usage
-----
    from app.core.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Loading document", extra={"path": "/data/report.pdf"})

If the ``rich`` package is installed the console handler uses Rich's
``RichHandler`` for coloured, highlighted output in development.
Otherwise it falls back to a plain ``StreamHandler``.
"""

import logging
import sys
from typing import Any

from app.core.config import get_settings

# Track whether the root logger has already been configured so we do not
# add duplicate handlers on repeated calls to ``get_logger``.
_configured: bool = False

# Log record format used when Rich is not available.
_PLAIN_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def _configure_root_logger() -> None:
    """
    Configure the root logger once per process.

    Selects between Rich and plain formatting based on availability,
    and applies the log level from application settings.
    """
    global _configured  # noqa: PLW0603
    if _configured:
        return

    settings = get_settings()
    level: int = getattr(logging, settings.log_level, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any pre-existing handlers to avoid duplicates.
    root.handlers.clear()

    handler: logging.Handler
    try:
        from rich.logging import RichHandler  # type: ignore[import-untyped]

        handler = RichHandler(
            level=level,
            show_time=True,
            show_path=True,
            rich_tracebacks=True,
            tracebacks_show_locals=settings.debug,
            markup=True,
        )
    except ImportError:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(fmt=_PLAIN_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        handler.setLevel(level)

    root.addHandler(handler)

    # Silence overly verbose third-party loggers.
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "anthropic",
        "chromadb",
        "qdrant_client",
        "sentence_transformers",
        "transformers",
        "torch",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger, configuring the root logger on first call.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module, which produces
        a dot-separated hierarchy (e.g. ``app.nodes.loader``).

    Returns
    -------
    logging.Logger
        A standard library ``Logger`` instance.
    """
    _configure_root_logger()
    return logging.getLogger(name)


class BoundLogger:
    """
    A thin wrapper around ``logging.Logger`` that binds a fixed set of
    context fields to every log record.

    Useful for adding per-request or per-execution context without
    threading it through every function call.

    Parameters
    ----------
    logger:
        The underlying ``logging.Logger`` to delegate to.
    context:
        A dictionary of key-value pairs added to every ``extra`` dict.

    Example
    -------
        logger = get_logger(__name__)
        bound = BoundLogger(logger, {"pipeline_id": "abc123", "node_id": "n1"})
        bound.info("Node started")
    """

    def __init__(self, logger: logging.Logger, context: dict[str, Any]) -> None:
        self._logger = logger
        self._context = context

    def _merge(self, extra: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(self._context)
        if extra:
            merged.update(extra)
        return merged

    def debug(self, msg: str, extra: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._logger.debug(msg, extra=self._merge(extra), **kwargs)

    def info(self, msg: str, extra: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._logger.info(msg, extra=self._merge(extra), **kwargs)

    def warning(self, msg: str, extra: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._logger.warning(msg, extra=self._merge(extra), **kwargs)

    def error(self, msg: str, extra: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._logger.error(msg, extra=self._merge(extra), **kwargs)

    def exception(self, msg: str, extra: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._logger.exception(msg, extra=self._merge(extra), **kwargs)

    def bind(self, **context: Any) -> "BoundLogger":
        """Return a new BoundLogger with additional context fields merged in."""
        return BoundLogger(self._logger, {**self._context, **context})
