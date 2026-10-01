from __future__ import annotations

import logging
from datetime import UTC, datetime

from rich.logging import RichHandler

LOG_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
LOG_FORMAT = "%(asctime)s %(message)s"


class UtcFormatter(logging.Formatter):
    """UTC ISO-8601 timestamps on every loco log line."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(record.created, tz=UTC).strftime(
            datefmt or LOG_TIME_FORMAT
        )


def utcnow_iso() -> str:
    return datetime.now(UTC).strftime(LOG_TIME_FORMAT)


def format_elapsed(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 10:
        return f"{seconds:.2f}s"
    return f"{seconds:.1f}s"


def setup_logging(level: str = "INFO") -> None:
    handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        show_time=False,
        markup=False,
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level.upper())
    console_handler.setFormatter(UtcFormatter(LOG_FORMAT))

    logging.basicConfig(
        level=level.upper(),
        handlers=[handler, console_handler],
        force=True,
    )
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
