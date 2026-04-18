"""Audit logging for moderation events."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class DecisionCode(str, Enum):
    """Possible outcomes from the moderation pipeline."""

    ALLOW = "ALLOW"
    BLOCK_DETERMINISTIC = "BLOCK_DETERMINISTIC"
    BLOCK_LLM = "BLOCK_LLM"
    BLOCK_POST = "BLOCK_POST"
    ERROR_FAIL_CLOSED = "ERROR_FAIL_CLOSED"


@dataclass
class ModerationEvent:
    """Structured record of a single moderation decision."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stage: str = ""
    decision_code: DecisionCode = DecisionCode.ALLOW
    matched_category: str | None = None
    model_used: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


_LEVEL_MAP: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class AuditLogger:
    """Logs :class:`ModerationEvent` instances via Python's stdlib ``logging``.

    Parameters
    ----------
    log_level:
        Severity level for audit messages (``"debug"``, ``"info"``, etc.).
    redact_content:
        When ``True``, raw user or model text is never written to the log.
    """

    def __init__(self, log_level: str = "info", redact_content: bool = True) -> None:
        self._level = _LEVEL_MAP.get(log_level.lower(), logging.INFO)
        self.redact_content = redact_content
        self._logger = logging.getLogger("offlineaihelper.audit")

    def log_event(self, event: ModerationEvent) -> None:
        """Emit *event* to the audit logger at the configured level."""
        self._logger.log(
            self._level,
            "MODERATION_EVENT event_id=%s stage=%s decision=%s category=%s model=%s ts=%s",
            event.event_id,
            event.stage,
            event.decision_code.value,
            event.matched_category or "—",
            event.model_used or "—",
            event.timestamp,
        )
