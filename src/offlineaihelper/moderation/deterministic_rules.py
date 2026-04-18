"""Deterministic text-matching rules for the moderation pipeline."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_RULE_SETS_DIR = Path(__file__).parent / "rule_sets"
_DEFAULT_MAX_LENGTH = 32768


@dataclass
class DeterministicResult:
    """Result returned by :class:`DeterministicChecker`."""

    passed: bool
    matched_category: str | None = None
    matched_pattern: str | None = None
    severity: str | None = None
    action: str | None = None


@dataclass
class _CompiledCategory:
    name: str
    severity: str
    action: str
    regexes: list[re.Pattern[str]] = field(default_factory=list)


def _load_categories(rule_sets_dir: Path) -> list[_CompiledCategory]:
    """Load and compile all rule-set JSON files from *rule_sets_dir*."""
    categories: list[_CompiledCategory] = []
    for json_file in sorted(rule_sets_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping rule set %s: %s", json_file, exc)
            continue
        for cat_name, cat_cfg in data.get("categories", {}).items():
            compiled = _CompiledCategory(
                name=cat_name,
                severity=cat_cfg.get("severity", "medium"),
                action=cat_cfg.get("action", "flag"),
            )
            for pattern in cat_cfg.get("patterns", []):
                try:
                    compiled.regexes.append(
                        re.compile(pattern, re.IGNORECASE | re.UNICODE)
                    )
                except re.error as exc:
                    logger.warning("Invalid regex '%s' in category '%s': %s", pattern, cat_name, exc)
            categories.append(compiled)
    return categories


class DeterministicChecker:
    """Rule-based content checker using regex patterns loaded from JSON rule sets.

    Parameters
    ----------
    rule_sets_dir:
        Directory containing ``*.json`` rule-set files.  Defaults to the
        bundled ``rule_sets/`` directory.
    enabled_categories:
        Whitelist of category names to evaluate.  ``None`` means all categories.
    max_length:
        Maximum allowed text length in characters.  Texts longer than this
        are treated as blocked to prevent DoS.
    """

    def __init__(
        self,
        rule_sets_dir: Path | str | None = None,
        enabled_categories: list[str] | None = None,
        max_length: int = _DEFAULT_MAX_LENGTH,
    ) -> None:
        _dir = Path(rule_sets_dir) if rule_sets_dir is not None else _DEFAULT_RULE_SETS_DIR
        all_categories = _load_categories(_dir)
        if enabled_categories is not None:
            self._categories = [c for c in all_categories if c.name in enabled_categories]
        else:
            self._categories = all_categories
        self._max_length = max_length
        logger.debug(
            "DeterministicChecker loaded %d categor(y/ies): %s",
            len(self._categories),
            [c.name for c in self._categories],
        )

    def check(self, text: str) -> DeterministicResult:
        """Check *text* against all enabled rules.

        Returns a :class:`DeterministicResult` with ``passed=False`` if any
        ``"block"`` pattern matches, or if the text exceeds ``max_length``.
        """
        if len(text) > self._max_length:
            logger.info("Text exceeds max length %d", self._max_length)
            return DeterministicResult(
                passed=False,
                matched_category="max_length",
                matched_pattern=None,
                severity="high",
                action="block",
            )

        for category in self._categories:
            for regex in category.regexes:
                match = regex.search(text)
                if match:
                    logger.info(
                        "Deterministic match: category=%s pattern=%r severity=%s action=%s",
                        category.name,
                        regex.pattern,
                        category.severity,
                        category.action,
                    )
                    if category.action == "block":
                        return DeterministicResult(
                            passed=False,
                            matched_category=category.name,
                            matched_pattern=regex.pattern,
                            severity=category.severity,
                            action=category.action,
                        )

        return DeterministicResult(passed=True)
