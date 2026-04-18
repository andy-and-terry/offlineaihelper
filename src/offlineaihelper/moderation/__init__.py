"""Moderation package — exports both sync (DeterministicRules-based) and async APIs."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Pattern

if TYPE_CHECKING:
    from offlineaihelper.sync_ollama import SyncOllamaClient

CATEGORIES = (
    "self_harm",
    "violence",
    "sexual_content",
    "hate_harassment",
    "illicit_behavior",
    "malware_hacking",
    "privacy_pii",
    "safe",
)

RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class ModerationResult:
    """Result of a single deterministic or LLM moderation check."""

    category: str
    risk: str
    reason: str
    source: str
    requires_confirmation: bool = False


class DeterministicRules:
    """Fast regex-based classifier for the sync moderation pipeline."""

    def __init__(self) -> None:
        self._patterns: dict[str, Pattern[str]] = {
            "self_harm": re.compile(r"\b(self[_\s-]harm|hurt myself|suicide)\b", re.IGNORECASE),
            "violence": re.compile(r"\b(shoot|bomb|stab|murder|attack)\b", re.IGNORECASE),
            "sexual_content": re.compile(r"\b(explicit sex|porn|sexual act|nude)\b", re.IGNORECASE),
            "hate_harassment": re.compile(r"\b(slur|hate (group|people)|harass|racist attack)\b", re.IGNORECASE),
            "illicit_behavior": re.compile(r"\b(steal|fraud|counterfeit|bypass law)\b", re.IGNORECASE),
            "malware_hacking": re.compile(r"\b(malware|ransomware|keylogger|phishing kit|exploit)\b", re.IGNORECASE),
            "privacy_pii": re.compile(
                r"\b(ssn|social security number|credit card number|private key|password dump)\b", re.IGNORECASE
            ),
        }
        self._destructive_commands = re.compile(
            r"\b(rm\s+-rf|del\s+/f|rmdir\s+/s|format\s+|diskpart|git\s+reset\s+--hard|git\s+clean\s+-fdx|push\s+-f|--force)\b",
            re.IGNORECASE,
        )

    def classify(self, text: str) -> ModerationResult:
        """Return a :class:`ModerationResult` for *text*."""
        candidate = text or ""
        for category, pattern in self._patterns.items():
            if pattern.search(candidate):
                return ModerationResult(
                    category=category,
                    risk="high",
                    reason="deterministic pattern match",
                    source="rules",
                )

        destructive = self._destructive_commands.search(candidate)
        if destructive:
            return ModerationResult(
                category="illicit_behavior",
                risk="medium",
                reason=f"potentially destructive command detected: {destructive.group(0)}",
                source="rules",
                requires_confirmation=True,
            )

        return ModerationResult(
            category="safe",
            risk="low",
            reason="no risky pattern detected",
            source="rules",
        )


class LLMModerator:
    """LLM-based moderation using a :class:`SyncOllamaClient`."""

    def __init__(self, client: "SyncOllamaClient", model: str) -> None:
        self.client = client
        self.model = model

    def classify(self, text: str, stage: str) -> ModerationResult:
        """Return a :class:`ModerationResult` by querying the LLM."""
        prompt = (
            "You are a moderation model. Return only compact JSON with keys: category, risk, reason. "
            f"Category must be one of: {', '.join(CATEGORIES)}. "
            "Risk must be one of: low, medium, high. "
            f"Stage: {stage}. Text: {text}"
        )
        raw = self.client.generate(self.model, prompt)
        parsed = _extract_json(raw)

        category = parsed.get("category", "safe")
        if category not in CATEGORIES:
            category = "safe"

        risk = parsed.get("risk", "low")
        if risk not in {"low", "medium", "high"}:
            risk = "low"

        return ModerationResult(
            category=category,
            risk=risk,
            reason=parsed.get("reason", "LLM moderation result"),
            source="llm",
        )


class ModerationPipeline:
    """Runs deterministic rules followed by LLM moderation."""

    def __init__(self, rules: DeterministicRules, llm_moderator: LLMModerator) -> None:
        self.rules = rules
        self.llm_moderator = llm_moderator

    def classify(self, text: str, stage: str) -> ModerationResult:
        """Classify *text* at *stage* (``"pre"`` or ``"post"``)."""
        rules_result = self.rules.classify(text)
        if rules_result.risk == "high":
            return rules_result

        llm_result = self.llm_moderator.classify(text, stage=stage)
        if rules_result.requires_confirmation:
            return ModerationResult(
                category=llm_result.category if llm_result.category != "safe" else rules_result.category,
                risk=max(llm_result.risk, rules_result.risk, key=_risk_value),
                reason=f"{rules_result.reason}; {llm_result.reason}",
                source="rules+llm",
                requires_confirmation=True,
            )

        return llm_result


def _extract_json(raw_text: str) -> dict:
    if not raw_text:
        return {}
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _risk_value(value: str) -> int:
    return RISK_ORDER.get(value, 0)
