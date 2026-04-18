"""Policy engine that orchestrates the full moderation pipeline."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from offlineaihelper.moderation.audit import AuditLogger, DecisionCode, ModerationEvent
from offlineaihelper.moderation.deterministic_rules import DeterministicChecker
from offlineaihelper.moderation.llm_moderator import LLMModerator
from offlineaihelper.ollama.client import OllamaClient, OllamaUnavailableError

logger = logging.getLogger(__name__)

_DEFAULT_POLICY_PATH = Path("config/moderation.policy.json")


@dataclass
class PolicyDecision:
    """Outcome from :meth:`ModerationPolicyEngine.evaluate`."""

    allowed: bool
    decision_code: DecisionCode
    reason: str
    audit_event: ModerationEvent


class ModerationPolicyEngine:
    """Evaluates text against the configured moderation policy.

    The policy is read from the JSON file indicated by the
    ``MODERATION_POLICY_PATH`` environment variable (or *policy_path*).

    Parameters
    ----------
    policy_path:
        Explicit path to the policy JSON.  Falls back to the env var, then
        the bundled default.
    ollama_client:
        Shared :class:`~offlineaihelper.ollama.client.OllamaClient` instance.
    moderator_model:
        Ollama model name to use for LLM checks.
    """

    def __init__(
        self,
        policy_path: Path | str | None = None,
        ollama_client: OllamaClient | None = None,
        moderator_model: str = "llama-guard3:1b",
    ) -> None:
        _path = (
            Path(policy_path)
            if policy_path is not None
            else Path(os.environ.get("MODERATION_POLICY_PATH", str(_DEFAULT_POLICY_PATH)))
        )
        self._policy = json.loads(_path.read_text(encoding="utf-8"))
        self._strict_mode: bool = self._policy.get("strict_mode", True)

        audit_cfg = self._policy.get("audit", {})
        self._audit = AuditLogger(
            log_level=audit_cfg.get("log_level", "info"),
            redact_content=audit_cfg.get("redact_content", True),
        )

        pre_cfg = self._policy.get("pre_check", {})
        post_cfg = self._policy.get("post_check", {})
        llm_cfg = self._policy.get("llm_check", {})

        self._pre_enabled: bool = pre_cfg.get("enabled", True)
        self._post_enabled: bool = post_cfg.get("enabled", True)
        self._llm_enabled: bool = llm_cfg.get("enabled", True)

        self._pre_categories: list[str] | None = pre_cfg.get("rule_categories") or None
        self._post_categories: list[str] | None = post_cfg.get("rule_categories") or None

        self._pre_checker = DeterministicChecker(enabled_categories=self._pre_categories)
        self._post_checker = DeterministicChecker(enabled_categories=self._post_categories)

        self._llm_moderator: LLMModerator | None = None
        if self._llm_enabled and ollama_client is not None:
            self._llm_moderator = LLMModerator(
                ollama_client=ollama_client,
                model=moderator_model,
                temperature=float(llm_cfg.get("temperature", 0.0)),
                max_tokens=int(llm_cfg.get("max_tokens", 64)),
            )

    # ------------------------------------------------------------------

    async def evaluate(
        self, text: str, stage: Literal["pre", "post"] = "pre"
    ) -> PolicyDecision:
        """Run the moderation pipeline for *stage* on *text*.

        Pipeline
        --------
        1. Deterministic check (if enabled for the current stage).
        2. LLM check (if enabled and stage is ``"pre"``).
        3. Return ALLOW if all checks pass.

        On any unexpected exception in strict mode the decision is
        ``ERROR_FAIL_CLOSED``; in non-strict mode the request is allowed
        with a warning.
        """
        event = ModerationEvent(stage=stage)

        try:
            # ── 1. Deterministic ──────────────────────────────────────
            if stage == "pre" and self._pre_enabled:
                result = self._pre_checker.check(text)
                if not result.passed:
                    event.decision_code = DecisionCode.BLOCK_DETERMINISTIC
                    event.matched_category = result.matched_category
                    self._audit.log_event(event)
                    return PolicyDecision(
                        allowed=False,
                        decision_code=DecisionCode.BLOCK_DETERMINISTIC,
                        reason=f"Blocked by deterministic rule: {result.matched_category}",
                        audit_event=event,
                    )

            if stage == "post" and self._post_enabled:
                result = self._post_checker.check(text)
                if not result.passed:
                    event.decision_code = DecisionCode.BLOCK_POST
                    event.matched_category = result.matched_category
                    self._audit.log_event(event)
                    return PolicyDecision(
                        allowed=False,
                        decision_code=DecisionCode.BLOCK_POST,
                        reason=f"Response blocked by post-check: {result.matched_category}",
                        audit_event=event,
                    )

            # ── 2. LLM check (pre stage only) ─────────────────────────
            if stage == "pre" and self._llm_enabled and self._llm_moderator is not None:
                llm_result = await self._llm_moderator.moderate(text)
                event.model_used = self._llm_moderator._model
                if not llm_result.safe:
                    event.decision_code = DecisionCode.BLOCK_LLM
                    event.matched_category = llm_result.category
                    self._audit.log_event(event)
                    return PolicyDecision(
                        allowed=False,
                        decision_code=DecisionCode.BLOCK_LLM,
                        reason=f"Blocked by LLM moderation: {llm_result.reason}",
                        audit_event=event,
                    )

        except OllamaUnavailableError as exc:
            logger.error("OllamaUnavailableError during moderation: %s", exc)
            if self._strict_mode:
                event.decision_code = DecisionCode.ERROR_FAIL_CLOSED
                self._audit.log_event(event)
                return PolicyDecision(
                    allowed=False,
                    decision_code=DecisionCode.ERROR_FAIL_CLOSED,
                    reason="Moderation service unavailable (strict mode: fail-closed)",
                    audit_event=event,
                )
            else:
                logger.warning("Strict mode OFF — allowing request despite moderation error")

        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error during moderation: %s", exc)
            if self._strict_mode:
                event.decision_code = DecisionCode.ERROR_FAIL_CLOSED
                self._audit.log_event(event)
                return PolicyDecision(
                    allowed=False,
                    decision_code=DecisionCode.ERROR_FAIL_CLOSED,
                    reason=f"Unexpected moderation error (strict mode): {exc}",
                    audit_event=event,
                )
            else:
                logger.warning("Strict mode OFF — allowing request despite unexpected error")

        # ── 3. ALLOW ──────────────────────────────────────────────────
        event.decision_code = DecisionCode.ALLOW
        self._audit.log_event(event)
        return PolicyDecision(
            allowed=True,
            decision_code=DecisionCode.ALLOW,
            reason="Content passed all moderation checks",
            audit_event=event,
        )
