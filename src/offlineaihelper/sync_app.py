"""Synchronous high-level API: SyncOfflineAIHelper + ResponseEnvelope.

This module provides the task-routing, sync pipeline API originally in the
flat ``offlineaihelper`` module.  The async FastAPI-backed pipeline lives in
:mod:`offlineaihelper.app`.
"""
from __future__ import annotations

from dataclasses import dataclass

from offlineaihelper.config import AppConfig, load_config
from offlineaihelper.moderation import DeterministicRules, LLMModerator, ModerationPipeline
from offlineaihelper.policy import ModerationPolicy
from offlineaihelper.sync_ollama import SyncOllamaClient


@dataclass(frozen=True)
class ResponseEnvelope:
    """Result returned by :meth:`SyncOfflineAIHelper.handle_request`."""

    text: str
    action: str
    category: str
    warning: str | None = None


class _ModelRouter:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def model_for_task(self, task: str) -> str:
        task_l = (task or "chat").lower()
        if task_l == "coding":
            return self.config.models.coding
        if task_l in {"qa", "q&a", "help"}:
            return self.config.models.qa
        if task_l in {"image", "svg"}:
            return self.config.models.image
        if task_l == "low_end":
            return self.config.models.low_end
        return self.config.models.chat


class SyncOfflineAIHelper:
    """Synchronous offline AI assistant with a built-in moderation pipeline.

    Parameters
    ----------
    config:
        Application configuration.  Loaded from ``config/models.json`` and
        ``OAH_*`` env vars when ``None``.
    ollama_client:
        Optional pre-built :class:`~offlineaihelper.sync_ollama.SyncOllamaClient`.
        A default instance is created when ``None``.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        ollama_client: SyncOllamaClient | None = None,
    ) -> None:
        self.config = config or load_config()
        self.ollama = ollama_client or SyncOllamaClient(base_url=self.config.ollama_base_url)
        self._router = _ModelRouter(self.config)
        self._pipeline = ModerationPipeline(
            rules=DeterministicRules(),
            llm_moderator=LLMModerator(self.ollama, self.config.models.moderation),
        )
        self._policy = ModerationPolicy(self.config.policy_actions)

    def verify_environment(self) -> None:
        """Raise if Ollama is not reachable."""
        self.ollama.ensure_available()

    def handle_request(self, user_input: str, task: str = "chat") -> ResponseEnvelope:
        """Process *user_input* through pre-check → generate → post-check.

        Returns
        -------
        ResponseEnvelope
            Contains the (possibly blocked) outcome of the request.
        """
        command_context = _is_command_context(user_input, task)
        pre = self._pipeline.classify(user_input, stage="pre")
        pre_decision = self._policy.decide(pre, is_command_context=command_context)
        if pre_decision.action == "block":
            return ResponseEnvelope(
                text=_safe_refusal(pre.category), action="block", category=pre.category
            )

        model = self._router.model_for_task(task)
        assistant_output = self.ollama.generate(model, user_input)

        post = self._pipeline.classify(assistant_output, stage="post")
        post_decision = self._policy.decide(post, is_command_context=command_context)

        if post_decision.action == "block":
            return ResponseEnvelope(
                text=_safe_refusal(post.category),
                action="block",
                category=post.category,
            )

        if pre_decision.action == "allow_with_warning" or post_decision.action == "allow_with_warning":
            warning = (
                pre_decision.message
                if pre_decision.action == "allow_with_warning"
                else post_decision.message
            )
            return ResponseEnvelope(
                text=f"{assistant_output}\n\nSafety note: {warning}",
                action="allow_with_warning",
                category=post.category,
                warning=warning,
            )

        return ResponseEnvelope(text=assistant_output, action="allow", category=post.category)


def _safe_refusal(category: str) -> str:
    if category in {"self_harm", "violence"}:
        return (
            "I can't help with harm. If someone may be in immediate danger, contact local emergency services now. "
            "I can help with safety planning and finding crisis support resources."
        )
    if category in {"malware_hacking", "illicit_behavior"}:
        return "I can't assist with harmful or illegal actions. I can provide defensive, legal alternatives instead."
    if category == "privacy_pii":
        return "I can't help expose private information. I can explain secure handling and redaction best practices."
    return "I can't help with that request, but I can help with a safer alternative."


def _is_command_context(user_input: str, task: str) -> bool:
    sample = f"{task} {user_input}".lower()
    markers = ("bash", "command", "script", "batch", "powershell", "terminal", "git")
    return any(token in sample for token in markers)
