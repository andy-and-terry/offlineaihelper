from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig, load_config
from .moderation import DeterministicRules, LLMModerator, ModerationPipeline
from .ollama import OllamaClient
from .policy import ModerationPolicy


@dataclass(frozen=True)
class ResponseEnvelope:
    text: str
    action: str
    category: str
    warning: str | None = None


class ModelRouter:
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


class OfflineAIHelper:
    def __init__(self, config: AppConfig | None = None, ollama_client: OllamaClient | None = None) -> None:
        self.config = config or load_config()
        self.ollama = ollama_client or OllamaClient(base_url=self.config.ollama_base_url)
        self.router = ModelRouter(self.config)
        self.pipeline = ModerationPipeline(
            rules=DeterministicRules(),
            llm_moderator=LLMModerator(self.ollama, self.config.models.moderation),
        )
        self.policy = ModerationPolicy(self.config.policy_actions)

    def verify_environment(self) -> None:
        self.ollama.ensure_available()

    def handle_request(self, user_input: str, task: str = "chat") -> ResponseEnvelope:
        command_context = _is_command_context(user_input, task)
        pre = self.pipeline.classify(user_input, stage="pre")
        pre_decision = self.policy.decide(pre, is_command_context=command_context)
        if pre_decision.action == "block":
            return ResponseEnvelope(text=_safe_refusal(pre.category), action="block", category=pre.category)

        model = self.router.model_for_task(task)
        assistant_output = self.ollama.generate(model, user_input)

        post = self.pipeline.classify(assistant_output, stage="post")
        post_decision = self.policy.decide(post, is_command_context=command_context)

        if post_decision.action == "block":
            return ResponseEnvelope(
                text=_safe_refusal(post.category),
                action="block",
                category=post.category,
            )

        if pre_decision.action == "allow_with_warning" or post_decision.action == "allow_with_warning":
            warning = pre_decision.message if pre_decision.action == "allow_with_warning" else post_decision.message
            return ResponseEnvelope(
                text=f"{assistant_output}\n\nSafety note: {warning}",
                action="allow_with_warning",
                category=post.category,
                warning=warning,
            )

        return ResponseEnvelope(text=assistant_output, action="allow", category=post.category)


def _safe_refusal(category: str) -> str:
    if category in {"self_harm", "violence"}:
        return "I can’t help with harm. I can help with safety planning or finding immediate support resources instead."
    if category in {"malware_hacking", "illicit_behavior"}:
        return "I can’t assist with harmful or illegal actions. I can provide defensive, legal alternatives instead."
    if category == "privacy_pii":
        return "I can’t help expose private information. I can explain secure handling and redaction best practices."
    return "I can’t help with that request, but I can help with a safer alternative."


def _is_command_context(user_input: str, task: str) -> bool:
    sample = f"{task} {user_input}".lower()
    markers = ("bash", "command", "script", "batch", "powershell", "terminal", "git")
    return any(token in sample for token in markers)
