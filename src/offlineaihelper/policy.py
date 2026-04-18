"""Synchronous policy decision layer."""
from __future__ import annotations

from dataclasses import dataclass

from offlineaihelper.moderation import ModerationResult


@dataclass(frozen=True)
class SyncPolicyDecision:
    """Outcome of a synchronous moderation policy evaluation."""

    action: str
    message: str


class ModerationPolicy:
    """Maps moderation categories to policy actions.

    Parameters
    ----------
    actions_by_category:
        Dict mapping category names to actions (``"allow"``,
        ``"block"``, ``"allow_with_warning"``).
    """

    def __init__(self, actions_by_category: dict[str, str]) -> None:
        self.actions_by_category = actions_by_category

    def decide(self, moderation: ModerationResult, is_command_context: bool) -> SyncPolicyDecision:
        """Return a :class:`SyncPolicyDecision` for *moderation*."""
        action = self.actions_by_category.get(moderation.category, "allow")

        if moderation.requires_confirmation and is_command_context:
            return SyncPolicyDecision(
                action="allow_with_warning",
                message="This operation may be destructive. Please confirm by typing 'I confirm' before proceeding.",
            )

        if action == "block":
            return SyncPolicyDecision(
                action="block",
                message=(
                    f"Request blocked due to {moderation.category}. "
                    "Provide a safer alternative focused on legal, non-harmful, privacy-preserving guidance."
                ),
            )

        if action == "allow_with_warning" or moderation.risk == "medium":
            return SyncPolicyDecision(
                action="allow_with_warning",
                message=(
                    f"Content flagged for {moderation.category}. "
                    "Continue with a safe completion and include a short caution."
                ),
            )

        return SyncPolicyDecision(action="allow", message="Allowed")
