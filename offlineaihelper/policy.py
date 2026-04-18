from dataclasses import dataclass

from .moderation import ModerationResult


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    message: str


class ModerationPolicy:
    def __init__(self, actions_by_category: dict[str, str]) -> None:
        self.actions_by_category = actions_by_category

    def decide(self, moderation: ModerationResult, is_command_context: bool) -> PolicyDecision:
        action = self.actions_by_category.get(moderation.category, "allow")

        if moderation.requires_confirmation and is_command_context:
            return PolicyDecision(
                action="allow_with_warning",
                message="This operation may be destructive. Please confirm by typing 'I confirm' before proceeding.",
            )

        if action == "block":
            return PolicyDecision(
                action="block",
                message=(
                    f"Request blocked due to {moderation.category}. "
                    "Provide a safer alternative focused on legal, non-harmful, privacy-preserving guidance."
                ),
            )

        if action == "allow_with_warning" or moderation.risk == "medium":
            return PolicyDecision(
                action="allow_with_warning",
                message=(
                    f"Content flagged for {moderation.category}. Continue with a safe completion and include a short caution."
                ),
            )

        return PolicyDecision(action="allow", message="Allowed")
