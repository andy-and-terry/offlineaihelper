import unittest

from offlineaihelper.moderation import DeterministicRules
from offlineaihelper.policy import ModerationPolicy


class TestDeterministicRules(unittest.TestCase):
    def test_self_harm_detection(self) -> None:
        result = DeterministicRules().classify("I am thinking about suicide")
        self.assertEqual(result.category, "self_harm")
        self.assertEqual(result.risk, "high")

    def test_destructive_command_requires_confirmation(self) -> None:
        result = DeterministicRules().classify("Use git reset --hard and git clean -fdx")
        self.assertEqual(result.category, "illicit_behavior")
        self.assertTrue(result.requires_confirmation)


class TestPolicy(unittest.TestCase):
    def test_blocked_category(self) -> None:
        policy = ModerationPolicy({"malware_hacking": "block", "safe": "allow"})
        result = policy.decide(
            moderation=type("M", (), {"category": "malware_hacking", "risk": "high", "requires_confirmation": False})(),
            is_command_context=False,
        )
        self.assertEqual(result.action, "block")

    def test_command_confirmation_warning(self) -> None:
        policy = ModerationPolicy({"safe": "allow"})
        result = policy.decide(
            moderation=type("M", (), {"category": "safe", "risk": "low", "requires_confirmation": True})(),
            is_command_context=True,
        )
        self.assertEqual(result.action, "allow_with_warning")


if __name__ == "__main__":
    unittest.main()
