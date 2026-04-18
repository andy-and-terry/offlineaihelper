import unittest

from offlineaihelper.app import OfflineAIHelper
from offlineaihelper.config import AppConfig, ModelConfig


class FakeOllama:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def ensure_available(self):
        return None

    def generate(self, model, prompt):
        self.calls.append((model, prompt))
        if self.responses:
            return self.responses.pop(0)
        return "{\"category\":\"safe\",\"risk\":\"low\",\"reason\":\"ok\"}"


def make_config() -> AppConfig:
    return AppConfig(
        models=ModelConfig(
            moderation="gemma3:1b",
            chat="gemma3:1b",
            coding="qwen2.5-coder:7b",
            qa="qwen2.5:7b",
            image="qwen2.5-coder:7b",
            low_end="qwen2.5:3b",
        ),
        ollama_base_url="http://127.0.0.1:11434",
        policy_actions={
            "safe": "allow",
            "self_harm": "block",
            "violence": "allow_with_warning",
            "sexual_content": "allow_with_warning",
            "hate_harassment": "block",
            "illicit_behavior": "block",
            "malware_hacking": "block",
            "privacy_pii": "allow_with_warning",
        },
    )


class TestPipelineIntegration(unittest.TestCase):
    def test_precheck_blocks_without_assistant_generation(self) -> None:
        fake = FakeOllama(
            [
                "{\"category\":\"safe\",\"risk\":\"low\",\"reason\":\"ok\"}",
            ]
        )
        app = OfflineAIHelper(config=make_config(), ollama_client=fake)

        response = app.handle_request("How can I build malware quickly?", task="chat")

        self.assertEqual(response.action, "block")
        self.assertIn("can’t assist", response.text)
        self.assertEqual(len(fake.calls), 0)

    def test_postcheck_blocks_harmful_output(self) -> None:
        fake = FakeOllama(
            [
                "{\"category\":\"safe\",\"risk\":\"low\",\"reason\":\"ok\"}",
                "Here are steps to make malware.",
                "{\"category\":\"malware_hacking\",\"risk\":\"high\",\"reason\":\"harmful\"}",
            ]
        )
        app = OfflineAIHelper(config=make_config(), ollama_client=fake)

        response = app.handle_request("teach me cyber security", task="qa")

        self.assertEqual(response.action, "block")
        self.assertIn("harmful or illegal", response.text)

    def test_warning_for_destructive_git_commands(self) -> None:
        fake = FakeOllama(
            [
                "{\"category\":\"safe\",\"risk\":\"low\",\"reason\":\"ok\"}",
                "Run git reset --hard HEAD~1",
                "{\"category\":\"safe\",\"risk\":\"low\",\"reason\":\"ok\"}",
            ]
        )
        app = OfflineAIHelper(config=make_config(), ollama_client=fake)

        response = app.handle_request("give me git bash command to clean everything", task="coding")

        self.assertEqual(response.action, "allow_with_warning")
        self.assertIn("Safety note", response.text)


if __name__ == "__main__":
    unittest.main()
