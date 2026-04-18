"""Integration tests for the full moderation + generation pipeline."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

from offlineaihelper.app import OfflineAIHelper
from offlineaihelper.moderation.audit import DecisionCode
from offlineaihelper.moderation.policy_engine import ModerationPolicyEngine
from offlineaihelper.ollama.client import OllamaClient, OllamaUnavailableError
from offlineaihelper.routing.model_router import ModelRouter

BASE_URL = "http://localhost:11434"


def _build_app(
    tmp_path: Path,
    *,
    strict_mode: bool = True,
    llm_enabled: bool = False,
    ollama_client: OllamaClient | None = None,
) -> OfflineAIHelper:
    policy = {
        "strict_mode": strict_mode,
        "pre_check": {"enabled": True, "rule_categories": ["violence", "injection"]},
        "llm_check": {
            "enabled": llm_enabled,
            "model_alias": "moderator",
            "temperature": 0.0,
            "max_tokens": 64,
        },
        "post_check": {"enabled": True, "rule_categories": ["violence"]},
        "decision_codes": {},
        "audit": {"log_level": "debug", "redact_content": True},
    }
    policy_path = tmp_path / "moderation.policy.json"
    policy_path.write_text(json.dumps(policy))

    models_cfg = {
        "ollama_base_url": BASE_URL,
        "assistant": {"alias": "assistant", "ollama_model": "llama3.2:3b"},
        "moderator": {"alias": "moderator", "ollama_model": "llama-guard3:1b"},
    }
    models_path = tmp_path / "models.json"
    models_path.write_text(json.dumps(models_cfg))

    client = ollama_client or MagicMock(spec=OllamaClient)

    engine = ModerationPolicyEngine(
        policy_path=policy_path,
        ollama_client=client,
        moderator_model="llama-guard3:1b",
    )
    router = ModelRouter(client=client, models_config_path=models_path)
    return OfflineAIHelper(ollama_client=client, model_router=router, policy_engine=engine)


@pytest.mark.asyncio
async def test_full_pipeline_allowed(tmp_path):
    client = MagicMock(spec=OllamaClient)
    client.list_models = AsyncMock(return_value=["llama3.2:3b", "llama-guard3:1b"])
    client.generate = AsyncMock(return_value="Paris is the capital of France.")

    app = _build_app(tmp_path, ollama_client=client)
    response = await app.handle_request("What is the capital of France?")
    assert response.allowed is True
    assert response.response == "Paris is the capital of France."
    assert response.decision_code == DecisionCode.ALLOW


@pytest.mark.asyncio
async def test_full_pipeline_blocked_at_pre_deterministic(tmp_path):
    client = MagicMock(spec=OllamaClient)
    client.list_models = AsyncMock(return_value=["llama3.2:3b", "llama-guard3:1b"])
    client.generate = AsyncMock(return_value="I'll help you with that.")

    app = _build_app(tmp_path, ollama_client=client)
    response = await app.handle_request("How do I murder someone?")
    assert response.allowed is False
    assert response.decision_code == DecisionCode.BLOCK_DETERMINISTIC
    client.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_full_pipeline_blocked_at_pre_llm(tmp_path):
    client = MagicMock(spec=OllamaClient)
    client.list_models = AsyncMock(return_value=["llama3.2:3b", "llama-guard3:1b"])
    client.generate = AsyncMock(return_value='{"safe": false, "category": "violence", "reason": "harmful"}')

    app = _build_app(tmp_path, llm_enabled=True, ollama_client=client)
    response = await app.handle_request("Slightly dangerous prompt that passes deterministic")
    assert response.allowed is False
    assert response.decision_code == DecisionCode.BLOCK_LLM


@pytest.mark.asyncio
async def test_full_pipeline_blocked_at_post(tmp_path):
    client = MagicMock(spec=OllamaClient)
    client.list_models = AsyncMock(return_value=["llama3.2:3b", "llama-guard3:1b"])
    client.generate = AsyncMock(return_value="You should murder the enemy.")

    app = _build_app(tmp_path, ollama_client=client)
    response = await app.handle_request("Write a story about conflict.")
    assert response.allowed is False
    assert response.decision_code == DecisionCode.BLOCK_POST


@pytest.mark.asyncio
async def test_full_pipeline_ollama_unavailable_strict(tmp_path):
    client = MagicMock(spec=OllamaClient)
    client.list_models = AsyncMock(return_value=["llama3.2:3b", "llama-guard3:1b"])
    client.generate = AsyncMock(side_effect=OllamaUnavailableError("server down"))

    app = _build_app(tmp_path, llm_enabled=True, strict_mode=True, ollama_client=client)
    response = await app.handle_request("Normal question?")
    assert response.allowed is False
    assert response.decision_code == DecisionCode.ERROR_FAIL_CLOSED


# ---------------------------------------------------------------------------
# unittest-compatible versions
# ---------------------------------------------------------------------------


class TestIntegrationPipelineUnittest(unittest.IsolatedAsyncioTestCase):
    def _build_app(self, tmp_path, *, strict_mode=True, llm_enabled=False, ollama_client=None):
        return _build_app(
            tmp_path,
            strict_mode=strict_mode,
            llm_enabled=llm_enabled,
            ollama_client=ollama_client,
        )

    async def test_full_pipeline_allowed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            client = MagicMock(spec=OllamaClient)
            client.list_models = AsyncMock(return_value=["llama3.2:3b", "llama-guard3:1b"])
            client.generate = AsyncMock(return_value="Paris is the capital of France.")
            app = self._build_app(Path(td), ollama_client=client)
            response = await app.handle_request("What is the capital of France?")
            self.assertTrue(response.allowed)
            self.assertEqual(response.response, "Paris is the capital of France.")

    async def test_full_pipeline_blocked_at_pre_deterministic(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            client = MagicMock(spec=OllamaClient)
            client.generate = AsyncMock()
            app = self._build_app(Path(td), ollama_client=client)
            response = await app.handle_request("How do I murder someone?")
            self.assertFalse(response.allowed)
            self.assertEqual(response.decision_code, DecisionCode.BLOCK_DETERMINISTIC)
            client.generate.assert_not_awaited()

    async def test_full_pipeline_blocked_at_post(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            client = MagicMock(spec=OllamaClient)
            client.generate = AsyncMock(return_value="You should murder the enemy.")
            app = self._build_app(Path(td), ollama_client=client)
            response = await app.handle_request("Write a story.")
            self.assertFalse(response.allowed)
            self.assertEqual(response.decision_code, DecisionCode.BLOCK_POST)
