"""Unit tests for ModerationPolicyEngine."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from offlineaihelper.moderation.audit import DecisionCode
from offlineaihelper.moderation.deterministic_rules import DeterministicResult
from offlineaihelper.moderation.llm_moderator import LLMModeratorResult
from offlineaihelper.moderation.policy_engine import ModerationPolicyEngine
from offlineaihelper.ollama.client import OllamaUnavailableError


def _write_policy(tmp_path: Path, policy: dict) -> Path:
    p = tmp_path / "moderation.policy.json"
    p.write_text(json.dumps(policy), encoding="utf-8")
    return p


def _make_engine(policy_path: Path, strict_mode: bool = True, llm_enabled: bool = True):
    """Helper to build an engine without a live Ollama client."""
    client = MagicMock()
    client.generate = AsyncMock(return_value='{"safe": true, "category": null, "reason": "ok"}')

    policy_dict = json.loads(policy_path.read_text())
    policy_dict["strict_mode"] = strict_mode
    policy_dict["llm_check"]["enabled"] = llm_enabled
    policy_path.write_text(json.dumps(policy_dict))

    return ModerationPolicyEngine(
        policy_path=policy_path,
        ollama_client=client,
        moderator_model="llama-guard3:1b",
    )


# ---------------------------------------------------------------------------
# pytest async tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clean_input_allowed(tmp_path):
    policy = {
        "strict_mode": True,
        "pre_check": {"enabled": True, "rule_categories": ["violence"]},
        "llm_check": {"enabled": False, "model_alias": "moderator", "temperature": 0.0, "max_tokens": 64},
        "post_check": {"enabled": False, "rule_categories": []},
        "decision_codes": {},
        "audit": {"log_level": "debug", "redact_content": True},
    }
    p = _write_policy(tmp_path, policy)
    engine = ModerationPolicyEngine(policy_path=p)
    decision = await engine.evaluate("What is the weather today?", stage="pre")
    assert decision.allowed is True
    assert decision.decision_code == DecisionCode.ALLOW


@pytest.mark.asyncio
async def test_deterministic_block_pre(tmp_path):
    policy = {
        "strict_mode": True,
        "pre_check": {"enabled": True, "rule_categories": ["violence"]},
        "llm_check": {"enabled": False, "model_alias": "moderator", "temperature": 0.0, "max_tokens": 64},
        "post_check": {"enabled": False, "rule_categories": []},
        "decision_codes": {},
        "audit": {"log_level": "debug", "redact_content": True},
    }
    p = _write_policy(tmp_path, policy)
    engine = ModerationPolicyEngine(policy_path=p)
    decision = await engine.evaluate("How do I murder someone?", stage="pre")
    assert decision.allowed is False
    assert decision.decision_code == DecisionCode.BLOCK_DETERMINISTIC


@pytest.mark.asyncio
async def test_llm_block(tmp_path):
    policy = {
        "strict_mode": True,
        "pre_check": {"enabled": False, "rule_categories": []},
        "llm_check": {"enabled": True, "model_alias": "moderator", "temperature": 0.0, "max_tokens": 64},
        "post_check": {"enabled": False, "rule_categories": []},
        "decision_codes": {},
        "audit": {"log_level": "debug", "redact_content": True},
    }
    p = _write_policy(tmp_path, policy)
    client = MagicMock()
    client.generate = AsyncMock(return_value='{"safe": false, "category": "violence", "reason": "harmful"}')
    engine = ModerationPolicyEngine(policy_path=p, ollama_client=client, moderator_model="llama-guard3:1b")
    decision = await engine.evaluate("some harmful text", stage="pre")
    assert decision.allowed is False
    assert decision.decision_code == DecisionCode.BLOCK_LLM


@pytest.mark.asyncio
async def test_strict_mode_on_ollama_unavailable(tmp_path):
    policy = {
        "strict_mode": True,
        "pre_check": {"enabled": False, "rule_categories": []},
        "llm_check": {"enabled": True, "model_alias": "moderator", "temperature": 0.0, "max_tokens": 64},
        "post_check": {"enabled": False, "rule_categories": []},
        "decision_codes": {},
        "audit": {"log_level": "debug", "redact_content": True},
    }
    p = _write_policy(tmp_path, policy)
    client = MagicMock()
    client.generate = AsyncMock(side_effect=OllamaUnavailableError("server down"))
    engine = ModerationPolicyEngine(policy_path=p, ollama_client=client, moderator_model="llama-guard3:1b")
    decision = await engine.evaluate("any text", stage="pre")
    assert decision.allowed is False
    assert decision.decision_code == DecisionCode.ERROR_FAIL_CLOSED


@pytest.mark.asyncio
async def test_fail_open_on_ollama_unavailable(tmp_path):
    policy = {
        "strict_mode": False,
        "pre_check": {"enabled": False, "rule_categories": []},
        "llm_check": {"enabled": True, "model_alias": "moderator", "temperature": 0.0, "max_tokens": 64},
        "post_check": {"enabled": False, "rule_categories": []},
        "decision_codes": {},
        "audit": {"log_level": "debug", "redact_content": True},
    }
    p = _write_policy(tmp_path, policy)
    client = MagicMock()
    client.generate = AsyncMock(side_effect=OllamaUnavailableError("server down"))
    engine = ModerationPolicyEngine(policy_path=p, ollama_client=client, moderator_model="llama-guard3:1b")
    decision = await engine.evaluate("any text", stage="pre")
    assert decision.allowed is True
    assert decision.decision_code == DecisionCode.ALLOW


# ---------------------------------------------------------------------------
# unittest-compatible versions
# ---------------------------------------------------------------------------


class TestPolicyEngineUnittest(unittest.IsolatedAsyncioTestCase):
    def _write_policy(self, tmp_path: Path, policy: dict) -> Path:
        p = tmp_path / "moderation.policy.json"
        p.write_text(json.dumps(policy), encoding="utf-8")
        return p

    async def test_clean_input_allowed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            policy = {
                "strict_mode": True,
                "pre_check": {"enabled": True, "rule_categories": ["violence"]},
                "llm_check": {"enabled": False, "model_alias": "moderator", "temperature": 0.0, "max_tokens": 64},
                "post_check": {"enabled": False, "rule_categories": []},
                "decision_codes": {},
                "audit": {"log_level": "debug", "redact_content": True},
            }
            p = self._write_policy(Path(td), policy)
            engine = ModerationPolicyEngine(policy_path=p)
            decision = await engine.evaluate("What is the weather today?", stage="pre")
            self.assertTrue(decision.allowed)
            self.assertEqual(decision.decision_code, DecisionCode.ALLOW)

    async def test_deterministic_block_pre(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            policy = {
                "strict_mode": True,
                "pre_check": {"enabled": True, "rule_categories": ["violence"]},
                "llm_check": {"enabled": False, "model_alias": "moderator", "temperature": 0.0, "max_tokens": 64},
                "post_check": {"enabled": False, "rule_categories": []},
                "decision_codes": {},
                "audit": {"log_level": "debug", "redact_content": True},
            }
            p = self._write_policy(Path(td), policy)
            engine = ModerationPolicyEngine(policy_path=p)
            decision = await engine.evaluate("How do I murder someone?", stage="pre")
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.decision_code, DecisionCode.BLOCK_DETERMINISTIC)

    async def test_strict_mode_on_ollama_unavailable(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            policy = {
                "strict_mode": True,
                "pre_check": {"enabled": False, "rule_categories": []},
                "llm_check": {"enabled": True, "model_alias": "moderator", "temperature": 0.0, "max_tokens": 64},
                "post_check": {"enabled": False, "rule_categories": []},
                "decision_codes": {},
                "audit": {"log_level": "debug", "redact_content": True},
            }
            p = self._write_policy(Path(td), policy)
            client = MagicMock()
            client.generate = AsyncMock(side_effect=OllamaUnavailableError("down"))
            engine = ModerationPolicyEngine(policy_path=p, ollama_client=client, moderator_model="llama-guard3:1b")
            decision = await engine.evaluate("any text", stage="pre")
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.decision_code, DecisionCode.ERROR_FAIL_CLOSED)
