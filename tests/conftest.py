"""Shared pytest fixtures for the offlineaihelper test suite."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from offlineaihelper.moderation.audit import DecisionCode, ModerationEvent
from offlineaihelper.moderation.llm_moderator import LLMModeratorResult
from offlineaihelper.ollama.client import OllamaClient


@pytest.fixture
def sample_policy_config() -> dict:
    return {
        "strict_mode": True,
        "pre_check": {
            "enabled": True,
            "rule_categories": ["violence", "self_harm", "explicit", "pii", "injection"],
        },
        "llm_check": {
            "enabled": True,
            "model_alias": "moderator",
            "temperature": 0.0,
            "max_tokens": 64,
            "prompt_template": "Analyze: {text}",
        },
        "post_check": {
            "enabled": True,
            "rule_categories": ["violence", "self_harm", "explicit", "pii", "injection"],
        },
        "decision_codes": {
            "ALLOW": "Passed",
            "BLOCK_DETERMINISTIC": "Deterministic block",
            "BLOCK_LLM": "LLM block",
            "BLOCK_POST": "Post block",
            "ERROR_FAIL_CLOSED": "Error fail-closed",
        },
        "audit": {
            "log_level": "debug",
            "redact_content": True,
        },
    }


@pytest.fixture
def sample_models_config() -> dict:
    return {
        "ollama_base_url": "${OLLAMA_BASE_URL}",
        "assistant": {
            "alias": "assistant",
            "ollama_model": "llama3.2:3b",
        },
        "moderator": {
            "alias": "moderator",
            "ollama_model": "llama-guard3:1b",
        },
    }


@pytest.fixture
def mock_ollama_client() -> OllamaClient:
    """Return a mock OllamaClient with safe/unsafe/timeout helpers."""
    client = MagicMock(spec=OllamaClient)
    client.generate = AsyncMock(return_value="This is a helpful response.")
    client.chat = AsyncMock(return_value="This is a helpful chat response.")
    client.list_models = AsyncMock(return_value=["llama3.2:3b", "llama-guard3:1b"])
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def tmp_config_dir(tmp_path: Path, sample_policy_config: dict, sample_models_config: dict) -> Path:
    """Write policy and models config files into a temp directory and return it."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    policy_path = config_dir / "moderation.policy.json"
    policy_path.write_text(json.dumps(sample_policy_config), encoding="utf-8")

    models_path = config_dir / "models.json"
    models_path.write_text(json.dumps(sample_models_config), encoding="utf-8")

    return config_dir
