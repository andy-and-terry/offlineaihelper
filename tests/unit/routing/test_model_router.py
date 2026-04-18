"""Unit tests for ModelRouter."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from offlineaihelper.moderation.llm_moderator import LLMModerator, LLMModeratorResult
from offlineaihelper.ollama.client import OllamaClient
from offlineaihelper.routing.model_router import ModelRouter, ModeratorUnavailableError


def _make_router(tmp_path: Path, client: OllamaClient, strict_mode: bool = True) -> ModelRouter:
    cfg = {
        "ollama_base_url": "http://localhost:11434",
        "assistant": {"alias": "assistant", "ollama_model": "llama3.2:3b"},
        "moderator": {"alias": "moderator", "ollama_model": "llama-guard3:1b"},
    }
    p = tmp_path / "models.json"
    p.write_text(json.dumps(cfg))
    return ModelRouter(client=client, models_config_path=p, strict_mode=strict_mode)


@pytest.mark.asyncio
async def test_get_assistant_model_returns_configured(tmp_path):
    client = MagicMock(spec=OllamaClient)
    router = _make_router(tmp_path, client)
    assert await router.get_assistant_model() == "llama3.2:3b"


@pytest.mark.asyncio
async def test_get_moderator_model_returns_configured(tmp_path):
    client = MagicMock(spec=OllamaClient)
    router = _make_router(tmp_path, client)
    assert await router.get_moderator_model() == "llama-guard3:1b"


@pytest.mark.asyncio
async def test_route_moderation_call_moderator_unavailable_strict(tmp_path):
    client = MagicMock(spec=OllamaClient)
    client.list_models = AsyncMock(return_value=["llama3.2:3b"])  # moderator absent
    router = _make_router(tmp_path, client, strict_mode=True)

    moderator = MagicMock(spec=LLMModerator)
    moderator.moderate = AsyncMock()

    with pytest.raises(ModeratorUnavailableError):
        await router.route_moderation_call("test text", moderator)


@pytest.mark.asyncio
async def test_route_moderation_call_success(tmp_path):
    client = MagicMock(spec=OllamaClient)
    client.list_models = AsyncMock(return_value=["llama3.2:3b", "llama-guard3:1b"])
    router = _make_router(tmp_path, client, strict_mode=True)

    expected = LLMModeratorResult(safe=True, category=None, reason="ok", raw_response='{"safe":true}')
    moderator = MagicMock(spec=LLMModerator)
    moderator.moderate = AsyncMock(return_value=expected)

    result = await router.route_moderation_call("safe text", moderator)
    assert result.safe is True
    moderator.moderate.assert_awaited_once_with("safe text")


# ---------------------------------------------------------------------------
# unittest-compatible versions
# ---------------------------------------------------------------------------


class TestModelRouterUnittest(unittest.IsolatedAsyncioTestCase):
    def _make_router(self, tmp_path: Path, client, strict_mode: bool = True) -> ModelRouter:
        cfg = {
            "ollama_base_url": "http://localhost:11434",
            "assistant": {"alias": "assistant", "ollama_model": "llama3.2:3b"},
            "moderator": {"alias": "moderator", "ollama_model": "llama-guard3:1b"},
        }
        p = tmp_path / "models.json"
        p.write_text(json.dumps(cfg))
        return ModelRouter(client=client, models_config_path=p, strict_mode=strict_mode)

    async def test_get_assistant_model_returns_configured(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            client = MagicMock(spec=OllamaClient)
            router = self._make_router(Path(td), client)
            self.assertEqual(await router.get_assistant_model(), "llama3.2:3b")

    async def test_get_moderator_model_returns_configured(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            client = MagicMock(spec=OllamaClient)
            router = self._make_router(Path(td), client)
            self.assertEqual(await router.get_moderator_model(), "llama-guard3:1b")

    async def test_route_moderation_call_moderator_unavailable_strict(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            client = MagicMock(spec=OllamaClient)
            client.list_models = AsyncMock(return_value=["llama3.2:3b"])
            router = self._make_router(Path(td), client, strict_mode=True)
            moderator = MagicMock(spec=LLMModerator)
            moderator.moderate = AsyncMock()
            with self.assertRaises(ModeratorUnavailableError):
                await router.route_moderation_call("text", moderator)

    async def test_route_moderation_call_success(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            client = MagicMock(spec=OllamaClient)
            client.list_models = AsyncMock(return_value=["llama3.2:3b", "llama-guard3:1b"])
            router = self._make_router(Path(td), client)
            expected = LLMModeratorResult(safe=True, category=None, reason="ok", raw_response='{"safe":true}')
            moderator = MagicMock(spec=LLMModerator)
            moderator.moderate = AsyncMock(return_value=expected)
            result = await router.route_moderation_call("safe text", moderator)
            self.assertTrue(result.safe)
