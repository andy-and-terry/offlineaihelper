"""Unit tests for the FastAPI server."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

from offlineaihelper.app import AppResponse, OfflineAIHelper
from offlineaihelper.moderation.audit import DecisionCode
from offlineaihelper.moderation.policy_engine import ModerationPolicyEngine, PolicyDecision
from offlineaihelper.moderation.audit import ModerationEvent
from offlineaihelper.ollama.client import OllamaUnavailableError


def _make_allow_decision() -> PolicyDecision:
    return PolicyDecision(
        allowed=True,
        decision_code=DecisionCode.ALLOW,
        reason="ok",
        audit_event=ModerationEvent(),
    )

def _make_block_decision(code: DecisionCode = DecisionCode.BLOCK_DETERMINISTIC) -> PolicyDecision:
    return PolicyDecision(
        allowed=False,
        decision_code=code,
        reason="blocked",
        audit_event=ModerationEvent(),
    )


@pytest.fixture
def mock_app():
    app = MagicMock(spec=OfflineAIHelper)
    app.handle_request = AsyncMock(
        return_value=AppResponse(
            allowed=True,
            response="Hello!",
            decision_code=DecisionCode.ALLOW,
            reason="ok",
        )
    )
    app._policy = MagicMock(spec=ModerationPolicyEngine)
    app._policy.evaluate = AsyncMock(return_value=_make_allow_decision())
    app._client = MagicMock()
    app._client.list_models = AsyncMock(return_value=["llama3.2:3b"])
    app._client.aclose = AsyncMock()
    app._router = MagicMock()
    app._router.get_assistant_model = AsyncMock(return_value="llama3.2:3b")
    app._router.get_moderator_model = AsyncMock(return_value="llama-guard3:1b")
    return app


@pytest.fixture
def api_with_mock(mock_app):
    import offlineaihelper.server as srv
    srv._app_instance = mock_app
    yield srv.api
    srv._app_instance = None


@pytest.mark.asyncio
async def test_health(api_with_mock):
    async with AsyncClient(transport=ASGITransport(app=api_with_mock), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_ask_allowed(api_with_mock, mock_app):
    async with AsyncClient(transport=ASGITransport(app=api_with_mock), base_url="http://test") as client:
        r = await client.post("/ask", json={"prompt": "Hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["allowed"] is True
    assert body["response"] == "Hello!"
    assert body["decision_code"] == "ALLOW"


@pytest.mark.asyncio
async def test_ask_blocked(api_with_mock, mock_app):
    mock_app.handle_request.return_value = AppResponse(
        allowed=False,
        response=None,
        decision_code=DecisionCode.BLOCK_DETERMINISTIC,
        reason="violence",
    )
    async with AsyncClient(transport=ASGITransport(app=api_with_mock), base_url="http://test") as client:
        r = await client.post("/ask", json={"prompt": "bad prompt"})
    assert r.status_code == 200
    body = r.json()
    assert body["allowed"] is False
    assert body["decision_code"] == "BLOCK_DETERMINISTIC"


@pytest.mark.asyncio
async def test_ask_ollama_unavailable(api_with_mock, mock_app):
    mock_app.handle_request.side_effect = OllamaUnavailableError("down")
    async with AsyncClient(transport=ASGITransport(app=api_with_mock), base_url="http://test") as client:
        r = await client.post("/ask", json={"prompt": "test"})
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_moderate_allowed(api_with_mock):
    async with AsyncClient(transport=ASGITransport(app=api_with_mock), base_url="http://test") as client:
        r = await client.post("/moderate", json={"text": "Hello", "stage": "pre"})
    assert r.status_code == 200
    assert r.json()["allowed"] is True


@pytest.mark.asyncio
async def test_models_endpoint(api_with_mock):
    async with AsyncClient(transport=ASGITransport(app=api_with_mock), base_url="http://test") as client:
        r = await client.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert "available" in body
    assert "configured" in body


# ---------------------------------------------------------------------------
# unittest-compatible versions
# ---------------------------------------------------------------------------
class TestServerUnittest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import offlineaihelper.server as srv
        self._srv = srv
        self._mock_app = MagicMock(spec=OfflineAIHelper)
        self._mock_app.handle_request = AsyncMock(
            return_value=AppResponse(
                allowed=True, response="ok", decision_code=DecisionCode.ALLOW, reason="ok"
            )
        )
        self._mock_app._policy = MagicMock(spec=ModerationPolicyEngine)
        self._mock_app._policy.evaluate = AsyncMock(return_value=_make_allow_decision())
        self._mock_app._client = MagicMock()
        self._mock_app._client.list_models = AsyncMock(return_value=["llama3.2:3b"])
        self._mock_app._client.aclose = AsyncMock()
        self._mock_app._router = MagicMock()
        self._mock_app._router.get_assistant_model = AsyncMock(return_value="llama3.2:3b")
        self._mock_app._router.get_moderator_model = AsyncMock(return_value="llama-guard3:1b")
        srv._app_instance = self._mock_app

    def tearDown(self):
        self._srv._app_instance = None

    async def test_health(self):
        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=self._srv.api), base_url="http://test") as c:
            r = await c.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    async def test_ask_allowed(self):
        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=self._srv.api), base_url="http://test") as c:
            r = await c.post("/ask", json={"prompt": "Hello"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["allowed"])
