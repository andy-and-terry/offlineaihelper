"""Unit tests for OllamaClient using respx to mock httpx."""

from __future__ import annotations

import unittest

import httpx
import pytest
import respx

from offlineaihelper.ollama.client import (
    OllamaClient,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
)

BASE_URL = "http://localhost:11434"


@respx.mock
@pytest.mark.asyncio
async def test_generate_success():
    respx.post(f"{BASE_URL}/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "hello"})
    )
    client = OllamaClient(base_url=BASE_URL, timeout=5, retries=0)
    result = await client.generate(model="llama3.2:3b", prompt="Hi")
    assert result == "hello"
    await client.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_generate_timeout_retries_then_raises():
    respx.post(f"{BASE_URL}/api/generate").mock(side_effect=httpx.TimeoutException("timed out"))
    client = OllamaClient(base_url=BASE_URL, timeout=1, retries=2)
    with pytest.raises(OllamaUnavailableError):
        await client.generate(model="llama3.2:3b", prompt="Hi")
    await client.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_generate_404_raises_model_not_found():
    respx.post(f"{BASE_URL}/api/generate").mock(
        return_value=httpx.Response(404, text="model not found")
    )
    client = OllamaClient(base_url=BASE_URL, timeout=5, retries=0)
    with pytest.raises(OllamaModelNotFoundError):
        await client.generate(model="no-such-model", prompt="Hi")
    await client.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_list_models_returns_names():
    respx.get(f"{BASE_URL}/api/tags").mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [
                    {"name": "llama3.2:3b"},
                    {"name": "llama-guard3:1b"},
                ]
            },
        )
    )
    client = OllamaClient(base_url=BASE_URL, timeout=5, retries=0)
    names = await client.list_models()
    assert names == ["llama3.2:3b", "llama-guard3:1b"]
    await client.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_chat_success():
    respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Hello there!"}},
        )
    )
    client = OllamaClient(base_url=BASE_URL, timeout=5, retries=0)
    result = await client.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert result == "Hello there!"
    await client.aclose()


# ---------------------------------------------------------------------------
# unittest-compatible equivalents (for `python -m unittest discover`)
# ---------------------------------------------------------------------------

class TestOllamaClientUnittest(unittest.IsolatedAsyncioTestCase):
    """Mirror of the pytest tests for unittest discovery."""

    @respx.mock
    async def test_generate_success(self):
        respx.post(f"{BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "hello"})
        )
        client = OllamaClient(base_url=BASE_URL, timeout=5, retries=0)
        result = await client.generate(model="llama3.2:3b", prompt="Hi")
        self.assertEqual(result, "hello")
        await client.aclose()

    @respx.mock
    async def test_generate_timeout_retries_then_raises(self):
        respx.post(f"{BASE_URL}/api/generate").mock(
            side_effect=httpx.TimeoutException("timed out")
        )
        client = OllamaClient(base_url=BASE_URL, timeout=1, retries=2)
        with self.assertRaises(OllamaUnavailableError):
            await client.generate(model="llama3.2:3b", prompt="Hi")
        await client.aclose()

    @respx.mock
    async def test_generate_404_raises_model_not_found(self):
        respx.post(f"{BASE_URL}/api/generate").mock(
            return_value=httpx.Response(404, text="not found")
        )
        client = OllamaClient(base_url=BASE_URL, timeout=5, retries=0)
        with self.assertRaises(OllamaModelNotFoundError):
            await client.generate(model="no-such-model", prompt="Hi")
        await client.aclose()

    @respx.mock
    async def test_list_models_returns_names(self):
        respx.get(f"{BASE_URL}/api/tags").mock(
            return_value=httpx.Response(
                200,
                json={"models": [{"name": "llama3.2:3b"}, {"name": "llama-guard3:1b"}]},
            )
        )
        client = OllamaClient(base_url=BASE_URL, timeout=5, retries=0)
        names = await client.list_models()
        self.assertEqual(names, ["llama3.2:3b", "llama-guard3:1b"])
        await client.aclose()

    @respx.mock
    async def test_chat_success(self):
        respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "Hello there!"}},
            )
        )
        client = OllamaClient(base_url=BASE_URL, timeout=5, retries=0)
        result = await client.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": "Hi"}],
        )
        self.assertEqual(result, "Hello there!")
        await client.aclose()
