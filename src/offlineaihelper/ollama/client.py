"""Async HTTP client for communicating with a local Ollama server."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OllamaUnavailableError(Exception):
    """Raised when the Ollama server is unreachable after all retries."""


class OllamaModelNotFoundError(Exception):
    """Raised when the requested model is not found on the Ollama server."""


class OllamaClient:
    """Async client for the Ollama REST API.

    Parameters
    ----------
    base_url:
        Base URL of the Ollama server, e.g. ``http://localhost:11434``.
    timeout:
        Per-request timeout in seconds.
    retries:
        How many additional attempts to make on timeout or 5xx responses.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST *payload* to *path* with retry logic.

        Retries on :class:`httpx.TimeoutException` and HTTP 5xx responses.
        Raises :class:`OllamaModelNotFoundError` on 404 and
        :class:`OllamaUnavailableError` when retries are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt > 0:
                backoff = 2 ** (attempt - 1)
                logger.debug("Ollama retry %d/%d, backing off %ss", attempt, self.retries, backoff)
                await asyncio.sleep(backoff)
            try:
                response = await self._client.post(path, json=payload)
                if response.status_code == 404:
                    raise OllamaModelNotFoundError(
                        f"Model not found at {path}: {response.text}"
                    )
                if response.status_code >= 500:
                    last_exc = httpx.HTTPStatusError(
                        f"Server error {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    logger.warning("Ollama 5xx on attempt %d: %s", attempt + 1, response.status_code)
                    continue
                response.raise_for_status()
                return response.json()
            except OllamaModelNotFoundError:
                raise
            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning("Ollama timeout on attempt %d", attempt + 1)
            except httpx.HTTPStatusError:
                raise
            except httpx.RequestError as exc:
                last_exc = exc
                logger.warning("Ollama request error on attempt %d: %s", attempt + 1, exc)

        raise OllamaUnavailableError(
            f"Ollama unreachable at {self.base_url} after {self.retries + 1} attempt(s)"
        ) from last_exc

    async def _get(self, path: str) -> dict[str, Any]:
        """GET *path* with a single attempt (no retry needed for tag listing)."""
        try:
            response = await self._client.get(path)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise OllamaUnavailableError(
                f"Ollama GET {path} timed out"
            ) from exc
        except httpx.RequestError as exc:
            raise OllamaUnavailableError(
                f"Ollama GET {path} failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a completion for *prompt* using *model*.

        Returns the text of the generated response.
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        data = await self._post("/api/generate", payload)
        return data["response"]

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat *messages* list to *model* and return the reply text.

        Each message dict must have ``role`` and ``content`` keys.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        data = await self._post("/api/chat", payload)
        return data["message"]["content"]

    async def list_models(self) -> list[str]:
        """Return a list of model names available on the Ollama server."""
        data = await self._get("/api/tags")
        return [m["name"] for m in data.get("models", [])]

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()
