"""Synchronous Ollama HTTP client using only stdlib (no httpx dependency)."""
import json
import shutil
import urllib.error
import urllib.request


class SyncOllamaUnavailableError(RuntimeError):
    """Raised when the Ollama server is not reachable (sync client)."""


class SyncOllamaClient:
    """Synchronous Ollama client backed by :mod:`urllib`.

    Use :class:`~offlineaihelper.ollama.client.OllamaClient` for the async
    httpx-based version.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def ensure_available(self) -> None:
        """Raise :exc:`SyncOllamaUnavailableError` if Ollama is not reachable."""
        if shutil.which("ollama") is None:
            raise SyncOllamaUnavailableError(
                "Ollama CLI was not found. Run install.ps1 first, then setup-models.ps1 to pull models."
            )
        if not self.health_check():
            raise SyncOllamaUnavailableError(
                "Ollama is installed but not reachable on http://127.0.0.1:11434. Start Ollama and retry."
            )

    def health_check(self) -> bool:
        """Return ``True`` if the Ollama server responds on ``/api/tags``."""
        request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return 200 <= response.status < 300
        except OSError:
            return False

    def generate(self, model: str, prompt: str) -> str:
        """Call ``/api/generate`` and return the response text."""
        payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
                return body.get("response", "")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Ollama generate failed with status {exc.code}") from exc
        except OSError as exc:
            raise RuntimeError("Failed to reach Ollama API") from exc
