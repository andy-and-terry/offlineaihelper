import json
import shutil
import urllib.error
import urllib.request


class OllamaUnavailableError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def ensure_available(self) -> None:
        if shutil.which("ollama") is None:
            raise OllamaUnavailableError(
                "Ollama CLI was not found. Run install.ps1 first, then setup-models.ps1 to pull models."
            )
        if not self.health_check():
            raise OllamaUnavailableError(
                "Ollama is installed but not reachable on http://127.0.0.1:11434. Start Ollama and retry."
            )

    def health_check(self) -> bool:
        request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return 200 <= response.status < 300
        except OSError:
            return False

    def generate(self, model: str, prompt: str) -> str:
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
