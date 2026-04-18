"""Model routing — maps alias names to real Ollama model identifiers."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from offlineaihelper.moderation.llm_moderator import LLMModerator, LLMModeratorResult
from offlineaihelper.ollama.client import OllamaClient

logger = logging.getLogger(__name__)

_DEFAULT_MODELS_CONFIG_PATH = Path("config/models.json")
_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


class ModeratorUnavailableError(Exception):
    """Raised when the moderator model is required but not available."""


def _resolve_env_vars(value: str) -> str:
    """Replace ``${VAR}`` placeholders with the corresponding env-var values."""
    return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)


class ModelRouter:
    """Resolves model aliases and validates model availability via Ollama.

    Parameters
    ----------
    client:
        Shared :class:`~offlineaihelper.ollama.client.OllamaClient` instance.
    models_config_path:
        Path to ``models.json``.  Defaults to the ``MODELS_CONFIG_PATH``
        env var or the bundled ``config/models.json``.
    strict_mode:
        When ``True``, raises :class:`ModeratorUnavailableError` if the
        moderator model is absent from the server.
    """

    def __init__(
        self,
        client: OllamaClient,
        models_config_path: Path | str | None = None,
        strict_mode: bool = True,
    ) -> None:
        self._client = client
        self._strict_mode = strict_mode

        _path = (
            Path(models_config_path)
            if models_config_path is not None
            else Path(os.environ.get("MODELS_CONFIG_PATH", str(_DEFAULT_MODELS_CONFIG_PATH)))
        )
        raw = json.loads(_path.read_text(encoding="utf-8"))

        self._assistant_model: str = raw["assistant"]["ollama_model"]
        self._moderator_model: str = raw["moderator"]["ollama_model"]

    async def get_assistant_model(self) -> str:
        """Return the configured assistant model name."""
        return self._assistant_model

    async def get_moderator_model(self) -> str:
        """Return the configured moderator model name."""
        return self._moderator_model

    async def is_model_available(self, model: str) -> bool:
        """Return ``True`` if *model* is present in the Ollama server's model list."""
        try:
            available = await self._client.list_models()
            return any(m == model or m.startswith(model) for m in available)
        except Exception:  # noqa: BLE001
            return False

    async def route_moderation_call(
        self, text: str, llm_moderator: LLMModerator
    ) -> LLMModeratorResult:
        """Run *llm_moderator* on *text*, checking model availability first.

        Raises
        ------
        ModeratorUnavailableError
            If strict mode is on and the moderator model is not available.
        """
        moderator_model = await self.get_moderator_model()
        available = await self.is_model_available(moderator_model)
        if not available:
            if self._strict_mode:
                raise ModeratorUnavailableError(
                    f"Moderator model '{moderator_model}' is not available on the Ollama server"
                )
            logger.warning(
                "Moderator model '%s' unavailable; skipping LLM moderation (strict=False)",
                moderator_model,
            )
        return await llm_moderator.moderate(text)
