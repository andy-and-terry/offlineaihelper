"""LLM-based content moderation using an Ollama model."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from offlineaihelper.ollama.client import OllamaClient, OllamaUnavailableError

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    "You are a content safety classifier. Analyze the following text and respond ONLY "
    'with a JSON object in this exact format: {{"safe": true/false, "category": "<category or null>", '
    '"reason": "<brief reason>"}}.\n\nText to analyze:\n{text}'
)


@dataclass
class LLMModeratorResult:
    """Result from the LLM moderation call."""

    safe: bool
    category: str | None
    reason: str | None
    raw_response: str


class LLMModerator:
    """Uses an Ollama LLM to evaluate whether a piece of text is safe.

    Parameters
    ----------
    ollama_client:
        Pre-configured :class:`~offlineaihelper.ollama.client.OllamaClient`.
    model:
        Ollama model name to use for moderation.
    temperature:
        Sampling temperature (``0.0`` for deterministic output).
    max_tokens:
        Maximum tokens the model may generate in its JSON reply.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> None:
        self._client = ollama_client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def moderate(self, text: str) -> LLMModeratorResult:
        """Ask the moderation LLM whether *text* is safe.

        Raises
        ------
        OllamaUnavailableError
            Propagated from the underlying client so that the policy engine
            can apply fail-closed logic.
        """
        prompt = _PROMPT_TEMPLATE.format(text=text)
        raw = await self._client.generate(
            model=self._model,
            prompt=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> LLMModeratorResult:
        """Parse the JSON response from the LLM.

        On any parse failure the content is treated as **unsafe** (strict
        interpretation), because a well-behaved safety model should always
        return valid JSON.
        """
        stripped = raw.strip()
        start = stripped.find("{")
        end = stripped.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("LLM moderator: no JSON object found in response, treating as unsafe")
            return LLMModeratorResult(safe=False, category="parse_error", reason="No JSON found", raw_response=raw)

        try:
            data = json.loads(stripped[start:end])
        except json.JSONDecodeError as exc:
            logger.warning("LLM moderator: JSON parse failed (%s), treating as unsafe", exc)
            return LLMModeratorResult(safe=False, category="parse_error", reason=str(exc), raw_response=raw)

        safe = bool(data.get("safe", False))
        category = data.get("category") or None
        reason = data.get("reason") or None
        return LLMModeratorResult(safe=safe, category=category, reason=reason, raw_response=raw)
