"""Main application class and factory function."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from offlineaihelper.moderation.audit import DecisionCode
from offlineaihelper.moderation.policy_engine import ModerationPolicyEngine
from offlineaihelper.ollama.client import OllamaClient
from offlineaihelper.routing.model_router import ModelRouter

logger = logging.getLogger(__name__)


@dataclass
class AppResponse:
    """The result returned to the caller of :meth:`OfflineAIHelper.handle_request`."""

    allowed: bool
    response: str | None
    decision_code: DecisionCode
    reason: str


class OfflineAIHelper:
    """Full request pipeline: pre-check → generate → post-check.

    Parameters
    ----------
    ollama_client:
        Shared async HTTP client for Ollama.
    model_router:
        Resolves model aliases and checks availability.
    policy_engine:
        Runs the moderation pipeline.
    assistant_temperature:
        Sampling temperature for the assistant model.
    assistant_max_tokens:
        Token budget for the assistant model.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        model_router: ModelRouter,
        policy_engine: ModerationPolicyEngine,
        assistant_temperature: float = 0.7,
        assistant_max_tokens: int = 1024,
    ) -> None:
        self._client = ollama_client
        self._router = model_router
        self._policy = policy_engine
        self._temperature = assistant_temperature
        self._max_tokens = assistant_max_tokens

    async def handle_request(self, user_prompt: str) -> AppResponse:
        """Process *user_prompt* through the full moderation + generation pipeline.

        Returns
        -------
        AppResponse
            Contains the (possibly blocked) outcome of the request.
        """
        # ── 1. Pre-check ──────────────────────────────────────────────
        pre_decision = await self._policy.evaluate(user_prompt, stage="pre")
        if not pre_decision.allowed:
            return AppResponse(
                allowed=False,
                response=None,
                decision_code=pre_decision.decision_code,
                reason=pre_decision.reason,
            )

        # ── 2. Generate ───────────────────────────────────────────────
        assistant_model = await self._router.get_assistant_model()
        generated = await self._client.generate(
            model=assistant_model,
            prompt=user_prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        # ── 3. Post-check ─────────────────────────────────────────────
        post_decision = await self._policy.evaluate(generated, stage="post")
        if not post_decision.allowed:
            return AppResponse(
                allowed=False,
                response=None,
                decision_code=post_decision.decision_code,
                reason=post_decision.reason,
            )

        return AppResponse(
            allowed=True,
            response=generated,
            decision_code=DecisionCode.ALLOW,
            reason="Content passed all moderation checks",
        )


def create_app(env_path: str | Path = ".env") -> OfflineAIHelper:
    """Load environment variables from *env_path* and wire up the application.

    Returns
    -------
    OfflineAIHelper
        A fully configured application instance ready to handle requests.
    """
    load_dotenv(dotenv_path=env_path, override=False)

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    timeout = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "30"))
    retries = int(os.environ.get("OLLAMA_MODERATION_RETRIES", "2"))

    client = OllamaClient(base_url=base_url, timeout=timeout, retries=retries)

    strict_mode = os.environ.get("MODERATION_STRICT_MODE", "true").lower() not in ("false", "0", "no")
    router = ModelRouter(client=client, strict_mode=strict_mode)

    models_config_path = os.environ.get("MODELS_CONFIG_PATH", "config/models.json")
    import json

    models_cfg = json.loads(Path(models_config_path).read_text(encoding="utf-8"))
    moderator_model = models_cfg["moderator"]["ollama_model"]

    policy_engine = ModerationPolicyEngine(
        ollama_client=client,
        moderator_model=moderator_model,
    )

    return OfflineAIHelper(
        ollama_client=client,
        model_router=router,
        policy_engine=policy_engine,
    )
