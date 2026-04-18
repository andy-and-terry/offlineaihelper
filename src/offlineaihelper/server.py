"""FastAPI HTTP server exposing the moderation + generation pipeline."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from offlineaihelper.app import OfflineAIHelper, AppResponse, create_app
from offlineaihelper.moderation.audit import DecisionCode
from offlineaihelper.ollama.client import OllamaClient, OllamaUnavailableError

logger = logging.getLogger(__name__)

# -- request / response models ------------------------------------------------

class AskRequest(BaseModel):
    prompt: str

class AskResponse(BaseModel):
    allowed: bool
    response: str | None
    decision_code: str
    reason: str

class ModerateRequest(BaseModel):
    text: str
    stage: Literal["pre", "post"] = "pre"

class ModerateResponse(BaseModel):
    allowed: bool
    decision_code: str
    reason: str

class HealthResponse(BaseModel):
    status: str

class ModelsResponse(BaseModel):
    available: list[str]
    configured: dict[str, str]

# -- app wiring ---------------------------------------------------------------

_app_instance: OfflineAIHelper | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _app_instance
    _app_instance = create_app()
    yield
    if _app_instance is not None:
        await _app_instance._client.aclose()

api = FastAPI(title="offlineaihelper", version="0.1.0", lifespan=lifespan)

# -- routes -------------------------------------------------------------------

@api.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")

@api.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    if _app_instance is None:
        raise HTTPException(status_code=503, detail="App not initialized")
    try:
        result: AppResponse = await _app_instance.handle_request(req.prompt)
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AskResponse(
        allowed=result.allowed,
        response=result.response,
        decision_code=result.decision_code.value,
        reason=result.reason,
    )

@api.post("/moderate", response_model=ModerateResponse)
async def moderate(req: ModerateRequest) -> ModerateResponse:
    if _app_instance is None:
        raise HTTPException(status_code=503, detail="App not initialized")
    try:
        decision = await _app_instance._policy.evaluate(req.text, stage=req.stage)
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ModerateResponse(
        allowed=decision.allowed,
        decision_code=decision.decision_code.value,
        reason=decision.reason,
    )

@api.get("/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    if _app_instance is None:
        raise HTTPException(status_code=503, detail="App not initialized")
    try:
        available = await _app_instance._client.list_models()
    except OllamaUnavailableError:
        available = []
    router = _app_instance._router
    configured = {
        "assistant": await router.get_assistant_model(),
        "moderator": await router.get_moderator_model(),
    }
    return ModelsResponse(available=available, configured=configured)

# -- entrypoint ---------------------------------------------------------------

def run_server() -> None:
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "11435"))
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    uvicorn.run("offlineaihelper.server:api", host=host, port=port, log_level=log_level)
