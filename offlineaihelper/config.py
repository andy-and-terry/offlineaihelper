import json
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "models.json"


@dataclass(frozen=True)
class ModelConfig:
    moderation: str
    chat: str
    coding: str
    qa: str
    image: str
    low_end: str


@dataclass(frozen=True)
class AppConfig:
    models: ModelConfig
    ollama_base_url: str
    policy_actions: dict[str, str]


DEFAULT_POLICY_ACTIONS = {
    "safe": "allow",
    "self_harm": "block",
    "violence": "allow_with_warning",
    "sexual_content": "allow_with_warning",
    "hate_harassment": "block",
    "illicit_behavior": "block",
    "malware_hacking": "block",
    "privacy_pii": "allow_with_warning",
}


def _read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(config_path: str | None = None) -> AppConfig:
    raw = _read_config(Path(config_path) if config_path else DEFAULT_CONFIG_PATH)
    model_cfg = raw.get("models", {})

    models = ModelConfig(
        moderation=os.getenv("OAH_MODERATION_MODEL", model_cfg.get("moderation", "gemma3:1b")),
        chat=os.getenv("OAH_CHAT_MODEL", model_cfg.get("chat", "gemma3:1b")),
        coding=os.getenv("OAH_CODING_MODEL", model_cfg.get("coding", "qwen2.5-coder:7b")),
        qa=os.getenv("OAH_QA_MODEL", model_cfg.get("qa", "qwen2.5:7b")),
        image=os.getenv("OAH_IMAGE_MODEL", model_cfg.get("image", "qwen2.5-coder:7b")),
        low_end=os.getenv("OAH_LOW_END_MODEL", model_cfg.get("low_end", "qwen2.5:3b")),
    )

    policy_actions = {**DEFAULT_POLICY_ACTIONS, **raw.get("policy", {}).get("actions", {})}
    for key, value in os.environ.items():
        if key.startswith("OAH_POLICY_ACTION_"):
            category = key.removeprefix("OAH_POLICY_ACTION_").lower()
            policy_actions[category] = value

    return AppConfig(
        models=models,
        ollama_base_url=os.getenv("OAH_OLLAMA_BASE_URL", raw.get("ollama", {}).get("base_url", "http://127.0.0.1:11434")),
        policy_actions=policy_actions,
    )
