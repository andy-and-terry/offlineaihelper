"""Command-line interface for offlineaihelper."""

from __future__ import annotations

import asyncio
import sys

import click

from offlineaihelper.app import create_app
from offlineaihelper.ollama.client import OllamaUnavailableError


@click.group()
@click.version_option(package_name="offlineaihelper")
def main() -> None:
    """Offline AI assistant with built-in moderation pipeline."""


@main.command("ask")
@click.option("--prompt", required=True, help="The user prompt to send to the assistant.")
@click.option("--env", default=".env", show_default=True, help="Path to .env file.")
def ask(prompt: str, env: str) -> None:
    """Send PROMPT to the assistant and print the response."""
    try:
        app = create_app(env_path=env)
        result = asyncio.run(app.handle_request(prompt))
    except OllamaUnavailableError as exc:
        click.secho(f"Error: Ollama is not reachable — {exc}", fg="red", err=True)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        click.secho(f"Unexpected error: {exc}", fg="red", err=True)
        sys.exit(1)

    if result.allowed:
        click.echo(result.response)
    else:
        click.secho(
            f"[BLOCKED] {result.decision_code.value}: {result.reason}",
            fg="yellow",
            err=True,
        )
        sys.exit(2)


@main.command("check-models")
@click.option("--env", default=".env", show_default=True, help="Path to .env file.")
def check_models(env: str) -> None:
    """List available Ollama models and verify configured models are present."""
    from dotenv import load_dotenv
    import os
    import json
    from pathlib import Path
    from offlineaihelper.ollama.client import OllamaClient

    load_dotenv(dotenv_path=env, override=False)

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    timeout = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "30"))
    models_config_path = os.environ.get("MODELS_CONFIG_PATH", "config/models.json")

    async def _run() -> None:
        async with OllamaClient(base_url=base_url, timeout=timeout) as client:
            try:
                available = await client.list_models()
            except OllamaUnavailableError as exc:
                click.secho(f"Error: Ollama is not reachable — {exc}", fg="red", err=True)
                sys.exit(1)

            click.secho("\nAvailable models on Ollama server:", bold=True)
            for m in available:
                click.echo(f"  • {m}")

            cfg = json.loads(Path(models_config_path).read_text(encoding="utf-8"))
            configured = {
                cfg["assistant"]["alias"]: cfg["assistant"]["ollama_model"],
                cfg["moderator"]["alias"]: cfg["moderator"]["ollama_model"],
            }

            click.secho("\nConfigured model status:", bold=True)
            for alias, model in configured.items():
                present = any(m == model or m.startswith(model) for m in available)
                status = click.style("✓ present", fg="green") if present else click.style("✗ missing", fg="red")
                click.echo(f"  [{alias}] {model}  {status}")

    asyncio.run(_run())


@main.command("moderate")
@click.option("--text", required=True, help="Text to run through the moderation pipeline.")
@click.option("--env", default=".env", show_default=True, help="Path to .env file.")
def moderate(text: str, env: str) -> None:
    """Run TEXT through the moderation pipeline and print the decision."""
    from dotenv import load_dotenv
    import os
    import json
    from pathlib import Path
    from offlineaihelper.ollama.client import OllamaClient
    from offlineaihelper.moderation.policy_engine import ModerationPolicyEngine

    load_dotenv(dotenv_path=env, override=False)

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    timeout = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "30"))
    models_config_path = os.environ.get("MODELS_CONFIG_PATH", "config/models.json")

    async def _run() -> None:
        async with OllamaClient(base_url=base_url, timeout=timeout) as client:
            cfg = json.loads(Path(models_config_path).read_text(encoding="utf-8"))
            moderator_model = cfg["moderator"]["ollama_model"]
            engine = ModerationPolicyEngine(ollama_client=client, moderator_model=moderator_model)
            decision = await engine.evaluate(text, stage="pre")

        color = "green" if decision.allowed else "yellow"
        click.secho(
            f"\nDecision : {decision.decision_code.value}",
            fg=color,
            bold=True,
        )
        click.echo(f"Allowed  : {decision.allowed}")
        click.echo(f"Reason   : {decision.reason}")

    try:
        asyncio.run(_run())
    except OllamaUnavailableError as exc:
        click.secho(f"Error: Ollama is not reachable — {exc}", fg="red", err=True)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        click.secho(f"Unexpected error: {exc}", fg="red", err=True)
        sys.exit(1)


@main.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host.")
@click.option("--port", default=11435, show_default=True, type=int, help="Bind port.")
@click.option("--env", default=".env", show_default=True, help="Path to .env file.")
def serve(host: str, port: int, env: str) -> None:
    """Start the FastAPI HTTP server (used by the Node.js CLI layer)."""
    from dotenv import load_dotenv
    import os
    load_dotenv(dotenv_path=env, override=False)
    os.environ.setdefault("API_HOST", host)
    os.environ.setdefault("API_PORT", str(port))
    try:
        from offlineaihelper.server import run_server
        run_server()
    except ImportError:
        click.secho(
            "Error: fastapi and uvicorn are required for the server. "
            "Install with: pip install 'offlineaihelper[server]'",
            fg="red",
            err=True,
        )
        sys.exit(1)
