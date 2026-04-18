# offlineaihelper

Offline AI assistant with a strict built-in moderation pipeline, powered by [Ollama](https://ollama.com). Runs entirely locally — no embedded model weights, no cloud calls.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Node.js CLI  (node/src/cli.js)                      │
│  Commands: ask · moderate · check-models · health    │
└───────────────────┬──────────────────────────────────┘
                    │ HTTP (localhost:11435)
┌───────────────────▼──────────────────────────────────┐
│  Python FastAPI Server  (src/offlineaihelper/server) │
│  POST /ask · POST /moderate · GET /models            │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Moderation Pipeline                            │ │
│  │  Pre-check → Generate → Post-check              │ │
│  │  Deterministic rules + LLM moderator            │ │
│  └────────────────────┬────────────────────────────┘ │
└───────────────────────┼──────────────────────────────┘
                        │ HTTP (localhost:11434)
┌───────────────────────▼──────────────────────────────┐
│  Ollama  (configurable models — see config/models.json) │
└──────────────────────────────────────────────────────┘
```

## Default model routing

| Alias / task | Model | Purpose |
|---|---|---|
| `assistant` | `llama3.2:3b` | Main assistant (FastAPI pipeline) |
| `moderator` | `llama-guard3:1b` (~400 MB) | Content safety (FastAPI pipeline) |
| `moderation` | `gemma3:1b` | Moderation (flat-module API) |
| `chat` | `gemma3:1b` | Chat task |
| `coding` | `qwen2.5-coder:7b` | Coding task |
| `qa` | `qwen2.5:7b` | Q&A task |
| `image` | `qwen2.5-coder:7b` | Image/SVG task |
| `low_end` | `qwen2.5:3b` | Low-resource fallback |

All model defaults live in `config/models.json` and can be overridden with environment variables (see Configuration below).

## Safety / moderation behavior

Every request passes through three stages:

1. **Pre-check** — fast deterministic regex rules (violence, self-harm, PII, injection patterns)
2. **LLM check** — small moderator model evaluates the prompt via Ollama
3. **Post-check** — same rules applied to the generated response

Decision codes: `ALLOW` · `BLOCK_DETERMINISTIC` · `BLOCK_LLM` · `BLOCK_POST` · `ERROR_FAIL_CLOSED`

Moderation categories: self-harm · violence · sexual content · hate/harassment · illicit behavior · malware/hacking · privacy/PII · safe

In `strict_mode=true` (default), any moderation error blocks the request.

## Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com) installed and running

## Installation

```powershell
# Windows (PowerShell) — installs Ollama if needed, Python package, Node deps, pulls models
.\install.ps1
```

Or manually:
```bash
pip install -e ".[dev]"
cd node && npm install
.\setup-models.ps1   # pull configured Ollama models
```

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `API_HOST` | `127.0.0.1` | Python API server bind host |
| `API_PORT` | `11435` | Python API server port |
| `MODERATION_STRICT_MODE` | `true` | Block on moderation errors |
| `OLLAMA_TIMEOUT_SECONDS` | `30` | Per-request Ollama timeout |
| `OAH_MODERATION_MODEL` | `gemma3:1b` | Override moderation model |
| `OAH_CHAT_MODEL` | `gemma3:1b` | Override chat model |
| `OAH_CODING_MODEL` | `qwen2.5-coder:7b` | Override coding model |
| `OAH_QA_MODEL` | `qwen2.5:7b` | Override Q&A model |
| `OAH_IMAGE_MODEL` | `qwen2.5-coder:7b` | Override image model |
| `OAH_LOW_END_MODEL` | `qwen2.5:3b` | Override low-end fallback |
| `OAH_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama URL (flat-module API) |
| `OAH_POLICY_ACTION_<CATEGORY>` | _(see models.json)_ | Override policy per category |

## Usage

### Option A — Node.js CLI + Python server

Start the Python API server first:
```bash
offlineaihelper serve
```

Then use the Node.js CLI:
```bash
# Ask the assistant
node node/src/cli.js ask --prompt "What is the capital of France?"

# Moderate a piece of text
node node/src/cli.js moderate --text "Hello, how are you?"

# Check which models are available
node node/src/cli.js check-models

# Verify the server is running
node node/src/cli.js health
```

### Option B — Python CLI (direct, no server needed)

```bash
offlineaihelper ask --prompt "Hello!"
offlineaihelper moderate --text "Test text"
offlineaihelper check-models
```

### Option C — Programmatic Python API

```python
from offlineaihelper import OfflineAIHelper

helper = OfflineAIHelper()
helper.verify_environment()  # helpful error if Ollama missing/not running
response = helper.handle_request("Help me write safe Git Bash steps", task="coding")
print(response.text)
```

## Development

### Run Python tests
```bash
python -m unittest discover -s tests -v
# or
pytest
```

### Run Node.js tests
```bash
cd node && npm test
```

### Run all tests
```bash
python -m unittest discover -s tests -v && cd node && npm test
```
