# offlineaihelper

An offline AI assistant with a strict built-in moderation pipeline, powered by [Ollama](https://ollama.com).

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
│  Ollama  (llama3.2:3b assistant + llama-guard3:1b)   │
└──────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com) installed and running

## Installation

```powershell
# Windows (PowerShell)
./install.ps1
```

Or manually:
```bash
pip install -e ".[dev]"
cd node && npm install
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

## Usage

### Start the Python API server

```bash
offlineaihelper serve
# or
python -m offlineaihelper.server
```

### Node.js CLI

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

### Python CLI (direct, no server needed)

```bash
offlineaihelper ask --prompt "Hello!"
offlineaihelper moderate --text "Test text"
offlineaihelper check-models
```

## Moderation Pipeline

Every request passes through three stages:

1. **Pre-check** — fast deterministic regex rules (violence, self-harm, PII, injection patterns)
2. **LLM check** — `llama-guard3:1b` (~400 MB) evaluates the prompt via Ollama
3. **Post-check** — same deterministic rules applied to the generated response

Decision codes: `ALLOW` · `BLOCK_DETERMINISTIC` · `BLOCK_LLM` · `BLOCK_POST` · `ERROR_FAIL_CLOSED`

In `strict_mode=true` (default), any moderation error blocks the request.

## Models

| Alias | Model | Size | Purpose |
|---|---|---|---|
| assistant | `llama3.2:3b` | ~2 GB | Main assistant |
| moderator | `llama-guard3:1b` | ~400 MB | Content safety |

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
