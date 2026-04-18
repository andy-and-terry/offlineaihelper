# Offline AI Helper

Offline AI Helper uses local Ollama models only (no embedded model weights). It includes a moderation pipeline with a small default moderation model (~400MB class, configurable).

## Default model routing

- Moderation model: `gemma3:1b` (override with `OAH_MODERATION_MODEL`)
- Chat: `gemma3:1b`
- Coding: `qwen2.5-coder:7b`
- Q&A: `qwen2.5:7b`
- Image/SVG: `qwen2.5-coder:7b`
- Low-end fallback: `qwen2.5:3b`

All model defaults live in `config/models.json` and can be overridden with environment variables.

## Safety / moderation behavior

The app applies moderation for every request category (chat/coding/Q&A/image):

1. **Pre-check** user input
2. **Post-check** assistant output before returning it
3. **Deterministic rules layer** (regex/heuristics) before LLM moderation
4. **Policy decision layer** with actions: `allow`, `block`, `allow_with_warning`

Moderation categories:

- self-harm
- violence
- sexual content
- hate/harassment
- illicit behavior
- malware/hacking
- privacy/PII
- safe

For blocked requests, the app returns a refusal plus safe alternative direction. For risky/borderline output, it returns a safer completion with warning. For command/script contexts, destructive operations trigger confirmation requirements.

## Windows setup

1. Install Ollama:

```powershell
.\install.ps1
```

2. Pull configured models:

```powershell
.\setup-models.ps1
```

3. Run app code (from Python):

```python
from offlineaihelper import OfflineAIHelper

helper = OfflineAIHelper()
helper.verify_environment()  # helpful error if Ollama missing/not running
response = helper.handle_request("Help me write safe Git Bash steps", task="coding")
print(response.text)
```

## Configuration

Environment variable overrides:

- `OAH_MODERATION_MODEL`
- `OAH_CHAT_MODEL`
- `OAH_CODING_MODEL`
- `OAH_QA_MODEL`
- `OAH_IMAGE_MODEL`
- `OAH_LOW_END_MODEL`
- `OAH_OLLAMA_BASE_URL`
- `OAH_POLICY_ACTION_<CATEGORY>` (for example `OAH_POLICY_ACTION_VIOLENCE=block`)

## Tests

Run tests with:

```bash
python -m unittest discover -s tests -v
```
