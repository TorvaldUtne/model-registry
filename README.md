# mr — Model Registry

A single-file CLI for tracking, rating, and managing AI models across three local backends: **Ollama** (via Docker), **llama.cpp** (`.gguf` files), and **ComfyUI** (`.safetensors`, `.ckpt`, etc.). State is stored in a local SQLite database.

## Features

- Scan all three backends in one command and keep a unified registry
- Rate models 1–5 and assign statuses (`active`, `favorite`, `on_hold`, `blacklisted`, …)
- Pull models from HuggingFace or CivitAI with download tracking
- Enrich registry entries with HF download counts, likes, architecture, and CivitAI trigger words
- Blacklist models with a reason so you're warned before accidentally re-downloading them
- Cross-backend duplicate detection (same `hf_repo` on multiple backends)
- Audit log: every mutating operation is recorded in an `events` table

## Requirements

- Python 3.10+
- Docker (for the Ollama backend)
- pip packages: `click`, `rich`, `requests`, `huggingface_hub`

## Installation

```bash
git clone https://github.com/yourname/model-registry.git
cd model-registry
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt
```

### Optional: add `mr` to your PATH

On Windows you can copy or symlink `mr.py` and create a small wrapper so you can type `mr` anywhere:

```bat
@echo off
python "C:\path\to\model-registry\mr.py" %*
```

Save it as `mr.bat` somewhere on your PATH.

## Quick start

```bash
# First-time setup (interactive wizard, writes config.json)
python mr.py init

# Scan all enabled backends
python mr.py scan

# List locally installed models
python mr.py list

# Show full details for a model
python mr.py show llama3

# Rate a model interactively
python mr.py rate llama3

# Pull a model from HuggingFace (llama.cpp backend)
python mr.py pull bartowski/Llama-3.2-3B-Instruct-GGUF

# Pull from CivitAI into ComfyUI
python mr.py pull "urn:air:sdxl:checkpoint:civitai:2218365@2741096"

# Fetch HuggingFace + CivitAI metadata for all registry entries
python mr.py enrich

# Summary report
python mr.py report
```

## Configuration

`mr init` generates `config.json` next to `mr.py`. You can also copy `config.example.json` as a starting point:

```json
{
  "registry_db": "",
  "backends": {
    "ollama": {
      "enabled": true,
      "mode": "docker",
      "docker_container": "ollama"
    },
    "llamacpp": {
      "enabled": true,
      "model_dir": "/path/to/gguf/models",
      "extensions": [".gguf"]
    },
    "comfyui": {
      "enabled": false,
      "base_dir": "/path/to/ComfyUI/models",
      "extensions": [".safetensors", ".ckpt", ".pt", ".pth", ".bin"]
    }
  },
  "huggingface": {
    "token_env_var": "HF_TOKEN"
  },
  "civitai": {
    "token_env_var": "CIVITAI_API_KEY"
  },
  "display": {
    "date_format": "%Y-%m-%d",
    "max_name_width": 60
  }
}
```

Set `registry_db` to an absolute path to store the database somewhere other than the script directory. Leave blank to use `registry.db` next to `mr.py`.

### API tokens

Tokens are read from environment variables (not stored in config):

```bash
export HF_TOKEN=hf_...          # HuggingFace (required for gated repos)
export CIVITAI_API_KEY=...      # CivitAI (required for gated models)
```

## Commands

| Command | Description |
|---|---|
| `init` | Interactive setup wizard. Writes `config.json` and initialises the DB. |
| `scan` | Scan all enabled backends and update the registry. |
| `list` | List registry entries (local & non-blacklisted by default). |
| `show MODEL` | Full detail panel for a model. |
| `rate MODEL` | Set rating (1–5), status, and an optional note interactively. |
| `status MODEL STATUS` | Change status without touching the rating. |
| `note MODEL [TEXT]` | Append a timestamped note. |
| `touch MODEL` | Mark `last_used = now` (for tracking use outside this tool). |
| `report` | Summary: totals by backend/status, unrated list, blacklisted list, cross-backend duplicates. |
| `pull REF` | Download a model and register it. See below. |
| `rename MODEL NEW_NAME` | Rename a model file on disk and update the registry entry. |
| `delete MODEL` | Remove a model from Ollama or disk; DB record is kept with `status=deleted`. |
| `blacklist MODEL [REASON]` | Blacklist and auto-delete a model. Future pulls warn before proceeding. |
| `enrich` | Fetch HuggingFace and CivitAI metadata for all registered models. |
| `search TERM` | Search registry by name, HF repo, or notes. |

### `list` options

```
--backend   ollama | llamacpp | comfyui
--status    active | unrated | blacklisted | deleted | on_hold | testing | keep | favorite
--unrated   Show only models with no rating
--all       Include non-local and blacklisted models
```

### `pull` reference formats

Backend is auto-detected from the ref format:

| Format | Example | Backend |
|---|---|---|
| Plain Ollama name | `llama3.2:latest` | ollama |
| `org/repo` (HF) | `bartowski/Llama-3.2-3B-GGUF` | llamacpp |
| `.gguf` filename | `model.Q4_K_M.gguf` | llamacpp |
| `hf.co/org/repo:tag` | `hf.co/bartowski/Llama-3.2-3B-GGUF:Q4_K_M` | ollama |
| HF resolve URL | `https://huggingface.co/org/repo/resolve/main/file.safetensors` | comfyui |
| AIR tag | `urn:air:sdxl:checkpoint:civitai:1234@5678` | comfyui |
| CivitAI version ID | `civitai:12345` | comfyui |
| CivitAI URL | `https://civitai.com/models/1234?modelVersionId=5678` | comfyui |

Use `--backend` to override auto-detection. For ComfyUI downloads, `--subdir` selects the destination subdirectory (e.g. `checkpoints`, `loras`, `vae`). The subdir is auto-detected from AIR tags and the CivitAI API when possible.

```bash
# Force a HF repo into ComfyUI/loras
python mr.py pull stabilityai/stable-diffusion-xl-base-1.0 --backend comfyui --subdir checkpoints

# Specific GGUF variant via glob pattern
python mr.py pull bartowski/Llama-3.2-3B-Instruct-GGUF --file "*Q4_K_M*"
```

### Model statuses

| Status | Meaning |
|---|---|
| `unrated` | Default; not yet evaluated |
| `active` | In regular use |
| `favorite` | Top-tier model |
| `keep` | Worth keeping, not actively used |
| `testing` | Under evaluation |
| `on_hold` | Paused — may reconsider later |
| `blacklisted` | Do not use; warns on future pull attempts |
| `deleted` | Removed from disk; DB record retained |

## Database

The SQLite database (`registry.db`) has two tables:

- **`models`** — one row per model with all metadata (rating, status, size, source URL, HF stats, CivitAI trigger words, etc.)
- **`events`** — append-only audit log of every mutating operation

The schema migrates automatically when new columns are added, so existing databases are never broken by upgrades.

## License

MIT — see [LICENSE](LICENSE).
