# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`mr` (Model Registry) is a single-file CLI tool (`mr.py`) that tracks, rates, and manages AI models across three backends: **Ollama** (via Docker), **llama.cpp** (local `.gguf` files), and **ComfyUI** (local `.safetensors`/`.ckpt`/etc.). It stores state in a local SQLite database.

## Setup

```bash
# Create and activate venv
python -m venv venv
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# First-time config (interactive wizard)
python mr.py init
```

`mr init` writes `config.json` (gitignored) from prompts and initializes `registry.db`. Copy `config.example.json` as a starting point if you want to pre-fill values.

## Running commands

```bash
python mr.py <command>
```

All commands: `init`, `scan`, `list`, `show`, `rate`, `status`, `note`, `touch`, `report`, `pull`, `rename`, `delete`, `blacklist`, `enrich`, `search`.

## Architecture

Everything lives in `mr.py` — no modules, no packages. The flow is:

1. **Config** (`load_config`) reads `config.json` next to `mr.py`.
2. **Database** (`get_db` / `init_db`) opens/creates `registry.db` (path from config or same dir). Schema: `models` table + `events` audit log. Column migrations are handled inline with `ALTER TABLE … ADD COLUMN` wrapped in try/except.
3. **CLI** is built with `click` (`@cli.command()`). Output uses `rich` (Console, Table, Panel, Progress).
4. **Backends**:
   - *Ollama*: `docker exec <container> ollama list/pull/rm`
   - *llama.cpp*: filesystem glob of `.gguf` files; downloads via `huggingface_hub`
   - *ComfyUI*: filesystem scan of subdirs under `base_dir`; downloads from HuggingFace or CivitAI

### Key conventions

- `find_model(conn, name)` — fuzzy match on `display_name` / `ollama_name`; prompts user to pick if ambiguous.
- Every mutating operation writes an entry to the `events` table.
- `source_type` values: `ollama_direct`, `ollama_hf`, `llamacpp`, `comfyui_unknown`, `comfyui_hf`, `comfyui_civitai`.
- `source_url` stores either a CivitAI AIR tag (verbatim) or a `https://` browse URL. `get_model_link(row)` converts AIR tags to browse URLs on read.
- CivitAI token must be passed as a query param (`?token=…`), not an `Authorization` header, because the CDN redirect strips headers.

## Adding new columns

Add the column to the `CREATE TABLE` block in `init_db` **and** add a migration entry in the `for col, definition in [...]` loop below it (same function). This pattern lets the schema evolve without breaking existing databases.
