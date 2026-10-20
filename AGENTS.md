# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this project is

`mr` (Model Registry) tracks, rates, and manages AI models across three backends: **Ollama** (via Docker), **llama.cpp** (local `.gguf` files), and **ComfyUI** (local `.safetensors`/`.ckpt`/etc.). It stores state in a local SQLite database.

It ships three entry points:

- **`mr.py`** — interactive CLI (click + rich). Thin rendering layer over the engine.
- **`mr_mcp.py`** — MCP server (Streamable HTTP) exposing the same engine to agents over the LAN without shell/filesystem access.
- **`mr_core.py`** — the engine: all logic and data access, no UI dependencies.

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

## Running tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

The suite (`tests/test_mr_core.py`) runs entirely against a temporary SQLite DB and temp model dirs — it never touches the real registry or model files.

## Running commands

```bash
python mr.py <command>
```

All commands: `init`, `scan`, `list`, `show`, `rate`, `status`, `note`, `touch`, `report`, `pull`, `rename`, `delete`, `blacklist`, `enrich`, `search`, `backends`, `copy`, `remove`, `removeall`, `restore`, `tag`, `untag`.

## Running the MCP server

```bash
python mr_mcp.py                    # read-only, LAN (0.0.0.0:8321)
python mr_mcp.py --read-write       # expose write tools too
mr-serve.bat --read-write           # same via launcher
```

MCP settings (`host`, `port`, `mode`, `stateful`) live in the `mcp` section of `config.json`. The Streamable HTTP transport is **stateless by default** (`stateful: false`): it mints no session id and ignores any `mcp-session-id` header, so a cached/stale session id from a previous run or after the 30-minute idle expiry can never cause a hard `404`. Pass `--stateful` or set `"stateful": true` only if a client needs resumability or server-initiated requests. Destructive MCP tools (`pull`, `delete`, `remove`, `removeall`, `rename`, `copy`, `blacklist`, `restore`) additionally require `confirm=True` per call. No auth — trusted LAN only.

## Architecture

The engine lives in `mr_core.py`. Both `mr.py` and `mr_mcp.py` are thin clients over it:

1. **Config** (`mr_core.load_config`) reads `config.json` next to the scripts.
2. **Database** (`mr_core.get_db` / `mr_core.init_db`) opens/creates `registry.db` (path from config or script dir). Schema: `models` table + `events` audit log. Column migrations are handled inline with `ALTER TABLE … ADD COLUMN` wrapped in try/except.
3. **CLI** (`mr.py`) is built with `click` (`@cli.command()`); every handler calls an `engine_*` function from `mr_core` and renders the returned dict/list with rich. Keep prompting + rendering here only.
4. **MCP** (`mr_mcp.py`) registers `engine_*` functions as typed `MCPServer` tools (names prefixed `mr_`).
5. **Backends**:
   - *Ollama*: `docker exec <container> ollama list/pull/rm`
   - *llama.cpp*: filesystem glob of `.gguf` files; downloads via `huggingface_hub`
   - *ComfyUI*: filesystem scan of subdirs under `base_dir`; downloads from HuggingFace or CivitAI

### Key conventions

- **Engine functions return structured data** (`list[dict]` / `dict`), never print, never `sys.exit`. They raise `MrError` subclasses:
  - `ModelNotFound(name, suggestions)`, `AmbiguousModel(name, matches)`, `DestructiveOperation`.
- Engine writes are gated: call `mr_core.set_writes_enabled(True)` before allowing mutations (the CLI does this at import; the MCP server only in read-write mode). Destructive-to-disk engines require a `confirm` parameter.
- Interactive disambiguation: `mr_core.resolve_model(conn, name, index)` raises `AmbiguousModel`; the CLI catches it and prompts (`click.prompt`), the MCP client passes `index`/revises.
- Every mutating operation writes an entry to the `events` table.
- `source_type` values: `ollama_direct`, `ollama_hf`, `llamacpp`, `comfyui_unknown`, `comfyui_hf`, `comfyui_civitai`.
- `source_url` stores either a CivitAI AIR tag (verbatim) or a `https://` browse URL. `mr_core.get_model_link(row, config)` converts AIR tags to browse URLs on read, using the configured `civitai.web_host` (default `civitai.red`; `civitai.green` is SFW-only). `civitai.api_host` (default `civitai.com`) is used for API calls, except that a CivitAI URL ref uses the host from that URL.
- CivitAI token must be passed as a query param (`?token=…`), not an `Authorization` header, because the CDN redirect strips headers.

## Adding a command

1. Add an `engine_<name>(config=None, …) -> dict|list` function in `mr_core.py`. Return structured data; raise `MrError` on failure; append progress lines to an optional `log: list[str]` param for long operations.
2. Call it from `mr.py` (prompting/rendering) and register it as an `@mcp.tool()` `mr_<name>` in `mr_mcp.py` (read-only vs write sets in `register_read_tools` / `build_write_tools`).

## Adding new columns

Add the column to the `CREATE TABLE` block in `mr_core.init_db` **and** add a migration entry in the `for col, definition in [...]` loop below it (same function). This pattern lets the schema evolve without breaking existing databases.