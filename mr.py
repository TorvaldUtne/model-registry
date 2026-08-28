#!/usr/bin/env python3
"""Model Registry (mr) - Track, rate, and manage AI models across Ollama and llama.cpp backends."""

import fnmatch
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
from huggingface_hub import hf_hub_download, HfApi
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
import requests

console = Console()

__version__ = "1.2.5"

SCRIPT_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "config.json"
CONFIG_EXAMPLE = SCRIPT_DIR / "config.example.json"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config():
    if not CONFIG_FILE.exists():
        console.print("[red]config.json not found. Run [bold]mr init[/bold] first.[/red]")
        sys.exit(1)
    with open(CONFIG_FILE) as f:
        return json.load(f)


# ─── Database ─────────────────────────────────────────────────────────────────

def get_db_path(config):
    p = config.get("registry_db", "")
    if p:
        return Path(p)
    return SCRIPT_DIR / "registry.db"


def get_db(config):
    db_path = get_db_path(config)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS models (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            display_name      TEXT NOT NULL,
            hf_repo           TEXT,
            variant           TEXT,
            backend           TEXT NOT NULL,
            source_type       TEXT,
            ollama_name       TEXT,
            file_path         TEXT,
            status            TEXT DEFAULT 'unrated',
            rating            INTEGER,
            tags              TEXT,
            notes             TEXT,
            size_gb           REAL,
            currently_local   INTEGER DEFAULT 1,
            times_downloaded  INTEGER DEFAULT 0,
            first_seen        TEXT,
            last_used         TEXT,
            last_updated      TEXT,
            param_count       TEXT,
            architecture      TEXT,
            hf_downloads      INTEGER,
            hf_likes          INTEGER,
            hf_last_modified  TEXT,
            source_url        TEXT,
            base_model        TEXT,
            trigger_words     TEXT,
            context_window    INTEGER
        );

        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id    INTEGER REFERENCES models(id),
            event_type  TEXT,
            timestamp   TEXT,
            detail      TEXT
        );
    """)
    conn.commit()

    # Migrations for columns added after initial release
    for col, definition in [
        ("source_url",    "TEXT"),
        ("base_model",    "TEXT"),
        ("trigger_words", "TEXT"),
        ("context_window", "INTEGER"),
    ]:
        try:
            conn.execute(f"ALTER TABLE models ADD COLUMN {col} {definition}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists


# ─── DB helpers ───────────────────────────────────────────────────────────────

def _last_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT last_insert_rowid()").fetchone()
    assert row is not None
    return int(row[0])


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    assert row is not None
    return row[0]


# ─── Model matching ───────────────────────────────────────────────────────────

def find_model(conn: sqlite3.Connection, name: str) -> sqlite3.Row:
    """Return a single models row matching name (partial on display_name and ollama_name)."""
    rows: list[sqlite3.Row] = conn.execute(
        """SELECT * FROM models
           WHERE display_name LIKE ? OR ollama_name LIKE ?
           ORDER BY display_name""",
        (f"%{name}%", f"%{name}%"),
    ).fetchall()

    if not rows:
        console.print(f"[red]No model matching '{name}' found.[/red]")
        all_names = [r["display_name"] for r in conn.execute("SELECT display_name FROM models").fetchall()]
        name_lower = name.lower()
        suggestions = [n for n in all_names if name_lower in n.lower()][:5]
        if suggestions:
            console.print("Did you mean:")
            for s in suggestions:
                console.print(f"  {s}")
        sys.exit(1)

    if len(rows) == 1:
        return rows[0]

    console.print(f"[yellow]Multiple models match '{name}':[/yellow]")
    for i, row in enumerate(rows, 1):
        local_status = "(local)" if row["currently_local"] else "(remote)"
        console.print(f"  {i}. {row['display_name']}  [{row['backend']}] {local_status}")
    choice = click.prompt("Pick a number", type=click.IntRange(1, len(rows)))
    return rows[choice - 1]


# ─── Ollama parsing ───────────────────────────────────────────────────────────

def parse_ollama_size(size_str):
    """Convert '13 GB', '637 MB', etc. to float GB."""
    m = re.match(r"([\d.]+)\s*(GB|MB|KB)", size_str.strip(), re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "MB":
        val /= 1024
    elif unit == "KB":
        val /= 1024 * 1024
    return round(val, 2)


def parse_hf_repo_from_ollama(ollama_name):
    """Extract (hf_repo, variant) from an ollama model name string."""
    # hf.co/org/repo:variant  or  huggingface.co/org/repo:tag
    m = re.match(r"(?:hf\.co|huggingface\.co)/([^:]+?)(?::(.+))?$", ollama_name, re.IGNORECASE)
    if m:
        repo = m.group(1).strip("/")
        variant = m.group(2)
        return repo, variant

    # org/repo:tag with exactly one slash (Ollama namespace or direct HF shorthand)
    # Must be exactly "word/word" — tweaked/YanLabs/model has two slashes and won't match
    m = re.match(r"^([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?::(.+))?$", ollama_name)
    if m:
        return m.group(1), m.group(2)

    return None, None


def get_source_type(ollama_name):
    """Determine source_type from ollama model name."""
    if re.match(r"(?:hf\.co|huggingface\.co)/", ollama_name, re.IGNORECASE):
        return "ollama_hf"
    return "ollama_direct"


def get_gguf_backend_names(config):
    """Return all file-based GGUF backend names (everything except 'ollama' and 'comfyui')."""
    return [
        name for name in config.get("backends", {})
        if name not in ("ollama", "comfyui")
    ]


def run_ollama_list(container):
    result = subprocess.run(
        ["docker", "exec", container, "ollama", "list"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker exec failed: {result.stderr.strip()}")
    return result.stdout.strip().splitlines()


def parse_ollama_list_lines(lines):
    """Parse ollama list output into list of dicts with ollama_name and size_gb."""
    models = []
    for line in lines[1:]:  # skip header
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        # NAME  ID  SIZE_NUM  SIZE_UNIT  MODIFIED...
        name = parts[0]
        size_gb = parse_ollama_size(parts[2] + " " + parts[3])
        models.append({"ollama_name": name, "size_gb": size_gb})
    return models

def get_hf_metadata(hf_repo, hf_token):
    """Fetch metadata (param_count, architecture, downloads, likes, last_modified) from HuggingFace model card."""
    if not hf_repo:
        return {}
    try:
        from huggingface_hub import HfApi
        hf_api = HfApi(token=hf_token)
        try:
            model_info = hf_api.model_info(repo_id=hf_repo)
        except Exception:
            # Catch 404 (RepositoryNotFoundError), 401, 403, network drops, etc.
            return {}

        metadata = {}
        if hasattr(model_info, 'safetensors') and model_info.safetensors and isinstance(model_info.safetensors, dict):
            for key in ["total_params", "num_params"]:
                if key in model_info.safetensors:
                    metadata["param_count"] = model_info.safetensors[key]
                    break

        card_data = getattr(model_info, 'card_data', None) or getattr(model_info, 'cardData', None)
        if card_data and isinstance(card_data, dict):
            if "architectures" in card_data and isinstance(card_data["architectures"], list) and card_data["architectures"]:
                metadata["architecture"] = card_data["architectures"][0]
            elif "model_name" in card_data:
                name_lower = str(card_data["model_name"]).lower()
                if "llama" in name_lower:
                    metadata["architecture"] = "Llama"
                elif "mistral" in name_lower:
                    metadata["architecture"] = "Mistral"

        metadata["hf_downloads"] = getattr(model_info, 'downloads', None)
        metadata["hf_likes"] = getattr(model_info, 'likes', None)
        last_mod = getattr(model_info, 'lastModified', None) or getattr(model_info, 'last_modified', None)
        metadata["hf_last_modified"] = last_mod.isoformat() if hasattr(last_mod, 'isoformat') else str(last_mod) if last_mod else None

        return metadata
    except Exception:
        return {}

def get_hf_context_window(hf_repo, hf_token):
    """Fetch context window from HuggingFace model config."""
    if not hf_repo:
        return None
    try:
        from huggingface_hub import hf_hub_download, HfApi
        import json
        import requests

        def _get_context_from_config(repo, token):
            try:
                config_path = hf_hub_download(repo_id=repo, filename="config.json", token=token)
                with open(config_path, "r") as f:
                    config = json.load(f)
                for key in ["max_position_embeddings", "max_sequence_length", "n_ctx", "seq_length", "max_seq_len", "sliding_window", "context_length"]:
                    if key in config and isinstance(config[key], int):
                        return config[key]
            except Exception:
                return None

        # 1. Try repo itself
        ctx = _get_context_from_config(hf_repo, hf_token)
        if ctx is not None:
            return ctx

        # 2. Check base_model from model card/tags
        try:
            api = HfApi(token=hf_token)
            info = api.model_info(hf_repo)
            # Check card_data for base_model
            card_data = getattr(info, "card_data", None) or getattr(info, "cardData", None)
            base_models = []
            if card_data and getattr(card_data, "base_model", None):
                bm = card_data.base_model
                if isinstance(bm, list):
                    base_models.extend(bm)
                elif isinstance(bm, str):
                    base_models.append(bm)
            for tag in getattr(info, "tags", []):
                if tag.startswith("base_model:"):
                    base_models.append(tag.split(":", 1)[1].replace("quantized:", ""))

            for bm in base_models:
                ctx = _get_context_from_config(bm, hf_token)
                if ctx is not None:
                    return ctx
        except Exception:
            pass

        return None
    except Exception as e:
        console.print(f"[yellow]  Warning: Could not fetch HF context window for {hf_repo}: {e}[/yellow]")
        return None


# ─── llama.cpp helpers ────────────────────────────────────────────────────────

def parse_variant_from_filename(filename):
    """Extract quant variant from a .gguf filename, e.g. Q4_K_M, IQ4_XS."""
    m = re.search(r"\.(Q\d[^.]*|IQ\d[^.]*|f16|f32|bf16)\.gguf$", filename, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None





def parse_context_window_from_gguf(file_path):
    """Extract context window and architecture from GGUF file metadata using the gguf library or raw fallback."""
    if not file_path or not Path(file_path).exists():
        return None

    # 1. Prefer standard official `gguf` library if available
    try:
        import gguf
        reader = gguf.GGUFReader(file_path)
        arch = None
        if "general.architecture" in reader.fields:
            part = reader.fields["general.architecture"].parts[-1]
            arch = bytes(part).decode("utf-8", errors="ignore")

        if arch and f"{arch}.context_length" in reader.fields:
            return int(reader.fields[f"{arch}.context_length"].parts[-1][0])

        for k, v in reader.fields.items():
            if k.endswith(".context_length"):
                return int(v.parts[-1][0])
    except Exception:
        pass

    # 2. Raw binary parser fallback
    try:
        import struct
        with open(file_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None

            version = struct.unpack("<I", f.read(4))[0]
            if version >= 3:
                f.read(8)  # skip tensor count & metadata kv count header padding if needed

            # Read kv pairs looking for context length
            # Note: GGUF header stores <arch>.context_length (e.g. llama.context_length, qwen2.context_length, etc.)
            while True:
                key_type_bytes = f.read(4)
                if not key_type_bytes or len(key_type_bytes) < 4:
                    break
                key_len = struct.unpack("<Q" if version >= 3 else "<I", f.read(8 if version >= 3 else 4))[0]
                if key_len > 256:  # sanity check
                    break
                key = f.read(key_len).decode("utf-8", errors="ignore")
                val_type = struct.unpack("<I", f.read(4))[0]

                if key.endswith(".context_length") or key == "context_length":
                    if val_type in (0, 1):  # 8-bit
                        return struct.unpack("<B", f.read(1))[0]
                    elif val_type in (2, 3):  # 16-bit
                        return struct.unpack("<H", f.read(2))[0]
                    elif val_type in (4, 5):  # 32-bit
                        return struct.unpack("<I", f.read(4))[0]
                    elif val_type in (10, 11):  # 64-bit
                        return struct.unpack("<Q", f.read(8))[0]
                    break
                else:
                    # Skip values according to type
                    type_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
                    if val_type in type_sizes:
                        f.read(type_sizes[val_type])
                    elif val_type == 8:  # string
                        slen = struct.unpack("<Q" if version >= 3 else "<I", f.read(8 if version >= 3 else 4))[0]
                        f.read(slen)
                    elif val_type == 9:  # array
                        atype = struct.unpack("<I", f.read(4))[0]
                        alen = struct.unpack("<Q" if version >= 3 else "<I", f.read(8 if version >= 3 else 4))[0]
                        if atype == 8:
                            for _ in range(alen):
                                slen = struct.unpack("<Q" if version >= 3 else "<I", f.read(8 if version >= 3 else 4))[0]
                                f.read(slen)
                        elif atype in type_sizes:
                            f.read(type_sizes[atype] * alen)
                        else:
                            break
                    else:
                        break
    except Exception:
        pass

    return None


# ─── CivitAI / AIR helpers ───────────────────────────────────────────────────

_CIVITAI_DOMAIN_RE = r"civitai\.(?:com|green|red)"

# AIR type field → ComfyUI subdir name
AIR_TYPE_TO_SUBDIR = {
    "checkpoint": "checkpoints",
    "model":      "checkpoints",
    "vae":        "vae",
    "lora":       "loras",
    "locon":      "loras",
    "lycoris":    "loras",
    "embedding":  "embeddings",
    "textualinversion": "embeddings",
    "hypernet":   "hypernetworks",
    "controlnet": "controlnet",
    "upscaler":   "upscale_models",
    "ipadapter":  "ipadapter",
    "clipvision": "clip_vision",
}

# CivitAI API model type field → ComfyUI subdir name
CIVITAI_API_TYPE_TO_SUBDIR = {
    "checkpoint":        "checkpoints",
    "textualinversion":  "embeddings",
    "hypernetwork":      "hypernetworks",
    "lora":              "loras",
    "locon":             "loras",
    "controlnet":        "controlnet",
    "upscaler":          "upscale_models",
    "motionmodule":      "animatediff_models",
    "vae":               "vae",
    "poses":             "poses",
}


def parse_air_tag(ref):
    """Parse an AIR URN for a CivitAI resource. Returns dict or None.

    Handles both full and shorthand forms per the spec (urn: and air: are optional):
      urn:air:{ecosystem}:{type}:civitai:{model_id}@{version_id}
      e.g. urn:air:sdxl:checkpoint:civitai:2218365@2741096
           sdxl:checkpoint:civitai:2218365@2741096
    """
    # Strip optional urn: and air: prefixes
    s = re.sub(r"^(?:urn:)?(?:air:)?", "", ref.strip(), flags=re.IGNORECASE)
    m = re.match(
        r"^([^:]+):([^:]+):civitai:(\d+)@(\d+)$",
        s, re.IGNORECASE,
    )
    if not m:
        return None
    return {
        "ecosystem":  m.group(1).lower(),
        "type":       m.group(2).lower(),
        "model_id":   m.group(3),
        "version_id": m.group(4),
    }


def parse_civitai_version_id(ref):
    """Extract CivitAI version ID from an AIR tag, URL, or 'civitai:<id>' shorthand."""
    air = parse_air_tag(ref)
    if air:
        return air["version_id"]
    # civitai:12345
    m = re.match(r"^civitai:(\d+)$", ref, re.IGNORECASE)
    if m:
        return m.group(1)
    # https://civitai.com/api/download/models/12345  (also .green / .red)
    m = re.search(_CIVITAI_DOMAIN_RE + r"/api/download/models/(\d+)", ref)
    if m:
        return m.group(1)
    # https://civitai.com/models/12345?modelVersionId=67890  (also .green / .red)
    m = re.search(r"[?&]modelVersionId=(\d+)", ref)
    if m:
        return m.group(1)
    return None


def parse_civitai_model_id(ref):
    """Extract the model ID from a CivitAI browse URL (any domain variant)."""
    m = re.search(_CIVITAI_DOMAIN_RE + r"/models/(\d+)", ref, re.IGNORECASE)
    return m.group(1) if m else None


def fetch_civitai_model_info(model_id, token=None, host="civitai.com"):
    """Call CivitAI API v1 for a model. Returns (version_id, subdir_hint) or (None, None)."""
    try:
        import requests as _requests
    except ImportError:
        return None, None
    url = f"https://{host}/api/v1/models/{model_id}"
    params = {}
    if token:
        params["token"] = token
    try:
        resp = _requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return None, None
        data = resp.json()
    except (_requests.RequestException, json.JSONDecodeError):
        return None, None
    # Default/latest version is first in the list
    versions = data.get("modelVersions", [])
    version_id = str(versions[0]["id"]) if versions else None
    api_type = (data.get("type") or "").lower().replace(" ", "")
    subdir = CIVITAI_API_TYPE_TO_SUBDIR.get(api_type)
    return version_id, subdir


def civitai_source_url(ref, version_id, model_id=None):
    """Return the value to store as source_url for a CivitAI download.

    AIR tags are stored as-is (get_model_link converts them on read).
    Model page URLs are normalized. civitai:NNN shorthand builds what it can.
    """
    if parse_air_tag(ref):
        return ref  # store the AIR tag verbatim
    if model_id:
        return f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
    if re.search(_CIVITAI_DOMAIN_RE + r"/models/", ref, re.IGNORECASE):
        m = re.match(r"(https://" + _CIVITAI_DOMAIN_RE + r"/models/\d+(?:/[^?#]*)?)", ref, re.IGNORECASE)
        base = m.group(1).rstrip("/") if m else "https://civitai.com/models"
        return f"{base}?modelVersionId={version_id}"
    # civitai:NNN or raw download URL — only version ID known
    return f"https://civitai.com/models?modelVersionId={version_id}"


# ─── Link helpers ────────────────────────────────────────────────────────────

def get_model_link(row):
    """Return a browse URL for a model row, or None."""
    if row["source_url"]:
        air = parse_air_tag(row["source_url"])
        if air:
            return f"https://civitai.com/models/{air['model_id']}?modelVersionId={air['version_id']}"
        return row["source_url"]
    if row["hf_repo"]:
        return f"https://huggingface.co/{row['hf_repo']}"
    if row["backend"] == "ollama":
        oname = row["ollama_name"] or row["display_name"] or ""
        if oname and "/" not in oname:
            # Plain ollama library model e.g. llava:latest, moondream:latest
            return f"https://ollama.com/library/{oname}"
    return None


# ─── Status display ───────────────────────────────────────────────────────────

STATUS_COLORS = {
    "active": "green",
    "unrated": "white",
    "blacklisted": "red",
    "deleted": "dim",
    "on_hold": "yellow",
    "testing": "cyan",
    "keep": "green",
    "favorite": "magenta",
}


# ─── ComfyUI scan ────────────────────────────────────────────────────────────

def scan_comfyui(config, conn):
    """Scan all immediate subdirs of the ComfyUI models base_dir. Returns (added, updated)."""
    comfy_cfg = config["backends"].get("comfyui", {})
    base_dir = Path(comfy_cfg.get("base_dir", ""))
    extensions = comfy_cfg.get("extensions", [".safetensors", ".ckpt", ".pt", ".pth", ".bin"])
    now = now_iso()
    added = updated = 0

    if not base_dir.exists():
        console.print(f"[red]ComfyUI base_dir does not exist: {base_dir}[/red]")
        return added, updated

    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        console.print(f"  [yellow]No subdirectories found in {base_dir}[/yellow]")
        return added, updated

    seen_paths = set()
    for subdir in sorted(subdirs):
        variant = subdir.name
        files = []
        for ext in extensions:
            files.extend(subdir.glob(f"*{ext}"))
        for f in sorted(files):
            fpath = str(f)
            seen_paths.add(fpath)
            size_gb = round(f.stat().st_size / (1024 ** 3), 4)

            existing = conn.execute(
                "SELECT * FROM models WHERE file_path=?", (fpath,)
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE models SET size_gb=?, currently_local=1, last_updated=? WHERE id=?",
                    (size_gb, now, existing["id"]),
                )
                conn.execute(
                    "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                    (existing["id"], "scan_updated", now, json.dumps({"size_gb": size_gb})),
                )
                updated += 1
            else:
                conn.execute(
                    """INSERT INTO models
                       (display_name, variant, backend, source_type,
                        file_path, size_gb, currently_local, first_seen, last_updated)
                       VALUES (?,?,?,?,?,?,1,?,?)""",
                    (f.stem, variant, "comfyui", "comfyui_unknown", fpath, size_gb, now, now),
                )
                mid = _last_id(conn)
                conn.execute(
                    "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                    (mid, "scan_added", now, json.dumps({"file_path": fpath})),
                )
                added += 1

    # Mark disappeared files as not-local
    db_comfy = conn.execute(
        "SELECT id, file_path FROM models WHERE backend='comfyui' AND currently_local=1"
    ).fetchall()
    for row in db_comfy:
        if row["file_path"] not in seen_paths:
            conn.execute(
                "UPDATE models SET currently_local=0, last_updated=? WHERE id=?",
                (now, row["id"]),
            )
            console.print(f"  [yellow]Marked not-local: {row['file_path']}[/yellow]")

    return added, updated


# ─── CLI group ────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Model Registry - track, rate, and manage AI models."""
    pass


@cli.command()
def backends():
    """List configured backend names."""
    config = load_config()
    backends = list(config.get("backends", {}).keys())
    console.print(f"[bold]Configured backends:[/bold]")
    for b in backends:
        status = "enabled" if config["backends"][b].get("enabled", False) else "disabled"
        console.print(f"  - {b} [dim]({status})[/dim]")


# ─── init ─────────────────────────────────────────────────────────────────────

@cli.command()
def init():
    """First-time setup wizard. Generates config.json."""
    console.print("[bold]Model Registry Setup[/bold]\n")

    if CONFIG_FILE.exists():
        if not click.confirm("config.json already exists. Overwrite?", default=False):
            sys.exit(0)

    # Load defaults from example if available
    defaults = {}
    if CONFIG_EXAMPLE.exists():
        with open(CONFIG_EXAMPLE) as f:
            defaults = json.load(f)

    container = click.prompt(
        "Ollama Docker container name",
        default=defaults.get("backends", {}).get("ollama", {}).get("docker_container", "ollama"),
    )

    gguf_dir = click.prompt(
        "Path to GGUF model directory (leave blank to disable llamacpp)",
        default="",
    )

    db_path = click.prompt(
        "Registry DB path (leave blank for same dir as mr.py)",
        default="",
    )

    hf_env_var = click.prompt(
        "HuggingFace token environment variable name",
        default=defaults.get("huggingface", {}).get("token_env_var", "HF_TOKEN"),
    )

    # Verify Docker connectivity
    console.print("\nChecking Docker connectivity...")
    try:
        result = subprocess.run(
            ["docker", "exec", container, "ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            console.print(f"[green]✓ Docker container '{container}' is reachable.[/green]")
        else:
            console.print(f"[yellow]⚠ Docker exec returned error: {result.stderr.strip()}[/yellow]")
            console.print("[yellow]  (Ollama backend may not work until Docker is running)[/yellow]")
    except FileNotFoundError:
        console.print("[yellow]⚠ 'docker' command not found. Ollama backend will not work.[/yellow]")
    except subprocess.TimeoutExpired:
        console.print("[yellow]⚠ Docker exec timed out.[/yellow]")

    # Verify GGUF dir
    llamacpp_enabled = bool(gguf_dir)
    if gguf_dir:
        p = Path(gguf_dir)
        if p.exists() and p.is_dir():
            console.print(f"[green]✓ GGUF directory exists: {p}[/green]")
        else:
            console.print(f"[yellow]⚠ GGUF directory not found: {p}[/yellow]")

    # ComfyUI setup
    comfyui_enabled = click.confirm("\nEnable ComfyUI backend?", default=False)
    comfyui_base_dir = ""
    if comfyui_enabled:
        default_comfy = defaults.get("backends", {}).get("comfyui", {}).get(
            "base_dir", r"M:\Programs\ComfyUI\ComfyUI\models"
        )
        comfyui_base_dir = click.prompt("ComfyUI models base directory", default=default_comfy)
        p = Path(comfyui_base_dir)
        if p.exists() and p.is_dir():
            console.print(f"[green]✓ ComfyUI models directory exists: {p}[/green]")
        else:
            console.print(f"[yellow]⚠ ComfyUI directory not found: {p}[/yellow]")

    civitai_env_var = click.prompt(
        "CivitAI token environment variable name (leave blank to skip)",
        default=defaults.get("civitai", {}).get("token_env_var", "CIVITAI_API_KEY"),
    )

    config = {
        "registry_db": db_path,
        "backends": {
            "ollama": {
                "enabled": True,
                "mode": "docker",
                "docker_container": container,
            },
            "llamacpp": {
                "enabled": llamacpp_enabled,
                "model_dir": gguf_dir,
                "extensions": [".gguf"],
            },
            "comfyui": {
                "enabled": comfyui_enabled,
                "base_dir": comfyui_base_dir,
                "extensions": [".safetensors", ".ckpt", ".pt", ".pth", ".bin"],
            },
        },
        "huggingface": {
            "token_env_var": hf_env_var,
        },
        "civitai": {
            "token_env_var": civitai_env_var,
        },
        "display": {
            "date_format": "%Y-%m-%d",
            "max_name_width": 60,
        },
    }

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    console.print(f"\n[green]✓ config.json written.[/green]")

    conn = get_db(config)
    init_db(conn)
    conn.close()
    console.print(f"[green]✓ Database initialized at {get_db_path(config)}[/green]")
    console.print("\n[bold]Setup complete.[/bold] Run [bold]mr scan[/bold] to populate the registry.")


# ─── scan ─────────────────────────────────────────────────────────────────────

@cli.command()
def scan():
    """Scan Ollama, llama.cpp, and ComfyUI backends, update the registry."""
    config = load_config()
    conn = get_db(config)
    init_db(conn)
    now = now_iso()
    added = updated = 0

    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_cfg = config["backends"].get("ollama", {})
    hf_token = os.environ.get(config.get("huggingface", {}).get("token_env_var"))
    if ollama_cfg.get("enabled", False):
        container = ollama_cfg.get("docker_container", "ollama")
        console.print(f"Scanning Ollama (container: [bold]{container}[/bold])...")
        try:
            lines = run_ollama_list(container)
            ollama_models = parse_ollama_list_lines(lines)
            console.print(f"  Found {len(ollama_models)} model(s) in Ollama.")

            seen_hf_repos = {}  # hf_repo -> first ollama_name seen (dedup)

            for om in ollama_models:
                oname = om["ollama_name"]
                size_gb = om["size_gb"]
                hf_repo, variant = parse_hf_repo_from_ollama(oname)
                source_type = get_source_type(oname)
                context_window = None
                if hf_repo:
                    context_window = get_hf_context_window(hf_repo, hf_token)
                # Deduplicate: same hf_repo already registered this scan pass
                if hf_repo and hf_repo in seen_hf_repos:
                    console.print(
                        f"  [dim]Skipping duplicate: {oname} "
                        f"(same hf_repo as {seen_hf_repos[hf_repo]})[/dim]"
                    )
                    continue
                if hf_repo:
                    seen_hf_repos[hf_repo] = oname

                # Look up existing record (prefer hf_repo match, fall back to ollama_name)
                existing = None
                if hf_repo:
                    existing = conn.execute(
                        "SELECT * FROM models WHERE hf_repo=? AND backend='ollama'",
                        (hf_repo,),
                    ).fetchone()
                if not existing:
                    existing = conn.execute(
                        "SELECT * FROM models WHERE ollama_name=? AND backend='ollama'",
                        (oname,),
                    ).fetchone()

                if existing:
                    update_fields = {"size_gb": size_gb, "currently_local": 1, "last_updated": now, "ollama_name": oname}
                    if context_window is not None:
                        update_fields["context_window"] = context_window

                    set_clauses = [f"{k}=?" for k in update_fields.keys()]
                    params = list(update_fields.values()) + [existing["id"]]

                    conn.execute(
                        f"UPDATE models SET {', '.join(set_clauses)} WHERE id=?",
                        params,
                    )

                    event_detail = {"size_gb": size_gb, "ollama_name": oname}
                    if context_window is not None:
                        event_detail["context_window"] = context_window

                    conn.execute(
                        "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                        (existing["id"], "scan_updated", now, json.dumps(event_detail)),
                    )
                    updated += 1
                else:
                    conn.execute(
                        """INSERT INTO models
                           (display_name, hf_repo, variant, backend, source_type,
                            ollama_name, size_gb, context_window, currently_local, first_seen, last_updated)
                           VALUES (?,?,?,?,?,?,?,?,1,?,?)""",
                        (oname, hf_repo, variant, "ollama", source_type, oname, size_gb, context_window, now, now),
                    )
                    mid = _last_id(conn)
                    conn.execute(
                        "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                        (mid, "scan_added", now, json.dumps({"ollama_name": oname, "context_window": context_window})),
                    )
                    added += 1

            # Mark models that disappeared from Ollama as not-local
            db_ollama = conn.execute(
                "SELECT id, ollama_name FROM models WHERE backend='ollama' AND currently_local=1"
            ).fetchall()
            seen_names = {om["ollama_name"] for om in ollama_models}
            for row in db_ollama:
                if row["ollama_name"] not in seen_names:
                    conn.execute(
                        "UPDATE models SET currently_local=0, last_updated=? WHERE id=?",
                        (now, row["id"]),
                    )
                    conn.execute(
                        "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                        (row["id"], "scan_updated", now, json.dumps({"currently_local": 0})),
                    )
                    console.print(f"  [yellow]Marked not-local: {row['ollama_name']}[/yellow]")

        except RuntimeError as e:
            console.print(f"[red]Ollama scan failed: {e}[/red]")
        except FileNotFoundError:
            console.print("[red]'docker' command not found. Is Docker installed and on PATH?[/red]")

    # ── GGUF file backends (llamacpp, llamaserver, etc.) ─────────────────────
    for bname in get_gguf_backend_names(config):
        bcfg = config["backends"][bname]
        if not bcfg.get("enabled", False):
            continue
        model_dir = Path(bcfg.get("model_dir", ""))
        extensions = bcfg.get("extensions", [".gguf"])
        console.print(f"\nScanning {bname} models in [bold]{model_dir}[/bold]...")
        if not model_dir.exists():
            console.print(f"[red]{bname} model_dir does not exist: {model_dir}[/red]")
            continue

        # Scan subdirectories first (each subdir = one model with multiple GGUF files)
        # Then scan top-level .gguf files
        subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
        top_level_files = []
        for ext in extensions:
            top_level_files.extend(model_dir.glob(f"*{ext}"))
        # Filter out files that are inside subdirs
        top_level_files = [f for f in top_level_files if f.parent == model_dir]

        total_files = 0
        seen_paths = set()

        # Process subdirectories (models with multiple GGUF files)
        for subdir in sorted(subdirs):
            subdir_gguf_files = []
            for ext in extensions:
                subdir_gguf_files.extend(subdir.glob(f"*{ext}"))

            if not subdir_gguf_files:
                continue

            total_files += len(subdir_gguf_files)
            variant = subdir.name
            # Use the largest file in the subdir for size and name
            main_file = max(subdir_gguf_files, key=lambda f: f.stat().st_size)
            fpath = str(main_file)
            seen_paths.add(fpath)
            size_gb = round(sum(f.stat().st_size for f in subdir_gguf_files) / (1024 ** 3), 4)

            # Use subdir name as display_name, variant is subdir name too
            existing = conn.execute(
                "SELECT * FROM models WHERE file_path=?", (fpath,)
            ).fetchone()

            if existing:
                context_window = parse_context_window_from_gguf(fpath)
                update_fields = {"size_gb": size_gb, "currently_local": 1, "last_updated": now, "variant": variant}
                if context_window is not None:
                    update_fields["context_window"] = context_window

                set_clauses = [f"{k}=?" for k in update_fields.keys()]
                params = list(update_fields.values()) + [existing["id"]]

                conn.execute(
                    f"UPDATE models SET {', '.join(set_clauses)} WHERE id=?",
                    params,
                )

                event_detail = {"size_gb": size_gb, "variant": variant}
                if context_window is not None:
                    event_detail["context_window"] = context_window

                conn.execute(
                    "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                    (existing["id"], "scan_updated", now, json.dumps(event_detail)),
                )
                updated += 1
            else:
                conn.execute(
                    """INSERT INTO models
                       (display_name, variant, backend, source_type,
                        file_path, size_gb, currently_local, first_seen, last_updated)
                       VALUES (?,?,?,?,?,?,1,?,?)""",
                    (subdir.name, variant, bname, bname, fpath, size_gb, now, now),
                )
                mid = _last_id(conn)
                conn.execute(
                    "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                    (mid, "scan_added", now, json.dumps({"file_path": fpath, "subdir": subdir.name})),
                )
                added += 1

        # Process top-level .gguf files (single-file models)
        for f in sorted(top_level_files):
            fpath = str(f)
            seen_paths.add(fpath)
            size_gb = round(f.stat().st_size / (1024 ** 3), 4)
            variant = parse_variant_from_filename(f.name)
            context_window = parse_context_window_from_gguf(fpath)

            existing = conn.execute(
                "SELECT * FROM models WHERE file_path=?", (fpath,)
            ).fetchone()

            if existing:
                update_fields = {"size_gb": size_gb, "currently_local": 1, "last_updated": now}
                if context_window is not None:
                    update_fields["context_window"] = context_window
                set_clauses = [f"{k}=?" for k in update_fields.keys()]
                params = list(update_fields.values()) + [existing["id"]]
                conn.execute(
                    f"UPDATE models SET {', '.join(set_clauses)} WHERE id=?",
                    params,
                )
                conn.execute(
                    "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                    (existing["id"], "scan_updated", now, json.dumps(update_fields)),
                )
                updated += 1
            else:
                display_name = f.stem
                context_window = parse_context_window_from_gguf(fpath)
                conn.execute(
                    """INSERT INTO models
                       (display_name, variant, backend, source_type,
                        file_path, size_gb, context_window, currently_local, first_seen, last_updated)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (display_name, variant, bname, bname, fpath, size_gb, context_window, now, now),
                )
                mid = _last_id(conn)
                conn.execute(
                    "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                    (mid, "scan_added", now, json.dumps({"file_path": fpath})),
                )
                added += 1

        console.print(f"  Found {total_files + len(top_level_files)} GGUF file(s) in {len(subdirs) + len(top_level_files)} model(s).")

        # Mark disappeared files/subdirs as not-local
        db_rows = conn.execute(
            "SELECT id, file_path FROM models WHERE backend=? AND currently_local=1", (bname,)
        ).fetchall()
        for row in db_rows:
            if row["file_path"] not in seen_paths:
                conn.execute(
                    "UPDATE models SET currently_local=0, last_updated=? WHERE id=?",
                    (now, row["id"]),
                )
                console.print(f"  [yellow]Marked not-local: {row['file_path']}[/yellow]")

    # ── ComfyUI ──────────────────────────────────────────────────────────────
    comfy_cfg = config["backends"].get("comfyui", {})
    if comfy_cfg.get("enabled", False):
        base_dir = comfy_cfg.get("base_dir", "")
        console.print(f"\nScanning ComfyUI models in [bold]{base_dir}[/bold]...")
        c_added, c_updated = scan_comfyui(config, conn)
        added += c_added
        updated += c_updated
        console.print(f"  ComfyUI: [green]{c_added}[/green] added, [cyan]{c_updated}[/cyan] updated")

    conn.commit()
    conn.close()
    console.print(
        f"\n[green]Scan complete.[/green] "
        f"Added: [bold]{added}[/bold]  Updated: [bold]{updated}[/bold]"
    )




# ─── list ─────────────────────────────────────────────────────────────────────

@cli.command("list")
@click.option("--backend", type=str, default=None)
@click.option(
    "--status",
    type=click.Choice(["active", "unrated", "blacklisted", "deleted", "on_hold", "testing", "keep", "favorite"]),
    default=None,
)
@click.option("--unrated", is_flag=True, default=False, help="Show only models with no rating")
@click.option("--all", "show_all", is_flag=True, default=False, help="Include non-local and blacklisted models")
@click.option("--deleted", is_flag=True, default=False, help="Show only deleted models")
def list_models(backend, status, unrated, show_all, deleted):
    """List models in the registry. By default shows only locally installed, non-blacklisted models."""
    config = load_config()
    conn = get_db(config)
    init_db(conn)

    query = "SELECT * FROM models WHERE 1=1"
    params = []
    if deleted:
        query += " AND status='deleted'"
    elif not show_all:
        query += " AND currently_local=1 AND status != 'blacklisted'"
    if backend:
        query += " AND backend=?"
        params.append(backend)
    if status and not deleted:
        query += " AND status=?"
        params.append(status)
    if unrated:
        query += " AND rating IS NULL"
    query += " ORDER BY display_name"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        console.print("No models found.")
        return

    date_fmt = config.get("display", {}).get("date_format", "%Y-%m-%d")

    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
    table.add_column("Name", no_wrap=True)
    if backend != "comfyui":
        table.add_column("Backend", width=9)
    if backend == "comfyui":
        table.add_column("Type", width=14)
    table.add_column("Status", width=12)
    table.add_column("Rating", width=7, justify="center")
    table.add_column("Size", width=9, justify="right")
    table.add_column("Last Used", width=12)
    table.add_column("Tags", width=30)

    for row in rows:
        status_val = row["status"] or "unrated"
        color = STATUS_COLORS.get(status_val, "white")
        rating_str = f"{row['rating']}/5" if row["rating"] else "-"
        if row["size_gb"] is None:
            size_str = "-"
        elif row["size_gb"] < 0.1:
            size_str = f"{row['size_gb'] * 1024:.0f} MB"
        else:
            size_str = f"{row['size_gb']:.1f} GB"

        last_used = row["last_used"]
        if last_used:
            try:
                dt = datetime.fromisoformat(last_used.replace("Z", "+00:00"))
                last_used = dt.strftime(date_fmt)
            except ValueError:
                pass

        tags_str = ""
        if row["tags"]:
            try:
                tags = json.loads(row["tags"])
                tags_str = ", ".join(tags)
            except json.JSONDecodeError:
                tags_str = row["tags"]

        not_local = "" if row["currently_local"] else " [dim](not local)[/dim]"
        row_cells = [f"[{color}]{row['display_name']}{not_local}[/{color}]"]
        if backend != "comfyui":
            row_cells.append(row["backend"])
        if backend == "comfyui":
            row_cells.append(row["variant"] or "-")
        row_cells += [
            f"[{color}]{status_val}[/{color}]",
            rating_str,
            size_str,
            last_used or "-",
            tags_str,
        ]
        table.add_row(*row_cells)

    console.print(table)
    console.print(f"[dim]{len(rows)} model(s)[/dim]")


# ─── show ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("model")
def show(model):
    """Show full details for a model."""
    config = load_config()
    conn = get_db(config)
    init_db(conn)
    row = find_model(conn, model)

    tags = []
    if row["tags"]:
        try:
            tags = json.loads(row["tags"])
        except json.JSONDecodeError:
            tags = [row["tags"]]

    lines = []
    lines.append(f"[bold cyan]Name:[/bold cyan]       {row['display_name']}")
    lines.append(f"[bold]Backend:[/bold]    {row['backend']}")
    lines.append(f"[bold]Status:[/bold]     {row['status'] or 'unrated'}")
    if row["rating"]:
        lines.append(f"[bold]Rating:[/bold]     {row['rating']}/5")
    else:
        lines.append("[bold]Rating:[/bold]     unrated")
    if row["hf_repo"]:
        lines.append(f"[bold]HF Repo:[/bold]    {row['hf_repo']}")
    link = get_model_link(row)
    if link:
        lines.append(f"[bold]Link:[/bold]       {link}")
    if row["variant"]:
        lines.append(f"[bold]Variant:[/bold]    {row['variant']}")
    if row["ollama_name"]:
        lines.append(f"[bold]Ollama:[/bold]     {row['ollama_name']}")
    if row["file_path"]:
        lines.append(f"[bold]File:[/bold]       {row['file_path']}")
    lines.append(f"[bold]Size:[/bold]       {row['size_gb']:.2f} GB" if row["size_gb"] is not None else "[bold]Size:[/bold]       -")
    try:
        context_window = row["context_window"]
        if context_window:
            lines.append(f"[bold]Context:[/bold]    {context_window:,} tokens")
    except (KeyError, TypeError):
        pass  # context_window column doesn't exist in older DBs
    lines.append(f"[bold]Local:[/bold]      {'yes' if row['currently_local'] else 'no'}")
    lines.append(f"[bold]Downloads:[/bold]  {row['times_downloaded']}")
    if tags:
        lines.append(f"[bold]Tags:[/bold]       {', '.join(tags)}")
    if row["first_seen"]:
        lines.append(f"[bold]First seen:[/bold] {row['first_seen']}")
    if row["last_used"]:
        lines.append(f"[bold]Last used:[/bold]  {row['last_used']}")
    if row["last_updated"]:
        lines.append(f"[bold]Updated:[/bold]    {row['last_updated']}")

    # Phase 3 HF enrichment fields
    if any(row[k] is not None for k in ("param_count", "architecture", "hf_downloads", "hf_likes", "hf_last_modified")):
        lines.append("")
        lines.append("[bold dim]─── HF Metadata ───[/bold dim]")
        if row["param_count"] is not None: # Check for None explicitly, as it could be 0
            lines.append(f"[bold]Params:[/bold]     {row['param_count']:,}")
        if row["architecture"]:
            lines.append(f"[bold]Arch:[/bold]       {row['architecture']}")
        if row["hf_downloads"] is not None:
            lines.append(f"[bold]DL count:[/bold]   {row['hf_downloads']:,}")
        if row["hf_likes"] is not None:
            lines.append(f"[bold]Likes:[/bold]      {row['hf_likes']:,}")
        if row["hf_last_modified"]:
            lines.append(f"[bold]HF updated:[/bold] {row['hf_last_modified']}")

    if row["base_model"] or row["trigger_words"]:
        lines.append("")
        lines.append("[bold dim]─── CivitAI Metadata ───[/bold dim]")
        if row["base_model"]:
            lines.append(f"[bold]Base model:[/bold] {row['base_model']}")
        if row["trigger_words"]:
            try:
                words = json.loads(row["trigger_words"])
            except json.JSONDecodeError:
                words = [row["trigger_words"]]
            lines.append(f"[bold]Triggers:[/bold]   {', '.join(words)}")

    if row["notes"]:
        lines.append("")
        lines.append("[bold dim]─── Notes ───[/bold dim]")
        lines.append(row["notes"].strip())

    # Recent events
    events = conn.execute(
        "SELECT * FROM events WHERE model_id=? ORDER BY timestamp DESC LIMIT 10",
        (row["id"],),
    ).fetchall()
    if events:
        lines.append("")
        lines.append("[bold dim]─── Recent Events ───[/bold dim]")
        for e in events:
            lines.append(f"  [dim]{e['timestamp']}[/dim]  {e['event_type']}  {e['detail'] or ''}")

    conn.close()
    console.print(
        Panel("\n".join(lines), title=f"[bold]{row['display_name']}[/bold]", expand=False)
    )


# ─── rate ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("model")
def rate(model):
    """Interactively rate a model (1-5), set status, add optional note."""
    config = load_config()
    conn = get_db(config)
    init_db(conn)
    row = find_model(conn, model)
    now = now_iso()

    console.print(f"\n[bold]Rating:[/bold] {row['display_name']}")
    if row["rating"]:
        console.print(f"  Current: {row['rating']}/5  status={row['status']}")

    new_rating = click.prompt("Rating (1-5)", type=click.IntRange(1, 5))
    new_status = click.prompt(
        "Status",
        type=click.Choice(["active", "unrated", "blacklisted", "deleted", "on_hold", "testing", "keep", "favorite"]),
        default=row["status"] or "active",
    )
    note_text = click.prompt("Note (blank to skip)", default="", show_default=False)

    notes = row["notes"] or ""
    if note_text:
        notes += f"\n[{now}] {note_text}"

    conn.execute(
        "UPDATE models SET rating=?, status=?, notes=?, last_updated=? WHERE id=?",
        (new_rating, new_status, notes, now, row["id"]),
    )
    conn.execute(
        "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
        (row["id"], "rate", now, json.dumps({"rating": new_rating, "status": new_status})),
    )
    if note_text:
        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (row["id"], "note", now, note_text),
        )

    conn.commit()
    conn.close()
    console.print(f"[green]✓ Rated {new_rating}/5, status: {new_status}[/green]")


# ─── status ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("model")
@click.argument("status", type=click.Choice(["active", "unrated", "blacklisted", "deleted", "on_hold", "testing", "keep", "favorite"]))
def status(model, status):
    """Set a model's status without requiring a rating."""
    config = load_config()
    conn = get_db(config)
    init_db(conn)
    row = find_model(conn, model)
    now = now_iso()

    conn.execute(
        "UPDATE models SET status=?, last_updated=? WHERE id=?",
        (status, now, row["id"]),
    )
    conn.execute(
        "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
        (row["id"], "setstatus", now, json.dumps({"status": status})),
    )
    conn.commit()
    conn.close()
    console.print(f"[green]✓ Status set to: {status}[/green]")


# ─── note ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("model")
@click.argument("text", nargs=-1, required=False)
def note(model, text):
    """Append a timestamped note to a model."""
    config = load_config()
    conn = get_db(config)
    init_db(conn)
    row = find_model(conn, model)
    now = now_iso()

    note_text = " ".join(text) if text else click.prompt("Note")

    notes = row["notes"] or ""
    notes += f"\n[{now}] {note_text}"

    conn.execute(
        "UPDATE models SET notes=?, last_updated=? WHERE id=?",
        (notes, now, row["id"]),
    )
    conn.execute(
        "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
        (row["id"], "note", now, note_text),
    )
    conn.commit()
    conn.close()
    console.print("[green]✓ Note added.[/green]")


# ─── touch ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("model")
def touch(model):
    """Update last_used to now (use when you ran a model outside this tool)."""
    config = load_config()
    conn = get_db(config)
    init_db(conn)
    row = find_model(conn, model)
    now = now_iso()

    conn.execute(
        "UPDATE models SET last_used=?, last_updated=? WHERE id=?",
        (now, now, row["id"]),
    )
    conn.execute(
        "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
        (row["id"], "touch", now, None),
    )
    conn.commit()
    conn.close()
    console.print(f"[green]✓ last_used updated for {row['display_name']}[/green]")


# ─── report ───────────────────────────────────────────────────────────────────

@cli.command()
def report():
    """Summary: total models, GB by backend, unrated list, blacklisted list, cross-backend duplicates."""
    config = load_config()
    conn = get_db(config)
    init_db(conn)

    total = _scalar(conn, "SELECT COUNT(*) FROM models")
    total_gb = _scalar(conn, "SELECT COALESCE(SUM(size_gb), 0) FROM models WHERE currently_local=1")

    by_backend = conn.execute(
        """SELECT backend, COUNT(*) as cnt, SUM(size_gb) as gb
           FROM models WHERE currently_local=1
           GROUP BY backend"""
    ).fetchall()

    by_status = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM models GROUP BY status ORDER BY cnt DESC"
    ).fetchall()

    unrated = conn.execute(
        """SELECT display_name, backend, size_gb
           FROM models WHERE rating IS NULL AND currently_local=1
           ORDER BY display_name"""
    ).fetchall()

    blacklisted = conn.execute(
        "SELECT display_name, rating, notes FROM models WHERE status='blacklisted'"
    ).fetchall()

    dupes = conn.execute(
        """SELECT hf_repo,
                  COUNT(DISTINCT backend) AS backend_count,
                  GROUP_CONCAT(backend || ': ' || display_name, '  |  ') AS names
           FROM models
           WHERE hf_repo IS NOT NULL AND currently_local=1
           GROUP BY hf_repo
           HAVING backend_count > 1"""
    ).fetchall()

    console.print(f"\n[bold]Model Registry Report[/bold]")
    console.print(f"Total models tracked: [bold]{total}[/bold]  |  Local storage: [bold]{total_gb:.1f} GB[/bold]")

    console.print("\n[bold]By Backend (local):[/bold]")
    for r in by_backend:
        console.print(f"  {r['backend']:12s}  {r['cnt']} models   {(r['gb'] or 0):.1f} GB")

    console.print("\n[bold]By Status:[/bold]")
    for r in by_status:
        console.print(f"  {(r['status'] or 'unrated'):12s}  {r['cnt']}")

    if unrated:
        console.print(f"\n[bold]Unrated Models ({len(unrated)}):[/bold]")
        for r in unrated:
            size_str = f"  {r['size_gb']:.1f} GB" if r["size_gb"] is not None else ""
            console.print(f"  [{r['backend']}] {r['display_name']}{size_str}")

    if blacklisted:
        console.print(f"\n[bold red]Blacklisted ({len(blacklisted)}):[/bold red]")
        for r in blacklisted:
            rating_str = f"rating={r['rating']}" if r["rating"] else "unrated"
            first_note = (r["notes"] or "").strip().splitlines()[0] if r["notes"] else ""
            console.print(f"  {r['display_name']}  ({rating_str})  {first_note}")

    if dupes:
        console.print(f"\n[bold yellow]Cross-backend Duplicates ({len(dupes)}):[/bold yellow]")
        for r in dupes:
            console.print(f"  [yellow]{r['hf_repo']}[/yellow]")
            console.print(f"    {r['names']}")
    else:
        console.print("\n[dim]No cross-backend duplicates detected.[/dim]")
        gguf_backends = get_gguf_backend_names(config)
        placeholders = ",".join("?" * len(gguf_backends))
        no_hf = _scalar(
            conn,
            f"SELECT COUNT(*) FROM models WHERE backend IN ({placeholders}) AND hf_repo IS NULL AND currently_local=1",
            tuple(gguf_backends),
        ) if gguf_backends else 0
        if no_hf:
            console.print(
                f"[dim]  ({no_hf} GGUF model(s) have no hf_repo — "
                "run 'mr enrich' for accurate duplicate detection)[/dim]"
            )

    conn.close()


# ─── enrich ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--all", "enrich_all", is_flag=True, default=False, help="Enrich all models with hf_repo, not just local ones")
def enrich(enrich_all):
    """Fetch additional metadata (context window, parameters, architecture, stats) from HuggingFace Hub."""
    config = load_config()
    conn = get_db(config)
    init_db(conn)
    now = now_iso()
    hf_token = os.environ.get(config.get("huggingface", {}).get("token_env_var"))

    where_clause = "WHERE (hf_repo IS NOT NULL AND hf_repo != '') OR (file_path IS NOT NULL AND file_path LIKE '%.gguf')"
    if not enrich_all:
        where_clause += " AND currently_local=1"

    rows = conn.execute(f"""
        SELECT id, display_name, hf_repo, backend, file_path, ollama_name,
               context_window, param_count, architecture, hf_downloads, hf_likes, hf_last_modified
        FROM models
        {where_clause}
        ORDER BY display_name
    """).fetchall()

    if not rows:
        console.print("[yellow]No models with hf_repo found to enrich.[/yellow]")
        conn.close()
        return

    console.print(f"[bold]Enriching {len(rows)} model(s) from HuggingFace Hub...[/bold]")

    updated_count = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Enriching models", total=len(rows))
        for row in rows:
            progress.update(task, description=f"Enriching [bold]{row['display_name'][:40]}[/bold]")
            update_fields = {}

            # Context window
            if row["context_window"] is None:
                # First try local GGUF file if path exists
                if row["file_path"] and Path(row["file_path"]).exists():
                    ctx = parse_context_window_from_gguf(row["file_path"])
                    if ctx is not None:
                        update_fields["context_window"] = ctx

                # If still not found, try Hugging Face
                if "context_window" not in update_fields and row["hf_repo"]:
                    ctx = get_hf_context_window(row["hf_repo"], hf_token)
                    if ctx is not None:
                        update_fields["context_window"] = ctx

            # Metadata from HF API
            if any(row[k] is None for k in ("param_count", "architecture", "hf_downloads", "hf_likes", "hf_last_modified")):
                meta = get_hf_metadata(row["hf_repo"], hf_token)
                if meta:
                    for k, v in meta.items():
                        if v is not None and row[k] is None:
                            update_fields[k] = v

                # Convert any datetime objects to ISO strings
                for k, v in list(update_fields.items()):
                    if isinstance(v, datetime):
                        update_fields[k] = v.isoformat()

                update_fields["last_updated"] = now
                set_clauses = [f"{k}=?" for k in update_fields.keys()]
                params = list(update_fields.values()) + [row["id"]]
                conn.execute(
                    f"UPDATE models SET {', '.join(set_clauses)} WHERE id=?",
                    params,
                )
                conn.execute(
                    "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                    (row["id"], "enrich_updated", now, json.dumps(update_fields)),
                )
                updated_count += 1

            progress.advance(task)

    conn.commit()
    conn.close()
    console.print(f"\n[green]Enrichment complete.[/green] Updated [bold]{updated_count}[/bold] of [bold]{len(rows)}[/bold] model(s).")


# ─── search ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("term")
def search(term):
    """Search local registry by name, hf_repo, notes, or tags."""
    config = load_config()
    conn = get_db(config)
    init_db(conn)

    rows = conn.execute(
        """SELECT * FROM models
           WHERE display_name LIKE ? OR hf_repo LIKE ? OR notes LIKE ? OR tags LIKE ?
           ORDER BY display_name""",
        (f"%{term}%", f"%{term}%", f"%{term}%", f"%{term}%"),
    ).fetchall()
    conn.close()

    if not rows:
        console.print(f"No results for '{term}'.")
        return

    for row in rows:
        color = STATUS_COLORS.get(row["status"] or "unrated", "white")
        rating_str = f"{row['rating']}/5" if row["rating"] else "unrated"
        size_str = f"  {row['size_gb']:.1f} GB" if row["size_gb"] is not None else ""
        console.print(
            f"[{color}]{row['display_name']}[/{color}]  "
            f"[dim][{row['backend']}][/dim]  {rating_str}{size_str}"
        )
        link = get_model_link(row)
        if link:
            console.print(f"  [dim]{link}[/dim]")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) == 1:
        console.print(f"[bold]Model Registry v{__version__}[/bold]")
        console.print("Run [bold]mr --help[/bold] for available commands.")
        sys.exit(0)
    cli()
