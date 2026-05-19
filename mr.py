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
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

console = Console()

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
            trigger_words     TEXT
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
        suggestions = [n for n in all_names if any(c in n.lower() for c in name_lower)][:5]
        if suggestions:
            console.print("Did you mean:")
            for s in suggestions:
                console.print(f"  {s}")
        sys.exit(1)

    if len(rows) == 1:
        return rows[0]

    console.print(f"[yellow]Multiple models match '{name}':[/yellow]")
    for i, row in enumerate(rows, 1):
        console.print(f"  {i}. {row['display_name']}  [{row['backend']}]")
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


# ─── llama.cpp helpers ────────────────────────────────────────────────────────

def parse_variant_from_filename(filename):
    """Extract quant variant from a .gguf filename, e.g. Q4_K_M, IQ4_XS."""
    m = re.search(r"\.(Q\d[^.]*|IQ\d[^.]*|f16|f32|bf16)\.gguf$", filename, re.IGNORECASE)
    if m:
        return m.group(1).upper()
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
    except Exception:
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
                    conn.execute(
                        """UPDATE models
                           SET size_gb=?, currently_local=1, last_updated=?, ollama_name=?
                           WHERE id=?""",
                        (size_gb, now, oname, existing["id"]),
                    )
                    conn.execute(
                        "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                        (existing["id"], "scan_updated", now, json.dumps({"size_gb": size_gb})),
                    )
                    updated += 1
                else:
                    conn.execute(
                        """INSERT INTO models
                           (display_name, hf_repo, variant, backend, source_type,
                            ollama_name, size_gb, currently_local, first_seen, last_updated)
                           VALUES (?,?,?,?,?,?,?,1,?,?)""",
                        (oname, hf_repo, variant, "ollama", source_type, oname, size_gb, now, now),
                    )
                    mid = _last_id(conn)
                    conn.execute(
                        "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                        (mid, "scan_added", now, json.dumps({"ollama_name": oname})),
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

        files = []
        for ext in extensions:
            files.extend(model_dir.glob(f"*{ext}"))
        console.print(f"  Found {len(files)} file(s).")

        seen_paths = set()
        for f in sorted(files):
            fpath = str(f)
            seen_paths.add(fpath)
            size_gb = round(f.stat().st_size / (1024 ** 3), 4)
            variant = parse_variant_from_filename(f.name)

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
                display_name = f.stem
                conn.execute(
                    """INSERT INTO models
                       (display_name, variant, backend, source_type,
                        file_path, size_gb, currently_local, first_seen, last_updated)
                       VALUES (?,?,?,?,?,?,1,?,?)""",
                    (display_name, variant, bname, bname, fpath, size_gb, now, now),
                )
                mid = _last_id(conn)
                conn.execute(
                    "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                    (mid, "scan_added", now, json.dumps({"file_path": fpath})),
                )
                added += 1

        # Mark disappeared files as not-local
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
def list_models(backend, status, unrated, show_all):
    """List models in the registry. By default shows only locally installed, non-blacklisted models."""
    config = load_config()
    conn = get_db(config)

    query = "SELECT * FROM models WHERE 1=1"
    params = []
    if not show_all:
        query += " AND currently_local=1 AND status != 'blacklisted'"
    if backend:
        query += " AND backend=?"
        params.append(backend)
    if status:
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
            except Exception:
                pass

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
    row = find_model(conn, model)

    tags = []
    if row["tags"]:
        try:
            tags = json.loads(row["tags"])
        except Exception:
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
    if any(row[k] is not None for k in ("param_count", "architecture", "hf_downloads", "hf_likes")):
        lines.append("")
        lines.append("[bold dim]─── HF Metadata ───[/bold dim]")
        if row["param_count"]:
            lines.append(f"[bold]Params:[/bold]     {row['param_count']}")
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
            except Exception:
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


# ─── pull ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("ref")
@click.option("--backend", type=str, default=None)
@click.option("--file", "file_pattern", default=None, help="Glob pattern for file in HF repo")
@click.option("--subdir", default=None, help="ComfyUI subdir to save into (e.g. checkpoints, loras)")
def pull(ref, backend, file_pattern, subdir):
    """Pull a model. Warns if blacklisted or previously deleted.

    For ComfyUI models, --subdir is required. Supports HuggingFace repos
    (org/repo format) and CivitAI downloads (civitai:<versionId> or CivitAI URL).
    """
    config = load_config()
    conn = get_db(config)
    init_db(conn)
    now = now_iso()

    # Auto-detect backend
    if backend is None:
        if parse_civitai_version_id(ref) is not None:
            backend = "comfyui"
        elif ref.endswith(".gguf") or ("/" in ref and not re.search(r"(?:hf\.co|huggingface\.co)/", ref, re.IGNORECASE)):
            backend = "llamacpp"
        else:
            backend = "ollama"

    # Pre-pull: check if blacklisted or deleted
    hf_repo, _ = parse_hf_repo_from_ollama(ref)
    existing = None
    if hf_repo:
        existing = conn.execute(
            "SELECT * FROM models WHERE hf_repo=?", (hf_repo,)
        ).fetchone()
    if not existing:
        existing = conn.execute(
            "SELECT * FROM models WHERE display_name LIKE ? OR ollama_name=?",
            (f"%{ref}%", ref),
        ).fetchone()

    if existing:
        if existing["status"] == "blacklisted":
            console.print("[bold red]⚠ WARNING: This model is BLACKLISTED[/bold red]")
            console.print(f"  Rating: {existing['rating']}/5" if existing["rating"] else "  Unrated")
            if existing["notes"]:
                for line in (existing["notes"] or "").strip().splitlines()[-3:]:
                    console.print(f"  {line}")
            if not click.confirm("Proceed anyway?", default=False):
                conn.close()
                return
        elif existing["status"] == "deleted":
            console.print("[yellow]⚠ This model was previously deleted.[/yellow]")
            events = conn.execute(
                "SELECT * FROM events WHERE model_id=? ORDER BY timestamp DESC LIMIT 5",
                (existing["id"],),
            ).fetchall()
            for e in events:
                console.print(f"  [dim]{e['timestamp']}[/dim]  {e['event_type']}  {e['detail'] or ''}")
            if not click.confirm("Proceed anyway?", default=False):
                conn.close()
                return

    # ── Ollama pull ──────────────────────────────────────────────────────────
    if backend == "ollama":
        container = config["backends"]["ollama"]["docker_container"]
        console.print(f"Pulling [bold]{ref}[/bold] via Ollama...")
        result = subprocess.run(
            ["docker", "exec", container, "ollama", "pull", ref],
            text=True,
        )
        if result.returncode != 0:
            console.print("[red]Pull failed.[/red]")
            conn.close()
            return

        hf_repo2, variant2 = parse_hf_repo_from_ollama(ref)
        source_type = get_source_type(ref)

        if existing:
            conn.execute(
                """UPDATE models
                   SET currently_local=1, times_downloaded=times_downloaded+1,
                       last_used=?, last_updated=?
                   WHERE id=?""",
                (now, now, existing["id"]),
            )
            mid = existing["id"]
        else:
            conn.execute(
                """INSERT INTO models
                   (display_name, hf_repo, variant, backend, source_type,
                    ollama_name, currently_local, times_downloaded, first_seen, last_used, last_updated)
                   VALUES (?,?,?,?,?,?,1,1,?,?,?)""",
                (ref, hf_repo2, variant2, "ollama", source_type, ref, now, now, now),
            )
            mid = _last_id(conn)

        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (mid, "pull", now, json.dumps({"ref": ref})),
        )
        conn.commit()
        console.print("[green]✓ Pull complete. Registry updated.[/green]")

    # ── GGUF file backends (llamacpp, llamaserver, etc.) ────────────────────
    elif backend in get_gguf_backend_names(config):
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            console.print("[red]huggingface_hub not installed. Run: pip install huggingface_hub[/red]")
            conn.close()
            return

        model_dir = Path(config["backends"][backend]["model_dir"])
        token_env = config["huggingface"]["token_env_var"]
        token = os.environ.get(token_env)

        parsed_repo, parsed_variant = parse_hf_repo_from_ollama(ref)
        repo_id = parsed_repo if parsed_repo else ref
        if parsed_variant and not file_pattern:
            file_pattern = f"*{parsed_variant}*.gguf"

        if "/" not in repo_id:
            console.print(f"[red]For {backend}, ref must be 'org/repo' or 'hf.co/org/repo:tag' format.[/red]")
            conn.close()
            return

        all_files = list(list_repo_files(repo_id, token=token))
        gguf_files = [f for f in all_files if f.endswith(".gguf")]

        if not gguf_files:
            console.print(f"[red]No .gguf files found in {repo_id}[/red]")
            conn.close()
            return

        if file_pattern:
            matches = [f for f in gguf_files if fnmatch.fnmatch(f, file_pattern)]
            if not matches:
                console.print(f"[red]No files matching '{file_pattern}' in {repo_id}[/red]")
                conn.close()
                return
            chosen_file = matches[0]
        elif len(gguf_files) == 1:
            chosen_file = gguf_files[0]
        else:
            console.print(f"Multiple GGUF files in [bold]{repo_id}[/bold]:")
            for i, f in enumerate(gguf_files, 1):
                console.print(f"  {i}. {f}")
            idx = click.prompt("Pick a number", type=click.IntRange(1, len(gguf_files)))
            chosen_file = gguf_files[idx - 1]

        console.print(f"Downloading [bold]{chosen_file}[/bold] from {repo_id}...")
        local_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=chosen_file,
                local_dir=str(model_dir),
                token=token,
            )
        )

        size_gb = round(local_path.stat().st_size / (1024 ** 3), 2)
        variant = parse_variant_from_filename(local_path.name)

        if existing:
            conn.execute(
                """UPDATE models
                   SET currently_local=1, times_downloaded=times_downloaded+1,
                       file_path=?, size_gb=?, last_used=?, last_updated=?
                   WHERE id=?""",
                (str(local_path), size_gb, now, now, existing["id"]),
            )
            mid = existing["id"]
        else:
            conn.execute(
                """INSERT INTO models
                   (display_name, hf_repo, variant, backend, source_type,
                    file_path, size_gb, currently_local, times_downloaded,
                    first_seen, last_used, last_updated)
                   VALUES (?,?,?,?,?,?,?,1,1,?,?,?)""",
                (local_path.stem, repo_id, variant, backend, backend,
                 str(local_path), size_gb, now, now, now),
            )
            mid = _last_id(conn)

        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (mid, "pull", now, json.dumps({"repo_id": repo_id, "file": chosen_file})),
        )
        conn.commit()
        console.print(
            f"[green]✓ Downloaded to {local_path} ({size_gb:.2f} GB). Registry updated.[/green]"
        )

    # ── ComfyUI download (HuggingFace or CivitAI) ────────────────────────────
    elif backend == "comfyui":
        comfy_cfg = config["backends"].get("comfyui", {})
        base_dir = Path(comfy_cfg.get("base_dir", ""))
        if not base_dir.exists():
            console.print(f"[red]ComfyUI base_dir does not exist: {base_dir}[/red]")
            conn.close()
            return

        civitai_cfg = config.get("civitai", {})
        token_env = civitai_cfg.get("token_env_var", "CIVITAI_API_KEY")
        civitai_token = os.environ.get(token_env)

        civitai_version_id = parse_civitai_version_id(ref)
        _civitai_model_id = parse_civitai_model_id(ref)

        # Browse URL with model ID but no version ID — resolve via API
        if civitai_version_id is None and _civitai_model_id:
            _dm = re.search(_CIVITAI_DOMAIN_RE, ref)
            _host = _dm.group(0) if _dm else "civitai.com"
            console.print(f"Fetching model info from CivitAI API (model {_civitai_model_id})...")
            civitai_version_id, _api_subdir = fetch_civitai_model_info(
                _civitai_model_id, token=civitai_token, host=_host
            )
            if civitai_version_id:
                console.print(f"  Using latest version [bold]{civitai_version_id}[/bold]")
            if not subdir and _api_subdir:
                subdir = _api_subdir
                console.print(f"  Auto-detected subdir [bold]{subdir}[/bold] from CivitAI model type")

        if not subdir:
            # Auto-detect from AIR tag type field
            air = parse_air_tag(ref)
            if air:
                subdir = AIR_TYPE_TO_SUBDIR.get(air["type"])
                if subdir:
                    console.print(f"  Auto-detected subdir [bold]{subdir}[/bold] from AIR type '{air['type']}'")
        if not subdir:
            subdir = click.prompt("ComfyUI subdir (e.g. checkpoints, loras, vae)")

        dest_dir = base_dir / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)

        if civitai_version_id:
            # ── CivitAI download ─────────────────────────────────────────────
            try:
                import requests as _requests
            except ImportError:
                console.print("[red]requests not installed. Run: pip install requests[/red]")
                conn.close()
                return

            # Token must be a query param — Authorization header is stripped on CDN redirect
            _dm = re.search(_CIVITAI_DOMAIN_RE, ref)
            _civitai_host = _dm.group(0) if _dm else "civitai.com"
            download_url = f"https://{_civitai_host}/api/download/models/{civitai_version_id}"
            params = {}
            if civitai_token:
                params["token"] = civitai_token
            else:
                console.print("[yellow]⚠ No CivitAI API key found. Download may fail for gated models.[/yellow]")

            console.print(f"Downloading from CivitAI (version {civitai_version_id})...")
            resp = _requests.get(download_url, params=params, stream=True, timeout=(30, None))
            if resp.status_code == 401:
                console.print(f"[red]CivitAI download failed: unauthorized. Check your {token_env} env var.[/red]")
                conn.close()
                return
            if resp.status_code != 200:
                console.print(f"[red]CivitAI download failed (HTTP {resp.status_code})[/red]")
                conn.close()
                return

            # Get filename from Content-Disposition header
            cd = resp.headers.get("Content-Disposition", "")
            filename_match = re.search(r'filename="?([^";\r\n]+)"?', cd)
            if filename_match:
                filename = filename_match.group(1).strip()
            else:
                filename = click.prompt("Filename to save as (no path)")

            local_path = dest_dir / filename
            total = int(resp.headers.get("Content-Length", 0)) or None

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Downloading {filename}", total=total)
                with open(local_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        fh.write(chunk)
                        progress.advance(task, len(chunk))

            size_gb = round(local_path.stat().st_size / (1024 ** 3), 4)
            fpath = str(local_path)

            air = parse_air_tag(ref)
            civitai_url = civitai_source_url(ref, civitai_version_id, air["model_id"] if air else None)

            # Check if already in DB by file_path
            existing_by_path = conn.execute(
                "SELECT * FROM models WHERE file_path=?", (fpath,)
            ).fetchone()
            if existing_by_path or existing:
                row_to_update = existing_by_path or existing
                conn.execute(
                    """UPDATE models
                       SET currently_local=1, times_downloaded=times_downloaded+1,
                           file_path=?, size_gb=?, last_used=?, last_updated=?,
                           source_type='comfyui_civitai', source_url=?
                       WHERE id=?""",
                    (fpath, size_gb, now, now, civitai_url, row_to_update["id"]),
                )
                mid = row_to_update["id"]
            else:
                conn.execute(
                    """INSERT INTO models
                       (display_name, variant, backend, source_type, source_url,
                        file_path, size_gb, currently_local, times_downloaded,
                        first_seen, last_used, last_updated)
                       VALUES (?,?,?,?,?,?,?,1,1,?,?,?)""",
                    (local_path.stem, subdir, "comfyui", "comfyui_civitai", civitai_url,
                     fpath, size_gb, now, now, now),
                )
                mid = _last_id(conn)

            conn.execute(
                "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                (mid, "pull", now, json.dumps({"civitai_version_id": civitai_version_id, "file": filename})),
            )
            conn.commit()
            console.print(
                f"[green]✓ Downloaded to {local_path} ({size_gb:.2f} GB). Registry updated.[/green]"
            )

        else:
            # ── HuggingFace download to ComfyUI subdir ───────────────────────
            try:
                from huggingface_hub import hf_hub_download, list_repo_files
            except ImportError:
                console.print("[red]huggingface_hub not installed. Run: pip install huggingface_hub[/red]")
                conn.close()
                return

            token_env = config["huggingface"]["token_env_var"]
            token = os.environ.get(token_env)

            # Detect full HF resolve URLs: https://huggingface.co/org/repo/resolve/ref/path/file
            _hf_resolve = re.match(
                r"https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$",
                ref, re.IGNORECASE,
            )
            if _hf_resolve:
                repo_id     = _hf_resolve.group(1)
                revision    = _hf_resolve.group(2)
                chosen_file = _hf_resolve.group(3)
                console.print(
                    f"Downloading [bold]{chosen_file}[/bold] from {repo_id} @ {revision}..."
                )
            else:
                repo_id  = ref
                revision = None

                if "/" not in repo_id:
                    console.print("[red]For HuggingFace, ref must be 'org/repo' format or a full resolve URL.[/red]")
                    conn.close()
                    return

                comfy_exts = tuple(comfy_cfg.get("extensions", [".safetensors", ".ckpt", ".pt", ".pth", ".bin"]))
                all_files = list(list_repo_files(repo_id, token=token))
                model_files = [f for f in all_files if f.lower().endswith(comfy_exts)]

                if not model_files:
                    console.print(f"[red]No model files found in {repo_id}[/red]")
                    conn.close()
                    return

                if file_pattern:
                    matches = [f for f in model_files if fnmatch.fnmatch(f, file_pattern)]
                    if not matches:
                        console.print(f"[red]No files matching '{file_pattern}' in {repo_id}[/red]")
                        conn.close()
                        return
                    chosen_file = matches[0]
                elif len(model_files) == 1:
                    chosen_file = model_files[0]
                else:
                    console.print(f"Multiple model files in [bold]{repo_id}[/bold]:")
                    for i, f in enumerate(model_files, 1):
                        console.print(f"  {i}. {f}")
                    idx = click.prompt("Pick a number", type=click.IntRange(1, len(model_files)))
                    chosen_file = model_files[idx - 1]

                console.print(f"Downloading [bold]{chosen_file}[/bold] from {repo_id}...")
            _hf_kwargs = dict(repo_id=repo_id, filename=chosen_file,
                              local_dir=str(dest_dir), token=token)
            if revision:
                _hf_kwargs["revision"] = revision
            local_path = Path(hf_hub_download(**_hf_kwargs))

            size_gb = round(local_path.stat().st_size / (1024 ** 3), 4)
            fpath = str(local_path)

            existing_by_path = conn.execute(
                "SELECT * FROM models WHERE file_path=?", (fpath,)
            ).fetchone()
            if existing_by_path or existing:
                row_to_update = existing_by_path or existing
                conn.execute(
                    """UPDATE models
                       SET currently_local=1, times_downloaded=times_downloaded+1,
                           file_path=?, size_gb=?, hf_repo=?, last_used=?, last_updated=?,
                           source_type='comfyui_hf'
                       WHERE id=?""",
                    (fpath, size_gb, repo_id, now, now, row_to_update["id"]),
                )
                mid = row_to_update["id"]
            else:
                conn.execute(
                    """INSERT INTO models
                       (display_name, hf_repo, variant, backend, source_type,
                        file_path, size_gb, currently_local, times_downloaded,
                        first_seen, last_used, last_updated)
                       VALUES (?,?,?,?,?,?,?,1,1,?,?,?)""",
                    (local_path.stem, repo_id, subdir, "comfyui", "comfyui_hf",
                     fpath, size_gb, now, now, now),
                )
                mid = _last_id(conn)

            conn.execute(
                "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                (mid, "pull", now, json.dumps({"repo_id": repo_id, "file": chosen_file})),
            )
            conn.commit()
            console.print(
                f"[green]✓ Downloaded to {local_path} ({size_gb:.2f} GB). Registry updated.[/green]"
            )

    conn.close()


# ─── restore ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would be downloaded without downloading")
@click.pass_context
def restore(ctx, dry_run):
    """Re-download all missing ComfyUI models that have a known source.

    Models with source_type 'comfyui_civitai' are re-pulled via their stored
    source_url. Models with source_type 'comfyui_hf' are re-pulled via their
    hf_repo, using the stored file_path basename to select the right file.
    """
    config = load_config()
    conn = get_db(config)
    init_db(conn)

    restorable = conn.execute(
        """SELECT * FROM models
           WHERE backend='comfyui' AND currently_local=0
             AND (source_url IS NOT NULL OR hf_repo IS NOT NULL)
           ORDER BY display_name"""
    ).fetchall()

    no_source = conn.execute(
        """SELECT display_name FROM models
           WHERE backend='comfyui' AND currently_local=0
             AND source_url IS NULL AND hf_repo IS NULL
           ORDER BY display_name"""
    ).fetchall()

    conn.close()

    if not restorable and not no_source:
        console.print("[green]No missing ComfyUI models found.[/green]")
        return

    if no_source:
        console.print(
            f"[yellow]WARNING: {len(no_source)} model(s) have no recorded source and cannot be restored:[/yellow]"
        )
        for row in no_source:
            console.print(f"  [dim]- {row['display_name']}[/dim]")

    if not restorable:
        return

    console.print(f"\n[bold]{len(restorable)} model(s) queued for restore:[/bold]")
    for row in restorable:
        src = row["source_url"] or row["hf_repo"]
        subdir_label = f"[dim] -> {row['variant']}[/dim]" if row["variant"] else ""
        console.print(f"  - {row['display_name']}{subdir_label}  [dim]({src})[/dim]")

    if dry_run:
        return

    if not click.confirm(f"\nDownload {len(restorable)} model(s)?", default=True):
        return

    failed = []
    for row in restorable:
        console.rule(f"[bold]{row['display_name']}[/bold]")

        subdir = row["variant"] or None

        if row["source_url"]:
            ref = row["source_url"]
            file_pattern = None
        else:
            ref = row["hf_repo"]
            file_pattern = Path(row["file_path"]).name if row["file_path"] else None

        if not subdir and row["file_path"]:
            # Derive subdir from the stored file path relative to base_dir
            comfy_cfg = config["backends"].get("comfyui", {})
            base_dir = Path(comfy_cfg.get("base_dir", ""))
            try:
                rel = Path(row["file_path"]).relative_to(base_dir)
                subdir = rel.parts[0] if len(rel.parts) > 1 else None
            except ValueError:
                pass

        try:
            ctx.invoke(pull, ref=ref, backend="comfyui", file_pattern=file_pattern, subdir=subdir)
        except SystemExit:
            failed.append(row["display_name"])
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            failed.append(row["display_name"])

    console.rule()
    if failed:
        console.print(f"[red]Failed to restore {len(failed)} model(s):[/red]")
        for name in failed:
            console.print(f"  - {name}")
    else:
        console.print(f"[green]All {len(restorable)} model(s) restored.[/green]")


# ─── rename ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("model")
@click.argument("new_name")
def rename(model, new_name):
    """Rename a model file and registry entry, keeping all metadata intact.

    NEW_NAME should be given without the file extension — the extension is
    preserved automatically.  display_name and file_path are updated; all
    other fields (source_url, ratings, notes, events, etc.) are unchanged.
    """
    config = load_config()
    conn = get_db(config)
    row = find_model(conn, model)
    now = now_iso()

    old_display = row["display_name"]

    if row["backend"] != "ollama":
        if not row["file_path"]:
            console.print("[red]No file_path recorded for this model — cannot rename on disk.[/red]")
            conn.close()
            return
        old_path = Path(row["file_path"])
        if not old_path.exists():
            console.print(f"[yellow]File not found on disk: {old_path}[/yellow]")
            if not click.confirm("Update registry name anyway?", default=False):
                conn.close()
                return
            new_path = old_path.with_stem(new_name)
        else:
            new_path = old_path.with_stem(new_name)
            if new_path.exists():
                console.print(f"[red]A file already exists at {new_path} — aborting.[/red]")
                conn.close()
                return
            old_path.rename(new_path)
            console.print(f"[green]✓ Renamed file: {old_path.name} → {new_path.name}[/green]")

        conn.execute(
            "UPDATE models SET display_name=?, file_path=?, last_updated=? WHERE id=?",
            (new_name, str(new_path), now, row["id"]),
        )

    else:
        conn.execute(
            "UPDATE models SET display_name=?, last_updated=? WHERE id=?",
            (new_name, now, row["id"]),
        )

    conn.execute(
        "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
        (row["id"], "rename", now, f"{old_display} → {new_name}"),
    )
    conn.commit()
    conn.close()
    console.print(f"[green]✓ Registry updated: '{old_display}' → '{new_name}'[/green]")


# ─── delete ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("model")
def delete(model):
    """Delete a model from Ollama or disk. Keeps DB record, sets status=deleted."""
    config = load_config()
    conn = get_db(config)
    row = find_model(conn, model)
    now = now_iso()

    if not click.confirm(f"Delete '{row['display_name']}'?", default=False):
        conn.close()
        return

    if row["backend"] == "ollama":
        container = config["backends"]["ollama"]["docker_container"]
        result = subprocess.run(
            ["docker", "exec", container, "ollama", "rm", row["ollama_name"]],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            console.print(f"[red]Delete failed: {result.stderr.strip()}[/red]")
            conn.close()
            return
        console.print("[green]✓ Removed from Ollama.[/green]")

    elif row["backend"] != "ollama":
        if row["file_path"]:
            p = Path(row["file_path"])
            if p.exists():
                p.unlink()
                console.print(f"[green]✓ Deleted file: {p}[/green]")
            else:
                console.print(f"[yellow]File not found (already gone?): {p}[/yellow]")

    conn.execute(
        "UPDATE models SET currently_local=0, status='deleted', last_updated=? WHERE id=?",
        (now, row["id"]),
    )
    conn.execute(
        "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
        (row["id"], "delete", now, None),
    )
    conn.commit()
    conn.close()
    console.print("[green]✓ Registry updated (status=deleted).[/green]")


# ─── blacklist ────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("model")
@click.argument("reason", nargs=-1, required=False)
def blacklist(model, reason):
    """Set a model's status to blacklisted, record reason, and delete it.

    Partial name matching works: 'Peach' matches the full ollama model name.
    Works even if the model isn't in the registry yet (creates a new entry).
    REASON can be passed inline or left blank to be prompted.
    """
    config = load_config()
    conn = get_db(config)
    init_db(conn)
    now = now_iso()

    # Manual lookup so we can handle "not found" ourselves
    rows = conn.execute(
        """SELECT * FROM models
           WHERE display_name LIKE ? OR ollama_name LIKE ?
           ORDER BY display_name""",
        (f"%{model}%", f"%{model}%"),
    ).fetchall()

    if not rows:
        console.print(f"[yellow]'{model}' not found in registry.[/yellow]")
        if not click.confirm("Add it as a new blacklisted entry?", default=True):
            conn.close()
            return
        hf_repo, variant = parse_hf_repo_from_ollama(model)
        _bl_backends = ["ollama"] + get_gguf_backend_names(config) + ["comfyui"]
        backend = click.prompt(
            "Backend", type=click.Choice(_bl_backends), default="ollama"
        )
        conn.execute(
            """INSERT INTO models
               (display_name, hf_repo, variant, backend, source_type,
                ollama_name, currently_local, first_seen, last_updated)
               VALUES (?,?,?,?,?,?,0,?,?)""",
            (model, hf_repo, variant, backend, get_source_type(model),
             model if backend == "ollama" else None, now, now),
        )
        mid = _last_id(conn)
        row = conn.execute("SELECT * FROM models WHERE id=?", (mid,)).fetchone()
    elif len(rows) == 1:
        row = rows[0]
    else:
        console.print(f"[yellow]Multiple models match '{model}':[/yellow]")
        for i, r in enumerate(rows, 1):
            console.print(f"  {i}. {r['display_name']}  [{r['backend']}]")
        choice = click.prompt("Pick a number", type=click.IntRange(1, len(rows)))
        row = rows[choice - 1]

    reason_text = " ".join(reason) if reason else click.prompt("Reason for blacklisting")
    notes = row["notes"] or ""
    notes += f"\n[{now}] BLACKLISTED: {reason_text}"

    conn.execute(
        "UPDATE models SET status='blacklisted', notes=?, last_updated=? WHERE id=?",
        (notes, now, row["id"]),
    )
    conn.execute(
        "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
        (row["id"], "blacklist", now, reason_text),
    )

    # Auto-delete if currently local
    if row["currently_local"]:
        if row["backend"] == "ollama" and row["ollama_name"]:
            container = config["backends"]["ollama"]["docker_container"]
            result = subprocess.run(
                ["docker", "exec", container, "ollama", "rm", row["ollama_name"]],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                console.print("[green]✓ Removed from Ollama.[/green]")
            else:
                console.print(f"[yellow]⚠ Ollama delete failed: {result.stderr.strip()}[/yellow]")
        elif row["backend"] != "ollama" and row["file_path"]:
            p = Path(row["file_path"])
            if p.exists():
                p.unlink()
                console.print(f"[green]✓ Deleted file: {p}[/green]")
        conn.execute(
            "UPDATE models SET currently_local=0 WHERE id=?", (row["id"],)
        )
        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (row["id"], "delete", now, "auto-deleted on blacklist"),
        )

    conn.commit()
    conn.close()
    console.print(f"[red]✓ {row['display_name']} blacklisted.[/red]")


# ─── enrich ───────────────────────────────────────────────────────────────────

@cli.command()
def enrich():
    """Fetch HuggingFace metadata for all models with hf_repo set."""
    try:
        import requests
    except ImportError:
        console.print("[red]requests not installed. Run: pip install requests[/red]")
        return

    config = load_config()
    conn = get_db(config)
    now = now_iso()

    token_env = config["huggingface"]["token_env_var"]
    token = os.environ.get(token_env)
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    rows = conn.execute(
        "SELECT id, hf_repo, display_name FROM models WHERE hf_repo IS NOT NULL"
    ).fetchall()

    if not rows:
        console.print("No models with hf_repo to enrich.")
        conn.close()
        return

    console.print(f"Enriching [bold]{len(rows)}[/bold] model(s) from HuggingFace API...")

    def fetch_hf(repo_id, retries=3):
        url = f"https://huggingface.co/api/models/{repo_id}"
        for attempt in range(retries):
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    console.print(f"[yellow]Rate limited — waiting {wait}s...[/yellow]")
                    time.sleep(wait)
                    continue
                if resp.status_code == 404:
                    return None
                return None
            except requests.RequestException:
                time.sleep(1)
        return None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Enriching...", total=len(rows))

        for row in rows:
            progress.update(task, description=row["display_name"][:45])
            data = fetch_hf(row["hf_repo"])

            if data:
                # Attempt to extract param count from tags
                param_count = None
                for tag in data.get("tags", []):
                    m = re.match(r"^(\d+(?:\.\d+)?[BbMmKk])$", tag)
                    if m:
                        param_count = m.group(1).upper()
                        break

                # Architecture from model config
                architecture = None
                cfg = data.get("config")
                if isinstance(cfg, dict):
                    architecture = cfg.get("model_type")

                hf_downloads = data.get("downloads")
                hf_likes = data.get("likes")
                hf_last_modified = data.get("lastModified")

                conn.execute(
                    """UPDATE models SET
                       param_count      = COALESCE(?, param_count),
                       architecture     = COALESCE(?, architecture),
                       hf_downloads     = ?,
                       hf_likes         = ?,
                       hf_last_modified = ?,
                       last_updated     = ?
                       WHERE id=?""",
                    (param_count, architecture, hf_downloads,
                     hf_likes, hf_last_modified, now, row["id"]),
                )
                conn.execute(
                    "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                    (row["id"], "enrich", now, json.dumps({"hf_downloads": hf_downloads})),
                )

            progress.advance(task)
            time.sleep(0.5)

    conn.commit()

    # ── CivitAI enrichment phase ─────────────────────────────────────────────
    civitai_rows = conn.execute(
        "SELECT id, display_name, source_url, source_type FROM models"
        " WHERE source_type='comfyui_civitai' AND source_url IS NOT NULL"
    ).fetchall()

    if civitai_rows:
        civitai_cfg = config.get("civitai", {})
        civitai_token_env = civitai_cfg.get("token_env_var", "CIVITAI_API_KEY")
        civitai_token = os.environ.get(civitai_token_env)

        console.print(f"\nEnriching [bold]{len(civitai_rows)}[/bold] CivitAI model(s)...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("CivitAI...", total=len(civitai_rows))

            for row in civitai_rows:
                progress.update(task, description=row["display_name"][:45])
                version_id = parse_civitai_version_id(row["source_url"])
                if not version_id:
                    progress.advance(task)
                    continue

                url = f"https://civitai.com/api/v1/model-versions/{version_id}"
                params = {"token": civitai_token} if civitai_token else {}
                try:
                    resp = requests.get(url, params=params, timeout=15)
                    if resp.status_code != 200:
                        progress.advance(task)
                        time.sleep(0.5)
                        continue
                    data = resp.json()
                except Exception:
                    progress.advance(task)
                    time.sleep(0.5)
                    continue

                base_model = data.get("baseModel")
                trained_words = data.get("trainedWords") or []
                trigger_words_json = json.dumps(trained_words) if trained_words else None

                conn.execute(
                    """UPDATE models SET
                       base_model    = COALESCE(?, base_model),
                       trigger_words = COALESCE(?, trigger_words),
                       last_updated  = ?
                       WHERE id=?""",
                    (base_model, trigger_words_json, now, row["id"]),
                )
                conn.execute(
                    "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                    (row["id"], "enrich", now,
                     json.dumps({"base_model": base_model, "trigger_words": trained_words})),
                )
                progress.advance(task)
                time.sleep(0.5)

    conn.commit()
    conn.close()
    console.print("[green]✓ Enrichment complete.[/green]")


# ─── search ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("term")
def search(term):
    """Search local registry by name, hf_repo, or notes."""
    config = load_config()
    conn = get_db(config)

    rows = conn.execute(
        """SELECT * FROM models
           WHERE display_name LIKE ? OR hf_repo LIKE ? OR notes LIKE ?
           ORDER BY display_name""",
        (f"%{term}%", f"%{term}%", f"%{term}%"),
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
    cli()
