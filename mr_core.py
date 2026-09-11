#!/usr/bin/env python3
"""Engine layer for Model Registry (mr).

Pure logic + data access, with no CLI (click) or UI (rich) dependencies so it can
be reused by both the `mr.py` CLI and the `mr_mcp.py` MCP server.

Writes are gated: by default all write operations raise DestructiveOperation.
Call `set_writes_enabled(True)` (e.g. from the CLI, or from the MCP server when
started in read-write mode) to allow them. Destructive-to-disk operations still
require an explicit `confirm=True` argument.
"""

import difflib
import fnmatch
import json
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from huggingface_hub import hf_hub_download, list_repo_files, HfApi


# ─── Exceptions ───────────────────────────────────────────────────────────────


class MrError(Exception):
    """Base exception for Model Registry errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class ModelNotFound(MrError):
    """Model not found in registry."""

    def __init__(self, name: str, suggestions: list[str] | None = None):
        self.name = name
        self.suggestions = suggestions or []
        super().__init__(
            f"Model '{name}' not found", {"suggestions": self.suggestions}
        )


class AmbiguousModel(MrError):
    """Multiple models match the search term."""

    def __init__(self, name: str, matches: list[dict]):
        self.name = name
        self.matches = matches
        super().__init__(
            f"Multiple models match '{name}'", {"matches": self.matches}
        )


class DestructiveOperation(MrError):
    """Write operation blocked because writes are disabled or confirmation missing."""

    def __init__(self, message: str = "Operation not permitted."):
        super().__init__(message)


# ─── Write gating ─────────────────────────────────────────────────────────────

_writes_enabled = False


def set_writes_enabled(enabled: bool) -> None:
    """Enable or disable write operations (registry/disk mutations)."""
    global _writes_enabled
    _writes_enabled = bool(enabled)


def writes_enabled() -> bool:
    return _writes_enabled


def _require_writes() -> None:
    if not _writes_enabled:
        raise DestructiveOperation(
            "Writes are disabled in this mode. Start the server with --read-write "
            "to allow modifying the registry."
        )


def _require_confirm(confirm: bool) -> None:
    if not confirm:
        raise DestructiveOperation(
            "This operation is destructive. Pass confirm=True to proceed."
        )


# ─── Constants ────────────────────────────────────────────────────────────────


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

__version__ = "1.4.0"

SCRIPT_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "config.json"
CONFIG_EXAMPLE = SCRIPT_DIR / "config.example.json"

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


# ─── Helpers ──────────────────────────────────────────────────────────────────


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(log: list[str] | None, msg: str) -> None:
    if log is not None:
        log.append(msg)


# ─── Config ───────────────────────────────────────────────────────────────────


def load_config() -> dict:
    """Load config.json. Raises MrError if not found."""
    if not CONFIG_FILE.exists():
        raise MrError("config.json not found. Run 'mr init' first.")
    with open(CONFIG_FILE) as f:
        return json.load(f)


def get_db_path(config: dict) -> Path:
    p = config.get("registry_db", "")
    if p:
        return Path(p)
    return SCRIPT_DIR / "registry.db"


def get_db(config: dict) -> sqlite3.Connection:
    db_path = get_db_path(config)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize database schema with inline migrations."""
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


# ─── Model resolution ─────────────────────────────────────────────────────────


def resolve_model(conn: sqlite3.Connection, name: str, index: int = 0) -> sqlite3.Row:
    """Return a single models row matching name (partial on display_name/ollama_name).

    - Exactly one match → returns it.
    - Multiple matches and index provided → returns match at that index.
    - Multiple matches and index not provided → raises AmbiguousModel.
    - No matches → raises ModelNotFound with close-match suggestions.

    Args:
        index: 0-based index to select among ambiguous matches.
    """
    rows = conn.execute(
        """SELECT * FROM models
           WHERE display_name LIKE ? OR ollama_name LIKE ?
           ORDER BY display_name""",
        (f"%{name}%", f"%{name}%"),
    ).fetchall()

    if not rows:
        all_names = [
            r["display_name"]
            for r in conn.execute("SELECT display_name FROM models").fetchall()
        ]
        suggestions = difflib.get_close_matches(name, all_names, n=5, cutoff=0.4)
        raise ModelNotFound(name, suggestions)

    if len(rows) == 1:
        return rows[0]

    if index < 0 or index >= len(rows):
        raise MrError(f"Index {index} out of range for {len(rows)} matches")

    return rows[index]


def find_matching_rows(conn: sqlite3.Connection, name: str) -> list[sqlite3.Row]:
    """Return all models rows matching name (used by blacklist's manual lookup)."""
    return conn.execute(
        """SELECT * FROM models
           WHERE display_name LIKE ? OR ollama_name LIKE ?
           ORDER BY display_name""",
        (f"%{name}%", f"%{name}%"),
    ).fetchall()


# ─── Ollama parsing ───────────────────────────────────────────────────────────


def parse_ollama_size(size_str: str) -> float | None:
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


def parse_hf_repo_from_ollama(ollama_name: str) -> tuple[str | None, str | None]:
    """Extract (hf_repo, variant) from an ollama model name string."""
    m = re.match(
        r"(?:hf\.co|huggingface\.co)/([^:]+?)(?::(.+))?$", ollama_name, re.IGNORECASE
    )
    if m:
        return m.group(1).strip("/"), m.group(2)

    m = re.match(r"^([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?::(.+))?$", ollama_name)
    if m:
        return m.group(1), m.group(2)

    return None, None


def get_source_type(ollama_name: str) -> str:
    """Determine source_type from ollama model name."""
    if re.match(r"(?:hf\.co|huggingface\.co)/", ollama_name, re.IGNORECASE):
        return "ollama_hf"
    return "ollama_direct"


def get_gguf_backend_names(config: dict) -> list[str]:
    """Return all file-based GGUF backend names (everything except 'ollama'/'comfyui')."""
    return [
        name for name in config.get("backends", {})
        if name not in ("ollama", "comfyui")
    ]


def run_ollama_list(container: str) -> list[str]:
    result = subprocess.run(
        ["docker", "exec", container, "ollama", "list"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker exec failed: {result.stderr.strip()}")
    return result.stdout.strip().splitlines()


def parse_ollama_list_lines(lines: list[str]) -> list[dict]:
    """Parse ollama list output into list of dicts with ollama_name and size_gb."""
    models = []
    for line in lines[1:]:  # skip header
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        name = parts[0]
        size_gb = parse_ollama_size(parts[2] + " " + parts[3])
        models.append({"ollama_name": name, "size_gb": size_gb})
    return models


# ─── HuggingFace helpers ──────────────────────────────────────────────────────


def get_hf_metadata(hf_repo: str, hf_token: str | None) -> dict:
    """Fetch metadata (param_count, architecture, downloads, likes, last_modified)."""
    if not hf_repo:
        return {}
    try:
        hf_api = HfApi(token=hf_token)
        try:
            model_info = hf_api.model_info(repo_id=hf_repo)
        except Exception:
            return {}

        metadata = {}
        if (
            hasattr(model_info, "safetensors")
            and model_info.safetensors
            and isinstance(model_info.safetensors, dict)
        ):
            st = model_info.safetensors
            if st.get("total") is not None:
                metadata["param_count"] = st["total"]
            elif isinstance(st.get("parameters"), dict) and st["parameters"]:
                metadata["param_count"] = sum(
                    v for v in st["parameters"].values() if isinstance(v, (int, float))
                )

        card_data = getattr(model_info, "card_data", None) or getattr(
            model_info, "cardData", None
        )
        if card_data and isinstance(card_data, dict):
            if (
                "architectures" in card_data
                and isinstance(card_data["architectures"], list)
                and card_data["architectures"]
            ):
                metadata["architecture"] = card_data["architectures"][0]
            elif "model_name" in card_data:
                name_lower = str(card_data["model_name"]).lower()
                if "llama" in name_lower:
                    metadata["architecture"] = "Llama"
                elif "mistral" in name_lower:
                    metadata["architecture"] = "Mistral"

        metadata["hf_downloads"] = getattr(model_info, "downloads", None)
        metadata["hf_likes"] = getattr(model_info, "likes", None)
        last_mod = getattr(model_info, "lastModified", None) or getattr(
            model_info, "last_modified", None
        )
        metadata["hf_last_modified"] = (
            last_mod.isoformat() if hasattr(last_mod, "isoformat")
            else str(last_mod) if last_mod else None
        )

        return metadata
    except Exception:
        return {}


def get_hf_context_window(hf_repo: str, hf_token: str | None) -> int | None:
    """Fetch context window from HuggingFace model config (repo itself, then base_model)."""
    if not hf_repo:
        return None
    try:
        def _get_context_from_config(repo, token):
            try:
                config_path = hf_hub_download(
                    repo_id=repo, filename="config.json", token=token
                )
                with open(config_path, "r") as f:
                    config = json.load(f)
                for key in [
                    "max_position_embeddings",
                    "max_sequence_length",
                    "n_ctx",
                    "seq_length",
                    "max_seq_len",
                    "sliding_window",
                    "context_length",
                ]:
                    if key in config and isinstance(config[key], int):
                        return config[key]
            except Exception:
                return None

        ctx = _get_context_from_config(hf_repo, hf_token)
        if ctx is not None:
            return ctx

        try:
            api = HfApi(token=hf_token)
            info = api.model_info(hf_repo)
            card_data = getattr(info, "card_data", None) or getattr(
                info, "cardData", None
            )
            base_models = []
            if card_data and getattr(card_data, "base_model", None):
                bm = card_data.base_model
                if isinstance(bm, list):
                    base_models.extend(bm)
                elif isinstance(bm, str):
                    base_models.append(bm)
            for tag in getattr(info, "tags", []):
                if tag.startswith("base_model:"):
                    base_models.append(
                        tag.split(":", 1)[1].replace("quantized:", "")
                    )

            for bm in base_models:
                ctx = _get_context_from_config(bm, hf_token)
                if ctx is not None:
                    return ctx
        except Exception:
            pass

        return None
    except Exception:
        return None


# ─── llama.cpp helpers ────────────────────────────────────────────────────────


def parse_variant_from_filename(filename: str) -> str | None:
    """Extract quant variant from a .gguf filename, e.g. Q4_K_M, IQ4_XS, UD-Q8_K_XL."""
    m = re.search(
        r"[-.](i\d+-[A-Za-z0-9_]+|UD-[A-Za-z0-9_]+|Q\d[^.]*|IQ\d[^.]*|MXFP\d[^.]*|f16|f32|bf16)\.gguf$",
        filename,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    m = re.search(r"\.(Q\d[^.]*|IQ\d[^.]*|f16|f32|bf16)\.gguf$", filename, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def flatten_hf_subdir(directory: Path) -> None:
    """If directory contains exactly one subdirectory, move its contents up and remove it."""
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        subdir = subdirs[0]
        for item in subdir.iterdir():
            shutil.move(str(item), str(directory / item.name))
        subdir.rmdir()


def resolve_local_file_path(file_path: str, config: dict | None = None) -> Path | None:
    """Resolve a local file path, checking configured model directories. Path or None."""
    if not file_path:
        return None

    normalized_path_str = str(file_path).replace("\\", "/")
    p = Path(normalized_path_str)

    if p.exists():
        return p

    if config:
        for bname, bcfg in config.get("backends", {}).items():
            if not bcfg.get("enabled", False) or "model_dir" not in bcfg:
                continue
            model_dir = Path(bcfg["model_dir"])
            if not model_dir.exists():
                continue

            candidate = model_dir / p.name
            if candidate.exists():
                return candidate

            parts = [seg for seg in normalized_path_str.split("/") if seg and ":" not in seg]
            for i in range(len(parts)):
                candidate = model_dir.joinpath(*parts[i:])
                if candidate.exists():
                    return candidate

            try:
                matches = list(model_dir.rglob(p.name))
                if matches:
                    return matches[0]
            except Exception:
                pass

    return None


def parse_context_window_from_gguf(file_path, config: dict | None = None) -> int | None:
    """Extract context window from GGUF metadata, with multi-shard fallback."""
    resolved_path = resolve_local_file_path(file_path, config)
    if not resolved_path:
        return None
    resolved_path = str(resolved_path)

    ctx = _read_gguf_context_length(resolved_path)
    if ctx is not None:
        return ctx

    m = re.search(r"-(\d+)-of-(\d+)\.gguf$", resolved_path, re.IGNORECASE)
    if m:
        width = len(m.group(1))
        shard1_name = re.sub(
            r"-\d+-of-(\d+)\.gguf$",
            f"-{1:0{width}d}-of-\\1.gguf",
            resolved_path,
            flags=re.IGNORECASE,
        )
        if shard1_name != resolved_path and Path(shard1_name).exists():
            ctx = _read_gguf_context_length(shard1_name)
            if ctx is not None:
                return ctx

    return None


def _read_gguf_context_length(resolved_path: str) -> int | None:
    """Low-level: read context_length metadata from a single GGUF file path."""
    try:
        with open(resolved_path, "rb") as f:
            magic = f.read(4)
            if magic == b"GGUF":
                version = struct.unpack("<I", f.read(4))[0]
                if version in (1, 2, 3):
                    if version == 1:
                        _tensor_count = struct.unpack("<I", f.read(4))[0]
                        kv_count = struct.unpack("<I", f.read(4))[0]
                    else:
                        _tensor_count = struct.unpack("<Q", f.read(8))[0]
                        kv_count = struct.unpack("<Q", f.read(8))[0]

                    for _ in range(kv_count):
                        key_len = struct.unpack(
                            "<Q" if version >= 2 else "<I",
                            f.read(8 if version >= 2 else 4),
                        )[0]
                        if key_len > 256:
                            break
                        key = f.read(key_len).decode("utf-8", errors="ignore")
                        val_type = struct.unpack("<I", f.read(4))[0]

                        if key.endswith(".context_length") or key == "context_length":
                            if val_type in (0, 1):
                                return struct.unpack("<B", f.read(1))[0]
                            elif val_type in (2, 3):
                                return struct.unpack("<H", f.read(2))[0]
                            elif val_type in (4, 5):
                                return struct.unpack("<I", f.read(4))[0]
                            elif val_type in (10, 11):
                                return struct.unpack("<Q", f.read(8))[0]
                            break
                        else:
                            type_sizes = {
                                0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                                7: 1, 10: 8, 11: 8, 12: 8,
                            }
                            if val_type in type_sizes:
                                f.read(type_sizes[val_type])
                            elif val_type == 8:
                                slen = struct.unpack(
                                    "<Q" if version >= 2 else "<I",
                                    f.read(8 if version >= 2 else 4),
                                )[0]
                                f.read(slen)
                            elif val_type == 9:
                                atype = struct.unpack("<I", f.read(4))[0]
                                alen = struct.unpack(
                                    "<Q" if version >= 2 else "<I",
                                    f.read(8 if version >= 2 else 4),
                                )[0]
                                if atype == 8:
                                    for _ in range(alen):
                                        slen = struct.unpack(
                                            "<Q" if version >= 2 else "<I",
                                            f.read(8 if version >= 2 else 4),
                                        )[0]
                                        f.read(slen)
                                elif atype in type_sizes:
                                    f.read(type_sizes[atype] * alen)
                                else:
                                    break
                            else:
                                break
    except Exception:
        pass

    try:
        import gguf
        reader = gguf.GGUFReader(resolved_path)
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

    return None


# ─── CivitAI / AIR helpers ────────────────────────────────────────────────────


def parse_air_tag(ref: str) -> dict | None:
    """Parse an AIR URN for a CivitAI resource. Returns dict or None.

    urn:air:{ecosystem}:{type}:civitai:{model_id}@{version_id}
    e.g. urn:air:sdxl:checkpoint:civitai:2218365@2741096

    Also accepts bundled/multi-version tags where a secondary model version is
    appended with '+', e.g. civitai:133005@782002+695423 — the primary version
    (the first id after '@') is used.
    """
    s = re.sub(r"^(?:urn:)?(?:air:)?", "", ref.strip(), flags=re.IGNORECASE)
    m = re.match(
        r"^([^:]+):([^:]+):civitai:(\d+)@(\d+)(?:\+\d+)*$",
        s, re.IGNORECASE,
    )
    if not m:
        return None
    return {
        "ecosystem": m.group(1).lower(),
        "type": m.group(2).lower(),
        "model_id": m.group(3),
        "version_id": m.group(4),
    }


def parse_civitai_version_id(ref: str) -> str | None:
    """Extract CivitAI version ID from an AIR tag, URL, or 'civitai:<id>' shorthand."""
    air = parse_air_tag(ref)
    if air:
        return air["version_id"]
    m = re.match(r"^civitai:(\d+)$", ref, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(_CIVITAI_DOMAIN_RE + r"/api/download/models/(\d+)", ref)
    if m:
        return m.group(1)
    m = re.search(r"[?&]modelVersionId=(\d+)", ref)
    if m:
        return m.group(1)
    return None


def parse_civitai_model_id(ref: str) -> str | None:
    """Extract the model ID from a CivitAI browse URL (any domain variant)."""
    m = re.search(_CIVITAI_DOMAIN_RE + r"/models/(\d+)", ref, re.IGNORECASE)
    return m.group(1) if m else None


def fetch_civitai_model_info(
    model_id, token=None, host="civitai.com"
) -> tuple[str | None, str | None]:
    """Call CivitAI API v1 for a model. Returns (version_id, subdir_hint) or (None, None)."""
    url = f"https://{host}/api/v1/models/{model_id}"
    params = {}
    if token:
        params["token"] = token
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return None, None
        data = resp.json()
    except (requests.RequestException, json.JSONDecodeError):
        return None, None
    versions = data.get("modelVersions", [])
    version_id = str(versions[0]["id"]) if versions else None
    api_type = (data.get("type") or "").lower().replace(" ", "")
    subdir = CIVITAI_API_TYPE_TO_SUBDIR.get(api_type)
    return version_id, subdir


def civitai_source_url(ref: str, version_id, model_id=None) -> str:
    """Return the value to store as source_url for a CivitAI download."""
    if parse_air_tag(ref):
        return ref
    if model_id:
        return f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
    if re.search(_CIVITAI_DOMAIN_RE + r"/models/", ref, re.IGNORECASE):
        m = re.match(
            r"(https://" + _CIVITAI_DOMAIN_RE + r"/models/\d+(?:/[^?#]*)?)",
            ref,
            re.IGNORECASE,
        )
        base = m.group(1).rstrip("/") if m else "https://civitai.com/models"
        return f"{base}?modelVersionId={version_id}"
    return f"https://civitai.com/models?versionId={version_id}"


def get_model_link(row) -> str | None:
    """Get a browse URL for a model row."""
    if row["source_url"]:
        air = parse_air_tag(row["source_url"])
        if air:
            return f"https://civitai.com/models/{air['model_id']}?modelVersionId={air['version_id']}"
        return row["source_url"]
    if row["hf_repo"]:
        return f"https://huggingface.co/{row['hf_repo']}"
    return None


def _sanitize_filename(name: str) -> str:
    """Strip path components and characters unsafe for filenames."""
    name = name.strip().replace("\\", "/").split("/")[-1]
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip().lstrip(".")
    return name


# ─── ComfyUI scanning ─────────────────────────────────────────────────────────


def scan_comfyui(config: dict, conn: sqlite3.Connection) -> tuple[int, int]:
    """Scan all immediate subdirs of the ComfyUI models base_dir. Returns (added, updated)."""
    comfy_cfg = config["backends"].get("comfyui", {})
    base_dir = Path(comfy_cfg.get("base_dir", ""))
    extensions = comfy_cfg.get("extensions", [".safetensors", ".ckpt", ".pt", ".pth", ".bin"])
    now = now_iso()
    added = updated = 0

    if not base_dir.exists():
        raise MrError(f"ComfyUI base_dir does not exist: {base_dir}")

    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
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
                update_fields = {}
                if existing["currently_local"] != 1:
                    update_fields["currently_local"] = 1
                if (existing["size_gb"] or 0) != size_gb:
                    update_fields["size_gb"] = size_gb
                if update_fields:
                    update_fields["last_updated"] = now
                    set_clauses = [f"{k}=?" for k in update_fields.keys()]
                    params = list(update_fields.values()) + [existing["id"]]
                    conn.execute(
                        f"UPDATE models SET {', '.join(set_clauses)} WHERE id=?",
                        params,
                    )
                    event_detail = {
                        k: v for k, v in update_fields.items() if k != "last_updated"
                    }
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
                    (f.stem, variant, "comfyui", "comfyui_unknown", fpath, size_gb, now, now),
                )
                mid = _last_id(conn)
                conn.execute(
                    "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                    (mid, "scan_added", now, json.dumps({"file_path": fpath})),
                )
                added += 1

    # Mark disappeared files as not-local
    if seen_paths:
        placeholders = ",".join("?" * len(seen_paths))
        conn.execute(
            f"""UPDATE models SET currently_local=0, last_updated=?
                WHERE backend='comfyui' AND file_path NOT IN ({placeholders})""",
            (now, *seen_paths),
        )
    else:
        conn.execute(
            "UPDATE models SET currently_local=0, last_updated=? WHERE backend='comfyui'",
            (now,),
        )

    return added, updated


# ─── File deletion (shared by delete/blacklist) ───────────────────────────────


def _delete_model_files(row, config: dict, conn: sqlite3.Connection) -> bool:
    """Delete a model's files from disk. Returns True if removal succeeded."""
    if row["backend"] == "ollama":
        container = config["backends"]["ollama"].get("docker_container", "ollama")
        result = subprocess.run(
            ["docker", "exec", container, "ollama", "rm", row["ollama_name"]],
            capture_output=True, text=True,
        )
        return result.returncode == 0

    if not row["file_path"]:
        return False

    resolved = resolve_local_file_path(row["file_path"], config) or Path(row["file_path"])
    if not resolved.exists():
        return False

    resolved.unlink()

    m = re.match(r"^(.+?)-(\d+)-of-(\d+)\.gguf$", resolved.name, re.IGNORECASE)
    if m:
        shard_re = re.compile(
            rf"^{re.escape(m.group(1))}-\d+-of-{m.group(3)}\.gguf$", re.IGNORECASE
        )
        for sibling in resolved.parent.iterdir():
            if sibling != resolved and sibling.is_file() and shard_re.match(sibling.name):
                sibling.unlink()

    parent = resolved.parent
    try:
        parent_res = parent.resolve()
    except OSError:
        parent_res = parent.absolute()

    protected = False
    inside_backend_root = False
    for bname, bcfg in config.get("backends", {}).items():
        if not bcfg.get("enabled", False):
            continue
        d = bcfg.get("model_dir") or bcfg.get("base_dir")
        if not d:
            continue
        try:
            root = Path(d).resolve()
        except OSError:
            continue
        if parent_res == root:
            protected = True
        elif parent_res.is_relative_to(root):
            inside_backend_root = True
            if bname == "comfyui" and root == parent_res.parent:
                protected = True  # e.g. <base>/checkpoints holds many models

    if protected or not inside_backend_root:
        return True

    others = conn.execute(
        "SELECT file_path FROM models WHERE id != ? AND file_path IS NOT NULL",
        (row["id"],),
    ).fetchall()
    for other in others:
        try:
            if Path(other["file_path"]).resolve().is_relative_to(parent_res):
                return True  # another registered model lives in this directory
        except (OSError, ValueError):
            continue

    shutil.rmtree(parent)
    return True


# ─── Engine functions: reads ──────────────────────────────────────────────────


def engine_backends(config: dict | None = None) -> list[dict]:
    """List configured backend names and enabled/disabled status."""
    if config is None:
        config = load_config()
    return [
        {
            "name": name,
            "status": "enabled" if cfg.get("enabled", False) else "disabled",
        }
        for name, cfg in config.get("backends", {}).items()
    ]


def engine_list(
    config: dict | None = None,
    backend: str | None = None,
    status: str | None = None,
    unrated: bool = False,
    show_all: bool = False,
    deleted: bool = False,
) -> list[dict]:
    """List models in the registry (by default local, non-blacklisted)."""
    if config is None:
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
    return [dict(r) for r in rows]


def engine_show(config: dict | None = None, name: str | None = None) -> dict:
    """Show full details (incl. tags, notes, link, recent events) for a model."""
    if config is None:
        config = load_config()
    if name is None:
        raise MrError("name is required")

    conn = get_db(config)
    init_db(conn)
    try:
        row = resolve_model(conn, name)
        result = dict(row)

        tags = []
        if row["tags"]:
            try:
                tags = json.loads(row["tags"])
            except json.JSONDecodeError:
                tags = [row["tags"]]
        result["tags"] = tags

        result["notes_list"] = (
            row["notes"].strip().splitlines() if row["notes"] else []
        )
        result["link"] = get_model_link(row)

        events = conn.execute(
            "SELECT * FROM events WHERE model_id=? ORDER BY timestamp DESC LIMIT 10",
            (row["id"],),
        ).fetchall()
        result["recent_events"] = [dict(e) for e in events]

        return result
    finally:
        conn.close()


def engine_report(config: dict | None = None) -> dict:
    """Summary: totals, GB by backend, unrated/blacklisted lists, duplicates."""
    if config is None:
        config = load_config()
    conn = get_db(config)
    init_db(conn)

    total = _scalar(conn, "SELECT COUNT(*) FROM models")
    total_gb = _scalar(
        conn, "SELECT COALESCE(SUM(size_gb), 0) FROM models WHERE currently_local=1"
    )

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

    conn.close()

    return {
        "total": total,
        "total_gb": round(total_gb, 1),
        "by_backend": [dict(r) for r in by_backend],
        "by_status": [dict(r) for r in by_status],
        "unrated": [dict(r) for r in unrated],
        "blacklisted": [dict(r) for r in blacklisted],
        "duplicates": [dict(r) for r in dupes],
    }


def engine_search(
    config: dict | None = None,
    term: str | None = None,
) -> list[dict]:
    """Search models by name, hf_repo, notes, or tags."""
    if config is None:
        config = load_config()
    if term is None:
        raise MrError("term is required")

    conn = get_db(config)
    init_db(conn)
    rows = conn.execute(
        """SELECT * FROM models
           WHERE display_name LIKE ? OR hf_repo LIKE ? OR notes LIKE ? OR tags LIKE ?
           ORDER BY display_name""",
        (f"%{term}%", f"%{term}%", f"%{term}%", f"%{term}%"),
    ).fetchall()
    conn.close()

    results = [dict(r) for r in rows]
    for r in results:
        r["link"] = get_model_link(r)
    return results


# ─── Engine functions: writes ─────────────────────────────────────────────────


def engine_scan(
    config: dict | None = None,
    log: list[str] | None = None,
) -> dict:
    """Scan Ollama, llama.cpp, and ComfyUI backends; update the registry."""
    _require_writes()
    if config is None:
        config = load_config()
    conn = get_db(config)
    init_db(conn)
    now = now_iso()
    added = updated = 0

    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_cfg = config["backends"].get("ollama", {})
    hf_token = os.environ.get(
        config.get("huggingface", {}).get("token_env_var", "")
    ) or None

    if ollama_cfg.get("enabled", False):
        container = ollama_cfg.get("docker_container", "ollama")
        _log(log, f"Scanning Ollama (container: {container})...")
        try:
            lines = run_ollama_list(container)
            ollama_models = parse_ollama_list_lines(lines)
            _log(log, f"  Found {len(ollama_models)} model(s) in Ollama.")

            seen_hf_repos = {}

            for om in ollama_models:
                oname = om["ollama_name"]
                size_gb = om["size_gb"]
                hf_repo, variant = parse_hf_repo_from_ollama(oname)
                source_type = get_source_type(oname)
                if hf_repo and hf_repo in seen_hf_repos:
                    _log(
                        log,
                        f"  Skipping duplicate: {oname} (same hf_repo as {seen_hf_repos[hf_repo]})",
                    )
                    continue
                if hf_repo:
                    seen_hf_repos[hf_repo] = oname

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

                context_window = None
                if hf_repo and (existing is None or existing["context_window"] is None):
                    context_window = get_hf_context_window(hf_repo, hf_token)

                if existing:
                    update_fields = {}
                    if existing["currently_local"] != 1:
                        update_fields["currently_local"] = 1
                    if (existing["size_gb"] or 0) != size_gb:
                        update_fields["size_gb"] = size_gb
                    if existing["ollama_name"] != oname:
                        update_fields["ollama_name"] = oname
                    if context_window is not None and existing["context_window"] != context_window:
                        update_fields["context_window"] = context_window

                    if update_fields:
                        update_fields["last_updated"] = now
                        set_clauses = [f"{k}=?" for k in update_fields.keys()]
                        params = list(update_fields.values()) + [existing["id"]]
                        conn.execute(
                            f"UPDATE models SET {', '.join(set_clauses)} WHERE id=?",
                            params,
                        )
                        event_detail = {
                            k: v for k, v in update_fields.items() if k != "last_updated"
                        }
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
                    _log(log, f"  Marked not-local: {row['ollama_name']}")

        except RuntimeError as e:
            _log(log, f"[red]Ollama scan failed: {e}[/red]")
        except FileNotFoundError:
            _log(log, "[red]'docker' command not found. Is Docker installed and on PATH?[/red]")

    # ── GGUF file backends ──────────────────────────────────────────────────
    for bname in get_gguf_backend_names(config):
        bcfg = config["backends"][bname]
        if not bcfg.get("enabled", False):
            continue
        model_dir = Path(bcfg.get("model_dir", ""))
        extensions = bcfg.get("extensions", [".gguf"])
        _log(log, f"\nScanning {bname} models in {model_dir}...")
        if not model_dir.exists():
            _log(log, f"[red]{bname} model_dir does not exist: {model_dir}[/red]")
            continue

        subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
        top_level_files = []
        for ext in extensions:
            top_level_files.extend(model_dir.glob(f"*{ext}"))
        top_level_files = [f for f in top_level_files if f.parent == model_dir]

        total_files = 0
        seen_paths = set()

        for subdir in sorted(subdirs):
            subdir_gguf_files = []
            for ext in extensions:
                subdir_gguf_files.extend(subdir.glob(f"*{ext}"))

            if not subdir_gguf_files:
                continue

            total_files += len(subdir_gguf_files)
            variant = subdir.name
            main_file = max(subdir_gguf_files, key=lambda f: f.stat().st_size)
            fpath = str(main_file)
            seen_paths.add(fpath)
            size_gb = round(sum(f.stat().st_size for f in subdir_gguf_files) / (1024 ** 3), 4)

            context_window = parse_context_window_from_gguf(fpath, config)

            existing = conn.execute(
                "SELECT * FROM models WHERE file_path=?", (fpath,)
            ).fetchone()

            if existing:
                update_fields = {}
                if existing["currently_local"] != 1:
                    update_fields["currently_local"] = 1
                if (existing["size_gb"] or 0) != size_gb:
                    update_fields["size_gb"] = size_gb
                if existing["variant"] != variant:
                    update_fields["variant"] = variant
                if context_window is not None and existing["context_window"] != context_window:
                    update_fields["context_window"] = context_window

                if update_fields:
                    update_fields["last_updated"] = now
                    set_clauses = [f"{k}=?" for k in update_fields.keys()]
                    params = list(update_fields.values()) + [existing["id"]]
                    conn.execute(
                        f"UPDATE models SET {', '.join(set_clauses)} WHERE id=?",
                        params,
                    )
                    event_detail = {
                        k: v for k, v in update_fields.items() if k != "last_updated"
                    }
                    conn.execute(
                        "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                        (existing["id"], "scan_updated", now, json.dumps(event_detail)),
                    )
                    updated += 1
            else:
                conn.execute(
                    """INSERT INTO models
                       (display_name, variant, backend, source_type,
                        file_path, size_gb, context_window, currently_local, first_seen, last_updated)
                       VALUES (?,?,?,?,?,?,?,1,?,?)""",
                    (subdir.name, variant, bname, bname, fpath, size_gb, context_window, now, now),
                )
                mid = _last_id(conn)
                conn.execute(
                    "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                    (mid, "scan_added", now, json.dumps({"file_path": fpath, "subdir": subdir.name})),
                )
                added += 1

        for f in sorted(top_level_files):
            fpath = str(f)
            seen_paths.add(fpath)
            size_gb = round(f.stat().st_size / (1024 ** 3), 4)
            variant = parse_variant_from_filename(f.name)
            context_window = parse_context_window_from_gguf(fpath, config)

            existing = conn.execute(
                "SELECT * FROM models WHERE file_path=?", (fpath,)
            ).fetchone()

            if existing:
                update_fields = {}
                if existing["currently_local"] != 1:
                    update_fields["currently_local"] = 1
                if (existing["size_gb"] or 0) != size_gb:
                    update_fields["size_gb"] = size_gb
                if context_window is not None and existing["context_window"] != context_window:
                    update_fields["context_window"] = context_window

                if update_fields:
                    update_fields["last_updated"] = now
                    set_clauses = [f"{k}=?" for k in update_fields.keys()]
                    params = list(update_fields.values()) + [existing["id"]]
                    conn.execute(
                        f"UPDATE models SET {', '.join(set_clauses)} WHERE id=?",
                        params,
                    )
                    event_detail = {
                        k: v for k, v in update_fields.items() if k != "last_updated"
                    }
                    conn.execute(
                        "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                        (existing["id"], "scan_updated", now, json.dumps(event_detail)),
                    )
                    updated += 1
            else:
                display_name = f.stem
                conn.execute(
                    """INSERT INTO models
                       (display_name, variant, backend, source_type,
                        file_path, size_gb, context_window, currently_local, first_seen, last_updated)
                       VALUES (?,?,?,?,?,?,?,1,?,?)""",
                    (display_name, variant, bname, bname, fpath, size_gb, context_window, now, now),
                )
                mid = _last_id(conn)
                conn.execute(
                    "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                    (mid, "scan_added", now, json.dumps({"file_path": fpath})),
                )
                added += 1

        _log(
            log,
            f"  Found {total_files + len(top_level_files)} GGUF file(s) in "
            f"{len(subdirs) + len(top_level_files)} model(s).",
        )

        db_rows = conn.execute(
            "SELECT id, file_path FROM models WHERE backend=? AND currently_local=1", (bname,)
        ).fetchall()
        for row in db_rows:
            if row["file_path"] not in seen_paths:
                conn.execute(
                    "UPDATE models SET currently_local=0, last_updated=? WHERE id=?",
                    (now, row["id"]),
                )
                _log(log, f"  Marked not-local: {row['file_path']}")

    # ── ComfyUI ─────────────────────────────────────────────────────────────
    comfy_cfg = config["backends"].get("comfyui", {})
    if comfy_cfg.get("enabled", False):
        base_dir = comfy_cfg.get("base_dir", "")
        _log(log, f"\nScanning ComfyUI models in {base_dir}...")
        c_added, c_updated = scan_comfyui(config, conn)
        added += c_added
        updated += c_updated
        _log(log, f"  ComfyUI: {c_added} added, {c_updated} updated")

    conn.commit()
    conn.close()
    return {"added": added, "updated": updated, "timestamp": now}


def engine_enrich(
    config: dict | None = None,
    enrich_all: bool = False,
    log: list[str] | None = None,
) -> dict:
    """Fetch additional metadata (context, params, architecture, HF/CivitAI stats)."""
    _require_writes()
    if config is None:
        config = load_config()
    conn = get_db(config)
    init_db(conn)
    now = now_iso()
    hf_token = os.environ.get(
        config.get("huggingface", {}).get("token_env_var", "")
    ) or None

    where_clause = "WHERE 1=1"
    if not enrich_all:
        where_clause = (
            "WHERE currently_local=1 AND "
            "(status IS NULL OR status NOT IN ('deleted', 'blacklisted'))"
        )

    rows = conn.execute(f"""
        SELECT id, display_name, hf_repo, backend, file_path, ollama_name,
               context_window, param_count, architecture, hf_downloads, hf_likes, hf_last_modified
        FROM models
        {where_clause}
        ORDER BY display_name
    """).fetchall()

    if not rows:
        conn.close()
        return {"updated": 0, "civitai_updated": 0, "timestamp": now}

    _log(log, f"Enriching {len(rows)} model(s) from HuggingFace Hub...")

    updated_count = 0
    for row in rows:
        update_fields = {}

        if row["file_path"] and row["context_window"] is None:
            ctx = parse_context_window_from_gguf(row["file_path"], config)
            if ctx is not None:
                update_fields["context_window"] = ctx

        if row["hf_repo"]:
            if row["context_window"] is None and "context_window" not in update_fields:
                ctx = get_hf_context_window(row["hf_repo"], hf_token)
                if ctx is not None:
                    update_fields["context_window"] = ctx

            if row["hf_repo"] and any(
                row[k] is None
                for k in ("param_count", "architecture", "hf_downloads", "hf_likes", "hf_last_modified")
            ):
                meta = get_hf_metadata(row["hf_repo"], hf_token)
                if meta:
                    for k, v in meta.items():
                        if v is not None and row[k] is None:
                            update_fields[k] = v

        if update_fields:
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
            conn.commit()
            updated_count += 1
            _log(log, f"  Enriched {row['display_name']}")

    # ── CivitAI enrichment ──────────────────────────────────────────────────
    civitai_where = "WHERE source_type='comfyui_civitai' AND source_url IS NOT NULL"
    if not enrich_all:
        civitai_where += (
            " AND currently_local=1 AND "
            "(status IS NULL OR status NOT IN ('deleted', 'blacklisted'))"
        )
    civitai_rows = conn.execute(
        f"SELECT id, display_name, source_url, source_type, base_model, trigger_words FROM models {civitai_where}"
    ).fetchall()

    civitai_updated_count = 0
    if civitai_rows:
        civitai_cfg = config.get("civitai", {})
        token_env = civitai_cfg.get("token_env_var", "CIVITAI_API_KEY")
        civitai_token = os.environ.get(token_env)

        _log(log, f"Enriching {len(civitai_rows)} CivitAI model(s)...")

        for row in civitai_rows:
            version_id = parse_civitai_version_id(row["source_url"])
            if not version_id:
                continue

            url = f"https://civitai.com/api/v1/model-versions/{version_id}"
            params = {"token": civitai_token} if civitai_token else {}
            try:
                resp = requests.get(url, params=params, timeout=15)
                if resp.status_code != 200:
                    time.sleep(0.5)
                    continue
                data = resp.json()
            except (requests.RequestException, json.JSONDecodeError):
                time.sleep(0.5)
                continue

            base_model = data.get("baseModel")
            trained_words = data.get("trainedWords") or []
            trigger_words_json = json.dumps(trained_words) if trained_words else None

            c_updates = {}
            if base_model and not row["base_model"]:
                c_updates["base_model"] = base_model
            if trigger_words_json and not row["trigger_words"]:
                c_updates["trigger_words"] = trigger_words_json

            if c_updates:
                c_updates["last_updated"] = now
                set_clauses = [f"{k}=?" for k in c_updates.keys()]
                params = list(c_updates.values()) + [row["id"]]
                conn.execute(
                    f"UPDATE models SET {', '.join(set_clauses)} WHERE id=?",
                    params,
                )
                conn.execute(
                    "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                    (row["id"], "enrich_updated", now, json.dumps(c_updates)),
                )
                civitai_updated_count += 1
                _log(log, f"  Enriched CivitAI: {row['display_name']}")

            time.sleep(0.5)

        conn.commit()

    conn.close()
    return {
        "updated": updated_count,
        "civitai_updated": civitai_updated_count,
        "timestamp": now,
    }


def engine_rate(
    config: dict | None = None,
    name: str | None = None,
    rating: int | None = None,
    status: str | None = None,
    note: str | None = None,
) -> dict:
    """Rate a model (1-5) and set its status, with optional note."""
    _require_writes()
    if config is None:
        config = load_config()
    if name is None or rating is None:
        raise MrError("name and rating are required")
    if rating < 1 or rating > 5:
        raise MrError("Rating must be between 1 and 5")

    valid = ["active", "unrated", "blacklisted", "deleted", "on_hold", "testing", "keep", "favorite"]
    if status and status not in valid:
        raise MrError(f"Invalid status. Must be one of: {', '.join(valid)}")

    conn = get_db(config)
    init_db(conn)
    now = now_iso()
    try:
        row = resolve_model(conn, name)

        notes = row["notes"] or ""
        if note:
            notes += f"\n[{now}] {note}"

        conn.execute(
            "UPDATE models SET rating=?, status=?, notes=?, last_updated=? WHERE id=?",
            (rating, status, notes, now, row["id"]),
        )
        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (row["id"], "rate", now, json.dumps({"rating": rating, "status": status})),
        )
        if note:
            conn.execute(
                "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                (row["id"], "note", now, note),
            )
        conn.commit()
        return {
            "status": "rated",
            "display_name": row["display_name"],
            "rating": rating,
            "status_value": status,
            "timestamp": now,
        }
    finally:
        conn.close()


def engine_status(
    config: dict | None = None,
    name: str | None = None,
    status: str | None = None,
) -> dict:
    """Set a model's status without requiring a rating."""
    _require_writes()
    if config is None:
        config = load_config()
    if name is None or status is None:
        raise MrError("name and status are required")

    valid = ["active", "unrated", "blacklisted", "deleted", "on_hold", "testing", "keep", "favorite"]
    if status not in valid:
        raise MrError(f"Invalid status. Must be one of: {', '.join(valid)}")

    conn = get_db(config)
    init_db(conn)
    now = now_iso()
    try:
        row = resolve_model(conn, name)
        conn.execute(
            "UPDATE models SET status=?, last_updated=? WHERE id=?",
            (status, now, row["id"]),
        )
        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (row["id"], "setstatus", now, json.dumps({"status": status})),
        )
        conn.commit()
        return {
            "status": "updated",
            "display_name": row["display_name"],
            "new_status": status,
            "timestamp": now,
        }
    finally:
        conn.close()


def engine_note(
    config: dict | None = None,
    name: str | None = None,
    text: str | None = None,
) -> dict:
    """Append a timestamped note to a model."""
    _require_writes()
    if config is None:
        config = load_config()
    if name is None or text is None:
        raise MrError("name and text are required")

    conn = get_db(config)
    init_db(conn)
    now = now_iso()
    try:
        row = resolve_model(conn, name)
        notes = row["notes"] or ""
        notes += f"\n[{now}] {text}"
        conn.execute(
            "UPDATE models SET notes=?, last_updated=? WHERE id=?",
            (notes, now, row["id"]),
        )
        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (row["id"], "note", now, text),
        )
        conn.commit()
        return {
            "status": "note_added",
            "display_name": row["display_name"],
            "note": text,
            "timestamp": now,
        }
    finally:
        conn.close()


def engine_touch(
    config: dict | None = None,
    name: str | None = None,
) -> dict:
    """Update last_used to now (use when you ran a model outside this tool)."""
    _require_writes()
    if config is None:
        config = load_config()
    if name is None:
        raise MrError("name is required")

    conn = get_db(config)
    init_db(conn)
    now = now_iso()
    try:
        row = resolve_model(conn, name)
        conn.execute(
            "UPDATE models SET last_used=?, last_updated=? WHERE id=?",
            (now, now, row["id"]),
        )
        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (row["id"], "touch", now, None),
        )
        conn.commit()
        return {
            "status": "touched",
            "display_name": row["display_name"],
            "timestamp": now,
        }
    finally:
        conn.close()


def engine_tag(
    config: dict | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Add tags to a model."""
    _require_writes()
    if config is None:
        config = load_config()
    if name is None or not tags:
        raise MrError("name and at least one tag are required")

    conn = get_db(config)
    init_db(conn)
    now = now_iso()
    try:
        row = resolve_model(conn, name)

        current = []
        if row["tags"]:
            try:
                current = json.loads(row["tags"])
            except json.JSONDecodeError:
                current = [row["tags"]]

        for t in tags:
            if t not in current:
                current.append(t)

        conn.execute(
            "UPDATE models SET tags=?, last_updated=? WHERE id=?",
            (json.dumps(current), now, row["id"]),
        )
        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (row["id"], "tag", now, json.dumps(current)),
        )
        conn.commit()
        return {
            "status": "tags_updated",
            "display_name": row["display_name"],
            "tags": current,
            "timestamp": now,
        }
    finally:
        conn.close()


def engine_untag(
    config: dict | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Remove tags from a model. If tags is empty, removes all tags."""
    _require_writes()
    if config is None:
        config = load_config()
    if name is None:
        raise MrError("name is required")

    conn = get_db(config)
    init_db(conn)
    now = now_iso()
    try:
        row = resolve_model(conn, name)

        current = []
        if row["tags"]:
            try:
                current = json.loads(row["tags"])
            except json.JSONDecodeError:
                current = [row["tags"]]

        if tags:
            for t in tags:
                if t in current:
                    current.remove(t)
        else:
            current = []

        conn.execute(
            "UPDATE models SET tags=?, last_updated=? WHERE id=?",
            (json.dumps(current) if current else None, now, row["id"]),
        )
        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (row["id"], "untag", now, json.dumps(current) if current else None),
        )
        conn.commit()
        return {
            "status": "tags_updated",
            "display_name": row["display_name"],
            "tags": current,
            "timestamp": now,
        }
    finally:
        conn.close()


def engine_delete(
    config: dict | None = None,
    name: str | None = None,
    confirm: bool = False,
) -> dict:
    """Delete a model from Ollama/disk. Keeps DB record, sets status='deleted'."""
    _require_writes()
    _require_confirm(confirm)
    if config is None:
        config = load_config()
    if name is None:
        raise MrError("name is required")

    conn = get_db(config)
    init_db(conn)
    now = now_iso()
    try:
        row = resolve_model(conn, name)

        removed = _delete_model_files(row, config, conn)
        if row["backend"] == "ollama" and not removed:
            raise MrError("Model still present in Ollama; registry not updated.")

        conn.execute(
            "UPDATE models SET currently_local=0, status='deleted', last_updated=? WHERE id=?",
            (now, row["id"]),
        )
        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (row["id"], "delete", now, None),
        )
        conn.commit()
        return {
            "status": "deleted",
            "display_name": row["display_name"],
            "file_removed": removed,
            "timestamp": now,
        }
    finally:
        conn.close()


def engine_remove(
    config: dict | None = None,
    name: str | None = None,
    confirm: bool = False,
) -> dict:
    """Remove a model from the registry entirely (hard delete of DB row)."""
    _require_writes()
    _require_confirm(confirm)
    if config is None:
        config = load_config()
    if name is None:
        raise MrError("name is required")

    conn = get_db(config)
    init_db(conn)
    now = now_iso()
    try:
        row = resolve_model(conn, name)
        conn.execute("DELETE FROM events WHERE model_id=?", (row["id"],))
        conn.execute("DELETE FROM models WHERE id=?", (row["id"],))
        conn.commit()
        return {
            "status": "removed",
            "display_name": row["display_name"],
            "timestamp": now,
        }
    finally:
        conn.close()


def engine_removeall(
    config: dict | None = None,
    dry_run: bool = False,
    confirm: bool = False,
) -> dict:
    """Purge all models with status='deleted' from the registry (hard delete)."""
    _require_writes()
    if config is None:
        config = load_config()

    conn = get_db(config)
    init_db(conn)
    now = now_iso()
    try:
        deleted = conn.execute(
            "SELECT * FROM models WHERE status='deleted' ORDER BY display_name"
        ).fetchall()

        if not deleted:
            return {"status": "no_deleted_models", "count": 0, "timestamp": now}

        if dry_run:
            return {
                "status": "dry_run",
                "count": len(deleted),
                "models": [dict(r) for r in deleted],
                "timestamp": now,
            }

        _require_confirm(confirm)

        removed = 0
        for row in deleted:
            conn.execute("DELETE FROM events WHERE model_id=?", (row["id"],))
            conn.execute("DELETE FROM models WHERE id=?", (row["id"],))
            conn.execute(
                "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (NULL, 'purge', ?, ?)",
                (now, f"purged: {row['display_name']} [{row['backend']}]"),
            )
            removed += 1

        conn.commit()
        return {"status": "purged", "count": removed, "timestamp": now}
    finally:
        conn.close()


def engine_copy(
    config: dict | None = None,
    src_backend: str | None = None,
    dst_backend: str | None = None,
    model_name: str | None = None,
    confirm: bool = False,
) -> dict:
    """Copy a model from one GGUF backend to another."""
    _require_writes()
    _require_confirm(confirm)
    if config is None:
        config = load_config()
    if not src_backend or not dst_backend or not model_name:
        raise MrError("src_backend, dst_backend, and model_name are required")

    conn = get_db(config)
    init_db(conn)
    now = now_iso()
    try:
        rows = conn.execute(
            """SELECT * FROM models
               WHERE display_name LIKE ? AND backend=?
               ORDER BY display_name""",
            (f"%{model_name}%", src_backend),
        ).fetchall()

        if not rows:
            raise MrError(f"Model '{model_name}' not found in backend '{src_backend}'")
        if len(rows) > 1:
            raise AmbiguousModel(model_name, [dict(r) for r in rows])
        row = rows[0]

        if row["backend"] == "ollama":
            raise MrError("Cannot copy from Ollama - use 'mr pull' instead.")
        if row["backend"] == "comfyui":
            raise MrError("Cannot copy from ComfyUI - only GGUF backends are supported.")
        if dst_backend == "ollama":
            raise MrError("Cannot copy to Ollama - use 'mr pull' instead.")
        if dst_backend == "comfyui":
            raise MrError("Cannot copy to ComfyUI - only GGUF backends are supported.")

        src_path = Path(row["file_path"]) if row["file_path"] else None
        if not src_path or not src_path.exists():
            raise MrError(f"Source file not found: {row['file_path']}")

        dst_cfg = config["backends"].get(dst_backend, {})
        dst_model_dir = Path(dst_cfg.get("model_dir", ""))
        if not dst_model_dir.exists():
            raise MrError(f"Destination directory does not exist: {dst_model_dir}")

        repo_name = src_path.parent.name
        dst_dir = dst_model_dir / repo_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        for f in src_path.parent.glob("*"):
            if f.is_file():
                shutil.copy2(f, dst_dir / f.name)

        dst_files = list(dst_dir.glob("*.gguf"))
        if not dst_files:
            raise MrError("No GGUF files found in destination directory")

        main_file = max(dst_files, key=lambda f: f.stat().st_size)
        main_path = Path(main_file)
        total_size = sum(f.stat().st_size for f in dst_files)
        size_gb = round(total_size / (1024 ** 3), 2)
        variant = parse_variant_from_filename(main_path.name)

        conn.execute(
            """INSERT INTO models
               (display_name, hf_repo, variant, backend, source_type,
                file_path, size_gb, currently_local, first_seen, last_updated)
               VALUES (?,?,?,?,?,?,?,1,?,?)""",
            (dst_dir.name, row["hf_repo"], variant, dst_backend, dst_backend,
             str(main_path), size_gb, now, now),
        )
        mid = _last_id(conn)
        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (mid, "copy", now, json.dumps({
                "from": f"{src_backend}:{row['display_name']}",
                "to": f"{dst_backend}:{dst_dir.name}",
            })),
        )
        conn.commit()
        return {
            "status": "copied",
            "display_name": dst_dir.name,
            "backend": dst_backend,
            "size_gb": size_gb,
            "timestamp": now,
        }
    finally:
        conn.close()


def engine_rename(
    config: dict | None = None,
    name: str | None = None,
    new_name: str | None = None,
    confirm: bool = False,
) -> dict:
    """Rename the directory containing a model, keeping the file name unchanged."""
    _require_writes()
    _require_confirm(confirm)
    if config is None:
        config = load_config()
    if name is None or new_name is None:
        raise MrError("name and new_name are required")

    conn = get_db(config)
    init_db(conn)
    now = now_iso()
    try:
        row = resolve_model(conn, name)

        if row["backend"] == "ollama":
            raise MrError("Ollama models don't have directories to rename.")
        if not row["file_path"]:
            raise MrError("No file_path recorded for this model - cannot rename on disk.")

        old_path = Path(row["file_path"])
        old_dir = old_path.parent
        new_dir = old_dir.with_name(new_name)

        if not old_dir.exists():
            raise MrError(f"Directory not found on disk: {old_dir}")
        if new_dir.exists():
            raise MrError(f"Directory already exists: {new_dir}")

        old_dir.rename(new_dir)

        new_path = new_dir / old_path.name
        conn.execute(
            "UPDATE models SET file_path=?, last_updated=? WHERE id=?",
            (str(new_path), now, row["id"]),
        )
        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (row["id"], "rename", now, f"{old_dir.name} → {new_dir.name}"),
        )
        conn.commit()
        return {
            "status": "renamed",
            "display_name": row["display_name"],
            "old_dir": str(old_dir),
            "new_dir": str(new_dir),
            "timestamp": now,
        }
    finally:
        conn.close()


def engine_blacklist(
    config: dict | None = None,
    name: str | None = None,
    reason: str | None = None,
    backend: str | None = None,
    confirm: bool = False,
) -> dict:
    """Set a model's status to blacklisted, record reason, and delete it from disk.

    Works even if the model isn't in the registry yet (creates a new entry),
    in which case `backend` is required.
    """
    _require_writes()
    _require_confirm(confirm)
    if config is None:
        config = load_config()
    if name is None:
        raise MrError("name is required")

    conn = get_db(config)
    init_db(conn)
    now = now_iso()
    try:
        rows = find_matching_rows(conn, name)

        if not rows:
            if not backend:
                raise MrError(
                    "Model not found in registry. backend is required to add a new blacklisted entry."
                )
            hf_repo, variant = parse_hf_repo_from_ollama(name)
            conn.execute(
                """INSERT INTO models
                   (display_name, hf_repo, variant, backend, source_type,
                    ollama_name, currently_local, first_seen, last_updated)
                   VALUES (?,?,?,?,?,?,0,?,?)""",
                (name, hf_repo, variant, backend, get_source_type(name),
                 name if backend == "ollama" else None, now, now),
            )
            mid = _last_id(conn)
            row = conn.execute("SELECT * FROM models WHERE id=?", (mid,)).fetchone()
        elif len(rows) == 1:
            row = rows[0]
        else:
            raise AmbiguousModel(name, [dict(r) for r in rows])

        reason_text = reason or "no reason specified"
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

        if row["currently_local"]:
            _delete_model_files(row, config, conn)
            conn.execute(
                "UPDATE models SET currently_local=0 WHERE id=?", (row["id"],)
            )
            conn.execute(
                "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                (row["id"], "delete", now, "auto-deleted on blacklist"),
            )

        conn.commit()
        return {
            "status": "blacklisted",
            "display_name": row["display_name"],
            "reason": reason_text,
            "timestamp": now,
        }
    finally:
        conn.close()


# ─── Engine functions: pull ───────────────────────────────────────────────────


def engine_pull(
    config: dict | None = None,
    ref: str | None = None,
    variant: str | None = None,
    backend: str | None = None,
    file_pattern: str | None = None,
    subdir: str | None = None,
    download_all: bool = False,
    filename: str | None = None,
    allow_blacklisted: bool = False,
    confirm: bool = False,
    log: list[str] | None = None,
) -> dict:
    """Pull a model from Ollama, HuggingFace, or CivitAI.

    ref may be: an ollama name, 'org/repo' or 'hf.co/org/repo:tag' (GGUF/ComfyUI),
    a CivitAI version/model URL or AIR tag, or 'civitai:<versionId>'.

    Ambiguous multi-file downloads require file_pattern (or download_all=True for GGUF).
    """
    _require_writes()
    _require_confirm(confirm)
    if config is None:
        config = load_config()
    if ref is None:
        raise MrError("ref (model identifier) is required")

    conn = get_db(config)
    init_db(conn)
    now = now_iso()

    try:
        # Variant shorthand: 'mr pull org/repo Q5_K_M'
        if variant and not re.search(r"(?:hf\.co|huggingface\.co)/", ref, re.IGNORECASE):
            if "/" in ref:
                ref = f"{ref}:{variant}"
            elif backend in get_gguf_backend_names(config):
                raise MrError(
                    f"For {backend}, ref must be 'org/repo' format when using the variant argument."
                )

        # Auto-detect backend
        if backend is None:
            if parse_civitai_version_id(ref) is not None:
                backend = "comfyui"
            elif ref.endswith(".gguf") or (
                "/" in ref and not re.search(r"(?:hf\.co|huggingface\.co)/", ref, re.IGNORECASE)
            ):
                backend = "llamacpp"
            else:
                backend = "ollama"

        # Pre-pull blacklist/deleted check
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

        if existing and existing["status"] == "blacklisted":
            if not allow_blacklisted:
                raise MrError(
                    f"Model is blacklisted: {existing['display_name']}. "
                    "Set allow_blacklisted=True to proceed."
                )
        if existing and existing["status"] == "deleted" and not allow_blacklisted:
            _log(
                log,
                f"Warning: model previously deleted ({existing['display_name']}). "
                "Downloading anyway (status will be reset).",
            )

        # ── Ollama pull ──────────────────────────────────────────────────────
        if backend == "ollama":
            container = config["backends"]["ollama"]["docker_container"]
            _log(log, f"Pulling {ref} via Ollama...")
            result = subprocess.run(
                ["docker", "exec", container, "ollama", "pull", ref],
                text=True,
            )
            if result.returncode != 0:
                raise MrError("Pull failed.")

            hf_repo2, variant2 = parse_hf_repo_from_ollama(ref)
            source_type = get_source_type(ref)

            existing_ollama = conn.execute(
                "SELECT * FROM models WHERE backend='ollama' AND (ollama_name=? OR display_name=?)",
                (ref, ref),
            ).fetchone()

            if existing_ollama:
                conn.execute(
                    """UPDATE models
                       SET currently_local=1, times_downloaded=times_downloaded+1,
                           status=CASE WHEN status='deleted' THEN NULL ELSE status END,
                           last_used=?, last_updated=?
                       WHERE id=?""",
                    (now, now, existing_ollama["id"]),
                )
                mid = existing_ollama["id"]
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
            _log(log, "Pull complete. Registry updated.")
            return {"status": "pulled", "backend": "ollama", "ref": ref, "model_id": mid, "timestamp": now}

        # ── GGUF file backends ───────────────────────────────────────────────
        elif backend in get_gguf_backend_names(config):
            backend_cfg = config["backends"].get(backend, {})
            model_dir = Path(backend_cfg.get("model_dir", ""))
            if not model_dir.exists():
                raise MrError(f"Model directory for {backend} does not exist: {model_dir}")

            token_env = config.get("huggingface", {}).get("token_env_var", "HF_TOKEN")
            token = os.environ.get(token_env)

            parsed_repo, parsed_variant = parse_hf_repo_from_ollama(ref)
            repo_id = parsed_repo if parsed_repo else ref
            if parsed_variant and not file_pattern:
                file_pattern = f"*{parsed_variant}*.gguf"

            if "/" not in repo_id:
                raise MrError(f"For {backend}, ref must be 'org/repo' or 'hf.co/org/repo:tag' format.")

            try:
                all_files = list(list_repo_files(repo_id, token=token))
            except Exception as e:
                raise MrError(f"Failed to list repo files for {repo_id}: {e}")

            gguf_files = [f for f in all_files if f.endswith(".gguf")]
            if not gguf_files:
                raise MrError(f"No .gguf files found in {repo_id}")

            if file_pattern:
                pat = file_pattern
                if not any(c in pat for c in "*?[]") and not pat.endswith(".gguf"):
                    pat = f"*{pat}*.gguf"
                matches = [f for f in gguf_files if fnmatch.fnmatch(f.lower(), pat.lower())]
                if not matches:
                    raise MrError(f"No files matching '{file_pattern}' in {repo_id}")
                files_to_download = matches
            elif len(gguf_files) == 1:
                files_to_download = gguf_files
            else:
                if not download_all:
                    raise MrError(
                        f"Multiple GGUF files in {repo_id}. Specify file_pattern or set download_all=True.",
                        {"kind": "gguf_multi", "files": gguf_files},
                    )
                files_to_download = gguf_files

            repo_name = repo_id.split("/")[1] if "/" in repo_id else repo_id
            repo_name = re.sub(r"[^a-zA-Z0-9_.-]", "-", repo_name)
            target_dir = model_dir / repo_name
            target_dir.mkdir(parents=True, exist_ok=True)

            _log(log, f"Downloading {len(files_to_download)} file(s) to {target_dir}...")
            downloaded_paths = []
            for gguf_file in files_to_download:
                downloaded_file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=gguf_file,
                    local_dir=str(target_dir),
                    token=token,
                )
                local_path = Path(downloaded_file_path)
                if local_path.parent != target_dir:
                    final_path = target_dir / local_path.name
                    shutil.move(str(local_path), str(final_path))
                    local_path = final_path
                    _log(log, f"  Flattened {gguf_file} to: {local_path}")
                else:
                    _log(log, f"  Downloaded: {gguf_file}")
                downloaded_paths.append(str(local_path))

            total_size = sum(Path(p).stat().st_size for p in downloaded_paths)
            size_gb = round(total_size / (1024 ** 3), 2)

            main_file = max(downloaded_paths, key=lambda p: Path(p).stat().st_size)
            main_path = Path(main_file)
            variant_val = parse_variant_from_filename(main_path.name) or parsed_variant
            context_window = parse_context_window_from_gguf(main_path, config)
            if context_window is None:
                context_window = get_hf_context_window(repo_id, token)

            existing_by_dir = None
            for p in downloaded_paths:
                existing_by_dir = conn.execute(
                    "SELECT * FROM models WHERE file_path=? AND backend=?", (p, backend)
                ).fetchone()
                if existing_by_dir:
                    break
            if not existing_by_dir:
                existing_by_dir = conn.execute(
                    "SELECT * FROM models WHERE display_name=? AND backend=?", (repo_name, backend)
                ).fetchone()

            if existing_by_dir:
                conn.execute(
                    """UPDATE models
                       SET currently_local=1, times_downloaded=times_downloaded+1,
                           file_path=?, size_gb=?, last_used=?, last_updated=?, variant=?,
                           context_window=COALESCE(?, context_window),
                           status=CASE WHEN status='deleted' THEN NULL ELSE status END
                       WHERE id=?""",
                    (main_file, size_gb, now, now, variant_val, context_window, existing_by_dir["id"]),
                )
                mid = existing_by_dir["id"]
            else:
                conn.execute(
                    """INSERT INTO models
                       (display_name, hf_repo, variant, backend, source_type,
                        file_path, size_gb, context_window, currently_local, times_downloaded,
                        first_seen, last_used, last_updated)
                       VALUES (?,?,?,?,?,?,?,?,1,1,?,?,?)""",
                    (repo_name, repo_id, variant_val, backend, backend,
                     main_file, size_gb, context_window, now, now, now),
                )
                mid = _last_id(conn)

            conn.execute(
                "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                (mid, "pull", now, json.dumps({
                    "repo_id": repo_id, "files": files_to_download, "subdir": repo_name,
                })),
            )
            conn.commit()
            _log(
                log,
                f"Downloaded {len(files_to_download)} file(s) to {target_dir} ({size_gb:.2f} GB). Registry updated.",
            )
            return {
                "status": "pulled",
                "backend": backend,
                "repo_id": repo_id,
                "target_dir": str(target_dir),
                "size_gb": size_gb,
                "files": files_to_download,
                "model_id": mid,
                "timestamp": now,
            }

        # ── ComfyUI download (HuggingFace or CivitAI) ────────────────────────
        elif backend == "comfyui":
            comfy_cfg = config["backends"].get("comfyui", {})
            base_dir = Path(comfy_cfg.get("base_dir", ""))
            if not base_dir.exists():
                raise MrError(f"ComfyUI base_dir does not exist: {base_dir}")

            civitai_cfg = config.get("civitai", {})
            token_env = civitai_cfg.get("token_env_var", "CIVITAI_API_KEY")
            civitai_token = os.environ.get(token_env)

            civitai_version_id = parse_civitai_version_id(ref)
            _civitai_model_id = parse_civitai_model_id(ref)

            # Browse URL with model ID but no version ID → resolve via API
            if civitai_version_id is None and _civitai_model_id:
                _dm = re.search(_CIVITAI_DOMAIN_RE, ref)
                _host = _dm.group(0) if _dm else "civitai.com"
                _log(log, f"Fetching model info from CivitAI API (model {_civitai_model_id})...")
                civitai_version_id, _api_subdir = fetch_civitai_model_info(
                    _civitai_model_id, token=civitai_token, host=_host
                )
                if civitai_version_id:
                    _log(log, f"  Using latest version {civitai_version_id}")
                if not subdir and _api_subdir:
                    subdir = _api_subdir
                    _log(log, f"  Auto-detected subdir {subdir} from CivitAI model type")

            if not subdir:
                air = parse_air_tag(ref)
                if air:
                    subdir = AIR_TYPE_TO_SUBDIR.get(air["type"])
                    if subdir:
                        _log(log, f"  Auto-detected subdir {subdir} from AIR type '{air['type']}'")
            if not subdir:
                raise MrError("subdir is required for ComfyUI downloads (e.g. checkpoints, loras, vae)")

            dest_dir = base_dir / subdir
            dest_dir.mkdir(parents=True, exist_ok=True)

            if civitai_version_id:
                # ── CivitAI download ─────────────────────────────────────────
                _dm = re.search(_CIVITAI_DOMAIN_RE, ref)
                _civitai_host = _dm.group(0) if _dm else "civitai.com"
                download_url = f"https://{_civitai_host}/api/download/models/{civitai_version_id}"
                params = {}
                if civitai_token:
                    params["token"] = civitai_token
                else:
                    _log(log, "Warning: No CivitAI API key found. Download may fail for gated models.")

                _log(log, f"Downloading from CivitAI (version {civitai_version_id})...")
                resp = requests.get(download_url, params=params, stream=True, timeout=(30, None))
                if resp.status_code == 401:
                    raise MrError(f"CivitAI download failed: unauthorized. Check your {token_env} env var.")
                if resp.status_code != 200:
                    raise MrError(f"CivitAI download failed (HTTP {resp.status_code})")

                cd = resp.headers.get("Content-Disposition", "")
                filename_match = re.search(r'filename="?([^";\r\n]+)"?', cd)
                out_filename = _sanitize_filename(filename_match.group(1)) if filename_match else ""
                if not out_filename:
                    out_filename = _sanitize_filename(filename or "")
                if not out_filename:
                    out_filename = f"civitai_{civitai_version_id}.bin"

                local_path = dest_dir / out_filename

                try:
                    with open(local_path, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=8192):
                            fh.write(chunk)
                except Exception:
                    local_path.unlink(missing_ok=True)
                    raise

                size_gb = round(local_path.stat().st_size / (1024 ** 3), 4)
                fpath = str(local_path)

                air = parse_air_tag(ref)
                civitai_url = civitai_source_url(ref, civitai_version_id, air["model_id"] if air else None)

                existing_by_path = conn.execute(
                    "SELECT * FROM models WHERE file_path=?", (fpath,)
                ).fetchone()
                existing_comfy = existing if (existing and existing["backend"] == "comfyui") else None
                if existing_by_path or existing_comfy:
                    row_to_update = existing_by_path or existing_comfy
                    conn.execute(
                        """UPDATE models
                           SET currently_local=1, times_downloaded=times_downloaded+1,
                               file_path=?, size_gb=?, last_used=?, last_updated=?,
                               source_type='comfyui_civitai', source_url=?,
                               status=CASE WHEN status='deleted' THEN NULL ELSE status END
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
                    (mid, "pull", now, json.dumps({
                        "civitai_version_id": civitai_version_id, "file": out_filename,
                    })),
                )
                conn.commit()
                _log(log, f"Downloaded to {local_path} ({size_gb:.2f} GB). Registry updated.")
                return {
                    "status": "pulled",
                    "backend": "comfyui",
                    "source": "civitai",
                    "file_path": str(local_path),
                    "size_gb": size_gb,
                    "model_id": mid,
                    "timestamp": now,
                }

            else:
                # ── HuggingFace download to ComfyUI subdir ──────────────────
                token_env = config.get("huggingface", {}).get("token_env_var", "HF_TOKEN")
                token = os.environ.get(token_env)

                _hf_resolve = re.match(
                    r"https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$",
                    ref, re.IGNORECASE,
                )
                if _hf_resolve:
                    repo_id = _hf_resolve.group(1)
                    revision = _hf_resolve.group(2)
                    chosen_file = _hf_resolve.group(3)
                    _log(log, f"Downloading {chosen_file} from {repo_id} @ {revision}...")
                else:
                    repo_id = ref
                    revision = None

                    if "/" not in repo_id:
                        raise MrError(
                            "For HuggingFace, ref must be 'org/repo' format or a full resolve URL."
                        )

                    comfy_exts = tuple(comfy_cfg.get(
                        "extensions", [".safetensors", ".ckpt", ".pt", ".pth", ".bin"]
                    ))
                    try:
                        all_files = list(list_repo_files(repo_id, token=token))
                    except Exception as e:
                        raise MrError(f"Failed to list repo files for {repo_id}: {e}")
                    model_files = [f for f in all_files if f.lower().endswith(comfy_exts)]

                    if not model_files:
                        raise MrError(f"No model files found in {repo_id}")

                    if file_pattern:
                        matches = [f for f in model_files if fnmatch.fnmatch(f.lower(), file_pattern.lower())]
                        if not matches:
                            raise MrError(f"No files matching '{file_pattern}' in {repo_id}")
                        chosen_file = matches[0]
                    elif len(model_files) == 1:
                        chosen_file = model_files[0]
                    else:
                        raise MrError(
                            f"Multiple model files in {repo_id}. Specify file_pattern to choose one.",
                            {"kind": "comfyui_hf_multi", "files": model_files},
                        )

                    _log(log, f"Downloading {chosen_file} from {repo_id}...")

                _hf_kwargs = dict(
                    repo_id=repo_id, filename=chosen_file,
                    local_dir=str(dest_dir), token=token,
                )
                if revision:
                    _hf_kwargs["revision"] = revision
                local_path = Path(hf_hub_download(**_hf_kwargs))

                if dest_dir.exists():
                    flatten_hf_subdir(dest_dir)
                if not local_path.exists():
                    local_path = dest_dir / local_path.name

                size_gb = round(local_path.stat().st_size / (1024 ** 3), 4)
                fpath = str(local_path)

                existing_by_path = conn.execute(
                    "SELECT * FROM models WHERE file_path=?", (fpath,)
                ).fetchone()
                existing_comfy = existing if (existing and existing["backend"] == "comfyui") else None
                if existing_by_path or existing_comfy:
                    row_to_update = existing_by_path or existing_comfy
                    conn.execute(
                        """UPDATE models
                           SET currently_local=1, times_downloaded=times_downloaded+1,
                               file_path=?, size_gb=?, hf_repo=?, last_used=?, last_updated=?,
                               source_type='comfyui_hf',
                               status=CASE WHEN status='deleted' THEN NULL ELSE status END
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
                _log(log, f"Downloaded to {local_path} ({size_gb:.2f} GB). Registry updated.")
                return {
                    "status": "pulled",
                    "backend": "comfyui",
                    "source": "huggingface",
                    "file_path": str(local_path),
                    "size_gb": size_gb,
                    "model_id": mid,
                    "timestamp": now,
                }

        else:
            avail = ["ollama"] + get_gguf_backend_names(config) + ["comfyui"]
            raise MrError(f"Unknown backend: '{backend}'. Available backends: {', '.join(avail)}")
    finally:
        conn.close()


def engine_restore(
    config: dict | None = None,
    dry_run: bool = False,
    confirm: bool = False,
    log: list[str] | None = None,
) -> dict:
    """Re-download all missing ComfyUI models that have a known source."""
    _require_writes()
    if config is None:
        config = load_config()

    conn = get_db(config)
    init_db(conn)
    try:
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
    finally:
        conn.close()

    if not restorable and not no_source:
        return {"status": "no_missing_models", "restored": 0, "failed": [], "no_source": []}

    no_source_names = [r["display_name"] for r in no_source]
    _log(log, f"WARNING: {len(no_source_names)} model(s) have no recorded source and cannot be restored:")
    for n in no_source_names:
        _log(log, f"  - {n}")

    if not restorable:
        return {"status": "no_restorable", "restored": 0, "failed": [], "no_source": no_source_names}

    _log(log, f"{len(restorable)} model(s) queued for restore:")
    for row in restorable:
        src = row["source_url"] or row["hf_repo"]
        label = f" -> {row['variant']}" if row["variant"] else ""
        _log(log, f"  - {row['display_name']}{label}  ({src})")

    if dry_run:
        return {
            "status": "dry_run",
            "count": len(restorable),
            "models": [dict(r) for r in restorable],
            "no_source": no_source_names,
        }

    _require_confirm(confirm)

    failed = []
    for row in restorable:
        subdir = row["variant"] or None

        if row["source_url"]:
            ref = row["source_url"]
            file_pattern = None
        else:
            ref = row["hf_repo"]
            file_pattern = Path(row["file_path"]).name if row["file_path"] else None

        if not subdir and row["file_path"]:
            comfy_cfg = config["backends"].get("comfyui", {})
            base_dir = Path(comfy_cfg.get("base_dir", ""))
            try:
                rel = Path(row["file_path"]).relative_to(base_dir)
                subdir = rel.parts[0] if len(rel.parts) > 1 else None
            except ValueError:
                pass

        try:
            engine_pull(
                config=config,
                ref=ref,
                backend="comfyui",
                file_pattern=file_pattern,
                subdir=subdir,
                confirm=True,
                log=log,
            )
        except Exception as e:
            _log(log, f"Error restoring {row['display_name']}: {e}")
            failed.append(row["display_name"])

    if failed:
        _log(log, f"Failed to restore {len(failed)} model(s): {', '.join(failed)}")
    else:
        _log(log, f"All {len(restorable)} model(s) restored.")

    return {
        "status": "restored" if not failed else "partial",
        "count": len(restorable),
        "restored": len(restorable) - len(failed),
        "failed": failed,
        "no_source": no_source_names,
        "timestamp": now_iso(),
    }
