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
import hashlib
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

    def __init__(self, name: str, suggestions: list[str] | None = None, hint: str | None = None):
        self.name = name
        self.suggestions = suggestions or []
        self.hint = hint
        message = f"Model '{name}' not found"
        if hint:
            message += f" ({hint})"
        details = {"suggestions": self.suggestions}
        if hint:
            details["hint"] = hint
        super().__init__(message, details)


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

# Shared User-Agent for outbound HTTP. CivitAI (and others) reject requests with
# the default urllib/requests agent with 403, so always send a stable one.
_HTTP_HEADERS = {
    "User-Agent": f"model-registry/{__version__} (+https://github.com/TorvaldUtne/model-registry)"
}

# CivitAI split its web frontend: green = SFW only, red = includes NSFW.
# The red host serves everything, so it is the default browse host; set
# civitai.web_host to civitai.green to force SFW-only links. The API is served
# on either host (civitai.com remains the documented default api_host).
CIVITAI_DEFAULT_WEB_HOST = "civitai.red"
CIVITAI_DEFAULT_API_HOST = "civitai.com"


def civitai_web_host(config: dict | None = None) -> str:
    """Host used for human-facing CivitAI browse URLs (default: civitai.green)."""
    host = (config or {}).get("civitai", {}).get("web_host")
    return (host or CIVITAI_DEFAULT_WEB_HOST).strip() or CIVITAI_DEFAULT_WEB_HOST


def civitai_api_host(config: dict | None = None) -> str:
    """Host used for CivitAI API calls (default: civitai.com)."""
    host = (config or {}).get("civitai", {}).get("api_host")
    return (host or CIVITAI_DEFAULT_API_HOST).strip() or CIVITAI_DEFAULT_API_HOST

# AIR type field → ComfyUI subdir name
AIR_TYPE_TO_SUBDIR = {
    "checkpoint":        "checkpoints",
    "checkpoints":       "checkpoints",
    "model":             "checkpoints",
    "vae":               "vae",
    "lora":              "loras",
    "loras":             "loras",
    "locon":             "loras",
    "lycoris":           "loras",
    "embedding":         "embeddings",
    "embeddings":        "embeddings",
    "textualinversion":  "embeddings",
    "hypernet":          "hypernetworks",
    "hypernetwork":      "hypernetworks",
    "hypernetworks":     "hypernetworks",
    "controlnet":        "controlnet",
    "upscaler":          "upscale_models",
    "ipadapter":         "ipadapter",
    "clipvision":        "clip_vision",
    "clip":              "clip",
    "diffusionmodel":    "diffusion_models",
    "diffusion":         "diffusion_models",
    "diffusers":         "diffusion_models",
    "textencoder":       "text_encoders",
    "textencoders":      "text_encoders",
    "unet":              "unet",
    "vaeapprox":         "vae_approx",
    "motionmodule":      "animatediff_models",
    "gligen":            "gligen",
    "photomaker":        "photomaker",
    "stylemodels":       "style_models",
    "stylemodel":        "style_models",
    "sam":               "sams",
    "sams":              "sams",
    "facerestoremodels": "facerestore_models",
    "workflows":         "workflows",
    "poses":             "poses",
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
    "diffusionmodel":    "diffusion_models",
    "diffusion":         "diffusion_models",
    "diffusers":         "diffusion_models",
    "textencoder":       "text_encoders",
    "workflows":         "workflows",
    "wildcards":         "wildcards",
}

# CivitAI per-file "type" from /api/v1/model-versions/{id} → ComfyUI subdir name
# (e.g. a version can ship a 'Diffusion Model' plus a 'VAE' or 'Text Encoder')
CIVITAI_FILE_TYPE_TO_SUBDIR = {
    "diffusionmodel": "diffusion_models",
    "checkpoint":     "checkpoints",
    "vae":            "vae",
    "lora":           "loras",
    "controlnet":     "controlnet",
    "upscaler":       "upscale_models",
    "textencoder":    "text_encoders",
    "motionmodule":   "animatediff_models",
    "embedding":      "embeddings",
    "clip":           "clip",
    "clipvision":     "clip_vision",
}


def comfyui_subdir_for_type(type_: str | None) -> str | None:
    """Map a CivitAI/AIR type string to a ComfyUI subdir.

    Handles case/punctuation differences ('Diffusion Model', 'diffusionmodel',
    'DiffusionModel') and the singular→plural mismatch ('diffusionmodel' → the
    plural subdir 'diffusion_models').
    """
    if not type_:
        return None
    normalized = re.sub(r"[^a-z0-9]", "", type_.lower())
    for table in (
        AIR_TYPE_TO_SUBDIR,
        CIVITAI_API_TYPE_TO_SUBDIR,
        CIVITAI_FILE_TYPE_TO_SUBDIR,
    ):
        if normalized in table:
            return table[normalized]
        if normalized.endswith("s"):
            singular = table.get(normalized[:-1])
            if singular:
                return singular
    return None


# Canonical ComfyUI subdir names (values of the type→subdir maps above), used to
# recognize repo-supplied directory prefixes such as 'loras/foo.safetensors'.
_COMFYUI_SUBDIRS = {
    v.lower()
    for table in (AIR_TYPE_TO_SUBDIR, CIVITAI_API_TYPE_TO_SUBDIR, CIVITAI_FILE_TYPE_TO_SUBDIR)
    for v in table.values()
}


def detect_comfyui_subdir(rel_path: str) -> tuple[str | None, str]:
    """Detect a ComfyUI subdir inside a repo-relative file path.

    Many HuggingFace repos ship files under a directory named after the ComfyUI
    type, e.g. 'loras/krea2.safetensors' or 'text_encoders/qwen.safetensors'.
    Returns (subdir, remainder) using the *last* known subdir segment, so
    'loras/loras/x' → ('loras', 'x') and
    'text_encoders/split_files/text_encoders/x' → ('text_encoders', 'x').
    Returns (None, rel_path) when no known subdir is present.
    """
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]
    anchor = None
    for i, part in enumerate(parts[:-1]):  # never treat the file name itself as a dir
        if part.lower() in _COMFYUI_SUBDIRS:
            anchor = i
    if anchor is None:
        return None, rel_path
    return parts[anchor].lower(), "/".join(parts[anchor + 1:])


# ─── Helpers ──────────────────────────────────────────────────────────────────


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(log: list[str] | None, msg: str) -> None:
    if log is not None:
        log.append(msg)


def _file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str | None:
    """Return the lowercase SHA-256 hex digest of a file, or None if unreadable.

    Streams in chunks so multi-GB checkpoints don't load into memory.
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


# ─── Config ───────────────────────────────────────────────────────────────────


def load_config() -> dict:
    """Load config.json. Raises MrError if not found or not valid JSON."""
    if not CONFIG_FILE.exists():
        raise MrError("config.json not found. Run 'mr init' first.")
    with open(CONFIG_FILE) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise MrError(f"config.json is not valid JSON: {e}") from e


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
            context_window    INTEGER,
            file_hash         TEXT,
            hash_checked      TEXT
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
        ("file_hash",     "TEXT"),
        ("hash_checked",  "TEXT"),
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


# Model file extensions users may paste along with a name (e.g. after an `ls`).
_MODEL_EXTENSIONS = (
    ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx", ".sft",
)


def _strip_model_extension(name: str) -> str | None:
    """Return name without a known model file extension, or None if none present."""
    lower = name.lower()
    for ext in _MODEL_EXTENSIONS:
        if lower.endswith(ext):
            return name[: -len(ext)]
    return None


def _escape_like(term: str) -> str:
    """Escape LIKE wildcards so a user term is matched literally.

    Callers must pair this with ``LIKE ? ESCAPE '\\'``. Without it a name
    containing ``_`` or ``%`` (common in ComfyUI filenames) silently matches
    unrelated rows.
    """
    return (
        term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    )


def _query_by_name(conn: sqlite3.Connection, name: str) -> list[sqlite3.Row]:
    # Deterministic ordering is required so that an index chosen from one call
    # selects the same row on a later call with the same name.
    like = f"%{_escape_like(name)}%"
    return conn.execute(
        """SELECT * FROM models
           WHERE display_name LIKE ? ESCAPE '\\' OR ollama_name LIKE ? ESCAPE '\\'
           ORDER BY display_name, backend, id""",
        (like, like),
    ).fetchall()


def resolve_model(
    conn: sqlite3.Connection, name: str, index: int | None = None
) -> sqlite3.Row:
    """Return a single models row matching name (partial on display_name/ollama_name).

    - Exactly one match → returns it.
    - Multiple matches and index provided → returns match at that index.
    - Multiple matches and index not provided → raises AmbiguousModel.
    - No matches → raises ModelNotFound with close-match suggestions.

    A trailing model file extension (e.g. '.safetensors') is tolerated: if the
    raw name matches nothing it is retried without the extension.

    Args:
        index: 0-based index to select among ambiguous matches. Omit it (None)
            to force an AmbiguousModel error when more than one row matches.

    The same (name, index) pair always resolves to the same row because
    ``_query_by_name`` orders deterministically, so callers can prompt for a
    choice once and pass the index on to a later engine call.
    """
    rows = _query_by_name(conn, name)

    # Tolerate a pasted filename: 'foo.safetensors' → 'foo'.
    if not rows:
        stem = _strip_model_extension(name)
        if stem and stem != name:
            rows = _query_by_name(conn, stem)

    if not rows:
        search = _strip_model_extension(name) or name
        all_names = [
            r["display_name"]
            for r in conn.execute("SELECT display_name FROM models").fetchall()
        ]
        suggestions = difflib.get_close_matches(search, all_names, n=5, cutoff=0.4)
        hint = None
        if search != name:
            hint = f"tried '{search}' without the file extension"
        raise ModelNotFound(name, suggestions, hint)

    if len(rows) == 1:
        return rows[0]

    if index is None:
        raise AmbiguousModel(name, [dict(r) for r in rows])

    if index < 0 or index >= len(rows):
        raise MrError(f"Index {index} out of range for {len(rows)} matches")

    return rows[index]


def find_matching_rows(conn: sqlite3.Connection, name: str) -> list[sqlite3.Row]:
    """Return all models rows matching name (partial on display_name/ollama_name).

    Ordered identically to ``resolve_model`` so an index chosen from one call
    selects the same row on another.
    """
    like = f"%{_escape_like(name)}%"
    return conn.execute(
        """SELECT * FROM models
           WHERE display_name LIKE ? ESCAPE '\\' OR ollama_name LIKE ? ESCAPE '\\'
           ORDER BY display_name, backend, id""",
        (like, like),
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


def _backend_model_dir(config: dict, backend: str) -> Path | None:
    """Return a GGUF backend's configured model_dir, or None if unset.

    Guards against the Path("") == Path(".") trap, which would otherwise make
    an unconfigured backend silently operate on the current working directory.
    """
    d = config.get("backends", {}).get(backend, {}).get("model_dir")
    return Path(d) if d else None


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

        card_data = getattr(model_info, "card_data", None)
        if card_data is not None:
            # huggingface_hub exposes this as a CardData/ModelCardData object
            # (not a dict), but older versions used a plain dict.
            if isinstance(card_data, dict):
                archs = card_data.get("architectures")
                model_name = card_data.get("model_name")
            else:
                archs = getattr(card_data, "architectures", None)
                model_name = getattr(card_data, "model_name", None)
            if archs and isinstance(archs, (list, tuple)) and archs:
                metadata["architecture"] = archs[0]
            elif model_name:
                name_lower = str(model_name).lower()
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


def place_downloaded_file(local_path: Path, dest_dir: Path, relative_name: str | None = None) -> Path:
    """Move a hub-downloaded file into dest_dir at relative_name (default: basename).

    HuggingFace honours the repo-relative path, so a file requested as
    'loras/foo.safetensors' lands in dest_dir/loras/foo.safetensors. When
    dest_dir already *is* the 'loras' subdir this yields the doubled
    models/loras/loras/foo.safetensors. Callers pass the path remainder after
    the ComfyUI-type prefix so the file lands at models/loras/foo.safetensors,
    and any emptied source directories are pruned.
    """
    target = dest_dir / (relative_name or local_path.name)
    if target == local_path:
        return local_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(local_path), str(target))
    parent = local_path.parent
    while parent != dest_dir and parent.is_dir():
        try:
            parent.rmdir()  # fails (and we stop) if not empty
        except OSError:
            break
        parent = parent.parent
    return target


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

    Also accepts bundled/file-specific tags where a CivitAI file id is appended
    with '+', e.g. civitai:2831028@3305034+3190059 — the version is the id after
    '@' and the trailing id is the CivitAI file id (see /api/v1/model-versions,
    file.downloadUrl '?fileId=…').
    """
    s = re.sub(r"^(?:urn:)?(?:air:)?", "", ref.strip(), flags=re.IGNORECASE)
    m = re.match(
        r"^([^:]+):([^:]+):civitai:(\d+)@(\d+)((?:\+\d+)+)?$",
        s, re.IGNORECASE,
    )
    if not m:
        return None
    file_suffix = m.group(5) or ""
    file_id = re.match(r"\+(\d+)", file_suffix).group(1) if file_suffix else None
    return {
        "ecosystem": m.group(1).lower(),
        "type": m.group(2).lower(),
        "model_id": m.group(3),
        "version_id": m.group(4),
        "file_id": file_id,
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
    m = re.search(r"[?&]versionId=(\d+)", ref)
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
        resp = requests.get(url, params=params, headers=_HTTP_HEADERS, timeout=15)
        if resp.status_code != 200:
            return None, None
        data = resp.json()
    except (requests.RequestException, json.JSONDecodeError):
        return None, None
    versions = data.get("modelVersions", [])
    version_id = str(versions[0]["id"]) if versions else None
    api_type = (data.get("type") or "").lower().replace(" ", "")
    subdir = comfyui_subdir_for_type(api_type)
    return version_id, subdir


def _parse_civitai_version_payload(data: dict) -> dict:
    """Normalize a CivitAI model-version payload.

    Both /model-versions/{id} and /model-versions/by-hash/{hash} return the same
    shape, so parse once here to keep the two lookups in sync.
    """
    files = []
    for f in data.get("files", []):
        meta = f.get("metadata") or {}
        files.append({
            "id": str(f.get("id")),
            "name": f.get("name"),
            "type": f.get("type"),
            "fp": meta.get("fp"),
            "format": meta.get("format"),
            "primary": bool(f.get("primary")),
            "download_url": f.get("downloadUrl"),
            "sha256": (f.get("hashes") or {}).get("SHA256"),
        })
    model = data.get("model") or {}
    return {
        "base_model": data.get("baseModel"),
        "trigger_words": data.get("trainedWords") or [],
        "model_id": str(model.get("id") or data.get("modelId") or ""),
        "model_name": model.get("name"),
        "model_type": model.get("type"),
        "version_id": str(data.get("id") or ""),
        "version_name": data.get("name"),
        "files": files,
    }


def fetch_civitai_version_info(
    version_id, token=None, host="civitai.com"
) -> dict | None:
    """Call CivitAI API v1 for a model version.

    Returns a dict with:
      base_model:      baseModel string (e.g. 'Krea 2' / 'SDXL 1.0') or None
      trigger_words:   list[str] of trained trigger words (may be empty)
      model_id:        CivitAI model id
      model_name:      CivitAI model display name (e.g. 'CyberRealistic Krea 2')
      model_type:      CivitAI model type (e.g. 'LORA', 'Checkpoint')
      version_id:      version id
      version_name:    version label (e.g. 'v3.0')
      files:           list of dicts with id/name/type/(fp,format)/primary/download_url/sha256
    Returns None on network/API failure (callers should fall back to defaults).
    """
    url = f"https://{host}/api/v1/model-versions/{version_id}"
    params = {}
    if token:
        params["token"] = token
    try:
        resp = requests.get(url, params=params, headers=_HTTP_HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except (requests.RequestException, json.JSONDecodeError):
        return None
    parsed = _parse_civitai_version_payload(data)
    if not parsed["version_id"]:
        parsed["version_id"] = str(version_id)
    return parsed


def fetch_civitai_version_by_hash(
    file_hash, token=None, host="civitai.com"
) -> dict | None:
    """Look up a CivitAI model version from a local file's SHA-256 hash.

    Uses GET /api/v1/model-versions/by-hash/{hash}; the query-parameter form
    returns 400, so the hash must be in the path. Returns the same shape as
    fetch_civitai_version_info, or None when the hash is unknown/not public or
    the request fails.
    """
    url = f"https://{host}/api/v1/model-versions/by-hash/{file_hash}"
    params = {}
    if token:
        params["token"] = token
    try:
        resp = requests.get(url, params=params, headers=_HTTP_HEADERS, timeout=20)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except (requests.RequestException, json.JSONDecodeError):
        return None
    return _parse_civitai_version_payload(data)


def civitai_metadata_filename(
    stem: str,
    version_info: dict | None,
    selected_file: dict | None = None,
    model_id: str | None = None,
) -> str:
    """Return an enriched file stem for a ComfyUI CivitAI model.

    Appends only the metadata tokens that are actually known, in the form
    {stem}_{v{version}}_{baseModel}_{fp}_civitai-{modelId}, so files are easier
    to identify on disk while still traceable back to the CivitAI page.
    """
    tokens = []
    if version_info:
        vname = version_info.get("version_name") or ""
        vtoken = re.sub(r"[^A-Za-z0-9.]+", "", vname) or version_info.get("version_id") or ""
        if vtoken:
            tokens.append(f"v{vtoken}" if not vtoken.lower().startswith("v") else vtoken)
    if version_info and version_info.get("base_model"):
        tokens.append(re.sub(r"[^A-Za-z0-9]+", "", version_info["base_model"]))
    if selected_file and selected_file.get("fp"):
        tokens.append(re.sub(r"[^A-Za-z0-9]+", "", selected_file["fp"]))
    if model_id:
        tokens.append(f"civitai-{model_id}")
    if not tokens:
        return _sanitize_filename(stem).strip("._") or _sanitize_filename(stem)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", _sanitize_filename(stem)).strip("._") or "model"
    return f"{slug}_{'_'.join(tokens)}"


def civitai_file_download_params(version_info, file_id) -> dict:
    """Return download query params that pin the download to the requested file.

    CivitAI's download endpoint selects a specific file via 'fileId' (matching
    the file.downloadUrl returned by /api/v1/model-versions). Falls back to {}
    (the version's primary file) when the id is unknown.
    """
    if not file_id:
        return {}
    if version_info and not any(f["id"] == file_id for f in version_info["files"]):
        return {}
    return {"fileId": file_id}


def _civitai_download_to_file(
    download_url: str,
    params: dict,
    dest_dir: Path,
    *,
    version_id: str,
    token_env: str,
    selected_file: dict | None = None,
    version_info: dict | None = None,
    filename: str | None = None,
    enrich_filename: bool = False,
    air: dict | None = None,
    log: list[str] | None = None,
    progress_callback=None,
) -> tuple[Path, int]:
    """Download a CivitAI model version into dest_dir, resuming across runs.

    Streams to '<final_name>.part' and atomically renames to the final name only
    once the transfer is complete. If a '.part' file from a previous interrupted
    run is present and the server advertises byte-range support, the transfer
    resumes from the current file size via a 'Range: bytes=<n>-' request. An
    already-complete final file is left untouched.

    Returns (final_path, resumed_from_bytes).
    """
    try:
        resp = requests.get(
            download_url, params=params, headers=_HTTP_HEADERS,
            stream=True, timeout=(30, None),
        )
    except requests.RequestException as e:
        raise MrError(f"CivitAI download failed: {e}") from e

    if resp.status_code == 401:
        resp.close()
        raise MrError(
            f"CivitAI download failed: unauthorized. Check your {token_env} env var."
        )
    if resp.status_code != 200:
        resp.close()
        raise MrError(f"CivitAI download failed (HTTP {resp.status_code})")

    cd = resp.headers.get("Content-Disposition", "")
    filename_match = re.search(r'filename="?([^";\r\n]+)"?', cd)
    out_filename = _sanitize_filename(filename_match.group(1)) if filename_match else ""
    if not out_filename and selected_file and selected_file.get("name"):
        out_filename = _sanitize_filename(selected_file["name"])
    if not out_filename:
        out_filename = _sanitize_filename(filename or "")
    if not out_filename:
        out_filename = f"civitai_{version_id}.bin"

    local_path = dest_dir / out_filename

    # Optional metadata-enriched filename (name + version + base model + fp + id).
    if enrich_filename and version_info:
        base_stem = version_info.get("model_name") or Path(out_filename).stem
        model_id_for_name = (
            (air.get("model_id") if air else None) or version_info.get("model_id") or None
        )
        enriched_stem = civitai_metadata_filename(
            base_stem, version_info, selected_file, model_id_for_name
        )
        enriched_name = f"{enriched_stem}{Path(out_filename).suffix or '.safetensors'}"
        if enriched_name != out_filename:
            target_path = dest_dir / enriched_name
            if target_path.exists():
                _log(log, f"  {enriched_name} already exists — keeping {out_filename}")
            else:
                out_filename = enriched_name
                local_path = target_path
                _log(log, f"  Saving as {enriched_name}")

    full_total = int(resp.headers.get("Content-Length", 0)) or None

    # Already fully downloaded on a previous run: nothing to do.
    if local_path.exists() and local_path.stat().st_size > 0:
        size = local_path.stat().st_size
        if full_total is None or size == full_total:
            resp.close()
            _log(log, f"Already downloaded: {local_path} ({size} bytes)")
            return local_path, 0

    part_path = local_path.with_name(local_path.name + ".part")
    offset = part_path.stat().st_size if part_path.exists() else 0
    accept_ranges = resp.headers.get("Accept-Ranges", "").lower() == "bytes"
    total = full_total
    mode = "wb"

    # A leftover .part may belong to a different/older remote file. Only trust
    # it when it is no larger than the current file; finalize an exact match and
    # discard a larger one so a corrupt file is never renamed into place.
    if full_total is not None:
        if offset == full_total:
            resp.close()
            os.replace(part_path, local_path)
            _log(log, f"Finalized {local_path.name} from {offset} cached bytes.")
            return local_path, offset
        if offset > full_total:
            _log(log, "  Cached .part is larger than the remote file; restarting download.")
            offset = 0
            mode = "wb"

    if offset > 0 and accept_ranges:
        resp.close()
        resume_headers = dict(_HTTP_HEADERS)
        resume_headers["Range"] = f"bytes={offset}-"
        _log(log, f"  Resuming {out_filename} from {offset} bytes...")
        try:
            resp = requests.get(
                download_url, params=params, headers=resume_headers,
                stream=True, timeout=(30, None),
            )
        except requests.RequestException as e:
            raise MrError(
                f"CivitAI download interrupted after {offset} bytes: {e}. "
                "Rerun the same command to resume."
            ) from e
        if resp.status_code == 206:
            content_range = resp.headers.get("Content-Range", "")
            m = re.search(r"/(\d+)\s*$", content_range)
            total = int(m.group(1)) if m else full_total
            mode = "ab"
        elif resp.status_code == 200:
            _log(log, "  Server did not honor Range; restarting download.")
            offset = 0
            mode = "wb"
            total = int(resp.headers.get("Content-Length", 0)) or None
        elif resp.status_code == 416:
            # Stale/oversized .part: drop it and restart the whole transfer.
            resp.close()
            _log(log, "  Server rejected Range request; restarting download.")
            resp = requests.get(
                download_url, params=params, headers=_HTTP_HEADERS,
                stream=True, timeout=(30, None),
            )
            if resp.status_code != 200:
                resp.close()
                raise MrError(f"CivitAI download failed (HTTP {resp.status_code})")
            offset = 0
            mode = "wb"
            total = int(resp.headers.get("Content-Length", 0)) or None
        else:
            resp.close()
            raise MrError(f"CivitAI download failed (HTTP {resp.status_code})")
    elif offset > 0:
        _log(log, "  Server does not advertise Range support; restarting download.")
        offset = 0

    if total is not None and offset == total:
        # The .part file already holds the whole file; just finalize it.
        resp.close()
        os.replace(part_path, local_path)
        return local_path, offset

    downloaded = offset
    try:
        with open(part_path, mode) as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                if progress_callback is not None:
                    progress_callback(downloaded, total, out_filename)
    except requests.RequestException as e:
        # Keep the .part file so a rerun can resume it.
        raise MrError(
            f"CivitAI download interrupted after {downloaded} bytes: {e}. "
            "Rerun the same command to resume."
        ) from e
    finally:
        resp.close()

    final_size = part_path.stat().st_size
    if total is not None and final_size < total:
        raise MrError(
            f"CivitAI download incomplete: {final_size}/{total} bytes. "
            "Rerun the same command to resume."
        )

    os.replace(part_path, local_path)
    return local_path, offset


def civitai_source_url(ref: str, version_id, model_id=None, config: dict | None = None) -> str:
    """Return the value to store as source_url for a CivitAI download.

    Constructed (non-URL) refs use the configured web host; explicit URLs keep
    whatever domain the user supplied.
    """
    web = civitai_web_host(config)
    if parse_air_tag(ref):
        return ref
    if re.search(_CIVITAI_DOMAIN_RE + r"/models/", ref, re.IGNORECASE):
        m = re.match(
            r"(https://" + _CIVITAI_DOMAIN_RE + r"/models/\d+(?:/[^?#]*)?)",
            ref,
            re.IGNORECASE,
        )
        base = m.group(1).rstrip("/") if m else f"https://{web}/models"
        return f"{base}?modelVersionId={version_id}"
    if model_id:
        return f"https://{web}/models/{model_id}?modelVersionId={version_id}"
    return f"https://{web}/models?versionId={version_id}"


def get_model_link(row, config: dict | None = None) -> str | None:
    """Get a browse URL for a model row (AIR tags use the configured web host)."""
    if row["source_url"]:
        air = parse_air_tag(row["source_url"])
        if air:
            return f"https://{civitai_web_host(config)}/models/{air['model_id']}?modelVersionId={air['version_id']}"
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


def scan_comfyui(config: dict, conn: sqlite3.Connection) -> tuple[int, int, int]:
    """Scan all category subdirs of the ComfyUI models base_dir.

    Returns (added, updated, merged). Category subdirs are searched
    recursively, so files that `pull` placed under an extra folder
    (e.g. loras/author/foo.safetensors) are still tracked.
    """
    comfy_cfg = config["backends"].get("comfyui", {})
    base_dir = Path(comfy_cfg.get("base_dir", "") or "")
    extensions = comfy_cfg.get("extensions", [".safetensors", ".ckpt", ".pt", ".pth", ".bin"])
    now = now_iso()
    added = updated = 0

    if not comfy_cfg.get("base_dir"):
        raise MrError("ComfyUI base_dir is not configured.")
    if not base_dir.exists():
        raise MrError(f"ComfyUI base_dir does not exist: {base_dir}")

    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return added, updated, 0

    seen_paths = set()
    for subdir in sorted(subdirs):
        variant = subdir.name
        files = []
        for ext in extensions:
            files.extend(subdir.rglob(f"*{ext}"))
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

    # Mark disappeared files as not-local (with an audit event per row).
    if seen_paths:
        placeholders = ",".join("?" * len(seen_paths))
        gone = conn.execute(
            f"""SELECT id FROM models
                WHERE backend='comfyui' AND currently_local=1
                  AND file_path NOT IN ({placeholders})""",
            tuple(seen_paths),
        ).fetchall()
        if gone:
            conn.execute(
                f"""UPDATE models SET currently_local=0, last_updated=?
                    WHERE backend='comfyui' AND file_path NOT IN ({placeholders})""",
                (now, *seen_paths),
            )
    else:
        gone = conn.execute(
            "SELECT id FROM models WHERE backend='comfyui' AND currently_local=1"
        ).fetchall()
        if gone:
            conn.execute(
                "UPDATE models SET currently_local=0, last_updated=? WHERE backend='comfyui'",
                (now,),
            )
    for r in gone:
        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (r["id"], "scan_updated", now, json.dumps({"currently_local": 0})),
        )

    removed = _consolidate_comfyui_duplicates(conn, now)
    return added, updated, removed


def _consolidate_comfyui_duplicates(conn: sqlite3.Connection, now: str) -> int:
    """Merge duplicate comfyui rows that track the same file under different paths.

    The comfyui scan keys rows by exact file_path, so when a file moves drives
    (or gets pulled once per location) the same model ends up as several rows —
    one live copy plus stale duplicates. Group by (display_name, subdir) and:

      - exactly one live row  → the live row is the keeper; absorb missing
        metadata (civitai/hf source, base model, trigger words, tags, rating,
        notes) from every duplicate, then delete the duplicates.
      - no live row          → keep the row whose file still exists on disk (or
        the most recently updated), delete only duplicates whose file exists
        nowhere.
      - several live rows    → ambiguous; leave untouched.

    Returns the number of duplicate rows removed.
    """
    rows = conn.execute(
        "SELECT * FROM models WHERE backend='comfyui'"
    ).fetchall()

    groups: dict[tuple[str, str | None], list[sqlite3.Row]] = {}
    for r in rows:
        groups.setdefault((r["display_name"], r["variant"]), []).append(r)

    removed = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        local = [r for r in members if r["currently_local"]]
        if len(local) == 1:
            keeper = local[0]
            # Only absorb duplicates whose file is genuinely gone; a row whose
            # file still exists on disk is a real second copy, not stale.
            donors = [
                r for r in members
                if r is not keeper
                and not (r["file_path"] and Path(r["file_path"]).exists())
            ]
        elif len(local) == 0:
            on_disk = [
                r for r in members
                if r["file_path"] and Path(r["file_path"]).exists()
            ]
            keeper = (
                on_disk[0] if on_disk
                else max(members, key=lambda r: r["last_updated"] or "")
            )
            donors = []
            for r in members:
                if r is keeper:
                    continue
                still_exists = r["file_path"] and Path(r["file_path"]).exists()
                if not still_exists:
                    donors.append(r)
        else:
            continue

        if not donors:
            continue
        for donor in donors:
            merged = _absorb_comfyui_row(conn, keeper, donor, now)
            donor_id = donor["id"]
            conn.execute(
                "UPDATE events SET model_id=? WHERE model_id=?", (keeper["id"], donor_id)
            )
            conn.execute("DELETE FROM models WHERE id=?", (donor_id,))
            conn.execute(
                "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                (
                    keeper["id"], "scan_merged", now,
                    json.dumps({"absorbed_id": donor_id, "fields": merged}),
                ),
            )
            removed += 1
            # Refresh the keeper snapshot so a later donor merges onto the
            # fields written by earlier donors instead of overwriting them.
            keeper = conn.execute(
                "SELECT * FROM models WHERE id=?", (keeper["id"],)
            ).fetchone()
    if removed:
        conn.commit()
    return removed


def _absorb_comfyui_row(
    conn: sqlite3.Connection,
    keeper: sqlite3.Row,
    donor: sqlite3.Row,
    now: str,
) -> list[str]:
    """Copy a duplicate row's missing metadata into the keeper. Returns changed fields."""
    changes: dict[str, object] = {}

    for col in ("source_url", "base_model", "trigger_words", "hf_repo"):
        if not keeper[col] and donor[col]:
            changes[col] = donor[col]

    # Upgrade an "unknown" row to the donor's proper provenance when the donor
    # carries a CivitAI/HF source tag (otherwise enrich/rename would skip it).
    if (
        (keeper["source_type"] in (None, "comfyui_unknown"))
        and donor["source_type"] in ("comfyui_civitai", "comfyui_hf")
    ):
        changes["source_type"] = donor["source_type"]

    if donor["tags"]:
        ktags: set[str] = set()
        if keeper["tags"]:
            try:
                ktags = {str(t) for t in json.loads(keeper["tags"])}
            except json.JSONDecodeError:
                ktags = {str(keeper["tags"])}
        donor_tags: set[str] = set()
        try:
            donor_tags = {str(t) for t in json.loads(donor["tags"])}
        except json.JSONDecodeError:
            donor_tags = {str(donor["tags"])}
        merged_tags = ktags | donor_tags
        new_tags = json.dumps(sorted(merged_tags))
        if new_tags != keeper["tags"]:
            changes["tags"] = new_tags

    for col in ("rating", "notes"):
        if keeper[col] is None and donor[col] is not None:
            changes[col] = donor[col]

    if donor["times_downloaded"]:
        changes["times_downloaded"] = (keeper["times_downloaded"] or 0) + donor["times_downloaded"]
    if donor["first_seen"] and (not keeper["first_seen"] or donor["first_seen"] < keeper["first_seen"]):
        changes["first_seen"] = donor["first_seen"]
    if donor["last_used"] and (not keeper["last_used"] or donor["last_used"] > keeper["last_used"]):
        changes["last_used"] = donor["last_used"]

    if changes:
        changes["last_updated"] = now
        set_clauses = [f"{c}=?" for c in changes]
        conn.execute(
            f"UPDATE models SET {', '.join(set_clauses)} WHERE id=?",
            [*changes.values(), keeper["id"]],
        )
    return list(changes.keys())


# ─── Content identity / blacklist lookup ──────────────────────────────────────


def record_file_hash(conn: sqlite3.Connection, row, config: dict) -> str | None:
    """Compute and persist a file-backed model's SHA-256.

    Called before a file is deleted (delete/blacklist) so a model stays
    identifiable by content after its file is gone. Returns the digest, or None
    for Ollama models / unreadable files.
    """
    if row["file_hash"]:
        return row["file_hash"]
    if row["backend"] == "ollama" or not row["file_path"]:
        return None
    resolved = resolve_local_file_path(row["file_path"], config) or Path(row["file_path"])
    digest = _file_sha256(resolved)
    if digest:
        conn.execute("UPDATE models SET file_hash=? WHERE id=?", (digest, row["id"]))
    return digest


def find_blacklisted_by_hash(conn: sqlite3.Connection, file_hash: str | None):
    """Return a blacklisted model row with this SHA-256, or None."""
    if not file_hash:
        return None
    return conn.execute(
        "SELECT * FROM models WHERE status='blacklisted' AND file_hash=?",
        (file_hash,),
    ).fetchone()


def _civitai_source_patterns(ref: str) -> list[str]:
    """Build SQL LIKE patterns matching a CivitAI ref against stored source_url.

    Stored source_urls take several forms: an AIR tag (``...civitai:MODEL@VER``),
    or a browse URL (``.../models/MODEL?modelVersionId=VER`` or the
    ``?versionId=VER`` fallback). Return a pattern for each so the lookup works
    regardless of how the row was recorded.
    """
    air = parse_air_tag(ref)
    model_id = air["model_id"] if air else parse_civitai_model_id(ref)
    version_id = air["version_id"] if air else parse_civitai_version_id(ref)
    if model_id and version_id:
        return [f"%civitai:{model_id}@{version_id}%", f"%versionId={version_id}%"]
    if model_id:
        return [f"%civitai:{model_id}@%", f"%/models/{model_id}%"]
    if version_id:
        return [f"%versionId={version_id}%"]
    return []


def find_civitai_existing(conn: sqlite3.Connection, ref: str):
    """Find a registry row referenced by a CivitAI ref via its source_url.

    Blacklisted rows win so a re-pull of blacklisted content is blocked even
    when other (non-blacklisted) versions of the same model also exist.
    """
    patterns = _civitai_source_patterns(ref)
    if not patterns:
        return None
    where = " OR ".join("source_url LIKE ?" for _ in patterns)
    return conn.execute(
        f"""SELECT * FROM models WHERE {where}
           ORDER BY CASE status WHEN 'blacklisted' THEN 0 ELSE 1 END,
                    currently_local DESC, id DESC
           LIMIT 1""",
        tuple(patterns),
    ).fetchone()


def _has_blacklisted_hash(conn: sqlite3.Connection) -> bool:
    return _scalar(
        conn, "SELECT COUNT(*) FROM models WHERE status='blacklisted' AND file_hash IS NOT NULL"
    ) > 0


def guard_blacklisted_by_hash(
    conn: sqlite3.Connection,
    path: str | Path,
    *,
    allow_blacklisted: bool,
    expected_hash: str | None = None,
) -> str | None:
    """Delete a just-downloaded file and raise if its SHA-256 is blacklisted.

    Uses the source-provided `expected_hash` when known; otherwise hashes the
    file on disk, but only when a blacklisted model actually has a stored hash.
    """
    digest = expected_hash
    if not digest:
        if not _has_blacklisted_hash(conn):
            return None
        digest = _file_sha256(path)
    row = find_blacklisted_by_hash(conn, digest)
    if row and not allow_blacklisted:
        try:
            Path(path).unlink()
        except OSError:
            pass
        raise MrError(
            f"Downloaded file matches blacklisted model '{row['display_name']}' "
            f"(SHA-256 {digest}). Set allow_blacklisted=True to proceed."
        )
    return digest


def _require_not_blacklisted(row, allow_blacklisted: bool) -> None:
    """Raise if a matched row is blacklisted and the caller didn't opt in."""
    if row is not None and row["status"] == "blacklisted" and not allow_blacklisted:
        raise MrError(
            f"Model is blacklisted: {row['display_name']}. "
            "Set allow_blacklisted=True to proceed."
        )


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

    # A model can exist as several registry rows (re-scanned into a new install
    # dir, same model pulled more than once, etc.), and tags often end up on
    # only one of them — leaving the tag column empty on the row `mr list`
    # actually shows. Aggregate tags across every row sharing a (backend,
    # display_name) so any tagged duplicate keeps the tag column populated for
    # all of the listed rows.
    tags_by_key: dict[tuple[str, str], set[str]] = {}
    for agg in conn.execute(
        "SELECT backend, display_name, tags FROM models WHERE tags IS NOT NULL AND tags != ''"
    ).fetchall():
        key = (agg["backend"], agg["display_name"])
        raw = agg["tags"]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [raw]
        tags_by_key.setdefault(key, set()).update(str(t) for t in parsed)
    conn.close()

    result = [dict(r) for r in rows]
    for r in result:
        union = sorted(tags_by_key.get((r["backend"], r["display_name"]), set()))
        if union:
            r["tags"] = json.dumps(union)

    return result


def engine_show(
    config: dict | None = None, name: str | None = None, index: int | None = None
) -> dict:
    """Show full details (incl. tags, notes, link, recent events) for a model."""
    if config is None:
        config = load_config()
    if name is None:
        raise MrError("name is required")

    conn = get_db(config)
    init_db(conn)
    try:
        row = resolve_model(conn, name, index)
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
        result["link"] = get_model_link(row, config)

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
        r["link"] = get_model_link(r, config)
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
                source_type = get_source_type(oname)
                hf_repo, variant = parse_hf_repo_from_ollama(oname)
                # A bare 'namespace/model' name is an Ollama registry model, not
                # an HF repo. Only an explicit hf.co/huggingface.co prefix means
                # the model came from HuggingFace; otherwise the same namespace
                # would dedup-away unrelated tags (e.g. user/model:8b vs :70b).
                if source_type != "ollama_hf":
                    hf_repo = None
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
        if not bcfg.get("model_dir"):
            _log(log, f"[red]{bname} has no model_dir configured; skipping.[/red]")
            continue
        model_dir = Path(bcfg["model_dir"])
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
                conn.execute(
                    "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                    (row["id"], "scan_updated", now, json.dumps({"currently_local": 0})),
                )
                _log(log, f"  Marked not-local: {row['file_path']}")

    # ── ComfyUI ─────────────────────────────────────────────────────────────
    comfy_cfg = config["backends"].get("comfyui", {})
    if comfy_cfg.get("enabled", False):
        base_dir = comfy_cfg.get("base_dir", "")
        _log(log, f"\nScanning ComfyUI models in {base_dir}...")
        c_added, c_updated, c_merged = scan_comfyui(config, conn)
        added += c_added
        updated += c_updated
        _log(log, f"  ComfyUI: {c_added} added, {c_updated} updated")
        if c_merged:
            _log(log, f"  ComfyUI: merged {c_merged} duplicate row(s) into their live counterparts")

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
    civitai_cfg = config.get("civitai", {})
    civitai_token = os.environ.get(
        civitai_cfg.get("token_env_var", "CIVITAI_API_KEY")
    )

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

    # Local ComfyUI files with no known source: candidates for a SHA-256 lookup.
    # hash_checked is a negative cache so unmatched files aren't re-queried on
    # every run; --all ignores the cache and re-checks every candidate.
    hash_where = (
        "WHERE backend='comfyui' AND file_path IS NOT NULL AND source_url IS NULL"
    )
    if not enrich_all:
        hash_where += (
            " AND currently_local=1 AND hash_checked IS NULL AND "
            "(status IS NULL OR status NOT IN ('deleted', 'blacklisted'))"
        )
    hash_rows = conn.execute(
        f"SELECT id, display_name, file_path, file_hash, base_model, trigger_words "
        f"FROM models {hash_where} ORDER BY display_name"
    ).fetchall()

    if not rows and not hash_rows:
        conn.close()
        return {
            "updated": 0,
            "civitai_updated": 0,
            "civitai_hash_updated": 0,
            "timestamp": now,
        }

    if rows:
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

    # ── CivitAI by-hash enrichment (local files with unknown provenance) ─────
    hash_updated_count = 0
    if hash_rows:
        _log(log, f"Looking up {len(hash_rows)} ComfyUI file(s) by SHA-256...")
        for row in hash_rows:
            file_hash = row["file_hash"]
            if not file_hash:
                file_hash = _file_sha256(row["file_path"])
                if not file_hash:
                    _log(log, f"  Skipping {row['display_name']} (unreadable)")
                    continue
                # Persist the hash even on a miss so we only ever hash once.
                conn.execute(
                    "UPDATE models SET file_hash=? WHERE id=?", (file_hash, row["id"])
                )

            info = fetch_civitai_version_by_hash(
                file_hash, token=civitai_token, host=civitai_api_host(config)
            )
            if not info:
                conn.execute(
                    "UPDATE models SET hash_checked=? WHERE id=?", (now, row["id"])
                )
                conn.execute(
                    "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                    (row["id"], "enrich_hash_miss", now, json.dumps({"file_hash": file_hash})),
                )
                conn.commit()
                _log(log, f"  No CivitAI match for {row['display_name']}")
                time.sleep(0.5)
                continue

            trigger_words = info.get("trigger_words") or []
            trigger_words_json = json.dumps(trigger_words) if trigger_words else None

            h_updates = {"source_type": "comfyui_civitai", "hash_checked": now}
            model_id = info.get("model_id")
            version_id = info.get("version_id")
            if model_id and version_id:
                h_updates["source_url"] = (
                    f"https://{civitai_web_host(config)}/models/{model_id}?modelVersionId={version_id}"
                )
            if info.get("base_model") and not row["base_model"]:
                h_updates["base_model"] = info["base_model"]
            if trigger_words_json and not row["trigger_words"]:
                h_updates["trigger_words"] = trigger_words_json
            h_updates["last_updated"] = now

            set_clauses = [f"{k}=?" for k in h_updates.keys()]
            query_params = list(h_updates.values()) + [row["id"]]
            conn.execute(
                f"UPDATE models SET {', '.join(set_clauses)} WHERE id=?",
                query_params,
            )
            conn.execute(
                "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                (row["id"], "enrich_updated", now, json.dumps(h_updates)),
            )
            conn.commit()
            hash_updated_count += 1
            _log(log, f"  Matched CivitAI by hash: {row['display_name']} -> "
                      f"{info.get('model_name') or model_id or version_id}")

            time.sleep(0.5)

    # ── CivitAI enrichment (known source URLs) ──────────────────────────────
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
        _log(log, f"Enriching {len(civitai_rows)} CivitAI model(s)...")

        for row in civitai_rows:
            version_id = parse_civitai_version_id(row["source_url"])
            if not version_id:
                continue

            url = f"https://{civitai_api_host(config)}/api/v1/model-versions/{version_id}"
            params = {"token": civitai_token} if civitai_token else {}
            try:
                resp = requests.get(url, params=params, headers=_HTTP_HEADERS, timeout=15)
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
                update_params = list(c_updates.values()) + [row["id"]]
                conn.execute(
                    f"UPDATE models SET {', '.join(set_clauses)} WHERE id=?",
                    update_params,
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
        "civitai_hash_updated": hash_updated_count,
        "timestamp": now,
    }


def engine_rate(
    config: dict | None = None,
    name: str | None = None,
    rating: int | None = None,
    status: str | None = None,
    note: str | None = None,
    index: int | None = None,
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
        row = resolve_model(conn, name, index)

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
    index: int | None = None,
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
        row = resolve_model(conn, name, index)
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
    index: int | None = None,
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
        row = resolve_model(conn, name, index)
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
    index: int | None = None,
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
        row = resolve_model(conn, name, index)
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
    index: int | None = None,
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
        row = resolve_model(conn, name, index)

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
    index: int | None = None,
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
        row = resolve_model(conn, name, index)

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
    index: int | None = None,
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
        row = resolve_model(conn, name, index)

        # A hash only has lookup value after deletion if the model is
        # blacklisted (it is what blocks a re-pull by content). Hashing a
        # multi-GB checkpoint on a plain delete can take many minutes with no
        # visible progress, so only do it when it actually matters.
        if row["status"] == "blacklisted":
            record_file_hash(conn, row, config)
        removed = _delete_model_files(row, config, conn)
        if row["backend"] == "ollama" and not removed:
            raise MrError("Model still present in Ollama; registry not updated.")

        # Deleting a blacklisted model must not silently clear its blacklist.
        new_status = "blacklisted" if row["status"] == "blacklisted" else "deleted"
        conn.execute(
            "UPDATE models SET currently_local=0, status=?, last_updated=? WHERE id=?",
            (new_status, now, row["id"]),
        )
        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (row["id"], "delete", now, None),
        )
        conn.commit()
        return {
            "status": new_status,
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
    index: int | None = None,
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
        row = resolve_model(conn, name, index)
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
    index: int | None = None,
) -> dict:
    """Copy a model from one GGUF backend to another.

    A directory-model (files under <model_dir>/<Name>/) is copied wholesale to
    <dst_model_dir>/<Name>/; a top-level single-file model (<model_dir>/x.gguf)
    is copied to <dst_model_dir>/x.gguf. Pass index to disambiguate when more
    than one row matches.
    """
    _require_writes()
    _require_confirm(confirm)
    if config is None:
        config = load_config()
    if not src_backend or not dst_backend or not model_name:
        raise MrError("src_backend, dst_backend, and model_name are required")

    if src_backend not in config.get("backends", {}):
        raise MrError(f"Unknown source backend: {src_backend}")
    if dst_backend not in config.get("backends", {}):
        raise MrError(f"Unknown destination backend: {dst_backend}")
    if src_backend in ("ollama", "comfyui"):
        raise MrError(f"Cannot copy from {src_backend} - only GGUF backends are supported.")
    if dst_backend in ("ollama", "comfyui"):
        raise MrError(f"Cannot copy to {dst_backend} - use 'mr pull' instead.")

    src_model_dir = _backend_model_dir(config, src_backend)
    dst_model_dir = _backend_model_dir(config, dst_backend)
    if src_model_dir is None:
        raise MrError(f"Source backend '{src_backend}' has no model_dir configured.")
    if dst_model_dir is None:
        raise MrError(f"Destination backend '{dst_backend}' has no model_dir configured.")
    if not dst_model_dir.exists():
        raise MrError(f"Destination directory does not exist: {dst_model_dir}")

    conn = get_db(config)
    init_db(conn)
    now = now_iso()
    try:
        rows = conn.execute(
            """SELECT * FROM models
               WHERE display_name LIKE ? ESCAPE '\\' AND backend=?
               ORDER BY display_name, id""",
            (f"%{_escape_like(model_name)}%", src_backend),
        ).fetchall()

        if not rows:
            raise MrError(f"Model '{model_name}' not found in backend '{src_backend}'")
        if len(rows) == 1:
            row = rows[0]
        elif index is None:
            raise AmbiguousModel(model_name, [dict(r) for r in rows])
        elif index < 0 or index >= len(rows):
            raise MrError(f"Index {index} out of range for {len(rows)} matches")
        else:
            row = rows[index]

        if row["backend"] != src_backend:
            raise MrError(f"Model is not in backend '{src_backend}'.")

        src_path = Path(row["file_path"]) if row["file_path"] else None
        if not src_path or not src_path.exists():
            raise MrError(f"Source file not found: {row['file_path']}")

        src_dir = src_path.parent
        try:
            is_top_level = src_dir.resolve() == src_model_dir.resolve()
        except OSError:
            is_top_level = src_dir == src_model_dir

        if is_top_level:
            model_display = src_path.stem
            files_to_copy = [src_path]
            dst_dir = dst_model_dir
        else:
            model_display = src_dir.name
            files_to_copy = [f for f in src_dir.iterdir() if f.is_file()]
            dst_dir = dst_model_dir / src_dir.name

        copied_ggufs = [f for f in files_to_copy if f.suffix.lower() == ".gguf"]
        if not copied_ggufs:
            raise MrError("Source directory contains no GGUF files")

        dst_main = dst_dir / max(copied_ggufs, key=lambda f: f.stat().st_size).name
        if conn.execute(
            "SELECT 1 FROM models WHERE backend=? AND file_path=?",
            (dst_backend, str(dst_main)),
        ).fetchone():
            raise MrError(
                f"Destination already registered: {dst_main}. Remove it first."
            )

        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in files_to_copy:
            shutil.copy2(f, dst_dir / f.name)

        dst_files = [dst_dir / f.name for f in copied_ggufs]
        main_path = max(dst_files, key=lambda f: f.stat().st_size)
        total_size = sum(f.stat().st_size for f in dst_files)
        size_gb = round(total_size / (1024 ** 3), 2)
        variant = parse_variant_from_filename(main_path.name)

        conn.execute(
            """INSERT INTO models
               (display_name, hf_repo, variant, backend, source_type,
                file_path, size_gb, currently_local, first_seen, last_updated)
               VALUES (?,?,?,?,?,?,?,1,?,?)""",
            (model_display, row["hf_repo"], variant, dst_backend, dst_backend,
             str(main_path), size_gb, now, now),
        )
        mid = _last_id(conn)
        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (mid, "copy", now, json.dumps({
                "from": f"{src_backend}:{row['display_name']}",
                "to": f"{dst_backend}:{model_display}",
            })),
        )
        conn.commit()
        return {
            "status": "copied",
            "display_name": model_display,
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
    index: int | None = None,
) -> dict:
    """Rename a model on disk.

    llama.cpp-style backends: renames the directory containing the model,
    keeping the file name unchanged. ComfyUI: renames the *file* in place (the
    subdir must stay put) and enriches the new name with CivitAI metadata
    (version, base model, precision, civitai id) when available.
    """
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
        row = resolve_model(conn, name, index)

        if row["backend"] == "ollama":
            raise MrError("Ollama models don't have directories to rename.")
        if not row["file_path"]:
            raise MrError("No file_path recorded for this model - cannot rename on disk.")

        # If the resolved record points at a file that isn't on disk (stale
        # path left behind when the file moved/copied), fall back to a sibling
        # record with the same display_name whose file is actually on disk.
        # This keeps rename tracking the live file instead of failing on a
        # stale row.
        if not Path(row["file_path"]).exists():
            siblings = [
                r
                for r in conn.execute(
                    """SELECT * FROM models
                       WHERE display_name = ? AND id != ?
                         AND file_path IS NOT NULL AND file_path != ''""",
                    (row["display_name"], row["id"]),
                ).fetchall()
                if Path(r["file_path"]).exists()
            ]
            if len(siblings) == 1:
                row = siblings[0]
            elif len(siblings) > 1:
                raise AmbiguousModel(
                    row["display_name"],
                    [dict(r) for r in siblings],
                )

        old_path = Path(row["file_path"])
        old_dir = old_path.parent

        if not old_dir.exists():
            raise MrError(f"Directory not found on disk: {old_dir}")
        if not old_path.exists():
            raise MrError(f"File not found on disk: {old_path}")

        new_name = _sanitize_filename(new_name)
        clean_name = _strip_model_extension(new_name) or new_name
        if not clean_name:
            raise MrError("new_name must not be empty.")
        if row["backend"] == "comfyui":
            # ── ComfyUI: rename the file in place, enriched with metadata ──
            version_info = None
            selected_file = None
            model_id = None
            if row["source_type"] == "comfyui_civitai" and row["source_url"]:
                air = parse_air_tag(row["source_url"])
                model_id = air["model_id"] if air else None
                civitai_version_id = parse_civitai_version_id(row["source_url"])
                if civitai_version_id:
                    civitai_cfg = config.get("civitai", {})
                    token_env = civitai_cfg.get("token_env_var", "CIVITAI_API_KEY")
                    civitai_token = os.environ.get(token_env)
                    version_info = fetch_civitai_version_info(
                        civitai_version_id, token=civitai_token,
                        host=(re.search(_CIVITAI_DOMAIN_RE, row["source_url"]).group(0)
                              if re.search(_CIVITAI_DOMAIN_RE, row["source_url"]) else civitai_api_host(config)),
                    )
                    if version_info and air and air.get("file_id"):
                        selected_file = next(
                            (f for f in version_info["files"] if f["id"] == air["file_id"]), None
                        )
            new_stem = civitai_metadata_filename(clean_name, version_info, selected_file, model_id) \
                if row["source_type"] == "comfyui_civitai" else clean_name
            new_path = old_dir / f"{new_stem}{old_path.suffix}"
            if new_path == old_path:
                display_name = old_path.stem
            else:
                if new_path.exists():
                    raise MrError(f"File already exists: {new_path}")
                old_path.rename(new_path)
                display_name = new_path.stem
        else:
            model_dir = _backend_model_dir(config, row["backend"])
            try:
                is_top_level = model_dir is not None and old_dir.resolve() == model_dir.resolve()
            except OSError:
                is_top_level = model_dir is not None and old_dir == model_dir
            if is_top_level:
                # ── top-level single-file model: rename the file in place ──
                new_path = old_dir / f"{clean_name}{old_path.suffix}"
                if new_path == old_path:
                    display_name = old_path.stem
                else:
                    if new_path.exists():
                        raise MrError(f"File already exists: {new_path}")
                    old_path.rename(new_path)
                    display_name = new_path.stem
            else:
                # ── llama-style: rename the directory, keep the file name ──
                new_dir = old_dir.with_name(clean_name)
                if new_dir.exists():
                    raise MrError(f"Directory already exists: {new_dir}")
                old_dir.rename(new_dir)
                new_path = new_dir / old_path.name
                display_name = new_dir.name

        conn.execute(
            "UPDATE models SET file_path=?, display_name=?, last_updated=? WHERE id=?",
            (str(new_path), display_name, now, row["id"]),
        )
        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (row["id"], "rename", now, json.dumps({"old": str(old_path), "new": str(new_path)})),
        )
        conn.commit()
        return {
            "status": "renamed",
            "display_name": display_name,
            "old_path": str(old_path),
            "new_path": str(new_path),
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
    index: int | None = None,
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
        try:
            row = resolve_model(conn, name, index)
        except ModelNotFound:
            if not backend:
                raise MrError(
                    "Model not found in registry. backend is required to add a new blacklisted entry."
                )
            source_type = get_source_type(name)
            hf_repo, variant = parse_hf_repo_from_ollama(name)
            if source_type != "ollama_hf":
                hf_repo = None
            conn.execute(
                """INSERT INTO models
                   (display_name, hf_repo, variant, backend, source_type,
                    ollama_name, currently_local, first_seen, last_updated)
                   VALUES (?,?,?,?,?,?,0,?,?)""",
                (name, hf_repo, variant, backend, source_type,
                 name if backend == "ollama" else None, now, now),
            )
            mid = _last_id(conn)
            row = conn.execute("SELECT * FROM models WHERE id=?", (mid,)).fetchone()

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
            record_file_hash(conn, row, config)
            removed = _delete_model_files(row, config, conn)
            if not removed and row["backend"] == "ollama":
                raise MrError("Model still present in Ollama; registry not updated.")
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
    enrich_filename: bool = False,
    confirm: bool = False,
    log: list[str] | None = None,
    progress_callback=None,
) -> dict:
    """Pull a model from Ollama, HuggingFace, or CivitAI.

    ref may be: an ollama name, 'org/repo' or 'hf.co/org/repo:tag' (GGUF/ComfyUI),
    a CivitAI version/model URL or AIR tag, or 'civitai:<versionId>'.

    Ambiguous multi-file downloads require file_pattern (or download_all=True for GGUF).

    progress_callback: optional callable(downloaded_bytes, total_bytes, label) called
    as CivitAI download bytes arrive (used by the CLI to drive a progress bar; MCP
    clients can pass None).
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

        # Pre-pull blacklist/deleted check. A previously-seen entry is matched by
        # HF repo, pasted/partial name, or the CivitAI model@version/versionId in
        # its source_url; blacklisted files deleted before their hash was
        # recorded are still caught this way. Content identity (SHA-256) is
        # checked again after download (ComfyUI and GGUF) for files re-uploaded
        # under a different id.
        hf_repo, _ = parse_hf_repo_from_ollama(ref)
        existing = None
        if hf_repo:
            existing = conn.execute(
                """SELECT * FROM models WHERE hf_repo=?
                   ORDER BY CASE status WHEN 'blacklisted' THEN 0 ELSE 1 END,
                            currently_local DESC, id DESC
                   LIMIT 1""",
                (hf_repo,),
            ).fetchone()
        if not existing:
            existing = conn.execute(
                """SELECT * FROM models
                   WHERE display_name LIKE ? ESCAPE '\\' OR ollama_name=?
                   ORDER BY CASE status WHEN 'blacklisted' THEN 0 ELSE 1 END,
                            currently_local DESC, id DESC
                   LIMIT 1""",
                (f"%{_escape_like(ref)}%", ref),
            ).fetchone()
        if not existing:
            existing = find_civitai_existing(conn, ref)

        _require_not_blacklisted(existing, allow_blacklisted)
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
            model_dir = _backend_model_dir(config, backend)
            if model_dir is None:
                raise MrError(f"Backend '{backend}' has no model_dir configured.")
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
                _log(log, f"  Downloading {gguf_file}...")
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
                # Content check: refuse (and remove) a GGUF whose SHA-256 matches
                # a blacklisted model, even when re-pulled under a new repo name.
                guard_blacklisted_by_hash(
                    conn, local_path, allow_blacklisted=allow_blacklisted,
                )
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

            _dm_host = re.search(_CIVITAI_DOMAIN_RE, ref)
            _civitai_host = _dm_host.group(0) if _dm_host else civitai_api_host(config)

            civitai_version_id = parse_civitai_version_id(ref)
            _civitai_model_id = parse_civitai_model_id(ref)
            air = parse_air_tag(ref)
            air_file_id = air["file_id"] if air else None

            # Browse URL with model ID but no version ID → resolve via API
            if civitai_version_id is None and _civitai_model_id:
                _log(log, f"Fetching model info from CivitAI API (model {_civitai_model_id})...")
                civitai_version_id, _api_subdir = fetch_civitai_model_info(
                    _civitai_model_id, token=civitai_token, host=_civitai_host
                )
                if civitai_version_id:
                    _log(log, f"  Using latest version {civitai_version_id}")
                if not subdir and _api_subdir:
                    subdir = _api_subdir
                    _log(log, f"  Auto-detected subdir {subdir} from CivitAI model type")

            # Version details: exact file list (fileId), base model, trigger words
            version_info = None
            selected_file = None
            if civitai_version_id:
                version_info = fetch_civitai_version_info(
                    civitai_version_id, token=civitai_token, host=_civitai_host
                )
                if version_info:
                    if air_file_id:
                        selected_file = next(
                            (f for f in version_info["files"] if f["id"] == air_file_id), None
                        )
                    else:
                        selected_file = next(
                            (f for f in version_info["files"] if f.get("primary")), None
                        ) or (version_info["files"][0] if version_info["files"] else None)

            if not subdir:
                air = parse_air_tag(ref)
                if air:
                    subdir = comfyui_subdir_for_type(air["type"])
                    if subdir:
                        _log(log, f"  Auto-detected subdir {subdir} from AIR type '{air['type']}'")
            if not subdir and selected_file:
                subdir = comfyui_subdir_for_type(selected_file["type"])
                if subdir:
                    _log(log, f"  Auto-detected subdir {subdir} from CivitAI file type '{selected_file['type']}'")
            # HF downloads: many repos nest files under a ComfyUI-type dir
            # (e.g. 'loras/foo.safetensors'). Infer the subdir from --file so we
            # don't prompt and later double it to models/loras/loras/foo.
            if not subdir and file_pattern:
                _detected, _ = detect_comfyui_subdir(file_pattern)
                if _detected:
                    subdir = _detected
                    _log(log, f"  Auto-detected subdir {subdir} from --file path")

            # CivitAI needs a subdir up front; HF can still infer it from the
            # repo's file list (or the resolve URL path) further down.
            if not subdir and civitai_version_id:
                raise MrError("subdir is required for ComfyUI downloads (e.g. checkpoints, loras, vae)")

            dest_dir = base_dir / subdir if subdir else None
            if dest_dir:
                dest_dir.mkdir(parents=True, exist_ok=True)

            if civitai_version_id:
                # ── CivitAI download ─────────────────────────────────────────
                download_url = f"https://{_civitai_host}/api/download/models/{civitai_version_id}"
                params = {}
                if civitai_token:
                    params["token"] = civitai_token
                else:
                    _log(log, "Warning: No CivitAI API key found. Download may fail for gated models.")

                # Pin the download to the exact file when the tag specifies one
                # (AIR suffix '+<fileId>'); without it CivitAI serves the default
                # (primary) file, which may not be the requested variant.
                file_params = civitai_file_download_params(version_info, air_file_id)
                if file_params:
                    params.update(file_params)
                    _log(log, f"  Requesting exact file id {air_file_id} from the AIR tag")
                _log(log, f"Downloading from CivitAI (version {civitai_version_id})...")
                local_path, resumed_from = _civitai_download_to_file(
                    download_url,
                    params,
                    dest_dir,
                    version_id=civitai_version_id,
                    token_env=token_env,
                    selected_file=selected_file,
                    version_info=version_info,
                    filename=filename,
                    enrich_filename=enrich_filename,
                    air=air,
                    log=log,
                    progress_callback=progress_callback,
                )

                size_gb = round(local_path.stat().st_size / (1024 ** 3), 4)
                fpath = str(local_path)

                # Content check: refuse (and remove) a file whose SHA-256 matches
                # a blacklisted model, even if its id/name changed upstream.
                guard_blacklisted_by_hash(
                    conn, local_path, allow_blacklisted=allow_blacklisted,
                    expected_hash=(selected_file or {}).get("sha256"),
                )

                air = parse_air_tag(ref)
                model_id_for_url = (
                    (air["model_id"] if air else None)
                    or (version_info.get("model_id") if version_info else None)
                )
                civitai_url = civitai_source_url(
                    ref, civitai_version_id, model_id_for_url, config
                )

                # Store CivitAI metadata (trigger words / base model) on pull so
                # `mr show` can display them without a separate enrich step.
                base_model_val = version_info["base_model"] if version_info else None
                trigger_words_val = (
                    json.dumps(version_info["trigger_words"])
                    if version_info and version_info["trigger_words"] else None
                )

                existing_by_path = conn.execute(
                    "SELECT * FROM models WHERE file_path=?", (fpath,)
                ).fetchone()
                existing_comfy = existing if (existing and existing["backend"] == "comfyui") else None
                if existing_by_path or existing_comfy:
                    row_to_update = existing_by_path or existing_comfy
                    if row_to_update["status"] == "blacklisted" and not allow_blacklisted:
                        local_path.unlink(missing_ok=True)
                        raise MrError(
                            f"Model is blacklisted: {row_to_update['display_name']}. "
                            "Set allow_blacklisted=True to proceed."
                        )
                    conn.execute(
                        """UPDATE models
                           SET currently_local=1, times_downloaded=times_downloaded+1,
                               file_path=?, size_gb=?, last_used=?, last_updated=?,
                               source_type='comfyui_civitai', source_url=?,
                               base_model=COALESCE(?, base_model),
                               trigger_words=COALESCE(?, trigger_words),
                               status=CASE WHEN status='deleted' THEN NULL ELSE status END
                           WHERE id=?""",
                        (fpath, size_gb, now, now, civitai_url,
                         base_model_val, trigger_words_val, row_to_update["id"]),
                    )
                    mid = row_to_update["id"]
                else:
                    conn.execute(
                        """INSERT INTO models
                           (display_name, variant, backend, source_type, source_url,
                            file_path, size_gb, base_model, trigger_words,
                            currently_local, times_downloaded, first_seen, last_used, last_updated)
                           VALUES (?,?,?,?,?,?,?,?,?,1,1,?,?,?)""",
                        (local_path.stem, subdir, "comfyui", "comfyui_civitai", civitai_url,
                         fpath, size_gb, base_model_val, trigger_words_val, now, now, now),
                    )
                    mid = _last_id(conn)

                conn.execute(
                    "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                    (mid, "pull", now, json.dumps({
                        "civitai_version_id": civitai_version_id, "file": local_path.name,
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
                    "resumed_from": resumed_from,
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

                # Resolve/redirect the subdir from the repo's actual file path so
                # 'loras/foo.safetensors' lands in <base>/loras/foo.safetensors
                # instead of prompting and nesting <base>/loras/loras/foo.
                _detected, _remainder = detect_comfyui_subdir(chosen_file)
                if _detected and not subdir:
                    subdir = _detected
                    _log(log, f"  Auto-detected subdir {subdir} from repo path '{chosen_file}'")
                elif _detected and subdir != _detected:
                    _log(log, f"  Note: '{chosen_file}' is under '{_detected}'; keeping --subdir '{subdir}'")
                if not subdir:
                    raise MrError("subdir is required for ComfyUI downloads (e.g. checkpoints, loras, vae)")
                dest_dir = base_dir / subdir
                dest_dir.mkdir(parents=True, exist_ok=True)

                _hf_kwargs = dict(
                    repo_id=repo_id, filename=chosen_file,
                    local_dir=str(dest_dir), token=token,
                )
                if revision:
                    _hf_kwargs["revision"] = revision
                local_path = place_downloaded_file(
                    Path(hf_hub_download(**_hf_kwargs)), dest_dir,
                    _remainder if _detected else None,
                )

                size_gb = round(local_path.stat().st_size / (1024 ** 3), 4)
                fpath = str(local_path)

                # Content check: refuse (and remove) a file whose SHA-256 matches
                # a blacklisted model, even if its id/name changed upstream.
                guard_blacklisted_by_hash(
                    conn, local_path, allow_blacklisted=allow_blacklisted,
                )

                existing_by_path = conn.execute(
                    "SELECT * FROM models WHERE file_path=?", (fpath,)
                ).fetchone()
                existing_comfy = existing if (existing and existing["backend"] == "comfyui") else None
                if existing_by_path or existing_comfy:
                    row_to_update = existing_by_path or existing_comfy
                    if row_to_update["status"] == "blacklisted" and not allow_blacklisted:
                        local_path.unlink(missing_ok=True)
                        raise MrError(
                            f"Model is blacklisted: {row_to_update['display_name']}. "
                            "Set allow_blacklisted=True to proceed."
                        )
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
                 AND (status IS NULL OR status != 'blacklisted')
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
