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
            hf_last_modified  TEXT
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


# ─── Model matching ───────────────────────────────────────────────────────────

def find_model(conn, name, *, prompt_on_multiple=True):
    """Return a single models row matching name (partial on display_name and ollama_name)."""
    rows = conn.execute(
        """SELECT * FROM models
           WHERE display_name LIKE ? OR ollama_name LIKE ?
           ORDER BY display_name""",
        (f"%{name}%", f"%{name}%"),
    ).fetchall()

    if not rows:
        console.print(f"[red]No model matching '{name}' found.[/red]")
        # Offer fuzzy suggestions from all names
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

    if not prompt_on_multiple:
        return rows  # caller handles list

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


# ─── Link helpers ────────────────────────────────────────────────────────────

def get_model_link(row):
    """Return a browse URL for a model row, or None."""
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
}


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
                "enabled": False,
                "model_dirs": {},
            },
        },
        "huggingface": {
            "token_env_var": hf_env_var,
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
    """Scan Ollama and llama.cpp backends, update the registry."""
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
                    mid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
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

    # ── llama.cpp ────────────────────────────────────────────────────────────
    llamacpp_cfg = config["backends"].get("llamacpp", {})
    if llamacpp_cfg.get("enabled", False):
        model_dir = Path(llamacpp_cfg.get("model_dir", ""))
        extensions = llamacpp_cfg.get("extensions", [".gguf"])
        console.print(f"\nScanning llama.cpp models in [bold]{model_dir}[/bold]...")
        if not model_dir.exists():
            console.print(f"[red]llama.cpp model_dir does not exist: {model_dir}[/red]")
        else:
            files = []
            for ext in extensions:
                files.extend(model_dir.glob(f"*{ext}"))
            console.print(f"  Found {len(files)} GGUF file(s).")

            seen_paths = set()
            for f in sorted(files):
                fpath = str(f)
                seen_paths.add(fpath)
                size_gb = round(f.stat().st_size / (1024 ** 3), 2)
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
                    display_name = f.stem  # filename without .gguf extension
                    conn.execute(
                        """INSERT INTO models
                           (display_name, variant, backend, source_type,
                            file_path, size_gb, currently_local, first_seen, last_updated)
                           VALUES (?,?,?,?,?,?,1,?,?)""",
                        (display_name, variant, "llamacpp", "llamacpp", fpath, size_gb, now, now),
                    )
                    mid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    conn.execute(
                        "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
                        (mid, "scan_added", now, json.dumps({"file_path": fpath})),
                    )
                    added += 1

            # Mark disappeared files as not-local
            db_llamacpp = conn.execute(
                "SELECT id, file_path FROM models WHERE backend='llamacpp' AND currently_local=1"
            ).fetchall()
            for row in db_llamacpp:
                if row["file_path"] not in seen_paths:
                    conn.execute(
                        "UPDATE models SET currently_local=0, last_updated=? WHERE id=?",
                        (now, row["id"]),
                    )
                    console.print(f"  [yellow]Marked not-local: {row['file_path']}[/yellow]")

    conn.commit()
    conn.close()
    console.print(
        f"\n[green]Scan complete.[/green] "
        f"Added: [bold]{added}[/bold]  Updated: [bold]{updated}[/bold]"
    )


# ─── list ─────────────────────────────────────────────────────────────────────

@cli.command("list")
@click.option("--backend", type=click.Choice(["ollama", "llamacpp"]), default=None)
@click.option(
    "--status",
    type=click.Choice(["active", "unrated", "blacklisted", "deleted", "on_hold"]),
    default=None,
)
@click.option("--unrated", is_flag=True, default=False, help="Show only models with no rating")
def list_models(backend, status, unrated):
    """List models in the registry."""
    config = load_config()
    conn = get_db(config)

    query = "SELECT * FROM models WHERE 1=1"
    params = []
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
    table.add_column("Backend", width=9)
    table.add_column("Status", width=12)
    table.add_column("Rating", width=7, justify="center")
    table.add_column("Size", width=9, justify="right")
    table.add_column("Last Used", width=12)

    for row in rows:
        status_val = row["status"] or "unrated"
        color = STATUS_COLORS.get(status_val, "white")
        rating_str = f"{row['rating']}/5" if row["rating"] else "-"
        size_str = f"{row['size_gb']:.1f} GB" if row["size_gb"] else "-"

        last_used = row["last_used"]
        if last_used:
            try:
                dt = datetime.fromisoformat(last_used.replace("Z", "+00:00"))
                last_used = dt.strftime(date_fmt)
            except Exception:
                pass

        not_local = "" if row["currently_local"] else " [dim](not local)[/dim]"
        table.add_row(
            f"[{color}]{row['display_name']}{not_local}[/{color}]",
            row["backend"],
            f"[{color}]{status_val}[/{color}]",
            rating_str,
            size_str,
            last_used or "-",
        )

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
    lines.append(f"[bold]Size:[/bold]       {row['size_gb']:.2f} GB" if row["size_gb"] else "[bold]Size:[/bold]       -")
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
        type=click.Choice(["active", "unrated", "blacklisted", "deleted", "on_hold"]),
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

    total = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    total_gb = conn.execute(
        "SELECT SUM(size_gb) FROM models WHERE currently_local=1"
    ).fetchone()[0] or 0

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
            size_str = f"  {r['size_gb']:.1f} GB" if r["size_gb"] else ""
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
        no_hf = conn.execute(
            "SELECT COUNT(*) FROM models WHERE backend='llamacpp' AND hf_repo IS NULL AND currently_local=1"
        ).fetchone()[0]
        if no_hf:
            console.print(
                f"[dim]  ({no_hf} llamacpp model(s) have no hf_repo — "
                "run 'mr enrich' for accurate duplicate detection)[/dim]"
            )

    conn.close()


# ─── pull ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("ref")
@click.option("--backend", type=click.Choice(["ollama", "llamacpp"]), default=None)
@click.option("--file", "file_pattern", default=None, help="Glob pattern for GGUF file in HF repo")
def pull(ref, backend, file_pattern):
    """Pull a model. Warns if blacklisted or previously deleted."""
    config = load_config()
    conn = get_db(config)
    init_db(conn)
    now = now_iso()

    # Auto-detect backend
    if backend is None:
        backend = "llamacpp" if ref.endswith(".gguf") else "ollama"

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
            mid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (mid, "pull", now, json.dumps({"ref": ref})),
        )
        conn.commit()
        console.print("[green]✓ Pull complete. Registry updated.[/green]")

    # ── llama.cpp / HF download ──────────────────────────────────────────────
    elif backend == "llamacpp":
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            console.print("[red]huggingface_hub not installed. Run: pip install huggingface_hub[/red]")
            conn.close()
            return

        model_dir = Path(config["backends"]["llamacpp"]["model_dir"])
        token_env = config["huggingface"]["token_env_var"]
        token = os.environ.get(token_env)
        repo_id = ref

        if "/" not in repo_id:
            console.print("[red]For llamacpp, ref must be 'org/repo' format.[/red]")
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
                (local_path.stem, repo_id, variant, "llamacpp", "llamacpp",
                 str(local_path), size_gb, now, now, now),
            )
            mid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            "INSERT INTO events (model_id, event_type, timestamp, detail) VALUES (?,?,?,?)",
            (mid, "pull", now, json.dumps({"repo_id": repo_id, "file": chosen_file})),
        )
        conn.commit()
        console.print(
            f"[green]✓ Downloaded to {local_path} ({size_gb:.2f} GB). Registry updated.[/green]"
        )

    conn.close()


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

    elif row["backend"] == "llamacpp":
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
        backend = click.prompt(
            "Backend", type=click.Choice(["ollama", "llamacpp"]), default="ollama"
        )
        conn.execute(
            """INSERT INTO models
               (display_name, hf_repo, variant, backend, source_type,
                ollama_name, currently_local, first_seen, last_updated)
               VALUES (?,?,?,?,?,?,0,?,?)""",
            (model, hf_repo, variant, backend, get_source_type(model),
             model if backend == "ollama" else None, now, now),
        )
        mid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
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
        elif row["backend"] == "llamacpp" and row["file_path"]:
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
        size_str = f"  {row['size_gb']:.1f} GB" if row["size_gb"] else ""
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
