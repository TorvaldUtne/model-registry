#!/usr/bin/env python3
"""Model Registry (mr) - CLI for the Model Registry engine.

This is a thin, interactive wrapper over the `mr_core` engine. All business
logic lives in `mr_core.py`; this file only handles the click command group,
keyboard prompts, and rich rendering so the same engine can be exposed over MCP
(`mr_mcp.py`) without shell/filesystem access.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

import mr_core
from mr_core import (
    MrError,
    ModelNotFound,
    AmbiguousModel,
    DestructiveOperation,
)

# The CLI is interactive — the user is at the terminal, so writes are enabled.
mr_core.set_writes_enabled(True)

console = Console()

__version__ = mr_core.__version__


def load_config():
    """Load config or print an error and exit."""
    try:
        return mr_core.load_config()
    except MrError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


# ─── Rendering helpers ────────────────────────────────────────────────────────


def _fmt_size(size_gb):
    if size_gb is None:
        return "-"
    if size_gb < 0.1:
        return f"{size_gb * 1024:.0f} MB"
    return f"{size_gb:.1f} GB"


def _fmt_date(config, iso):
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime(config.get("display", {}).get("date_format", "%Y-%m-%d"))
    except ValueError:
        return iso


def _parse_tags(tags):
    if not tags:
        return []
    try:
        return json.loads(tags)
    except json.JSONDecodeError:
        return [tags]


class _LiveLog(list):
    """A log list that prints each line to the console as it is appended."""

    def __init__(self, console_):
        super().__init__()
        self._console = console_

    def append(self, item):
        super().append(item)
        self._console.print(item)


def _handle_engine_error(e):
    """Print an engine error. Returns exit code."""
    if isinstance(e, ModelNotFound):
        console.print(f"[red]{e}[/red]")
        if e.suggestions:
            console.print("Did you mean:")
            for s in e.suggestions:
                console.print(f"  {s}")
    elif isinstance(e, AmbiguousModel):
        console.print(f"[yellow]{e}[/yellow]")
        for i, m in enumerate(e.matches, 1):
            console.print(f"  {i}. {m['display_name']}  [{m['backend']}]")
    else:
        console.print(f"[red]Error: {e}[/red]")
    return 1


def resolve_model_interactive(config, name):
    """Resolve a model, prompting the user to disambiguate if needed."""
    conn = mr_core.get_db(config)
    mr_core.init_db(conn)
    try:
        try:
            return mr_core.resolve_model(conn, name)
        except AmbiguousModel as e:
            console.print(f"[yellow]Multiple models match '{name}':[/yellow]")
            for i, m in enumerate(e.matches, 1):
                local = "(local)" if m["currently_local"] else "(remote)"
                console.print(f"  {i}. {m['display_name']}  [{m['backend']}] {local}")
            choice = click.prompt("Pick a number", type=click.IntRange(1, len(e.matches)))
            return mr_core.resolve_model(conn, name, index=choice - 1)
    finally:
        conn.close()


def _list_models(config, rows):
    """Render the model list table (shared by list and search)."""
    if not rows:
        console.print("No models found.")
        return

    date_fmt = config.get("display", {}).get("date_format", "%Y-%m-%d")
    backend_filter = config.get("_backend_filter")

    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
    table.add_column("Name", no_wrap=True)
    if backend_filter != "comfyui":
        table.add_column("Backend", width=9)
    if backend_filter == "comfyui":
        table.add_column("Type", width=14)
    table.add_column("Status", width=12)
    table.add_column("Rating", width=7, justify="center")
    table.add_column("Size", width=9, justify="right")
    table.add_column("Last Used", width=12)
    table.add_column("Tags", width=30)

    for row in rows:
        status_val = row["status"] or "unrated"
        color = mr_core.STATUS_COLORS.get(status_val, "white")
        rating_str = f"{row['rating']}/5" if row["rating"] else "-"
        size_str = _fmt_size(row["size_gb"])

        last_used = _fmt_date(config, row["last_used"])
        tags_str = ", ".join(_parse_tags(row["tags"]))

        not_local = "" if row["currently_local"] else " [dim](not local)[/dim]"
        row_cells = [f"[{color}]{row['display_name']}{not_local}[/{color}]"]
        if backend_filter != "comfyui":
            row_cells.append(row["backend"])
        if backend_filter == "comfyui":
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


# ─── CLI group ────────────────────────────────────────────────────────────────


@click.group()
def cli():
    """Model Registry - track, rate, and manage AI models."""
    pass


# ─── backends ─────────────────────────────────────────────────────────────────


@cli.command()
def backends():
    """List configured backend names."""
    config = load_config()
    console.print("[bold]Configured backends:[/bold]")
    for b in mr_core.engine_backends(config):
        console.print(f"  - {b['name']} [dim]({b['status']})[/dim]")


# ─── init ─────────────────────────────────────────────────────────────────────


@cli.command()
def init():
    """First-time setup wizard. Generates config.json."""
    from mr_core import CONFIG_FILE, CONFIG_EXAMPLE

    console.print("[bold]Model Registry Setup[/bold]\n")

    if CONFIG_FILE.exists():
        if not click.confirm("config.json already exists. Overwrite?", default=False):
            sys.exit(0)

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

    import subprocess
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
    except FileNotFoundError:
        console.print("[yellow]⚠ 'docker' command not found. Ollama backend will not work.[/yellow]")
    except subprocess.TimeoutExpired:
        console.print("[yellow]⚠ Docker exec timed out.[/yellow]")

    llamacpp_enabled = bool(gguf_dir)
    if gguf_dir:
        p = Path(gguf_dir)
        if p.exists() and p.is_dir():
            console.print(f"[green]✓ GGUF directory exists: {p}[/green]")
        else:
            console.print(f"[yellow]⚠ GGUF directory not found: {p}[/yellow]")

    comfyui_enabled = click.confirm("\nEnable ComfyUI backend?", default=False)
    comfyui_base_dir = ""
    if comfyui_enabled:
        default_comfy = defaults.get("backends", {}).get("comfyui", {}).get(
            "base_dir", r"X:\Models\comfy"
        )
        comfyui_base_dir = click.prompt("ComfyUI models base directory", default=default_comfy)
        p = Path(comfyui_base_dir)
        if p.exists() and p.is_dir():
            console.print(f"[green]✓ ComfyUI directory exists: {p}[/green]")
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

    conn = mr_core.get_db(config)
    mr_core.init_db(conn)
    conn.close()
    console.print(f"[green]✓ Database initialized at {mr_core.get_db_path(config)}[/green]")
    console.print("\n[bold]Setup complete.[/bold] Run [bold]mr scan[/bold] to populate the registry.")


# ─── scan ─────────────────────────────────────────────────────────────────────


@cli.command()
def scan():
    """Scan Ollama, llama.cpp, and ComfyUI backends, update the registry."""
    config = load_config()
    log = []
    try:
        result = mr_core.engine_scan(config, log=log)
    except DestructiveOperation as e:
        console.print(f"[red]{e}[/red]")
        return
    for line in log:
        console.print(line)
    console.print(
        f"\n[green]Scan complete.[/green] "
        f"Added: [bold]{result['added']}[/bold]  Updated: [bold]{result['updated']}[/bold]"
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
    rows = mr_core.engine_list(config, backend=backend, status=status, unrated=unrated, show_all=show_all, deleted=deleted)
    config["_backend_filter"] = backend
    _list_models(config, rows)


# ─── show ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("model")
def show(model):
    """Show full details for a model."""
    config = load_config()
    try:
        data = mr_core.engine_show(config, name=model)
    except (MrError, DestructiveOperation) as e:
        sys.exit(_handle_engine_error(e))
    row = data

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
    if row["link"]:
        lines.append(f"[bold]Link:[/bold]       {row['link']}")
    if row["variant"]:
        lines.append(f"[bold]Variant:[/bold]    {row['variant']}")
    if row["ollama_name"]:
        lines.append(f"[bold]Ollama:[/bold]     {row['ollama_name']}")
    if row["file_path"]:
        lines.append(f"[bold]File:[/bold]       {row['file_path']}")
    lines.append(
        f"[bold]Size:[/bold]       {row['size_gb']:.2f} GB" if row["size_gb"] is not None
        else "[bold]Size:[/bold]       -"
    )
    try:
        context_window = row["context_window"]
        if context_window:
            try:
                lines.append(f"[bold]Context:[/bold]    {int(context_window):,} tokens")
            except (ValueError, TypeError):
                lines.append(f"[bold]Context:[/bold]    {context_window} tokens")
    except (KeyError, IndexError, TypeError):
        pass
    lines.append(f"[bold]Local:[/bold]      {'yes' if row['currently_local'] else 'no'}")
    lines.append(f"[bold]Downloads:[/bold]  {row['times_downloaded']}")
    if row["tags"]:
        lines.append(f"[bold]Tags:[/bold]       {', '.join(row['tags'])}")
    if row["first_seen"]:
        lines.append(f"[bold]First seen:[/bold] {row['first_seen']}")
    if row["last_used"]:
        lines.append(f"[bold]Last used:[/bold]  {row['last_used']}")
    if row["last_updated"]:
        lines.append(f"[bold]Updated:[/bold]    {row['last_updated']}")

    if any(row[k] is not None for k in ("param_count", "architecture", "hf_downloads", "hf_likes", "hf_last_modified")):
        lines.append("")
        lines.append("[bold dim]--- HF Metadata ---[/bold dim]")
        if row["param_count"] is not None:
            p_val = row["param_count"]
            try:
                lines.append(f"[bold]Params:[/bold]     {int(p_val):,}")
            except (ValueError, TypeError):
                lines.append(f"[bold]Params:[/bold]     {p_val}")
        if row["architecture"]:
            lines.append(f"[bold]Arch:[/bold]       {row['architecture']}")
        if row["hf_downloads"] is not None:
            try:
                lines.append(f"[bold]DL count:[/bold]   {int(row['hf_downloads']):,}")
            except (ValueError, TypeError):
                lines.append(f"[bold]DL count:[/bold]   {row['hf_downloads']}")
        if row["hf_likes"] is not None:
            try:
                lines.append(f"[bold]Likes:[/bold]      {int(row['hf_likes']):,}")
            except (ValueError, TypeError):
                lines.append(f"[bold]Likes:[/bold]      {row['hf_likes']}")
        if row["hf_last_modified"]:
            lines.append(f"[bold]HF updated:[/bold] {row['hf_last_modified']}")

    if row["base_model"] or row["trigger_words"]:
        lines.append("")
        lines.append("[bold dim]--- CivitAI Metadata ---[/bold dim]")
        if row["base_model"]:
            lines.append(f"[bold]Base model:[/bold] {row['base_model']}")
        if row["trigger_words"]:
            words = _parse_tags(row["trigger_words"])
            lines.append(f"[bold]Triggers:[/bold]   {', '.join(words)}")

    if row["notes"]:
        lines.append("")
        lines.append("[bold dim]--- Notes ---[/bold dim]")
        lines.append(row["notes"].strip())

    events = row.get("recent_events") or []
    if events:
        lines.append("")
        lines.append("[bold dim]--- Recent Events ---[/bold dim]")
        for e in events:
            lines.append(f"  [dim]{e['timestamp']}[/dim]  {e['event_type']}  {e['detail'] or ''}")

    console.print(
        Panel("\n".join(lines), title=f"[bold]{row['display_name']}[/bold]", expand=False)
    )


# ─── rate ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("model")
def rate(model):
    """Interactively rate a model (1-5), set status, add optional note."""
    config = load_config()
    row = resolve_model_interactive(config, model)

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

    try:
        result = mr_core.engine_rate(
            config, name=row["display_name"], rating=new_rating,
            status=new_status, note=note_text or None,
        )
    except (MrError, DestructiveOperation) as e:
        sys.exit(_handle_engine_error(e))
    console.print(f"[green]✓ Rated {new_rating}/5, status: {new_status}[/green]")


# ─── status ───────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("model")
@click.argument("status", type=click.Choice(["active", "unrated", "blacklisted", "deleted", "on_hold", "testing", "keep", "favorite"]))
def status(model, status):
    """Set a model's status without requiring a rating."""
    config = load_config()
    row = resolve_model_interactive(config, model)
    try:
        mr_core.engine_status(config, name=row["display_name"], status=status)
    except (MrError, DestructiveOperation) as e:
        sys.exit(_handle_engine_error(e))
    console.print(f"[green]✓ Status set to: {status}[/green]")


# ─── note ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("model")
@click.argument("text", nargs=-1, required=False)
def note(model, text):
    """Append a timestamped note to a model."""
    config = load_config()
    row = resolve_model_interactive(config, model)
    note_text = " ".join(text) if text else click.prompt("Note")
    try:
        mr_core.engine_note(config, name=row["display_name"], text=note_text)
    except (MrError, DestructiveOperation) as e:
        sys.exit(_handle_engine_error(e))
    console.print("[green]✓ Note added.[/green]")


# ─── touch ────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("model")
def touch(model):
    """Update last_used to now (use when you ran a model outside this tool)."""
    config = load_config()
    row = resolve_model_interactive(config, model)
    try:
        mr_core.engine_touch(config, name=row["display_name"])
    except (MrError, DestructiveOperation) as e:
        sys.exit(_handle_engine_error(e))
    console.print(f"[green]✓ last_used updated for {row['display_name']}[/green]")


# ─── report ───────────────────────────────────────────────────────────────────


@cli.command()
def report():
    """Summary: total models, GB by backend, unrated list, blacklisted list, cross-backend duplicates."""
    config = load_config()
    r = mr_core.engine_report(config)

    console.print(f"\n[bold]Model Registry Report[/bold]")
    console.print(f"Total models tracked: [bold]{r['total']}[/bold]  |  Local storage: [bold]{r['total_gb']:.1f} GB[/bold]")

    console.print("\n[bold]By Backend (local):[/bold]")
    for br in r["by_backend"]:
        console.print(f"  {br['backend']:12s}  {br['cnt']} models   {(br['gb'] or 0):.1f} GB")

    console.print("\n[bold]By Status:[/bold]")
    for s in r["by_status"]:
        console.print(f"  {(s['status'] or 'unrated'):12s}  {s['cnt']}")

    if r["unrated"]:
        console.print(f"\n[bold]Unrated Models ({len(r['unrated'])}):[/bold]")
        for m in r["unrated"]:
            size_str = f"  {m['size_gb']:.1f} GB" if m["size_gb"] is not None else ""
            console.print(f"  [{m['backend']}] {m['display_name']}{size_str}")

    if r["blacklisted"]:
        console.print(f"\n[bold red]Blacklisted ({len(r['blacklisted'])}):[/bold red]")
        for m in r["blacklisted"]:
            rating_str = f"rating={m['rating']}" if m["rating"] else "unrated"
            first_note = (m["notes"] or "").strip().splitlines()[0] if m["notes"] else ""
            console.print(f"  {m['display_name']}  ({rating_str})  {first_note}")

    if r["duplicates"]:
        console.print(f"\n[bold yellow]Cross-backend Duplicates ({len(r['duplicates'])}):[/bold yellow]")
        for d in r["duplicates"]:
            console.print(f"  [yellow]{d['hf_repo']}[/yellow]")
            console.print(f"    {d['names']}")
    else:
        console.print("\n[dim]No cross-backend duplicates detected.[/dim]")


# ─── enrich ───────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--all", "enrich_all", is_flag=True, default=False, help="Enrich all models with hf_repo, not just local ones")
def enrich(enrich_all):
    """Fetch additional metadata (context window, parameters, architecture, stats) from HuggingFace Hub."""
    config = load_config()
    log = []
    try:
        result = mr_core.engine_enrich(config, enrich_all=enrich_all, log=log)
    except DestructiveOperation as e:
        console.print(f"[red]{e}[/red]")
        return
    for line in log:
        console.print(line)
    console.print(
        f"\n[green]Enrichment complete.[/green] Updated [bold]{result['updated']}[/bold] of [bold]HF[/bold] model(s)"
        f" and [bold]{result['civitai_updated']}[/bold] CivitAI model(s)."
    )


# ─── pull ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("ref")
@click.argument("variant", required=False)
@click.option("--backend", type=str, default=None)
@click.option("--file", "file_pattern", default=None, help="Glob pattern for file in HF repo")
@click.option("--subdir", default=None, help="ComfyUI subdir to save into (e.g. checkpoints, loras)")
@click.option("--rename", "enrich_filename", is_flag=True, default=False,
              help="CivitAI ComfyUI pulls: save under a metadata-enriched name (name + version + base model + fp + civitai id)")
def pull(ref, variant, backend, file_pattern, subdir, enrich_filename):
    """Pull a model. Warns if blacklisted or previously deleted.

    For ComfyUI models, --subdir is required. Supports HuggingFace repos
    (org/repo format) and CivitAI downloads (civitai:<versionId> or CivitAI URL).

    Returns True on success, False on failure.
    """
    config = load_config()

    def _run(**overrides):
        log = _LiveLog(console)
        progress = None
        task = None

        def _cb(done, total, label):
            nonlocal progress, task
            if progress is None:
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console,
                    transient=True,
                )
                progress.start()
                task = progress.add_task(f"Downloading {label}", total=total)
            if total:
                progress.update(task, completed=done)
            else:
                progress.update(task, total=None, completed=done)

        try:
            return mr_core.engine_pull(
                config,
                ref=ref,
                variant=variant,
                backend=backend,
                file_pattern=overrides.get("file_pattern", file_pattern),
                subdir=overrides.get("subdir", subdir),
                download_all=overrides.get("download_all", False),
                filename=overrides.get("filename", None),
                allow_blacklisted=overrides.get("allow_blacklisted", False),
                enrich_filename=overrides.get("enrich_filename", enrich_filename),
                confirm=True,
                log=log,
                progress_callback=_cb,
            )
        finally:
            if progress is not None:
                progress.stop()

    while True:
        try:
            _run()
            return
        except MrError as e:
            if e.details.get("kind") == "gguf_multi":
                files = e.details["files"]
                console.print(f"Multiple GGUF files in [bold]{ref}[/bold]:")
                for i, f in enumerate(files, 1):
                    console.print(f"  {i}. {f}")
                if click.confirm("Download all GGUF files to a subdirectory?", default=True):
                    _run(download_all=True)
                    return
                user_pattern = click.prompt("Enter pattern (e.g., *Q4_K_M*)")
                _run(file_pattern=user_pattern)
                return
            if e.details.get("kind") == "comfyui_hf_multi":
                files = e.details["files"]
                console.print(f"Multiple model files in [bold]{ref}[/bold]:")
                for i, f in enumerate(files, 1):
                    console.print(f"  {i}. {f}")
                idx = click.prompt("Pick a number", type=click.IntRange(1, len(files)))
                _run(file_pattern=files[idx - 1])
                return
            if "blacklisted" in str(e) and "allow_blacklisted" in str(e):
                if not click.confirm("Model is BLACKLISTED. Proceed anyway?", default=False):
                    return
                _run(allow_blacklisted=True)
                return
            if "subdir is required" in str(e):
                subdir = click.prompt("ComfyUI subdir (e.g. checkpoints, loras, vae)")
                continue
            console.print(f"[red]Error: {e}[/red]")
            return


# ─── tag / untag ──────────────────────────────────────────────────────────────


@cli.command()
@click.argument("model")
@click.argument("tags", nargs=-1)
def tag(model, tags):
    """Add tags to a model. Multiple tags can be provided."""
    config = load_config()
    row = resolve_model_interactive(config, model)
    try:
        result = mr_core.engine_tag(config, name=row["display_name"], tags=list(tags))
    except (MrError, DestructiveOperation) as e:
        sys.exit(_handle_engine_error(e))
    console.print(f"[green]✓ Tags updated: {', '.join(result['tags'])}[/green]")


@cli.command()
@click.argument("model")
@click.argument("tags", nargs=-1)
def untag(model, tags):
    """Remove tags from a model. If no tags provided, removes all tags."""
    config = load_config()
    row = resolve_model_interactive(config, model)
    try:
        result = mr_core.engine_untag(config, name=row["display_name"], tags=list(tags))
    except (MrError, DestructiveOperation) as e:
        sys.exit(_handle_engine_error(e))
    if result["tags"]:
        console.print(f"[green]✓ Tags updated: {', '.join(result['tags'])}[/green]")
    else:
        console.print("[green]✓ All tags removed.[/green]")


# ─── removeall ────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would be removed without actually removing")
def removeall(dry_run):
    """Purge all models with status='deleted' from the registry (hard delete)."""
    config = load_config()
    try:
        preview = mr_core.engine_removeall(config, dry_run=True, confirm=False)
    except DestructiveOperation as e:
        console.print(f"[yellow]{e}[/yellow]")
        return

    if preview["status"] == "no_deleted_models":
        console.print("[green]No deleted models found.[/green]")
        return
    if dry_run:
        console.print(f"[yellow]Would purge {preview['count']} model(s) from the registry:[/yellow]")
        for m in preview["models"]:
            console.print(f"  - {m['display_name']} [{m['backend']}]")
        return

    if not click.confirm(
        f"Purge {preview['count']} model(s) from the registry? This cannot be undone.", default=False
    ):
        return
    final = mr_core.engine_removeall(config, dry_run=False, confirm=True)
    console.print(f"\n[green]✓ Purged {final['count']} model(s) from registry.[/green]")


# ─── restore ──────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would be downloaded without downloading")
def restore(dry_run):
    """Re-download all missing ComfyUI models that have a known source."""
    config = load_config()
    log = []
    try:
        plan = mr_core.engine_restore(config, dry_run=True, confirm=False, log=log)
    except DestructiveOperation as e:
        console.print(f"[yellow]{e}[/yellow]")
        return

    for line in log:
        console.print(line)

    if plan["status"] in ("no_missing_models", "no_restorable"):
        return

    if dry_run:
        return

    if not click.confirm(f"\nDownload {plan['count']} model(s)?", default=True):
        return

    log.clear()
    final = mr_core.engine_restore(config, dry_run=False, confirm=True, log=log)
    for line in log:
        console.print(line)
    console.rule()
    if final["failed"]:
        console.print(f"[red]Failed to restore {len(final['failed'])} model(s):[/red]")
        for name in final["failed"]:
            console.print(f"  - {name}")
    else:
        console.print(f"[green]All {final['count']} model(s) restored.[/green]")


# ─── copy ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("src_backend")
@click.argument("dst_backend")
@click.argument("model_name")
def copy(src_backend, dst_backend, model_name):
    """Copy a model from one GGUF backend to another.

    Example: mr copy llamaserver llamacpp Gembrain
    """
    config = load_config()
    conn = mr_core.get_db(config)
    mr_core.init_db(conn)
    try:
        rows = conn.execute(
            """SELECT * FROM models
               WHERE display_name LIKE ? AND backend=?
               ORDER BY display_name""",
            (f"%{model_name}%", src_backend),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        console.print(f"[red]Model '{model_name}' not found in backend '{src_backend}'[/red]")
        return
    if len(rows) > 1:
        console.print(f"[yellow]Multiple models match '{model_name}' in '{src_backend}':[/yellow]")
        for i, r in enumerate(rows, 1):
            console.print(f"  {i}. {r['display_name']}")
        choice = click.prompt("Pick a number", type=click.IntRange(1, len(rows)))
        chosen = rows[choice - 1]
    else:
        chosen = rows[0]

    try:
        result = mr_core.engine_copy(
            config, src_backend=src_backend, dst_backend=dst_backend,
            model_name=chosen["display_name"], confirm=True,
        )
    except (MrError, DestructiveOperation) as e:
        sys.exit(_handle_engine_error(e))
    console.print(f"[green]✓ Copied to {result['display_name']} ({result['size_gb']:.2f} GB). Registry updated.[/green]")


# ─── rename ───────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("model")
@click.argument("new_name")
def rename(model, new_name):
    """Rename a model on disk.

    For llamacpp-style backends, renames the directory (keeping the file name
    unchanged). For ComfyUI, renames the FILE in place (the subdir stays put)
    and enriches the new name with CivitAI metadata when available.
    """
    config = load_config()
    row = resolve_model_interactive(config, model)
    try:
        result = mr_core.engine_rename(config, name=row["display_name"], new_name=new_name, confirm=True)
    except (MrError, DestructiveOperation) as e:
        if "not found on disk" in str(e):
            console.print(f"[yellow]{e}[/yellow]")
            return
        sys.exit(_handle_engine_error(e))
    kind = "Renamed file" if row["backend"] == "comfyui" else "Renamed directory"
    console.print(f"[green]✓ {kind}: {result['old_path']} → {result['new_path']}[/green]")


# ─── delete / remove / blacklist ──────────────────────────────────────────────


@cli.command()
@click.argument("model")
def delete(model):
    """Delete a model from Ollama or disk. Keeps DB record, sets status=deleted."""
    config = load_config()
    row = resolve_model_interactive(config, model)
    if not click.confirm(f"Delete '{row['display_name']}'?", default=False):
        return
    try:
        result = mr_core.engine_delete(config, name=row["display_name"], confirm=True)
    except (MrError, DestructiveOperation) as e:
        sys.exit(_handle_engine_error(e))
    console.print("[green]✓ Registry updated (status=deleted).[/green]")


@cli.command()
@click.argument("model")
def remove(model):
    """Remove a model from the registry (hard delete). Completely removes the DB entry."""
    config = load_config()
    row = resolve_model_interactive(config, model)
    if not click.confirm(f"Delete '{row['display_name']}' from registry?", default=False):
        return
    try:
        mr_core.engine_remove(config, name=row["display_name"], confirm=True)
    except (MrError, DestructiveOperation) as e:
        sys.exit(_handle_engine_error(e))
    console.print("[green]✓ Model deleted from registry.[/green]")


@cli.command()
@click.argument("model")
@click.argument("reason", nargs=-1, required=False)
def blacklist(model, reason):
    """Set a model's status to blacklisted, record reason, and delete it.

    Works even if the model isn't in the registry yet (creates a new entry).
    REASON can be passed inline or left blank to be prompted.
    """
    config = load_config()
    reason_text = " ".join(reason) if reason else None

    conn = mr_core.get_db(config)
    mr_core.init_db(conn)
    try:
        try:
            row = mr_core.resolve_model(conn, model)
        except AmbiguousModel as e:
            console.print(f"[yellow]Multiple models match '{model}':[/yellow]")
            for i, m in enumerate(e.matches, 1):
                console.print(f"  {i}. {m['display_name']}  [{m['backend']}]")
            choice = click.prompt("Pick a number", type=click.IntRange(1, len(e.matches)))
            row = mr_core.resolve_model(conn, model, index=choice - 1)
        except ModelNotFound:
            row = None
    finally:
        conn.close()

    if row is None:
        if not click.confirm("Add it as a new blacklisted entry?", default=True):
            return
        bl_backends = ["ollama"] + mr_core.get_gguf_backend_names(config) + ["comfyui"]
        backend = click.prompt("Backend", type=click.Choice(bl_backends), default="ollama")
    else:
        backend = row["backend"]

    if reason_text is None:
        reason_text = click.prompt("Reason for blacklisting")

    resolved_name = row["display_name"] if row is not None else model
    try:
        result = mr_core.engine_blacklist(
            config, name=resolved_name, reason=reason_text, backend=backend, confirm=True
        )
    except (MrError, DestructiveOperation) as e:
        sys.exit(_handle_engine_error(e))
    console.print(f"[red]✓ {result['display_name']} blacklisted.[/red]")


# ─── search ───────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("term")
def search(term):
    """Search local registry by name, hf_repo, notes, or tags."""
    config = load_config()
    rows = mr_core.engine_search(config, term=term)
    if not rows:
        console.print(f"No results for '{term}'.")
        return
    for row in rows:
        color = mr_core.STATUS_COLORS.get(row["status"] or "unrated", "white")
        rating_str = f"{row['rating']}/5" if row["rating"] else "unrated"
        size_str = f"  {row['size_gb']:.1f} GB" if row["size_gb"] is not None else ""
        console.print(
            f"[{color}]{row['display_name']}[/{color}]  "
            f"[dim][{row['backend']}][/dim]  {rating_str}{size_str}"
        )
        if row["link"]:
            console.print(f"  [dim]{row['link']}[/dim]")


# ─── Entry point ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    if len(sys.argv) == 1:
        console.print(f"[bold]Model Registry v{__version__}[/bold]")
        console.print("Run [bold]mr --help[/bold] for available commands.")
        sys.exit(0)
    cli()
