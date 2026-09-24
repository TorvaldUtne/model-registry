#!/usr/bin/env python3
"""Model Registry (mr) MCP server.

Exposes the `mr_core` engine as Model Context Protocol (MCP) tools over the
local network (Streamable HTTP), so agents (e.g. Hermes) can manage the model
registry WITHOUT being given shell or filesystem access.

Modes:
  readonly (default)  — registers only read-only tools (list/show/report/...).
                        Every write operation also raises even if invoked, as a
                        second line of defense.
  readwrite           — pass --read-write (or set "mcp": {"mode": "readwrite"}
                        in config.json). Registers write tools too. Destructive
                        operations (pull/delete/remove/removeall/rename/copy/
                        blacklist/restore) additionally require confirm=True.

No authentication is performed — suitable only for a trusted local network.

Example:
  venv\\Scripts\\python.exe mr_mcp.py --read-write --host 0.0.0.0 --port 8321
"""

import argparse
import sys

from mcp import MCPError
from mcp.server.mcpserver import MCPServer

import mr_core
from mr_core import MrError, DestructiveOperation


def _load_config():
    try:
        return mr_core.load_config()
    except MrError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)


def _call(fn, **kwargs):
    """Call an engine function, converting engine errors to MCP errors."""
    try:
        return fn(**kwargs)
    except (MrError, DestructiveOperation) as e:
        raise MCPError(code=-32603, message=str(e), data=e.details or {})
    except Exception as e:
        raise MCPError(code=-32603, message=f"Internal error: {e}")


def _with_log(fn, **kwargs):
    """Call an engine function and attach its progress log lines to the result."""
    log = []
    result = _call(fn, log=log, **kwargs)
    if log:
        result["_log"] = log
    return result


# ─── Tools ────────────────────────────────────────────────────────────────────


def register_read_tools(mcp: MCPServer) -> None:
    """Register read-only tools (list, show, report, search, backends)."""

    @mcp.tool()
    def mr_backends() -> list[dict]:
        """Return the configured backend names and whether each is enabled.

        Backends are typically: ollama, llamacpp, llamaserver, comfyui.
        """
        return _call(mr_core.engine_backends)

    @mcp.tool()
    def mr_list(
        backend: str | None = None,
        status: str | None = None,
        unrated: bool = False,
        show_all: bool = False,
        deleted: bool = False,
    ) -> list[dict]:
        """List models in the registry.

        By default returns only locally installed, non-blacklisted models. Set
        show_all=True to include non-local and blacklisted models, or deleted=True
        to list only status='deleted' models. Filter with optional backend and
        status. The status values are: active|unrated|blacklisted|deleted|
        on_hold|testing|keep|favorite.
        """
        return _call(
            mr_core.engine_list,
            backend=backend, status=status, unrated=unrated,
            show_all=show_all, deleted=deleted,
        )

    @mcp.tool()
    def mr_show(name: str) -> dict:
        """Show full details for one model (by partial name match).

        Returns tags, notes, link, plus the 10 most recent events. If the name
        is ambiguous, an error with the matching models is returned.
        """
        return _call(mr_core.engine_show, name=name)

    @mcp.tool()
    def mr_report() -> dict:
        """Return a summary report: total models, local storage GB per backend,
        counts by status, unrated list, blacklisted list, and cross-backend
        duplicates.
        """
        return _call(mr_core.engine_report)

    @mcp.tool()
    def mr_search(term: str) -> list[dict]:
        """Search the registry by name, hf_repo, notes, or tags. Returns
        matching model rows.
        """
        return _call(mr_core.engine_search, term=term)


def build_write_tools(mcp: MCPServer) -> None:
    """Register write tools. Only registered in read-write mode; the engine still
    enforces its own read-only gate as a second layer of defense."""

    @mcp.tool()
    def mr_scan() -> dict:
        """Scan the Ollama, llama.cpp, and ComfyUI backends and update the
        registry (adds/updates/marks-not-local models). Writes to the DB.
        """
        return _with_log(mr_core.engine_scan)

    @mcp.tool()
    def mr_enrich(enrich_all: bool = False) -> dict:
        """Fetch additional metadata (context window, params, architecture, HF and
        CivitAI stats) from HuggingFace Hub / CivitAI for tracked models. Writes
        to the DB. enrich_all=True also enriches non-local/deleted models.
        """
        return _with_log(mr_core.engine_enrich, enrich_all=enrich_all)

    @mcp.tool()
    def mr_rate(
        name: str,
        rating: int,
        status: str | None = None,
        note: str | None = None,
    ) -> dict:
        """Rate a model 1-5, optionally set its status and append a note.
        status: active|unrated|blacklisted|deleted|on_hold|testing|keep|favorite.
        """
        return _call(mr_core.engine_rate, name=name, rating=rating, status=status, note=note)

    @mcp.tool()
    def mr_status(name: str, status: str) -> dict:
        """Set a model's status without requiring a rating.
        status: active|unrated|blacklisted|deleted|on_hold|testing|keep|favorite.
        """
        return _call(mr_core.engine_status, name=name, status=status)

    @mcp.tool()
    def mr_note(name: str, text: str) -> dict:
        """Append a timestamped note to a model."""
        return _call(mr_core.engine_note, name=name, text=text)

    @mcp.tool()
    def mr_touch(name: str) -> dict:
        """Update a model's last_used timestamp to now (call when you used a model
        outside this tool)."""
        return _call(mr_core.engine_touch, name=name)

    @mcp.tool()
    def mr_tag(name: str, tags: list[str]) -> dict:
        """Add tags to a model."""
        return _call(mr_core.engine_tag, name=name, tags=tags)

    @mcp.tool()
    def mr_untag(name: str, tags: list[str] | None = None) -> dict:
        """Remove tags from a model. If tags is empty, removes all tags."""
        return _call(mr_core.engine_untag, name=name, tags=tags)

    @mcp.tool()
    def mr_pull(
        ref: str,
        backend: str | None = None,
        variant: str | None = None,
        file_pattern: str | None = None,
        subdir: str | None = None,
        download_all: bool = False,
        filename: str | None = None,
        allow_blacklisted: bool = False,
        enrich_filename: bool = False,
        confirm: bool = True,
    ) -> dict:
        """Pull/download a model. Downloads files to disk and may create registry
        entries, so it is DESTRUCTIVE: pass confirm=True to proceed.

        ref: an ollama name; an HF repo 'org/repo' (optionally with ':tag');
        a CivitAI URL, AIR tag, or 'civitai:<versionId>'.

        backend auto-detects (ollama by default). For llama.cpp backends pass
        e.g. backend='llamacpp' and optionally variant/file_pattern. For ComfyUI
        pass backend='comfyui' and subdir='checkpoints'|'loras'|'vae'|...
        enrich_filename: for CivitAI ComfyUI pulls, save the file under a
        metadata-enriched name (name + version + base model + fp + civitai id).
        """
        return _with_log(
            mr_core.engine_pull,
            ref=ref, backend=backend, variant=variant, file_pattern=file_pattern,
            subdir=subdir, download_all=download_all, filename=filename,
            allow_blacklisted=allow_blacklisted, enrich_filename=enrich_filename,
            confirm=confirm,
        )

    @mcp.tool()
    def mr_delete(name: str, confirm: bool = True) -> dict:
        """Delete a model from Ollama/disk. Keeps the registry row (sets
        status='deleted') but deletes the files. DESTRUCTIVE; pass confirm=True.
        """
        return _call(mr_core.engine_delete, name=name, confirm=confirm)

    @mcp.tool()
    def mr_remove(name: str, confirm: bool = True) -> dict:
        """Remove a model's registry entry entirely (hard DB delete). Does NOT
        delete files on disk. DESTRUCTIVE; pass confirm=True.
        """
        return _call(mr_core.engine_remove, name=name, confirm=confirm)

    @mcp.tool()
    def mr_removeall(dry_run: bool = False, confirm: bool = True) -> dict:
        """Purge all models with status='deleted' from the registry (hard DB
        delete). Use dry_run=True first to preview. DESTRUCTIVE; pass confirm=True.
        """
        return _call(mr_core.engine_removeall, dry_run=dry_run, confirm=confirm)

    @mcp.tool()
    def mr_rename(name: str, new_name: str, confirm: bool = True) -> dict:
        """Rename a model on disk. For llama.cpp-style backends, renames the
        directory (file name unchanged). For ComfyUI, renames the FILE in place
        and enriches it with CivitAI metadata (version, base model, fp, civitai
        id) when known. DESTRUCTIVE; pass confirm=True.
        """
        return _call(mr_core.engine_rename, name=name, new_name=new_name, confirm=confirm)

    @mcp.tool()
    def mr_copy(
        src_backend: str,
        dst_backend: str,
        model_name: str,
        confirm: bool = True,
    ) -> dict:
        """Copy a model from one GGUF backend to another (copies files, adds a
        new registry entry). DESTRUCTIVE; pass confirm=True.
        """
        return _call(
            mr_core.engine_copy,
            src_backend=src_backend, dst_backend=dst_backend,
            model_name=model_name, confirm=confirm,
        )

    @mcp.tool()
    def mr_blacklist(
        name: str,
        reason: str | None = None,
        backend: str | None = None,
        confirm: bool = True,
    ) -> dict:
        """Set a model's status to blacklisted, record a reason, and delete its
        files if local. Works even if the model isn't in the registry yet (in
        which case backend is required: ollama|llamacpp|...|comfyui).
        DESTRUCTIVE; pass confirm=True.
        """
        return _call(
            mr_core.engine_blacklist,
            name=name, reason=reason, backend=backend, confirm=confirm,
        )

    @mcp.tool()
    def mr_restore(dry_run: bool = False, confirm: bool = True) -> dict:
        """Re-download all missing ComfyUI models that have a known source
        (CivitAI or HuggingFace). Use dry_run=True first to preview.
        DESTRUCTIVE; pass confirm=True.
        """
        return _with_log(mr_core.engine_restore, dry_run=dry_run, confirm=confirm)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    config = _load_config()
    mcp_cfg = config.get("mcp", {})

    parser = argparse.ArgumentParser(
        prog="mr_mcp",
        description="Model Registry MCP server (Streamable HTTP over the local network, no auth).",
    )
    parser.add_argument(
        "--read-write", action="store_true", default=None,
        help="Register write tools as well as reads. Default is read-only.",
    )
    parser.add_argument(
        "--readonly", action="store_true", default=False,
        help="Force read-only mode even if config.json enables read-write.",
    )
    parser.add_argument("--host", default=None, help="Interface to bind (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind (default 8321)")
    args = parser.parse_args()

    # Mode: CLI --read-write or config mcp.mode; --readonly always wins
    if args.readonly:
        mode = "readonly"
    elif args.read_write:
        mode = "readwrite"
    else:
        mode = mcp_cfg.get("mode", "readonly")

    readwrite = mode == "readwrite"
    mr_core.set_writes_enabled(readwrite)

    host = args.host or mcp_cfg.get("host", "0.0.0.0")
    port = args.port or int(mcp_cfg.get("port", 8321))

    name = "model-registry"
    mcp = MCPServer(
        name,
        title="Model Registry (mr)",
description=(
            "Track, rate, and manage AI models across Ollama, llama.cpp, and "
            "ComfyUI backends. " + ("READ-WRITE mode." if readwrite else "READ-ONLY mode.")
        ),
        version=mr_core.__version__,
        warn_on_duplicate_tools=True,
    )

    register_read_tools(mcp)
    if readwrite:
        build_write_tools(mcp)

    print(f"Model Registry MCP server starting", file=sys.stderr)
    print(f"  URL:    http://{host}:{port}/mcp", file=sys.stderr)
    print(f"  Mode:   {'READ-WRITE' if readwrite else 'READ-ONLY'}", file=sys.stderr)
    print("  Tools:  reads + writes" if readwrite else "  Tools:  reads only", file=sys.stderr)
    print("  NOTE:   No authentication - trusted local network only.", file=sys.stderr)

    try:
        mcp.run(
            transport="streamable-http",
            host=host,
            port=port,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()