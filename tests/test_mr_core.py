"""Regression tests for the mr engine.

These run entirely against a temporary SQLite DB and temp model directories, so
they never touch the real registry or the user's model files.
"""

import json
from pathlib import Path

import pytest

import mr_core


@pytest.fixture(autouse=True)
def _enable_writes():
    mr_core.set_writes_enabled(True)
    yield
    mr_core.set_writes_enabled(False)


def make_config(tmp_path: Path, *, gguf2: bool = False, comfy: bool = True) -> dict:
    backends = {"ollama": {"enabled": False}}
    (tmp_path / "gguf").mkdir(exist_ok=True)
    backends["llamacpp"] = {
        "enabled": True, "model_dir": str(tmp_path / "gguf"), "extensions": [".gguf"],
    }
    if gguf2:
        (tmp_path / "gguf2").mkdir(exist_ok=True)
        backends["llamacpp2"] = {
            "enabled": True, "model_dir": str(tmp_path / "gguf2"), "extensions": [".gguf"],
        }
    if comfy:
        (tmp_path / "comfy").mkdir(exist_ok=True)
        backends["comfyui"] = {
            "enabled": True, "base_dir": str(tmp_path / "comfy"),
            "extensions": [".safetensors", ".ckpt", ".pt", ".pth", ".bin"],
        }
    return {
        "registry_db": str(tmp_path / "registry.db"),
        "backends": backends,
        "huggingface": {"token_env_var": "HF_TOKEN"},
        "civitai": {"web_host": "civitai.red", "api_host": "civitai.com"},
    }


def insert_model(conn, display_name, *, backend="comfyui", variant=None,
                 file_path=None, currently_local=1, status=None, source_url=None,
                 source_type="comfyui_unknown"):
    conn.execute(
        """INSERT INTO models
           (display_name, variant, backend, source_type, source_url, file_path,
            currently_local, status, first_seen, last_updated)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (display_name, variant, backend, source_type, source_url, file_path,
         currently_local, status, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
    )
    conn.commit()


# ─── resolve_model ────────────────────────────────────────────────────────────


def test_resolve_model_ambiguous_requires_index(tmp_path):
    cfg = make_config(tmp_path)
    conn = mr_core.get_db(cfg)
    mr_core.init_db(conn)
    insert_model(conn, "same", file_path="/a/same.safetensors")
    insert_model(conn, "same", file_path="/b/same.safetensors")

    with pytest.raises(mr_core.AmbiguousModel) as ei:
        mr_core.resolve_model(conn, "same")
    assert len(ei.value.matches) == 2

    r0 = mr_core.resolve_model(conn, "same", index=0)
    r1 = mr_core.resolve_model(conn, "same", index=1)
    assert r0["id"] != r1["id"]
    with pytest.raises(mr_core.MrError):
        mr_core.resolve_model(conn, "same", index=5)
    conn.close()


def test_resolve_model_escapes_like_wildcards(tmp_path):
    cfg = make_config(tmp_path)
    conn = mr_core.get_db(cfg)
    mr_core.init_db(conn)
    insert_model(conn, "sdxl_vae")
    insert_model(conn, "sdxlXvae")

    # Without escaping, '_' would match 'X' and this would be ambiguous.
    row = mr_core.resolve_model(conn, "sdxl_vae")
    assert row["display_name"] == "sdxl_vae"
    conn.close()


# ─── scan_comfyui ─────────────────────────────────────────────────────────────


def test_scan_comfyui_blank_base_dir_raises(tmp_path):
    cfg = make_config(tmp_path)
    cfg["backends"]["comfyui"]["base_dir"] = ""
    conn = mr_core.get_db(cfg)
    mr_core.init_db(conn)
    with pytest.raises(mr_core.MrError):
        mr_core.scan_comfyui(cfg, conn)
    conn.close()


def test_scan_comfyui_empty_base_dir_returns_tuple(tmp_path):
    cfg = make_config(tmp_path)
    conn = mr_core.get_db(cfg)
    mr_core.init_db(conn)
    # No category subdirs: must return a 3-tuple, not crash.
    added, updated, merged = mr_core.scan_comfyui(cfg, conn)
    assert (added, updated, merged) == (0, 0, 0)
    conn.close()


def test_scan_tracks_nested_comfy_file(tmp_path):
    cfg = make_config(tmp_path)
    nested = tmp_path / "comfy" / "loras" / "author" / "foo.safetensors"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"x")

    mr_core.engine_scan(cfg)
    mr_core.engine_scan(cfg)

    conn = mr_core.get_db(cfg)
    row = conn.execute(
        "SELECT variant, currently_local FROM models WHERE file_path=?", (str(nested),)
    ).fetchone()
    assert row is not None
    assert row["variant"] == "loras"
    assert row["currently_local"] == 1
    conn.close()


# ─── engine_copy ──────────────────────────────────────────────────────────────


def test_copy_top_level_single_file(tmp_path):
    cfg = make_config(tmp_path, gguf2=True)
    (tmp_path / "gguf" / "top.gguf").write_bytes(b"hello")
    (tmp_path / "gguf" / "DirModel").mkdir()
    (tmp_path / "gguf" / "DirModel" / "a.gguf").write_bytes(b"a" * 10)
    mr_core.engine_scan(cfg)

    res = mr_core.engine_copy(
        cfg, src_backend="llamacpp", dst_backend="llamacpp2",
        model_name="top", confirm=True,
    )
    assert (tmp_path / "gguf2" / "top.gguf").exists()
    assert not (tmp_path / "gguf2" / "DirModel").exists()
    assert res["display_name"] == "top"


def test_copy_directory_model(tmp_path):
    cfg = make_config(tmp_path, gguf2=True)
    d = tmp_path / "gguf" / "DirModel"
    d.mkdir()
    (d / "a.gguf").write_bytes(b"a" * 10)
    (d / "b.gguf").write_bytes(b"b" * 20)
    mr_core.engine_scan(cfg)

    res = mr_core.engine_copy(
        cfg, src_backend="llamacpp", dst_backend="llamacpp2",
        model_name="DirModel", confirm=True,
    )
    assert (tmp_path / "gguf2" / "DirModel" / "a.gguf").exists()
    assert (tmp_path / "gguf2" / "DirModel" / "b.gguf").exists()
    assert res["display_name"] == "DirModel"


def test_copy_unknown_backend_rejected(tmp_path):
    cfg = make_config(tmp_path, gguf2=True)
    (tmp_path / "gguf" / "top.gguf").write_bytes(b"x")
    mr_core.engine_scan(cfg)
    with pytest.raises(mr_core.MrError):
        mr_core.engine_copy(
            cfg, src_backend="llamacpp", dst_backend="nope",
            model_name="top", confirm=True,
        )


def test_copy_duplicate_destination_rejected(tmp_path):
    cfg = make_config(tmp_path, gguf2=True)
    (tmp_path / "gguf" / "top.gguf").write_bytes(b"x")
    mr_core.engine_scan(cfg)
    mr_core.engine_copy(cfg, src_backend="llamacpp", dst_backend="llamacpp2",
                        model_name="top", confirm=True)
    with pytest.raises(mr_core.MrError):
        mr_core.engine_copy(cfg, src_backend="llamacpp", dst_backend="llamacpp2",
                            model_name="top", confirm=True)


# ─── engine_rename ────────────────────────────────────────────────────────────


def test_rename_top_level_file(tmp_path):
    cfg = make_config(tmp_path)
    (tmp_path / "gguf" / "top.gguf").write_bytes(b"x")
    mr_core.engine_scan(cfg)

    res = mr_core.engine_rename(cfg, name="top", new_name="top2", confirm=True)
    assert (tmp_path / "gguf" / "top2.gguf").exists()
    assert not (tmp_path / "gguf" / "top.gguf").exists()
    assert res["display_name"] == "top2"


def test_rename_directory_uses_dir_name(tmp_path):
    cfg = make_config(tmp_path)
    d = tmp_path / "gguf" / "DirModel"
    d.mkdir()
    (d / "a.gguf").write_bytes(b"a" * 10)
    mr_core.engine_scan(cfg)

    res = mr_core.engine_rename(cfg, name="DirModel", new_name="DirModel2", confirm=True)
    assert (tmp_path / "gguf" / "DirModel2" / "a.gguf").exists()
    # display_name must track the directory, not the shard file stem.
    assert res["display_name"] == "DirModel2"


def test_rename_strips_pasted_extension(tmp_path):
    cfg = make_config(tmp_path)
    (tmp_path / "gguf" / "top.gguf").write_bytes(b"x")
    mr_core.engine_scan(cfg)
    res = mr_core.engine_rename(cfg, name="top", new_name="renamed.gguf", confirm=True)
    assert (tmp_path / "gguf" / "renamed.gguf").exists()
    assert res["display_name"] == "renamed"


# ─── consolidation / civitai / hf / config ────────────────────────────────────


def test_consolidate_keeps_on_disk_donor(tmp_path):
    cfg = make_config(tmp_path)
    keep = tmp_path / "comfy" / "loras" / "keep.safetensors"
    keep.parent.mkdir(parents=True)
    keep.write_bytes(b"k")
    dup = tmp_path / "comfy" / "checkpoints" / "keep.safetensors"
    dup.parent.mkdir(parents=True)
    dup.write_bytes(b"d")

    conn = mr_core.get_db(cfg)
    mr_core.init_db(conn)
    insert_model(conn, "keep", variant="loras", file_path=str(keep), currently_local=1)
    insert_model(conn, "keep", variant="loras", file_path=str(dup), currently_local=0)

    mr_core._consolidate_comfyui_duplicates(conn, mr_core.now_iso())
    conn.commit()
    rows = conn.execute(
        "SELECT file_path FROM models WHERE display_name='keep'"
    ).fetchall()
    assert any(r["file_path"] == str(dup) for r in rows), "on-disk donor was wrongly deleted"
    conn.close()


def test_find_civitai_existing_matches_browse_url(tmp_path):
    cfg = make_config(tmp_path)
    conn = mr_core.get_db(cfg)
    mr_core.init_db(conn)
    insert_model(
        conn, "citmodel", source_type="comfyui_civitai",
        source_url="https://civitai.com/models/123?modelVersionId=456",
        currently_local=0, status="blacklisted",
    )
    found = mr_core.find_civitai_existing(
        conn, "https://civitai.red/models/123?modelVersionId=456"
    )
    assert found is not None and found["display_name"] == "citmodel"
    conn.close()


def test_parse_civitai_version_id_variants():
    assert mr_core.parse_civitai_version_id("civitai:123") == "123"
    assert mr_core.parse_civitai_version_id(
        "https://civitai.com/models/1?modelVersionId=456"
    ) == "456"
    assert mr_core.parse_civitai_version_id(
        "https://civitai.red/models?versionId=789"
    ) == "789"


def test_get_hf_metadata_reads_card_data_dataclass(monkeypatch):
    class FakeCard:
        architectures = ["LlamaForCausalLM"]
        model_name = "x"

    class FakeInfo:
        safetensors = {"total": 123}
        card_data = FakeCard()
        downloads = 5
        likes = 1
        lastModified = None

    class FakeApi:
        def __init__(self, *a, **k):
            pass

        def model_info(self, repo_id):
            return FakeInfo()

    monkeypatch.setattr(mr_core, "HfApi", FakeApi)
    meta = mr_core.get_hf_metadata("org/repo", None)
    assert meta["architecture"] == "LlamaForCausalLM"
    assert meta["param_count"] == 123


def test_load_config_bad_json_raises_merror(tmp_path, monkeypatch):
    bad = tmp_path / "config.json"
    bad.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(mr_core, "CONFIG_FILE", bad)
    with pytest.raises(mr_core.MrError):
        mr_core.load_config()


def test_blacklist_indexed(tmp_path):
    cfg = make_config(tmp_path)
    conn = mr_core.get_db(cfg)
    mr_core.init_db(conn)
    insert_model(conn, "dup", file_path="/a/dup.safetensors")
    insert_model(conn, "dup", file_path="/b/dup.safetensors")

    with pytest.raises(mr_core.AmbiguousModel):
        mr_core.engine_blacklist(cfg, name="dup", reason="x", confirm=True)

    res = mr_core.engine_blacklist(cfg, name="dup", reason="x", confirm=True, index=1)
    assert res["status"] == "blacklisted"
    rows = conn.execute(
        "SELECT status FROM models WHERE display_name='dup' ORDER BY id"
    ).fetchall()
    assert rows[0]["status"] != "blacklisted"
    assert rows[1]["status"] == "blacklisted"
    conn.close()
