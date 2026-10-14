import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bootstrap", ROOT / "scripts" / "bootstrap.py")
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


@pytest.fixture
def source_root(tmp_path):
    lock = json.loads((ROOT / "upstreams.lock.json").read_text())
    (tmp_path / "upstreams.lock.json").write_text(json.dumps(lock))
    for spec in lock.values():
        archive = tmp_path / spec["archive"]
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.symlink_to(ROOT / spec["archive"])
    return tmp_path


def test_clean_restore_without_git_preserves_local_edits(source_root, monkeypatch):
    def no_git(*args, **kwargs):
        pytest.fail("Clean source restoration must not invoke Git")

    monkeypatch.setattr(bootstrap.subprocess, "check_output", no_git)
    bootstrap.restore_upstreams(source_root)
    vendor = source_root / "vendor"
    assert (vendor / "RSI-Jev" / "rsijev" / "arch.py").is_file()
    assert (vendor / "RSI-Jev" / "serve" / "decider.py").is_file()
    assert (vendor / "PlayJev" / "playjev" / "env.py").is_file()
    assert (vendor / "PlayJev" / "games" / "_shared" / "fonts" / "fonts.conf").is_file()
    for game in ("snake", "2048", "tetris", "pacman", "racer", "invaders",
                 "sokoban", "mario", "flappy", "breakout"):
        directory = vendor / "PlayJev" / "games" / game
        spec = json.loads((directory / "pj.json").read_text())
        assert (directory / spec["entry"]).is_file()
        assert (directory / "pj_hook.js").is_file()
    for name in ("RSI-Jev", "PlayJev"):
        assert (vendor / name / "LICENSE").is_file()
        assert not (vendor / name / ".git").exists()
    edited = vendor / "RSI-Jev" / "rsijev" / "arch.py"
    edited.write_text("local changes\n")
    bootstrap.restore_upstreams(source_root)
    assert edited.read_text() == "local changes\n"


def test_archive_checksum_checked_before_extraction(source_root):
    archive = source_root / "upstreams" / "PlayJev.tar.gz"
    archive.unlink()
    archive.write_bytes(b"corrupt archive")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        bootstrap.restore_upstreams(source_root)
    assert not (source_root / "vendor").exists()


def test_different_snapshot_is_kept_intact(source_root):
    directory = source_root / "vendor" / "PlayJev"
    directory.mkdir(parents=True)
    (directory / ".game-rsi-upstream.json").write_text(json.dumps({"commit": "old"}))
    local = directory / "local.txt"
    local.write_text("keep me")
    with pytest.raises(RuntimeError, match="Move this directory aside"):
        bootstrap.restore_upstreams(source_root)
    assert local.read_text() == "keep me"
    assert not (directory / "playjev").exists()


@pytest.mark.parametrize("matches", [True, False])
def test_existing_git_checkout_is_never_fetched_or_overwritten(source_root, monkeypatch, matches):
    lock = json.loads((source_root / "upstreams.lock.json").read_text())
    for name in lock:
        (source_root / "vendor" / name / ".git").mkdir(parents=True)

    def rev_parse(args, *, cwd, text):
        assert args == ["git", "rev-parse", "HEAD"]
        return lock[cwd.name]["commit"] if matches else "different-commit"

    monkeypatch.setattr(bootstrap.subprocess, "check_output", rev_parse)
    if matches:
        bootstrap.restore_upstreams(source_root)
    else:
        with pytest.raises(RuntimeError, match="existing files have been kept"):
            bootstrap.restore_upstreams(source_root)
    assert (source_root / "vendor" / "PlayJev" / ".git").is_dir()
