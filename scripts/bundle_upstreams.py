"""Refresh runtime archives from the exact local commits in upstreams.lock.json."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock_path = ROOT / "upstreams.lock.json"
    lock = json.loads(lock_path.read_text())
    # Check every checkout before changing any archive.
    for name, spec in lock.items():
        directory = ROOT / "vendor" / name
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
        if actual != spec["commit"]:
            raise RuntimeError(f"Expected {spec['commit']}, found {actual}: {directory}")
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=directory, text=True).strip():
            raise RuntimeError(f"Refusing to bundle a modified checkout: {directory}")
    for name, spec in lock.items():
        archive = ROOT / spec["archive"]
        archive.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "archive", "--format=tar.gz", f"--output={archive}",
                        spec["commit"], "--", *spec["paths"]], cwd=ROOT / "vendor" / name, check=True)
        spec["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        print(f"{name}: {archive.stat().st_size:,} bytes")
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")


if __name__ == "__main__":
    main()
