"""Install isolated runtime and pinned sources; optionally reuse an existing CUDA stack."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=None):
    subprocess.run([str(x) for x in args], cwd=cwd, check=True)


def restore_upstreams(root=ROOT):
    """Restore bundled runtime sources without contacting upstream Git servers."""
    for name, spec in json.loads((root / "upstreams.lock.json").read_text()).items():
        archive = root / spec["archive"]
        if hashlib.sha256(archive.read_bytes()).hexdigest() != spec["sha256"]:
            raise RuntimeError(f"Source archive checksum mismatch: {archive}")
        directory = root / "vendor" / name
        identity = {"commit": spec["commit"], "sha256": spec["sha256"]}
        marker = directory / ".game-rsi-upstream.json"
        if directory.exists():
            if (directory / ".git").exists():
                actual = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
                if actual == spec["commit"]:
                    continue
            elif marker.is_file() and json.loads(marker.read_text()) == identity:
                continue
            raise RuntimeError(
                f"Source version differs from upstreams.lock.json: {directory}. "
                "Move this directory aside and rerun bootstrap to restore the bundled version; "
                "existing files have been kept.")
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=directory.parent) as temporary:
            staged = Path(temporary) / "source"
            staged.mkdir()
            with tarfile.open(archive, "r:gz") as source:
                source.extractall(staged, filter="data")
            (staged / marker.name).write_text(json.dumps(identity, indent=2) + "\n")
            staged.rename(directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Install PyTorch and model training extras")
    parser.add_argument("--reuse-python", type=Path, help="Reuse installed GPU packages read-only, without changing that environment")
    parser.add_argument("--browser", action="store_true", help="Install Chromium for the PlayJev game harness")
    parser.add_argument("--sources-only", action="store_true", help="Restore bundled sources without installing Python packages or Chromium")
    args = parser.parse_args()
    restore_upstreams()
    if args.sources_only:
        print("Bundled upstream sources are ready in vendor/.")
        return
    env = ROOT / ".venv"
    if not env.exists():
        run("uv", "venv", "--python", "3.12", env)
    python = env / "bin" / "python"
    if args.reuse_python:
        packages = subprocess.check_output([str(args.reuse_python.absolute()), "-c",
                                            "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip()
        target = subprocess.check_output([str(python), "-c",
                                          "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip()
        (Path(target) / "cuda-reuse.pth").write_text(packages + "\n")
        # Keep torch untouched; install only missing browser/test dependencies locally.
        run("uv", "pip", "install", "--python", python, "--no-deps", "-e", ROOT)
        run("uv", "pip", "install", "--python", python, "playwright>=1.56,<2", "pytest>=8", "psutil>=6", "einops")
        run("uv", "pip", "install", "--python", python, "--no-deps", "accelerate>=1.15")
    else:
        extra = "[test,train]" if args.train else "[test]"
        run("uv", "pip", "install", "--python", python, "-e", str(ROOT) + extra)
    if args.browser:
        run(python, "-m", "playwright", "install", "chromium")
    print("Ready: .venv/bin/game-rsi serve --port 8092")


if __name__ == "__main__":
    main()
