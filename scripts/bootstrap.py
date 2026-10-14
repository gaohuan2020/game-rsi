"""Install isolated runtime and pinned sources; optionally reuse an existing CUDA stack."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=None):
    subprocess.run([str(x) for x in args], cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Install PyTorch and model training extras")
    parser.add_argument("--reuse-python", type=Path, help="Reuse installed GPU packages read-only, without changing that environment")
    parser.add_argument("--browser", action="store_true", help="Install Chromium for the PlayJev game harness")
    args = parser.parse_args()
    for name, spec in json.loads((ROOT / "upstreams.lock.json").read_text()).items():
        directory = ROOT / "vendor" / name
        if not directory.exists():
            directory.parent.mkdir(parents=True, exist_ok=True)
            run("git", "clone", "--depth", "1", spec["url"], directory)
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
        if actual != spec["commit"]:
            if subprocess.check_output(["git", "status", "--porcelain"], cwd=directory, text=True).strip():
                raise RuntimeError(f"Refusing to change modified source: {directory}")
            run("git", "fetch", "--depth", "1", "origin", spec["commit"], cwd=directory)
            run("git", "checkout", "--detach", spec["commit"], cwd=directory)
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
