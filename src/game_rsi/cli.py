from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Real game play + native Jev self-improvement")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Launch the local game/training dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8092, type=int)
    for name in ("play", "evolve"):
        task = sub.add_parser(name, help="Run a task without the dashboard")
        task.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        from .server import create_app
        print(f"Game RSI: http://{args.host}:{args.port}", flush=True)
        uvicorn.run(create_app(), host=args.host, port=args.port, access_log=False,
                    timeout_graceful_shutdown=3)
    else:
        from .config import ARTIFACTS, RunConfig
        from .storage import atomic_json
        from .worker import main as worker_main
        import sys
        config = RunConfig.model_validate_json(args.config.read_text())
        directory = ARTIFACTS / "cli" / f"{int(time.time())}-{os.getpid()}"
        atomic_json(directory / "config.json", config.model_dump())
        sys.argv = [sys.argv[0], args.command, str(directory)]
        worker_main()


if __name__ == "__main__":
    main()
