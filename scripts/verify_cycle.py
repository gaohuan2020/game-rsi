"""Run a real GPU acceptance experiment through the dashboard API."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((args.config or root / "configs" / "rsi-snake.json").read_text())
    if not args.config:
        cfg.update(rounds=1, collect_steps=256, train_steps=80, max_episode_steps=100)
    if args.smoke:
        cfg.update(collect_steps=64, train_steps=2, eval_episodes=2, max_episode_steps=10)
    with httpx.Client(base_url="http://127.0.0.1:8092", timeout=30) as client:
        response = client.post("/api/evolve", json=cfg)
        response.raise_for_status()
        job = response.json()
        print("job=" + job["id"], flush=True)
        (root / "artifacts" / "acceptance-job.json").write_text(json.dumps(job, indent=2))
        last = ""
        started = time.monotonic()
        heartbeat = started
        while time.monotonic() - started < 1800:
            detail = client.get("/api/jobs/" + job["id"]).json()
            state = detail["job"]
            message = f"{state.get('status')} / {state.get('phase')}"
            if message != last or time.monotonic() - heartbeat > 30:
                print(message + " " + json.dumps({k: state.get(k) for k in ('collected','train_step','loss','eval_episode') if k in state}), flush=True)
                last, heartbeat = message, time.monotonic()
            if state.get("status") in {"completed", "failed", "stopped", "interrupted"}:
                if state["status"] != "completed":
                    print(detail["logs"][-5000:], flush=True)
                    raise RuntimeError(state.get("error", state["status"]))
                training_rounds = [e for e in detail["events"] if e["kind"] == "trained"]
                evaluations = [e for e in detail["events"] if e["kind"] == "evaluation"]
                trained, evaluation = training_rounds[-1], evaluations[-1]
                report = {"job": job["id"], "training": trained, "evaluation": evaluation,
                          "training_rounds": training_rounds, "evaluations": evaluations,
                          "real_gpu_run": True,
                          "smoke_only": args.smoke or cfg.get("train_steps", 0) < 10 or cfg.get("eval_episodes", 0) < 4}
                (root / "artifacts" / f"acceptance-{job['id']}.json").write_text(json.dumps(report, indent=2))
                (root / "artifacts" / "acceptance-report.json").write_text(json.dumps(report, indent=2))
                print(json.dumps(report, ensure_ascii=False), flush=True)
                return
            time.sleep(3)
        raise TimeoutError("Acceptance worker has not finished; inspect dashboard before retrying")


if __name__ == "__main__":
    main()
