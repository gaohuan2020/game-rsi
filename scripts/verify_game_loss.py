"""Measure real pixel-model learning on fresh game data without promoting a champion.

Validation seeds stay separate from training. A fixed training probe distinguishes
fitting from generalization; changing minibatch loss is not used as a success test.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import fcntl
import json
import math
import time
from pathlib import Path

from game_rsi.config import ARTIFACTS, RunConfig, upstream_paths
from game_rsi.policies import Policy, release_policy
from game_rsi.storage import Journal, atomic_json
from game_rsi.training import train_candidate
from game_rsi.worker import collect


def compare_metrics(initial, final):
    before, after = initial["val_loss"], final["val_loss"]
    return {"initial_loss": before, "final_loss": after, "loss_delta": after - before,
            "relative_decrease": (before - after) / max(abs(before), 1e-12),
            "initial_kl": initial["val_kl"], "final_kl": final["val_kl"],
            "initial_agreement": initial["agreement"], "final_agreement": final["agreement"],
            "decreased": after < before - 1e-6}


async def run_game(args, game, root, parent):
    cfg = RunConfig(game=game, backend="playjev", input_mode="pixels", model=args.model,
                    rounds=1, collect_steps=args.collect_steps, train_steps=args.steps,
                    batch_size=args.batch_size, learning_rate=args.learning_rate,
                    teacher_mix=args.teacher_mix, seed=args.seed, data_strategy="trajectory")
    directory = root / game
    journal = Journal(directory)
    atomic_json(directory / "config.json", cfg.model_dump())
    print(f"{game}: collecting {cfg.collect_steps} observations", flush=True)
    policy = Policy(cfg, parent)
    try:
        await collect(cfg, policy, directory / "data", 1, journal)
    finally:
        release_policy(policy)
    journal.phase("training", round=1)
    print(f"{game}: training {cfg.train_steps} steps", flush=True)
    report = await asyncio.to_thread(train_candidate, cfg, parent, directory / "data",
                                    directory / "checkpoint", journal, 1, probe_training=True)
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    metrics = [event for event in events if event["kind"] == "metric"]
    validation = [event for event in metrics if "val_loss" in event]
    train_metrics = [event for event in metrics if "loss" in event]
    if not all(math.isfinite(e["loss"]) for e in train_metrics):
        raise FloatingPointError("Training produced non-finite losses")
    result = {"game": game, "config": cfg.model_dump(), "checkpoint": report["checkpoint"],
              "validation": compare_metrics(report["initial"], report["final"]),
              "training_probe": compare_metrics(report["training_probe"]["initial"],
                                                report["training_probe"]["final"]),
              "train_probe_samples": report["training_probe"]["samples"],
              "train_samples": report["train_samples"], "dev_samples": report["dev_samples"],
              "validation_curve": [{key: event[key] for key in ("step", "val_loss", "val_kl", "agreement")}
                                   for event in validation],
              "reload_parity": report["reload_parity"], "seconds": report["seconds"],
              "scope": "One seed and recipe; loss verification, not a game-score or statistical significance claim"}
    with (directory / "metrics.csv").open("w", newline="") as stream:
        columns = ["step", "loss", "kl", "target_entropy", "val_loss", "val_kl", "agreement"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    atomic_json(directory / "report.json", result)
    journal.publish(status="completed", phase="completed", progress=1)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", choices=["tetris", "mario"], default=["tetris", "mario"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lock-root", type=Path, default=ARTIFACTS)
    parser.add_argument("--model", default="OmniJev/PlayJev-0.8B")
    parser.add_argument("--collect-steps", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--teacher-mix", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    upstream_paths()
    import torch
    from huggingface_hub import try_to_load_from_cache
    if not torch.cuda.is_available():
        raise RuntimeError("This verification requires a real CUDA training device")
    cached = try_to_load_from_cache(args.model, "config.json") if not Path(args.model).is_dir() else None
    parent = str(Path(cached).parent) if isinstance(cached, str) else args.model
    root = (args.output or ARTIFACTS / "qa" / f"game-loss-{time.strftime('%Y%m%d-%H%M%S')}").resolve()
    root.mkdir(parents=True, exist_ok=False)
    args.lock_root.mkdir(parents=True, exist_ok=True)
    with (args.lock_root / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another training worker owns the GPU lock") from error
        atomic_json(root / "experiment.json", {"registered_at": time.time(), "games": args.games,
                    "model": args.model, "resolved_parent": parent,
                    "collect_steps": args.collect_steps, "steps": args.steps,
                    "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                    "teacher_mix": args.teacher_mix, "seed": args.seed,
                    "criterion": "Final loss on the same seed-separated validation set is lower than initial loss",
                    "champion_updates": False, "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(0)})
        results = []
        for game in args.games:
            results.append(await run_game(args, game, root, parent))
            atomic_json(root / "report.json", {"results": results,
                        "all_validation_losses_decreased": all(r["validation"]["decreased"] for r in results)})
        print(f"Reports: {root}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
