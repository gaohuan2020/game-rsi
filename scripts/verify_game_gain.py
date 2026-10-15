"""Fresh paired game-score verification of saved candidates, with optional promotion."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import time
from pathlib import Path

from game_rsi.config import ARTIFACTS, RunConfig, champion, registry_key, upstream_paths
from game_rsi.evaluation import evaluation_seeds, paired_report
from game_rsi.policies import Policy, choose, noop_rule, probabilities, release_policy
from game_rsi.storage import Journal, atomic_json, fingerprint


def publish_dashboard_history(loss_run, output, games):
    """Expose completed, real training + score-verification traces in the dashboard."""
    published = []
    for game in games:
        training = loss_run / game
        evaluation = output / game
        result = json.loads((evaluation / "evaluation.json").read_text())
        train_state = json.loads((training / "state.json").read_text())
        eval_state = json.loads((evaluation / "state.json").read_text())
        if train_state["status"] != "completed" or eval_state["status"] != "completed":
            raise ValueError("Only completed verification runs can enter dashboard history")
        created = eval_state["updated_at"]
        job_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(created)) + f"-gain-{game}"
        directory = ARTIFACTS / "jobs" / job_id
        cfg = RunConfig.model_validate({**json.loads((training / "config.json").read_text()),
                                       "promotion_rule": result["promotion_rule"],
                                       "eval_episodes": result["pairs"],
                                       "max_episode_steps": result["max_episode_steps"]})
        atomic_json(directory / "config.json", cfg.model_dump())
        atomic_json(directory / "state.json", {**train_state, "status": "completed", "phase": "completed",
                    "progress": 1, "game": game, "backend": cfg.backend, "config": cfg.model_dump(),
                    "last_evaluation": result, "champion": champion(cfg), "completed_at": created})
        events = (training / "events.jsonl").read_text() + (evaluation / "events.jsonl").read_text()
        (directory / "events.jsonl").write_text(events)
        if (training / "frame.json").exists():
            atomic_json(directory / "frame.json", json.loads((training / "frame.json").read_text()))
        atomic_json(directory / "experiment.json", {"training": json.loads((loss_run / "experiment.json").read_text()),
                    "evaluation": json.loads((output / "experiment.json").read_text())})
        # Publish the job metadata last, after all files are ready for readers.
        atomic_json(directory / "job.json", {"id": job_id, "kind": "evolve", "game": game,
                    "backend": cfg.backend, "config": cfg.model_dump(), "created_at": created,
                    "source": "verified training and fresh paired evaluation",
                    "training_artifacts": str(training.resolve()), "evaluation_artifacts": str(evaluation.resolve())})
        published.append(job_id)
    return published


async def evaluate_parallel(cfg, checkpoint, seeds, journal, label, batch_size):
    from playjev.env import VecGame
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache(checkpoint, "config.json") if not Path(checkpoint).is_dir() else None
    resolved = str(Path(cached).parent) if isinstance(cached, str) else checkpoint
    journal.phase("evaluating", evaluation_model=label)
    policy = await asyncio.to_thread(Policy, cfg, resolved)
    try:
        async with VecGame(cfg.game, n=len(seeds)) as env:
            obs = await env.reset(seeds)
            banned = [set() for _ in seeds]
            finished, lengths = set(), [0] * len(seeds)
            # Record whether initial decisions agree with normal single-frame inference.
            decisions = await asyncio.to_thread(policy.model.decide, [o["frame"] for o in obs],
                                                env.actions, batch_size=batch_size)
            singles = [await asyncio.to_thread(policy.decide, o, env.actions) for o in obs]
            parity = all(choose(d.probs, set()) == choose(p, set()) for d, p in zip(decisions, singles))
            journal.event("batch_parity", model=label, initial_actions_match=parity, batch_size=batch_size)
            if not parity:
                batch_size = 1
            for step in range(cfg.max_episode_steps):
                journal.check()
                active = [i for i in range(len(seeds)) if i not in finished]
                if not active:
                    break
                decisions = await asyncio.to_thread(policy.model.decide, [obs[i]["frame"] for i in active],
                                                    env.actions, batch_size=batch_size)
                actions = [choose(probabilities(d.probs, len(env.actions)), banned[i])
                           for i, d in zip(active, decisions)]
                after = await asyncio.gather(*(env.pages[i].step(action) for i, action in zip(active, actions)))
                for i, action, result in zip(active, actions, after):
                    noop_rule(obs[i], result, action, banned[i])
                    obs[i] = result
                    lengths[i] = step + 1
                    if result["done"] or step + 1 == cfg.max_episode_steps:
                        finished.add(i)
                        journal.event("episode", source=label, seed=seeds[i], score=result["score"],
                                      steps=step + 1, truncated=not result["done"])
                journal.publish(eval_step=step + 1, eval_episode=len(finished), eval_target=len(seeds),
                                scores=[o["score"] for o in obs], progress=len(finished) / len(seeds))
                if (step + 1) % 100 == 0:
                    print(f"{cfg.game}/{label}: step {step+1}, complete {len(finished)}/{len(seeds)}", flush=True)
            return {"scores": [float(o["score"]) for o in obs], "lengths": lengths,
                    "initial_batch_parity": parity, "inference_batch_size": batch_size}
    finally:
        release_policy(policy)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", nargs="+", default=["tetris", "mario"], choices=["tetris", "mario"])
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--publish-history", action="store_true")
    args = parser.parse_args()
    upstream_paths()
    args.output.mkdir(parents=True, exist_ok=False)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    seeds = evaluation_seeds(args.episodes, args.seed)
    with (ARTIFACTS / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another training worker owns the GPU lock") from error
        plan = {"registered_at": time.time(), "games": args.games, "seeds": seeds,
                "max_episode_steps": args.max_episode_steps, "promotion_rule": "positive_gain",
                "promote": args.promote, "inference_batch_size": args.batch_size,
                "scope": "Fresh seed panel, fixed before observing scores; one paired panel per game"}
        atomic_json(args.output / "experiment.json", plan)
        results = []
        for game in args.games:
            loss = json.loads((args.loss_run / game / "report.json").read_text())
            cfg = RunConfig.model_validate({**loss["config"], "seed": args.seed,
                    "eval_episodes": args.episodes, "max_episode_steps": args.max_episode_steps,
                    "promotion_rule": "positive_gain"})
            current = champion(cfg)
            checkpoint = loss["checkpoint"]
            journal = Journal(args.output / game)
            atomic_json(journal.directory / "config.json", cfg.model_dump())
            print(f"{game}: evaluating champion", flush=True)
            before = await evaluate_parallel(cfg, current["checkpoint"], seeds, journal, "champion", args.batch_size)
            print(f"{game}: evaluating candidate", flush=True)
            after = await evaluate_parallel(cfg, checkpoint, seeds, journal, "candidate", args.batch_size)
            result = paired_report(before["scores"], after["scores"], seed=args.seed,
                                   promotion_rule=cfg.promotion_rule)
            result.update(game=game, seeds=seeds, max_episode_steps=cfg.max_episode_steps,
                          checkpoint=checkpoint, parent=current["checkpoint"], round=1,
                          sha256=fingerprint(Path(checkpoint)), baseline_details=before, candidate_details=after,
                          loss_validation=loss["validation"], input_mode="pixels")
            result["status"] = "promoted" if result["passed"] and args.promote else "verified" if result["passed"] else "rejected"
            atomic_json(journal.directory / "evaluation.json", result)
            if args.promote and result["passed"]:
                # Same lock and registry contract as the normal evolve worker.
                updated = {"checkpoint": checkpoint, "sha256": result["sha256"],
                           "version": f"{args.output.name}/{game}/r1", "backend": cfg.backend,
                           "game": game, "promoted_at": time.time(), "previous": current,
                           "evaluation": result}
                atomic_json(ARTIFACTS / "champions" / f"{registry_key(cfg)}.json", updated)
                journal.publish(champion=updated)
            journal.event("evaluation", **result)
            journal.publish(status="completed", phase="completed", progress=1, last_evaluation=result)
            results.append(result)
            atomic_json(args.output / "report.json", {"experiment": plan, "results": results})
            print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.publish_history:
            print("Dashboard history: " + json.dumps(publish_dashboard_history(args.loss_run, args.output, args.games)), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
