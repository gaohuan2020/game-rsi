"""Read-only fresh-panel evaluation of the retained native RSI champion."""
import asyncio
import json

from game_rsi.config import ARTIFACTS, RunConfig, champion, upstream_paths
from game_rsi.evaluation import evaluation_seeds, paired_report
from game_rsi.policies import Policy, release_policy
from game_rsi.storage import Journal, atomic_json
from game_rsi.worker import evaluate


async def main():
    upstream_paths()
    cfg = RunConfig(input_mode="state", max_episode_steps=300, eval_episodes=8, seed=29)
    current = champion(cfg)
    if current["version"] == "base":
        raise RuntimeError("No promoted champion to evaluate")
    seeds = evaluation_seeds(cfg.eval_episodes, cfg.seed)
    if set(seeds) & set(current["evaluation"]["seeds"]):
        raise RuntimeError("Fresh panel overlaps the model-selection panel")
    journal = Journal(ARTIFACTS / "qa" / "fresh-panel")
    atomic_json(journal.directory / "experiment.json", {"seeds": seeds, "baseline": cfg.model,
                                                        "candidate": current["checkpoint"], "seed": cfg.seed,
                                                        "max_episode_steps": cfg.max_episode_steps})
    print("baseline on fresh seeds", flush=True)
    policy = Policy(cfg, cfg.model)
    baseline = await evaluate(cfg, policy, seeds, journal, "fresh-baseline")
    release_policy(policy)
    print("champion on fresh seeds", flush=True)
    policy = Policy(cfg, current["checkpoint"])
    scores = await evaluate(cfg, policy, seeds, journal, "fresh-champion")
    release_policy(policy)
    report = paired_report(baseline, scores, seed=cfg.seed)
    report.update(seeds=seeds, checkpoint=current["checkpoint"], input_mode="state",
                  scope="fresh Snake seeds; no training or champion selection on this panel")
    atomic_json(journal.directory / "report.json", report)
    journal.publish(status="completed", phase="completed", progress=1)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
