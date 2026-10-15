from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import random
import time
import traceback
from pathlib import Path

from .config import ARTIFACTS, STATE_CONTRACT, RunConfig, champion, registry_key, upstream_paths
from .evaluation import evaluation_seeds, paired_report, training_seed
from .policies import Policy, choose, game_state, noop_rule, release_policy
from .storage import Cancelled, Journal, atomic_json


async def collect(cfg, policy, data_root, round_index, journal):
    from playjev.env import VecGame
    from playjev.teachers import make_teacher
    out = data_root / cfg.game / f"round-{round_index:03d}"
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed + round_index)
    journal.phase("collecting", round=round_index, input_mode=policy.mode)
    candidates = cfg.trajectory_candidates if cfg.data_strategy == "trajectory" else 1
    # Preserve the raw step budget and leave room for both train and dev seed groups.
    episode_cap = min(cfg.max_episode_steps, max(4, cfg.collect_steps // max(8, 2 * candidates)))
    completed = 0
    async with VecGame(cfg.game, n=1) as env:
        with (out / "records.jsonl").open("w") as stream, (out / "episodes.jsonl").open("w") as outcomes:
            index, episode, seed_index, candidate = 0, 0, 0, 0
            while index < cfg.collect_steps:
                journal.check()
                seed = training_seed(round_index, seed_index)
                # Validation is never ranked: one rollout avoids spending a candidate
                # tournament's budget on observations that cannot train the model.
                seed_candidates = 1 if seed % 10 == 0 else candidates
                obs = (await env.reset([seed]))[0]
                initial_score = float(obs["score"])
                teacher = make_teacher(cfg.game, env.actions)
                teacher.reset()
                banned = set()
                guided = rng.random() < cfg.teacher_mix
                behavior = "teacher" if guided else "model"
                step, reason = 0, None
                while reason is None:
                    journal.check()
                    p = await asyncio.to_thread(policy.decide, obs, env.actions)
                    target = teacher.act(obs)
                    label_valid = not teacher.giveup()
                    teacher_action = max(range(len(target)), key=target.__getitem__)
                    action = teacher_action if guided else choose(p, banned)
                    source = behavior
                    if rng.random() < cfg.epsilon:
                        action = rng.randrange(len(p))
                        source = "random"
                    filename = f"frames/{index:07d}.jpg"
                    (out / filename).write_bytes(obs["frame"])
                    after = (await env.step([action]))[0]
                    stream.write(json.dumps({"schema_version": 2, "game": cfg.game, "shard": out.name,
                                             "seed": seed, "episode": episode, "candidate": candidate,
                                             "step": step, "frame": filename, "prev_frame": None,
                                             "observation_hash": hashlib.sha256(obs["frame"]).hexdigest(),
                                             "actions": [o["name"] for o in env.actions],
                                             "options": env.actions, "state": game_state(obs),
                                             "teacher_probs": target, "teacher_action": teacher_action,
                                             "label_valid": label_valid, "model_probs": p,
                                             "taken_action": action, "actor": cfg.backend,
                                             "behavior_mode": behavior, "action_source": source,
                                             "score": obs["score"], "reward": after["reward"],
                                             "done": after["done"], "t": obs.get("t", 0)}) + "\n")
                    journal.frame(after, env.actions, p, action, episode=episode, seed=seed,
                                  actor="teacher-guided" if guided else cfg.backend, model=policy.checkpoint,
                                  input_mode=policy.mode, latency_ms=policy.latency_ms, source="collection")
                    index += 1
                    step += 1
                    journal.publish(progress=index / cfg.collect_steps, collected=index,
                                    collect_target=cfg.collect_steps)
                    noop_rule(obs, after, action, banned)
                    obs = after
                    if obs["done"]:
                        reason = "environment_cap" if obs.get("truncated") else "terminal"
                    elif not label_valid or teacher.giveup():
                        reason = "teacher_giveup"
                    elif step >= episode_cap:
                        reason = "episode_cap"
                    elif index >= cfg.collect_steps:
                        reason = "budget"
                summary = {"episode": episode, "candidate": candidate, "seed": seed,
                           "steps": step, "horizon": episode_cap, "behavior_mode": behavior,
                           "initial_score": initial_score, "score": float(obs["score"]),
                           "score_gain": float(obs["score"]) - initial_score,
                           "done": bool(obs["done"]), "end_reason": reason,
                           "complete": reason not in {"budget", "teacher_giveup"}}
                outcomes.write(json.dumps(summary, allow_nan=False) + "\n")
                journal.event("episode", source="collection", **summary)
                completed += 1
                episode += 1
                candidate += 1
                if candidate >= seed_candidates:
                    candidate = 0
                    seed_index += 1
    journal.event("collected", samples=cfg.collect_steps, shard=str(out), episodes=completed,
                  data_strategy=cfg.data_strategy, trajectory_candidates=candidates)


async def evaluate(cfg, policy, seeds, journal, label):
    from playjev.env import VecGame
    journal.phase("evaluating", evaluation_model=label)
    scores = []
    async with VecGame(cfg.game, n=1) as env:
        for episode, seed in enumerate(seeds):
            journal.check()
            obs = (await env.reset([seed]))[0]
            banned = set()
            for step in range(cfg.max_episode_steps):
                journal.check()
                p = await asyncio.to_thread(policy.decide, obs, env.actions)
                action = choose(p, banned)
                after = (await env.step([action]))[0]
                if step % 4 == 0 or after["done"]:
                    journal.frame(after, env.actions, p, action, seed=seed, episode=episode,
                                  actor=cfg.backend, model=policy.checkpoint, input_mode=policy.mode,
                                  latency_ms=policy.latency_ms, source=f"evaluation/{label}")
                noop_rule(obs, after, action, banned)
                obs = after
                if obs["done"]:
                    break
            scores.append(float(obs["score"]))
            journal.publish(progress=(episode + 1) / len(seeds), eval_episode=episode + 1,
                            eval_target=len(seeds))
            journal.event("episode", source=label, seed=seed, score=obs["score"], steps=step + 1,
                          truncated=not obs["done"])
    return scores


async def evolve(cfg, journal):
    from .training import train_candidate
    data_root = journal.directory / "data"
    seeds = evaluation_seeds(cfg.eval_episodes, cfg.seed)
    for round_index in range(1, cfg.rounds + 1):
        journal.check()
        current = champion(cfg)
        parent = current["checkpoint"]
        # Register the recipe and gate before collecting or spending training GPU time.
        experiment = {"round": round_index, "parent": parent, "config": cfg.model_dump(),
                      "state_contract": STATE_CONTRACT,
                      "hypothesis": "Outcome-selected trajectories and balanced corrective replay improve held-out game score"
                      if cfg.data_strategy == "trajectory" else
                      "Teacher correction labels on uniformly replayed trajectories improve held-out game score",
                      "prediction": "mean paired game-score gain > 0" if cfg.promotion_rule == "positive_gain" else
                      f"paired relative gain > {cfg.min_gain}; CI lower bound > 0 when required",
                      "eval_seeds": seeds, "registered_at": time.time()}
        round_dir = journal.directory / "rounds" / f"{round_index:03d}"
        atomic_json(round_dir / "experiment.json", experiment)
        journal.event("experiment", **experiment)
        journal.phase("loading", round=round_index, champion=current)
        policy = await asyncio.to_thread(Policy, cfg, parent)
        await collect(cfg, policy, data_root, round_index, journal)
        baseline = await evaluate(cfg, policy, seeds, journal, "champion")
        release_policy(policy)
        journal.phase("training", round=round_index)
        trained = await asyncio.to_thread(train_candidate, cfg, parent, data_root,
                                         round_dir / "checkpoint", journal, round_index)
        journal.check()
        journal.phase("loading_candidate", round=round_index)
        policy = await asyncio.to_thread(Policy, cfg, trained["checkpoint"])
        scores = await evaluate(cfg, policy, seeds, journal, "candidate")
        release_policy(policy)
        report = paired_report(baseline, scores, min_gain=cfg.min_gain,
                               require_ci=cfg.require_ci, seed=cfg.seed, promotion_rule=cfg.promotion_rule)
        report.update(round=round_index, seeds=seeds, parent=parent,
                      checkpoint=trained["checkpoint"], sha256=trained["sha256"],
                      input_mode=journal.state.get("input_mode"), max_episode_steps=cfg.max_episode_steps,
                      status="promoted" if report["passed"] else "rejected")
        atomic_json(round_dir / "evaluation.json", report)
        journal.check()
        if report["passed"]:
            updated = {"checkpoint": trained["checkpoint"], "sha256": trained["sha256"],
                       "version": f"{journal.directory.name}/r{round_index}", "backend": cfg.backend,
                       "game": cfg.game, "promoted_at": time.time(), "previous": current,
                       "evaluation": report}
            # Only the evolve worker writes this registry, and replacement is atomic.
            atomic_json(ARTIFACTS / "champions" / f"{registry_key(cfg)}.json", updated)
            journal.publish(champion=updated)
        journal.event("evaluation", **report)
        journal.publish(last_evaluation=report)
        # A failed recipe gets a smaller next learning rate; the registered record keeps each change.
        if not report["passed"]:
            cfg = cfg.model_copy(update={"learning_rate": max(1e-7, cfg.learning_rate / 2)})
    journal.publish(status="completed", phase="completed", progress=1, completed_at=time.time())


async def play(cfg, journal):
    from playjev.teachers import make_teacher
    from .live import LiveBrowser, PolicyCache, SwitchGame, check_switch, show_frames
    cache = PolicyCache()
    token = None
    try:
        async with LiveBrowser() as browser:
            while True:
                try:
                    rng = random.Random(cfg.seed)
                    episode = 0
                    while True:
                        check_switch(journal, token)
                        current = champion(cfg)
                        policy = None
                        journal.phase("loading", game=cfg.game, backend=cfg.backend, config=cfg.model_dump(),
                                      control_token=token, actor=cfg.actor, champion=current)
                        if cfg.actor == "model":
                            policy = await cache.get(cfg, current["checkpoint"])
                        check_switch(journal, token)
                        page = await browser.game(cfg.game)
                        seed = cfg.seed + episode
                        obs = await browser.reset(page, seed, cfg.display_fps)
                        options = page.actions
                        teacher = make_teacher(cfg.game, options)
                        teacher.reset()
                        banned = set()
                        mode = policy.mode if policy else "teacher/state"
                        journal.phase("playing", game=cfg.game, input_mode=mode)
                        journal.frame(obs, options, game=cfg.game, seed=seed, episode=episode,
                                      actor=cfg.actor, model=current["checkpoint"] if policy else None,
                                      input_mode=mode, source="live", latency_ms=0)

                        async def decide(observation):
                            if cfg.actor == "teacher":
                                return teacher.act(observation)
                            if cfg.actor == "random":
                                p = [0.0] * len(options)
                                p[rng.randrange(len(p))] = 1.0
                                return p
                            return await asyncio.to_thread(policy.decide, observation, options)

                        p = await decide(obs)
                        for step in range(cfg.max_episode_steps):
                            check_switch(journal, token)
                            action = choose(p, banned)
                            latency = policy.latency_ms if policy else 0
                            after, frames, before = await browser.step(page, action)
                            noop_rule(obs, after, action, banned)
                            next_decision = asyncio.create_task(decide(after)) if not after["done"] else None
                            try:
                                await show_frames(journal, after, frames, before, options, p, action, cfg, token,
                                                  seed=seed, episode=episode, actor=cfg.actor,
                                                  model=current["checkpoint"] if policy else None,
                                                  input_mode=mode, latency_ms=latency)
                            finally:
                                # Finish the inference before switching/releasing its model.
                                if next_decision:
                                    await asyncio.gather(next_decision, return_exceptions=True)
                            check_switch(journal, token)
                            obs = after
                            if obs["done"]:
                                break
                            p = await next_decision
                            if cfg.delay_ms:
                                await asyncio.sleep(cfg.delay_ms / 1000)
                        journal.event("episode", source="live", game=cfg.game, score=obs["score"], steps=step + 1, seed=seed)
                        episode += 1
                        journal.publish(episodes=episode)
                        await asyncio.sleep(0.1)
                except SwitchGame as change:
                    cfg, token = change.config, change.token
                    journal.event("switch", game=cfg.game, backend=cfg.backend, token=token)
                    journal.phase("switching", game=cfg.game, backend=cfg.backend, config=cfg.model_dump(),
                                  control_token=token)
    finally:
        cache.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["play", "evolve"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    upstream_paths()
    cfg = RunConfig.model_validate_json((args.directory / "config.json").read_text())
    journal = Journal(args.directory)
    lockfile = None
    try:
        if args.kind == "evolve":
            import fcntl
            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            lockfile = (ARTIFACTS / "training.lock").open("a")
            try:
                fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another training worker owns the GPU training lock")
        asyncio.run(play(cfg, journal) if args.kind == "play" else evolve(cfg, journal))
    except Cancelled as error:
        journal.publish(status="stopped", phase="stopped", message=str(error))
        journal.event("stopped", message=str(error))
    except BaseException as error:
        traceback.print_exc()
        journal.publish(status="failed", phase="failed", error=str(error))
        journal.event("error", message=str(error))
        raise
    finally:
        if lockfile:
            lockfile.close()


if __name__ == "__main__":
    main()
