"""Outcome-selected trajectories and balanced replay, inspired by MASS Algorithm 3.

Games supply verifiable scores, so no language-model judge or Bradley–Terry fit is
needed. Outcomes select data; only observations and teacher actions reach the model.
"""
from __future__ import annotations

import math
import random
import json
from collections import Counter, defaultdict
from pathlib import Path

from .policies import probabilities


def _buckets(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["_round"], row["_role"], row["_trajectory"], row["_correction"])].append(row)
    return groups


def _cap(rows, limit, rng):
    """Keep a bounded, diverse buffer without letting a long episode fill it."""
    if len(rows) <= limit:
        return rows
    groups = list(_buckets(rows).values())
    for group in groups:
        rng.shuffle(group)
    rng.shuffle(groups)
    kept = []
    while groups and len(kept) < limit:
        remaining = []
        for group in groups:
            kept.append(group.pop())
            if group:
                remaining.append(group)
            if len(kept) == limit:
                break
        groups = remaining
    return kept


def _bounded_train(rows, limit, rng):
    if len(rows) <= limit:
        return rows
    latest = max(row["_round"] for row in rows)
    fresh = [r for r in rows if r["_round"] == latest]
    history = [r for r in rows if r["_round"] != latest]
    fresh_limit = min(len(fresh), max((limit + 1) // 2, limit - len(history)))
    return _cap(fresh, fresh_limit, rng) + _cap(history, limit - fresh_limit, rng)


def load_replay(data_root: Path, game: str, limit: int, seed: int, *, keep: int = 2):
    """Rank only comparable, completed candidates; never rank validation episodes.

    Legacy shards have no outcome manifest and remain usable without invented scores.
    A failed/incomplete candidate can supply corrective labels but never elite data.
    """
    train, dev, ranking = [], [], []
    counts = Counter()
    for manifest in sorted((data_root / game).glob("*/records.jsonl")):
        summaries = {}
        summary_path = manifest.with_name("episodes.jsonl")
        if summary_path.exists():
            for line in summary_path.read_text().splitlines():
                summary = json.loads(line)
                summaries[summary["episode"]] = summary
        episodes = defaultdict(list)
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            if not isinstance(row.get("seed"), int) or not 0 <= row["seed"] < 90000000:
                raise ValueError("Replay contains a missing/invalid seed or a protected evaluation seed")
            episodes[row.get("episode", row["seed"])].append(row)
        candidates = defaultdict(list)
        prepared = []
        for episode, rows in episodes.items():
            counts["raw_rows"] += len(rows)
            if len({row["seed"] for row in rows}) != 1:
                raise ValueError("A trajectory cannot mix training and validation seeds")
            summary = summaries.get(episode)
            if summary and (summary["seed"] != rows[0]["seed"] or summary["steps"] != len(rows)):
                raise ValueError("Trajectory outcome does not match its recorded observations")
            current = rows[0].get("schema_version", 1) >= 2
            if current and not summary:
                counts["unfinished_rows"] += len(rows)
                continue
            complete = bool(summary and summary.get("complete"))
            score = summary.get("score_gain") if complete else None
            if complete and (not isinstance(score, (int, float)) or not math.isfinite(score)):
                raise ValueError("Trajectory outcome must have a finite score")
            trajectory = f"{manifest.parent.name}/{episode}"
            valid, seen = [], set()
            for row in rows:
                try:
                    gold = probabilities(row["teacher_probs"], len(row["actions"]))
                    target = row["teacher_action"]
                    if not isinstance(target, int) or not 0 <= target < len(gold):
                        raise ValueError("Invalid teacher action")
                    if not math.isclose(gold[target], max(gold), abs_tol=1e-8):
                        raise ValueError("Teacher action disagrees with teacher distribution")
                except (KeyError, TypeError, ValueError):
                    counts["invalid_labels"] += 1
                    continue
                if row.get("label_valid") is False or max(gold) - min(gold) < 1e-8:
                    counts["uninformative_labels"] += 1
                    continue
                observation = row.get("observation_hash") or row.get("state")
                key = (observation, row.get("state"), tuple(gold))
                if observation is not None and key in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(key)
                try:
                    model = probabilities(row["model_probs"], len(gold))
                    predicted = max(range(len(model)), key=model.__getitem__)
                    # Any equally good teacher action is acceptable, not a model mistake.
                    correction = gold[predicted] < max(gold) - 1e-8
                except (KeyError, TypeError, ValueError):
                    correction = False
                row.update(frame_path=str(manifest.parent / row["frame"]), teacher_probs=gold,
                           _round=manifest.parent.name, _trajectory=trajectory,
                           _role=row.get("behavior_mode", row.get("action_source", "model")),
                           _correction=correction, _elite=not current)
                valid.append(row)
            item = {"rows": valid, "seed": rows[0]["seed"], "episode": episode,
                    "trajectory": trajectory, "score": score, "complete": complete,
                    "legacy": not current, "selected": not current}
            prepared.append(item)
            if current and complete and valid and item["seed"] % 10 != 0:
                candidates[(item["seed"], summary["horizon"])].append(item)
        # Shuffle ties reproducibly so equal scores do not always favor candidate zero.
        rank_rng = random.Random(f"{seed}:{manifest.parent.name}")
        for group in candidates.values():
            rank_rng.shuffle(group)
            group.sort(key=lambda item: item["score"], reverse=True)
            for rank, item in enumerate(group, 1):
                item["selected"] = rank <= keep
                ranking.append({k: item[k] for k in ("trajectory", "seed", "score", "selected")})
                ranking[-1]["rank"] = rank
        for item in prepared:
            if item["seed"] % 10 == 0:
                # Validation keeps all valid completed candidates, irrespective of outcome.
                if item["complete"] or item["legacy"]:
                    dev.extend(item["rows"])
                continue
            for row in item["rows"]:
                row["_elite"] = item["selected"]
                if row["_elite"] or row["_correction"]:
                    train.append(row)
                else:
                    counts["rejected_rows"] += 1
    eligible = len(train)
    train = _bounded_train(train, limit, random.Random(seed))
    # Independent RNG: changing train selection must not change the validation sample.
    dev = _cap(dev, 128, random.Random(seed + 104729))
    if not train or not dev:
        raise ValueError("Collection must include informative, disjoint train and validation episodes")
    report = {"strategy": "trajectory", "version": 1,
              "reference": "https://arxiv.org/abs/2610.12176", "trajectory_keep": keep,
              "eligible_train_rows": eligible, "train_samples": len(train), "dev_samples": len(dev),
              "correction_samples": sum(row["_correction"] for row in train),
              "elite_samples": sum(row["_elite"] for row in train),
              "train_seeds": sorted({row["seed"] for row in train}),
              "dev_seeds": sorted({row["seed"] for row in dev}),
              "rows_by_round": dict(Counter(row["_round"] for row in train)),
              "rows_by_role": dict(Counter(row["_role"] for row in train)),
              "filtered": dict(counts), "ranking": ranking,
              "fresh_fraction": .5, "correction_fraction": .25,
              "validation_selection": "all completed candidates on independent seed namespace"}
    return train, dev, report


class ReplaySampler:
    """Choose fresh/history, then correction/elite, then role and episode uniformly."""

    def __init__(self, rows, seed):
        self.rng = random.Random(seed)
        self.groups = {}
        self.draws = Counter()
        for row in rows:
            pools = (["elite"] if row["_elite"] else []) + (["correction"] if row["_correction"] else [])
            for pool in pools:
                roles = self.groups.setdefault(row["_round"], {}).setdefault(pool, {})
                roles.setdefault(row["_role"], {}).setdefault(row["_trajectory"], []).append(row)
        self.rounds = sorted(self.groups)
        if not self.rounds:
            raise ValueError("Cannot sample an empty replay buffer")

    def batch(self, size):
        result = []
        for _ in range(size):
            round_id = self.rounds[-1]
            if len(self.rounds) > 1 and self.rng.random() >= .5:
                round_id = self.rng.choice(self.rounds[:-1])
            pools = self.groups[round_id]
            pool = "correction" if self.rng.random() < .25 else "elite"
            if pool not in pools:
                pool = next(iter(pools))
            roles = pools[pool]
            role = self.rng.choice(list(roles))
            episode = self.rng.choice(list(roles[role]))
            row = self.rng.choice(roles[role][episode])
            result.append(row)
            self.draws[f"{round_id}/{pool}/{role}"] += 1
        return result
