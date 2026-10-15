import asyncio
import json
import sys
import types
from collections import Counter

import pytest

from game_rsi.config import RunConfig
from game_rsi.replay import ReplaySampler, load_replay
from game_rsi.storage import Journal
from game_rsi.worker import collect


def write_shard(root, round_index, episodes):
    shard = root / "snake" / f"round-{round_index:03d}"
    shard.mkdir(parents=True)
    rows, summaries = [], []
    for episode, spec in enumerate(episodes):
        length = spec.get("steps", 4)
        for step in range(length):
            rows.append({"schema_version": 2, "seed": spec["seed"], "episode": episode,
                         "step": step, "frame": f"frames/{episode}-{step}.jpg",
                         "observation_hash": f"{episode}/{step}", "state": '{"position":1}',
                         "actions": ["left", "right"], "teacher_probs": [1., 0.],
                         "teacher_action": 0, "model_probs": [0., 1.] if spec.get("correction") else [1., 0.],
                         "behavior_mode": spec.get("role", "model")})
        summaries.append({"episode": episode, "seed": spec["seed"], "steps": length,
                          "horizon": 8, "complete": spec.get("complete", True),
                          "score_gain": spec["score"]})
    (shard / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (shard / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in summaries))
    return shard, rows


def test_ranks_same_seed_and_keeps_corrections_from_rejected_trajectory(tmp_path):
    write_shard(tmp_path, 1, [
        {"seed": 101000, "score": 100}, {"seed": 101000, "score": 0},
        {"seed": 101001, "score": 1, "correction": True},
        {"seed": 101001, "score": 8}, {"seed": 101001, "score": 5},
        # A different, difficult seed must not compete with the high-scoring seed.
        {"seed": 101002, "score": 0},
    ])
    train, dev, report = load_replay(tmp_path, "snake", 100, 17, keep=2)
    assert [item["score"] for item in report["ranking"] if item["seed"] == 101001] == [8, 5, 1]
    assert {row["episode"] for row in train if row["_elite"]} == {3, 4, 5}
    assert {row["episode"] for row in train if not row["_elite"]} == {2}
    assert all(row["_correction"] for row in train if row["episode"] == 2)
    assert {row["episode"] for row in dev} == {0, 1}
    assert not set(report["train_seeds"]) & set(report["dev_seeds"])
    assert all(row["state"] == '{"position":1}' for row in train), "Outcomes must not enter model observations"
    _, other_dev, _ = load_replay(tmp_path, "snake", 100, 17, keep=1)
    assert other_dev == dev, "Training selection must not influence validation"


def test_filters_bad_labels_duplicates_and_incomplete_demonstrations(tmp_path):
    shard, rows = write_shard(tmp_path, 1, [
        {"seed": 101000, "score": 0}, {"seed": 101001, "score": 3},
        {"seed": 101001, "score": 100, "complete": False, "correction": True},
        {"seed": 101001, "score": 200, "complete": False},
    ])
    rows[5]["observation_hash"] = rows[4]["observation_hash"]
    rows[6]["teacher_probs"] = [.5, .5]
    rows[7]["teacher_probs"] = [-1., 2.]
    (shard / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    train, dev, report = load_replay(tmp_path, "snake", 100, 17)
    assert report["filtered"]["duplicates"] == 1
    assert report["filtered"]["uninformative_labels"] == 1
    assert report["filtered"]["invalid_labels"] == 1
    assert {row["episode"] for row in train if row["_elite"]} == {1}
    assert {row["episode"] for row in train if not row["_elite"]} == {2}
    assert len(dev) == 4


def test_protected_evaluation_seeds_and_mismatched_outcomes_are_rejected(tmp_path):
    shard, _ = write_shard(tmp_path, 1, [{"seed": 90000000, "score": 1}])
    with pytest.raises(ValueError, match="protected evaluation seed"):
        load_replay(tmp_path, "snake", 100, 17)
    path = shard / "records.jsonl"
    path.write_text(path.read_text().replace('90000000', '101001'))
    with pytest.raises(ValueError, match="does not match"):
        load_replay(tmp_path, "snake", 100, 17)


def test_legacy_shards_remain_usable_without_fabricated_rankings(tmp_path):
    shard, rows = write_shard(tmp_path, 1, [
        {"seed": 101000, "score": 0}, {"seed": 101001, "score": 1}])
    for row in rows:
        del row["schema_version"]
    (shard / "episodes.jsonl").unlink()
    (shard / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    train, dev, report = load_replay(tmp_path, "snake", 100, 17)
    assert train and dev and not report["ranking"]


def test_buffer_reserves_recent_data_and_sampler_balances_roles_and_episodes(tmp_path):
    for round_index in range(1, 5):
        write_shard(tmp_path, round_index, [
            {"seed": 100000 * round_index, "score": 0},
            {"seed": 100000 * round_index + 1, "score": 1, "steps": 80, "role": "teacher"},
            {"seed": 100000 * round_index + 2, "score": 1, "steps": 2},
            {"seed": 100000 * round_index + 3, "score": 1, "steps": 40},
        ])
    train, dev, report = load_replay(tmp_path, "snake", 32, 17)
    assert len(train) == 32 and len(dev) <= 128
    assert report["rows_by_round"]["round-004"] == 16
    sampler = ReplaySampler(train, 17)
    draws = sampler.batch(12000)
    assert .47 < sum(r["_round"] == "round-004" for r in draws) / len(draws) < .53
    assert .47 < sum(r["_role"] == "teacher" for r in draws) / len(draws) < .53
    latest_model = Counter(r["episode"] for r in draws if r["_round"] == "round-004" and r["_role"] == "model")
    assert .45 < latest_model[2] / sum(latest_model.values()) < .55
    assert ReplaySampler(train, 17).batch(50) == draws[:50]


def test_sampler_reserves_correction_draws():
    rows = [{"_round": "round-001", "_role": "model", "_trajectory": str(i),
             "_elite": i == 0, "_correction": i == 1} for i in range(2)]
    draws = ReplaySampler(rows, 17).batch(10000)
    assert .23 < sum(r["_correction"] for r in draws) / len(draws) < .27


def test_data_configuration_budget_and_defaults():
    assert RunConfig().collect_steps == 1000
    assert RunConfig().data_strategy == "trajectory"
    with pytest.raises(ValueError, match="trajectory_keep"):
        RunConfig(trajectory_keep=3)
    with pytest.raises(ValueError, match="four steps"):
        RunConfig(collect_steps=32, trajectory_candidates=8)
    assert RunConfig(collect_steps=32, trajectory_candidates=8, data_strategy="uniform")


def test_collection_obeys_1000_step_budget_and_repeats_seed_for_candidates(tmp_path, monkeypatch):
    class Environment:
        resets, calls = [], 0
        actions = [{"name": "left"}, {"name": "right"}]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def observation(self):
            return {"frame": f"frame-{self.step_index}".encode(), "info": {"x": self.step_index},
                    "score": self.step_index, "reward": 1, "done": self.step_index == 3}

        async def reset(self, seeds):
            self.resets.extend(seeds)
            self.step_index = 0
            return [self.observation()]

        async def step(self, actions):
            self.step_index += 1
            Environment.calls += 1
            return [self.observation()]

    class Teacher:
        def reset(self):
            pass

        def act(self, obs):
            return [0., 1.]

        def giveup(self):
            return False

    class Policy:
        mode, checkpoint, latency_ms = "state", "test", 0

        def decide(self, obs, options):
            return [1., 0.]

    monkeypatch.setitem(sys.modules, "playjev.env", types.SimpleNamespace(VecGame=lambda *a, **kw: Environment()))
    monkeypatch.setitem(sys.modules, "playjev.teachers", types.SimpleNamespace(make_teacher=lambda *a: Teacher()))
    cfg = RunConfig(epsilon=1, teacher_mix=1)
    asyncio.run(collect(cfg, Policy(), tmp_path / "data", 1, Journal(tmp_path / "job")))
    manifest = tmp_path / "data/snake/round-001/records.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows) == Environment.calls == 1000
    assert Environment.resets[:7] == [101000] + [101001] * 3 + [101002] * 3
    assert all(row["action_source"] == "random" and row["behavior_mode"] == "teacher" for row in rows)
    train, dev, report = load_replay(tmp_path / "data", "snake", cfg.replay_limit, cfg.seed)
    assert train and dev and report["strategy"] == "trajectory"
    outcomes = [json.loads(line) for line in manifest.with_name("episodes.jsonl").read_text().splitlines()]
    assert outcomes[-1]["end_reason"] == "budget" and not outcomes[-1]["complete"]
