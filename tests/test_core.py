import json
from pathlib import Path

import pytest

from game_rsi.config import RunConfig, registry_key
from game_rsi.evaluation import evaluation_seeds, paired_report, training_seed
from game_rsi.policies import choose, game_state, probabilities
from game_rsi.storage import Cancelled, Journal, atomic_json
from game_rsi.training import load_samples


def test_pairing_and_gate():
    assert paired_report([1, 2, 3, 4], [3, 4, 5, 6])["passed"]
    assert not paired_report([1, 2, 3, 4], [1, 2, 3, 4])["passed"]
    report = paired_report([10, 10, 10, 10], [30, 30, 30, 1], require_ci=False)
    assert report["checks"]["score_gain"]
    assert not report["checks"]["no_large_regression"]
    with pytest.raises(ValueError):
        paired_report([1, 2], [2])
    with pytest.raises(ValueError):
        paired_report([1, float("nan")], [2, 3])


def test_confidence_gate_rejects_uncertain_improvement():
    report = paired_report([10, 10, 10, 10], [12, 12, 12, 8])
    assert report["delta"] > 0
    assert not report["checks"]["confidence_interval"]


def test_seed_namespaces_and_replay_split(tmp_path):
    seeds = [training_seed(r, e) for r in range(1, 31) for e in range(20)]
    assert not set(seeds) & set(evaluation_seeds(64, 17))
    shard = tmp_path / "snake" / "round-001"
    shard.mkdir(parents=True)
    rows = [{"seed": seed, "frame": "frames/1.jpg"} for seed in seeds[:20]]
    (shard / "records.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    train, dev = load_samples(tmp_path, "snake", 100, 17)
    assert train and dev
    assert all(x["seed"] % 10 != 0 for x in train)
    assert all(x["seed"] % 10 == 0 for x in dev)
    assert not {x["seed"] for x in train} & {x["seed"] for x in dev}


def test_input_never_contains_label_or_score():
    state = game_state({"score": 9000, "teacher_action": 3,
                        "info": {"head": [1, 2], "maxX": 100, "won": True, "ticks": 42}})
    assert json.loads(state) == {"head": [1, 2]}


def test_snake_geometry_is_observation_not_a_label():
    obs = {"info": {"head": {"row": 1, "col": 1}, "food": [1, 2], "grid": "###|#.F|###"}}
    state = json.loads(game_state(obs))
    assert state["move_geometry"]["right"] == {"blocked": False, "food_distance": 0}
    assert state["move_geometry"]["up"]["blocked"]
    assert "teacher_action" not in state


def test_actor_probabilities_and_noop():
    assert probabilities([2, 1], 2) == [2/3, 1/3]
    for values in ([0, 0], [-1, 2], [float("nan"), 1], [0.5]):
        with pytest.raises(ValueError):
            probabilities(values, 2)
    assert choose([0.9, 0.1], {0}) == 1
    assert choose([0.9, 0.1], {0, 1}) == 0


def test_model_contracts_and_champion_isolation():
    assert RunConfig(backend="playjev").model == "OmniJev/PlayJev-0.8B"
    with pytest.raises(ValueError):
        RunConfig(backend="playjev", input_mode="state")
    with pytest.raises(ValueError):
        RunConfig(eval_episodes=1)
    assert registry_key(RunConfig(model="model-a")) != registry_key(RunConfig(model="model-b"))
    assert registry_key(RunConfig(input_mode="state")) != registry_key(RunConfig(input_mode="pixels"))


def test_auto_mode_resolves_same_champion_as_explicit_mode(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"spec": {}}))
    config = RunConfig(model=str(tmp_path), input_mode="auto")
    assert registry_key(config) == registry_key(config.model_copy(update={"input_mode": "state"}))


def test_journal_cancel_keeps_durable_records(tmp_path):
    journal = Journal(tmp_path)
    journal.event("metric", step=1, loss=0.5)
    journal.publish(phase="training")
    assert json.loads((tmp_path / "state.json").read_text())["phase"] == "training"
    (tmp_path / "stop").touch()
    with pytest.raises(Cancelled):
        journal.check()
    assert json.loads((tmp_path / "events.jsonl").read_text())["loss"] == 0.5
