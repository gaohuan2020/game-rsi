import asyncio
import json

import pytest

from game_rsi.config import RunConfig, registry_key
from game_rsi.storage import Journal


@pytest.mark.parametrize("rule,promoted", [("positive_gain", True), ("strict", False)])
def test_evolution_uses_registered_rule_when_updating_champion(tmp_path, monkeypatch, rule, promoted):
    from game_rsi import config, training, worker

    class Policy:
        def __init__(self, cfg, parent):
            self.checkpoint = parent

    async def collect(*args):
        pass

    async def evaluate(cfg, policy, seeds, journal, label):
        return [10., 10., 10., 10.] if label == "champion" else [30., 30., 30., 1.]

    def train(cfg, parent, data, out, journal, round_index):
        return {"checkpoint": str(out), "sha256": "test-weights"}

    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(worker, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(worker, "Policy", Policy)
    monkeypatch.setattr(worker, "release_policy", lambda policy: None)
    monkeypatch.setattr(worker, "collect", collect)
    monkeypatch.setattr(worker, "evaluate", evaluate)
    monkeypatch.setattr(training, "train_candidate", train)
    cfg = RunConfig(model="test-parent", input_mode="state", rounds=1,
                    eval_episodes=4, promotion_rule=rule)
    journal = Journal(tmp_path / "run")
    asyncio.run(worker.evolve(cfg, journal))
    report = journal.state["last_evaluation"]
    assert report["promotion_rule"] == rule
    assert report["passed"] is promoted
    assert report["delta"] > 0
    assert not report["checks"]["no_large_regression"]
    pointer = tmp_path / "champions" / f"{registry_key(cfg)}.json"
    assert pointer.exists() is promoted
    if promoted:
        assert json.loads(pointer.read_text())["evaluation"] == report
    assert json.loads((tmp_path / "run/rounds/001/evaluation.json").read_text()) == report
