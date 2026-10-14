import json

from fastapi.testclient import TestClient

from game_rsi.config import RunConfig
from game_rsi.storage import atomic_json


def seed_run(root, task_id, game, created_at, evaluations, kind="evolve"):
    directory = root / "jobs" / task_id
    atomic_json(directory / "job.json", {"id": task_id, "kind": kind, "game": game,
                                        "created_at": created_at,
                                        "config": RunConfig(game=game).model_dump()})
    atomic_json(directory / "state.json", {"status": "completed", "phase": "completed", "round": 1})
    (directory / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evaluations))
    return directory


def evaluation(round_index, score):
    return {"kind": "evaluation", "time": round_index, "round": round_index,
            "baseline": score, "candidate": score + 1, "passed": True}


def test_history_includes_runs_outside_recent_jobs_and_survives_restart(tmp_path, monkeypatch):
    from game_rsi import server
    monkeypatch.setattr(server, "ARTIFACTS", tmp_path)
    for i in range(105):
        seed_run(tmp_path, f"run-{i:03d}", "snake", i, [evaluation(1, i)])
    seed_run(tmp_path, "other-game", "mario", 200, [evaluation(1, 999)])
    seed_run(tmp_path, "exhibition", "snake", 201, [evaluation(1, 999)], kind="play")
    with TestClient(server.create_app()) as client:
        assert len(client.app.state.manager.jobs()) == 100
        result = client.get("/api/games/snake/comparisons").json()
        assert result["game"] == "snake"
        assert len(result["runs"]) == 105
        assert [run["job"]["id"] for run in result["runs"]] == [f"run-{i:03d}" for i in range(105)]
        assert result["runs"][0]["evaluations"][0]["baseline"] == 0
        assert client.get("/api/games/sokoban/comparisons").json()["runs"] == []
        assert client.get("/api/games/unknown/comparisons").status_code == 404
    with TestClient(server.create_app()) as client:
        assert client.get("/api/games/snake/comparisons").json() == result


def test_history_reads_appends_without_duplicates_and_preserves_partial_lines(tmp_path, monkeypatch):
    from game_rsi import server
    monkeypatch.setattr(server, "ARTIFACTS", tmp_path)
    directory = seed_run(tmp_path, "first", "snake", 1, [evaluation(1, 10)])
    path = directory / "events.jsonl"
    with TestClient(server.create_app()) as client:
        def rounds():
            return client.get("/api/games/snake/comparisons").json()["runs"][0]["evaluations"]
        assert len(rounds()) == 1
        assert len(rounds()) == 1
        with path.open("a") as stream:
            stream.write('invalid json\n' + json.dumps(evaluation(2, 20)))
        assert len(rounds()) == 1
        with path.open("a") as stream:
            stream.write("\n" + json.dumps(evaluation(1, 11)) + "\n")
        assert [(e["round"], e["baseline"]) for e in rounds()] == [(1, 11), (2, 20)]
        path.write_text(json.dumps(evaluation(3, 30)) + "\n")
        assert [(e["round"], e["baseline"]) for e in rounds()] == [(3, 30)]
