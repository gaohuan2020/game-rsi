from fastapi.testclient import TestClient

from game_rsi.server import create_app


def test_health_static_validation_and_origin(tmp_path, monkeypatch):
    import game_rsi.server as server
    monkeypatch.setattr(server, "ARTIFACTS", tmp_path)
    with TestClient(create_app()) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        assert "游戏自进化实验室" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200
        assert len(client.get("/api/catalog").json()["games"]) == 10
        assert client.get("/api/state").json()["jobs"] == []
        assert client.post("/api/evolve", json={"game": "invalid"}).status_code == 422
        assert client.post("/api/play", json={}, headers={"Origin": "https://evil.example"}).status_code == 403
        assert client.get("/api/jobs/not-found").status_code == 404


def test_incremental_events_preserve_partial_lines_and_evaluations(tmp_path, monkeypatch):
    import json
    import game_rsi.server as server
    from game_rsi.storage import atomic_json
    monkeypatch.setattr(server, "ARTIFACTS", tmp_path)
    directory = tmp_path / "jobs" / "history"
    atomic_json(directory / "job.json", {"id": "history", "kind": "evolve"})
    atomic_json(directory / "state.json", {"status": "completed"})
    path = directory / "events.jsonl"
    path.write_text(json.dumps({"time": 1, "kind": "evaluation", "passed": True}) + '\n{"time":2,')
    with TestClient(create_app()) as client:
        first = client.get("/api/jobs/history").json()["events"]
        assert len(first) == 1 and first[0]["passed"]
        with path.open("a") as stream:
            stream.write('"kind":"metric","step":1,"loss":0.5}\n')
        second = client.get("/api/jobs/history").json()["events"]
        assert len(second) == 2 and second[-1]["loss"] == 0.5
        assert client.get("/api/jobs/history").json()["events"] == second
