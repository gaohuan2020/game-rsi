import json

import pytest
from fastapi.testclient import TestClient

from game_rsi.config import RunConfig
from game_rsi.storage import Cancelled, Journal, atomic_json


def seed_task(manager, task_id, kind, game, phase):
    cfg = RunConfig(game=game).model_dump()
    directory = manager.jobs_dir / task_id
    journal = Journal(directory)
    journal.publish(status='running', phase=phase)
    meta = {'id': task_id, 'kind': kind, 'game': game, 'config': cfg}
    atomic_json(directory / 'job.json', meta)
    return {**meta, 'status': 'running', 'phase': phase}, journal


@pytest.mark.parametrize('phase', ['loading', 'collecting', 'training', 'saving', 'evaluating'])
def test_game_switch_cancels_evolution_at_every_stage(tmp_path, monkeypatch, phase):
    from game_rsi import server
    monkeypatch.setattr(server, 'ARTIFACTS', tmp_path)
    with TestClient(server.create_app()) as client:
        manager = client.app.state.manager
        play, _ = seed_task(manager, 'play', 'play', 'snake', 'playing')
        evolve, journal = seed_task(manager, 'evolve', 'evolve', 'snake', phase)
        monkeypatch.setattr(manager, 'jobs', lambda: [play, evolve])
        monkeypatch.setattr(manager, 'alive', lambda job_id: False)
        champion = tmp_path / 'champions' / 'retained.json'
        atomic_json(champion, {'checkpoint': 'retained-champion'})
        response = client.post('/api/play', json={'game': 'mario'})
        assert response.status_code == 200
        assert response.json()['stopped_evolution_ids'] == ['evolve']
        with pytest.raises(Cancelled):
            journal.check()
        state = json.loads((journal.directory / 'state.json').read_text())
        assert state['status'] == 'stopping'
        request = json.loads((manager.jobs_dir / 'play' / 'switch.json').read_text())
        assert request['config']['game'] == 'mario'
        assert json.loads(champion.read_text())['checkpoint'] == 'retained-champion'


def test_same_game_actor_change_keeps_evolution_running(tmp_path, monkeypatch):
    from game_rsi import server
    monkeypatch.setattr(server, 'ARTIFACTS', tmp_path)
    with TestClient(server.create_app()) as client:
        manager = client.app.state.manager
        play, _ = seed_task(manager, 'play', 'play', 'snake', 'playing')
        evolve, journal = seed_task(manager, 'evolve', 'evolve', 'snake', 'training')
        monkeypatch.setattr(manager, 'jobs', lambda: [play, evolve])
        response = client.post('/api/play', json={'game': 'snake', 'actor': 'teacher'})
        assert response.status_code == 200
        assert response.json()['stopped_evolution_ids'] == []
        journal.check()
        assert not (journal.directory / 'stop').exists()


def test_game_switch_cancels_evolution_without_a_live_worker(tmp_path, monkeypatch):
    from game_rsi import server
    monkeypatch.setattr(server, 'ARTIFACTS', tmp_path)
    with TestClient(server.create_app()) as client:
        manager = client.app.state.manager
        evolve, journal = seed_task(manager, 'evolve', 'evolve', 'snake', 'training')
        monkeypatch.setattr(manager, 'jobs', lambda: [evolve])
        monkeypatch.setattr(manager, 'alive', lambda job_id: False)
        monkeypatch.setattr(manager, 'start', lambda kind, cfg: {'id': 'new-play', 'game': cfg.game})
        response = client.post('/api/play', json={'game': 'mario'})
        assert response.json()['stopped_evolution_ids'] == ['evolve']
        with pytest.raises(Cancelled):
            journal.check()


def test_stopping_completed_job_does_not_reopen_it(tmp_path, monkeypatch):
    from game_rsi import server
    monkeypatch.setattr(server, 'ARTIFACTS', tmp_path)
    with TestClient(server.create_app()) as client:
        manager = client.app.state.manager
        _, journal = seed_task(manager, 'done', 'evolve', 'snake', 'completed')
        journal.publish(status='completed')
        response = client.post('/api/jobs/done/stop')
        assert response.json()['status'] == 'completed'
        assert not (journal.directory / 'stop').exists()
