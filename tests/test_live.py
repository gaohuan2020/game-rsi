import pytest

from game_rsi.config import RunConfig
from game_rsi.live import SwitchGame, check_switch
from game_rsi.storage import Journal, atomic_json


def test_hot_switch_latest_request_and_stop(tmp_path):
    journal = Journal(tmp_path)
    atomic_json(tmp_path / "switch.json", {"token": "new", "config": RunConfig(game="mario").model_dump()})
    with pytest.raises(SwitchGame) as result:
        check_switch(journal, "old")
    assert result.value.config.game == "mario"
    check_switch(journal, "new")


def test_video_reader_sends_unique_frames_without_full_snapshots(tmp_path, monkeypatch):
    from game_rsi import server
    monkeypatch.setattr(server, "ARTIFACTS", tmp_path)
    manager = server.Manager()
    try:
        directory = tmp_path / "jobs" / "video"
        directory.mkdir()
        manager.active_play_id = "video"
        assert manager.video_frame(None) is None
        (directory / "live.jpg").write_bytes(b"first")
        key, jpeg = manager.video_frame(None)
        assert jpeg == b"first"
        assert manager.video_frame(key) is None
        (directory / ".image.tmp").write_bytes(b"second")
        (directory / ".image.tmp").replace(directory / "live.jpg")
        assert manager.video_frame(key)[1] == b"second"
    finally:
        manager.close()


def test_playback_defaults_remove_fixed_delay():
    assert RunConfig().delay_ms == 0
    assert RunConfig().display_fps == 30
