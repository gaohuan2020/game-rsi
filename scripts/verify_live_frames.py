"""Check actual intermediate images and exact action/terminal-state parity."""
import asyncio
import hashlib
import json

from game_rsi.config import ROOT, upstream_paths
from game_rsi.live import LiveBrowser


async def main():
    upstream_paths()
    from playjev.env import VecGame
    report = []
    for game, actions in (("mario", [6,6,2,4,2,6,2,4]), ("flappy", [0,1,1,0,1,1,0,1])):
        reference = []
        async with VecGame(game, n=1) as env:
            await env.reset([17])
            for action in actions:
                obs = (await env.step([action]))[0]
                reference.append(obs)
        frames_count = 0
        hashes = set()
        async with LiveBrowser() as browser:
            page = await browser.game(game)
            await browser.reset(page, 17, 30)
            for action, expected in zip(actions, reference):
                obs, frames, before = await browser.step(page, action)
                for key in ("score", "done", "t", "steps", "info"):
                    assert obs.get(key) == expected.get(key), (game, key)
                assert obs["frame"] == expected["frame"], game
                frames_count += len(frames)
                hashes.update(hashlib.md5(frame["frame"]).hexdigest() for frame in frames)
        assert frames_count > len(actions), (game, frames_count)
        assert len(hashes) > len(actions), (game, len(hashes))
        entry = {"game": game, "actions": len(actions), "captured_frames": frames_count,
                 "unique_images": len(hashes), "terminal_state_and_frame_parity": True}
        print(json.dumps(entry), flush=True)
        report.append(entry)
    out = ROOT / "artifacts" / "qa" / "live-frame-parity.json"
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
