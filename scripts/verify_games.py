"""Exercise the actual browser hook and teacher for every bundled PlayJev game."""
import asyncio
import json
from pathlib import Path

from game_rsi.config import GAMES, ROOT, upstream_paths


async def main():
    upstream_paths()
    from playjev.env import VecGame
    from playjev.teachers import make_teacher
    results = []
    for game in GAMES:
        async with VecGame(game, n=1) as env:
            obs = (await env.reset([17]))[0]
            teacher = make_teacher(game, env.actions)
            teacher.reset()
            for _ in range(3):
                p = teacher.act(obs)
                assert abs(sum(p) - 1) < 1e-4
                obs = (await env.step([max(range(len(p)), key=p.__getitem__)]))[0]
                assert obs.get("frame")
                assert not obs.get("errors"), obs.get("errors")
            results.append({"game": game, "actions": len(env.actions), "score": obs["score"], "frame_bytes": len(obs["frame"])})
            print(json.dumps(results[-1]), flush=True)
    out = ROOT / "artifacts" / "qa" / "games-report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
