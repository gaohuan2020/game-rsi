"""Measure real MJPEG deliveries, model reuse and browser/UI layout; play only."""
import asyncio
import json
import time

import httpx
from playwright.async_api import async_playwright

from game_rsi.config import ROOT

BASE = "http://127.0.0.1:8092"


async def wait_frame(client, game, token=None):
    started = time.monotonic()
    while time.monotonic() - started < 90:
        state = (await client.get("/api/state")).json()
        play = state.get("play") or {}
        if play.get("job", {}).get("status") == "failed":
            raise RuntimeError(play["job"].get("error"))
        frame = play.get("frame") or {}
        if frame.get("game") == game and frame.get("display_sequence") and (
                token is None or play["job"].get("control_token") == token):
            return state
        await asyncio.sleep(.05)
    raise TimeoutError(game)


async def fps(client, seconds=5):
    count, started = 0, time.monotonic()
    pending = b""
    async with client.stream("GET", "/api/play/stream") as stream:
        stream.raise_for_status()
        async for chunk in stream.aiter_bytes():
            pending += chunk
            count += pending.count(b"--frame\r\n")
            pending = pending[-8:]
            if time.monotonic() - started > seconds:
                break
    return round(count / (time.monotonic() - started), 2)


async def main():
    result = {"games": [], "page_errors": []}
    async with httpx.AsyncClient(base_url=BASE, timeout=90) as client:
        state = (await client.get("/api/state")).json()
        current = state.get("play", {}).get("job", {})
        previous = current.get("config")
        cfg = {"game":"mario","backend":"playjev","model":"OmniJev/PlayJev-0.8B",
               "actor":"model","input_mode":"pixels","max_episode_steps":300,"delay_ms":0,"display_fps":30}
        job_id = None
        try:
            for game in ("mario", "flappy", "mario", "flappy"):
                cfg["game"] = game
                started = time.monotonic()
                response = await client.post("/api/play", json=cfg)
                response.raise_for_status()
                job = response.json()
                snapshot = await wait_frame(client, game, job.get("control_token"))
                if job_id:
                    assert job["id"] == job_id, "Switch restarted the worker"
                job_id = job["id"]
                elapsed = time.monotonic() - started
                delivered = await fps(client)
                result["games"].append({"game":game,"switch_to_first_frame_s":round(elapsed,3),
                                         "received_fps":delivered,"worker_pid":snapshot["play"]["job"]["pid"]})
                print(json.dumps(result["games"][-1]), flush=True)
                assert delivered > 8, "Playback is still at monitoring refresh speed"
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
                page = await browser.new_page(viewport={"width":1512,"height":1100})
                page.on("pageerror", lambda error: result["page_errors"].append(str(error)))
                await page.goto(BASE, wait_until="domcontentloaded")
                await page.wait_for_function("document.getElementById('game-frame').naturalWidth>0")
                widths = await page.evaluate("({game:document.querySelector('.game-panel').clientWidth,training:document.querySelector('.train-panel').clientWidth})")
                result["layout"] = widths
                assert widths["game"] > widths["training"] * 1.7
                out = ROOT / "artifacts" / "qa"
                await page.screenshot(path=str(out / "live-desktop.png"), full_page=True)
                # Use the actual UI dropdown; it should switch without a stop/start action.
                await page.select_option("#game", "flappy")
                await page.wait_for_function("document.getElementById('screen-source').textContent.startsWith('FLAPPY')", timeout=15000)
                result["dropdown_hot_switch"] = True
                await page.set_viewport_size({"width":390,"height":844})
                await page.screenshot(path=str(out / "live-mobile.png"), full_page=True)
                assert not await page.evaluate("document.documentElement.scrollWidth>innerWidth")
                await browser.close()
            assert not result["page_errors"]
        finally:
            # Restore the user's selection using the hot-switch API; no training is launched.
            if previous:
                previous.update(delay_ms=0, display_fps=30)
                await client.post("/api/play", json=previous)
        out = ROOT / "artifacts" / "qa" / "live-performance.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
