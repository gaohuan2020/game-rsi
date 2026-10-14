"""Smooth live viewing without changing the training/evaluation action contract."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from collections import OrderedDict

from .config import RunConfig
from .policies import Policy, release_policy

# Observe actual game timer callbacks. Do not split pj.step, inject extra ticks,
# repeat flap/jump inputs, or alter the terminal observation used by the model.
CAPTURE_JS = r"""
(() => {
  const capture = window.__gameRsiVideo = {enabled:false, frames:[], last:-Infinity, interval:1000/30};
  function sample() {
    if (!capture.enabled || !window.pj || !pj.hook?.canvas) return;
    const t = pj.now();
    if (t - capture.last < capture.interval - 0.01) return;
    try {
      const image = pj.frame();
      if (image) { capture.frames.push({t,image}); capture.last = t; }
      // Bound memory even when a hook advances through a long introduction.
      if (capture.frames.length > 120) capture.frames.shift();
    } catch (_) {}
  }
  for (const name of ['setTimeout','setInterval','requestAnimationFrame']) {
    const original = window[name];
    window[name] = function(callback, ...args) {
      if (typeof callback !== 'function') return original.call(window, callback, ...args);
      return original.call(window, function(...values) {
        const result = callback.apply(this, values);
        sample();
        return result;
      }, ...args);
    };
  }
  window.webkitRequestAnimationFrame = window.mozRequestAnimationFrame = window.requestAnimationFrame;
})();
"""


class SwitchGame(Exception):
    def __init__(self, config, token):
        self.config, self.token = config, token


def check_switch(journal, token):
    journal.check()
    path = journal.directory / "switch.json"
    if path.exists():
        message = json.loads(path.read_text())
        if message["token"] != token:
            raise SwitchGame(RunConfig.model_validate(message["config"]), message["token"])


class PolicyCache:
    """Keep two resident checkpoints across game changes; release older GPU allocations."""
    def __init__(self):
        self.entries = OrderedDict()

    async def get(self, cfg, checkpoint):
        key = (cfg.backend, checkpoint, cfg.device, cfg.input_mode)
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        # Evict before loading to bound the peak allocation.
        if len(self.entries) >= 2:
            _, old = self.entries.popitem(last=False)
            release_policy(old)
        policy = await asyncio.to_thread(Policy, cfg, checkpoint)
        self.entries[key] = policy
        return policy

    def close(self):
        for policy in self.entries.values():
            release_policy(policy)
        self.entries.clear()


class LiveBrowser:
    """One browser and bounded game pages reused by a live play worker."""
    async def __aenter__(self):
        from playwright.async_api import async_playwright
        from playjev.env import FONTS_CONF, GameServer
        self.server = GameServer()
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(headless=True,
            env={**os.environ, "FONTCONFIG_FILE": str(FONTS_CONF)},
            args=["--no-sandbox", "--disable-gpu", "--mute-audio", "--disable-dev-shm-usage",
                  "--disable-background-timer-throttling", "--disable-renderer-backgrounding"])
        self.context = await self.browser.new_context()
        self.pages = {}
        return self

    async def game(self, game):
        from playjev.env import GamePage, load_spec
        if game not in self.pages:
            page = GamePage(self.context, load_spec(game), self.server.base_url)
            await page.open()
            await page.page.add_init_script(CAPTURE_JS)
            self.pages[game] = page
        return self.pages[game]

    async def reset(self, page, seed, fps):
        obs = await page.reset(seed)
        await page.page.evaluate("fps => { const c=__gameRsiVideo; c.frames=[]; c.last=pj.now(); c.interval=1000/fps; c.enabled=true; }", fps)
        return obs

    async def step(self, page, action):
        before = await page.page.evaluate("pj.now()")
        obs = await page.step(action)
        captured = await page.page.evaluate("() => { const c=__gameRsiVideo; const frames=c.frames; c.frames=[]; return frames; }")
        frames = [{"t": item["t"], "frame": base64.b64decode(item["image"].split(",", 1)[1])}
                  for item in captured]
        # Always retain the exact final observation (including death/score changes).
        if not frames or frames[-1]["frame"] != obs["frame"]:
            frames.append({"t": obs.get("t", before), "frame": obs["frame"]})
        return obs, frames, before

    async def __aexit__(self, *exc):
        await self.browser.close()
        await self.pw.stop()
        self.server.close()


async def show_frames(journal, obs, frames, before, options, probs, action, cfg, token, **extra):
    """Pace actual intermediate images while the next inference runs concurrently."""
    duration = max(0, (obs.get("t", before) - before) / 1000)
    # Cap long hook-internal transitions rather than buffering seconds of stale video.
    duration = min(duration, 0.3)
    started = time.monotonic()
    span = max(obs.get("t", before) - before, 1)
    last_shown = -1.0
    for index, frame in enumerate(frames):
        check_switch(journal, token)
        offset = duration * max(0, min(1, (frame["t"] - before) / span))
        if index < len(frames) - 1 and offset - last_shown < 1 / cfg.display_fps - .001:
            continue
        wait = started + offset - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        check_switch(journal, token)
        journal.frame({**obs, "frame": frame["frame"]}, options, probs, action,
                      game=cfg.game, source="live", **extra)
        last_shown = offset
