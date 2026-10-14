from __future__ import annotations

import json
import math
import time

from .config import RunConfig, upstream_paths

INSTRUCTIONS = "Which move should the player make next?"


def game_state(obs: dict) -> str:
    # Structured-state RSI models get mechanics, never teacher labels or accumulated outcomes.
    drop = {"maxX", "ticks", "mapFallback", "dead", "won", "levelString"}
    info = {k: v for k, v in (obs.get("info") or {}).items() if k not in drop}
    # A text checkpoint benefits from explicit geometry. These are observations derived
    # from the grid, not a policy: no move is chosen and no teacher target is exposed.
    head, food, grid = info.get("head"), info.get("food"), info.get("grid")
    if isinstance(head, dict) and isinstance(food, list) and isinstance(grid, str):
        cells = grid.split("|")
        r, c = head["row"], head["col"]
        info["relative_food"] = {"row_delta": food[0] - r, "col_delta": food[1] - c}
        preview = {}
        for name, dr, dc in (("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1)):
            nr, nc = r + dr, c + dc
            blocked = not (0 <= nr < len(cells) and 0 <= nc < len(cells[nr])) or cells[nr][nc] == "#"
            preview[name] = {"blocked": blocked, "food_distance": abs(food[0] - nr) + abs(food[1] - nc)}
        info["move_geometry"] = preview
    return json.dumps(info, ensure_ascii=False, separators=(",", ":"))


def question(options):
    return {"action": {"type": "choice", "instructions": INSTRUCTIONS,
                       "criteria": {o["name"]: o["description"] for o in options}}}


def probabilities(values, n):
    if len(values) != n or any(not math.isfinite(x) or x < 0 for x in values) or sum(values) <= 0:
        raise ValueError("Model returned invalid action probabilities")
    return [float(x) / sum(values) for x in values]


class Policy:
    def __init__(self, cfg: RunConfig, checkpoint: str):
        upstream_paths()
        self.cfg, self.checkpoint = cfg, checkpoint
        self.latency_ms = 0
        if cfg.backend == "playjev":
            from playjev.model import PlayJevModel
            self.model = PlayJevModel(checkpoint, device=cfg.device,
                                      dtype="bfloat16" if cfg.device.startswith("cuda") else "float32").load()
            self.mode = "pixels"
        else:
            from rsijev import Decider
            self.model = Decider(checkpoint, device=cfg.device)
            has_vision = self.model.image_limits.get("supported", False)
            self.mode = ("pixels" if has_vision else "state") if cfg.input_mode == "auto" else cfg.input_mode
            if self.mode == "pixels" and not has_vision:
                raise ValueError("This RSI-Jev checkpoint is text-only. Choose state/auto or a VL release.")

    def decide(self, obs, options):
        started = time.perf_counter()
        if self.cfg.backend == "playjev":
            p = self.model.decide([obs["frame"]], options, batch_size=1)[0].probs
        else:
            state = "<image>" if self.mode == "pixels" else game_state(obs)
            images = [obs["frame"]] if self.mode == "pixels" else None
            pmap = self.model.decide(state, question(options), images=images)["action"]["probabilities"]
            p = [pmap[o["name"]] for o in options]
        self.latency_ms = (time.perf_counter() - started) * 1000
        return probabilities(p, len(options))


def choose(probs, banned):
    allowed = [i for i in range(len(probs)) if i not in banned] or list(range(len(probs)))
    return max(allowed, key=probs.__getitem__)


def noop_rule(before, after, action, banned):
    if not after["done"] and before["score"] == after["score"] and before["frame"] == after["frame"]:
        banned.add(action)
    else:
        banned.clear()


def release_policy(policy):
    import gc
    import torch
    del policy.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
