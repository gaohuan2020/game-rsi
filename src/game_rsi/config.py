from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

ROOT = Path(os.environ.get("GAME_RSI_ROOT", Path(__file__).resolve().parents[2])).resolve()
ARTIFACTS = Path(os.environ.get("GAME_RSI_ARTIFACTS", ROOT / "artifacts")).resolve()
GAMES = ("snake", "2048", "tetris", "pacman", "racer", "invaders", "sokoban", "mario", "flappy", "breakout")
Backend = Literal["playjev", "rsi"]
DEFAULT_MODELS = {"playjev": "OmniJev/PlayJev-0.8B", "rsi": "shgao/rsi-jev-v1.0-qwen3.5-0.8b"}
STATE_CONTRACT = "game-state-v2-geometry"


def upstream_paths():
    # Insert ahead of previously installed editable RSI-Jev checkouts.
    for name in ("RSI-Jev", "PlayJev"):
        path = ROOT / "vendor" / name
        if not path.is_dir():
            raise RuntimeError(f"Missing {path}. Run python scripts/bootstrap.py first.")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


class RunConfig(BaseModel):
    game: Literal["snake", "2048", "tetris", "pacman", "racer", "invaders", "sokoban", "mario", "flappy", "breakout"] = "snake"
    backend: Backend = "rsi"
    model: str = ""
    device: str = "cuda:0"
    rounds: int = Field(3, ge=1, le=30)
    collect_steps: int = Field(512, ge=32, le=100000)
    train_steps: int = Field(80, ge=1, le=20000)
    batch_size: int = Field(4, ge=1, le=64)
    learning_rate: float = Field(0.0001, ge=0.0000001, le=0.01)
    tune_tower: bool = False
    epsilon: float = Field(0.05, ge=0, le=1)
    teacher_mix: float = Field(0.25, ge=0, le=1)
    eval_episodes: int = Field(8, ge=2, le=64)
    max_episode_steps: int = Field(300, ge=10, le=10000)
    seed: int = Field(17, ge=0, le=100000000)
    min_gain: float = Field(0.02, ge=0, le=1)
    require_ci: bool = True
    replay_limit: int = Field(8192, ge=32, le=100000)
    delay_ms: int = Field(0, ge=0, le=2000)
    display_fps: int = Field(30, ge=10, le=60)
    actor: Literal["model", "teacher", "random"] = "model"
    input_mode: Literal["auto", "pixels", "state"] = "auto"

    @model_validator(mode="after")
    def defaults(self):
        self.model = self.model.strip() or DEFAULT_MODELS[self.backend]
        if self.backend == "playjev" and self.input_mode == "state":
            raise ValueError("PlayJev uses pixels; choose RSI-Jev for structured state input")
        return self


def registry_key(config: RunConfig) -> str:
    import hashlib
    # Different parent models/input contracts do not silently share a champion.
    parent = hashlib.sha256(config.model.encode()).hexdigest()[:12]
    mode = config.input_mode
    if config.backend == "playjev":
        mode = "pixels"
    elif mode == "auto":
        meta_path = Path(config.model) / "meta.json"
        if not meta_path.is_file():
            try:
                from huggingface_hub import try_to_load_from_cache
                cached = try_to_load_from_cache(config.model, "meta.json")
                if isinstance(cached, str):
                    meta_path = Path(cached)
            except (ImportError, ValueError):
                pass
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            vision = meta.get("vision") or (meta.get("release") or {}).get("vision") or (
                (meta.get("spec") or {}).get("fit_extra") or {}).get("vision")
            mode = "pixels" if vision else "state"
    return f"{config.backend}-{config.game}-{mode}-{parent}"


def champion(config: RunConfig) -> dict:
    path = ARTIFACTS / "champions" / f"{registry_key(config)}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"checkpoint": config.model, "version": "base", "backend": config.backend}
