from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import ARTIFACTS, GAMES, ROOT, RunConfig, registry_key
from .storage import atomic_json

TERMINAL = {"completed", "stopped", "failed", "interrupted"}


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


class Manager:
    def __init__(self):
        self.jobs_dir = ARTIFACTS / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.lockfile = (ARTIFACTS / "server.lock").open("a")
        try:
            fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A game-rsi server already owns this artifacts directory") from error
        self.processes = {}
        self.event_cache = {}
        self.comparison_cache = {}
        self.active_play_id = None
        self.mutex = threading.RLock()
        # Recover only our own worker processes, verified by command and creation time.
        import psutil
        for directory in self.jobs_dir.iterdir():
            meta = read_json(directory / "job.json", {})
            state = read_json(directory / "state.json", {})
            if state.get("status") not in TERMINAL and meta:
                try:
                    process = psutil.Process(meta["pid"])
                    cmd = process.cmdline()
                    if ("game_rsi.worker" in cmd and str(directory) in cmd
                            and abs(process.create_time() - meta["process_started"]) < 1):
                        # Existing workers keep their output; reattach instead of repeating a training run.
                        self.processes[directory.name] = process
                        continue
                except (psutil.Error, KeyError):
                    pass
                atomic_json(directory / "state.json", {**state, "status": "interrupted", "phase": "interrupted",
                                                       "error": "Worker exited before server recovery"})

    def alive(self, job_id):
        import psutil
        process = self.processes.get(job_id)
        if isinstance(process, subprocess.Popen):
            return process.poll() is None
        return bool(process and process.is_running() and process.status() != psutil.STATUS_ZOMBIE)

    def jobs(self):
        out = []
        for directory in sorted(self.jobs_dir.iterdir(), reverse=True)[:100]:
            meta = read_json(directory / "job.json")
            if not meta:
                continue
            state = read_json(directory / "state.json", {"status": "starting", "phase": "starting"})
            if state.get("status") not in TERMINAL and not self.alive(directory.name):
                state.update(status="failed", phase="failed", error=state.get("error") or "Worker exited; see worker.log")
                atomic_json(directory / "state.json", state)
            out.append({**meta, **state})
        return out

    def start(self, kind, cfg):
        with self.mutex:
            if any(job["kind"] == kind and job["status"] not in TERMINAL for job in self.jobs()):
                raise HTTPException(409, f"A {kind} task is already active; stop it first")
            job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            directory = self.jobs_dir / job_id
            directory.mkdir()
            atomic_json(directory / "config.json", cfg.model_dump())
            atomic_json(directory / "state.json", {"status": "starting", "phase": "starting"})
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            env["GAME_RSI_ROOT"] = str(ROOT)
            env["GAME_RSI_ARTIFACTS"] = str(ARTIFACTS)
            env["PYTHONUNBUFFERED"] = "1"
            with (directory / "worker.log").open("w") as log:
                process = subprocess.Popen([sys.executable, "-m", "game_rsi.worker", kind, str(directory)],
                                           stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            import psutil
            meta = {"id": job_id, "kind": kind, "game": cfg.game, "backend": cfg.backend,
                    "created_at": time.time(), "config": cfg.model_dump(), "pid": process.pid,
                    "process_started": psutil.Process(process.pid).create_time()}
            atomic_json(directory / "job.json", meta)
            self.processes[job_id] = process
            if kind == "play":
                self.active_play_id = job_id
            return meta

    def start_play(self, cfg):
        with self.mutex:
            jobs = self.jobs()
            playing = next((j for j in jobs if j["kind"] == "play" and j["status"] not in TERMINAL), None)
            game_changed = playing is not None and playing["config"]["game"] != cfg.game
            stopped_evolution_ids = []
            for job in jobs:
                if (job["kind"] == "evolve" and job["status"] not in TERMINAL
                        and (game_changed or job["config"]["game"] != cfg.game)):
                    self.stop(job["id"])
                    stopped_evolution_ids.append(job["id"])
            if playing is None:
                return {**self.start("play", cfg), "stopped_evolution_ids": stopped_evolution_ids}
            directory = self.jobs_dir / playing["id"]
            token = uuid.uuid4().hex
            atomic_json(directory / "switch.json", {"token": token, "config": cfg.model_dump()})
            atomic_json(directory / "config.json", cfg.model_dump())
            meta = read_json(directory / "job.json")
            meta.setdefault("initial_config", meta["config"])
            meta.update(game=cfg.game, backend=cfg.backend, config=cfg.model_dump())
            atomic_json(directory / "job.json", meta)
            self.active_play_id = playing["id"]
            return {**meta, "switched": True, "control_token": token,
                    "stopped_evolution_ids": stopped_evolution_ids}

    def video_frame(self, previous):
        job_id = self.active_play_id
        if job_id is None:
            jobs = self.jobs()
            job = next((j for j in jobs if j["kind"] == "play"), None)
            if not job:
                return None
            self.active_play_id = job_id = job["id"]
        try:
            with (self.jobs_dir / job_id / "live.jpg").open("rb") as stream:
                key = (job_id, os.fstat(stream.fileno()).st_mtime_ns)
                if key == previous:
                    return None
                return key, stream.read()
        except OSError:
            return None

    def start_evolution(self, cfg):
        with self.mutex:
            jobs = self.jobs()
            if any(j["kind"] == "evolve" and j["status"] not in TERMINAL for j in jobs):
                raise HTTPException(409, "Evolution is already running")
            playing = next((j for j in jobs if j["kind"] == "play" and j["status"] not in TERMINAL), None)
            if playing and (registry_key(RunConfig.model_validate(playing["config"])) != registry_key(cfg)
                            or playing["config"]["max_episode_steps"] != cfg.max_episode_steps):
                # Switching the selected environment also switches our own exhibition actor.
                process = self.processes[playing["id"]]
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                deadline = time.monotonic() + 3
                while self.alive(playing["id"]) and time.monotonic() < deadline:
                    time.sleep(0.05)
                if self.alive(playing["id"]):
                    os.killpg(process.pid, signal.SIGKILL)
                directory = self.jobs_dir / playing["id"]
                old = read_json(directory / "state.json", {})
                atomic_json(directory / "state.json", {**old, "status": "stopped", "phase": "stopped",
                                                       "message": "Switched to selected evolution environment"})
                playing = None
            if playing is None:
                self.start_play(cfg.model_copy(update={"actor": "model"}))
            return self.start("evolve", cfg)

    def stop(self, job_id):
        with self.mutex:
            directory = self.jobs_dir / job_id
            if not (directory / "job.json").is_file():
                raise HTTPException(404, "Unknown task")
            state = read_json(directory / "state.json", {})
            if state.get("status") in TERMINAL:
                return {"id": job_id, "status": state["status"]}
            if (directory / "stop").exists():
                return {"id": job_id, "status": "stopping"}
            atomic_json(directory / "state.json", {**state, "status": "stopping", "phase": "stopping"})
            (directory / "stop").touch()
            # Graceful cancellation checks run every game/training step; terminate a hung load after a deadline.
            def deadline():
                time.sleep(20)
                if self.alive(job_id):
                    process = self.processes[job_id]
                    os.killpg(process.pid, signal.SIGTERM)
                    time.sleep(3)
                    if self.alive(job_id):
                        os.killpg(process.pid, signal.SIGKILL)
                    state = read_json(directory / "state.json", {})
                    atomic_json(directory / "state.json", {**state, "status": "stopped", "phase": "stopped"})
            threading.Thread(target=deadline, daemon=True).start()
            return {"id": job_id, "status": "stopping"}

    def details(self, job_id):
        if not job_id or "/" in job_id or job_id in (".", ".."):
            raise HTTPException(404, "Unknown task")
        directory = self.jobs_dir / job_id
        meta = read_json(directory / "job.json")
        if not meta:
            raise HTTPException(404, "Unknown task")
        events = []
        path = directory / "events.jsonl"
        if path.exists():
            # Incremental reads keep long training runs from rereading hundreds of MB
            # on each SSE update. Keep every evaluation, plus recent chart/log events.
            with self.mutex:
                cache = self.event_cache.setdefault(job_id, {"offset": 0, "metrics": deque(maxlen=1800),
                                                              "recent": deque(maxlen=700), "evaluations": []})
                if path.stat().st_size < cache["offset"]:
                    cache.update(offset=0, metrics=deque(maxlen=1800), recent=deque(maxlen=700), evaluations=[])
                with path.open("rb") as stream:
                    stream.seek(cache["offset"])
                    data = stream.read(4 * 1024 * 1024)
                complete = data.rfind(b"\n")
                if complete >= 0:
                    cache["offset"] += complete + 1
                    for line in data[:complete].splitlines():
                        with contextlib.suppress(ValueError):
                            event = json.loads(line)
                            if event["kind"] == "evaluation":
                                cache["evaluations"].append(event)
                            else:
                                cache["metrics" if event["kind"] == "metric" else "recent"].append(event)
                events = sorted(list(cache["metrics"]) + list(cache["recent"]) + cache["evaluations"],
                                key=lambda event: event["time"])
        logs = directory / "worker.log"
        return {"job": {**meta, **read_json(directory / "state.json", {})}, "events": events,
                "frame": read_json(directory / "frame.json"),
                "logs": logs.read_text(errors="replace")[-12000:] if logs.exists() else ""}

    def comparisons(self, game):
        """Read every experiment for a game, independent of the recent-jobs limit."""
        with self.mutex:
            runs = []
            for directory in self.jobs_dir.iterdir():
                meta = read_json(directory / "job.json", {})
                if (meta.get("kind") != "evolve"
                        or (meta.get("game") or meta.get("config", {}).get("game")) != game):
                    continue
                path = directory / "events.jsonl"
                cache = self.comparison_cache.setdefault(directory.name, {"offset": 0, "evaluations": {}})
                if not path.exists() or path.stat().st_size < cache["offset"]:
                    cache.update(offset=0, evaluations={})
                if path.exists():
                    with path.open("rb") as stream:
                        stream.seek(cache["offset"])
                        for line in stream:
                            # A concurrent append may not have finished its last JSON line yet.
                            if not line.endswith(b"\n"):
                                break
                            cache["offset"] += len(line)
                            with contextlib.suppress(ValueError, TypeError):
                                event = json.loads(line)
                                if isinstance(event, dict) and event.get("kind") == "evaluation":
                                    # Replayed observations replace their round, rather than duplicating it.
                                    cache["evaluations"][event.get("round", 1)] = event
                state = read_json(directory / "state.json", {})
                job = {**meta, **state}
                summary = {key: job[key] for key in ("id", "kind", "backend", "created_at", "status", "phase", "round")
                           if key in job}
                summary["game"] = game
                runs.append({"job": summary,
                             "evaluations": sorted(cache["evaluations"].values(), key=lambda e: e.get("round", 1))})
            runs.sort(key=lambda run: (run["job"].get("created_at", 0), run["job"]["id"]))
            return {"game": game, "runs": runs}

    def snapshot(self, include_video=True):
        with self.mutex:
            jobs = self.jobs()
            play_job = next((j for j in jobs if j["kind"] == "play"), None)
            if play_job:
                self.active_play_id = play_job["id"]
            evolve_job = next((j for j in jobs if j["kind"] == "evolve"), None)
            # Preserve the latest frame after a stop and after a page refresh.
            play = self.details(play_job["id"]) if play_job else None
            evolution = self.details(evolve_job["id"]) if evolve_job else None
            if not include_video and play and (play.get("frame") or {}).get("display_sequence"):
                play["frame"].pop("image", None)
            return {"time": time.time(), "jobs": jobs, "play": play, "evolution": evolution,
                    "champions": [read_json(p) for p in (ARTIFACTS / "champions").glob("*.json")]}

    def close(self):
        # The foreground server owns the workers; shutdown releases GPU/browser resources.
        for job_id, process in self.processes.items():
            if self.alive(job_id):
                (self.jobs_dir / job_id / "stop").touch()
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
        fcntl.flock(self.lockfile, fcntl.LOCK_UN)
        self.lockfile.close()


def create_app():
    @asynccontextmanager
    async def lifespan(app):
        app.state.manager = Manager()
        yield
        app.state.manager.close()

    app = FastAPI(title="Game RSI", lifespan=lifespan)
    web = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.middleware("http")
    async def local_mutations(request: Request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            if (request.headers.get("sec-fetch-site") == "cross-site"
                    or (origin and urlparse(origin).netloc != request.headers.get("host"))):
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": "Cross-origin control requests are refused"}, status_code=403)
        return await call_next(request)

    @app.get("/")
    async def index():
        return FileResponse(web / "index.html")

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "project": "game-rsi", "artifacts": str(ARTIFACTS)}

    @app.get("/api/catalog")
    async def catalog():
        entries = []
        for game in GAMES:
            path = ROOT / "vendor" / "PlayJev" / "games" / game / "pj.json"
            spec = read_json(path, {})
            entries.append({"id": game, "title": spec.get("title", game), "available": bool(spec)})
        return {"games": entries, "defaults": RunConfig().model_dump(),
                "backend_models": {"playjev": "OmniJev/PlayJev-0.8B", "rsi": "shgao/rsi-jev-v1.0-qwen3.5-0.8b"}}

    @app.get("/api/state")
    async def state():
        return await asyncio.to_thread(app.state.manager.snapshot)

    @app.get("/api/jobs/{job_id}")
    async def detail(job_id: str):
        return await asyncio.to_thread(app.state.manager.details, job_id)

    @app.get("/api/games/{game}/comparisons")
    async def comparisons(game: str):
        if game not in GAMES:
            raise HTTPException(404, "Unknown game")
        return await asyncio.to_thread(app.state.manager.comparisons, game)

    @app.post("/api/play")
    async def start_play(cfg: RunConfig):
        return await asyncio.to_thread(app.state.manager.start_play, cfg)

    @app.get("/api/play/stream")
    async def video(request: Request):
        async def frames():
            previous = None
            while not await request.is_disconnected():
                item = await asyncio.to_thread(app.state.manager.video_frame, previous)
                if item:
                    previous, jpeg = item
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode()
                           + b"\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(1 / 60)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.post("/api/evolve")
    async def start_evolve(cfg: RunConfig):
        return await asyncio.to_thread(app.state.manager.start_evolution, cfg)

    @app.post("/api/jobs/{job_id}/stop")
    async def stop(job_id: str):
        if "/" in job_id or job_id in (".", ".."):
            raise HTTPException(404, "Unknown task")
        return app.state.manager.stop(job_id)

    @app.get("/api/events")
    async def events(request: Request):
        async def stream():
            while not await request.is_disconnected():
                snapshot = await asyncio.to_thread(app.state.manager.snapshot, False)
                yield "event: snapshot\ndata: " + json.dumps(snapshot, ensure_ascii=False) + "\n\n"
                await asyncio.sleep(0.6)
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
