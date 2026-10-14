from __future__ import annotations

import hashlib
import json
import os
import time
from collections import deque
from pathlib import Path


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    for file in sorted(path.glob("*.safetensors")):
        h.update(file.name.encode())
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


class Cancelled(Exception):
    pass


class Journal:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.state = {"status": "running", "phase": "starting", "round": 0, "progress": 0, "started_at": time.time()}
        self.display_times = deque()
        self.display_sequence = 0
        self.publish()

    def check(self):
        if (self.directory / "stop").exists():
            raise Cancelled("用户停止了任务；冠军模型保持原状")

    def event(self, kind: str, **values):
        entry = {"time": time.time(), "kind": kind, **values}
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")

    def publish(self, **values):
        self.state.update(values, updated_at=time.time())
        atomic_json(self.directory / "state.json", self.state)

    def phase(self, name: str, **values):
        self.check()
        self.publish(phase=name, progress=0, **values)
        self.event("phase", phase=name, **values)

    def frame(self, obs: dict, options: list, probs=None, action=None, **extra):
        import base64
        frame = obs.get("frame")
        if frame:
            if extra.get("source") == "live":
                self.display_sequence += 1
                now = time.monotonic()
                self.display_times.append(now)
                while len(self.display_times) > 1 and self.display_times[0] < now - 1:
                    self.display_times.popleft()
                jpeg = frame
                if frame.startswith(b"\x89PNG"):
                    import io
                    from PIL import Image
                    buffer = io.BytesIO()
                    Image.open(io.BytesIO(frame)).convert("RGB").save(buffer, "JPEG", quality=85)
                    jpeg = buffer.getvalue()
                target = self.directory / "live.jpg"
                temporary = self.directory / ".live.jpg.tmp"
                temporary.write_bytes(jpeg)
                os.replace(temporary, target)
                extra.update(display_sequence=self.display_sequence, display_fps=len(self.display_times))
            # The hook can return PNG as well as JPEG; preserve its true content type.
            mime = "image/png" if frame.startswith(b"\x89PNG") else "image/jpeg"
            atomic_json(self.directory / "frame.json", {
                "image": f"data:{mime};base64," + base64.b64encode(frame).decode(),
                "score": obs.get("score", 0), "step": obs.get("steps", 0), "done": obs.get("done", False),
                "actions": [x["name"] for x in options], "probabilities": probs,
                "action": options[action]["name"] if action is not None else None,
                "updated_at": time.time(), **extra,
            })
