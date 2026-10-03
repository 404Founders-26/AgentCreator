"""Run folder bookkeeping: log file, per-stage state, small JSON helpers."""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

STAGES = ["generate", "filter", "train", "eval", "export", "benchmark"]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    # On Windows os.replace fails with PermissionError while another process has the file open
    # (e.g. the web server reading state.json) - retry briefly instead of crashing the run.
    for attempt in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 + 0.01 * attempt)
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    for attempt in range(5):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default
        except PermissionError:  # Windows: file is being replaced right now
            time.sleep(0.05)
    return default


def read_jsonl(path: Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written last line after a crash


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


class Run:
    """One distillation run living in runs/<project_name>/."""

    def __init__(self, run_dir: str | Path, echo: bool = True):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "run.log"
        self.state_path = self.dir / "state.json"
        self.echo = echo
        self._lock = threading.Lock()

    # paths -----------------------------------------------------------------
    def path(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # logging ---------------------------------------------------------------
    def log(self, msg: str, stage: str | None = None) -> None:
        prefix = f"[{datetime.now().strftime('%H:%M:%S')}]"
        if stage:
            prefix += f" [{stage}]"
        line = f"{prefix} {msg}"
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        if self.echo:
            print(line, flush=True)

    # state -----------------------------------------------------------------
    def state(self) -> dict:
        st = read_json(self.state_path, None) or {}
        st.setdefault("stages", {})
        for s in STAGES:
            st["stages"].setdefault(s, {"status": "pending"})
        return st

    def set_stage(self, stage: str, status: str, **extra: Any) -> None:
        with self._lock:
            st = self.state()
            entry = st["stages"].get(stage, {})
            entry["status"] = status
            if status == "running":
                entry["started"] = now_iso()
                entry.pop("error", None)
                entry.pop("ended", None)
                entry.pop("progress", None)
            if status in ("done", "failed", "stopped", "skipped"):
                entry["ended"] = now_iso()
            entry.update(extra)
            st["stages"][stage] = entry
            st["updated"] = now_iso()
            write_json(self.state_path, st)

    def progress(self, stage: str, done: int, total: int, note: str = "") -> None:
        with self._lock:
            st = self.state()
            entry = st["stages"].setdefault(stage, {"status": "running"})
            entry["progress"] = {"done": done, "total": total, "note": note}
            st["updated"] = now_iso()
            write_json(self.state_path, st)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
