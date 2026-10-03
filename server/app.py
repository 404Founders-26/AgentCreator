"""FastAPI backend for the AgentCreator web UI. Each run executes in its own subprocess (one GPU job at a time)."""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from distiller.config import ROOT, deep_merge, load_config, parse_value, run_dir_for, safe_name, save_config, set_dotted
from distiller.presets import list_presets, load_preset
from distiller.runlog import STAGES, Run, read_json, read_jsonl

WEB = ROOT / "web"
app = FastAPI(title="AgentCreator")


class Job:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.run_name: str | None = None
        self.lock = threading.Lock()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


JOB = Job()
SHUTTING_DOWN = threading.Event()


def _stop_active_job(timeout: float = 15) -> None:
    proc = JOB.proc
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        proc.wait(timeout=timeout)
    except Exception:
        proc.kill()


@app.on_event("shutdown")
def _on_shutdown():
    SHUTTING_DOWN.set()
    _stop_active_job()


@app.post("/api/quit")
def quit_app():
    """Quit button: stop any running job, then end the server process (the launcher window closes too)."""
    SHUTTING_DOWN.set()

    def _bye():
        _stop_active_job()
        os._exit(0)

    threading.Timer(0.5, _bye).start()
    return {"quitting": True}


def runs_root() -> Path:
    return run_dir_for(load_config()).parent


def run_path(name: str) -> Path:
    p = runs_root() / safe_name(name)
    if not p.exists():
        raise HTTPException(404, f"run '{name}' not found")
    return p


def run_summary(p: Path) -> dict:
    run = Run(p, echo=False)
    st = run.state()
    active = JOB.alive() and JOB.run_name == p.name
    if not active:  # a stage left "running" by a crashed/killed process
        for s in st["stages"].values():
            if s.get("status") == "running":
                s["status"] = "interrupted"
    cfg = {}
    if (p / "config.yaml").exists():
        cfg = yaml.safe_load((p / "config.yaml").read_text(encoding="utf-8")) or {}
    return {
        "name": p.name, "active": active, "stages": st["stages"], "updated": st.get("updated"),
        "specialty": cfg.get("specialty"), "student": (cfg.get("student") or {}).get("model"),
        "teacher": (cfg.get("teacher") or {}).get("model"),
    }


# ----------------------------------------------------------------- config & presets
@app.get("/api/presets")
def presets():
    out = []
    for p in list_presets():
        p["config_overrides"] = load_preset(p["name"]).get("config_overrides") or {}
        out.append(p)
    return out


@app.get("/api/config/default")
def default_config():
    return load_config()


class TeacherCheck(BaseModel):
    teacher: dict


@app.post("/api/teacher/check")
def teacher_check(body: TeacherCheck):
    from distiller.teacher import check_teacher

    cfg = deep_merge(load_config(), {"teacher": body.teacher})
    return check_teacher(cfg)


class TeacherModels(BaseModel):
    base_url: str
    api_key: str | None = None


@app.post("/api/teacher/models")
def teacher_models(body: TeacherModels):
    """Models the teacher server has installed (Ollama: `ollama list`)."""
    from distiller.teacher import Teacher

    cfg = deep_merge(load_config(), {"teacher": {"base_url": body.base_url, "api_key": body.api_key or "ollama"}})
    t = Teacher(cfg)
    try:
        return {"ok": True, "models": sorted(m for m in t.list_models() if m)}
    except Exception as e:
        return {"ok": False, "models": [], "error": f"can't reach {body.base_url}: {e}"}
    finally:
        t.close()


@app.get("/api/catalog")
def catalog():
    from distiller.catalog import STUDENTS, TEACHER_SUGGESTIONS

    return {"students": STUDENTS, "teacher_suggestions": TEACHER_SUGGESTIONS, "gpu": gpu_info()}


class StudentAdvice(BaseModel):
    model: str
    load_in_4bit: bool = False
    max_seq_length: int = 2048
    batch_size: int = 2


@app.post("/api/student/advice")
def student_advice(body: StudentAdvice):
    from distiller.catalog import student_advice as advise

    gpu = gpu_info()
    vram = gpu["memory_total_mb"] / 1024 if gpu else None
    return advise(body.model, body.load_in_4bit, body.max_seq_length, body.batch_size, vram)


# ----------------------------------------------------------------- custom specialties
class CustomSpecialty(BaseModel):
    title: str
    description: str
    examples: list[str] = []
    answer_check: str = "none"
    num_prompts: int = 1000
    keep_thinking: bool = False
    system_prompt: str | None = None
    hf_dataset: dict | None = None
    upload: str | None = None
    eval_holdout: int = 60


@app.post("/api/specialties")
def create_specialty(body: CustomSpecialty):
    from distiller.presets import save_custom_preset

    try:
        preset = save_custom_preset(body.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"name": preset["name"], "preset": preset}


@app.delete("/api/specialties/{name}")
def delete_specialty(name: str):
    from distiller.presets import delete_custom_preset

    if not delete_custom_preset(name):
        raise HTTPException(404, "only custom specialties can be deleted")
    return {"deleted": name}


class PreviewSpecialty(BaseModel):
    teacher: dict
    title: str
    description: str
    examples: list[str] = []


@app.post("/api/specialties/preview")
def preview_specialty(body: PreviewSpecialty):
    from distiller.synth import preview_prompts

    cfg = deep_merge(load_config(), {"teacher": body.teacher})
    try:
        return {"prompts": preview_prompts(cfg, body.title, body.description, body.examples)}
    except Exception as e:
        raise HTTPException(502, f"teacher error: {e}")


class Upload(BaseModel):
    filename: str
    content: str


@app.post("/api/uploads")
def upload_prompts(body: Upload):
    """Accepts .jsonl (objects with a prompt/instruction/question field), .json (array) or .txt (one prompt per line)."""
    import json as _json
    import time as _time

    from distiller.presets import PRESET_DIR, PROMPT_FALLBACKS, slugify

    if len(body.content) > 30_000_000:
        raise HTTPException(413, "file too large (30 MB max)")
    name = body.filename.lower()
    rows: list[dict] = []
    try:
        if name.endswith(".jsonl"):
            for line in body.content.splitlines():
                if line.strip():
                    rows.append(_json.loads(line))
        elif name.endswith(".json"):
            data = _json.loads(body.content)
            rows = [d if isinstance(d, dict) else {"prompt": str(d)} for d in (data if isinstance(data, list) else [data])]
        else:
            rows = [{"prompt": l.strip()} for l in body.content.splitlines() if l.strip()]
    except ValueError as e:
        raise HTTPException(400, f"could not parse {body.filename}: {e}")
    clean = []
    for r in rows:
        prompt = next((r[k] for k in PROMPT_FALLBACKS if isinstance(r, dict) and r.get(k)), None)
        if prompt and str(prompt).strip():
            item = {"prompt": str(prompt).strip()}
            for extra in ("tests", "answer"):
                if r.get(extra):
                    item[extra] = r[extra]
            clean.append(item)
    if not clean:
        raise HTTPException(400, "no prompts found - use a 'prompt' (or instruction/question) field, or one prompt per line")
    out = PRESET_DIR / "custom" / "uploads" / f"{slugify(Path(body.filename).stem)}-{int(_time.time())}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(_json.dumps(c, ensure_ascii=False) for c in clean) + "\n", encoding="utf-8")
    return {"path": str(out.relative_to(ROOT)), "count": len(clean), "preview": [c["prompt"][:200] for c in clean[:3]]}


# ----------------------------------------------------------------- runs
@app.get("/api/runs")
def list_runs():
    root = runs_root()
    if not root.exists():
        return []
    items = [run_summary(p) for p in root.iterdir() if p.is_dir() and (p / "config.yaml").exists()]
    return sorted(items, key=lambda r: r.get("updated") or "", reverse=True)


class StartRun(BaseModel):
    config: dict
    stages: list[str] = STAGES
    overrides: list[str] = []  # "key.path=value" lines from the UI's extra-overrides box
    overwrite_config: bool = True


def _start(run_dir: Path, stages: list[str]):
    with JOB.lock:
        if JOB.alive():
            raise HTTPException(409, f"run '{JOB.run_name}' is still going - stop it first (one GPU job at a time)")
        bad = [s for s in stages if s not in STAGES]
        if bad or not stages:
            raise HTTPException(400, f"bad stages: {bad or 'none selected'}")
        log = open(run_dir / "process.log", "ab")
        kwargs = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        try:
            JOB.proc = subprocess.Popen(
                [sys.executable, "-m", "distiller", "run", "--run-dir", str(run_dir), "--stages", ",".join(stages)],
                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, **kwargs)
        finally:
            log.close()  # the child has its own handle; keeping ours open locks the file on Windows
        JOB.run_name = run_dir.name
    return {"started": run_dir.name, "pid": JOB.proc.pid, "stages": stages}


@app.post("/api/runs")
def start_run(body: StartRun):
    cfg = deep_merge(load_config(), body.config)
    for item in body.overrides:
        key, sep, val = item.partition("=")
        if not sep or not key.strip():
            raise HTTPException(400, f"bad override line: {item!r}")
        set_dotted(cfg, key.strip(), parse_value(val.strip()))
    cfg["project_name"] = safe_name(cfg.get("project_name") or "run")
    if JOB.alive():
        raise HTTPException(409, f"run '{JOB.run_name}' is still going - stop it first (one GPU job at a time)")
    run_dir = run_dir_for(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    if body.overwrite_config or not (run_dir / "config.yaml").exists():
        save_config(cfg, run_dir / "config.yaml")
    return _start(run_dir, body.stages)


class Resume(BaseModel):
    stages: list[str]


@app.post("/api/runs/{name}/resume")
def resume_run(name: str, body: Resume):
    return _start(run_path(name), body.stages)


@app.post("/api/runs/{name}/stop")
def stop_run(name: str):
    if not (JOB.alive() and JOB.run_name == safe_name(name)):
        raise HTTPException(400, "that run is not active")
    proc = JOB.proc
    assert proc is not None
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
    except Exception as e:
        print(f"graceful stop failed ({e!r}); terminating", flush=True)
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    run = Run(run_path(name), echo=False)
    for s, info in run.state()["stages"].items():
        if info.get("status") == "running":
            run.set_stage(s, "stopped")
    run.log("stopped from the web UI")
    return {"stopped": name}


@app.get("/api/runs/{name}")
def run_detail(name: str):
    p = run_path(name)
    info = run_summary(p)
    info["config"] = yaml.safe_load((p / "config.yaml").read_text(encoding="utf-8")) if (p / "config.yaml").exists() else {}
    info["data_stats"] = read_json(p / "data" / "stats.json")
    info["train"] = read_json(p / "train" / "summary.json")
    info["eval"] = read_json(p / "eval" / "results.json")
    info["export"] = read_json(p / "export" / "export.json")
    info["benchmark"] = read_json(p / "benchmark.json")
    hist = list(read_jsonl(p / "train" / "history.jsonl"))
    info["loss_curve"] = [{"step": h["step"], "loss": h.get("loss"), "eval_loss": h.get("eval_loss")}
                          for h in hist if "loss" in h or "eval_loss" in h]
    return info


@app.get("/api/runs/{name}/samples")
def samples(name: str, n: int = 5, split: str = "train"):
    p = run_path(name)
    fname = {"train": "data/train.jsonl", "raw": "data/raw.jsonl", "eval": "eval/distilled_outputs.jsonl"}.get(split)
    if not fname:
        raise HTTPException(400, "split must be train, raw or eval")
    out = []
    for row in read_jsonl(p / fname):
        out.append(row)
        if len(out) >= max(1, min(n, 50)):
            break
    return out


@app.get("/api/runs/{name}/logs")
async def logs(name: str, request: Request, tail: int = 300):
    path = run_path(name) / "run.log"

    async def stream():
        # read in binary and keep a byte offset: text-mode seek() is unreliable on Windows (\r\n)
        pos, partial = 0, b""
        if path.exists():
            data = path.read_bytes()
            for line in data.decode("utf-8", errors="replace").splitlines()[-tail:]:
                yield f"data: {line}\n\n"
            pos = len(data)
        idle = 0
        while not SHUTTING_DOWN.is_set() and not await request.is_disconnected():
            size = path.stat().st_size if path.exists() else 0
            if size < pos:  # file was recreated
                pos, partial = 0, b""
            if size > pos:
                with open(path, "rb") as f:
                    f.seek(pos)
                    chunk = partial + f.read()
                    pos = f.tell()
                *lines, partial = chunk.split(b"\n")
                for line in lines:
                    yield f"data: {line.decode('utf-8', errors='replace').rstrip(chr(13))}\n\n"
                idle = 0
            else:
                idle += 1
                if idle % 30 == 0:
                    yield ": keep-alive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/runs/{name}/process-log")
def process_log(name: str, lines: int = 200):
    p = run_path(name) / "process.log"
    if not p.exists():
        return {"text": ""}
    return {"text": "\n".join(p.read_bytes().decode("utf-8", errors="replace").splitlines()[-lines:])}


def gpu_info() -> dict | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            n, used, total, util = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
            return {"name": n, "memory_used_mb": int(used), "memory_total_mb": int(total), "utilization": int(util)}
    except Exception:
        pass
    return None


@app.get("/api/doctor")
def doctor():
    from distiller.doctor import run_checks

    return run_checks(load_config())


@app.get("/api/status")
def status():
    return {"active_run": JOB.run_name if JOB.alive() else None, "gpu": gpu_info()}


# ----------------------------------------------------------------- static UI
app.mount("/static", StaticFiles(directory=str(WEB)), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")
