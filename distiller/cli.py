"""Command line entry point:  python -m distiller <command> ..."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

import yaml

from .config import DEFAULT_CONFIG, deep_merge, load_config, run_dir_for
from .presets import list_presets, load_preset
from .runlog import STAGES, Run


def build_config(config_path: str | None, overrides: list[str] | None) -> dict:
    """defaults -> preset.config_overrides -> your config file -> --set flags."""
    user = {}
    if config_path:
        user = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    cfg = load_config(None)
    specialty = user.get("specialty", cfg.get("specialty"))
    for o in overrides or []:
        if o.startswith("specialty="):
            specialty = o.split("=", 1)[1].strip()
    cfg = deep_merge(cfg, load_preset(specialty).get("config_overrides") or {})
    cfg = deep_merge(cfg, user)
    return load_config_from_dict(cfg, overrides)


def load_config_from_dict(cfg: dict, overrides: list[str] | None) -> dict:
    from .config import parse_value, set_dotted

    for item in overrides or []:
        k, _, v = item.partition("=")
        set_dotted(cfg, k.strip(), parse_value(v.strip()))
    return cfg


def _raise_interrupt(*_):
    raise KeyboardInterrupt


def setup_console() -> None:
    """Windows consoles/pipes default to cp1252 - make every print UTF-8 so model text can't crash a run."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # HF cache warns on Windows without dev mode
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def cmd_run(a) -> int:
    if a.run_dir:  # used by the web UI: config already lives in the run folder
        run_dir = Path(a.run_dir)
        cfg = load_config(run_dir / "config.yaml", a.set)
    else:
        cfg = build_config(a.config, a.set)
        if a.name:
            cfg["project_name"] = a.name
        run_dir = run_dir_for(cfg)
    from .pipeline import parse_stages, run_pipeline

    stages = parse_stages(a.stages)
    # make the Stop button behave like Ctrl+C so state.json records "stopped":
    # Linux/WSL sends SIGTERM, Windows sends CTRL_BREAK_EVENT (SIGBREAK)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _raise_interrupt)
    return run_pipeline(cfg, Run(run_dir), stages)


def cmd_presets(a) -> int:
    for p in list_presets():
        print(f"{p['name']:<12} {p['title']}\n             sources: {', '.join(p['sources'])} | eval: {p['eval']}")
    return 0


def cmd_check(a) -> int:
    from .teacher import check_teacher

    cfg = build_config(a.config, a.set)
    res = check_teacher(cfg)
    print(json.dumps(res, indent=2))
    return 0 if res.get("ok") else 1


def cmd_doctor(a) -> int:
    from .doctor import print_report, run_checks

    return print_report(run_checks(build_config(a.config, a.set)))


def cmd_status(a) -> int:
    run = Run(a.run_dir, echo=False)
    print(json.dumps(run.state(), indent=2))
    return 0


def cmd_ui(a) -> int:
    import uvicorn

    from .config import ROOT

    if str(ROOT) not in sys.path:  # works no matter which folder the command is started from
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    print(f"AgentCreator UI on http://{a.host}:{a.port}  (Ctrl+C to quit)")
    # timeout_graceful_shutdown: the live-log streams never end on their own, so without it Ctrl+C hangs
    uvicorn.run("server.app:app", host=a.host, port=a.port, log_level="warning", timeout_graceful_shutdown=2)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="distiller", description="Distil an open LLM into a small specialist.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run pipeline stages")
    r.add_argument("--config", "-c", help=f"YAML config (merged over {DEFAULT_CONFIG.name})")
    r.add_argument("--name", help="project / run folder name")
    r.add_argument("--stages", default="all", help=f"comma list from: {','.join(STAGES)}")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. --set student.model=Qwen/Qwen3.5-0.8B")
    r.add_argument("--run-dir", help=argparse.SUPPRESS)
    r.set_defaults(fn=cmd_run)

    sub.add_parser("presets", help="list specialty presets").set_defaults(fn=cmd_presets)

    c = sub.add_parser("check-teacher", help="test the teacher endpoint")
    c.add_argument("--config", "-c")
    c.add_argument("--set", action="append", default=[])
    c.set_defaults(fn=cmd_check)

    d = sub.add_parser("doctor", help="check Python, GPU, packages, teacher server and exporters")
    d.add_argument("--config", "-c")
    d.add_argument("--set", action="append", default=[])
    d.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("status", help="print a run's state")
    s.add_argument("run_dir")
    s.set_defaults(fn=cmd_status)

    u = sub.add_parser("ui", help="start the web UI")
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--port", type=int, default=7860)
    u.set_defaults(fn=cmd_ui)

    a = p.parse_args(argv)
    setup_console()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
