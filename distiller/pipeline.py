"""Runs the stages in order, recording status in runs/<name>/state.json."""
from __future__ import annotations

import importlib
import traceback

from .config import save_config
from .runlog import STAGES, Run

STAGE_FUNCS = {
    "generate": ("distiller.generate", "run_generate"),
    "filter": ("distiller.filter", "run_filter"),
    "train": ("distiller.train", "run_train"),
    "eval": ("distiller.evaluate", "run_eval"),
    "export": ("distiller.export", "run_export"),
    "benchmark": ("distiller.benchmark", "run_benchmark"),
}


def parse_stages(text: str | list | None) -> list[str]:
    if not text or text == "all":
        return list(STAGES)
    items = text if isinstance(text, list) else [s.strip() for s in str(text).split(",")]
    bad = [s for s in items if s not in STAGES]
    if bad:
        raise ValueError(f"unknown stage(s) {bad}; valid: {', '.join(STAGES)}")
    return [s for s in STAGES if s in items]  # always canonical order


def run_pipeline(cfg: dict, run: Run, stages: list[str]) -> int:
    save_config(cfg, run.dir / "config.yaml")
    run.log(f"=== run '{run.dir.name}': specialty={cfg['specialty']} teacher={cfg['teacher']['model']} "
            f"student={cfg['student']['model']} stages={','.join(stages)} ===")
    for stage in stages:
        if stage in ("train", "eval", "benchmark"):  # these load the student onto the GPU
            from .teacher import unload_teacher

            if unload_teacher(cfg):
                run.log("asked Ollama to unload the teacher to free VRAM", stage)
        run.set_stage(stage, "running")
        mod_name, fn_name = STAGE_FUNCS[stage]
        try:
            fn = getattr(importlib.import_module(mod_name), fn_name)
            result = fn(cfg, run)
            run.set_stage(stage, "done", result=result if isinstance(result, dict) else None)
        except KeyboardInterrupt:
            run.set_stage(stage, "stopped")
            run.log("stopped by user", stage)
            return 130
        except Exception as e:
            run.set_stage(stage, "failed", error=f"{type(e).__name__}: {e}")
            run.log(f"FAILED: {type(e).__name__}: {e}", stage)
            tb = traceback.format_exc().rstrip()
            (run.dir / f"error-{stage}.txt").write_text(tb, encoding="utf-8")  # full chained traceback
            lines = tb.splitlines()
            # show the root cause too (the first exception in a chain), not just the last frames
            keep = lines[:12] + ["  ..."] + lines[-14:] if len(lines) > 30 else lines
            for line in keep:
                run.log("  " + line, stage)
            if stage == "export" and "benchmark" in stages:
                # the trained model is already saved - a failed GGUF export shouldn't block the speed test
                run.log("continuing: the benchmark will time the trained model directly", stage)
                continue
            return 1
    run.log("=== all requested stages finished ===")
    return 0
