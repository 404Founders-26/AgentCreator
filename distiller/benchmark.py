"""Stage 6: measure generation speed of the exported student (and the teacher, for comparison)."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from .runlog import Run, read_json, write_json
from .teacher import Teacher

STAGE = "benchmark"


def _llama_bench_bin(cfg: dict) -> str | None:
    from .export import find_llama_binary

    return find_llama_binary((cfg.get("export") or {}).get("llama_cpp_dir"), "llama-bench")


def bench_llama_bench(binary: str, gguf: str, n: int, log) -> dict:
    cmd = [binary, "-m", gguf, "-p", "512", "-n", str(n), "-ngl", "99", "-o", "json"]
    log("$ " + " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-500:])
    rows = json.loads(out.stdout)
    res = {"method": "llama-bench", "gpu_layers": 99}
    for r in rows:
        if r.get("n_gen"):
            res["generate_tokens_per_second"] = round(r["avg_ts"], 1)
        elif r.get("n_prompt"):
            res["prompt_tokens_per_second"] = round(r["avg_ts"], 1)
    return res


def bench_llama_cpp_python(gguf: str, prompt: str, n: int, runs: int, log) -> dict:
    from llama_cpp import Llama

    llm = Llama(model_path=gguf, n_gpu_layers=-1, n_ctx=2048, verbose=False)
    speeds = []
    for i in range(runs + 1):
        t0 = time.perf_counter()
        out = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=n, temperature=0)
        secs = time.perf_counter() - t0
        toks = out["usage"]["completion_tokens"]
        if i > 0:  # first run is warm-up
            speeds.append(toks / secs)
    return {"method": "llama-cpp-python", "generate_tokens_per_second": round(sum(speeds) / len(speeds), 1)}


def bench_transformers(path: str, prompt: str, n: int, runs: int, cfg: dict, log) -> dict:
    import torch

    from .modeling import free_gpu, load_for_inference, render_prompt, text_tokenizer

    model, tok = load_for_inference(path, cfg, log)
    t = text_tokenizer(tok)
    enc = t(render_prompt(tok, "", prompt, False), return_tensors="pt", add_special_tokens=False).to(model.device)
    speeds = []
    for i in range(runs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=n, min_new_tokens=n, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        if i > 0:
            speeds.append((out.shape[1] - enc["input_ids"].shape[1]) / secs)
    del model, tok
    free_gpu()
    return {"method": "transformers (16-bit, unquantized)", "generate_tokens_per_second": round(sum(speeds) / len(speeds), 1)}


def bench_teacher(cfg: dict, prompt: str, n: int, runs: int, model: str | None = None) -> dict:
    """Times a model behind the teacher's API (the teacher itself, or the student once imported into Ollama)."""
    if model:
        cfg = {**cfg, "teacher": {**cfg["teacher"], "model": model, "extra_body": {}}}
    t = Teacher(cfg)
    speeds = []
    try:
        for i in range(runs + 1):
            r = t.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=n, think=False)
            toks = (r["usage"] or {}).get("completion_tokens")
            if i > 0 and r.get("gen_tokens_per_second"):
                speeds.append(r["gen_tokens_per_second"])  # pure generation speed (Ollama reports it)
            elif i > 0 and toks and r["seconds"]:
                speeds.append(toks / r["seconds"])
    finally:
        t.close()
    if not speeds:
        return {"error": "teacher did not report token usage"}
    # includes HTTP + prompt processing, so it slightly understates pure generation speed
    return {"method": "API round-trip", "model": cfg["teacher"]["model"],
            "generate_tokens_per_second": round(sum(speeds) / len(speeds), 1)}


def run_benchmark(cfg: dict, run: Run) -> dict:
    log = lambda m: run.log(m, STAGE)
    b = cfg.get("benchmark") or {}
    prompt, n, runs = b.get("prompt", "Hello"), int(b.get("max_new_tokens", 256)), max(1, int(b.get("runs", 3)))
    exp = read_json(run.dir / "export" / "export.json", {}) or {}
    gguf = exp.get("gguf")
    results: dict = {"prompt": prompt, "max_new_tokens": n}

    student: dict | None = None
    errors = []
    # 1st choice: the student inside Ollama - same engine and GPU as the teacher, so the comparison is fair
    if exp.get("ollama_model"):
        try:
            student = bench_teacher(cfg, prompt, n, runs, model=exp["ollama_model"])
            student["method"] = f"Ollama ({exp.get('quantization')}) API round-trip"
            if "generate_tokens_per_second" not in student:
                errors.append(f"ollama: {student.get('error')}")
                student = None
        except Exception as e:
            errors.append(f"ollama: {e}")
    if student is None and gguf and Path(gguf).exists():
        binary = _llama_bench_bin(cfg)
        if binary:
            try:
                student = bench_llama_bench(binary, gguf, n, log)
            except Exception as e:
                errors.append(f"llama-bench: {e}")
        if student is None:
            try:
                student = bench_llama_cpp_python(gguf, prompt, n, runs, log)
            except Exception as e:
                errors.append(f"llama-cpp-python: {e}")
        if student:
            student["file"] = Path(gguf).name
    if student is None and exp.get("ollama_model"):
        try:
            student = bench_teacher(cfg, prompt, n, runs, model=exp["ollama_model"])
            student["method"] = f"Ollama ({exp.get('quantization')}) API round-trip"
            if "generate_tokens_per_second" not in student:
                errors.append(f"ollama: {student.get('error')}")
                student = None
        except Exception as e:
            errors.append(f"ollama: {e}")
    if student is None and (run.dir / "merged" / "config.json").exists():
        log("no GGUF runner available - timing the merged model with transformers instead")
        student = bench_transformers(str(run.dir / "merged"), prompt, n, runs, cfg, log)
    if student is None:
        raise RuntimeError("nothing to benchmark - run train/export first. " + "; ".join(errors))
    results["student"] = student
    log(f"student: {student['generate_tokens_per_second']} tokens/s ({student['method']})")

    if b.get("include_teacher", True):
        try:
            results["teacher"] = bench_teacher(cfg, prompt, n, runs)
            if "generate_tokens_per_second" in results["teacher"]:
                tps = results["teacher"]["generate_tokens_per_second"]
                results["speedup"] = round(student["generate_tokens_per_second"] / tps, 2) if tps else None
                log(f"teacher: {tps} tokens/s -> student is {results['speedup']}x faster")
        except Exception as e:
            results["teacher"] = {"error": str(e)}
            log(f"teacher benchmark skipped: {e}")
    if errors:
        results["notes"] = errors
    write_json(run.path("benchmark.json"), results)
    return results
