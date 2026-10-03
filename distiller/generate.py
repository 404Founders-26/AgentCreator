"""Stage 1: ask the teacher to answer every specialty prompt. Resumable."""
from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .presets import keep_thinking, load_preset, load_train_records
from .runlog import Run, read_jsonl
from .teacher import Teacher, TeacherError
from .config import to_json

STAGE = "generate"


def fmt_eta(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60} min"
    h, m = divmod(seconds // 60, 60)
    return f"{h} h {m:02d} min" if h < 48 else f"{h / 24:.1f} days"


def build_messages(preset: dict, user: str) -> list[dict]:
    msgs = []
    if preset.get("system_prompt"):
        msgs.append({"role": "system", "content": preset["system_prompt"].strip()})
    msgs.append({"role": "user", "content": user})
    return msgs


def select_records(cfg: dict, records: list[dict], holdout: int = 0) -> tuple[list[dict], list[dict]]:
    """Shuffle deterministically, set aside `holdout` prompts for evaluation, then apply data.max_prompts."""
    rng = random.Random(cfg.get("seed", 42))
    records = list(records)
    rng.shuffle(records)
    held, records = records[:holdout], records[holdout:]
    limit = (cfg.get("data") or {}).get("max_prompts")
    return (records[: int(limit)] if limit else records), held


def synthetic_records(cfg: dict, run: Run, preset: dict, log) -> list[dict]:
    from .synth import synthesize_prompts

    syn = dict(preset["synthetic"])
    limit = (cfg.get("data") or {}).get("max_prompts")
    if limit:  # quick runs: don't make the teacher write more prompts than will be used
        syn["num_prompts"] = min(int(syn.get("num_prompts", 500)), int(limit) + int((preset.get("eval") or {}).get("holdout", 60)))
    verifier = syn.get("verifier", "none")
    return [{"id": r["id"], "source": "synthetic", "verifier": verifier, "user": r["prompt"],
             "tests": [], "setup": "", "answer": None}
            for r in synthesize_prompts(cfg, run, {**preset, "synthetic": syn}, log)]


def run_generate(cfg: dict, run: Run) -> dict:
    log = lambda m: run.log(m, STAGE)
    preset = load_preset(cfg["specialty"])
    all_records = load_train_records(preset, log)
    if preset.get("synthetic"):
        all_records = synthetic_records(cfg, run, preset, log) + all_records
    if not all_records:
        raise RuntimeError("no prompts loaded - check the preset sources (internet access for Hugging Face datasets?)")
    holdout = 0
    ev = preset.get("eval") or {}
    if ev.get("verifier") == "judge":  # custom specialty: keep some prompts back to grade the student on
        holdout = min(int(ev.get("holdout", 60)), max(5, len(all_records) // 10))
    records, held = select_records(cfg, all_records, holdout)
    if held:
        log(f"holding back {len(held)} prompts as the evaluation set (teacher answers become the reference)")

    spp = max(1, int(cfg["data"].get("samples_per_prompt", 1)))
    out_path = run.path("data", "raw.jsonl")
    done = {(r["id"], r.get("k", 0)) for r in read_jsonl(out_path) if not r.get("error")}
    tasks = [(rec, k, "train") for rec in records for k in range(spp)] + [(rec, 0, "eval") for rec in held]
    todo = [t for t in tasks if (t[0]["id"], t[1]) not in done]
    total = len(tasks)
    log(f"{len(records)} prompts x {spp} samples + {len(held)} eval prompts = {total} requests; "
        f"{total - len(todo)} already done, {len(todo)} to go")
    log(f"teacher: {cfg['teacher']['model']} @ {cfg['teacher']['base_url']}")
    if not todo:
        return {"requests": total, "new": 0}

    teacher = Teacher(cfg)
    lock = threading.Lock()
    counters = {"ok": 0, "err": 0, "tokens": 0}
    # only ask the teacher to "think" when we keep the thinking as training data - otherwise it just
    # burns tokens (with Qwen3.5 thinking on, roughly half of every answer is hidden reasoning)
    think = keep_thinking(cfg, preset) if cfg["teacher"].get("think") is None else bool(cfg["teacher"]["think"])
    log(f"teacher thinking: {'on' if think else 'off'} | parallel requests: {cfg['teacher'].get('concurrency', 2)}")
    t_start = time.time()
    # higher temperature for extra samples so rejection sampling sees variety
    base_temp = teacher.temperature

    def work(rec: dict, k: int, split: str) -> dict:
        temp = base_temp if k == 0 else min(1.0, base_temp + 0.1 * k)
        row = {
            "id": rec["id"], "k": k, "split": split, "source": rec["source"], "verifier": rec["verifier"],
            "system": preset.get("system_prompt", "").strip(), "user": rec["user"],
            "tests": rec["tests"], "setup": rec["setup"], "answer": rec["answer"],
        }
        try:
            res = teacher.chat(build_messages(preset, rec["user"]), temperature=temp, think=think)
            row.update(res)
        except TeacherError as e:
            row["error"] = str(e)
        return row

    workers = max(1, int(cfg["teacher"].get("concurrency", 2)))
    processed = 0
    pool = ThreadPoolExecutor(workers)
    clean_exit = False
    try:
        with open(out_path, "a", encoding="utf-8") as fout:
            futures = [pool.submit(work, rec, k, split) for rec, k, split in todo]
            for fut in as_completed(futures):
                row = fut.result()
                with lock:
                    fout.write(to_json(row) + "\n")
                    fout.flush()
                    processed += 1
                    if row.get("error"):
                        counters["err"] += 1
                        if counters["err"] <= 5:
                            log(f"error on {row['id']}: {row['error'][:200]}")
                        if counters["err"] >= 20 and counters["ok"] == 0:
                            raise RuntimeError("20 failed requests and no successes - is the teacher server running?")
                    else:
                        counters["ok"] += 1
                        counters["tokens"] += int((row.get("usage") or {}).get("completion_tokens") or 0)
                    elapsed = time.time() - t_start
                    rate = processed / elapsed if elapsed else 0  # answers per second this session
                    eta = fmt_eta((len(todo) - processed) / rate) if rate else "?"
                    if processed <= 3 or processed % 5 == 0 or processed == len(todo):
                        run.progress(STAGE, total - len(todo) + processed, total,
                                     f"{counters['ok']} ok, {counters['err']} err - {rate * 60:.1f}/min - {eta} left")
                    if processed == 1 and not row.get("error"):
                        place = teacher.gpu_placement()
                        if place and place.get("gpu_fraction") is not None:
                            log(f"teacher in VRAM: {place['vram_gb']}/{place['size_gb']} GB ({place['gpu_fraction']:.0%})")
                            if place["gpu_fraction"] < 0.97:
                                log(f"WARNING: {1 - place['gpu_fraction']:.0%} of the teacher runs on the CPU, which makes generation "
                                    "several times slower. Pick a smaller teacher (qwen3.5:4b fits fully in 8 GB) or lower "
                                    "teacher.num_ctx / parallel requests.")
                    if processed in (1, 5, 20) or processed % 50 == 0:
                        tps = row.get("gen_tokens_per_second")
                        log(f"{processed}/{len(todo)} done ({counters['ok']} ok, {counters['err']} errors) - "
                            f"{rate * 60:.1f} answers/min" + (f", {tps} tokens/s" if tps else "") + f" - about {eta} left")
        clean_exit = True
    finally:
        # on error / Ctrl+C drop queued requests instead of waiting for all of them
        pool.shutdown(wait=clean_exit, cancel_futures=not clean_exit)
        teacher.close()
    log(f"finished: {counters['ok']} answers, {counters['err']} errors, {counters['tokens']} completion tokens")
    if counters["ok"] == 0 and not done:
        raise RuntimeError("the teacher produced no answers")
    return {"requests": total, "new": counters["ok"], "errors": counters["err"]}
