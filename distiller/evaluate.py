"""Stage 4: score the distilled student (and optionally the untouched base student) on the preset's held-out benchmark."""
from __future__ import annotations

import time

from .modeling import free_gpu, load_for_inference, render_prompt, text_tokenizer
from .presets import keep_thinking, load_eval_records, load_preset
from .runlog import Run, write_json, write_jsonl
from .verifiers import split_think, verify

STAGE = "eval"


def generate_batch(model, tok, prompts: list[str], max_new_tokens: int) -> tuple[list[str], int, float]:
    import torch

    t = text_tokenizer(tok)
    t.padding_side = "left"
    if t.pad_token_id is None:
        t.pad_token = t.eos_token
    enc = t(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=t.pad_token_id, use_cache=True)
    secs = time.perf_counter() - t0
    new = out[:, enc["input_ids"].shape[1]:]
    texts, n_tokens = [], 0
    for row in new:
        valid = row[row != t.pad_token_id]
        n_tokens += int(valid.numel())
        texts.append(t.decode(valid, skip_special_tokens=True))
    return texts, n_tokens, secs


def evaluate_model(label: str, path: str, records: list[dict], preset: dict, cfg: dict, run: Run, think: bool) -> dict:
    log = lambda m: run.log(m, STAGE)
    model, tok = load_for_inference(path, cfg, log)
    e, f = cfg["eval"], cfg.get("filter") or {}
    bs, mnt = int(e.get("batch_size", 4)), int(e.get("max_new_tokens", 1024))
    system = (preset.get("system_prompt") or "").strip()
    rows, passed, tokens, secs = [], 0, 0, 0.0
    for i in range(0, len(records), bs):
        chunk = records[i:i + bs]
        prompts = [render_prompt(tok, system, r["user"], think) for r in chunk]
        outs, n_tok, s = generate_batch(model, tok, prompts, mnt)
        tokens, secs = tokens + n_tok, secs + s
        for rec, text in zip(chunk, outs):
            _, answer = split_think(text)
            ok, why = verify(rec, answer, execute_code=bool(f.get("execute_code", True)),
                             timeout=float(f.get("code_timeout", 10)))
            passed += ok
            rows.append({"id": rec["id"], "ok": ok, "reason": why, "output": text})
        done = min(i + bs, len(records))
        run.progress(STAGE, done, len(records), f"{label}: {passed}/{done} correct")
        judged = records[0].get("verifier") == "judge"
        log(f"{label}: {done}/{len(records)} answers generated" + ("" if judged else f" - {passed} correct"))
    write_jsonl(run.path("eval", f"{label}_outputs.jsonl"), rows)
    del model, tok
    free_gpu()
    if records and records[0].get("verifier") == "judge":
        return judge_model(label, path, records, rows, tokens, secs, cfg, run)
    return {
        "model": str(path),
        "score": round(passed / max(1, len(records)), 4),
        "correct": passed,
        "total": len(records),
        "avg_output_tokens": round(tokens / max(1, len(records)), 1),
        "batched_tokens_per_second": round(tokens / secs, 1) if secs else None,
    }


def judge_model(label, path, records, rows, tokens, secs, cfg, run) -> dict:
    """Custom specialties: the teacher grades each answer 1-10 against its own reference answer."""
    from .synth import judge_outputs
    from .teacher import unload_teacher

    log = lambda m: run.log(m, STAGE)
    log(f"{label}: teacher is grading {len(rows)} answers")
    items = [{"id": row["id"], "user": rec["user"], "reference": rec["reference"], "candidate": row["output"]}
             for rec, row in zip(records, rows)]
    graded = judge_outputs(cfg, items, log, lambda d, t: run.progress(STAGE, d, t, f"{label}: grading"))
    scores = [g["judge_score"] for g in graded if g["judge_score"] is not None]
    out_rows = []
    for row, g in zip(rows, graded):
        sc = g["judge_score"]
        out_rows.append({**row, "judge_score": sc, "ok": bool(sc and sc >= 7),
                         "reason": f"judge {sc}/10" if sc else "judge failed"})
    write_jsonl(run.path("eval", f"{label}_outputs.jsonl"), out_rows)
    unload_teacher(cfg)  # give the GPU back before the next model loads
    mean = sum(scores) / len(scores) if scores else 0.0
    good = sum(1 for s in scores if s >= 7)
    log(f"{label}: mean judge score {mean:.2f}/10, {good}/{len(rows)} rated 7+")
    return {
        "model": str(path),
        "score": round(mean / 10, 4),
        "mean_judge_score": round(mean, 2),
        "rated_7_plus": good,
        "correct": good,
        "total": len(rows),
        "ungraded": len(rows) - len(scores),
        "avg_output_tokens": round(tokens / max(1, len(rows)), 1),
        "batched_tokens_per_second": round(tokens / secs, 1) if secs else None,
    }


def run_eval(cfg: dict, run: Run) -> dict:
    log = lambda m: run.log(m, STAGE)
    preset = load_preset(cfg["specialty"])
    think = keep_thinking(cfg, preset)
    limit = int(cfg["eval"].get("max_samples", 100))
    if (preset.get("eval") or {}).get("verifier") == "judge":
        from .runlog import read_jsonl

        records = list(read_jsonl(run.dir / "data" / "eval_refs.jsonl"))[:limit]
        if not records:
            raise RuntimeError("no held-out reference answers (data/eval_refs.jsonl) - re-run generate + filter")
        log(f"{len(records)} held-out prompts, graded by the teacher (1-10)")
    else:
        records = load_eval_records(preset, limit, log)
    if not records:
        raise RuntimeError(f"preset '{preset['name']}' has no eval section")
    merged = run.dir / "merged"
    if not (merged / "config.json").exists():
        raise RuntimeError("no merged model found - run the train stage first")
    bench = (preset.get("eval") or {}).get("name", "eval")
    metric = "teacher-judge score (mean/10)" if records[0].get("verifier") == "judge" else "pass@1 (greedy)"
    results = {"benchmark": bench, "metric": metric, "models": {}}
    results["models"]["distilled"] = evaluate_model("distilled", str(merged), records, preset, cfg, run, think)
    if "judge" not in metric:
        log(f"distilled score: {results['models']['distilled']['score']:.1%}")
    if cfg["eval"].get("compare_base", True):
        results["models"]["base"] = evaluate_model("base", cfg["student"]["model"], records, preset, cfg, run, think)
        b, d = results["models"]["base"]["score"], results["models"]["distilled"]["score"]
        results["gain"] = round(d - b, 4)
        if "judge" in metric:
            log(f"untrained student {b * 10:.1f}/10 -> distilled {d * 10:.1f}/10 ({(d - b) * 10:+.1f})")
        else:
            log(f"untrained student {b:.1%} -> distilled {d:.1%} ({(d - b) * 100:+.1f} points)")
    write_json(run.path("eval", "results.json"), results)
    return results
