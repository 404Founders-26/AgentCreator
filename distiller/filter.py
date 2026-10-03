"""Stage 2: keep only verified, well-formed teacher answers and write the training set."""
from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from .presets import keep_thinking, load_preset
from .runlog import Run, read_jsonl, write_json, write_jsonl
from .verifiers import verify

STAGE = "filter"


def build_target(row: dict, think: bool) -> str:
    content = (row.get("content") or "").strip()
    reasoning = (row.get("reasoning") or "").strip()
    if think and reasoning:
        return f"<think>\n{reasoning}\n</think>\n\n{content}"
    return content


def check_row(row: dict, cfg: dict) -> tuple[bool, str]:
    if row.get("error"):
        return False, "request_error"
    content = (row.get("content") or "").strip()
    if not content:
        return False, "empty_or_truncated"
    if row.get("finish_reason") == "length":
        return False, "truncated"
    d = cfg["data"]
    if len(content) < int(d.get("min_chars", 20)):
        return False, "too_short"
    if len(content) + len(row.get("reasoning") or "") > int(d.get("max_chars", 12000)):
        return False, "too_long"
    f = cfg.get("filter") or {}
    return verify(row, content, execute_code=bool(f.get("execute_code", True)), timeout=float(f.get("code_timeout", 10)))


def run_filter(cfg: dict, run: Run) -> dict:
    log = lambda m: run.log(m, STAGE)
    preset = load_preset(cfg["specialty"])
    think = keep_thinking(cfg, preset)
    all_rows = list(read_jsonl(run.path("data", "raw.jsonl")))
    if not all_rows:
        raise RuntimeError("data/raw.jsonl is empty - run the generate stage first")
    # held-out evaluation prompts (custom specialties): never trained on; teacher answer = reference
    refs = {}
    for r in all_rows:
        if r.get("split") == "eval" and not r.get("error") and (r.get("content") or "").strip() \
                and r.get("finish_reason") != "length":
            refs[r["id"]] = {"id": r["id"], "source": r["source"], "verifier": "judge", "user": r["user"],
                             "reference": r["content"].strip(), "tests": [], "setup": "", "answer": None}
    if refs:
        write_jsonl(run.path("data", "eval_refs.jsonl"), refs.values())
        log(f"{len(refs)} held-out prompts with teacher reference answers saved for evaluation")
    rows = [r for r in all_rows if r.get("split") != "eval"]
    log(f"checking {len(rows)} teacher answers (keep_thinking={think})")

    workers = max(1, int((cfg.get("filter") or {}).get("workers", 4)))
    results: list[tuple[bool, str]] = [None] * len(rows)  # type: ignore

    def job(i: int):
        results[i] = check_row(rows[i], cfg)
        return i

    with ThreadPoolExecutor(workers) as pool:
        for n, _ in enumerate(pool.map(job, range(len(rows))), 1):
            if n % 100 == 0 or n == len(rows):
                run.progress(STAGE, n, len(rows))

    reasons = Counter(r for _, r in results)
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    seen_hashes = set()
    dupes = 0
    for row, (ok, _) in zip(rows, results):
        if not ok:
            continue
        target = build_target(row, think)
        h = hashlib.sha1((row["user"] + "\x00" + " ".join(target.split())).encode()).hexdigest()
        if h in seen_hashes:
            dupes += 1
            continue
        seen_hashes.add(h)
        by_prompt[row["id"]].append({
            "id": row["id"], "source": row["source"], "system": row.get("system", ""),
            "user": row["user"], "assistant": target,
        })

    keep_n = max(1, int(cfg["data"].get("keep_per_prompt", 1)))
    kept: list[dict] = []
    for pid, items in by_prompt.items():
        items.sort(key=lambda x: len(x["assistant"]))  # prefer concise correct answers
        kept.extend(items[:keep_n])

    # split by prompt id so a prompt never appears in both train and validation
    rng = random.Random(cfg.get("seed", 42))
    ids = sorted(by_prompt)
    rng.shuffle(ids)
    n_val = int(len(ids) * float(cfg["data"].get("eval_fraction", 0.03)))
    val_ids = set(ids[:n_val]) if len(ids) >= 50 else set()
    train = [r for r in kept if r["id"] not in val_ids]
    valid = [r for r in kept if r["id"] in val_ids]
    rng.shuffle(train)

    write_jsonl(run.path("data", "train.jsonl"), train)
    write_jsonl(run.path("data", "valid.jsonl"), valid)
    per_source = Counter(r["source"] for r in kept)
    passed = sum(1 for ok, _ in results if ok)
    stats = {
        "raw": len(rows),
        "passed_checks": passed,
        "pass_rate": round(passed / len(rows), 3),
        "duplicates_removed": dupes,
        "prompts_with_answer": len(by_prompt),
        "train_examples": len(train),
        "valid_examples": len(valid),
        "reasons": dict(reasons.most_common()),
        "per_source": dict(per_source),
        "keep_thinking": think,
        "eval_references": len(refs),
    }
    write_json(run.path("data", "stats.json"), stats)
    log(f"kept {len(kept)} examples ({len(train)} train / {len(valid)} valid); pass rate {stats['pass_rate']:.1%}")
    log("reasons: " + ", ".join(f"{k}={v}" for k, v in reasons.most_common()))
    if not train:
        raise RuntimeError("no training examples survived filtering")
    return stats
