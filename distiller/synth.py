"""Custom specialties: the teacher writes the training prompts itself (self-instruct style),
and later grades the student against held-out reference answers (LLM-as-judge)."""
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .runlog import Run, read_json, read_jsonl, write_json
from .teacher import Teacher, TeacherError
from .verifiers import split_think

DIFFICULTY = ["easy", "medium", "hard", "medium", "hard", "expert-level"]


def parse_json_list(text: str) -> list:
    """Pull a JSON array out of a model reply; fall back to numbered / bulleted lines."""
    _, text = split_think(text or "")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    a, b = text.find("["), text.rfind("]")
    if a != -1 and b > a:
        try:
            data = json.loads(text[a:b + 1])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    items = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:\d+[.)]|[-*•])\s+(.+)$", line)
        if m:
            items.append(m.group(1).strip().strip('"'))
    return items


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()[:120]


def get_subtopics(teacher: Teacher, preset: dict, run: Run, log: Callable[[str], None]) -> list[str]:
    path = run.path("data", "subtopics.json")
    cached = read_json(path)
    if cached:
        return cached
    msg = (f"Specialty: {preset['title']}\nDescription: {preset.get('description', '')}\n\n"
           "List 24 distinct sub-topics or task types that together cover this specialty well, from basic to advanced. "
           "Return ONLY a JSON array of short strings.")
    topics: list[str] = []
    for _ in range(3):
        try:
            res = teacher.chat([{"role": "user", "content": msg}], temperature=0.7, max_tokens=4096, think=False)
            topics = [str(t).strip() for t in parse_json_list(res["content"]) if str(t).strip()]
        except TeacherError as e:
            log(f"sub-topic request failed: {e}")
        if len(topics) >= 4:
            break
    if len(topics) < 4:
        topics = ["core concepts", "practical tasks", "debugging and troubleshooting", "best practices",
                  "edge cases", "explanations for beginners", "advanced techniques", "real-world scenarios"]
        log("teacher did not return usable sub-topics - using generic ones")
    topics = topics[:40]
    write_json(path, topics)
    log(f"{len(topics)} sub-topics: {', '.join(topics[:8])}{' ...' if len(topics) > 8 else ''}")
    return topics


def synthesize_prompts(cfg: dict, run: Run, preset: dict, log: Callable[[str], None]) -> list[dict]:
    """Returns [{"id", "prompt", "topic"}]; resumable via data/synthetic_prompts.jsonl."""
    syn = preset["synthetic"]
    target = int(syn.get("num_prompts", 500))
    path = run.path("data", "synthetic_prompts.jsonl")
    have = list(read_jsonl(path))
    if len(have) >= target:
        log(f"reusing {target} teacher-written prompts")
        return have[:target]

    teacher = Teacher(cfg)
    try:
        topics = get_subtopics(teacher, preset, run, log)
        examples = syn.get("examples") or []
        batch = int(syn.get("batch", 8))
        seen = {_key(r["prompt"]) for r in have}
        lock = threading.Lock()
        counter = {"calls": len(have) // max(1, batch), "fails": 0}
        log(f"teacher is writing {target - len(have)} more prompts ({batch} per request)")

        def one(call_no: int) -> list[str]:
            topic = topics[call_no % len(topics)]
            level = DIFFICULTY[(call_no // len(topics)) % len(DIFFICULTY)]
            ex = ""
            if examples:
                ex = "Examples of the kind of request we want (do NOT copy them):\n" + "\n".join(
                    f"- {examples[(call_no + i) % len(examples)]}" for i in range(min(3, len(examples)))) + "\n\n"
            msg = (f"You are creating training data for an AI assistant that specialises in: {preset['title']}.\n"
                   f"Specialty description: {preset.get('description', '')}\n\n{ex}"
                   f"Write {batch} new, realistic, self-contained requests a user might send, all about the sub-topic "
                   f"\"{topic}\" at {level} difficulty. Make them varied in wording, length and format; include every detail "
                   "needed to answer (data, code, constraints). Do not answer them.\n"
                   "Return ONLY a JSON array of strings.")
            try:
                res = teacher.chat([{"role": "user", "content": msg}], temperature=1.0, max_tokens=4096, think=False)
            except TeacherError as e:
                with lock:
                    counter["fails"] += 1
                log(f"prompt-writing request failed: {e}")
                return []
            return [str(x).strip() for x in parse_json_list(res["content"]) if isinstance(x, (str, int, float))]

        max_calls = counter["calls"] + max(10, 3 * (target - len(have)) // batch)
        workers = max(1, int(cfg["teacher"].get("concurrency", 2)))
        with open(path, "a", encoding="utf-8") as f:
            call = counter["calls"]
            while len(have) < target and call < max_calls:
                calls = list(range(call, call + workers))
                call += workers
                with ThreadPoolExecutor(workers) as pool:
                    results = list(pool.map(one, calls))
                for call_no, items in zip(calls, results):
                    for text in items:
                        k = _key(text)
                        if len(text) < 15 or k in seen or len(have) >= target:
                            continue
                        seen.add(k)
                        rec = {"id": f"syn-{len(have)}", "prompt": text, "topic": topics[call_no % len(topics)]}
                        have.append(rec)
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                run.progress("generate", len(have), target, "teacher writing prompts")
                if counter["fails"] >= 10 and len(have) == 0:
                    raise RuntimeError("the teacher failed 10 prompt-writing requests - is it running?")
        log(f"{len(have)} unique prompts written by the teacher")
    finally:
        teacher.close()
    return have


# ----------------------------------------------------------------------------- judge
JUDGE_TEMPLATE = """You are grading an AI assistant's answer.

<task>
{task}
</task>

<reference_answer>
{reference}
</reference_answer>

<candidate_answer>
{candidate}
</candidate_answer>

Judge the candidate on correctness, completeness and usefulness for the task. Use the reference as a guide
(it was written by a strong model but may be imperfect). Penalise wrong facts, broken code and missing parts;
do not reward length for its own sake.
Finish with exactly one line in this form: Score: <integer from 1 to 10>"""


def parse_score(text: str) -> int | None:
    _, ans = split_think(text or "")
    hits = re.findall(r"score\s*[:=]\s*\**\s*(\d{1,2})", ans or text, flags=re.IGNORECASE)
    if not hits:
        hits = re.findall(r"\b(10|[1-9])\s*/\s*10\b", ans or text)
    if not hits:
        return None
    v = int(hits[-1])
    return v if 1 <= v <= 10 else None


def judge_outputs(cfg: dict, items: list[dict], log: Callable[[str], None],
                  progress: Callable[[int, int], None] | None = None) -> list[dict]:
    """items: [{"user", "reference", "candidate"}] -> adds "judge_score" (1-10 or None)."""
    teacher = Teacher(cfg)
    max_toks = int((cfg.get("eval") or {}).get("judge_max_tokens", 2048))
    done = [0]
    lock = threading.Lock()

    def one(it: dict) -> dict:
        _, cand = split_think(it["candidate"])
        msg = JUDGE_TEMPLATE.format(task=it["user"], reference=it["reference"][:8000], candidate=(cand or it["candidate"])[:8000])
        score = None
        for _ in range(2):
            try:
                res = teacher.chat([{"role": "user", "content": msg}], temperature=0, max_tokens=max_toks, think=False)
                score = parse_score(res["content"])
            except TeacherError as e:
                log(f"judge request failed: {e}")
            if score is not None:
                break
        with lock:
            done[0] += 1
            if progress:
                progress(done[0], len(items))
        return {**it, "judge_score": score}

    try:
        with ThreadPoolExecutor(max(1, int(cfg["teacher"].get("concurrency", 2)))) as pool:
            return list(pool.map(one, items))
    finally:
        teacher.close()


def preview_prompts(cfg: dict, title: str, description: str, examples: list[str], n: int = 5) -> list[str]:
    """One quick teacher call so the user can sanity-check a custom specialty before a long run."""
    teacher = Teacher(cfg)
    teacher.retries = 0
    try:
        ex = ("Examples of the kind of request we want (do NOT copy them):\n" + "\n".join(f"- {e}" for e in examples[:3]) + "\n\n") if examples else ""
        msg = (f"You are creating training data for an AI assistant that specialises in: {title}.\n"
               f"Specialty description: {description}\n\n{ex}"
               f"Write {n} varied, realistic, self-contained requests a user might send (mixed difficulty). Do not answer them.\n"
               "Return ONLY a JSON array of strings.")
        res = teacher.chat([{"role": "user", "content": msg}], temperature=0.9, max_tokens=4096, think=False)
        return [str(x).strip() for x in parse_json_list(res["content"])][:n]
    finally:
        teacher.close()
