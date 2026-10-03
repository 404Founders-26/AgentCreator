"""Specialty presets and the prompt sources they point to."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

import yaml

from .config import ROOT

PRESET_DIR = ROOT / "presets"
CUSTOM_SUBDIR = "custom"  # user-made specialties live in presets/custom/


def _preset_files() -> list[Path]:
    return sorted(PRESET_DIR.glob("*.yaml")) + sorted((PRESET_DIR / CUSTOM_SUBDIR).glob("*.yaml"))


def list_presets() -> list[dict]:
    out = []
    for p in _preset_files():
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        srcs = [s.get("name", s.get("path")) for s in data.get("sources", [])]
        if data.get("synthetic"):
            srcs.insert(0, f"teacher-written prompts ({data['synthetic'].get('num_prompts', '?')})")
        out.append({
            "name": data.get("name", p.stem),
            "title": data.get("title", p.stem),
            "description": (data.get("description") or "").strip(),
            "keep_thinking": bool(data.get("keep_thinking", False)),
            "custom": bool(data.get("custom", False)),
            "sources": srcs,
            "eval": (data.get("eval") or {}).get("name"),
        })
    return out


def preset_path(name: str) -> Path | None:
    for cand in (PRESET_DIR / f"{name}.yaml", PRESET_DIR / CUSTOM_SUBDIR / f"{name}.yaml"):
        if cand.exists():
            return cand
    return None


def load_preset(name: str) -> dict:
    path = preset_path(name)
    if path is None:
        known = ", ".join(p["name"] for p in list_presets())
        raise FileNotFoundError(f"Preset '{name}' not found in {PRESET_DIR} (available: {known})")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("name", name)
    data.setdefault("system_prompt", "")
    data.setdefault("sources", [])
    return data


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug[:48] or "custom"


def save_custom_preset(spec: dict) -> dict:
    """Create presets/custom/<slug>.yaml from the UI's "custom specialty" form.

    spec keys: title, description, examples [str], answer_check (none|has_code), num_prompts,
               keep_thinking, system_prompt?, hf_dataset {path, config?, split?, prompt_field?, limit?}?,
               upload (path to a jsonl already saved by the server)?, eval_holdout
    """
    title = (spec.get("title") or "").strip()
    desc = (spec.get("description") or "").strip()
    if not title or len(desc) < 10:
        raise ValueError("a custom specialty needs a name and a description of at least a few words")
    name = slugify(spec.get("name") or title)
    if (PRESET_DIR / f"{name}.yaml").exists():
        name = f"{name}-custom"  # never shadow a built-in preset
    check = spec.get("answer_check", "none")
    if check not in ("none", "has_code"):
        raise ValueError("answer_check must be 'none' or 'has_code'")
    examples = [e.strip() for e in (spec.get("examples") or []) if str(e).strip()][:20]
    system = (spec.get("system_prompt") or "").strip() or (
        f"You are a world-class expert in {title}. {desc}\n"
        "Give accurate, complete and well-structured answers. Show your reasoning briefly where it helps"
        + (", and put any code in a fenced code block." if check == "has_code" else ".")
    )
    sources: list[dict] = []
    if spec.get("upload"):
        sources.append({"name": "uploaded", "type": "jsonl", "path": str(spec["upload"]), "verifier": check})
    hf = spec.get("hf_dataset") or {}
    if hf.get("path"):
        src = {"name": hf["path"].split("/")[-1], "type": "hf", "path": hf["path"], "split": hf.get("split") or "train",
               "prompt_field": hf.get("prompt_field") or "prompt", "verifier": check, "limit": int(hf.get("limit") or 3000)}
        if hf.get("config"):
            src["config"] = hf["config"]
        sources.append(src)
    num = int(spec.get("num_prompts") or 0)
    if num <= 0 and not sources:
        raise ValueError("give the teacher a number of prompts to write, or add a dataset / prompt file")
    preset = {
        "name": name,
        "title": title,
        "custom": True,
        "description": desc,
        "keep_thinking": bool(spec.get("keep_thinking", False)),
        "system_prompt": system,
        "sources": sources,
        "eval": {"name": f"{name}-judge", "verifier": "judge", "holdout": int(spec.get("eval_holdout") or 60)},
        "config_overrides": {},
    }
    if num > 0:
        preset["synthetic"] = {"num_prompts": num, "examples": examples, "verifier": check, "batch": 8}
    if preset["keep_thinking"]:
        preset["config_overrides"] = {"teacher": {"max_tokens": 6144}, "student": {"max_seq_length": 4096},
                                      "data": {"max_chars": 24000}, "training": {"batch_size": 1, "grad_accum": 16},
                                      "eval": {"max_new_tokens": 3072, "batch_size": 2}}
    out = PRESET_DIR / CUSTOM_SUBDIR / f"{name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(preset, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return preset


def delete_custom_preset(name: str) -> bool:
    p = PRESET_DIR / CUSTOM_SUBDIR / f"{slugify(name)}.yaml"
    if p.exists():
        p.unlink()
        return True
    return False


# --------------------------------------------------------------------------- records
# A normalized record looks like:
#   {"id", "source", "verifier", "user": <rendered user message>,
#    "tests": [...], "setup": str, "answer": str|None}

def _render(template: str, values: dict) -> str:
    text = (template or "{prompt}").format_map(defaultdict(str, values))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_answer(raw, regex: str | None):
    if raw is None:
        return None
    raw = str(raw)
    if regex:
        m = re.search(regex, raw, flags=re.MULTILINE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return raw.strip()


PROMPT_FALLBACKS = ("prompt", "instruction", "question", "query", "input", "text", "problem", "task")


def normalize_row(row: dict, spec: dict, idx: int) -> dict | None:
    prompt = row.get(spec.get("prompt_field", "prompt"))
    if prompt is None and "prompt_field" not in spec:
        prompt = next((row[k] for k in PROMPT_FALLBACKS if row.get(k)), None)
    if not prompt or not str(prompt).strip():
        return None
    tests = row.get(spec.get("tests_field", "tests")) or []
    if isinstance(tests, str):
        tests = [tests]
    tests = [str(t) for t in tests if str(t).strip()]
    setup = row.get(spec.get("setup_field", "setup")) or ""
    extra_input = row.get(spec.get("input_field", "input")) or ""
    answer = _extract_answer(row.get(spec.get("answer_field", "answer")), spec.get("answer_regex"))
    src = spec.get("name") or spec.get("path", "src")
    user = _render(spec.get("template", "{prompt}"), {
        "prompt": str(prompt).strip(),
        "tests": "\n".join(tests),
        "input": str(extra_input).strip(),
    })
    return {
        "id": f"{src}-{row.get('id', row.get('task_id', idx))}",
        "source": src,
        "verifier": spec.get("verifier", "none"),
        "user": user,
        "tests": tests,
        "setup": str(setup),
        "answer": answer,
    }


def _iter_hf(spec: dict):
    from datasets import load_dataset  # heavy import, only when needed

    splits = spec.get("splits") or [spec.get("split", "train")]
    for split in splits:
        kwargs = {"split": split}
        if spec.get("config"):
            ds = load_dataset(spec["path"], spec["config"], **kwargs)
        else:
            ds = load_dataset(spec["path"], **kwargs)
        for row in ds:
            yield dict(row)


def _iter_jsonl(spec: dict):
    path = Path(spec["path"])
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_source(spec: dict, log: Callable[[str], None] = print) -> list[dict]:
    kind = spec.get("type", "hf")
    it = _iter_hf(spec) if kind == "hf" else _iter_jsonl(spec)
    limit = spec.get("limit")
    out = []
    for i, row in enumerate(it):
        rec = normalize_row(row, spec, i)
        if rec:
            out.append(rec)
        if limit and len(out) >= limit:
            break
    # make ids unique even if the dataset repeats them
    seen: dict[str, int] = {}
    for r in out:
        n = seen.get(r["id"], 0)
        seen[r["id"]] = n + 1
        if n:
            r["id"] = f"{r['id']}~{n}"
    log(f"loaded {len(out)} prompts from {spec.get('name', spec.get('path'))}")
    return out


def load_train_records(preset: dict, log: Callable[[str], None] = print) -> list[dict]:
    records: list[dict] = []
    for spec in preset.get("sources", []):
        if spec.get("enabled", True) is False:
            continue
        try:
            records.extend(load_source(spec, log))
        except Exception as e:  # one broken source should not kill the run
            log(f"WARNING: skipping source {spec.get('name', spec.get('path'))}: {e}")
    return records


def load_eval_records(preset: dict, limit: int | None, log: Callable[[str], None] = print) -> list[dict]:
    spec = dict(preset.get("eval") or {})
    if not spec:
        return []
    if limit:
        spec["limit"] = limit
    return load_source(spec, log)


def keep_thinking(cfg: dict, preset: dict) -> bool:
    val = (cfg.get("data") or {}).get("keep_thinking")
    return bool(preset.get("keep_thinking", False)) if val is None else bool(val)
