"""Config loading: YAML defaults + user file + dotted CLI overrides."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (extra or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def parse_value(raw: str) -> Any:
    """Turn a CLI string into a typed value (yaml rules: 2 -> int, true -> bool, null -> None)."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def get_dotted(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def load_config(path: str | Path | None = None, overrides: list[str] | dict | None = None) -> dict:
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    if isinstance(overrides, dict):
        cfg = deep_merge(cfg, overrides)
    elif overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"--set expects key=value, got {item!r}")
            k, v = item.split("=", 1)
            set_dotted(cfg, k.strip(), parse_value(v.strip()))
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def run_dir_for(cfg: dict) -> Path:
    workdir = Path(cfg.get("workdir") or "runs")
    if not workdir.is_absolute():
        workdir = ROOT / workdir
    name = safe_name(cfg.get("project_name") or "run")
    return workdir / name


WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_name(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in str(name).strip())
    keep = keep.strip("-.") or "run"
    if keep.split(".")[0].upper() in WINDOWS_RESERVED:  # CON, NUL, COM1... can't be folder names on Windows
        keep = f"{keep}-run"
    return keep[:80]


def to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
