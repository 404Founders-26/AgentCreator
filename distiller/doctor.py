"""`python -m distiller doctor` - checks that everything a run needs is installed and reachable.
Works the same on native Windows, WSL and Linux."""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys
import tempfile

OK, WARN, FAIL = "ok", "warn", "fail"


def _check(name: str, status: str, detail: str, fix: str = "") -> dict:
    return {"name": name, "status": status, "detail": detail, "fix": fix}


def _version(mod: str) -> str | None:
    try:
        return importlib.import_module(mod).__version__
    except Exception:
        return None


def run_checks(cfg: dict | None = None) -> list[dict]:
    from .config import ROOT, load_config, run_dir_for

    cfg = cfg or load_config()
    win = os.name == "nt"
    out: list[dict] = []

    # --- Python
    v = sys.version_info
    py = f"{v.major}.{v.minor}.{v.micro}"
    if v < (3, 10):
        out.append(_check("Python", FAIL, py, "Python 3.11-3.13 is recommended (3.13 is the newest Unsloth supports)"))
    elif v >= (3, 14) or v < (3, 11):
        out.append(_check("Python", WARN, f"{py} ({platform.system()} {platform.release()})",
                          "works, but Unsloth only supports Python 3.11-3.13, so training falls back to the slower "
                          "transformers + peft path. Install Python 3.13 alongside, delete .venv and re-run setup for full speed"))
    else:
        out.append(_check("Python", OK, f"{py} ({platform.system()} {platform.release()})"))
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix) or bool(os.environ.get("CONDA_PREFIX"))
    out.append(_check("Virtual environment", OK if in_venv else WARN, sys.prefix,
                      "" if in_venv else ("run setup_windows.bat, then start_ui.bat" if win else "source .venv/bin/activate")))

    # --- core packages
    missing = [m for m in ("fastapi", "uvicorn", "httpx", "yaml", "datasets") if importlib.util.find_spec(m) is None]
    out.append(_check("Core packages", FAIL if missing else OK, "missing: " + ", ".join(missing) if missing else "fastapi, uvicorn, httpx, pyyaml, datasets",
                      "pip install -r requirements.txt" if missing else ""))

    # --- PyTorch + GPU
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / 1024 ** 3
            out.append(_check("PyTorch + CUDA", OK, f"torch {torch.__version__}, CUDA {torch.version.cuda}, {props.name} ({vram:.1f} GB)"))
            out.append(_check("bf16 support", OK if torch.cuda.is_bf16_supported() else WARN,
                              "yes" if torch.cuda.is_bf16_supported() else "no - training falls back to fp16 (fine, a bit less stable)"))
        else:
            cpu_only = "+cpu" in torch.__version__ or torch.version.cuda is None
            out.append(_check("PyTorch + CUDA", FAIL, f"torch {torch.__version__} cannot see a GPU",
                              "install the CUDA build: pip install torch --index-url https://download.pytorch.org/whl/cu128"
                              if cpu_only else "update the NVIDIA driver, then reboot"))
    except Exception as e:
        out.append(_check("PyTorch + CUDA", FAIL, f"torch not importable ({e})",
                          "pip install torch --index-url https://download.pytorch.org/whl/cu128"))

    # --- training stack
    tv = _version("transformers")
    if not tv:
        out.append(_check("transformers", FAIL, "not installed", "pip install -r requirements-train.txt"))
    else:
        major = int(tv.split(".")[0])
        out.append(_check("transformers", OK if major >= 5 else WARN, tv,
                          "" if major >= 5 else "Qwen3.5 needs transformers v5: pip install -U \"transformers>=5\""))
    pv = _version("peft")
    out.append(_check("peft", OK if pv else FAIL, pv or "not installed", "" if pv else "pip install peft"))
    try:
        import unsloth  # noqa: F401

        out.append(_check("Unsloth", OK, _version("unsloth") or "installed"))
    except Exception as e:
        out.append(_check("Unsloth", WARN, f"not usable ({str(e).splitlines()[0][:120]})",
                          "optional - training falls back to transformers + peft (slower, more VRAM). pip install unsloth"))
    bv = _version("bitsandbytes")
    out.append(_check("bitsandbytes", OK if bv else WARN, bv or "not installed",
                      "" if bv else "optional - enables 8-bit AdamW (saves VRAM): pip install bitsandbytes"))

    # --- teacher server
    from .teacher import Teacher

    t = Teacher(cfg)
    try:
        models = t.list_models()
        want = cfg["teacher"]["model"]
        has = want in models
        out.append(_check("Teacher server", OK, f"{cfg['teacher']['base_url']} - {len(models)} model(s)"))
        out.append(_check("Teacher model", OK if has else WARN, want + (" installed" if has else " not installed"),
                          "" if has else f"ollama pull {want}   (or pick another model in the UI)"))
    except Exception as e:
        out.append(_check("Teacher server", FAIL, f"can't reach {cfg['teacher']['base_url']} ({type(e).__name__})",
                          "start Ollama (it runs in the system tray on Windows) or fix teacher.base_url"))
    finally:
        t.close()

    # --- export tools
    ollama = shutil.which("ollama")
    llama_dir = (cfg.get("export") or {}).get("llama_cpp_dir")
    from .export import find_llama_binary

    quant = find_llama_binary(llama_dir, "llama-quantize")
    if ollama or quant:
        out.append(_check("GGUF export", OK, "Ollama import" if ollama else f"llama.cpp ({quant})"))
    else:
        out.append(_check("GGUF export", WARN, "no ollama CLI or llama.cpp found",
                          "install Ollama (adds 'ollama' to PATH) or set export.llama_cpp_dir"))

    # --- Hugging Face
    try:
        import httpx

        r = httpx.get("https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B", timeout=10)
        out.append(_check("Hugging Face", OK if r.status_code < 400 else WARN, f"reachable (HTTP {r.status_code})"))
    except Exception as e:
        out.append(_check("Hugging Face", FAIL, f"unreachable ({type(e).__name__})", "check your internet / proxy"))
    hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    if win and len(hf_home) > 120:
        out.append(_check("HF cache path", WARN, hf_home, "very long path - set HF_HOME=C:\\hf to avoid Windows path-length errors"))

    # --- disk + write access
    runs = run_dir_for(cfg).parent
    try:
        runs.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=runs, delete=True):
            pass
        free = shutil.disk_usage(runs).free / 1024 ** 3
        out.append(_check("Disk space", OK if free > 25 else WARN, f"{free:.0f} GB free at {runs}",
                          "" if free > 25 else "a run needs ~10-25 GB (merged model, checkpoints, GGUF, HF cache)"))
    except Exception as e:
        out.append(_check("Runs folder", FAIL, f"cannot write to {runs}: {e}", "move the project out of a protected/synced folder"))

    # --- code-check sandbox (runs generated code against unit tests)
    from .verifiers import run_python_tests

    ok, why = run_python_tests("def f(x):\n    return x + 1", ["assert f(1) == 2"], timeout=20)
    out.append(_check("Code test runner", OK if ok else FAIL, "can run generated code in a subprocess" if ok else why,
                      "" if ok else "antivirus may be blocking python.exe in the temp folder - allow it, or set filter.execute_code: false"))
    if str(ROOT).lower().find("onedrive") != -1:
        out.append(_check("Project location", WARN, str(ROOT), "OneDrive sync can lock files mid-run - move the folder outside OneDrive"))
    return out


def print_report(checks: list[dict]) -> int:
    icon = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}
    for c in checks:
        print(f"{icon[c['status']]} {c['name']:<20} {c['detail']}")
        if c["fix"]:
            print(f"       -> {c['fix']}")
    fails = sum(c["status"] == FAIL for c in checks)
    warns = sum(c["status"] == WARN for c in checks)
    print(f"\n{fails} problem(s), {warns} warning(s)." + ("  Ready to train." if not fails else ""))
    return 1 if fails else 0
