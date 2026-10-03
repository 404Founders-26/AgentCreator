"""Stage 5: turn the merged model into a quantized GGUF (+ an Ollama Modelfile)."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .runlog import Run, write_json

STAGE = "export"


def find_llama_binary(llama_dir: Path | None, name: str) -> str | None:
    """Finds llama.cpp tools in a source build (build/bin, build/bin/Release) or an unzipped Windows release."""
    exe = name + (".exe" if os.name == "nt" else "")
    if llama_dir:
        llama_dir = Path(llama_dir).expanduser()
        for cand in [llama_dir / exe, llama_dir / "build" / "bin" / exe, llama_dir / "build" / "bin" / "Release" / exe]:
            if cand.exists():
                return str(cand)
        if llama_dir.exists():
            hits = sorted(llama_dir.rglob(exe))
            if hits:
                return str(hits[0])
    return shutil.which(name)


def _find_quantize(llama_dir: Path) -> str | None:
    return find_llama_binary(llama_dir, "llama-quantize")


def _stream(cmd: list[str], log, env: dict | None = None) -> None:
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, env=env)
    tail = []
    last = ""
    for line in proc.stdout:  # type: ignore[union-attr]
        line = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]|[\u2800-\u28ff]", "", line).strip()  # drop terminal colours/spinners
        if line == last:
            continue
        last = line
        tail = (tail + [line])[-20:]
        if "%|" in line:  # progress bars - too noisy for the run log
            continue
        if any(k in line.lower() for k in ("error", "writing", "llama_model_quantize", "converting", "model size")):
            log(line[:300])
    if proc.wait() != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n" + "\n".join(tail))


DIRECT_TYPES = {"f16", "bf16", "q8_0"}  # convert_hf_to_gguf.py writes these itself - no compiled tools needed


def export_llama_cpp(merged: Path, out: Path, quant: str, llama_dir: Path, keep_f16: bool, name: str, log) -> Path:
    convert = llama_dir / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise RuntimeError(f"{convert} not found - export.llama_cpp_dir must point to a llama.cpp checkout")
    env = dict(os.environ, PYTHONUTF8="1", NO_LOCAL_GGUF="")
    gguf_py = llama_dir / "gguf-py"
    if gguf_py.exists():  # use the gguf package that matches this convert script
        env["PYTHONPATH"] = str(gguf_py) + os.pathsep + env.get("PYTHONPATH", "")
    q = quant.lower()
    qbin = _find_quantize(llama_dir)
    if q not in DIRECT_TYPES and not qbin and os.name == "nt":
        try:
            fetch_llama_cpp_bins(llama_dir / "bin", log)
            qbin = _find_quantize(llama_dir)
        except Exception as e:
            log(f"could not download llama.cpp tools: {e}")
    if q not in DIRECT_TYPES and not qbin:
        log(f"llama-quantize not available - writing q8_0 instead of {q} (still small and fast for a small model)")
        q = "q8_0"
    direct = q if q in DIRECT_TYPES else "f16"
    first = out / f"{name}-{direct}.gguf"
    _stream([sys.executable, str(convert), str(merged), "--outfile", str(first), "--outtype", direct], log, env=env)
    if q in DIRECT_TYPES:
        return first
    final = out / f"{name}-{q}.gguf"
    _stream([qbin, str(first), str(final), q.upper()], log)
    if not keep_f16:
        first.unlink(missing_ok=True)
    return final


LLAMA_ZIP = "https://codeload.github.com/ggml-org/llama.cpp/zip/refs/heads/master"


def fetch_llama_cpp_tools(dest: Path, log) -> Path:
    """Download just the llama.cpp converter (python, ~140 files) - no compiler or build needed."""
    if (dest / "convert_hf_to_gguf.py").exists():
        return dest
    import io
    import zipfile

    import httpx

    log("downloading the llama.cpp GGUF converter (~40 MB, one time)")
    data = httpx.get(LLAMA_ZIP, follow_redirects=True, timeout=600).content
    z = zipfile.ZipFile(io.BytesIO(data))
    for name in z.namelist():
        rel = name.split("/", 1)[1] if "/" in name else ""
        if not rel or name.endswith("/"):
            continue
        if rel.startswith(("convert_hf_to_gguf.py", "gguf-py/", "conversion/")):
            p = dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(z.read(name))
    return dest


def fetch_llama_cpp_bins(dest: Path, log) -> Path:
    """Windows: download llama.cpp's prebuilt CPU tools (~20 MB) - llama-quantize is all we need from it."""
    if (dest / "llama-quantize.exe").exists():
        return dest
    import io
    import zipfile

    import httpx

    rels = httpx.get("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5", timeout=60).json()
    for rel in rels:
        for a in rel.get("assets", []):
            if a["name"].endswith("-bin-win-cpu-x64.zip"):
                log(f"downloading llama.cpp tools {rel['tag_name']} (~{a['size'] // 1_000_000} MB, one time)")
                data = httpx.get(a["browser_download_url"], follow_redirects=True, timeout=600).content
                dest.mkdir(parents=True, exist_ok=True)
                zipfile.ZipFile(io.BytesIO(data)).extractall(dest)
                return dest
    raise RuntimeError("no Windows build found in the latest llama.cpp releases")


def import_gguf_into_ollama(gguf: Path, out: Path, name: str, log) -> str | None:
    exe = shutil.which("ollama")
    if not exe:
        return None
    mf = out / "Modelfile"
    model = ollama_name(name)
    try:
        _stream([exe, "create", model, "-f", str(mf)], log)
        return model
    except RuntimeError as e:
        log(f"could not import the GGUF into Ollama: {str(e).splitlines()[-1][:200]}")
        return None


def export_unsloth(merged: Path, out: Path, quant: str, cfg: dict, log) -> Path:
    from unsloth import FastLanguageModel

    log("exporting with Unsloth's GGUF writer (it builds llama.cpp on first use, which takes a few minutes)")
    model, tok = FastLanguageModel.from_pretrained(model_name=str(merged), max_seq_length=int(cfg["student"]["max_seq_length"]),
                                                   load_in_4bit=False, dtype=None)
    model.save_pretrained_gguf(str(out), tok, quantization_method=quant.lower())
    ggufs = sorted(out.rglob("*.gguf"), key=lambda p: p.stat().st_mtime)
    if not ggufs:
        raise RuntimeError("Unsloth finished but no .gguf file was written")
    return ggufs[-1]


OLLAMA_QUANTS = {"q4_k_m": "q4_K_M", "q4_k_s": "q4_K_S", "q8_0": "q8_0", "f16": None, "bf16": None}


def ollama_name(run_name: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", run_name.lower()).strip("-.") or "distilled"


def export_ollama(merged: Path, out: Path, quant: str, name: str, log) -> dict:
    """Let Ollama import + quantize the safetensors model itself - no llama.cpp build needed (handy on Windows)."""
    exe = shutil.which("ollama")
    if not exe:
        raise RuntimeError("the 'ollama' command was not found on PATH")
    q = OLLAMA_QUANTS.get(quant.lower(), "missing")
    if q == "missing":
        log(f"Ollama can't quantize to {quant}; using q4_K_M")
        q = "q4_K_M"
    modelfile = out / "Modelfile.safetensors"
    src = merged.resolve().as_posix()  # forward slashes work for Ollama on Windows too
    if " " in src:
        src = f'"{src}"'
    modelfile.write_text(f"FROM {src}\nPARAMETER temperature 0.6\nPARAMETER top_p 0.95\nPARAMETER num_ctx 4096\n",
                         encoding="utf-8")
    model = ollama_name(name)
    cmd = [exe, "create", model, "-f", str(modelfile)] + (["--quantize", q] if q else [])
    try:
        _stream(cmd, log)
    except RuntimeError as e:
        # newer Ollama builds renamed the quantization types (int4, int8, nvfp4, mxfp4, mxfp8)
        msg = str(e)
        if not q or "unsupported --quantize" not in msg:
            raise
        alt = "int8" if q.startswith("q8") else "int4"
        log(f"this Ollama version doesn't know '{q}' - using its '{alt}' quantization instead")
        try:
            _stream([exe, "create", model, "-f", str(modelfile), "--quantize", alt], log)
            q = alt
        except RuntimeError as e2:
            # on Windows these quant types need Ollama's MLX runtime - import unquantized (16-bit) instead
            log(f"Ollama can't quantize here ({str(e2).splitlines()[-1][:120]}) - importing the 16-bit model instead")
            _stream([exe, "create", model, "-f", str(modelfile)], log)
            q = None
    return {"ollama_model": model, "quantization": q or "f16"}


def pick_method(e: dict) -> str:
    method = str(e.get("method") or "auto").lower()
    if method != "auto":
        return method
    if e.get("llama_cpp_dir"):
        return "llama_cpp"
    try:
        import unsloth  # noqa: F401
        has_unsloth = True
    except Exception:
        has_unsloth = False
    # On Windows: Unsloth's GGUF writer needs CMake + Visual Studio, and Ollama's safetensors import needs
    # MLX (not available) - so convert with llama.cpp's python converter, then load the GGUF into Ollama
    if os.name == "nt":
        return "llama_cpp"
    if has_unsloth:
        return "unsloth"
    if shutil.which("ollama"):
        return "ollama"
    return "llama_cpp"  # the converter is downloaded automatically


def run_export(cfg: dict, run: Run) -> dict:
    log = lambda m: run.log(m, STAGE)
    merged = run.dir / "merged"
    if not (merged / "config.json").exists():
        raise RuntimeError("no merged model found - run the train stage first")
    out = run.path("export", ".keep").parent
    e = cfg.get("export") or {}
    quant = str(e.get("quantization", "q4_k_m"))
    name = run.dir.name
    method = pick_method(e)
    log(f"export method: {method}")
    if method == "ollama":
        info = export_ollama(merged, out, quant, name, log)
        info.update({"method": "ollama", "ollama": f"ollama run {info['ollama_model']}"})
        write_json(run.path("export", "export.json"), info)
        log(f"model imported into Ollama as '{info['ollama_model']}' - try: {info['ollama']}")
        return info
    if method == "llama_cpp":
        from .config import ROOT

        ldir = Path(e["llama_cpp_dir"]).expanduser() if e.get("llama_cpp_dir") else fetch_llama_cpp_tools(ROOT / "tools" / "llama.cpp", log)
        gguf = export_llama_cpp(merged, out, quant, ldir, bool(e.get("keep_f16")), name, log)
    elif method == "unsloth":
        gguf = export_unsloth(merged, out, quant, cfg, log)
    else:
        raise RuntimeError(f"unknown export.method '{method}' (auto, llama_cpp, unsloth, ollama)")
    size_mb = round(gguf.stat().st_size / 1e6, 1)
    modelfile = out / "Modelfile"
    modelfile.write_text(
        f"FROM ./{gguf.relative_to(out).as_posix()}\n"
        "PARAMETER temperature 0.6\nPARAMETER top_p 0.95\nPARAMETER num_ctx 4096\n", encoding="utf-8")
    model = ollama_name(name)
    qname = gguf.stem.rsplit("-", 1)[-1]
    info = {"method": method, "gguf": str(gguf), "size_mb": size_mb, "quantization": qname,
            "ollama": f'cd "{out}" && ollama create {model} -f Modelfile'}
    imported = import_gguf_into_ollama(gguf, out, name, log)
    if imported:
        info["ollama_model"] = imported
        info["ollama"] = f"ollama run {imported}"
        log(f"model imported into Ollama as '{imported}'")
    write_json(run.path("export", "export.json"), info)
    log(f"GGUF ready: {gguf.name} ({size_mb} MB)")
    log(f"to use it in Ollama: {info['ollama']}")
    return info
