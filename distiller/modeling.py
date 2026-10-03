"""Model loading shared by train / eval / benchmark. Prefers Unsloth, falls back to transformers + peft."""
from __future__ import annotations

import gc
import subprocess
import sys
from typing import Callable

_ENV_OK = False


def _probe(code: str) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = (r.stderr or r.stdout or "").strip().splitlines()
    return r.returncode == 0, (lines[-1] if lines else "")


def ensure_environment(log: Callable[[str], None]) -> None:
    """Self-repair the most common Windows install problem: a torchvision built for a different torch
    ("operator torchvision::nms does not exist"). transformers imports torchvision whenever it is installed,
    so a mismatched one breaks peft, Unsloth and training. Checked in a separate process so a broken import
    can't poison this one."""
    global _ENV_OK
    if _ENV_OK:
        return
    ok, err = _probe("import torchvision")
    if ok:
        _ENV_OK = True
        return
    has_tv, _ = _probe("import importlib.metadata as m; m.version('torchvision')")
    if not has_tv:
        _ENV_OK = True
        return
    import torch

    base = torch.__version__.split("+")[0]
    major, minor, patch = (base.split(".") + ["0", "0"])[:3]
    cuda = (torch.version.cuda or "").replace(".", "")
    want = f"0.{int(minor) + 15}.{patch}"  # torch 2.N pairs with torchvision 0.(N+15)
    log(f"torchvision does not match torch {torch.__version__} ({err[:120]}) - installing torchvision {want}")
    pip = [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", f"torchvision=={want}"]
    if cuda:
        pip += ["--index-url", f"https://download.pytorch.org/whl/cu{cuda}"]
    r = subprocess.run(pip, capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok, err = _probe("import torchvision")
    if not ok:
        # text models don't need torchvision at all - removing it is a safe fallback
        log(f"could not install a matching torchvision ({(r.stderr or '').strip().splitlines()[-1:] }); uninstalling it instead")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"], capture_output=True)
    else:
        log(f"torchvision {want} installed")
    _ENV_OK = True


_UNSLOTH_ERR: str | None = None


def has_unsloth() -> bool:
    global _UNSLOTH_ERR
    try:
        import unsloth  # noqa: F401  (must be imported before transformers to patch it)
        return True
    except Exception as e:
        _UNSLOTH_ERR = f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
        return False


def text_tokenizer(tok):
    """Qwen3.5 checkpoints can ship a multimodal processor; we only need its text tokenizer."""
    return getattr(tok, "tokenizer", tok)


def render_prompt(tok, system: str, user: str, think: bool) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    t = text_tokenizer(tok)
    try:
        return t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    except TypeError:
        return t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def dtype_kw(dtype) -> dict:
    """transformers v5 renamed torch_dtype -> dtype."""
    import transformers

    major = int(transformers.__version__.split(".")[0])
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


def bf16_ok() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def load_for_training(cfg: dict, log: Callable[[str], None]):
    ensure_environment(log)
    s, t = cfg["student"], cfg["training"]
    name, seq, four = s["model"], int(s.get("max_seq_length", 2048)), bool(s.get("load_in_4bit", False))
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if has_unsloth():
        from unsloth import FastLanguageModel

        log(f"loading {name} with Unsloth ({'4-bit QLoRA' if four else '16-bit LoRA'})")
        kwargs = dict(model_name=name, max_seq_length=seq, load_in_4bit=four, dtype=None)
        if not four:
            kwargs["load_in_16bit"] = True
        try:
            model, tok = FastLanguageModel.from_pretrained(**kwargs)
        except TypeError:  # older Unsloth without load_in_16bit
            kwargs.pop("load_in_16bit", None)
            model, tok = FastLanguageModel.from_pretrained(**kwargs)
        model = FastLanguageModel.get_peft_model(
            model, r=int(t["lora_r"]), lora_alpha=int(t["lora_alpha"]), lora_dropout=float(t["lora_dropout"]),
            target_modules=targets, bias="none", use_gradient_checkpointing="unsloth", random_state=cfg.get("seed", 42),
        )
        return model, tok, "unsloth"

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"Unsloth not usable ({_UNSLOTH_ERR}) - loading {name} with transformers + peft")
    dtype = torch.bfloat16 if bf16_ok() else torch.float16
    kwargs: dict = {**dtype_kw(dtype), "device_map": "auto"}
    if four:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if four:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lcfg = LoraConfig(r=int(t["lora_r"]), lora_alpha=int(t["lora_alpha"]), lora_dropout=float(t["lora_dropout"]),
                      target_modules=targets, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model, tok, "peft"


def load_for_inference(path: str, cfg: dict, log: Callable[[str], None]):
    ensure_environment(log)
    seq = int(cfg["student"].get("max_seq_length", 2048))
    if has_unsloth():
        from unsloth import FastLanguageModel

        log(f"loading {path} for inference (Unsloth)")
        kwargs = dict(model_name=str(path), max_seq_length=seq + int(cfg["eval"].get("max_new_tokens", 1024)),
                      load_in_4bit=False, dtype=None)
        model, tok = FastLanguageModel.from_pretrained(**kwargs)
        FastLanguageModel.for_inference(model)
        return model, tok
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"loading {path} for inference (transformers)")
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_pretrained(
        str(path), **dtype_kw(torch.bfloat16 if bf16_ok() else torch.float16), device_map="auto")
    model.eval()
    return model, tok


def free_gpu(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
