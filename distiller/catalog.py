"""Model choices shown in the UI, with rough VRAM estimates for LoRA training on one GPU."""
from __future__ import annotations

import re

# params in billions; "gated" = needs a Hugging Face login + licence acceptance
STUDENTS = [
    {"id": "Qwen/Qwen3.5-0.8B", "params": 0.8, "family": "qwen3.5", "note": "fastest · same family as Qwen3.5 teachers"},
    {"id": "Qwen/Qwen3.5-2B", "params": 2.0, "family": "qwen3.5", "note": "best quality/speed balance on 8 GB"},
    {"id": "Qwen/Qwen3.5-4B", "params": 4.0, "family": "qwen3.5", "note": "strongest, needs ~10 GB in 16-bit"},
    {"id": "Qwen/Qwen3-0.6B", "params": 0.6, "family": "qwen3", "note": "older generation, tiny"},
    {"id": "Qwen/Qwen3-1.7B", "params": 1.7, "family": "qwen3", "note": "older generation"},
    {"id": "Qwen/Qwen3-4B", "params": 4.0, "family": "qwen3", "note": "older generation, 4-bit QLoRA fits 8 GB"},
    {"id": "meta-llama/Llama-3.2-1B-Instruct", "params": 1.2, "family": "llama", "note": "gated on Hugging Face"},
    {"id": "meta-llama/Llama-3.2-3B-Instruct", "params": 3.2, "family": "llama", "note": "gated on Hugging Face"},
    {"id": "google/gemma-3-1b-it", "params": 1.0, "family": "gemma", "note": "gated on Hugging Face"},
    {"id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "params": 1.7, "family": "smollm", "note": "fully open"},
]

TEACHER_SUGGESTIONS = ["qwen3.5:9b", "qwen3.5:4b", "qwen3:8b", "llama3.1:8b", "gemma3:12b", "deepseek-r1:8b"]


def guess_params(model_id: str) -> float | None:
    for s in STUDENTS:
        if s["id"].lower() == str(model_id).lower():
            return s["params"]
    m = re.search(r"(\d+(?:\.\d+)?)\s*([bm])\b", str(model_id).lower().replace("-", " ").replace("_", " "))
    if not m:
        return None
    v = float(m.group(1))
    return v / 1000 if m.group(2) == "m" else v


def family(model_id: str) -> str:
    s = str(model_id).lower()
    for fam in ("qwen3.5", "qwen3", "qwen2.5", "llama", "gemma", "mistral", "phi", "deepseek", "smollm"):
        if fam in s:
            return fam
    return "other"


def estimate_vram_gb(params_b: float | None, load_in_4bit: bool, seq_len: int, batch: int) -> float | None:
    """Very rough: weights + LoRA/optimizer states + activations (with gradient checkpointing)."""
    if not params_b:
        return None
    weights = params_b * (0.6 if load_in_4bit else 2.0)
    lora_and_optim = 0.15 + 0.05 * params_b
    activations = 0.35 * params_b ** 0.5 * (seq_len / 2048) * batch
    overhead = 0.8  # CUDA context, fragmentation
    return round(weights + lora_and_optim + activations + overhead, 1)


def student_advice(model_id: str, load_in_4bit: bool, seq_len: int, batch: int, vram_gb: float | None) -> dict:
    p = guess_params(model_id)
    need = estimate_vram_gb(p, load_in_4bit, seq_len, batch)
    out = {"params_b": p, "estimated_vram_gb": need, "family": family(model_id), "warnings": []}
    if need and vram_gb and need > vram_gb * 0.95:
        out["warnings"].append(f"~{need} GB needed but the GPU has {vram_gb:.0f} GB - "
                               + ("lower max sequence length / batch size" if load_in_4bit else "turn on 4-bit or pick a smaller student"))
    if load_in_4bit and out["family"] == "qwen3.5":
        out["warnings"].append("Unsloth advises against 4-bit (QLoRA) training for Qwen3.5 models")
    return out
