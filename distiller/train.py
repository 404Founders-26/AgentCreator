"""Stage 3: LoRA-distill the student on the filtered teacher answers (loss only on the answer tokens)."""
from __future__ import annotations

import inspect
import math
from pathlib import Path

from .modeling import bf16_ok, dtype_kw, free_gpu, load_for_training, render_prompt, text_tokenizer
from .presets import keep_thinking, load_preset
from .runlog import Run, read_jsonl, write_json

STAGE = "train"


def tokenize_examples(rows: list[dict], tok, max_len: int, think: bool) -> tuple[list[dict], int]:
    t = text_tokenizer(tok)
    eos = t.eos_token or ""
    out, dropped = [], 0
    for r in rows:
        prompt = render_prompt(tok, r.get("system", ""), r["user"], think)
        target = r["assistant"]
        # some templates already open the think block in the generation prompt
        if prompt.rstrip().endswith("<think>") and target.startswith("<think>"):
            target = target[len("<think>"):].lstrip("\n")
        p_ids = t(prompt, add_special_tokens=False)["input_ids"]
        a_ids = t(target + eos, add_special_tokens=False)["input_ids"]
        if len(p_ids) >= max_len - 16:
            dropped += 1
            continue
        ids = (p_ids + a_ids)[:max_len]
        labels = ([-100] * len(p_ids) + a_ids)[:max_len]
        if len(p_ids) + len(a_ids) > max_len:
            dropped += 1  # a cut-off answer teaches the model to stop mid-way; skip it
            continue
        out.append({"input_ids": ids, "labels": labels})
    return out, dropped


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch

        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            labels[i, :L] = torch.tensor(b["labels"])
            mask[i, :L] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def make_args(**kw):
    """Build TrainingArguments across transformers versions (eval_strategy vs evaluation_strategy)."""
    from transformers import TrainingArguments

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "eval_strategy" in kw:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    return TrainingArguments(**{k: v for k, v in kw.items() if k in params})


def last_checkpoint(folder: Path) -> str | None:
    cks = sorted(folder.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)
    return str(cks[-1]) if cks else None


def run_train(cfg: dict, run: Run) -> dict:
    log = lambda m: run.log(m, STAGE)
    preset = load_preset(cfg["specialty"])
    think = keep_thinking(cfg, preset)
    train_rows = list(read_jsonl(run.path("data", "train.jsonl")))
    valid_rows = list(read_jsonl(run.path("data", "valid.jsonl")))
    if not train_rows:
        raise RuntimeError("data/train.jsonl is empty - run generate + filter first")

    import torch

    if not torch.cuda.is_available():
        log("WARNING: no CUDA GPU visible - training on CPU will be extremely slow")
    model, tok, backend = load_for_training(cfg, log)
    t = text_tokenizer(tok)
    if t.pad_token_id is None:
        t.pad_token = t.eos_token
    max_len = int(cfg["student"].get("max_seq_length", 2048))
    train_ds, d1 = tokenize_examples(train_rows, tok, max_len, think)
    valid_ds, d2 = tokenize_examples(valid_rows, tok, max_len, think)
    log(f"{len(train_ds)} train / {len(valid_ds)} valid sequences (dropped {d1 + d2} longer than {max_len} tokens)")
    if not train_ds:
        raise RuntimeError("every example was longer than max_seq_length - raise student.max_seq_length")

    from transformers import Trainer, TrainerCallback

    tc = cfg["training"]
    steps_per_epoch = math.ceil(len(train_ds) / (int(tc["batch_size"]) * int(tc["grad_accum"])))
    total_steps = steps_per_epoch * int(tc["epochs"])

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if not logs:
                return
            parts = [f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in logs.items()
                     if k in ("loss", "eval_loss", "learning_rate", "grad_norm", "epoch")]
            log(f"step {state.global_step}/{state.max_steps}: " + " ".join(parts))
            hist = run.path("train", "history.jsonl")
            with open(hist, "a", encoding="utf-8") as f:
                import json
                f.write(json.dumps({"step": state.global_step, **logs}) + "\n")
            run.progress(STAGE, state.global_step, state.max_steps or total_steps)

    ckpt_dir = run.path("train", "checkpoints", ".keep").parent
    use_bf16 = bf16_ok()
    try:
        import bitsandbytes  # noqa: F401
        optim = "adamw_8bit"
    except Exception:
        optim = "adamw_torch"
    args = make_args(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=int(tc["batch_size"]),
        per_device_eval_batch_size=int(tc["batch_size"]),
        gradient_accumulation_steps=int(tc["grad_accum"]),
        num_train_epochs=float(tc["epochs"]),
        learning_rate=float(tc["learning_rate"]),
        warmup_ratio=float(tc["warmup_ratio"]),
        weight_decay=float(tc.get("weight_decay", 0.0)),
        lr_scheduler_type="cosine",
        logging_steps=int(tc.get("logging_steps", 5)),
        save_strategy="steps",
        save_steps=max(25, steps_per_epoch // 2),
        save_total_limit=2,
        eval_strategy="epoch" if valid_ds else "no",
        bf16=use_bf16,
        fp16=(not use_bf16) and torch.cuda.is_available(),
        optim=optim,
        report_to="none",
        remove_unused_columns=False,
        seed=int(cfg.get("seed", 42)),
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds or None,
                      data_collator=PadCollator(t.pad_token_id), callbacks=[Progress()])
    resume = last_checkpoint(ckpt_dir) if tc.get("resume", True) else None
    if resume:
        log(f"resuming from {resume}")
    log(f"training {total_steps} steps ({tc['epochs']} epochs, effective batch {int(tc['batch_size']) * int(tc['grad_accum'])}, {optim}, {'bf16' if use_bf16 else ('fp16' if torch.cuda.is_available() else 'fp32 on CPU')})")
    result = trainer.train(resume_from_checkpoint=resume)
    metrics = dict(result.metrics)
    if valid_ds:
        metrics.update(trainer.evaluate())

    adapter_dir = run.path("adapter", ".keep").parent
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    log(f"LoRA adapter saved to {adapter_dir}")

    merged_dir = run.path("merged", ".keep").parent
    if backend == "unsloth":
        model.save_pretrained_merged(str(merged_dir), tok, save_method="merged_16bit")
        del trainer, model
        free_gpu()
    else:
        del trainer, model
        free_gpu()
        from peft import AutoPeftModelForCausalLM

        merged = AutoPeftModelForCausalLM.from_pretrained(
            str(adapter_dir), **dtype_kw(torch.bfloat16 if use_bf16 else torch.float16), device_map="cpu")
        merged = merged.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tok.save_pretrained(str(merged_dir))
        del merged
        free_gpu()
    log(f"merged 16-bit model saved to {merged_dir}")
    summary = {"backend": backend, "train_sequences": len(train_ds), "valid_sequences": len(valid_ds),
               "steps": total_steps, "metrics": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in metrics.items()}}
    write_json(run.path("train", "summary.json"), summary)
    return summary
