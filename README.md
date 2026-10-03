# AgentCreator

Distil a big open-source LLM (the **teacher**) into a small, fast **specialist** (the **student**).
Pick a specialty such as coding or reasoning. The app gets the teacher to answer thousands of specialty
prompts, keeps only the answers it can **verify** (code that passes unit tests, maths that matches the
ground truth), and LoRA-trains the student on them. It then scores the student against its untouched base,
exports a GGUF file and measures tokens per second.

```
 teacher (Qwen3.5-9B via Ollama)                                 student (Qwen3.5-2B / 0.8B)
        │                                                                 ▲
  generate ──► filter (run tests / check answers) ──► train (LoRA, loss on answers only)
                                                                          │
                              benchmark ◄── export (GGUF + Ollama Modelfile) ◄── eval (vs base)
```

Default pairing: **Qwen3.5-9B teacher → Qwen3.5-2B student** (quality) or **Qwen3.5-0.8B** (speed).
Both models come from the same family, so they share a tokenizer. That also lets the student work as a
speculative-decoding draft model for the teacher.

---

## 1. Setup

### Option A: native Windows (one double-click)

You need an NVIDIA driver and Python 3.11-3.13 from python.org (tick *Add python.exe to PATH*).

**Double-click `AgentCreator.bat`.** That's all. It does the following:

1. **First run only:** creates `.venv` and installs the app, Unsloth and the GPU build of PyTorch.
   This takes 10-30 minutes, and progress is saved to `logs\setup.log`. If it's interrupted, run it
   again; finished steps are skipped. It also puts an **AgentCreator** shortcut on your Desktop.
2. Installs **Ollama** with winget if it's missing, and starts the Ollama server.
3. Downloads the teacher model (`qwen3.5:9b`, about 6.6 GB) in its own window if you don't have it,
   so you can set up a run in the meantime.
4. Starts the app and opens http://127.0.0.1:7860 in your browser.

From then on it opens in a few seconds. Keep its window open while you use the app; closing it stops
the app. If the app is already running, double-clicking again just reopens the browser tab.

### Option B: WSL2 (Ubuntu)

```bash
cd /mnt/c/Users/Kavyya/Desktop/AgentCreator
bash setup.sh                 # creates .venv and installs everything
source .venv/bin/activate
python -m distiller ui        # or double-click start_ui_wsl.bat from Windows
```

In WSL, `localhost:11434` reaches Ollama on Windows when mirrored networking is on. If it doesn't,
use the Windows host IP or run Ollama inside WSL. Keep the project on the Linux filesystem (`~/`)
for faster training I/O.

### Checking the setup

```bash
python -m distiller doctor
```

This checks Python, PyTorch/CUDA, VRAM, transformers v5, Unsloth, bitsandbytes, the teacher server and
model, exporters, Hugging Face access, disk space and the code-test runner. Each problem comes with the
command that fixes it. The **System check** button in the UI runs the same checks.

The app asks Ollama to unload the teacher (`keep_alive: 0`) before any stage that puts the student on
the GPU, so the two never share your 8 GB.

### Windows notes

* **Python version:** Unsloth supports 3.11-3.13. Python 3.14 works too, but without Unsloth, so training is slower and uses more VRAM. Several Python versions can be installed side by side; `setup_windows.bat` picks the best one through the `py` launcher.
* **Exporting to GGUF:** on Windows, `export.method: auto` uses **Ollama's own importer**
  (`ollama create --quantize q4_K_M`), so you don't need to build llama.cpp, CMake or Visual Studio.
  The model then appears in `ollama list` straight away. If Ollama can't import an architecture, download a
  llama.cpp release zip (`...-bin-win-cuda-x64.zip`) plus the llama.cpp source (for `convert_hf_to_gguf.py`)
  into one folder, and set **llama.cpp folder** in the UI. The `.exe` tools are found automatically.
* **Antivirus:** the filter stage runs generated code with `python.exe` in a temp folder. If Defender
  blocks it, the system check reports it.
* **OneDrive:** keep the project outside OneDrive-synced folders (such as a synced Desktop). Sync can lock
  files mid-run.
* **Long paths:** if Hugging Face downloads fail with path errors, set `HF_HOME=C:\hf` or enable Windows
  long paths.
* **Stop button:** on Windows it sends Ctrl+Break to the run. The run stops cleanly and the stage is
  marked *stopped*, so you can resume it later.

## 2. Use the web UI

```bash
python -m distiller ui        # → http://127.0.0.1:7860   (Windows: double-click start_ui.bat)
```

The left side is a four-step wizard:

1. **Teacher.** Enter the server URL and click **Connect**. The dropdown lists the models installed in
   Ollama, plus suggestions you can `ollama pull`. **Test speed** sends the teacher one question.
2. **Student.** Pick one of the listed small models (Qwen3.5 0.8B/2B/4B, Qwen3, Llama 3.2, Gemma 3,
   SmolLM2) or enter any Hugging Face id or local path. The app estimates the VRAM training needs and
   warns you if it won't fit your GPU.
3. **Specialty.** Pick a built-in one (Coding, Reasoning) or click **+ Custom** and describe your own
   (see section 5).
4. **Training.** Choose **Quick test**, **Standard** or **Thorough**. All the knobs are under *Advanced settings*.

Then click **Start training**. The right side shows each run's stages, live log, loss curve, training
samples, eval outputs and result cards (score vs the untrained student, tokens/s, GGUF size and the Ollama command).

Only one run uses the GPU at a time. **Stop** ends it cleanly. Tick stages on a run and click
**Run selected stages** to resume or redo part of the pipeline.

## 3. Or use the CLI

```bash
python -m distiller presets
python -m distiller check-teacher --set teacher.model=qwen3.5:9b
python -m distiller run --name coding-2b --set specialty=coding
python -m distiller run --name reason-0.8b --set specialty=reasoning --set student.model=Qwen/Qwen3.5-0.8B
python -m distiller run --name coding-2b --stages train,eval     # redo just some stages
python -m distiller run -c my_config.yaml                        # your YAML, merged over configs/default.yaml
```

Config precedence: `configs/default.yaml` → the preset's `config_overrides` → your `-c` file → `--set` flags.

## 4. What each stage writes (`runs/<name>/`)

| stage | output | notes |
|---|---|---|
| generate | `data/raw.jsonl` (+ `synthetic_prompts.jsonl` for custom) | **Resumable**: a re-run only asks for missing answers. `samples_per_prompt > 1` = rejection sampling |
| filter | `data/train.jsonl`, `valid.jsonl`, `stats.json` | runs code against tests in a separate time- and memory-limited process; checks maths answers; removes duplicates |
| train | `adapter/`, `merged/`, `train/summary.json` | Unsloth bf16 LoRA (falls back to transformers + peft). Loss is computed only on the answer. Resumes from checkpoints |
| eval | `eval/results.json`, `*_outputs.jsonl` | built-in: greedy pass@1 on the held-out test set · custom: teacher-graded 1-10. Distilled vs untrained |
| export | `export/*.gguf` + `Modelfile`, or a model in Ollama | `export.method`: Ollama import (Windows default), your llama.cpp folder, or Unsloth's GGUF writer |
| benchmark | `benchmark.json` | `llama-bench` → `llama-cpp-python` → the Ollama-imported model → transformers, plus the teacher for a speedup number |

Then use the model in Ollama: `cd runs/<name>/export && ollama create <name> -f Modelfile`.

## 5. Specialties

| preset | prompts | how answers are verified | eval |
|---|---|---|---|
| `coding` | MBPP (train/val/prompt) + 3k CodeAlpaca | MBPP: unit tests executed · CodeAlpaca: must contain code | MBPP test (500) |
| `reasoning` | GSM8K train (7.4k) | final `Answer:` must equal ground truth; keeps `<think>` traces | GSM8K test |
| *custom* | written by the teacher (+ your files / datasets) | format and length (or must contain code) | held-out tasks, graded 1-10 by the teacher |

### Custom specialties (from the UI)

Click **+ Custom** and give the specialty a name, a description of what the student should be expert at,
and optionally a few example requests. Then:

* **Tasks the teacher writes:** the teacher first lists about 24 sub-topics, then writes that many
  varied tasks across them, at mixed difficulty, with duplicates removed. **Preview tasks** shows 5 samples before you commit.
* **Answers contain:** *Text*, or *Code* (keeps only answers that include a code block).
* **Thinking traces:** keep the teacher's `<think>` reasoning in the training targets (longer sequences).
* **Your own prompts (optional):** upload a `.txt` (one per line), `.jsonl` or `.json` file, and/or
  name a Hugging Face dataset and its prompt column. These are mixed with the teacher-written tasks.

Custom specialties have no answer key, so the app sets aside about 10% of the tasks (up to 60) as an
evaluation set and never trains on them. The teacher's answers to those tasks become the references.
After training, the **teacher grades** the distilled and the untrained student's answers from 1 to 10.
Saved specialties are YAML files in `presets/custom/`, and you can edit them by hand.

**Add one by hand:** copy `presets/coding.yaml`, then point `sources` at Hugging Face datasets or local
`.jsonl` files (`{"prompt": ..., "tests": [...]}` or `{"prompt": ..., "answer": ...}`). Choose a
`verifier` (`python_tests`, `numeric_answer`, `has_code`, `none`) and add an `eval` set.
It appears in the UI automatically.

## 6. Tips for 8 GB VRAM

* A Qwen3.5-2B bf16 LoRA needs about 5 GB, and 0.8B about 3 GB. Unsloth advises **against 4-bit QLoRA
  for Qwen3.5**, so `load_in_4bit` defaults to false.
* For reasoning, the preset raises `max_seq_length` to 4096 with batch 1 × accumulation 16.
  If you run out of memory, lower `student.max_seq_length`. Longer examples are skipped, never cut off.
* More data beats more epochs. Rejection sampling (`samples_per_prompt: 4`) on MBPP multiplies the
  number of verified coding examples.
* Start small: `--set data.max_prompts=200 --set eval.max_samples=30` checks the whole loop in
  well under an hour. Then scale up.
* Qwen3.5 needs **transformers v5**.

## 7. Tests

```bash
pip install pytest && python -m pytest -q tests     # no GPU or internet needed (uses a fake teacher)
```

## Safety note

The filter stage **executes code written by the teacher model** to check it against unit tests.
It runs in a separate isolated Python process, with a timeout, a clean environment and (on Linux/WSL)
memory and CPU limits. It is **not** a full sandbox. If you add untrusted prompt sources, run the app
in WSL or a container, or set `filter.execute_code: false` (that only checks whether the code compiles).
