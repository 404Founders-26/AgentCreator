#!/usr/bin/env bash
# One-time setup inside WSL2 (Ubuntu). Usage: bash setup.sh [cu128|cu126|cu124]
set -euo pipefail
CUDA_TAG="${1:-cu128}"
cd "$(dirname "$0")"
if ! command -v nvidia-smi >/dev/null; then
  echo "WARNING: nvidia-smi not found - install the NVIDIA Windows driver with WSL support first." >&2
fi
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install -r requirements.txt -r requirements-train.txt
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
print("transformers", transformers.__version__)
try:
    import unsloth  # noqa
    print("unsloth OK")
except Exception as e:
    print("unsloth import failed (training will use transformers+peft):", e)
PY
echo
echo "Done. Next:  source .venv/bin/activate && python -m distiller ui"
