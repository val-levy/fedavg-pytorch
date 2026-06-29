#!/bin/bash


set -euo pipefail
cd "$(dirname "$0")"
CONCURRENCY="${1:-}"          # empty => run_cnn_fig2.sh auto-detects (one job per GPU)

echo "==> GPU:"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "  (no nvidia-smi — is this a CUDA pod?)"

echo "==> Python deps"
# RunPod PyTorch images already ship CUDA torch — do NOT reinstall it.
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
  && echo "  torch+CUDA OK" \
  || { echo "  installing CUDA torch (no CUDA torch found)"; pip install --quiet torch; }
python -c "import torchvision" 2>/dev/null || pip install --quiet torchvision
python -c "import matplotlib, numpy" 2>/dev/null || pip install --quiet matplotlib numpy

echo "==> Confirm device the code will use"
python - <<'PY'
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  training device: {dev}")
if dev == "cuda":
    print(f"  {torch.cuda.get_device_name(0)}")
PY

echo "==> Pre-download MNIST (so 18 workers don't race on it)"
python -c "from util import get_dataset; get_dataset(train=True); get_dataset(train=False); print('  MNIST ready')"

echo "==> Quick benchmark (3 rounds, heaviest config) so you know the pace"
python experiment.py --model cnn --bench 3 --trial E=20,B=10 || true

chmod +x run_cnn_fig2.sh
echo "==> Launching CNN Figure-2 run (one job per GPU)"
./run_cnn_fig2.sh $CONCURRENCY

echo "==> Plotting"
python plot_results.py --glob 'results/cnn_t2_*/lr_sweep_*.json' --target 0.99 --model-name CNN

echo "==> Packaging results (figures + raw JSON + logs) for download"
zip -r results.zip results logs >/dev/null && echo "  wrote results.zip ($(du -h results.zip | cut -f1))"

echo
echo "DONE."
echo "  Figure:   results/figures/accuracy_vs_rounds.png"
echo "  Download: results.zip  (contains both PNGs, all 18 per-config JSONs, and logs)"
echo "  Pull it back via the RunPod file browser, or:  runpodctl send results.zip"
