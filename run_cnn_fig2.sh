#!/bin/bash
#
# run_cnn_fig2.sh — fast reproduction of the paper's Figure 2 (MNIST CNN).
#
# Trims the full sweep down to a few-hour job:
#   * Table 2 configs only (E x B at C=0.1), both IID and non-IID  -> 18 runs
#   * NO learning-rate sweep: uses the paper's tuned CNN eta per config (1 run each)
#   * lower B=inf caps (FedSGD baselines don't need their full multi-thousand-round tail)
#
# Usage:
#   ./run_cnn_fig2.sh           # 2 concurrent jobs (good for single MPS GPU)
#   ./run_cnn_fig2.sh 1         # fully serial
#
# Then plot:
#   python plot_results.py --glob 'results/cnn_t2_*/lr_sweep_*.json' --target 0.99 --model-name CNN

set -u
cd "$(dirname "$0")"
CONCURRENCY="${1:-2}"          # MPS is one GPU; 2 is plenty, more just contends

if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
mkdir -p logs results

# paper's tuned CNN learning rates (C=0.1). B=inf/E=1 is the FedSGD row (eta=0.15);
# B=inf with E>1 has no published value, so we reuse the FedAvg-by-E eta as a sensible default.
eta_for() {  # eta_for <E> <B>
  local E="$1" B="$2"
  if [ "$B" = "600" ] && [ "$E" = "1" ]; then echo 0.15; return; fi
  case "$E" in 1) echo 0.30;; 5) echo 0.17;; 20) echo 0.15;; *) echo 0.15;; esac
}

JOBS=()
for dist in iid noniid; do
  for E in 1 5 20; do
    for B in 600 50 10; do
      Bs="$B"; [ "$B" = "600" ] && Bs="inf"
      L="$(eta_for "$E" "$B")"

      if [ "$dist" = "iid" ]; then
        niflag=""; dtag="iid"
        case "$B" in 600) mr=1000;; 50) mr=500;; *) mr=200;; esac
      else
        niflag="--non-iid"; dtag="ni2"
        case "$B" in 600) mr=1200;; 50) mr=800;; *) mr=600;; esac
      fi

      tag="cnn_t2_${dtag}_E${E}_B${Bs}"
      existing=( results/"${tag}"/lr_sweep_*.json )
      if [ -e "${existing[0]}" ]; then echo "skip (done): ${tag}"; continue; fi

      # single-value "sweep" = one training run, but keeps the JSON format plot_results reads
      JOBS+=("python experiment.py --model cnn --target 0.99 --trial E=${E},B=${B} --lr-sweep ${L} --max-rounds ${mr} ${niflag} --output results/${tag} > logs/${tag}.log 2>&1 && echo \"  done: ${tag} (eta=${L})\" || echo \"  FAILED: ${tag}\"")
    done
  done
done

total="${#JOBS[@]}"
[ "$total" -eq 0 ] && { echo "Nothing to run — all configs already done."; exit 0; }

echo "CNN Figure 2: launching ${total} jobs, ${CONCURRENCY} at a time..."
echo
printf '%s\0' "${JOBS[@]}" | xargs -0 -P "$CONCURRENCY" -n1 bash -c 'eval "$1"' _
echo
echo "Done. Plot with:"
echo "  python plot_results.py --glob 'results/cnn_t2_*/lr_sweep_*.json' --target 0.99 --model-name CNN"
