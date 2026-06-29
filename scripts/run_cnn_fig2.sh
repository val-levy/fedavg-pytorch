#!/bin/bash
#
# run_cnn_fig2.sh — fast reproduction of the paper's Figure 2 (MNIST CNN).
#
#   * Table 2 configs only (E x B at C=0.1), IID + non-IID  -> 18 runs
#   * NO learning-rate sweep: paper's tuned CNN eta per config (1 run each)
#   * lower B=inf caps (FedSGD baselines don't need their full tail)
#
# Multi-GPU aware: detects the GPU count, then load-balances the 18 jobs across
# the cards (longest-job-first) with one serial queue per worker, so no card sits
# idle while another doubles up. Each job is pinned to its card via CUDA_VISIBLE_DEVICES.
#
# Usage:
#   ./run_cnn_fig2.sh            # workers = #GPUs (one job per card at a time)
#   ./run_cnn_fig2.sh 16         # 16 workers (e.g. ~2 jobs/GPU on an 8-GPU box,
#                                #  to overlap this job's heavy CPU/Python stalls)
#
# Then plot:
#   python plot_results.py --glob 'results/cnn_t2_*/lr_sweep_*.json' --target 0.99 --model-name CNN

set -u
cd "$(dirname "$0")"

if command -v nvidia-smi >/dev/null 2>&1; then
  NG="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  NG=1
fi
[ "$NG" -ge 1 ] 2>/dev/null || NG=1
WORKERS="${1:-$NG}"            # number of parallel queues; card = worker % NG
echo "Detected ${NG} GPU(s); using ${WORKERS} parallel worker(s)."

if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
mkdir -p logs results

# paper's tuned CNN learning rates (C=0.1). B=inf/E=1 is the FedSGD row (eta=0.15);
# B=inf with E>1 has no published value, so reuse the FedAvg-by-E eta as a default.
eta_for() {  # eta_for <E> <B>
  local E="$1" B="$2"
  if [ "$B" = "600" ] && [ "$E" = "1" ]; then echo 0.15; return; fi
  case "$E" in 1) echo 0.30;; 5) echo 0.17;; 20) echo 0.15;; *) echo 0.15;; esac
}

# ---- build the job list (command + a weight proxy for load balancing) ----
CMDS=(); WEIGHTS=()
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

      # weight ~ images processed upper bound = 6000*E*max_rounds (good balancing proxy)
      WEIGHTS+=( $(( 6 * E * mr )) )
      CMDS+=("python experiment.py --model cnn --target 0.99 --trial E=${E},B=${B} --lr-sweep ${L} --max-rounds ${mr} ${niflag} --output results/${tag} > logs/${tag}.log 2>&1 && echo \"  done: ${tag} (eta=${L})\" || echo \"  FAILED: ${tag} (see logs/${tag}.log)\"")
    done
  done
done

total="${#CMDS[@]}"
[ "$total" -eq 0 ] && { echo "Nothing to run — all configs already done."; exit 0; }
[ "$WORKERS" -gt "$total" ] && WORKERS="$total"

# ---- LPT scheduling: assign heaviest jobs first to the currently-lightest worker ----
TMPD="$(mktemp -d)"
trap 'rm -rf "$TMPD"' EXIT
declare -a LOAD
for ((w=0; w<WORKERS; w++)); do LOAD[$w]=0; : > "$TMPD/q$w.sh"; done

# indices sorted by weight, descending
order=$(for i in "${!WEIGHTS[@]}"; do echo "$i ${WEIGHTS[$i]}"; done | sort -k2 -nr | awk '{print $1}')
for i in $order; do
  min=0
  for ((w=1; w<WORKERS; w++)); do (( LOAD[w] < LOAD[min] )) && min=$w; done
  gpu=$(( min % NG ))
  echo "CUDA_VISIBLE_DEVICES=${gpu} ${CMDS[$i]}" >> "$TMPD/q$min.sh"
  LOAD[$min]=$(( LOAD[min] + WEIGHTS[i] ))
done

echo "CNN Figure 2: ${total} jobs across ${WORKERS} workers on ${NG} GPU(s)..."
echo
pids=()
for ((w=0; w<WORKERS; w++)); do
  bash "$TMPD/q$w.sh" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo
if [ "$fail" -eq 0 ]; then echo "All jobs finished."; else echo "Some jobs failed — check logs/."; fi
echo "Plot with:"
echo "  python plot_results.py --glob 'results/cnn_t2_*/lr_sweep_*.json' --target 0.99 --model-name CNN"
