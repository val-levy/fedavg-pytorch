#!/bin/bash
#
# run_all.sh — reproduce the FedAvg MNIST 2NN experiments (Tables 1 & 2,
# IID + non-IID) with a controlled concurrency limit.
#
# Usage:
#   ./run_all.sh            # 4 jobs at a time (default)
#   ./run_all.sh 6          # 6 jobs at a time
#
# Each job runs experiment.py for one config, auto-tuning eta via --lr-sweep,
# writing results to results/<tag>/ and a log to logs/<tag>.log.
# Already-completed configs (a lr_sweep_*.json already exists) are SKIPPED,
# so the script is resumable — re-run it any time and it only does what's left.

set -u
cd "$(dirname "$0")"

CONCURRENCY="${1:-4}"
MODEL="${MODEL:-twonn}"     # twonn | cnn   (override: MODEL=cnn ./run_all.sh)
TARGET="${TARGET:-0.97}"    # 0.97 for 2NN, 0.99 for CNN  (override: TARGET=0.99 ...)

# activate venv (children inherit the exported PATH)
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

mkdir -p logs results

JOBS=()

# add_job <table> <E> <B> <C> <dist>
#   table: 1 or 2     dist: iid or noniid     B: 600 means B=inf (full shard)
add_job() {
  local table="$1" E="$2" B="$3" C="$4" dist="$5"

  local Bs="$B"; [ "$B" = "600" ] && Bs="inf"

  # eta grid: bigger for full-batch / FedSGD-like configs
  local grid="0.05,0.1,0.2,0.3"; [ "$B" = "600" ] && grid="0.1,0.3,0.5"

  # round caps: non-IID and full-batch need many more rounds to hit target
  local mr
  if [ "$dist" = "iid" ]; then
    case "$B" in 600) mr=2000;; 50) mr=500;; *) mr=200;; esac
  else
    case "$B" in 600) mr=2500;; 50) mr=1500;; *) mr=1200;; esac
  fi

  local niflag="" dtag="iid"
  if [ "$dist" = "noniid" ]; then
    niflag="--non-iid"
    # table-2 non-IID uses a fresh 'ni2' tag (old 'ni' runs used the buggy split)
    [ "$table" = "2" ] && dtag="ni2" || dtag="ni"
  fi

  local trial tag
  if [ "$table" = "2" ]; then
    trial="E=${E},B=${B}"                       # C defaults to 0.1
    tag="t2_${dtag}_E${E}_B${Bs}"
  else
    local Cs="${C/./}"                           # 0.0->00 0.2->02 0.5->05 1.0->10
    trial="E=${E},B=${B},C=${C}"
    tag="t1_${dtag}_B${Bs}_C${Cs}"
  fi

  # keep non-default models in their own namespace (cnn_*) so results don't collide
  [ "$MODEL" != "twonn" ] && tag="${MODEL}_${tag}"

  # skip if this config already produced results (portable glob check)
  local existing=( results/"${tag}"/lr_sweep_*.json )
  if [ -e "${existing[0]}" ]; then
    echo "skip (done): ${tag}"
    return
  fi

  JOBS+=("python experiment.py --model ${MODEL} --target ${TARGET} --trial ${trial} --lr-sweep ${grid} --max-rounds ${mr} ${niflag} --output results/${tag} > logs/${tag}.log 2>&1 && echo \"  done: ${tag}\" || echo \"  FAILED: ${tag} (see logs/${tag}.log)\"")
}

# ---- Table 2: increasing computation (C=0.1), vary E and B ----
for dist in iid noniid; do
  for E in 1 5 20; do
    for B in 600 50 10; do
      add_job 2 "$E" "$B" 0.1 "$dist"
    done
  done
done

# ---- Table 1: increasing parallelism (E=1), vary C (C=0.1 omitted; = Table 2) ----
for dist in iid noniid; do
  for C in 0.0 0.2; do            add_job 1 1 600 "$C" "$dist"; done   # B=inf
  for C in 0.0 0.2 0.5 1.0; do    add_job 1 1 10  "$C" "$dist"; done   # B=10
done

total="${#JOBS[@]}"
if [ "$total" -eq 0 ]; then
  echo "Nothing to run — all configs already have results."
  exit 0
fi

echo "Model=${MODEL}  target=${TARGET}  — launching ${total} jobs, ${CONCURRENCY} at a time..."
echo

# run the queue with a concurrency cap; each item is a full shell command
printf '%s\0' "${JOBS[@]}" \
  | xargs -0 -P "$CONCURRENCY" -n1 bash -c 'eval "$1"' _

echo
echo "All done. Per-config logs are in logs/, results in results/."
echo "Re-run this script to retry any FAILED configs (completed ones are skipped)."
