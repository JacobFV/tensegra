#!/usr/bin/env bash
set -euo pipefail
# Root releases one paired seed at a time; caller enforces timeout300.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
base=/tmp/campaign-a03-confirmation
mkdir -p "$base"
seed=${1:?Specify one released seed:201,202,203}
case "$seed" in 201|202|203) ;; *) exit 2;; esac
 for mode in soft4 soft8 context message hard; do
  /usr/bin/time -f "PROCESS_OCCUPANCY_SECONDS=%e" \
    "$HOME/topoformer-stage8-cuda/bin/python" -m topoformer.campaign_attention_study \
    --config "configs/campaign-a03-${mode}-${seed}.json" --output "$base/${mode}-${seed}" \
    > "$base/${mode}-${seed}.log" 2>&1
 done
