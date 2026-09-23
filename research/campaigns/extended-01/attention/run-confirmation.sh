#!/usr/bin/env bash
set -euo pipefail
# Root must release this job; caller enforces timeout600.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
base=/tmp/campaign-a03-confirmation
mkdir -p "$base"
for seed in 201 202 203; do
 for mode in soft4 soft8 context message hard; do
  /usr/bin/time -f "PROCESS_OCCUPANCY_SECONDS=%e" \
    "$HOME/topoformer-stage8-cuda/bin/python" -m topoformer.campaign_attention_study \
    --config "configs/campaign-a03-${mode}-${seed}.json" --output "$base/${mode}-${seed}" \
    > "$base/${mode}-${seed}.log" 2>&1
 done
done
