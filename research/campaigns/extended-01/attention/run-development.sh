#!/usr/bin/env bash
set -euo pipefail
# Root must explicitly release GPU slot before this script is invoked.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
base=/tmp/campaign-a01-results/development-53aa9f6
mkdir -p "$base"
for mode in soft context message hard none; do
  /usr/bin/time -f "PROCESS_OCCUPANCY_SECONDS=%e" \
    "$HOME/topoformer-stage8-cuda/bin/python" -m topoformer.campaign_attention_study \
    --config "configs/campaign-a01-dev-${mode}.json" --output "$base/$mode" \
    > "$base/${mode}.log" 2>&1
done
