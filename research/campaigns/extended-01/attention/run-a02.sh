#!/usr/bin/env bash
set -euo pipefail
# Invoke only after root queue release; caller wraps timeout120.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
base=/tmp/campaign-a02-results
mkdir -p "$base"
for name in dev-soft4 dev-soft8 dev-size_adjust frozen-unchanged frozen-override8 frozen-size_adjust; do
  /usr/bin/time -f "PROCESS_OCCUPANCY_SECONDS=%e" \
    "$HOME/topoformer-stage8-cuda/bin/python" -m topoformer.campaign_attention_study \
    --config "configs/campaign-a02-${name}.json" --output "$base/$name" \
    > "$base/${name}.log" 2>&1
done
