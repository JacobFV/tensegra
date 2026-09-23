#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p /tmp/campaign-a05-v2-results
for mode in soft hard context none; do
  /usr/bin/time -f 'PROCESS_OCCUPANCY_SECONDS=%e' "$HOME/topoformer-stage8-cuda/bin/python" -m topoformer.campaign_attention_selector_study --config "configs/campaign-a05-v2-$mode.json" --output "/tmp/campaign-a05-v2-results/$mode" > "/tmp/campaign-a05-v2-results/$mode.log" 2>&1
done
