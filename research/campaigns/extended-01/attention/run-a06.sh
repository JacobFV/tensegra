#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
seed="${1:?paired initialization seed required}"
case "$seed" in 601|602|603) ;; *) exit 2;; esac
mkdir -p "/tmp/campaign-a06-results/$seed"
for mode in soft hard context none; do
  /usr/bin/time -f 'PROCESS_OCCUPANCY_SECONDS=%e' "$HOME/topoformer-stage8-cuda/bin/python" -m topoformer.campaign_attention_selector_study --config "configs/campaign-a06-$mode-$seed.json" --output "/tmp/campaign-a06-results/$seed/$mode" > "/tmp/campaign-a06-results/$seed/$mode.log" 2>&1
done
