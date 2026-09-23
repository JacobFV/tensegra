#!/usr/bin/env bash
set -euo pipefail
seed="${1:?specify frozen confirmation seed}"
case "$seed" in 401|402|403) ;; *) exit 2;; esac
export PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p /tmp/campaign-a05-confirmation
for mode in soft hard context none; do
  /usr/bin/time -f 'PROCESS_OCCUPANCY_SECONDS=%e' "$HOME/topoformer-stage8-cuda/bin/python" -m topoformer.campaign_attention_selector_study --config "configs/campaign-a05-$mode-$seed.json" --output "/tmp/campaign-a05-confirmation/$mode-$seed" > "/tmp/campaign-a05-confirmation/$mode-$seed.log" 2>&1
done
