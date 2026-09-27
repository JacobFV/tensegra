#!/usr/bin/env bash
# Extended-05 Track B dev smoke (metered via metered.sh): per-update training cost of B0/BX2/BX3/BX1 and eval cost with
# --episode-rows on reduced b5 pools.  Not a result.  Usage: campaign05_trackb_smoke.sh OUTDIR [UPDATES]
set -eu
O="$1"; U="${2:-100}"
PY="${PY:-$HOME/structured-latent-dynamics-campaign03/env/bin/python}"
T=research/tools/campaign04_probeworld_train.py
mkdir -p "$O"
$PY $T labels --split-set b5 --out "$O/labels" --n-train 32 --n-train-x 32 --n-eval 16 --hold-target 4 \
  --hold-target-sensitive 1 --hold-shard 64
for arm in B0 BX1 BX2 BX3; do
  case $arm in
    B0) extra="--train-split b5_train" ;;
    BX1) extra="--train-split b5x_train" ;;
    BX2) extra="--train-split b5_train --inputs bx2" ;;
    BX3) extra="--train-split b5_train --arch modular" ;;
  esac
  $PY $T train --labels "$O/labels" --out "$O/$arm" --rung L1 --seed 10 --updates "$U" --log-every 1000 $extra >/dev/null
  $PY $T eval --labels "$O/labels" --run "$O/$arm" --split-set b5 --episode-rows --world-offset 900 \
    --split-worlds b5_hold_uc=1 b5_hold_se=1 >/dev/null
  $PY - "$O/$arm" <<'EOF'
import json, sys, pathlib
r = pathlib.Path(sys.argv[1])
m = json.loads((r / "train_meta.json").read_text()); e = json.loads((r / "eval.json").read_text())
print(r.name, "params", m["params"], "hidden", m["hidden"], "train_cpu_s", round(m["cpu_s_total"], 1),
      "per_update", round((m["cpu_s_total"] - m["cpu_s_label_load"]) / m["updates"], 4), "eval_cpu_s", round(e["cpu_s"], 1),
      "episodes", {s: v["free_running_greedy"]["summary"]["n"] for s, v in e["splits"].items()})
EOF
done
cat "$O/labels/labels_meta.json" | $PY -c "import json,sys; m=json.load(sys.stdin); print({k: (v.get('n_configs'), v.get('cpu_s')) for k, v in m.items() if isinstance(v, dict) and 'n_configs' in v})"
