#!/usr/bin/env bash
# Extended-06 Track B metered dev smoke (run under metered.sh from the repo root): unit costs for the job plan.
#   labels train part (B0 + replacement-matched B2) at N=384; B0 L1 and LEARNED-fuse training for a few updates;
#   small eval label part (dev/test 0, holds 16, hist 0); eval with episode rows on the holds.
set -eu
PY=${PY:-$HOME/structured-latent-dynamics-campaign03/env/bin/python}
T=research/tools/campaign04_probeworld_train.py
D=${D:-dev-smoke}
rm -rf "$D"; mkdir -p "$D"
t() { /usr/bin/time -f "[time] $1 cpu=%U+%S s maxrss=%M KB" "${@:2}"; }
t labels_train_B0_B2 $PY $T labels --split-set b6 --parts train --arms B0 B2 --n-train 384 --out $D/labels 2>&1 | grep -v '"stream"' | tail -4
t labels_eval_holds16 $PY $T labels --split-set b6 --parts eval --n-eval 0 --n-hold 16 --n-hist 0 --out $D/labels 2>&1 | tail -3
t train_B0_150 $PY $T train --labels $D/labels --train-split b6_B0 --rung L1 --seed 30 --updates 150 --log-every 50 --out $D/b0 2>&1 | tail -1
t train_LRN_100 $PY $T train --labels $D/labels --train-split b6_B0 --rung L1 --seed 30 --arch fuse --factor-mode learned --updates 100 --log-every 50 --out $D/lrn 2>&1 | tail -1
t train_SUP_100 $PY $T train --labels $D/labels --train-split b6_B0 --rung L1 --seed 30 --inputs factors6 --arch fuse --updates 100 --log-every 50 --out $D/sup 2>&1 | tail -1
t eval_B0_holds $PY $T eval --labels $D/labels --run $D/b0 --split-set b6 --splits b6_hold_SCE b6_hold_UCE --episode-rows --worlds 1 --world-offset 0 2>&1 | tail -2
$PY -c "import json;[print(r, json.load(open('$D/'+r+'/train_meta.json'))['params'], round(json.load(open('$D/'+r+'/train_meta.json'))['cpu_s_total'],1)) for r in ('b0','lrn','sup')]"
$PY -c "import json;m=json.load(open('$D/labels/labels_meta.train.json'));print({k:(v['n_configs'],v['states_total'],v['report'].get('unmatched_first_class')) for k,v in m.items() if k.startswith('b6_')}, m['train_cpu_s'], m['classified_configs'])"
