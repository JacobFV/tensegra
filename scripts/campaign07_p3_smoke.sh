#!/usr/bin/env bash
# Extended-07 P3 consumer-comparison smoke (dev only): every new batch type once at tiny scale on the pro6000, each
# step metered (metered.sh e07-dev-p3-smoke-<step>).  Usage: scripts/campaign07_p3_smoke.sh CODE_DIR OUT_DIR [steps...]
# Seed 47 (dev).  Read-only inputs: the P2 smoke bank / oof / full predictor / plain consumers (results/dev/p2-smoke3,
# seed 47, 20 bank updates, batch 64); outputs under OUT_DIR (results/dev).  b6d step: 4 configurations + 2 octets,
# only exit status / cost are read (the diag prints file counts only; the confirmation population is not scored).
set -u
CODE="$1"; OUT="$2"; shift 2
STEPS="${*:-cost cons evalc evald score}"
R6=$HOME/structured-latent-dynamics-campaign06/results
R7=$HOME/structured-latent-dynamics-campaign07/results
TB=$R6/e06-tb-labels/labels
TBC=$R6/e06-tbc-labels/labels
UCE=$R6/e06-tb-labels-uce/labels/cf_UCE.json
B6D=$R7/e07-p2c-labels/labels
SM=$R7/dev/p2-smoke3
PY=$HOME/structured-latent-dynamics-campaign03/env/bin/python
MET=$HOME/structured-latent-dynamics-campaign07/bin/metered.sh
S=47
BU=${BU:-20}
mkdir -p "$OUT"
run() { tag="$1"; shift; TAIL=${TAIL:-4} "$MET" "e07-dev-p3-smoke-$tag" "$CODE" env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src "$PY" "$@"; }
TR="research/tools/campaign04_probeworld_train.py"
DG="research/tools/campaign07_diag.py"
COMMON="--labels $TB --train-split b6_B0 --train-n 384 --rung L1 --batch 64 --seed $S --log-every 10"
models() {  # $1 = contract: plain CONS (P2 smoke) + the three variants, ::exact and ::pred=
  local c=$1
  local M="--model CONS3-lin-$c-exact-s$S=$SM/cons-$c/run::exact --model CONS3-lin-$c-pred-s$S=$SM/cons-$c/run::pred=$SM/pred-full/run"
  for a in mlp bil gate; do
    M="$M --model CONS3-$a-$c-exact-s$S=$OUT/cons-$a-$c/run::exact --model CONS3-$a-$c-pred-s$S=$OUT/cons-$a-$c/run::pred=$SM/pred-full/run"
  done
  echo "$M"
}
TAILA="--protocols B A-pistar --ivs none exact --phis own exact zero mean --phi-mean-bank $SM/bank.pkl --no-latent"
for step in $STEPS; do case $step in
cost) run cost research/tools/campaign07_p2.py cons-cost ;;
cons) for a in mlp bil gate; do for c in exact mix; do extra=""; [ $c != exact ] && extra="--oof $SM/oof.pt"
        run cons-$a-$c $TR train $COMMON --inputs factors6 --arch fuse --bank $SM/bank.pkl --bank-updates $BU \
          --phi-contract $c $extra --cons-arch $a --updates 0 --out $OUT/cons-$a-$c/run; done; done ;;
evalc) run eval-b6c $DG run --labels $TBC --pool b6c_hold_SCE --cf $TBC/cf_SCE.json --cf $UCE $(models mix) $TAILA \
         --n-configs 24 --n-octets 8 --tag p3smoke-mix-s$S --out $OUT/eval-mix ;;
evald) run eval-b6d $DG run --labels $B6D --pool b6d_hold_SCE --cf $B6D/cf_SCE.json $(models exact) $TAILA \
         --n-configs 4 --n-octets 2 --tag p3smoke-b6d-exact-s$S --out $OUT/eval-b6d ;;
score) C=""; for inp in exact pred; do for p in "bil mlp" "bil lin" "mlp lin" "gate lin" "bil gate"; do set -- $p
         for ph in own exact zero mean; do C="$C --contrast CONS3-$1-mix-$inp@$ph CONS3-$2-mix-$inp@$ph"; done; done; done
       run score research/tools/campaign07_diagscore.py --dirs $OUT/eval-mix --by-model-phi --n-boot 2000 \
         --boot-seed 7 $C --out $OUT/score.json --md $OUT/score.md ;;
esac; done
