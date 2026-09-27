#!/usr/bin/env bash
# Extended-07 Phase 2 smoke (dev only): every new batch type once at tiny scale on the pro6000, each step metered
# (metered.sh e07-dev-p2-smoke-<step>).  Usage: scripts/campaign07_p2_smoke.sh CODE_DIR OUT_DIR [steps...]
# Seed 47 (dev), bank worlds 12.95e9 (dev), 24 configurations of b6_B0, batch 64 (per-update costs at protocol scale).
set -u
CODE="$1"; OUT="$2"; shift 2
STEPS="${*:-bank init arms onp pred oof cons eval}"
R6=$HOME/structured-latent-dynamics-campaign06/results
TB=$R6/e06-tb-labels/labels
TBC=$R6/e06-tbc-labels/labels
PY=$HOME/structured-latent-dynamics-campaign03/env/bin/python
MET=$HOME/structured-latent-dynamics-campaign07/bin/metered.sh
S=47
BU=${BU:-20}
mkdir -p "$OUT"
run() { tag="$1"; shift; TAIL=${TAIL:-6} "$MET" "e07-dev-p2-smoke-$tag" "$CODE" env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src "$PY" "$@"; }
TR="research/tools/campaign04_probeworld_train.py"
P2="research/tools/campaign07_p2.py"
COMMON="--labels $TB --train-split b6_B0 --train-n 384 --rung L1 --batch 64 --seed $S --log-every 10"
for step in $STEPS; do case $step in
bank) run bank $P2 bank --labels $TB --pool b6_B0 --n-configs 24 --world-base 12950000000 \
        --source pistar:1 eps:0.2:1 "model:RAWF-s30=$R6/e06-tb-rawf-s30/run:1" --out $OUT/bank.pkl ;;
init) run init $P2 init --seed $S --out $OUT/init-s$S ;;
arms) for arm in "0 0" "1 0" "0 1" "1 1"; do set -- $arm
        run arm-s$1r$2 $TR train $COMMON --arch fuse --shape $1 --read $2 --init-from $OUT/init-s$S/init.pt \
          --bank $OUT/bank.pkl --bank-updates $BU --updates 0 --out $OUT/s$1r$2-bank/run; done ;;
onp)  for arm in "0 0" "1 1"; do set -- $arm
        run onp-s$1r$2 $TR train $COMMON --arch fuse --shape $1 --read $2 --init-from $OUT/init-s$S/init.pt \
          --updates 10 --out $OUT/s$1r$2-onp/run; done ;;
pred) for f in 0 1; do
        run pred-f$f $TR train $COMMON --arch fuse --shape 1 --read 0 --bank $OUT/bank.pkl --bank-updates $BU \
          --predictor-only --bank-folds 2 --bank-exclude-fold $f --updates 0 --out $OUT/pred-f$f/run; done
      run pred-full $TR train $COMMON --arch fuse --shape 1 --read 0 --bank $OUT/bank.pkl --bank-updates $BU \
          --predictor-only --updates 0 --out $OUT/pred-full/run ;;
oof)  run oof $P2 oof --bank $OUT/bank.pkl --fold-run 0=$OUT/pred-f0/run --fold-run 1=$OUT/pred-f1/run \
        --full-run $OUT/pred-full/run --out $OUT/oof.pt ;;
cons) for c in exact oof mix; do extra=""; [ $c != exact ] && extra="--oof $OUT/oof.pt"
        run cons-$c $TR train $COMMON --inputs factors6 --arch fuse --bank $OUT/bank.pkl --bank-updates $BU \
          --phi-contract $c $extra --updates 0 --out $OUT/cons-$c/run; done ;;
eval) M=""
      for a in s0r0 s1r0 s0r1 s1r1; do M="$M --model ${a^^}=$OUT/${a}-bank/run"; done
      for c in exact oof mix; do M="$M --model C${c}-X=$OUT/cons-$c/run::exact --model C${c}-P=$OUT/cons-$c/run::pred=$OUT/pred-full/run"; done
      run eval research/tools/campaign07_diag.py run --labels $TBC --pool b6c_hold_SCE --cf $TBC/cf_SCE.json $M \
        --protocols B A-pistar --ivs none exact --no-latent --n-configs 24 --n-octets 8 --tag smoke --out $OUT/eval ;;
esac; done
