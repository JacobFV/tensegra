"""Extended-07 Phase 1: print the diagnostic job matrix as root launch commands (campaign07_remote.py launch-cmd),
each validated with campaign07_remote's own parser and output-path guards (nothing is launched here).

  python research/tools/campaign07_diag_jobs.py SNAPSHOT_SHA

Order: support-* (8) -> dev-s30 (one-job smoke of the run type) -> dev-s31/32 -> score-dev (fix tolerances) ->
b6c-s35 (smoke; --check-historical must report 'identical' for all four models) -> b6c-s36..39 -> score-b6c-*.
"""
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign07_remote as rem  # noqa: E402

SHA = sys.argv[1]
R6 = "/home/brand/structured-latent-dynamics-campaign06/results"
R7 = "/home/brand/structured-latent-dynamics-campaign07/results"
PY = "/home/brand/structured-latent-dynamics-campaign03/env/bin/python"
TBC = f"{R6}/e06-tbc-labels/labels"
TB = f"{R6}/e06-tb-labels/labels"
UCE = f"{R6}/e06-tb-labels-uce/labels/cf_UCE.json"
ENV = ["env", "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=1", "PYTHONPATH=src", PY]
IVS = ("none pred exact iso:G1 iso:G2 iso:G3 iso:G4 dep:G1 dep:G2 dep:G3 dep:G4 up:G3 up:G4 "
       "keep:G1 keep:G2 keep:G3 keep:G4 gauss:1 gauss_pred:1 scale_err:0.5 mirror").split()
ROLL = "exact dep:G1 iso:G3 dep:G3".split()
out = []


def launch(job, caps, needs, mkdir, cmd):
    argv = [job, SHA, "--wall-cap", str(caps[0]), "--cpu-cap", str(caps[1])]
    for n in needs:
        argv += ["--needs", n]
    for m in mkdir:
        argv += ["--mkdir", m]
    argv += ["--"] + ENV + cmd
    a = rem.parse_launch(argv)
    assert a.job.startswith("e07-")
    rem.check_paths_in_cmd(a.cmd)
    for d in a.mkdir:
        assert d.startswith(f"{rem.ROOT}/results/")
    out.append("python research/tools/campaign07_remote.py launch-cmd " + shlex.join(argv))


def run_dir(arm, s):
    return f"{R6}/e06-tb-{arm}-s{s}/run"


def sup_file(s):
    return f"{R7}/e07-diag-support-s{s}/diag-v1-support-LRN-s{s}.json"


# stage 1: support references (LRN s30-32 dev lineages, s35-39)
for s in (35, 36, 37, 38, 39, 30, 31, 32):
    o = f"{R7}/e07-diag-support-s{s}"
    launch(f"e07-diag-support-s{s}", (900, 600), [f"{run_dir('lrn', s)}/model.pt", f"{TB}/b6_B0.pkl"], [o],
           ["research/tools/campaign07_diag.py", "support", "--labels", TB, "--pool", "b6_B0",
            "--model", f"LRN-s{s}={run_dir('lrn', s)}", "--n-configs", "384", "--out", o])


def models(s):
    m, needs = [], []
    for arm, name in (("lrn", "LRN"), ("rawf", "RAWF"), ("sup", "SUP"), ("b0", "B0")):
        m += ["--model", f"{name}-s{s}={run_dir(arm, s)}"]
        needs.append(f"{run_dir(arm, s)}/model.pt")
    return m, needs


# stage 2 (dev lineages s30-32, development only: fixes tolerances / sanity before s35-39): b6_hold_SCE + b6 SCE octets
for s in (30, 31, 32):
    o = f"{R7}/e07-diag-dev-s{s}"
    m, needs = models(s)
    launch(f"e07-diag-dev-s{s}", (3600, 2400), needs + [f"{TB}/b6_hold_SCE.shards.json", f"{TB}/cf_SCE.json", sup_file(s)],
           [o], ["research/tools/campaign07_diag.py", "run", "--labels", TB, "--pool", "b6_hold_SCE",
                 "--cf", f"{TB}/cf_SCE.json", *m, "--protocols", "B", "A-pistar", "A-own", "--ivs", *IVS,
                 "--rollout", *ROLL, "--support-ref", f"LRN-s{s}={sup_file(s)}", "--check-historical",
                 "--tag", f"dev-s{s}", "--out", o])

# stage 3: main b6c tranche, s35-39 (b6c_hold_SCE 400 configs + fresh SCE octets 320 + UCE octets 160)
for s in (35, 36, 37, 38, 39):
    o = f"{R7}/e07-diag-b6c-s{s}"
    m, needs = models(s)
    launch(f"e07-diag-b6c-s{s}", (3600, 2400),
           needs + [f"{TBC}/b6c_hold_SCE.shards.json", f"{TBC}/cf_SCE.json", UCE, sup_file(s)], [o],
           ["research/tools/campaign07_diag.py", "run", "--labels", TBC, "--pool", "b6c_hold_SCE",
            "--cf", f"{TBC}/cf_SCE.json", "--cf", UCE, *m, "--protocols", "B", "A-pistar", "A-own", "--ivs", *IVS,
            "--rollout", *ROLL, "--support-ref", f"LRN-s{s}={sup_file(s)}", "--check-historical",
            "--tag", f"b6c-s{s}", "--out", o])

# stage 4: scoring (n_boot 20000, fixed seed; semantic tolerances); dev first, b6c after tolerances are frozen
CON = ["--contrast", "LRN", "RAWF", "--contrast", "SUP", "RAWF", "--contrast", "LRN", "SUP", "--contrast", "RAWF", "B0"]
for job, seeds, stem, inc in (("e07-diag-score-dev", (30, 31, 32), "dev", ["diag-v1-*.jsonl.gz"]),
                              ("e07-diag-score-b6c-pool", (35, 36, 37, 38, 39), "b6c", ["diag-v1-[ABR]-*.jsonl.gz"]),
                              ("e07-diag-score-b6c-cf", (35, 36, 37, 38, 39), "b6c", ["diag-v1-cf-*.jsonl.gz"])):
    o = f"{R7}/{job}"
    dirs = [f"{R7}/e07-diag-{stem}-s{s}" for s in seeds]
    sups = [f"LRN-s{s}={sup_file(s)}" for s in seeds]
    launch(job, (3600, 2400), [f"{d}/diag-v1-summary-{stem}-s{s}.json" for d, s in zip(dirs, seeds)], [o],
           ["research/tools/campaign07_diagscore.py", "--dirs", *dirs, "--include", *inc, "--support-ref", *sups, *CON,
            "--n-boot", "20000", "--boot-seed", "7", "--out", f"{o}/diagscore-v1.json", "--md", f"{o}/diagscore-v1.md"])

print("\n".join(out))
