#!/usr/bin/env python3
"""Independent audit (Track B): model-variation control for B-LOC's 'belief-dependent' class.

For every first consequential error in bloc-final/records.jsonl.gz, drive (a) the 3 oracle-belief BO models and
(b) every OTHER public-input model of the same rung family (extended-04 L1 seeds 0-5, excluding the erring model
itself) along the erring model's visible history up to the error, and record whether each model's greedy argmax
is eps-optimal there.  If public-input models of other seeds 'fix' the error as often as BO does, the BO
diagnostic does not isolate belief.  Read-only; CPU; uses the trainer's replay_history (public inputs rebuilt
from the visible history).  Run from a source snapshot with PYTHONPATH=src.
Usage: campaign05_b_audit_bloc_control.py RECORDS OUT_JSON --bo RUN RUN RUN --l1 RUN ... --l4 RUN ...
"""
import gzip
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "research/tools")
import campaign04_probeworld_train as T  # noqa: E402
from tensegra import campaign04_probeworld as pw  # noqa: E402


def load(run):
    meta = json.loads((Path(run) / "train_meta.json").read_text())
    m = T.ProbeNet(meta.get("hidden", T.HIDDEN), inputs=meta.get("inputs") or "public",
                   arch=meta.get("arch") or "flat")
    m.load_state_dict(torch.load(Path(run) / "model.pt"))
    m.eval()
    return m


def main():
    torch.set_num_threads(1)
    t0 = time.process_time()
    args = sys.argv[1:]
    rec_path, out = args[0], args[1]
    groups, cur = {}, None
    for x in args[2:]:
        if x.startswith("--"):
            cur = x[2:]
            groups[cur] = []
        else:
            groups[cur].append(x)
    name = lambda r: Path(r).parent.name  # noqa: E731
    models = {name(r): load(r) for g in groups.values() for r in g}
    rows = []
    with gzip.open(rec_path, "rt") as f:
        for line in f:
            r = json.loads(line)
            fe = r.get("first_error")
            if not fe:
                continue
            t = fe["t"]
            cfg = pw.split_config("heldout_comp", r["cfg_idx"])
            hist = [tuple(d["rec"]) for d in r["decisions"][:t]]
            d = r["decisions"][t]
            res = {}
            for g, runs in groups.items():
                for run in runs:
                    nm = name(run)
                    if nm == r["model"]:
                        continue
                    av, logits, _ = T.replay_history(models[nm], cfg, hist, upto=t + 1)[t]
                    assert list(av) == d["avail"]
                    a = int(torch.argmax(logits))
                    res[nm] = [a, a in d["opt"]]
            rows.append({"model": r["model"], "cfg_idx": r["cfg_idx"], "world_seed": r["world_seed"], "t": t,
                         "class": fe["class"], "bo_argmax": fe["bo_argmax"], "ok": res})
    Path(out).write_text(json.dumps({"rows": rows, "cpu_s": time.process_time() - t0}))
    print(len(rows), "cpu", time.process_time() - t0)


if __name__ == "__main__":
    main()
