"""Golden headroom run for the bit-identity test of the extended-06 portfolio tool.

`python tests/campaign06_golden.py OUT.json` (run with the tool at the base commit de28b9f7,
before the A-CF-SMALL extension) wrote tests/fixtures/campaign06_headroom_golden.json.
tests/test_campaign06_confirm.py re-runs the same thing with the current tool and requires
exact equality (JSON round-trips float64 exactly)."""
import argparse, gzip, importlib.util, json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GOLDEN_SEEDS = range(2_300_000_000 + 100, 2_300_000_000 + 112)   # ext06 dev/smoke range (tests only)


def load_tool():
    spec = importlib.util.spec_from_file_location("c6p_golden", REPO / "research/tools/campaign06_portfolio.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["c6p_golden"] = mod   # picklable for the fold pool
    spec.loader.exec_module(mod)
    return mod


def write_records(pw, path):
    with gzip.open(path, "wt") as f:
        for s in GOLDEN_SEEDS:
            for r in pw.evaluate_episode(s, version="pw-v3"):
                f.write(json.dumps(r, separators=(",", ":")) + "\n")


def headroom_small(mod, rec_path, tmp):
    tmp = Path(tmp)
    mod.G.clear()
    mod.cmd_headroom(argparse.Namespace(inputs=[str(rec_path)], out=str(tmp / "hr"), folds=2, boot=50, inner=2, procs=1))
    s = json.loads((tmp / "hr/headroom.json").read_text())
    s.pop("cpu_s_main")
    z = np.load(tmp / "hr/per_instance.npz")
    return {"summary": s, "per_instance": {k: z[k].tolist() for k in sorted(z.files)}}


if __name__ == "__main__":
    import tempfile
    mod = load_tool()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "rec.jsonl.gz"
        write_records(mod.pw, p)
        out = headroom_small(mod, p, d)
    Path(sys.argv[1]).write_text(json.dumps(out))
    print("wrote", sys.argv[1], len(out["per_instance"]["eps"]), "instances")
