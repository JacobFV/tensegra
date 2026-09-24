"""Emit frozen finalist bindings {name,path,sha256} from completed population runs.

Runs on the host holding results. The finalist is the run's registered finalist
(final-round maximum development utility; single = latest). No other checkpoint
is eligible. Optionally adds bank member 0 per replicate as the no-RL baseline.
"""
import argparse, hashlib, json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--run", action="append", default=[], help="name=run_dir")
    p.add_argument("--bank", action="append", default=[], help="name=bank_run_dir (member 0)")
    a = p.parse_args()
    out = []
    for spec in a.run:
        name, run = spec.split("=")
        state = json.loads((a.results / run / "state.json").read_text())
        if state["status"] != "completed":
            raise SystemExit(f"{run} not completed")
        f = state["finalist"]
        path = a.results / run / f["checkpoint"]
        if sha(path) != f["checkpoint_sha256"]:
            raise SystemExit(f"hash mismatch {path}")
        out.append({"name": name, "path": str(path), "sha256": f["checkpoint_sha256"],
                    "development_utility": f["utility"], "development_success": f["success"], "slot": f["slot"]})
    for spec in a.bank:
        name, run = spec.split("=")
        state = json.loads((a.results / run / "state.json").read_text())
        m = state["members"][0]
        path = a.results / run / m["checkpoint"]
        if sha(path) != m["checkpoint_sha256"]:
            raise SystemExit(f"hash mismatch {path}")
        out.append({"name": name, "path": str(path), "sha256": m["checkpoint_sha256"]})
    print(json.dumps(out, indent=2))
