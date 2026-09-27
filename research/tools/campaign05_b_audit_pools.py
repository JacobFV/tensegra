#!/usr/bin/env python3
"""Independent audit (Track B): dump the configurations actually stored in the label pickles.

Read-only.  For every label pool (b5 split set, b5c split set, extended-04 b-labels) emit, per configuration,
(split, idx, config dict, flags).  Only the Config object of each pool entry is kept; the V/Q tables are dropped
immediately after each pickle is loaded (one pickle at a time, bounded memory).
Usage (run from a source snapshot with PYTHONPATH=src):
  campaign05_b_audit_pools.py OUT_JSON LABELS_DIR [LABELS_DIR ...]
"""
import dataclasses
import gc
import json
import pickle
import sys
from pathlib import Path


def main(out, dirs):
    res = {}
    for d in dirs:
        d = Path(d)
        for p in sorted(d.glob("*.pkl")):
            with open(p, "rb") as f:
                pool = pickle.load(f)
            ent = []
            for item in pool:
                idx, cfg = item[0], item[1]
                cd = dataclasses.asdict(cfg)
                cd = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cd.items()}
                ent.append({"idx": idx, "cfg": cd, "flags": list(cfg.flags), "n_states": len(item[2])})
            res[str(p)] = ent
            del pool
            gc.collect()
            print(p.name, len(ent), flush=True)
    Path(out).write_text(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
