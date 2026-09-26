#!/usr/bin/env python3
"""P2a auditor: non-tensor checkpoint fields (config, curves, data_hash) of C0 vs P1 X1-rl-r2,
and the per-update C1 anchor curve. Read-only; run with the snapshot env (torch)."""
import json
import sys

import torch

R = "/home/brand/tensegra-campaign03/results/"


def path(run, att):
    r, s = divmod(att, 6)
    return f"{R}{run}/checkpoints/round-{r}-slot-{s}-attempt-{att}.pt"


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if "second" not in k and "wall" not in k and "time" not in k}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def diff(x, y, p="", out=None, limit=40):
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y), key=str):
            if k not in x or k not in y:
                out.append(f"{p}{k}: only in {'P1' if k in y else 'C0'} = {json.dumps(y.get(k, x.get(k)), default=str)[:120]}")
            else:
                diff(x[k], y[k], f"{p}{k}.", out, limit)
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (u, v) in enumerate(zip(x, y)):
            diff(u, v, f"{p}[{i}].", out, limit)
    elif x != y:
        out.append(f"{p[:-1]}: C0={json.dumps(x, default=str)[:120]} P1={json.dumps(y, default=str)[:120]}")
    return out


res = {}
for att in [int(a) for a in sys.argv[2:]]:
    A = torch.load(path("p2a-c0-x1-r2", att), map_location="cpu", weights_only=False)
    B = torch.load(path("p1-rl-x1-r2", att), map_location="cpu", weights_only=False)
    C = torch.load(path("p2a-c1-x1-r2", att), map_location="cpu", weights_only=False)
    r = {"config_diff": diff(A["config"], B["config"]),
         "curves_len": [len(A["curves"]) if hasattr(A["curves"], "__len__") else None, len(B["curves"]) if hasattr(B["curves"], "__len__") else None],
         "curves_diff_timing_stripped": diff(strip(A["curves"]), strip(B["curves"])),
         "resume_history_diff": diff(strip(A["resume_history"]), strip(B["resume_history"]), limit=10),
         "data_hash": [str(A["data_hash"])[:200], str(B["data_hash"])[:200]],
         "c1_config_vs_c0": diff(A["config"], C["config"])}
    # C1 per-update anchor term
    cur = C["curves"]
    sample = cur[-1] if isinstance(cur, list) and cur else cur
    r["c1_curve_last_entry"] = json.dumps(sample, default=str)[:800]
    if isinstance(cur, list):
        ka = [(e.get("objective_parts") or {}).get("kl_to_anchor") for e in cur if isinstance(e, dict)]
        ka = [float(k) for k in ka if k is not None]
        kr = [float((e.get("objective_parts") or {}).get("kl_to_round_start")) for e in cur if isinstance(e, dict)
              and (e.get("objective_parts") or {}).get("kl_to_round_start") is not None]
        r["c1_kl_to_anchor"] = {"n": len(ka), "min": min(ka) if ka else None, "max": max(ka) if ka else None,
                                "mean": sum(ka) / len(ka) if ka else None, "zeros": sum(k == 0 for k in ka),
                                "last60_mean": sum(ka[-60:]) / len(ka[-60:]) if ka else None}
        r["c1_kl_to_round_start_last60_mean"] = sum(kr[-60:]) / len(kr[-60:]) if kr else None
        c0cur = A["curves"]
        k0 = [float((e.get("objective_parts") or {}).get("kl_to_round_start")) for e in c0cur if isinstance(e, dict)
              and (e.get("objective_parts") or {}).get("kl_to_round_start") is not None]
        r["c0_kl_to_round_start_last60_mean"] = sum(k0[-60:]) / len(k0[-60:]) if k0 else None
        r["c0_has_kl_to_anchor"] = any("kl_to_anchor" in json.dumps(e, default=str) for e in c0cur[-60:])
    res[att] = r
    print(att, json.dumps(r, default=str)[:2500], flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
