"""Quick descriptive summary of Track C label chunks (development aid; no claims).

    python research/tools/campaign04_c_inspect.py DIR [DIR ...]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main(argv=None):
    import torch
    from tensegra.campaign04_meta_train import STRATA, merge
    for d in (argv or sys.argv[1:]):
        files = sorted(Path(d).glob("labels-c*.pt"))
        data = merge([torch.load(f, weights_only=False) for f in files])
        dU = data["pt_dU"]
        s = data["pt_dU_sample"]
        gain = {"mask_top": dU[:, 1] - dU[:, 0], "stop": dU[:, 2] - dU[:, 0],
                "sample": s.nanmean(1) - dU[:, 0]}
        out = {"dir": d, "episodes": len(data["ep_seed"]), "points": len(dU),
               "success": float(data["ep_success"].float().mean()),
               "default_check": [int((data["pt_check"] == 1).sum()), int((data["pt_check"] >= 0).sum())]}
        for k, g in gain.items():
            ok = ~torch.isnan(g)
            out[f"gain_{k}"] = {"mean": float(g[ok].mean()) if ok.any() else None,
                                "positive_frac": float((g[ok] > 0).float().mean()) if ok.any() else None,
                                "max": float(g[ok].max()) if ok.any() else None}
        out["by_stratum"] = {}
        for i, name in enumerate(STRATA):
            sel = data["pt_stratum"] == i
            if sel.any():
                out["by_stratum"][name] = {"n": int(sel.sum()),
                                           **{k: float(g[sel][~torch.isnan(g[sel])].mean())
                                              if (~torch.isnan(g[sel])).any() else None for k, g in gain.items()}}
        out["sample_draw_var_mean"] = float(s.var(1).nanmean()) if s.shape[1] > 1 else None
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
