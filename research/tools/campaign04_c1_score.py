"""Score protocol-C1 (registered 2026-09-26T20:30Z) C-H1..C-H5.

Reads the Track C analysis table (c-analysis.json, IID group = mean of iid_f0/iid_f2) and, for the
C-H4 correlation clause, the raw per-episode branch points in c-eval-<base>-iid_f*/<cond>/appraisal_only-s0.jsonl.gz.
Usage: campaign04_c1_score.py <c-analysis.json> <results dir> [--out file]
"""
import gzip, json, random, sys
from pathlib import Path

U = ("sample", "mask_top", "stop")


def iid(table, arm, seed=None, key="utility"):
    rows = [r for r in table if r["arm"] == arm and r["condition"] in ("iid_f0", "iid_f2") and r["examples"] >= 256
            and (seed is None or r["meta_seed"] == seed)]
    by = {}
    for r in rows:
        by.setdefault(r["meta_seed"], []).append(r[key])
    vals = [sum(v) / len(v) for v in by.values() if len(v) == 2]
    return sum(vals) / len(vals) if vals else None


def iid128(table, arm, key="utility"):
    rows = [r for r in table if r["arm"] == arm and r["condition"] in ("iid_f0", "iid_f2") and r["examples"] == 128]
    return sum(r[key] for r in rows) / len(rows) if len(rows) == 2 else None


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
            for k in range(i, j + 1): r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y); n = len(x); mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** .5
    return num / den if den else 0.0


def corr_ci(pairs, B=400, seed=0):
    rng = random.Random(seed); x = [p[0] for p in pairs]; y = [p[1] for p in pairs]; s = spearman(x, y); bs = []
    for _ in range(B):
        idx = [rng.randrange(len(pairs)) for _ in pairs]; bs.append(spearman([x[i] for i in idx], [y[i] for i in idx]))
    bs.sort(); return s, bs[int(.025 * B)], bs[int(.975 * B) - 1], len(pairs)


def main():
    an = json.load(open(sys.argv[1])); res_dir = Path(sys.argv[2]); out = {"bases": {}, "hypotheses": {}}
    for base, b in an["bases"].items():
        t = b["table"]; fixed = {f: iid(t, f"fixed:{f}") for f in ("greedy", "sampled", "r_mask", "r_sample")}
        rec = {"type": b["base_type"], "fixed": fixed, "learned": iid(t, "learned"),
               "learned_per_seed": [iid(t, "learned", s) for s in (0, 1, 2)],
               "random_matched": iid(t, "random_matched"), "threshold": iid(t, "threshold"),
               "appraisal_only": iid(t, "appraisal_only"),
               "default_success": iid(t, "fixed:r_mask", key="success"), "learned_success": iid(t, "learned", key="success"),
               "default_cost": iid(t, "fixed:r_mask", key="cost"), "learned_cost": iid(t, "learned", key="cost"),
               "default_np": iid(t, "fixed:r_mask", key="no_progress_episode_rate"), "learned_np": iid(t, "learned", key="no_progress_episode_rate"),
               "learned_intervention_rate": iid(t, "learned", key="intervention_rate"),
               "shuffled_128": iid128(t, "model:shuffled"), "default_128": None}
        # causal comparison on the same 128-world prefix: learned s0 and default at 128 are not run; use shuffled vs learned-s0/default at 256 (disclosed)
        cal = an["calibration"][base]
        eces = [cal[f"{c}/appraisal_only/s0"]["ece_steps"] for c in ("iid_f0", "iid_f2")]
        rec["ece_iid"] = sum(eces) / 2
        pairs = {u: [] for u in U}
        for c in ("iid_f0", "iid_f2"):
            for l in gzip.open(res_dir / f"c-eval-{base}-{c}/{c}/appraisal_only-s0.jsonl.gz", "rt"):
                for p in json.loads(l).get("branch_points") or []:
                    for i, u in enumerate(U):
                        if u in p["actual_gain"]:
                            pairs[u].append((p["pred"]["adv"][i], p["actual_gain"][u]))
        rec["spearman_pooled"] = corr_ci([q for u in U for q in pairs[u]])
        rec["spearman_by_u"] = {u: corr_ci(pairs[u]) for u in U}
        out["bases"][base] = rec
    L = {k: v for k, v in out["bases"].items() if v["type"] == "loop_prone"}; C = {k: v for k, v in out["bases"].items() if v["type"] == "competent"}
    h1 = {k: v["learned"] >= max(v["fixed"].values()) + .02 and v["learned_success"] >= v["default_success"] - .02 for k, v in L.items()}
    h2 = {k: v["learned"] >= v["fixed"]["r_mask"] - .01 and (v["learned_cost"] < v["default_cost"] or v["learned_np"] < v["default_np"]) for k, v in C.items()}
    h3 = {t: {k: v["learned"] > v["random_matched"] and v["learned"] > v["threshold"] for k, v in d.items()} for t, d in (("loop_prone", L), ("competent", C))}
    h4 = {t: {k: v["ece_iid"] <= .05 and v["spearman_pooled"][1] > 0 for k, v in d.items()} for t, d in (("loop_prone", L), ("competent", C))}
    h5 = {k: {"learned_minus_default": v["learned"] - v["fixed"]["r_mask"], "shuffled128_minus_default256": (v["shuffled_128"] or 0) - v["fixed"]["r_mask"],
              "appraisal_only_minus_default": v["appraisal_only"] - v["fixed"]["r_mask"]} for k, v in out["bases"].items()}
    out["hypotheses"] = {"C-H1": {"lineages": h1, "supported": sum(h1.values()) >= 2},
                         "C-H2": {"lineages": h2, "supported": sum(h2.values()) >= 2},
                         "C-H3": {t: {"lineages": d, "supported": sum(d.values()) >= 2} for t, d in h3.items()},
                         "C-H4": {t: {"lineages": d, "supported": sum(d.values()) >= 2} for t, d in h4.items()},
                         "C-H5": {"per_base": h5, "note": "shuffled evaluated on 128-world causal tier vs default at 256 (same first 128 worlds are a subset); C-H5 is evaluable only where the learned gain is positive"}}
    s = json.dumps(out, indent=1)
    if "--out" in sys.argv: Path(sys.argv[sys.argv.index("--out") + 1]).write_text(s)
    for k, v in out["bases"].items():
        print(k, v["type"], "learned %.4f" % v["learned"], {f: round(x, 4) for f, x in v["fixed"].items()}, "rand %.4f thr %.4f" % (v["random_matched"], v["threshold"]),
              "irate %.3f" % v["learned_intervention_rate"], "ece %.3f" % v["ece_iid"], "rho pooled %.2f [%.2f,%.2f] n=%d" % v["spearman_pooled"],
              {u: "%.2f[%.2f,%.2f]" % x[:3] for u, x in v["spearman_by_u"].items()})
    print(json.dumps({k: (v.get("supported") if "supported" in v else {t: x["supported"] for t, x in v.items() if isinstance(x, dict) and "supported" in x}) for k, v in out["hypotheses"].items()}))


if __name__ == "__main__":
    main()
