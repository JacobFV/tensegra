"""Extra audit readings: F1/F1b paired effects on stalled episodes; B2 pooled commit-level AUROC. Prints JSON."""
import gzip, glob, json, math, os, random, sys

R = os.path.expanduser("~/tensegra-campaign04/results")
LIN = ["f-boot-x1-r3", "f-boot-x1-r4", "f-boot-x1-r5"]
out = {"F": {}, "B2": {}}


def load(p):
    d = {}
    with gzip.open(p, "rt") as f:
        for l in f:
            r = json.loads(l)
            o = r["outcome"]
            d[r["seed"]] = (o["utility"], bool(o["verified_success"]), o["cost"], o["steps"], r["progress"]["no_progress"],
                            bool(r.get("truncated")))
    return d


for exp in ("f1", "f1b"):
    for lin in LIN:
        diffs = []; np_eps = 0; flips_up = flips_down = 0; du_np = []; steps_g = steps_m = 0; cost_d = []
        for cond in ("iid_f0", "iid_f2"):
            G = load(glob.glob(f"{R}/{exp}-{lin}/{cond}-greedy/f-boot*.jsonl.gz")[0])
            M = load(glob.glob(f"{R}/{exp}-{lin}/{cond}-r_mask/f-boot*.jsonl.gz")[0])
            for s in G:
                g, m = G[s], M[s]
                diffs.append(m[0] - g[0]); cost_d.append(m[2] - g[2]); steps_g += g[3]; steps_m += m[3]
                if g[4] > 0:
                    np_eps += 1; du_np.append(m[0] - g[0])
                    flips_up += (m[1] and not g[1]); flips_down += (g[1] and not m[1])
        n = len(diffs); mu = sum(diffs) / n
        sd = math.sqrt(sum((x - mu) ** 2 for x in diffs) / (n - 1))
        out["F"][f"{exp}/{lin}"] = {"n": n, "dU": mu, "dU_se": sd / math.sqrt(n), "dU_ci95": [mu - 1.96 * sd / math.sqrt(n), mu + 1.96 * sd / math.sqrt(n)],
                                   "nonzero_dU_eps": sum(abs(x) > 1e-12 for x in diffs), "np_eps": np_eps,
                                   "mean_dU_on_np_eps": sum(du_np) / max(len(du_np), 1),
                                   "success_rescued": flips_up, "success_lost": flips_down,
                                   "d_cost": sum(cost_d) / n, "steps_per_ep_greedy": steps_g / n, "steps_per_ep_rmask": steps_m / n}


def auroc(pos, neg):
    if not pos or not neg:
        return None
    c = 0.0
    for p in pos:
        for q in neg:
            c += 1.0 if p > q else 0.5 if p == q else 0.0
    return c / (len(pos) * len(neg))


rng = random.Random(20260926)
for rung in ("L4", "L1"):
    for s in (0, 1, 2):
        fr = json.load(open(f"{R}/b2-train-{rung}-s{s}/run/failure_records.json"))
        recs = []
        for sp in ("heldout_price", "heldout_k", "heldout_comp"):
            recs += fr[sp]["free_running_greedy"]["commit"]
        pos = [-r["v_own"] for r in recs if r["wrong"]]
        neg = [-r["v_own"] for r in recs if not r["wrong"]]
        a = auroc(pos, neg)
        bs = []
        for _ in range(1000):
            bp = [rng.choice(pos) for _ in pos]; bn = [rng.choice(neg) for _ in neg]
            bs.append(auroc(bp, bn))
        bs.sort()
        out["B2"][f"{rung}-s{s}"] = {"n_pos": len(pos), "n_neg": len(neg), "auroc_v_own": a, "ci95": [bs[24], bs[974]],
                                    "keys": sorted(recs[0].keys()) if recs else None}
json.dump(out, sys.stdout, indent=1)
