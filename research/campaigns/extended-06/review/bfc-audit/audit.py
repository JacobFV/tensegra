"""Independent B-FACT-C audit analysis (own code; does not import campaign06_bscore / bprimary)."""
import gzip, json, os, sys, time
import numpy as np

B = os.path.dirname(os.path.abspath(__file__)) + "/bfc"
SEEDS = [35, 36, 37, 38, 39]
ARMS = ["lrn", "rawf", "sup", "b0"]
EPS = 0.5
ACT = ["probe", "exact_b1", "exact_b2", "inspect", "prop", "build", "use", "commit", "commit_infeasible", "abstain"]
t0 = time.process_time()


def ctx_of(prev, d):
    if d["step_in_query"] == 0:
        return "query_first"
    a, o = prev["rec"][0], prev["rec"][1]
    if a == 0 and o == 0:
        return "after_probe_solved"
    if a == 0 and o == 1:
        return "after_probe_failed"
    if a == 1 and o == 2:
        return "after_b1_timeout"
    return "other"


CTXS = ["query_first", "after_probe_solved", "after_probe_failed", "after_b1_timeout", "other"]


def load_run(arm, s):
    run = f"{B}/e06-tb-{arm}-s{s}/run"
    with gzip.open(run + "/eval_b6c_episodes.jsonl.gz", "rt") as f:
        head = json.loads(f.readline())["_meta"]
        rows = [json.loads(l) for l in f if l.strip()]
    assert head["seed"] == s
    rows = {(r["split"], r["cfg_idx"], r["rep"]): r for r in rows}
    cf = json.load(open(run + "/cf_eval_b6c.json"))
    return head, rows, cf


runs = {(a, s): load_run(a, s) for a in ARMS for s in SEEDS}
keys = sorted(set.intersection(*[set(r[1]) for r in runs.values()]))
assert len(keys) == 400, len(keys)
# world seeds identical across runs
for k in keys:
    ws = {runs[x][1][k]["world_seed"] for x in runs}
    assert len(ws) == 1

# ---------------------------------------------------------------- per-config arrays
def ep_arrays(rows):
    n = len(keys)
    gap = np.zeros(n); first = np.zeros(n)
    later_n = np.zeros(n); later_k = np.zeros(n)
    ctx_n = {c: np.zeros(n) for c in CTXS}; ctx_k = {c: np.zeros(n) for c in CTXS}
    # later decisions split by whether the model's action class
    build = np.zeros(n); steps = np.zeros(n); first_a = np.zeros(n, int); nprobe = np.zeros(n)
    qf_by_q = {}
    for j, k in enumerate(keys):
        r = rows[k]; ds = r["decisions"]
        gap[j] = r["gap_regret"]
        first[j] = ds[0]["a"] in set(ds[0]["opt"])
        first_a[j] = ds[0]["a"]
        build[j] = r["built"]
        steps[j] = len(ds)
        nprobe[j] = sum(d["a"] == 0 for d in ds)
        for p, d in zip(ds, ds[1:]):
            c = ctx_of(p, d); ok = d["delta"] <= EPS
            later_n[j] += 1; later_k[j] += ok
            ctx_n[c][j] += 1; ctx_k[c][j] += ok
    return dict(gap=gap, first=first, later_n=later_n, later_k=later_k, ctx_n=ctx_n, ctx_k=ctx_k, build=build,
                steps=steps, first_a=first_a, nprobe=nprobe)


EP = {x: ep_arrays(runs[x][1]) for x in runs}

# ---------------------------------------------------------------- octet arrays (per set, per decision type)
FAM = "SCE"


def cf_arrays(cf, htypes=None):
    sets = sorted(cf["families"][FAM], key=lambda s: s["index"])
    n = len(sets)
    out = {k: np.zeros(n) for k in ["flip_n", "flip_k", "nm_n", "nm_k", "fc_n", "fc_k", "oc_n", "oc_k",
                                     "bal_full_k", "nonflip_n", "nonflip_k"]}
    idx = []
    for j, s in enumerate(sets):
        idx.append(s["index"])
        for h, t in s["types"].items():
            if htypes and h not in htypes:
                continue
            mem = t["members"]; full = mem.get(FAM)
            if full is None:
                continue
            flip = t["flip"] and t["unique"]
            if flip:
                out["flip_n"][j] += 1; out["flip_k"][j] += full["ok"]
                if t["near_miss"] is not None:
                    out["nm_n"][j] += 1; out["nm_k"][j] += t["near_miss"]["ok"]
                    out["bal_full_k"][j] += full["ok"]
            else:
                out["nonflip_n"][j] += 1; out["nonflip_k"][j] += full["ok"]
            for key, m in mem.items():
                for f in FAM:
                    if f in key:
                        continue
                    on = "".join(x for x in FAM if x in key or x == f)
                    m2 = mem.get(on)
                    if m2 is None or not (m["unique"] and m2["unique"]):
                        continue
                    if m["opt"] != m2["opt"]:
                        out["oc_n"][j] += 1; out["oc_k"][j] += m["a"] != m2["a"]
                    else:
                        out["fc_n"][j] += 1; out["fc_k"][j] += m["a"] != m2["a"]
    return idx, out


CFA = {}
for x in runs:
    idx, CFA[x] = cf_arrays(runs[x][2])
    assert idx == list(range(320)), idx[:5]
HT = runs[("lrn", 35)][2]["_meta"]["decision_types"]
CFH = {h: {x: cf_arrays(runs[x][2], [h])[1] for x in runs} for h in HT}

# ---------------------------------------------------------------- endpoint functions on weights
def ep_val(e, name, w):
    if name == "gap_regret":
        return (w * e["gap"]).sum(-1) / w.sum(-1)
    if name == "later_acc":
        return (w * e["later_k"]).sum(-1) / (w * e["later_n"]).sum(-1)
    if name == "first_acc":
        return (w * e["first"]).sum(-1) / w.sum(-1)
    if name.startswith("ctx_"):
        c = name[4:]
        return (w * e["ctx_k"][c]).sum(-1) / (w * e["ctx_n"][c]).sum(-1)
    raise KeyError(name)


def cf_val(c, name, w):
    r = lambda a, b: (w * c[a]).sum(-1) / (w * c[b]).sum(-1)
    if name == "cf_near_miss_acc":
        return r("nm_k", "nm_n")
    if name == "iv_false_change":
        return r("fc_k", "fc_n")
    if name == "iv_change_given_opt_change":
        return r("oc_k", "oc_n")
    if name == "cf_flip_full_acc":
        return r("flip_k", "flip_n")
    if name == "cf_balanced_acc":
        return (r("bal_full_k", "nm_n") + r("nm_k", "nm_n")) / 2
    if name == "cf_nonflip_full_acc":
        return r("nonflip_k", "nonflip_n")
    raise KeyError(name)


EP_NAMES = ["gap_regret", "later_acc", "first_acc"]
CF_NAMES = ["cf_near_miss_acc", "iv_false_change", "cf_flip_full_acc", "cf_balanced_acc"]


def value(arm, s, name, w_ep=None, w_cf=None, cfsrc=None):
    if name in EP_NAMES or name.startswith("ctx_"):
        w = np.ones(400) if w_ep is None else w_ep
        return ep_val(EP[(arm, s)], name, w)
    w = np.ones(320) if w_cf is None else w_cf
    return cf_val((cfsrc or CFA)[(arm, s)], name, w)


def two_level(A, R, name, n_boot=1000, seed=0, shared=True, cfsrc=None):
    rng = np.random.default_rng(seed)
    P = len(SEEDS)
    pix = rng.integers(0, P, size=(n_boot, P))
    iscf = name not in EP_NAMES and not name.startswith("ctx_")
    n = 320 if iscf else 400
    if shared:
        W = rng.multinomial(n, [1 / n] * n, size=n_boot).astype(float)  # (n_boot, n)
        diffs = np.stack([value(A, s, name, *( (None, W) if iscf else (W, None)), cfsrc=cfsrc)
                          - value(R, s, name, *((None, W) if iscf else (W, None)), cfsrc=cfsrc) for s in SEEDS], 1)
        boots = np.take_along_axis(diffs, pix, 1).mean(1)
    else:  # independent configuration draw per drawn pair slot
        boots = np.zeros(n_boot)
        for slot in range(P):
            W = rng.multinomial(n, [1 / n] * n, size=n_boot).astype(float)
            d = np.stack([value(A, s, name, *((None, W) if iscf else (W, None)), cfsrc=cfsrc)
                          - value(R, s, name, *((None, W) if iscf else (W, None)), cfsrc=cfsrc) for s in SEEDS], 1)
            boots += d[np.arange(n_boot), pix[:, slot]]
        boots /= P
    boots = boots[np.isfinite(boots)]
    return np.percentile(boots, [2.5, 97.5]), boots


def per_pair(A, R, name, cfsrc=None):
    return np.array([float(value(A, s, name, cfsrc=cfsrc) - value(R, s, name, cfsrc=cfsrc)) for s in SEEDS])


out = {}
print("== point estimates (own code) ==")
for A, R in [("lrn", "rawf"), ("sup", "rawf"), ("lrn", "b0"), ("rawf", "b0"), ("sup", "b0")]:
    for name in EP_NAMES + CF_NAMES + ["iv_change_given_opt_change", "cf_nonflip_full_acc"]:
        pp = per_pair(A, R, name)
        print(f"{A}-{R} {name:28s} mean {pp.mean():+.4f} per pair " + " ".join(f"{x:+.3f}" for x in pp))
        out[f"{A}-{R}:{name}:per_pair"] = pp.tolist()

print("== LRN-RAWF two-level CIs, sensitivity ==")
for name in ["gap_regret", "later_acc", "cf_near_miss_acc", "iv_false_change", "cf_flip_full_acc", "cf_balanced_acc",
             "first_acc"]:
    for nb in (1000, 5000, 20000):
        los, his = [], []
        for sd in range(10 if nb < 20000 else 3):
            ci, _ = two_level("lrn", "rawf", name, nb, seed=sd)
            los.append(ci[0]); his.append(ci[1])
        print(f"{name:20s} n_boot {nb:6d} seeds {len(los)}: lo {min(los):+.4f}..{max(los):+.4f}  hi {min(his):+.4f}..{max(his):+.4f}")
        out[f"ci:{name}:{nb}"] = [min(los), max(los), min(his), max(his)]
    ci, b = two_level("lrn", "rawf", name, 20000, seed=123, shared=False)
    print(f"{name:20s} independent-config-draw-per-slot n_boot 20000: [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    out[f"ci_indep:{name}"] = ci.tolist()

# fraction of bootstrap mass above 0 for gap
_, b = two_level("lrn", "rawf", "gap_regret", 100000, seed=7)
print("gap_regret two-level: P(boot >= 0) =", (b >= 0).mean(), " 97.5% =", np.percentile(b, 97.5))
out["gap_p_ge0"] = float((b >= 0).mean())
# pair-level only: t interval and exact sign test
from math import comb
for name in ["gap_regret", "later_acc"]:
    pp = per_pair("lrn", "rawf", name)
    se = pp.std(ddof=1) / np.sqrt(5)
    t = 2.776
    print(f"{name} across-pair t4 95% CI: [{pp.mean() - t * se:+.4f}, {pp.mean() + t * se:+.4f}]  t={pp.mean() / se:.2f}")
    out[f"t4:{name}"] = [pp.mean() - t * se, pp.mean() + t * se]
print("sign test 5/5: p(one-sided)=", 1 / 32, " 4/5:", 6 / 32)

# ---------------------------------------------------------------- per decision type octet endpoints
print("== octet endpoints by decision type (LRN-RAWF per pair; mean; support LRN s35) ==")
for h in HT:
    for name in ["cf_flip_full_acc", "cf_near_miss_acc", "iv_false_change", "iv_change_given_opt_change"]:
        pp = per_pair("lrn", "rawf", name, cfsrc=CFH[h])
        c = CFH[h][("rawf", 35)]
        supp = {"cf_flip_full_acc": c["flip_n"].sum(), "cf_near_miss_acc": c["nm_n"].sum(),
                "iv_false_change": c["fc_n"].sum(), "iv_change_given_opt_change": c["oc_n"].sum()}[name]
        rm = np.mean([float(value("rawf", s, name, cfsrc=CFH[h])) for s in SEEDS])
        if np.all(np.isfinite(pp)):
            print(f"{h:20s} {name:28s} n={int(supp):4d} RAWF {rm:.3f} diff {np.nanmean(pp):+.4f} (" + " ".join(f"{x:+.3f}" for x in pp) + ")")
        out[f"byh:{h}:{name}"] = [int(supp), rm, pp.tolist()]

# discrimination: P(change|opt change) - P(change|opt same)
for A in ["lrn", "rawf", "sup", "b0"]:
    oc = [float(value(A, s, "iv_change_given_opt_change")) for s in SEEDS]
    fc = [float(value(A, s, "iv_false_change")) for s in SEEDS]
    fl = [float(value(A, s, "cf_flip_full_acc")) for s in SEEDS]
    nm = [float(value(A, s, "cf_near_miss_acc")) for s in SEEDS]
    print(f"{A:5s} P(chg|opt chg) {np.mean(oc):.3f}  P(chg|opt same) {np.mean(fc):.3f}  disc {np.mean(oc) - np.mean(fc):.3f}  flip {np.mean(fl):.3f} nm {np.mean(nm):.3f}")
    out[f"disc:{A}"] = [np.mean(oc), np.mean(fc), np.mean(fl), np.mean(nm)]

# ---------------------------------------------------------------- Q2 by context
print("== later-decision accuracy by context (LRN-RAWF) ==")
for c in CTXS:
    pp = per_pair("lrn", "rawf", "ctx_" + c)
    nl = np.mean([EP[("lrn", s)]["ctx_n"][c].sum() for s in SEEDS]); nr = np.mean([EP[("rawf", s)]["ctx_n"][c].sum() for s in SEEDS])
    rm = np.mean([float(value("rawf", s, "ctx_" + c)) for s in SEEDS])
    print(f"{c:20s} n LRN {nl:7.1f} RAWF {nr:7.1f}  RAWF acc {rm:.3f}  diff {pp.mean():+.4f} (" + " ".join(f"{x:+.3f}" for x in pp) + ")")
    out[f"ctx:{c}"] = [nl, nr, rm, pp.tolist()]
# contribution decomposition: errors per episode by context
print("== later-decision ERRORS per episode by context (LRN - RAWF) ==")
for c in CTXS:
    e = [(EP[("lrn", s)]["ctx_n"][c] - EP[("lrn", s)]["ctx_k"][c]).sum() / 400 - (EP[("rawf", s)]["ctx_n"][c] - EP[("rawf", s)]["ctx_k"][c]).sum() / 400 for s in SEEDS]
    print(f"{c:20s} {np.mean(e):+.4f} (" + " ".join(f"{x:+.3f}" for x in e) + ")")
    out[f"ctxerr:{c}"] = e
# later decisions count per episode
for A in ARMS:
    ln = [EP[(A, s)]["later_n"].sum() / 400 for s in SEEDS]
    print(A, "later decisions/episode", np.round(ln, 2), "mean", np.mean(ln).round(3))

# ---------------------------------------------------------------- behaviour
print("== behaviour ==")
for A in ARMS:
    br = [EP[(A, s)]["build"].mean() for s in SEEDS]
    st = [EP[(A, s)]["steps"].mean() for s in SEEDS]
    npb = [EP[(A, s)]["nprobe"].mean() for s in SEEDS]
    fa = np.mean([np.bincount(EP[(A, s)]["first_a"], minlength=10) / 400 for s in SEEDS], 0)
    print(f"{A:5s} build {np.mean(br):.3f} ({' '.join(f'{x:.2f}' for x in br)}) steps {np.mean(st):.2f} probes/ep {np.mean(npb):.2f} first-action mix " + " ".join(f"{ACT[i]}:{fa[i]:.2f}" for i in range(10) if fa[i] > 0.005))
    out[f"beh:{A}"] = dict(build=br, steps=st, nprobe=npb, first_mix=fa.tolist())
r0 = runs[("rawf", 35)][1][keys[0]]
print("pi* build rate", np.mean([runs[("rawf", 35)][1][k]["pi_star_built"] for k in keys]))

# action mix of LATER decisions and errors by action taken vs optimal
print("== later-decision action mix and error types ==")
for A in ["lrn", "rawf"]:
    mix = np.zeros(10); err_taken = np.zeros(10); err_opt = np.zeros(10); tot = 0
    for s in SEEDS:
        rows = runs[(A, s)][1]
        for k in keys:
            ds = rows[k]["decisions"]
            for d in ds[1:]:
                mix[d["a"]] += 1; tot += 1
                if d["delta"] > EPS:
                    err_taken[d["a"]] += 1
                    err_opt[d["opt"][0]] += 1
    print(A, "later mix", " ".join(f"{ACT[i]}:{mix[i] / tot:.3f}" for i in range(10) if mix[i]), "| errors/5x400eps by action taken", " ".join(f"{ACT[i]}:{err_taken[i] / 2000:.3f}" for i in range(10) if err_taken[i]), "| by optimum", " ".join(f"{ACT[i]}:{err_opt[i] / 2000:.3f}" for i in range(10) if err_opt[i]))
    out[f"latermix:{A}"] = dict(mix=(mix / tot).tolist(), err_taken=(err_taken / 2000).tolist(), err_opt=(err_opt / 2000).tolist())

# gap regret: where is it? split by build decision correctness / pi* build
print("== gap regret by pi* build (LRN-RAWF) ==")
for pb in (0, 1):
    sel = np.array([runs[("rawf", 35)][1][k]["pi_star_built"] == pb for k in keys])
    d = [EP[("lrn", s)]["gap"][sel].mean() - EP[("rawf", s)]["gap"][sel].mean() for s in SEEDS]
    bl = [EP[("lrn", s)]["build"][sel].mean() for s in SEEDS]; brw = [EP[("rawf", s)]["build"][sel].mean() for s in SEEDS]
    print(f"pi*_built={pb} n={sel.sum()} gap diff {np.mean(d):+.3f} ({' '.join(f'{x:+.2f}' for x in d)}) build LRN {np.mean(bl):.3f} RAWF {np.mean(brw):.3f}")
    out[f"gap_by_pibuild:{pb}"] = [int(sel.sum()), d, bl, brw]
# gap per k
ks = np.array([runs[("rawf", 35)][1][k]["k"] for k in keys])
for kk in sorted(set(ks)):
    sel = ks == kk
    d = [EP[("lrn", s)]["gap"][sel].mean() - EP[("rawf", s)]["gap"][sel].mean() for s in SEEDS]
    print(f"k={kk} n={sel.sum()} gap diff {np.mean(d):+.3f} ({' '.join(f'{x:+.2f}' for x in d)})")
    out[f"gap_by_k:{kk}"] = [int(sel.sum()), d]
# concentration: share of total gap diff from top 5% configs
for s in SEEDS:
    dd = EP[("lrn", s)]["gap"] - EP[("rawf", s)]["gap"]
    srt = np.sort(dd)
    print(f"s{s}: total {dd.sum():+.1f}; most-negative 20 configs {srt[:20].sum():+.1f}; most-positive 20 {srt[-20:].sum():+.1f}; median {np.median(dd):+.2f}; frac configs equal {np.mean(dd == 0):.2f}")
pooled = np.mean([EP[("lrn", s)]["gap"] - EP[("rawf", s)]["gap"] for s in SEEDS], 0)
srt = np.argsort(pooled)
print("pooled over pairs: total", pooled.sum().round(1), "top-20 configs", pooled[srt[:20]].sum().round(1), "without top 20 mean", np.delete(pooled, srt[:20]).mean().round(3))
out["gap_conc"] = [float(pooled.sum()), float(pooled[srt[:20]].sum()), float(np.delete(pooled, srt[:20]).mean())]

# ---------------------------------------------------------------- aux quality
print("== aux (train_log) ==")
aux = []
for s in SEEDS:
    L = json.load(open(f"{B}/e06-tb-lrn-s{s}/run/train_log.json"))
    fin = L[-1]["aux"]; last5 = np.mean([x["aux"] for x in L[-5:]])
    aux.append(last5)
    rr = [json.load(open(f"{B}/e06-tb-{a}-s{s}/run/train_log.json"))[-1]["regret"] for a in ("lrn", "rawf")]
    print(f"s{s} aux final {fin:.4f} last5 {last5:.4f} first {L[0]['aux']:.3f}; train-batch regret final LRN {rr[0]:+.2f} RAWF {rr[1]:+.2f}")
gp = per_pair("lrn", "rawf", "gap_regret"); la = per_pair("lrn", "rawf", "later_acc"); fl = per_pair("lrn", "rawf", "cf_flip_full_acc")
print("spearman-ish (rank corr) aux vs gap diff", np.corrcoef(np.argsort(np.argsort(aux)), np.argsort(np.argsort(gp)))[0, 1].round(2),
      "vs later diff", np.corrcoef(np.argsort(np.argsort(aux)), np.argsort(np.argsort(la)))[0, 1].round(2),
      "vs flip diff", np.corrcoef(np.argsort(np.argsort(aux)), np.argsort(np.argsort(fl)))[0, 1].round(2))
out["aux_last5"] = aux
print("cpu s", round(time.process_time() - t0, 1))
json.dump(out, open(os.path.dirname(B) + "/audit_out.json", "w"), indent=1, default=float)
