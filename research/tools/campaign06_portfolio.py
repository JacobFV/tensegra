"""Extended-06 Track A (portworld): arm evaluation and the HEADROOM study.

Subcommands
-----------
  evaluate --lo SEED --episodes E --out FILE.jsonl.gz [--knobs JSON]
      Evaluate every arm on every instance of episodes lo..lo+E-1 (resumable: episodes
      already present in FILE are skipped; FILE is appended per episode).
  headroom --inputs FILE... --out DIR [--folds 5] [--boot 2000] [--quick]
      The Phase-1 headroom study (design Track A): single methods (A0), tuned cascades
      (cheap-first / expensive-first ladders, reuse-first), threshold rules, depth<=3
      policy trees, a tuned softmax-logistic selector, the hand-written portfolio, the
      public-information bounded-portfolio estimates (cross-fitted multi-output GBT,
      one-shot and sequential), and the hidden-state oracle (ceiling only).
  plan --episodes E --chunks K --lo SEED --sha SHA
      Print root launch commands for a chunked registered run.

A-CF-SMALL confirmation pipeline (additive; see the section "A-CF-SMALL" below):
  cf-populations [--smoke]            print the train0..2 / select / confirm episode-seed populations
  cf-fit --lineage L --train F... --select F... --out DIR [--smoke]
      fit every family on train L, choose hyperparameters / best simple family on select, freeze + hash
  cf-score --fits DIR... --confirm F... --out DIR [--expect-sha H...] [--smoke]
      score the frozen selectors once on the confirm population (paired across lineages)
  cf-plan --sha SHA [--chunks 2] [--smoke]   print the root launch commands

All selectors are cross-fitted by EPISODE folds (instances of one episode share a fold)
and evaluated out-of-fold. CIs are episode-clustered bootstrap percentiles.
Pure numpy; runs with OMP/BLAS threads pinned to 1.
"""
from __future__ import annotations

import argparse, gzip, json, math, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
try:
    from tensegra import campaign06_portworld as pw
except ImportError:   # local machine without torch: import the module file directly
    import importlib.util
    _p = Path(__file__).resolve().parents[2] / "src/tensegra/campaign06_portworld.py"
    _spec = importlib.util.spec_from_file_location("campaign06_portworld", _p)
    pw = importlib.util.module_from_spec(_spec); sys.modules["campaign06_portworld"] = pw; _spec.loader.exec_module(pw)


# ---------------------------------------------------------------------------
# evaluate

def cmd_evaluate(a):
    knobs = json.loads(a.knobs) if a.knobs else None
    if knobs:
        knobs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in knobs.items()}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with gzip.open(out, "rt") as f:
            for line in f:
                done.add(json.loads(line)["episode"])
    t0 = time.process_time()
    n_inst = n_fail = 0
    for seed in range(a.lo, a.lo + a.episodes):
        if seed in done:
            continue
        try:
            recs = pw.evaluate_episode(seed, knobs, version=a.version)
            lines = [json.dumps(r, separators=(",", ":")) for r in recs]
        except RuntimeError as e:   # exact optimum not certified within the cap
            lines = [json.dumps({"episode": seed, "uncertified": str(e)})]
            n_fail += 1
            recs = []
        with gzip.open(out, "at") as f:
            f.write("\n".join(lines) + "\n")
        n_inst += len(recs)
    dt = time.process_time() - t0
    print(json.dumps({"instances": n_inst, "uncertified_episodes": n_fail, "cpu_s": round(dt, 1),
                      "cpu_per_instance": round(dt / max(n_inst, 1), 3)}))


def cmd_pilot(a):
    """Oracle-cost pilot (design v2 revision 10) and price-crossover quantities (review F3):
    per (n, correlation class) cell, `per` episodes; exact oracle CPU/work on instance t=0
    (full arm evaluation on the first `full` of each cell). A cell whose cap-hit rate
    exceeds 25% stops early, and larger n of that class are skipped."""
    out = Path(a.out)
    seed = a.lo
    skip = set()
    with gzip.open(out, "at") as f:
        for cls in a.classes:
            for n in a.n:
                if cls in skip:
                    seed += a.per; continue
                hits = 0
                for j in range(a.per):
                    kn = dict(pw.PILOT_KNOBS, n_range=(n, n), corr_classes=(cls,))
                    insts, kd = pw.generate_episode(seed + j, kn, "pw-v2")
                    t0 = time.process_time()
                    base = pw.exact_optimum(insts[0])
                    t1 = time.process_time()
                    inst = insts[1]; inst.cache = base[0]
                    ex = pw.exact_optimum(inst)
                    t2 = time.process_time()
                    row = {"seed": seed + j, "n": n, "cls": cls, "exact_cpu": t2 - t1, "exact_work": ex[2], "certified": ex[3],
                           "n_hidden": inst.n_hidden()}
                    hits += not ex[3]
                    if j < a.full:
                        rec, _ = pw.evaluate_instance(inst, opt=ex)
                        t3 = time.process_time()
                        row["eval_cpu"] = t3 - t2
                        for arm in ("insp:GR", "insp:PD1000000", "cons:PD1000000", "insp:RUPD1000000", "cons:GR"):
                            row[arm] = [rec["arms"][arm]["v"] / rec["opt"], pw.total_work(rec, arm)]
                    f.write(json.dumps(row) + "\n"); f.flush()
                    if j >= 7 and hits / (j + 1) > 0.25:
                        break
                print(json.dumps({"cls": cls, "n": n, "done": j + 1, "cap_hits": hits}), flush=True)
                if hits / (j + 1) > 0.25:
                    skip.add(cls)
                seed += a.per


def load(paths):
    recs, unc = [], []
    for p in paths:
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                (unc if "uncertified" in r else recs).append(r)
    recs.sort(key=lambda r: (r["episode"], r["t"]))
    return recs, unc


# ---------------------------------------------------------------------------
# Feature / utility matrices

FEATURES = None   # filled from the first record (sorted public feature names)
DRIVERS = ["n", "corr_class", "tight_loc_mean", "tight_glob", "conf_density", "n_comp", "hidden_frac", "m", "log_c", "log_o"]


def feature_matrix(recs):
    names = sorted(recs[0]["pub"].keys())
    return np.array([[float(r["pub"][k]) for k in names] for r in recs]), names


def util_matrix(recs, menu):
    return np.array([[pw.utility(r, m) for m in menu] for r in recs])


def episode_folds(recs, K):
    eps = np.array([r["episode"] for r in recs])
    h = (eps.astype(np.int64) * 2654435761) % (2 ** 32)
    return (h % K).astype(int), eps


# ---------------------------------------------------------------------------
# Cascades (simple tuned sequential portfolios; conditions use the method's own
# public status/telemetry only). Charged: the root bound they read (root_work).

LADDERS = ("PD", "RUPD")


def cascade_actions(rec, mode, start, fam, cap, tau, budgets):
    """Arms run by the cascade. start RV: verify the cache; stop if it is feasible for the
    plan. start GR: greedy+repair. Then escalate fam@b for b = budgets up to cap (cheap
    first), stopping when certified or when the root-bound gap of the best output <= tau."""
    tel = rec["tel"][mode]
    run = []
    if start == "RV":
        run.append(f"{mode}:RV")
        if tel["cache_feasible"]:
            return run
    run.append(f"{mode}:GR")
    rb = max(tel["root_bound"], 1e-9)
    best = max(rec["arms"][x]["v"] for x in run)
    if (rb - best) / rb <= tau:
        return run
    for b in budgets:
        if b > cap:
            break
        arm = f"{mode}:{fam}{b}"
        run.append(arm)
        best = max(best, rec["arms"][arm]["v"])
        if tel[f"{fam.lower()}{b}_finished"] or (rb - best) / rb <= tau:
            break
    return run


def cascade_grid(budgets):
    out = []
    for mode in pw.MODES:
        for start in ("RV", "GR"):
            for fam in LADDERS:
                for cap in (0,) + tuple(budgets):
                    for tau in (0.0, 0.005, 0.01, 0.02, 0.04, 0.08):
                        if cap == 0 and fam != "PD":
                            continue
                        out.append((mode, start, fam, cap, tau))
    return out


def cascade_matrix(recs, grid, budgets):
    U = np.zeros((len(recs), len(grid)))
    for i, r in enumerate(recs):
        for j, c in enumerate(grid):
            m = c[0]
            U[i, j] = pw.utility(r, cascade_actions(r, *c, budgets), extra_work=r["tel"][m]["root_work"])
    return U


# ---------------------------------------------------------------------------
# Mandatory strong simple baseline family (design v2 revision 8):
#   inspect iff theta * hidden_frac * L > o * n_hidden (else plan `else_mode`: cons or opt);
#   always verify/repair the cached solution and warm-start from it (RUPD: propagation +
#   decomposition always on); B&B budget scaled to the compute price (b = kappa / c, read
#   off the trace); certified-gap stopping rule (stop at the first trace point b with
#   (UB_b - inc_b) / inc_b < gamma * c * b); node-count switch rule (if not certified at
#   b_sw, stop searching and switch to beam width 16; commit the best output).

SB_GRID = [(th, em, ka, ga, sw) for th in (0.25, 1.0, 4.0) for em in ("cons", "opt")
           for ka in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2) for ga in (0.0, 0.5, 1.0, 2.0, 4.0) for sw in (0, 10_000, 100_000)]


def strong_baseline(rec, th, em, ka, ga, sw):
    p = rec["pub"]
    c, o, L = rec["prices"]["c"], rec["prices"]["o"], rec["prices"]["L"]
    nh = rec["n_hidden"]
    mode = "insp" if nh > 0 and th * p["hidden_frac"] * L > o * nh else em
    bmax = ka / c
    grid = [b for b in pw.TRACE_GRID if b <= bmax]
    run = [f"{mode}:GR", f"{mode}:RU"]
    arm = None
    for b in grid:
        arm = pw.trace_arm(rec, mode, "RUPD", b)
        if arm["fin"]:
            break
        if ga > 0 and (arm["ub"] - arm["v"]) / max(arm["v"], 1) < ga * c * b:
            break
        if sw and b >= sw:
            run.append(f"{mode}:BM16")
            break
    if arm is None:
        arm = pw.trace_arm(rec, mode, "RUPD", 0)
    return run + [arm]


def sb_matrix(recs):
    return np.array([[pw.utility(r, strong_baseline(r, *h)) for h in SB_GRID]
                     for r in recs])


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Binning shared by trees and GBT

def make_bins(X, B):
    edges = []
    for f in range(X.shape[1]):
        q = np.unique(np.quantile(X[:, f], np.linspace(0, 1, B + 1)[1:-1]))
        edges.append(q)
    return edges


def apply_bins(X, edges):
    return np.stack([np.searchsorted(e, X[:, f], side="right") for f, e in enumerate(edges)], 1).astype(np.int64)


def node_hist(codes, R, B):
    """Per (feature, bin) sums of R (n x A) and counts."""
    n, F = codes.shape
    A = R.shape[1]
    key = codes + (np.arange(F) * B)[None, :]
    cnt = np.bincount(key.ravel(), minlength=F * B).reshape(F, B)
    k2 = (key[:, :, None] * A + np.arange(A)[None, None, :]).ravel()
    H = np.bincount(k2, weights=np.broadcast_to(R[:, None, :], (n, F, A)).ravel(), minlength=F * B * A).reshape(F, B, A)
    return H, cnt


# ---------------------------------------------------------------------------
# Utility-maximizing policy tree (depth <= 3)

class PolicyTree:
    def __init__(self, depth, min_leaf, B=16, lookahead=True, feats=None):
        self.depth, self.min_leaf, self.B, self.lookahead, self.feats = depth, min_leaf, B, lookahead, feats

    def _best_split(self, codes, U):
        H, cnt = node_hist(codes, U, self.B)
        L = np.cumsum(H, 1)[:, :-1, :]            # left = bins <= b
        nL = np.cumsum(cnt, 1)[:, :-1]
        T = U.sum(0)
        val = L.max(2) + (T[None, None, :] - L).max(2)
        ok = (nL >= self.min_leaf) & (len(U) - nL >= self.min_leaf)
        val = np.where(ok, val, -np.inf)
        f, b = np.unravel_index(np.argmax(val), val.shape)
        return (int(f), int(b), float(val[f, b])) if np.isfinite(val[f, b]) else None

    def _grow(self, codes, U, d):
        leaf = {"a": int(np.argmax(U.sum(0)))}
        if d == 0 or len(U) < 2 * self.min_leaf:
            return leaf, float(U.sum(0).max())
        if self.lookahead and d >= 2:
            # root lookahead over the top-16 greedy split candidates (exhaustive two-level
            # search restricted to the most promising first splits)
            H, cnt = node_hist(codes, U, self.B)
            L = np.cumsum(H, 1)[:, :-1, :]
            nL = np.cumsum(cnt, 1)[:, :-1]
            ok = (nL >= self.min_leaf) & (len(U) - nL >= self.min_leaf)
            val = np.where(ok, L.max(2) + (U.sum(0)[None, None, :] - L).max(2), -np.inf)
            flat = np.argsort(-val, axis=None)[:16]
            cand = [np.unravel_index(t, val.shape) for t in flat if np.isfinite(val.flat[t])]
            best = None
            for f, b in cand:
                m = codes[:, f] <= b
                _, vl = self._grow1(codes[m], U[m])
                _, vr = self._grow1(codes[~m], U[~m])
                if best is None or vl + vr > best[2]:
                    best = (int(f), int(b), vl + vr)
            sp = best
        else:
            sp = self._best_split(codes, U)
        if sp is None or sp[2] <= U.sum(0).max() + 1e-12:
            return leaf, float(U.sum(0).max())
        f, b, _ = sp
        m = codes[:, f] <= b
        lt, vl = self._grow(codes[m], U[m], d - 1)
        rt, vr = self._grow(codes[~m], U[~m], d - 1)
        return {"f": f, "b": b, "l": lt, "r": rt}, vl + vr

    def _grow1(self, codes, U):
        base = float(U.sum(0).max())
        if len(U) < 2 * self.min_leaf:
            return None, base
        sp = self._best_split(codes, U)
        return (None, base) if sp is None else (None, max(base, sp[2]))

    def fit(self, X, U):
        Xs = X if self.feats is None else X[:, self.feats]
        self.edges = make_bins(Xs, self.B)
        codes = apply_bins(Xs, self.edges)
        Uc = U - U.mean(1, keepdims=True)
        self.tree, _ = self._grow(codes, Uc, self.depth)
        return self

    def predict(self, X):
        Xs = X if self.feats is None else X[:, self.feats]
        codes = apply_bins(Xs, self.edges)
        out = np.zeros(len(X), int)
        for i in range(len(X)):
            t = self.tree
            while "a" not in t:
                t = t["l"] if codes[i, t["f"]] <= t["b"] else t["r"]
            out[i] = t["a"]
        return out

    def describe(self, names, menu, t=None, ind=""):
        t = self.tree if t is None else t
        if "a" in t:
            return f"{ind}-> {menu[t['a']]}\n"
        fn = names[t["f"]] if self.feats is None else names[self.feats[t["f"]]]
        thr = self.edges[t["f"]][t["b"]] if t["b"] < len(self.edges[t["f"]]) else float("inf")
        return (f"{ind}if {fn} < {thr:.4g}:\n" + self.describe(names, menu, t["l"], ind + "  ")
                + f"{ind}else:\n" + self.describe(names, menu, t["r"], ind + "  "))


# ---------------------------------------------------------------------------
# Softmax-logistic selector (expected-utility objective, L2 tuned on inner folds)

class LogisticSelector:
    def __init__(self, lam=1e-3, steps=250, lr=0.05, seed=0):
        self.lam, self.steps, self.lr, self.seed = lam, steps, lr, seed

    def fit(self, X, U):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        Uc = U - U.mean(1, keepdims=True)
        s = Uc.std() + 1e-9
        Uc = Uc / s
        A = U.shape[1]
        W = np.zeros((Z.shape[1], A))
        m = np.zeros_like(W); v = np.zeros_like(W)
        for t in range(1, self.steps + 1):
            S = Z @ W
            S -= S.max(1, keepdims=True)
            P = np.exp(S); P /= P.sum(1, keepdims=True)
            eu = (P * Uc).sum(1, keepdims=True)
            G = -(Z.T @ (P * (Uc - eu))) / len(Z) + 2 * self.lam * W
            m = 0.9 * m + 0.1 * G; v = 0.999 * v + 0.001 * G * G
            W -= self.lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        self.W = W
        return self

    def predict(self, X):
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        return np.argmax(Z @ self.W, 1)


def tuned_logistic(Xtr, Utr, eps_tr, lams=(1e-3, 1e-2, 1e-1)):
    ue = np.unique(eps_tr)
    inner = np.isin(eps_tr, ue[::3]), np.isin(eps_tr, ue[1::3]), np.isin(eps_tr, ue[2::3])
    best = None
    for lam in lams:
        sc = 0.0
        for te in inner:
            if te.sum() == 0 or (~te).sum() == 0:
                continue
            p = LogisticSelector(lam).fit(Xtr[~te], Utr[~te]).predict(Xtr[te])
            sc += Utr[te][np.arange(te.sum()), p].sum()
        if best is None or sc > best[0]:
            best = (sc, lam)
    return LogisticSelector(best[1]).fit(Xtr, Utr), best[1]


# ---------------------------------------------------------------------------
# Multi-output gradient-boosted regression trees (squared loss, shared tree structure)

class MGBT:
    def __init__(self, rounds=200, lr=0.05, depth=3, min_leaf=15, B=32, l2=1.0, subsample=0.8, seed=0):
        self.rounds, self.lr, self.depth, self.min_leaf, self.B, self.l2 = rounds, lr, depth, min_leaf, B, l2
        self.subsample, self.seed = subsample, seed

    def _tree(self, codes, R):
        nodes = []

        def grow(idx, d):
            Rn = R[idx]
            val = Rn.sum(0) / (len(idx) + self.l2)
            if d == 0 or len(idx) < 2 * self.min_leaf:
                nodes.append(("leaf", val)); return len(nodes) - 1
            H, cnt = node_hist(codes[idx], Rn, self.B)
            L = np.cumsum(H, 1)[:, :-1, :]
            nL = np.cumsum(cnt, 1)[:, :-1]
            T = Rn.sum(0)
            nR = len(idx) - nL
            gain = (L ** 2).sum(2) / (nL + self.l2) + ((T[None, None, :] - L) ** 2).sum(2) / (nR + self.l2)
            gain = np.where((nL >= self.min_leaf) & (nR >= self.min_leaf), gain, -np.inf)
            f, b = np.unravel_index(np.argmax(gain), gain.shape)
            if not np.isfinite(gain[f, b]) or gain[f, b] <= (T ** 2).sum() / (len(idx) + self.l2) + 1e-12:
                nodes.append(("leaf", val)); return len(nodes) - 1
            m = codes[idx, f] <= b
            me = len(nodes); nodes.append(None)
            li = grow(idx[m], d - 1); ri = grow(idx[~m], d - 1)
            nodes[me] = ("split", int(f), int(b), li, ri)
            return me
        grow(np.arange(len(R)) if self._rows is None else self._rows, self.depth)
        return nodes

    def _apply(self, nodes, codes):
        out = np.zeros((len(codes), self.A))
        idx = np.arange(len(codes))
        stack = [(0, idx)]
        while stack:
            k, ix = stack.pop()
            nd = nodes[k]
            if nd[0] == "leaf":
                out[ix] = nd[1]
            else:
                m = codes[ix, nd[1]] <= nd[2]
                stack.append((nd[3], ix[m])); stack.append((nd[4], ix[~m]))
        return out

    def fit(self, X, Y):
        rng = np.random.default_rng(self.seed)
        self.edges = make_bins(X, self.B)
        codes = apply_bins(X, self.edges)
        self.A = Y.shape[1]
        self.base = Y.mean(0)
        pred = np.tile(self.base, (len(Y), 1))
        self.trees = []
        for _ in range(self.rounds):
            R = Y - pred
            self._rows = np.sort(rng.choice(len(Y), int(self.subsample * len(Y)), replace=False))
            nodes = self._tree(codes, R)
            pred += self.lr * self._apply(nodes, codes)
            self.trees.append(nodes)
        return self

    def predict(self, X):
        codes = apply_bins(X, self.edges)
        out = np.tile(self.base, (len(X), 1))
        for nodes in self.trees:
            out += self.lr * self._apply(nodes, codes)
        return out


# ---------------------------------------------------------------------------
# Hand-written portfolio (sequential; uses public features, prices and the telemetry
# of the calls it makes). Written from the method contracts, not fitted.

HAND_GRID = [(alpha, gamma, mreuse) for alpha in (0.25, 0.5, 1.0, 2.0) for gamma in (0.05, 0.1, 0.2, 0.4, 0.8)
             for mreuse in (0.0, 0.2, 0.5, 2.0)]


def hand_portfolio(rec, budgets, alpha=1.0, gamma=0.2, mreuse=0.2):
    """Hand-written sequential portfolio (structure written from the method contracts;
    three constants tuned on training folds). Returns the list of arms it runs.

    1. Information mode (only when there are hidden parameters): run greedy+repair on the
       optimistic plan (opt:GR, charged) and read its exact posterior feasibility p. Choose
       the cheapest expected loss among: inspect (o * n_hidden), conservative
       (alpha * voi_gap, the public root-bound gap between optimistic and conservative
       plans) and optimistic ((1 - p) * (1 + L)).
    2. If the change since the cache is small (m <= mreuse): verify the cache (RV); commit it
       as is when it verifies and m == 0; otherwise it warm-starts the search.
    3. Greedy + repair; gap = (root bound - best value) / root bound.
    4. Escalate B&B budgets 1e3, 1e4, ... (propagation + decomposition always on; warm
       started from the verified cache when it verified) while the price of the next step
       is below gamma x gap and the run is not certified.
    """
    p = rec["pub"]
    c, o, L = rec["prices"]["c"], rec["prices"]["o"], rec["prices"]["L"]
    nh = rec["n_hidden"]
    run = []
    mode = "cons"
    if nh > 0:
        run.append("opt:GR")
        pf = rec["tel"]["opt"]["gr_post_feas"]
        losses = {"insp": o * nh, "cons": alpha * p["voi_gap"], "opt": (1 - pf) * (1 + L)}
        mode = min(losses, key=losses.get)
    tel = rec["tel"][mode]
    cache_ok = False
    if p["m"] <= mreuse:
        run.append(f"{mode}:RV")
        cache_ok = bool(tel["cache_feasible"])
        if cache_ok and p["m"] == 0.0:
            return run
    run.append(f"{mode}:GR")
    rb = max(tel["root_bound"], 1e-9)
    best = max(rec["arms"][x]["v"] for x in run if x.startswith(mode))
    gap = (rb - best) / rb
    fam = "RUPD" if cache_ok else "PD"
    prev = 0
    for b in budgets:
        if c * (b - prev) >= gamma * gap:
            break
        arm = f"{mode}:{fam}{b}"
        run.append(arm)
        best = max(best, rec["arms"][arm]["v"])
        gap = (rb - best) / rb
        if tel[f"{fam.lower()}{b}_finished"]:
            break
        prev = b
    if mode != "opt" and "opt:GR" in run:
        run.remove("opt:GR")   # its output is not committed (charged separately below)
    return run


def hand_utility(rec, h, budgets):
    run = hand_portfolio(rec, budgets, *h)
    extra = rec["tel"]["cons"]["root_work"] + rec["feature_work"]
    if rec["n_hidden"] > 0 and "opt:GR" not in run:
        extra += pw.total_work(rec, "opt:GR")   # the opt probe was run and paid for
    return pw.utility(rec, run, extra_work=extra)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Bootstrap

def cluster_boot(vals_list, eps, B=2000, seed=0):
    """Episode-clustered percentile CIs for the mean of each vector in vals_list."""
    ue, inv = np.unique(eps, return_inverse=True)
    E = len(ue)
    cnt = np.bincount(inv, minlength=E).astype(float)
    rng = np.random.default_rng(seed)
    Wt = rng.multinomial(E, np.full(E, 1 / E), size=B).astype(float)
    out = []
    for v in vals_list:
        s = np.bincount(inv, weights=v, minlength=E)
        means = (Wt @ s) / (Wt @ cnt)
        out.append((float(v.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))))
    return out


# ---------------------------------------------------------------------------
# headroom (design v2 revisions 7, 8, 11)
#
# Families are functions fam(tr, te) -> per-instance OOF utility on te (selector charges
# included). "Simple" families are A1 candidates; the best simple rule is selected on
# INNER folds of each outer training set (no winner's curse) and scored out of fold.
# BPE one-shot = structured GBT (quality and log-work models per arm, combined with the
# public prices); BPE sequential = cross-fitted first-call selector over {one-shot,
# probe -> learned continuation}. GBT hyperparameters are chosen by inner CV.

G = {}   # global data shared with forked fold workers


def _split_inner(tr_idx, eps, K):
    ue = np.unique(eps[tr_idx])
    order = np.argsort((ue.astype(np.int64) * 40503 + 7) % 1000003, kind="stable")
    h = np.empty(len(ue), int); h[order] = np.arange(len(ue)) % K   # balanced episode folds
    out = []
    for k in range(K):
        m = np.isin(eps[tr_idx], ue[h == k])
        if m.any() and (~m).any():
            out.append((tr_idx[~m], tr_idx[m]))
    return out


def _pick(U, tr, te):
    return U[te][:, int(np.argmax(U[tr].mean(0)))]


def _menu(tr, K=5):
    """arms + the top-K (on the training rows) of each tuned rule family."""
    cols = [G["UA"]]
    for key in ("UC", "USB", "UH"):
        top = np.argsort(-G[key][tr].mean(0))[:K]
        cols.append(G[key][:, top])
    return np.hstack(cols)


def fam_single(tr, te):
    return _pick(G["UA"], tr, te)


def fam_cascade(tr, te):
    return _pick(G["UC"], tr, te)


def fam_strong(tr, te):
    return _pick(G["USB"], tr, te)


def fam_hand(tr, te):
    return _pick(G["UH"], tr, te)


def _feat_charge(te, inference_units):
    return G["c"][te] * (G["fwork"][te] + inference_units)


def fam_tree(depth, feats=None):
    def f(tr, te):
        M = _menu(tr)
        t = PolicyTree(depth, G["minleaf"], feats=feats, lookahead=depth >= 2).fit(G["X"][tr], M[tr])
        return M[te][np.arange(len(te)), t.predict(G["X"][te])] - _feat_charge(te, depth)
    return f


def fam_logistic(tr, te):
    M = _menu(tr)
    sel, lam = tuned_logistic(G["X"][tr], M[tr], G["eps"][tr])
    return M[te][np.arange(len(te)), sel.predict(G["X"][te])] - _feat_charge(te, G["X"].shape[1] * M.shape[1])


GBT_GRID = ((100, 3), (200, 3))


def _gbt_structured_fit(Xs, Q, LW, tr, cfg):
    gq = MGBT(rounds=cfg[0], depth=cfg[1]).fit(Xs[tr], Q[tr])
    gw = MGBT(rounds=cfg[0], depth=cfg[1]).fit(Xs[tr], LW[tr])
    smear = np.exp(LW[tr] - gw.predict(Xs[tr])).mean(0)
    return gq, gw, smear


def _structured(Xs, Q, LW, OB, US, FAIL, tr, te, cfg=None):
    """Hyperparameters (GBT_GRID) chosen on 2 inner folds of tr unless cfg is given, then fit
    on tr and choose on te. FAIL: per-arm failure indicator (opt-mode arms can fail); its
    probability is modelled too."""
    def choose(trn, tst, cfg):
        gq, gw, smear = _gbt_structured_fit(Xs, Q, LW, trn, cfg)
        gf = MGBT(rounds=cfg[0], depth=cfg[1]).fit(Xs[trn], FAIL[trn]) if FAIL[trn].any() else None
        pf = np.clip(gf.predict(Xs[tst]), 0, 1) if gf is not None else 0.0
        q = gq.predict(Xs[tst])
        Uh = q - G["c"][tst, None] * (np.expm1(gw.predict(Xs[tst])) * smear) - G["o"][tst, None] * OB[tst] \
            - G["L"][tst, None] * pf
        return np.argmax(Uh, 1)
    if cfg is None:
        best = None
        for c in GBT_GRID:
            sc = 0.0
            for itr, ite in _split_inner(tr, G["eps"], 2):
                sc += US[ite][np.arange(len(ite)), choose(itr, ite, c)].sum()
            if best is None or sc > best[0]:
                best = (sc, c)
        cfg = best[1]
    ch = choose(tr, te, cfg)
    return US[te][np.arange(len(te)), ch], cfg


def fam_learned_oneshot(tr, te, cfg=None):
    Xs = G["X"][:, G["struct_cols"]]
    u, cfg = _structured(Xs, G["Q"], np.log1p(G["Wk"]), G["Ob"], G["UA"], G["FAILA"], tr, te, cfg)
    return u - _feat_charge(te, 3 * cfg[0] * cfg[1]), cfg


def fam_learned_direct(tr, te):
    M = _menu(tr)
    Mc = M - M.mean(1, keepdims=True)
    g = MGBT(rounds=200).fit(G["X"][tr], Mc[tr])
    return M[te][np.arange(len(te)), np.argmax(g.predict(G["X"][te]), 1)] - _feat_charge(te, 200 * 3)


def fam_learned_seq(probe):
    def f(tr, te, cfg=None):
        S = G["SEQ"][probe]
        u, cfg = _structured(S["XT"], S["Q"], np.log1p(S["W"]), S["OB"], S["U"], S["FAIL"], tr, te, cfg)
        return u - _feat_charge(te, 3 * cfg[0] * cfg[1]), cfg
    return f


SIMPLE = {"A0_single": fam_single, "cascade_tuned": fam_cascade, "strong_baseline": fam_strong, "hand": fam_hand,
          "threshold_any": fam_tree(1), "tree_d2": fam_tree(2), "tree_d3": fam_tree(3), "logistic_tuned": fam_logistic}


def _run_family(name, fn, tr, te):
    out = fn(tr, te)
    return out[0] if isinstance(out, tuple) else out


def outer_fold(k):
    """Everything for outer fold k: every family's OOF utility on the test fold, the
    inner-fold selection of the best simple family, and the learned estimates."""
    fold = G["fold"]
    tr, te = np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]
    res, info = {}, {}
    for name, fn in SIMPLE.items():
        res[name] = _run_family(name, fn, tr, te)
    # inner-fold selection of the best simple family
    inner_scores = {name: 0.0 for name in SIMPLE}
    for itr, ite in _split_inner(tr, G["eps"], G["inner"]):
        for name, fn in SIMPLE.items():
            inner_scores[name] += _run_family(name, fn, itr, ite).sum()
    best_simple = max(inner_scores, key=inner_scores.get)
    res["best_simple_inner"] = res[best_simple]
    res["best_single_inner"] = res["A0_single"]
    info["best_simple_family"] = best_simple
    info["inner_scores"] = {n: v / len(tr) for n, v in inner_scores.items()}
    u, cfg = fam_learned_oneshot(tr, te); res["learned_oneshot"] = u; info["oneshot_cfg"] = cfg
    res["learned_oneshot_direct"] = fam_learned_direct(tr, te)
    for probe in G["SEQ"]:   # hyperparameters: the one-shot model's inner-CV choice
        u, _ = fam_learned_seq(probe)(tr, te, cfg); res[f"learned_seq[{probe}]"] = u
    # first-call choice: cross-fitted inside the training set (inner OOF utilities of each
    # candidate), then a GBT on public features picks the candidate per test instance
    cands = ["learned_oneshot"] + [f"learned_seq[{p}]" for p in G["SEQ"]]
    inner_u = {c: np.zeros(len(G["fold"])) for c in cands}
    for itr, ite in _split_inner(tr, G["eps"], 2):
        inner_u["learned_oneshot"][ite] = fam_learned_oneshot(itr, ite, cfg)[0]
        for p in G["SEQ"]:
            inner_u[f"learned_seq[{p}]"][ite] = fam_learned_seq(p)(itr, ite, cfg)[0]
    S = np.stack([inner_u[c] for c in cands], 1)
    g = MGBT(rounds=100).fit(G["X"][tr], S[tr] - S[tr].mean(1, keepdims=True))
    pick = np.argmax(g.predict(G["X"][te]), 1)
    Ste = np.stack([res[c] for c in cands], 1)
    res["learned_sequential"] = Ste[np.arange(len(te)), pick] - G["c"][te] * 100 * 3
    info["first_call_share"] = {c: float((pick == j).mean()) for j, c in enumerate(cands)}
    return k, te, res, info


SEQ_PROBES = {"cons:GR": ["root_bound", "gr_value", "g_value", "gr_moves"],
              "cons:PD1000": ["root_bound", "gr_value", "pd_nfree", "pd_ncomp", "pd_maxcomp", "pd_nfixed_in", "pd0_value",
                              "pd0_ub", "pd1000_value", "pd1000_ub", "pd1000_finished"],
              "cons:RV+cons:GR": ["root_bound", "gr_value", "cache_feasible", "cache_violations", "cache_value", "reuse_value"],
              "opt:GR": ["root_bound", "gr_value", "gr_post_feas", "cache_post_feas"]}


def population_matrices(recs, X, names, budgets=pw.BUDGETS):
    """Per-instance matrices of one record population (every entry is a function of that
    instance's record only, so populations can be built independently). Shared by headroom
    (unchanged numbers; tests/fixtures/campaign06_headroom_golden.json) and the A-CF-SMALL
    confirm pipeline."""
    arms = pw.arm_names(budgets)
    P = {}
    P["UA"] = util_matrix(recs, arms)
    grid = cascade_grid(budgets)
    P["UC"] = cascade_matrix(recs, grid, budgets)
    P["USB"] = sb_matrix(recs)
    P["UH"] = np.array([[hand_utility(r, h, budgets) for h in HAND_GRID] for r in recs])
    P["c"] = np.array([r["prices"]["c"] for r in recs]); P["o"] = np.array([r["prices"]["o"] for r in recs])
    P["L"] = np.array([r["prices"]["L"] for r in recs])
    P["fwork"] = np.array([r["feature_work"] for r in recs], float)
    P["Q"] = np.array([[(r["arms"][m]["v"] / r["opt"]) if (r["opt"] > 0 and r["arms"][m]["ok"]) else 0.0 for m in arms] for r in recs])
    P["FAILA"] = np.array([[float(not r["arms"][m]["ok"]) for m in arms] for r in recs])
    P["Wk"] = np.array([[pw.total_work(r, m) for m in arms] for r in recs], float)
    P["Ob"] = np.array([[r["arms"][m]["obs"] for m in arms] for r in recs], float)
    P["struct_cols"] = [i for i, f in enumerate(names) if f not in ("log_c", "log_o", "L", "log_obs_cost", "hid_x_L")]
    P["SEQ"] = {}
    for probe, tk in SEQ_PROBES.items():
        pa = probe.split("+")
        mode = pa[0].split(":")[0]
        T = np.array([[float(r["tel"][mode].get(k, 0.0)) for k in tk] for r in recs])
        rb = np.maximum(T[:, 0:1], 1e-9)
        T = np.hstack([T, T[:, 1:] / rb])
        XT = np.hstack([X[:, P["struct_cols"]], T])
        U = np.array([[pw.utility(r, pa + [m]) for m in arms] + [pw.utility(r, pa)] for r in recs])
        # quality / failure of the committed output under the utility's commit rule
        Qs, F = np.zeros_like(U), np.zeros_like(U)
        for i, r in enumerate(recs):
            for j, m in enumerate(arms + [None]):
                seq = pa + ([m] if m else [])
                uq = pw.utility(dict(r, prices={"c": 0.0, "o": 0.0, "L": 1.0}), seq)
                F[i, j] = float(uq < 0)
                Qs[i, j] = max(uq, 0.0)
        W = np.array([[pw.total_work(r, pa + [m]) for m in arms] + [pw.total_work(r, pa)] for r in recs], float)
        OB = np.hstack([P["Ob"], P["Ob"][:, [arms.index(pa[0])]]])
        P["SEQ"][probe] = {"XT": XT, "U": U, "Q": Qs, "FAIL": F, "W": W, "OB": OB}
    return P, grid


def cmd_headroom(a):
    recs, unc = load(a.inputs)
    budgets = pw.BUDGETS
    arms = pw.arm_names(budgets)
    X, names = feature_matrix(recs)
    N = len(recs)
    fold, eps = episode_folds(recs, a.folds)
    t0 = time.process_time()
    G.update(X=X, names=names, fold=fold, eps=eps, inner=a.inner, minleaf=max(20, int(0.03 * N * (a.folds - 1) / a.folds)))
    P, grid = population_matrices(recs, X, names, budgets)
    G.update(P)
    print(f"[headroom] N={N} episodes={len(np.unique(eps))} arms={len(arms)} cascades={len(grid)} SB={len(SB_GRID)} "
          f"hand={len(HAND_GRID)} matrices {time.process_time() - t0:.1f}s", flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(min(a.procs, a.folds)) as pool:
        outs = pool.map(outer_fold, range(a.folds))
    res, infos = {}, {}
    for k, te, r, info in outs:
        infos[k] = info
        for name, u in r.items():
            res.setdefault(name, np.zeros(N))[te] = u
    UA = G["UA"]
    res["oracle_arms (hidden-state, non-deployable)"] = UA.max(1)
    keys = list(res)
    cis = dict(zip(keys, cluster_boot([res[k] for k in keys], eps, a.boot)))
    gates = {}
    for est in ("learned_oneshot", "learned_sequential"):
        d1 = res[est] - res["best_simple_inner"]
        d2 = res[est] - res["best_single_inner"]
        (m1, l1, h1), (m2, l2, h2) = cluster_boot([d1, d2], eps, a.boot, seed=1)
        gates[est] = {"GA1_vs_best_simple": {"mean": m1, "lo": l1, "hi": h1, "pass": bool(m1 >= 0.02 and l1 > 0.005)},
                      "GA2_vs_best_single": {"mean": m2, "lo": l2, "hi": h2, "pass": bool(m2 >= 0.02 and l2 > 0.005)},
                      "phase2_margin_max": max(0.0, m1 / 2)}
    spread = cluster_boot([res["best_simple_inner"] - res["best_single_inner"],
                           res["oracle_arms (hidden-state, non-deployable)"] - res["best_simple_inner"]], eps, a.boot, seed=2)
    best_arm = np.argmax(UA, 1)
    freq = {arms[j]: float((best_arm == j).mean()) for j in np.unique(best_arm)}
    meth = {}
    for j, m in enumerate(arms):
        fam = m.split(":")[1].rstrip("0123456789")
        meth[fam] = meth.get(fam, 0.0) + float((best_arm == j).mean())
    modes = {}
    for j, m in enumerate(arms):
        modes[m.split(":")[0]] = modes.get(m.split(":")[0], 0.0) + float((best_arm == j).mean())
    drivers = {}
    for dname in DRIVERS:
        x = X[:, names.index(dname)]
        qs = np.unique(np.quantile(x, [1 / 3, 2 / 3]))
        b = np.searchsorted(qs, x, side="right")
        rows = []
        for kb in np.unique(b):
            m = b == kb
            ba = np.bincount(best_arm[m], minlength=len(arms))
            rows.append({"bin": int(kb), "range": [float(x[m].min()), float(x[m].max())], "n": int(m.sum()),
                         **{k: float(res[k][m].mean()) for k in ("best_single_inner", "best_simple_inner", "hand", "strong_baseline",
                                                                   "learned_oneshot", "learned_sequential",
                                                                   "oracle_arms (hidden-state, non-deployable)")},
                         "modal_best_arm": arms[int(np.argmax(ba))], "modal_best_frac": float(ba.max() / m.sum())})
        drivers[dname] = rows
    certified = float(np.mean([r["certified"] for r in recs]))
    summary = {
        "generator": recs[0].get("generator"), "N": N, "episodes": int(len(np.unique(eps))), "folds": a.folds,
        "inner_folds": a.inner, "boot": a.boot, "uncertified_episodes_legacy": len(unc),
        "oracle_certified_fraction": certified, "uncertified_instances_scored_vs_UB": int(sum(not r["certified"] for r in recs)),
        "utilities": {k: {"mean": cis[k][0], "lo": cis[k][1], "hi": cis[k][2]} for k in keys},
        "gates": gates,
        "best_simple_minus_best_single": dict(zip(("mean", "lo", "hi"), spread[0])),
        "oracle_minus_best_simple": dict(zip(("mean", "lo", "hi"), spread[1])),
        "best_simple_family_per_fold": {k: infos[k]["best_simple_family"] for k in infos},
        "fold_info": infos,
        "best_arm_frequency": dict(sorted(freq.items(), key=lambda t: -t[1])),
        "best_method_share": dict(sorted(meth.items(), key=lambda t: -t[1])),
        "methods_below_5pct": [m for m, v in meth.items() if v < 0.05],
        "best_mode_share": modes,
        "drivers": drivers,
        "exact_work": {"mean": float(np.mean([r["exact_work"] for r in recs])), "p50": float(np.median([r["exact_work"] for r in recs])),
                       "p99": float(np.quantile([r["exact_work"] for r in recs], 0.99)), "max": float(np.max([r["exact_work"] for r in recs]))},
        "selector_feature_work_mean": float(G["fwork"].mean()),
        "cpu_s_main": round(time.process_time() - t0, 1),
    }
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "headroom.json").write_text(json.dumps(summary, indent=1, default=float))
    np.savez_compressed(out / "per_instance.npz", eps=eps, fold=fold,
                        **{"".join(ch if ch.isalnum() else "_" for ch in k): v for k, v in res.items()})
    md = report_md(summary)
    (out / "headroom.md").write_text(md)
    print(md)


def report_md(s):
    L = [f"# portworld headroom ({s['generator']}; N={s['N']} instances, {s['episodes']} episodes; "
         f"oracle certified {s['oracle_certified_fraction']:.3f}, {s['uncertified_instances_scored_vs_UB']} scored vs UB)", "",
         "| policy | U | 95% CI (episode-clustered) |", "|---|---|---|"]
    for k, v in sorted(s["utilities"].items(), key=lambda t: -t[1]["mean"]):
        L.append(f"| {k} | {v['mean']:.4f} | {v['lo']:.4f} – {v['hi']:.4f} |")
    L += ["", "## Gates (design v2 revision 7)", "", "| estimate | GA-1 vs best simple (inner-selected) | GA-2 vs best single | pass GA-1 / GA-2 |", "|---|---|---|---|"]
    for est, g in s["gates"].items():
        a, b = g["GA1_vs_best_simple"], g["GA2_vs_best_single"]
        L.append(f"| {est} | {a['mean']:.4f} [{a['lo']:.4f}, {a['hi']:.4f}] | {b['mean']:.4f} [{b['lo']:.4f}, {b['hi']:.4f}] | {a['pass']} / {b['pass']} |")
    sp, om = s["best_simple_minus_best_single"], s["oracle_minus_best_simple"]
    L += ["", f"best simple − best single: {sp['mean']:.4f} [{sp['lo']:.4f}, {sp['hi']:.4f}]; "
          f"oracle − best simple: {om['mean']:.4f} [{om['lo']:.4f}, {om['hi']:.4f}]",
          f"best simple family per outer fold (inner-selected): {s['best_simple_family_per_fold']}",
          "", "best-method share (hindsight): " + ", ".join(f"{k} {v:.2f}" for k, v in s["best_method_share"].items()),
          f"methods best < 5%: {s['methods_below_5pct']}; best-mode share: " + ", ".join(f"{k} {v:.2f}" for k, v in s["best_mode_share"].items()),
          "", "best-arm frequency: " + ", ".join(f"{k} {v:.2f}" for k, v in list(s["best_arm_frequency"].items())[:12]),
          "", "## drivers (tertiles)", "",
          "| driver | range | n | best single | best simple | hand | strong | learned 1-shot | learned seq | oracle | modal best arm |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d, rows in s["drivers"].items():
        for r in rows:
            L.append(f"| {d} | {r['range'][0]:.3g}–{r['range'][1]:.3g} | {r['n']} | {r['best_single_inner']:.3f} | {r['best_simple_inner']:.3f} | "
                     f"{r['hand']:.3f} | {r['strong_baseline']:.3f} | {r['learned_oneshot']:.3f} | {r['learned_sequential']:.3f} | "
                     f"{r['oracle_arms (hidden-state, non-deployable)']:.3f} | {r['modal_best_arm']} ({r['modal_best_frac']:.2f}) |")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# A-CF-SMALL confirmation pipeline (registry A-CF-SMALL; registered after the A-HS gate
# failure). Additive: evaluate/headroom/plan are unchanged (bit-identity test:
# tests/test_campaign06_confirm.py against tests/fixtures/campaign06_headroom_golden.json).
#
#   (1) evaluate  (the existing subcommand) on 3 disjoint training populations, one shared
#       select population and one shared confirm population (cf_populations).
#   (2) cf-fit    per lineage L: every simple family and the learned one-shot / sequential
#       selectors are FIT on train L only; hyperparameters (logistic L2, GBT config) and the
#       best simple family are CHOSEN on the select population only; the frozen artifacts
#       are pickled and hashed. cf-fit refuses any record outside the role's seed range, so
#       it cannot read the confirm population.
#   (3) cf-score  the frozen selectors of all lineages are applied ONCE to the confirm
#       population (paired: the same confirm instances for every lineage); refuses a second
#       run into the same directory, a hash mismatch, or any population overlap.
#
# Semantics of every family, its menu and its selector charge are those of headroom (same
# classes, grids, menus and charges); only the fit/choose/evaluate split differs:
#   A0_single / cascade_tuned / strong_baseline / hand: constants = argmax of the TRAIN mean;
#   threshold_any / tree_d2 / tree_d3: menu (arms + top-5 of each tuned family on TRAIN),
#       min_leaf = max(20, int(.03 N_train));
#   logistic_tuned: one model per L2 in LOGIT_LAMS fit on TRAIN, L2 chosen on SELECT;
#   learned_oneshot: structured GBT (quality, log-work, failure) for each GBT_GRID config
#       fit on TRAIN, config chosen on SELECT (charged utility);
#   learned_seq[probe]: same structure on the probe telemetry, the one-shot's chosen config;
#   learned_sequential: first-call GBT over {one-shot, seq[probe]...}, trained on 2-fold
#       inner out-of-fold utilities WITHIN the training population (as headroom);
#   best_simple: argmax over the 8 simple families of the SELECT mean.

CF_EPISODES = 640
CF_LINEAGES = 3
CF_LINEAGE_STRIDE = 1_000_000     # lineage L trains on 310,000,000 + L * 1,000,000 + i
CF_SMOKE_BASE = 303_000_000       # dev_builder smoke (never protocol)
CF_SIMPLE = ("A0_single", "cascade_tuned", "strong_baseline", "hand", "threshold_any", "tree_d2", "tree_d3", "logistic_tuned")
LOGIT_LAMS = (1e-3, 1e-2, 1e-1)


def cf_populations(episodes=CF_EPISODES, lineages=CF_LINEAGES, smoke=False):
    """Episode-seed populations {name: (lo, hi)} (hi exclusive; one seed per episode).

    Registered: train L = [310,000,000 + L*1,000,000, + episodes) for L < lineages;
    select = [320,000,000, + episodes); confirm = [330,000,000, + episodes).
    Smoke (dev_builder only): train L = 303,000,000 + L*10,000 + i; select 303,100,000 + i;
    confirm 303,200,000 + i. Asserts containment in the role's registered sub-range and
    pairwise disjointness (hence disjointness from dev_builder/dev_gate for registered runs)."""
    R = pw.SEED_RANGES
    if smoke:
        assert episodes <= 10_000 and lineages <= 10
        b = CF_SMOKE_BASE
        pops = {f"train{L}": (b + L * 10_000, b + L * 10_000 + episodes) for L in range(lineages)}
        pops["select"] = (b + 100_000, b + 100_000 + episodes)
        pops["confirm"] = (b + 200_000, b + 200_000 + episodes)
        roles = {k: "dev_builder" for k in pops}
    else:
        assert 0 < episodes <= CF_LINEAGE_STRIDE and 0 < lineages <= 10
        t0 = R["train"][0]
        pops = {f"train{L}": (t0 + L * CF_LINEAGE_STRIDE, t0 + L * CF_LINEAGE_STRIDE + episodes) for L in range(lineages)}
        pops["select"] = (R["select"][0], R["select"][0] + episodes)
        pops["confirm"] = (R["confirm"][0], R["confirm"][0] + episodes)
        roles = {k: ("train" if k.startswith("train") else k) for k in pops}
    assert pw.check_seed_ranges()
    for k, (lo, hi) in pops.items():
        r0, r1 = R[roles[k]]
        assert r0 <= lo < hi <= r1, (k, lo, hi, roles[k])
        if not smoke:
            for other in ("dev_builder", "dev_gate"):
                assert hi <= R[other][0] or R[other][1] <= lo, (k, other)
    items = sorted(pops.values())
    for (a0, a1), (b0, b1) in zip(items, items[1:]):
        assert a1 <= b0, "A-CF-SMALL populations overlap"
    return pops


def _sha(b: bytes):
    import hashlib
    return hashlib.sha256(b).hexdigest()


def _file_sha(p):
    return _sha(Path(p).read_bytes())


def _check_population(recs, rng, what, unc=()):
    """Every record's episode lies in rng = [lo, hi) and the population is complete (every
    episode of the range present, as instance records or legacy uncertified-episode lines).
    Returns the sorted episode list of the instance records."""
    eps = sorted({int(r["episode"]) for r in recs})
    allep = set(eps) | {int(r["episode"]) for r in unc}
    bad = sorted(e for e in allep if not (rng[0] <= e < rng[1]))
    if bad or not eps:
        raise SystemExit(f"refusing: {what} population has {len(bad)} episodes outside [{rng[0]},{rng[1]}) "
                         f"(e.g. {bad[:3]}) or is empty")
    if len(allep) != rng[1] - rng[0]:
        raise SystemExit(f"refusing: {what} population incomplete: {len(allep)} of {rng[1] - rng[0]} episodes")
    return eps


def _rows(P, idx):
    """Row subset of a population-matrix dict."""
    out = {}
    for k, v in P.items():
        if k == "SEQ":
            out[k] = {p: {kk: vv[idx] for kk, vv in S.items()} for p, S in v.items()}
        elif k in ("struct_cols", "names"):
            out[k] = v
        else:
            out[k] = v[idx]
    return out


def cf_matrices(recs):
    X, names = feature_matrix(recs)
    P, _ = population_matrices(recs, X, names)
    P["X"], P["names"] = X, names
    P["eps"] = np.array([r["episode"] for r in recs])
    return P


_MODEL_CLASSES = {"PolicyTree": PolicyTree, "LogisticSelector": LogisticSelector, "MGBT": MGBT}


def _freeze(obj):
    """Plain-data snapshot of a fitted model (module-independent pickles)."""
    if obj is None:
        return None
    return {"cls": type(obj).__name__, "state": {k: v for k, v in vars(obj).items() if k != "_rows"}}


def _thaw(d):
    if d is None:
        return None
    o = _MODEL_CLASSES[d["cls"]].__new__(_MODEL_CLASSES[d["cls"]])
    o.__dict__.update(d["state"])
    return o


def cf_menu_spec(P, K=5):
    """headroom's _menu: arms + top-K (on this population's rows) of each tuned family."""
    return {key: np.argsort(-P[key].mean(0))[:K] for key in ("UC", "USB", "UH")}


def cf_menu(P, spec):
    return np.hstack([P["UA"]] + [P[key][:, spec[key]] for key in ("UC", "USB", "UH")])


def _struct_source(P, src):
    if src == "A":
        return P["X"][:, P["struct_cols"]], P["Q"], np.log1p(P["Wk"]), P["Ob"], P["UA"], P["FAILA"]
    S = P["SEQ"][src]
    return S["XT"], S["Q"], np.log1p(S["W"]), S["OB"], S["U"], S["FAIL"]


def _struct_fit(P, src, cfg):
    """headroom _structured's choose(): fit on every row of P."""
    Xs, Q, LW, _, _, FAIL = _struct_source(P, src)
    gq = MGBT(rounds=cfg[0], depth=cfg[1]).fit(Xs, Q)
    gw = MGBT(rounds=cfg[0], depth=cfg[1]).fit(Xs, LW)
    smear = np.exp(LW - gw.predict(Xs)).mean(0)
    gf = MGBT(rounds=cfg[0], depth=cfg[1]).fit(Xs, FAIL) if FAIL.any() else None
    return {"kind": "struct", "src": src, "cfg": list(cfg), "gq": _freeze(gq), "gw": _freeze(gw), "smear": smear,
            "gf": _freeze(gf), "units": 3 * cfg[0] * cfg[1]}


def cf_charge(P, art, arts=None):
    """Selector compute charged to each instance (utility units)."""
    k = art["kind"]
    if k == "pick":
        return np.zeros(len(P["c"]))   # the tuned families' own charges are inside their utility matrices
    if k == "sequential":
        cands = np.stack([cf_charge(P, arts[c], arts) for c in art["cands"]], 1)
        pick = np.argmax(_thaw(art["model"]).predict(P["X"]), 1)
        return cands[np.arange(len(pick)), pick] + P["c"] * art["units"]
    return P["c"] * (P["fwork"] + art["units"])


def cf_apply(P, art, arts=None):
    """Per-instance utility (selector charges included) of a frozen artifact on population P."""
    k = art["kind"]
    n = len(P["c"])
    if k == "pick":
        return P[art["mat"]][:, art["j"]]
    if k in ("tree", "logit", "direct"):
        M = cf_menu(P, art["menu"])
        m = _thaw(art["model"])
        ch = np.argmax(m.predict(P["X"]), 1) if k == "direct" else m.predict(P["X"])
        return M[np.arange(n), ch] - cf_charge(P, art)
    if k == "struct":
        Xs, _, _, OB, US, _ = _struct_source(P, art["src"])
        gq, gw, gf = _thaw(art["gq"]), _thaw(art["gw"]), _thaw(art["gf"])
        pf = np.clip(gf.predict(Xs), 0, 1) if gf is not None else 0.0
        Uh = gq.predict(Xs) - P["c"][:, None] * (np.expm1(gw.predict(Xs)) * art["smear"]) - P["o"][:, None] * OB \
            - P["L"][:, None] * pf
        return US[np.arange(n), np.argmax(Uh, 1)] - cf_charge(P, art)
    if k == "sequential":
        S = np.stack([cf_apply(P, arts[c], arts) for c in art["cands"]], 1)
        pick = np.argmax(_thaw(art["model"]).predict(P["X"]), 1)
        return S[np.arange(n), pick] - P["c"] * art["units"]
    raise ValueError(k)


def cf_resolve(arts, chosen, name):
    while name in chosen:   # aliases may chain: best_simple -> logistic_tuned -> logistic[lam=..]
        name = chosen[name]
    return arts[name]


def cf_fit(PT, PS, log=print):
    """Fit every candidate on the training population PT; choose on the select population PS.
    Returns (artifacts, chosen, select_scores). PS is used only through cf_apply(PS, .) means."""
    NT = len(PT["c"])
    arts = {}
    for name, mat in (("A0_single", "UA"), ("cascade_tuned", "UC"), ("strong_baseline", "USB"), ("hand", "UH")):
        arts[name] = {"kind": "pick", "mat": mat, "j": int(np.argmax(PT[mat].mean(0)))}
    spec = cf_menu_spec(PT)
    MT = cf_menu(PT, spec)
    minleaf = max(20, int(0.03 * NT))
    for name, d in (("threshold_any", 1), ("tree_d2", 2), ("tree_d3", 3)):
        t = PolicyTree(d, minleaf, feats=None, lookahead=d >= 2).fit(PT["X"], MT)
        arts[name] = {"kind": "tree", "menu": spec, "model": _freeze(t), "units": d}
    for lam in LOGIT_LAMS:
        arts[f"logistic[lam={lam:g}]"] = {"kind": "logit", "menu": spec, "units": PT["X"].shape[1] * MT.shape[1],
                                          "model": _freeze(LogisticSelector(lam).fit(PT["X"], MT)), "lam": lam}
    for cfg in GBT_GRID:
        arts[f"learned_oneshot[cfg={cfg[0]}x{cfg[1]}]"] = _struct_fit(PT, "A", cfg)
    Mc = MT - MT.mean(1, keepdims=True)
    arts["learned_oneshot_direct"] = {"kind": "direct", "menu": spec, "units": 200 * 3,
                                      "model": _freeze(MGBT(rounds=200).fit(PT["X"], Mc))}
    log(f"[cf-fit] simple families + one-shot candidates fit on N_train={NT}")
    sel = {n: float(cf_apply(PS, a, arts).mean()) for n, a in arts.items()}
    chosen = {}
    chosen["logistic_tuned"] = max((n for n in arts if n.startswith("logistic[")), key=lambda n: sel[n])
    chosen["learned_oneshot"] = max((n for n in arts if n.startswith("learned_oneshot[")), key=lambda n: sel[n])
    cfg = tuple(arts[chosen["learned_oneshot"]]["cfg"])
    for p in SEQ_PROBES:
        arts[f"learned_seq[{p}]"] = _struct_fit(PT, p, cfg)
    # first-call selector: inner 2-fold OOF utilities of each candidate within the training population
    cands = [chosen["learned_oneshot"]] + [f"learned_seq[{p}]" for p in SEQ_PROBES]
    S = np.zeros((NT, len(cands)))
    for itr, ite in _split_inner(np.arange(NT), PT["eps"], 2):
        Ptr, Pte = _rows(PT, itr), _rows(PT, ite)
        S[ite, 0] = cf_apply(Pte, _struct_fit(Ptr, "A", cfg))
        for j, p in enumerate(SEQ_PROBES, 1):
            S[ite, j] = cf_apply(Pte, _struct_fit(Ptr, p, cfg))
    g = MGBT(rounds=100).fit(PT["X"], S - S.mean(1, keepdims=True))
    arts["learned_sequential"] = {"kind": "sequential", "cands": cands, "model": _freeze(g), "units": 100 * 3}
    for n in list(arts):
        if n not in sel:
            sel[n] = float(cf_apply(PS, arts[n], arts).mean())
    for n in ("logistic_tuned", "learned_oneshot"):
        sel[n] = sel[chosen[n]]
    chosen["best_simple"] = max(CF_SIMPLE, key=lambda n: sel[n])
    chosen["best_single"] = "A0_single"
    return arts, chosen, sel


def _pickle(obj):
    import pickle
    return pickle.dumps(obj, protocol=4)


def cmd_cf_fit(a):
    import pickle
    t0 = time.process_time()
    pops = cf_populations(a.episodes, a.lineages, a.smoke)
    tr_recs, tr_unc = load(a.train)
    se_recs, se_unc = load(a.select)
    tr_eps = _check_population(tr_recs, pops[f"train{a.lineage}"], f"train{a.lineage}", tr_unc)
    se_eps = _check_population(se_recs, pops["select"], "select", se_unc)
    PT, PS = cf_matrices(tr_recs), cf_matrices(se_recs)
    assert PT["names"] == PS["names"], "feature names differ between populations"
    print(f"[cf-fit] lineage {a.lineage}: train {len(tr_recs)} inst / {len(tr_eps)} ep; select {len(se_recs)} inst / "
          f"{len(se_eps)} ep; matrices {time.process_time() - t0:.1f}s", flush=True)
    arts, chosen, sel = cf_fit(PT, PS)
    blob = _pickle({"artifacts": arts, "chosen": chosen, "names": PT["names"], "lineage": a.lineage})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "fit.pkl").exists():
        sys.exit(f"refusing: {out}/fit.pkl exists (artifacts are frozen once)")
    (out / "fit.pkl").write_bytes(blob)
    meta = {
        "registry": "A-CF-SMALL", "lineage": a.lineage, "smoke": a.smoke, "populations": pops,
        "train": {"range": pops[f"train{a.lineage}"], "episodes": len(tr_eps), "instances": len(tr_recs),
                  "uncertified_episodes_legacy": len(tr_unc), "episode_list_sha256": _sha(json.dumps(tr_eps).encode()),
                  "files": {str(p): _file_sha(p) for p in a.train}},
        "select": {"range": pops["select"], "episodes": len(se_eps), "instances": len(se_recs),
                   "uncertified_episodes_legacy": len(se_unc), "episode_list_sha256": _sha(json.dumps(se_eps).encode()),
                   "files": {str(p): _file_sha(p) for p in a.select}},
        "confirm_touched": False,
        "min_leaf": max(20, int(0.03 * len(tr_recs))),
        "chosen": chosen, "select_scores": dict(sorted(sel.items(), key=lambda t: -t[1])),
        "artifact_sha256": {n: _sha(_pickle(x)) for n, x in arts.items()},
        "fit_pkl_sha256": _sha(blob),
        "tool_sha256": _file_sha(__file__),
        "cpu_s": round(time.process_time() - t0, 1),
    }
    (out / "train_episodes.json").write_text(json.dumps(tr_eps))
    (out / "select_episodes.json").write_text(json.dumps(se_eps))
    (out / "fit.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"lineage": a.lineage, "chosen": chosen, "fit_pkl_sha256": meta["fit_pkl_sha256"],
                      "select_best_simple": sel[chosen["best_simple"]], "select_learned_oneshot": sel["learned_oneshot"],
                      "select_learned_sequential": sel["learned_sequential"], "cpu_s": meta["cpu_s"]}, indent=1))


def cf_load_fit(d, expect_sha=None):
    import pickle
    d = Path(d)
    meta = json.loads((d / "fit.json").read_text())
    blob = (d / "fit.pkl").read_bytes()
    h = _sha(blob)
    if h != meta["fit_pkl_sha256"] or (expect_sha and not h.startswith(expect_sha)):
        raise SystemExit(f"refusing: {d}/fit.pkl sha256 {h} does not match fit.json / expected {expect_sha}")
    fit = pickle.loads(blob)
    fit["meta"] = meta
    fit["train_eps"] = set(json.loads((d / "train_episodes.json").read_text()))
    fit["select_eps"] = set(json.loads((d / "select_episodes.json").read_text()))
    return fit


CF_REPORT = ("best_single", "best_simple") + CF_SIMPLE + ("learned_oneshot", "learned_oneshot_direct") + \
    tuple(f"learned_seq[{p}]" for p in SEQ_PROBES) + ("learned_sequential",)


def cf_score(fits, PC, boot=2000):
    """Apply every lineage's frozen selectors once to the confirm population PC."""
    eps = PC["eps"]
    per = {}
    for L, f in fits.items():
        arts, ch = f["artifacts"], f["chosen"]
        per[L] = {n: cf_apply(PC, cf_resolve(arts, ch, n), arts) for n in CF_REPORT}
        per[L]["_charge"] = {n: cf_charge(PC, cf_resolve(arts, ch, n), arts) for n in CF_REPORT}
    oracle = PC["UA"].max(1)
    lin = sorted(per)
    out = {"lineages": {}}
    for L in lin:
        u = per[L]
        keys = list(CF_REPORT)
        cis = dict(zip(keys, cluster_boot([u[k] for k in keys], eps, boot)))
        diffs = {}
        for est in ("learned_oneshot", "learned_sequential"):
            (m1, l1, h1), (m2, l2, h2) = cluster_boot([u[est] - u["best_simple"], u[est] - u["best_single"]], eps, boot, seed=1)
            diffs[est] = {"primary_vs_best_simple_select": {"mean": m1, "lo": l1, "hi": h1, "pass": bool(m1 > 0 and l1 > 0)},
                          "GA2_analogue_vs_best_single": {"mean": m2, "lo": l2, "hi": h2, "pass": bool(m2 >= 0.02 and l2 > 0.005)}}
        # reported, not a gate: margin over the strongest individual simple family in hindsight on confirm
        hind = max(CF_SIMPLE, key=lambda n: u[n].mean())
        (mh, lh, hh), = cluster_boot([u["learned_oneshot"] - u[hind]], eps, boot, seed=3)
        out["lineages"][str(L)] = {
            "chosen": fits[L]["chosen"], "fit_pkl_sha256": fits[L]["meta"]["fit_pkl_sha256"],
            "select_scores": {k: fits[L]["meta"]["select_scores"].get(fits[L]["chosen"].get(k, k)) for k in CF_REPORT},
            "utilities": {k: {"mean": cis[k][0], "lo": cis[k][1], "hi": cis[k][2]} for k in keys},
            "estimates": diffs,
            "hindsight_best_simple_on_confirm": {"family": hind, "oneshot_minus": {"mean": mh, "lo": lh, "hi": hh}},
            "selector_charge_mean": {k: float(per[L]["_charge"][k].mean()) for k in keys},
            "selector_charge_mean_work_units": {k: float((per[L]["_charge"][k] / PC["c"]).mean()) for k in keys},
        }
    # paired across lineages (the same bootstrap episode weights for every lineage)
    for est in ("learned_oneshot", "learned_sequential"):
        ds = [per[L][est] - per[L]["best_simple"] for L in lin]
        r = cluster_boot(ds + [np.mean(ds, 0)], eps, boot, seed=4)
        out.setdefault("paired", {})[est] = {
            "per_lineage": {str(L): dict(zip(("mean", "lo", "hi"), r[i])) for i, L in enumerate(lin)},
            "mean_over_lineages": dict(zip(("mean", "lo", "hi"), r[-1])),
            "lineages_passing": int(sum(out["lineages"][str(L)]["estimates"][est]["primary_vs_best_simple_select"]["pass"] for L in lin)),
        }
    out["primary_pass_3of3_oneshot"] = bool(len(lin) == CF_LINEAGES and out["paired"]["learned_oneshot"]["lineages_passing"] == len(lin))
    out["sequential_pass_all"] = bool(out["paired"]["learned_sequential"]["lineages_passing"] == len(lin))
    (mo, lo_, ho), = cluster_boot([oracle], eps, boot)
    out["oracle_arms_hidden_state"] = {"mean": mo, "lo": lo_, "hi": ho}
    X, names = PC["X"], PC["names"]
    drivers = {}
    for dname in DRIVERS:
        x = X[:, names.index(dname)]
        qs = np.unique(np.quantile(x, [1 / 3, 2 / 3]))
        b = np.searchsorted(qs, x, side="right")
        rows = []
        for kb in np.unique(b):
            m = b == kb
            row = {"bin": int(kb), "range": [float(x[m].min()), float(x[m].max())], "n": int(m.sum()),
                   "oracle": float(oracle[m].mean())}
            for L in lin:
                row[str(L)] = {k: float(per[L][k][m].mean()) for k in ("best_single", "best_simple", "learned_oneshot", "learned_sequential")}
            rows.append(row)
        drivers[dname] = rows
    out["drivers"] = drivers
    return out, per, oracle


def cmd_cf_score(a):
    t0 = time.process_time()
    out = Path(a.out)
    if (out / "score.json").exists():
        sys.exit(f"refusing: {out}/score.json exists (the confirm population is scored once)")
    pops = cf_populations(a.episodes, a.lineages, a.smoke)
    if not a.smoke and len(a.fits) != CF_LINEAGES:
        sys.exit(f"refusing: {len(a.fits)} fits given, registered design has {CF_LINEAGES} lineages")
    expect = a.expect_sha or [None] * len(a.fits)
    fits = {}
    for d, e in zip(a.fits, expect):
        f = cf_load_fit(d, e)
        fits[f["lineage"]] = f
    if len(fits) != len(a.fits):
        sys.exit("refusing: duplicate lineage among --fits")
    sel_sets = {f["meta"]["select"]["episode_list_sha256"] for f in fits.values()}
    if len(sel_sets) != 1:
        sys.exit("refusing: lineages were selected on different select populations")
    L = sorted(fits)
    for i in L:
        for j in L:
            if i < j and fits[i]["train_eps"] & fits[j]["train_eps"]:
                sys.exit(f"refusing: training populations of lineages {i} and {j} overlap")
    recs, unc = load(a.confirm)
    ceps = set(_check_population(recs, pops["confirm"], "confirm", unc))
    for f in fits.values():
        if ceps & (f["train_eps"] | f["select_eps"]):
            sys.exit("refusing: confirm episodes overlap a training/select population")
    PC = cf_matrices(recs)
    for f in fits.values():
        assert f["names"] == PC["names"], "feature names differ from the fit"
    res, per, oracle = cf_score(fits, PC, a.boot)
    certified = float(np.mean([r["certified"] for r in recs]))
    res.update({
        "registry": "A-CF-SMALL", "smoke": a.smoke, "populations": pops,
        "confirm": {"episodes": len(ceps), "instances": len(recs), "uncertified_episodes_legacy": len(unc),
                    "files": {str(p): _file_sha(p) for p in a.confirm}},
        "oracle_certified_fraction": certified,
        "uncertified_instances_scored_vs_UB": int(sum(not r["certified"] for r in recs)),
        "selector_feature_work_mean": float(PC["fwork"].mean()),
        "cpu_s": round(time.process_time() - t0, 1),
    })
    out.mkdir(parents=True, exist_ok=True)
    (out / "score.json").write_text(json.dumps(res, indent=1, default=float))
    flat = {"eps": PC["eps"], "oracle": oracle}
    for Lk, u in per.items():
        for k, v in u.items():
            if k != "_charge":
                flat[f"L{Lk}__" + "".join(ch if ch.isalnum() else "_" for ch in k)] = v
    np.savez_compressed(out / "per_instance.npz", **flat)
    md = cf_report_md(res)
    (out / "score.md").write_text(md)
    print(md)


def cf_report_md(s):
    c = s["confirm"]
    L = [f"# A-CF-SMALL confirmation{' (SMOKE, dev_builder seeds; not protocol)' if s['smoke'] else ''}", "",
         f"Confirm population: {c['instances']} instances / {c['episodes']} episodes; oracle certified "
         f"{s['oracle_certified_fraction']:.3f} ({s['uncertified_instances_scored_vs_UB']} scored vs UB). "
         f"Hidden-state oracle {s['oracle_arms_hidden_state']['mean']:.4f}.", "",
         "## Primary (per lineage): U(learned) − U(best simple chosen on select), episode-clustered 95% CI", "",
         "| lineage | best simple (select) | one-shot − best simple | pass | sequential − best simple | pass | one-shot − best single (GA-2 analogue) | seq − best single |",
         "|---|---|---|---|---|---|---|---|"]
    for Lk, r in s["lineages"].items():
        e1, e2 = r["estimates"]["learned_oneshot"], r["estimates"]["learned_sequential"]
        p1, p2 = e1["primary_vs_best_simple_select"], e2["primary_vs_best_simple_select"]
        g1, g2 = e1["GA2_analogue_vs_best_single"], e2["GA2_analogue_vs_best_single"]
        L.append(f"| {Lk} | {r['chosen']['best_simple']} | {p1['mean']:+.4f} [{p1['lo']:+.4f}, {p1['hi']:+.4f}] | {p1['pass']} | "
                 f"{p2['mean']:+.4f} [{p2['lo']:+.4f}, {p2['hi']:+.4f}] | {p2['pass']} | "
                 f"{g1['mean']:+.4f} [{g1['lo']:+.4f}, {g1['hi']:+.4f}] | {g2['mean']:+.4f} [{g2['lo']:+.4f}, {g2['hi']:+.4f}] |")
    for est, p in s["paired"].items():
        m = p["mean_over_lineages"]
        L.append(f"\n{est}: lineages passing {p['lineages_passing']}/{len(s['lineages'])}; mean over lineages (paired) "
                 f"{m['mean']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}]")
    L += ["", f"**Primary (one-shot, 3/3 lineages): {'PASS' if s['primary_pass_3of3_oneshot'] else 'FAIL'}**; "
          f"sequential all lineages: {s['sequential_pass_all']}.",
          "A pass establishes a small margin below the registered .02 practical-headroom threshold (registry practical_note).", "",
          "## Utilities on confirm (per lineage)", "",
          "U select = the select-population score used for choosing; charge = selector compute (features + inference) in "
          "work units, already subtracted from U (the tuned single/cascade/strong/hand families carry their own charges "
          "inside their utilities and show 0 here).", ""]
    for Lk, r in s["lineages"].items():
        L += [f"### lineage {Lk} (chosen: {r['chosen']}; fit.pkl {r['fit_pkl_sha256'][:12]})", "",
              "| policy | U confirm | 95% CI | U select | charge (work units) |", "|---|---|---|---|---|"]
        for k, v in sorted(r["utilities"].items(), key=lambda t: -t[1]["mean"]):
            ss = r["select_scores"].get(k)
            L.append(f"| {k} | {v['mean']:.4f} | {v['lo']:.4f} – {v['hi']:.4f} | {'' if ss is None else f'{ss:.4f}'} | "
                     f"{r['selector_charge_mean_work_units'][k]:.0f} |")
        h = r["hindsight_best_simple_on_confirm"]
        L += ["", f"Reported, not a gate: one-shot − strongest individual simple family in hindsight on confirm "
              f"({h['family']}): {h['oneshot_minus']['mean']:+.4f} [{h['oneshot_minus']['lo']:+.4f}, {h['oneshot_minus']['hi']:+.4f}]", ""]
    lins = list(s["lineages"])
    L += ["## Drivers (confirm tertiles; one-shot − best simple per lineage)", "",
          "| driver | range | n | oracle | " + " | ".join(f"L{x} best simple | L{x} 1-shot − simple | L{x} seq − simple" for x in lins) + " |",
          "|---|---|---|---|" + "---|" * (3 * len(lins))]
    for d, rows in s["drivers"].items():
        for r in rows:
            cells = []
            for x in lins:
                q = r[x]
                cells += [f"{q['best_simple']:.3f}", f"{q['learned_oneshot'] - q['best_simple']:+.4f}",
                          f"{q['learned_sequential'] - q['best_simple']:+.4f}"]
            L.append(f"| {d} | {r['range'][0]:.3g}–{r['range'][1]:.3g} | {r['n']} | {r['oracle']:.3f} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def cmd_cf_plan(a):
    """Root launch commands for A-CF-SMALL: evaluate chunks for train0..2 + select, then one
    fit job per lineage, then (after the fits are frozen) the confirm evaluate chunks, then
    one score job. Every job is 1 process."""
    pops = cf_populations(a.episodes, a.lineages, a.smoke)
    R = "/home/brand/structured-latent-dynamics-campaign06/results"
    env = ("env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src "
           "/home/brand/structured-latent-dynamics-campaign03/env/bin/python research/tools/campaign06_portfolio.py")
    rc = "python research/tools/campaign06_remote.py launch-cmd"
    sm = " --smoke" if a.smoke else ""
    geo = f" --episodes {a.episodes} --lineages {a.lineages}{sm}"
    per = math.ceil(a.episodes / a.chunks)
    files = {}

    def evals(pop):
        lo0, hi0 = pops[pop]
        for k in range(a.chunks):
            lo = lo0 + k * per
            n = min(per, hi0 - lo)
            if n <= 0:
                break
            job = f"{a.prefix}-{pop}-c{k}"
            files.setdefault(pop, []).append(f"{R}/{job}/records.jsonl.gz")
            print(f"{rc} {job} {a.sha} --wall-cap {a.wall_cap} --cpu-cap {a.cpu_cap} -- "
                  f"{env} evaluate --lo {lo} --episodes {n} --version {a.version} --out {R}/{job}/records.jsonl.gz")
    print("# step 1: evaluate the training populations and the shared select population (parallel)")
    for L in range(a.lineages):
        evals(f"train{L}")
    evals("select")
    print("# step 2 (after every step-1 chunk exits 0): fit + freeze one lineage per job (parallel)")
    for L in range(a.lineages):
        print(f"{rc} {a.prefix}-fit{L} {a.sha} --wall-cap {a.wall_cap} --cpu-cap {a.fit_cpu_cap} -- {env} cf-fit --lineage {L}{geo} "
              f"--train {' '.join(files[f'train{L}'])} --select {' '.join(files['select'])} --out {R}/{a.prefix}-fit{L}")
    print("# step 3 (after the fits exit 0; record each fit.json fit_pkl_sha256): evaluate the confirm population")
    evals("confirm")
    print("# step 4 (after the confirm chunks exit 0): score once (pass the recorded hashes as --expect-sha)")
    print(f"{rc} {a.prefix}-score {a.sha} --wall-cap {a.wall_cap} --cpu-cap {a.cpu_cap} -- {env} cf-score{geo} "
          f"--fits {' '.join(f'{R}/{a.prefix}-fit{L}' for L in range(a.lineages))} --expect-sha <SHA0> <SHA1> <SHA2> "
          f"--confirm {' '.join(files['confirm'])} --out {R}/{a.prefix}-score")


# ---------------------------------------------------------------------------
# plan

def cmd_plan(a):
    """Root launch commands for a chunked registered run: K evaluate chunks (1 core each),
    then (after all chunks finish) one headroom analysis job (4 processes)."""
    per = math.ceil(a.episodes / a.chunks)
    R = "/home/brand/structured-latent-dynamics-campaign06/results"
    env = ("env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src "
           "/home/brand/structured-latent-dynamics-campaign03/env/bin/python research/tools/campaign06_portfolio.py")
    ins = []
    for k in range(a.chunks):
        lo = a.lo + k * per
        n = min(per, a.lo + a.episodes - lo)
        job = f"{a.prefix}-c{k}"
        ins.append(f"{R}/{job}/records.jsonl.gz")
        print(f"python research/tools/campaign06_remote.py launch-cmd {job} {a.sha} --wall-cap {a.wall_cap} --cpu-cap {a.cpu_cap} -- "
              f"{env} evaluate --lo {lo} --episodes {n} --version {a.version} --out {R}/{job}/records.jsonl.gz")
    print(f"# after all chunks exit 0:")
    print(f"python research/tools/campaign06_remote.py launch-cmd {a.prefix}-analysis {a.sha} --wall-cap {a.wall_cap} "
          f"--cpu-cap {4 * a.cpu_cap} -- {env} headroom --inputs {' '.join(ins)} --out {R}/{a.prefix}-analysis --procs 4")


def main():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    e = s.add_parser("evaluate"); e.add_argument("--lo", type=int, required=True); e.add_argument("--episodes", type=int, required=True)
    e.add_argument("--out", required=True); e.add_argument("--knobs", default=None)
    e.add_argument("--version", default=None, help="generator knob set (default pw.GENERATOR_VERSION)"); e.set_defaults(f=cmd_evaluate)
    h = s.add_parser("headroom"); h.add_argument("--inputs", nargs="+", required=True); h.add_argument("--out", required=True)
    h.add_argument("--folds", type=int, default=5); h.add_argument("--boot", type=int, default=2000)
    h.add_argument("--inner", type=int, default=4); h.add_argument("--procs", type=int, default=4)
    h.set_defaults(f=cmd_headroom)
    pi = s.add_parser("pilot"); pi.add_argument("--lo", type=int, required=True); pi.add_argument("--out", required=True)
    pi.add_argument("--n", type=int, nargs="+", default=[20, 30, 40, 50]); pi.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    pi.add_argument("--per", type=int, default=24); pi.add_argument("--full", type=int, default=8); pi.set_defaults(f=cmd_pilot)
    pl = s.add_parser("plan"); pl.add_argument("--episodes", type=int, required=True); pl.add_argument("--chunks", type=int, required=True)
    pl.add_argument("--lo", type=int, required=True); pl.add_argument("--sha", required=True)
    pl.add_argument("--wall-cap", type=int, default=7200); pl.add_argument("--cpu-cap", type=int, default=7200)
    pl.add_argument("--prefix", default="e06-pw-ga"); pl.add_argument("--version", default=pw.GENERATOR_VERSION)
    pl.set_defaults(f=cmd_plan)

    def geo(sp):
        sp.add_argument("--episodes", type=int, default=CF_EPISODES); sp.add_argument("--lineages", type=int, default=CF_LINEAGES)
        sp.add_argument("--smoke", action="store_true", help="dev_builder smoke populations (never protocol)")
    cp = s.add_parser("cf-populations"); geo(cp)
    cp.set_defaults(f=lambda a: print(json.dumps(cf_populations(a.episodes, a.lineages, a.smoke), indent=1)))
    cf = s.add_parser("cf-fit"); cf.add_argument("--lineage", type=int, required=True)
    cf.add_argument("--train", nargs="+", required=True); cf.add_argument("--select", nargs="+", required=True)
    cf.add_argument("--out", required=True); geo(cf); cf.set_defaults(f=cmd_cf_fit)
    cs = s.add_parser("cf-score"); cs.add_argument("--fits", nargs="+", required=True); cs.add_argument("--confirm", nargs="+", required=True)
    cs.add_argument("--out", required=True); cs.add_argument("--boot", type=int, default=2000)
    cs.add_argument("--expect-sha", nargs="+", default=None, help="fit.pkl sha256 (prefixes) recorded after step 2, in --fits order")
    geo(cs); cs.set_defaults(f=cmd_cf_score)
    cpl = s.add_parser("cf-plan"); cpl.add_argument("--sha", required=True); cpl.add_argument("--chunks", type=int, default=2)
    cpl.add_argument("--wall-cap", type=int, default=3600); cpl.add_argument("--cpu-cap", type=int, default=1200)
    cpl.add_argument("--fit-cpu-cap", type=int, default=3600); cpl.add_argument("--prefix", default="e06-pw-acf")
    cpl.add_argument("--version", default=pw.GENERATOR_VERSION); geo(cpl); cpl.set_defaults(f=cmd_cf_plan)
    a = p.parse_args()
    if a.cmd in ("evaluate", "pilot"):
        lo, hi = a.lo, a.lo + (a.episodes if a.cmd == "evaluate" else a.per * len(a.n) * len(a.classes))
        assert pw.check_seed_ranges()
        ok = any(r0 <= lo and hi <= r1 for r0, r1 in pw.SEED_RANGES.values())
        if not ok:
            sys.exit(f"refusing: episode seeds [{lo},{hi}) not inside one registered sub-range {pw.SEED_RANGES}")
    a.f(a)


if __name__ == "__main__":
    main()
