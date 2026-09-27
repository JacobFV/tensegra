"""Extended-06 Track A: the portfolio world ("portworld").

A family of constrained packing/assignment instances with complementary solution
methods whose best choice depends on observable structure and on public prices.
Pure Python + numpy (numpy only for seeded sampling). See
research/campaigns/extended-06/portworld.md for the specification, the method
contracts and the generator revision log.

Instance (true)
---------------
n items, each with integer value v_i >= 1, group g(i) in 0..G-1, local weight wl_i
and global weight wg_i. Constraints:

* local capacity per group: sum_{i in S, g(i)=g} wl_i <= capl_g;
* global capacity: sum_{i in S} wg_i <= capg (may be non-binding);
* conflicts: known conflict pairs (public) + *possible* conflict pairs (public
  candidates, each real with public prior probability 1/2, true status hidden);
* requires: i in S => parent(i) in S (a forest inside each group, public);
* hidden capacities: for a hidden group only a public interval [lo, hi] is known;
  the true capacity is uniform on the integers of the interval (explicit posterior).

``inspect`` reveals all hidden parameters of an instance at obs_price per parameter.

Episodes: k related instances. Instance t is a perturbation of instance t-1 (the
episode's instance -1 is an unevaluated base). The cache for instance t is the
certified optimum of the true instance t-1 (a supplied, policy-independent cached
solution; it may be reusable or stale). The perturbation summary (counts of changed
values/weights/capacities/conflicts and the magnitude m) is public.

Planning modes
--------------
* ``cons``: plan on the conservative public instance (hidden capacities at lo,
  possible conflicts treated as real). Every solution is truly feasible.
* ``insp``: inspect everything first (n_hidden observations), plan on the truth.

Work units are deterministic counts of item-level operations (see each method).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

GENERATOR_VERSION = "pw-v3"   # default knob set; see KNOB_SETS and the revision log

# ---------------------------------------------------------------------------
# Seed ranges (episode seeds; one seed per episode). Asserted disjoint and inside
# the registered extended-06 Track A block [300,000,000, 340,000,000).
SEED_RANGES = {
    "dev_builder": (300_000_000, 305_000_000),   # builder development iterations (this study)
    "dev_gate": (305_000_000, 310_000_000),      # registered headroom gate run (root, fresh)
    "train": (310_000_000, 320_000_000),         # Phase-2 learner training
    "select": (320_000_000, 330_000_000),        # Phase-2 model/hyperparameter selection
    "confirm": (330_000_000, 340_000_000),       # Phase-2 sealed confirmation
}
TRACK_A_BLOCK = (300_000_000, 340_000_000)


def check_seed_ranges(ranges=SEED_RANGES):
    items = sorted(ranges.values())
    for lo, hi in items:
        assert TRACK_A_BLOCK[0] <= lo < hi <= TRACK_A_BLOCK[1], (lo, hi)
    for (a0, a1), (b0, b1) in zip(items, items[1:]):
        assert a1 <= b0, "seed sub-ranges overlap"
    return True


# ---------------------------------------------------------------------------
# Generator knobs (drawn per episode; all resulting quantities are public except
# the hidden capacities and the possible-conflict statuses).
DEFAULT_KNOBS = {
    "n_range": (16, 52),
    "groups_choices": (1, 2, 3, 4, 6, 8),
    "corr_range": (0.0, 1.0),          # value-weight correlation (hardness)
    "tight_loc_range": (0.2, 0.8),
    "tight_glob_range": (0.2, 1.4),    # >= ~1: global constraint non-binding
    "p_in_range": (0.0, 0.25),         # within-group conflict density
    "p_cross_zero": 0.5, "p_cross_range": (0.0, 0.03),
    "req_range": (0.0, 0.3),
    "hidden_cap_zero": 0.4, "hidden_cap_range": (0.2, 1.0),
    "width_range": (0.1, 0.5),
    "possible_zero": 0.5, "possible_range": (0.0, 0.25),  # per item, within-group candidates
    "k_range": (3, 6),
    "m_zero": 0.2, "m_range": (0.02, 0.5),
    "c_log10": (-9.0, -5.5),           # compute price per work unit
    "o_log10": (-3.5, -1.5),           # observation price per hidden parameter
    "L_range": (0.5, 2.0),             # failure loss
    "pert_value": (0.7, 1.3),          # value factor for a changed item
    "pert_weight_rate": 0.5,           # weight-change probability = rate * m
    "pert_weight": (0.8, 1.25),
    "pert_swap": 1 / 3,                # conflict edges swapped = round(swap * m * |E|)
}
# Revision log (reasons and numbers in research/campaigns/extended-06/portworld.md):
# pw-v2: observation price range widened upward (inspect vs conservative was one-sided:
#   inspection paid on ~all instances); perturbations made able to break caches
#   (magnitude up to 1, larger value/weight changes, more conflict churn); conflict
#   density range widened.
PILOT_KNOBS = dict(DEFAULT_KNOBS, o_log10=(-3.0, -0.7), m_range=(0.05, 1.0), pert_value=(0.5, 1.5),
                   pert_weight_rate=0.6, pert_weight=(0.7, 1.4), pert_swap=0.5, p_in_range=(0.0, 0.35),
                   corr_classes=(0, 1, 2, 3, 4), n_max_by_class={0: 999, 1: 999, 2: 999, 3: 999, 4: 999},
                   L_range=(0.2, 2.0))
KNOB_SETS = {
    "pw-v1": dict(DEFAULT_KNOBS),
    "pw-v2": dict(DEFAULT_KNOBS, o_log10=(-3.0, -0.7), m_range=(0.05, 1.0), pert_value=(0.5, 1.5),
                  pert_weight_rate=0.6, pert_weight=(0.7, 1.4), pert_swap=0.5, p_in_range=(0.0, 0.35)),
    # pw-v3 (design v2 compliance): Pisinger correlation classes with a public per-class size
    # rule from the oracle pilot (p99 certification <= ~5 core-s: classes 2/3 capped at 40);
    # prices centred by the stated crossover rule (pilot, insp mode): c* = median(B&B gain
    # over greedy) / median(B&B work) = 5.0e-7, o* = median(inspection gain) / median(n_hidden)
    # = 9.7e-3; each sampled +-1 decade (log-uniform); failure loss U[0.2, 2].
    "pw-v3": dict(PILOT_KNOBS, n_range=(16, 50), n_max_by_class={0: 50, 1: 50, 2: 40, 3: 40, 4: 50},
                  c_log10=(-7.3, -5.3), o_log10=(-3.0, -1.0), L_range=(0.2, 2.0)),
}


@dataclass
class Instance:
    n: int
    G: int
    v: list
    grp: list
    wl: list
    wg: list
    capl_true: list
    cap_lo: list
    cap_hi: list
    hidden_cap: list
    capg: int
    conflicts: list            # known (public) pairs (i<j)
    possible: list             # public candidate pairs (i<j)
    possible_real: list        # hidden truth
    parent: list
    cache: int | None = None   # bitmask of the cached solution (public)
    prices: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)   # public episode/perturbation info

    def n_hidden(self):
        return sum(self.hidden_cap) + len(self.possible)


@dataclass(frozen=True)
class PublicView:
    """Everything a policy may see before inspecting. No hidden truth is stored."""
    n: int
    G: int
    v: tuple
    grp: tuple
    wl: tuple
    wg: tuple
    cap_lo: tuple
    cap_hi: tuple
    hidden_cap: tuple
    capg: int
    conflicts: tuple
    possible: tuple
    parent: tuple
    cache: int | None
    prices: tuple   # (c, o, L)
    meta: tuple     # sorted (key, value) pairs


def public_view(inst: Instance) -> PublicView:
    lo = tuple(inst.cap_lo[g] if inst.hidden_cap[g] else inst.capl_true[g] for g in range(inst.G))
    hi = tuple(inst.cap_hi[g] if inst.hidden_cap[g] else inst.capl_true[g] for g in range(inst.G))
    return PublicView(inst.n, inst.G, tuple(inst.v), tuple(inst.grp), tuple(inst.wl), tuple(inst.wg), lo, hi,
                      tuple(inst.hidden_cap), inst.capg, tuple(inst.conflicts), tuple(inst.possible), tuple(inst.parent),
                      inst.cache, (inst.prices["c"], inst.prices["o"], inst.prices["L"]), tuple(sorted(inst.meta.items())))


class Inspector:
    """Holds the hidden truth; reveals it at a counted price."""

    def __init__(self, inst: Instance):
        self._inst = inst
        self.count = 0

    def reveal_all(self):
        self.count += self._inst.n_hidden()
        return list(self._inst.capl_true), [p for p, r in zip(self._inst.possible, self._inst.possible_real) if r]


# ---------------------------------------------------------------------------
# Generator

def _draw_structure(rng, kn):
    cls = -1
    if kn.get("corr_classes"):
        cls = int(rng.choice(kn["corr_classes"]))
        # public size rule per class (n_max from the oracle pilot)
        n = int(rng.integers(kn["n_range"][0], min(kn["n_range"][1], kn["n_max_by_class"][cls]) + 1))
    else:
        n = int(rng.integers(kn["n_range"][0], kn["n_range"][1] + 1))
    G = int(rng.choice(kn["groups_choices"]))
    G = max(1, min(G, n // 3))
    grp = sorted(int(x) for x in np.concatenate([np.arange(G), rng.integers(0, G, n - G)]))
    corr = float(rng.uniform(*kn["corr_range"]))
    wl = [int(x) for x in rng.integers(5, 61, n)]
    wg = [int(x) for x in rng.integers(5, 61, n)]
    if cls >= 0:
        # Pisinger-style value-weight correlation classes on the mean weight (public class id)
        wbar = [(a + b) / 2 for a, b in zip(wl, wg)]
        if cls == 0:     # uncorrelated
            v = [int(x) for x in rng.integers(5, 61, n)]
        elif cls == 1:   # weakly correlated
            v = [max(1, int(round(x + rng.uniform(-10, 10)))) for x in wbar]
        elif cls == 2:   # strongly correlated
            v = [int(round(x)) + 10 for x in wbar]
        elif cls == 3:   # almost strongly correlated
            v = [max(1, int(round(x + 10 + rng.uniform(-2, 2)))) for x in wbar]
        else:            # subset-sum-like
            v = [max(1, int(round(x))) for x in wbar]
        corr = {0: 0.0, 1: 0.5, 2: 1.0, 3: 0.95, 4: 1.0}[cls]
    else:
        v = [max(1, int(round(corr * (a + b) / 2 + (1 - corr) * rng.uniform(5, 60) + rng.uniform(-4, 4)))) for a, b in zip(wl, wg)]
    tl = float(rng.uniform(*kn["tight_loc_range"]))
    tg = float(rng.uniform(*kn["tight_glob_range"]))
    width = float(rng.uniform(*kn["width_range"]))
    hid_frac = 0.0 if rng.random() < kn["hidden_cap_zero"] else float(rng.uniform(*kn["hidden_cap_range"]))
    hidden_cap = [bool(rng.random() < hid_frac) for _ in range(G)]
    mu = []
    for g in range(G):
        s = sum(wl[i] for i in range(n) if grp[i] == g)
        mu.append(max(5, int(round(s * float(np.clip(tl + rng.uniform(-0.1, 0.1), 0.1, 0.95))))))
    capg = max(5, int(round(tg * sum(wg))))
    p_in = float(rng.uniform(*kn["p_in_range"]))
    p_cross = 0.0 if rng.random() < kn["p_cross_zero"] else float(rng.uniform(*kn["p_cross_range"]))
    conflicts = []
    cand_in = []
    for i in range(n):
        for j in range(i + 1, n):
            if grp[i] == grp[j]:
                if rng.random() < p_in:
                    conflicts.append((i, j))
                else:
                    cand_in.append((i, j))
            elif rng.random() < p_cross:
                conflicts.append((i, j))
    poss_rate = 0.0 if rng.random() < kn["possible_zero"] else float(rng.uniform(*kn["possible_range"]))
    n_poss = min(len(cand_in), int(round(poss_rate * n)))
    idx = rng.permutation(len(cand_in))[:n_poss] if n_poss else []
    possible = sorted(cand_in[int(t)] for t in idx)
    possible_real = [bool(rng.random() < 0.5) for _ in possible]
    req = float(rng.uniform(*kn["req_range"]))
    parent = [-1] * n
    for i in range(n):
        earlier = [j for j in range(i) if grp[j] == grp[i]]
        if earlier and rng.random() < req:
            parent[i] = int(earlier[int(rng.integers(len(earlier)))])
    knobs = {"corr": corr, "corr_class": cls, "tight_loc": tl, "tight_glob": tg, "width": width, "p_in": p_in, "p_cross": p_cross,
             "hid_frac": hid_frac, "poss_rate": poss_rate, "req": req}
    return dict(n=n, G=G, grp=grp, v=v, wl=wl, wg=wg, mu=mu, width=width, hidden_cap=hidden_cap, capg=capg,
                conflicts=conflicts, possible=possible, possible_real=possible_real, parent=parent), knobs


def _cap_interval(mu, width, rng):
    lo = max(1, int(round(mu * (1 - width))))
    hi = max(lo, int(round(mu * (1 + width))))
    return lo, hi, int(rng.integers(lo, hi + 1))


def _materialize(st, rng, prices, meta, cache=None):
    G = st["G"]
    lo, hi, tru = [], [], []
    for g in range(G):
        if st["hidden_cap"][g]:
            a, b, t = st["cap_draw"][g]
        else:
            a = b = t = st["mu"][g]
        lo.append(a); hi.append(b); tru.append(t)
    return Instance(n=st["n"], G=G, v=list(st["v"]), grp=list(st["grp"]), wl=list(st["wl"]), wg=list(st["wg"]),
                    capl_true=tru, cap_lo=lo, cap_hi=hi, hidden_cap=list(st["hidden_cap"]), capg=st["capg"],
                    conflicts=list(st["conflicts"]), possible=list(st["possible"]), possible_real=list(st["possible_real"]),
                    parent=list(st["parent"]), cache=cache, prices=dict(prices), meta=dict(meta))


def _perturb(st, m, rng, kn):
    st = {k: (list(x) if isinstance(x, list) else x) for k, x in st.items()}
    n, G = st["n"], st["G"]
    nv = nw = ncap = 0
    for i in range(n):
        if rng.random() < m:
            st["v"][i] = max(1, int(round(st["v"][i] * rng.uniform(*kn["pert_value"])))); nv += 1
        if rng.random() < m * kn["pert_weight_rate"]:
            st["wl"][i] = max(1, int(round(st["wl"][i] * rng.uniform(*kn["pert_weight"]))))
            st["wg"][i] = max(1, int(round(st["wg"][i] * rng.uniform(*kn["pert_weight"])))); nw += 1
    st["cap_draw"] = list(st["cap_draw"])
    for g in range(G):
        if rng.random() < m:
            st["mu"][g] = max(5, int(round(st["mu"][g] * rng.uniform(0.8, 1.2)))); ncap += 1
            st["cap_draw"][g] = _cap_interval(st["mu"][g], st["width"], rng)
    if rng.random() < m:
        st["capg"] = max(5, int(round(st["capg"] * rng.uniform(0.85, 1.15)))); ncap += 1
    E = len(st["conflicts"])
    nswap = min(E, int(round(kn["pert_swap"] * m * E))) if E else 0
    nadd = 0
    if nswap:
        keep = sorted(int(t) for t in rng.permutation(E)[: E - nswap])
        conf = [st["conflicts"][t] for t in keep]
        cs = set(conf) | set(st["possible"])
        tries = 0
        while nadd < nswap and tries < 50 * nswap:
            tries += 1
            i, j = sorted(int(x) for x in rng.integers(0, n, 2))
            if i != j and st["grp"][i] == st["grp"][j] and (i, j) not in cs:
                conf.append((i, j)); cs.add((i, j)); nadd += 1
        st["conflicts"] = sorted(conf)
    return st, {"m": float(m), "n_value_changed": nv, "n_weight_changed": nw, "n_cap_changed": ncap,
                "n_conf_removed": nswap, "n_conf_added": nadd}


def generate_episode(seed: int, knobs=None, version=None):
    """Return (instances, knob-draw) for episode seed. Caches are filled by evaluate_episode."""
    kn = dict(KNOB_SETS[version or GENERATOR_VERSION], **(knobs or {}))
    rng = np.random.default_rng(seed)
    st, kd = _draw_structure(rng, kn)
    st["cap_draw"] = [_cap_interval(st["mu"][g], st["width"], rng) for g in range(st["G"])]
    k = int(rng.integers(kn["k_range"][0], kn["k_range"][1] + 1))
    prices = {"c": float(10 ** rng.uniform(*kn["c_log10"])), "o": float(10 ** rng.uniform(*kn["o_log10"])),
              "L": float(rng.uniform(*kn["L_range"]))}
    states, metas = [st], [{"m": 0.0}]
    for t in range(k):
        m = 0.0 if rng.random() < kn["m_zero"] else float(rng.uniform(*kn["m_range"]))
        nst, pm = _perturb(states[-1], m, rng, kn)
        states.append(nst); metas.append(pm)
    insts = []
    for t, (s, pm) in enumerate(zip(states, metas)):
        meta = dict(pm, episode_seed=seed, t=t - 1, k=k, width=kd["width"], corr_class=kd["corr_class"])
        insts.append(_materialize(s, rng, prices, meta))
    return insts, kd   # insts[0] is the unevaluated base (t = -1)


# ---------------------------------------------------------------------------
# Planning instances and solvers

class Work:
    __slots__ = ("units",)

    def __init__(self):
        self.units = 0


class Plan:
    """A concrete (fully specified) instance to plan on, possibly a restriction."""

    def __init__(self, v, grp, wl, wg, capl, capg, conf_pairs, parent, ids=None, surrogate=False):
        n = len(v)
        self.surrogate = surrogate
        self.n = n
        self.v, self.wl, self.wg = list(v), list(wl), list(wg)
        gids = sorted(set(grp))
        remap = {g: t for t, g in enumerate(gids)}
        self.gorig = gids
        self.g = [remap[x] for x in grp]
        self.G = len(gids)
        self.capl = [capl[g] for g in gids]
        self.capg = capg
        self.conf = [0] * n
        self.adj = [[] for _ in range(n)]
        for i, j in conf_pairs:
            if i != j and not (self.conf[i] >> j) & 1:
                self.conf[i] |= 1 << j; self.conf[j] |= 1 << i
                self.adj[i].append(j); self.adj[j].append(i)
        self.parent = list(parent)
        self.children = [[] for _ in range(n)]
        for i, p in enumerate(parent):
            if p >= 0:
                self.children[p].append(i)
        self.ids = list(ids) if ids is not None else list(range(n))
        sl = [0] * self.G
        for i in range(n):
            sl[self.g[i]] += self.wl[i]
        self.nbl = [sl[g] <= self.capl[g] for g in range(self.G)]
        self.nbg = sum(self.wg) <= capg
        self.members = [[i for i in range(n) if self.g[i] == g] for g in range(self.G)]
        # ratio order (parents before children), per-constraint fractional orders
        self.ratio = [self._ratio(i) for i in range(n)]
        base = sorted(range(n), key=lambda i: (-self.ratio[i], i))
        order, placed = [], [False] * n

        def place(i):
            if placed[i]:
                return
            if self.parent[i] >= 0:
                place(self.parent[i])
            placed[i] = True; order.append(i)
        for i in base:
            place(i)
        self.order = order
        self.pos = [0] * n
        for k, i in enumerate(order):
            self.pos[i] = k
        self.gorder = sorted(range(n), key=lambda i: (-self.v[i] / self.wg[i], i))
        self.lorder = [sorted(self.members[g], key=lambda i: (-self.v[i] / self.wl[i], i)) for g in range(self.G)]
        if surrogate:
            # surrogate relaxation (oracle only): binding rows aggregated with multipliers
            # 1/capacity; the fractional knapsack of the aggregate row is a valid bound
            self.mu = [0.0 if self.nbl[g] else 1.0 / max(self.capl[g], 1) for g in range(self.G)]
            self.nu = 0.0 if self.nbg else 1.0 / max(capg, 1)
            self.agg = [self.mu[self.g[i]] * self.wl[i] + self.nu * self.wg[i] for i in range(n)]
            self.sorder = sorted(range(n), key=lambda i: (-(self.v[i] / self.agg[i]) if self.agg[i] > 0 else -math.inf, i))

    def _ratio(self, i):
        d = self.wl[i] / max(1, self.capl[self.g[i]])
        if not self.nbg:
            d += self.wg[i] / max(1, self.capg)
        return self.v[i] / max(d, 1e-9)

    # --- helpers
    def value(self, sel):
        s, i = 0, 0
        while sel:
            if sel & 1:
                s += self.v[i]
            sel >>= 1; i += 1
        return s

    def items(self, sel):
        out, i = [], 0
        while sel:
            if sel & 1:
                out.append(i)
            sel >>= 1; i += 1
        return out

    def loads(self, sel):
        ll = [0] * self.G
        lg = 0
        for i in self.items(sel):
            ll[self.g[i]] += self.wl[i]; lg += self.wg[i]
        return ll, lg

    def feasible(self, sel):
        ll, lg = self.loads(sel)
        if lg > self.capg or any(ll[g] > self.capl[g] for g in range(self.G)):
            return False
        for i in self.items(sel):
            if self.conf[i] & sel:
                return False
            p = self.parent[i]
            if p >= 0 and not (sel >> p) & 1:
                return False
        return True

    def restrict(self, items, capl, capg, fixed_in):
        """Sub-plan on `items` (plan indices) with remaining capacities; parents in fixed_in dropped."""
        idx = {i: t for t, i in enumerate(items)}
        conf = [(idx[i], idx[j]) for i in items for j in self.adj[i] if j in idx and idx[i] < idx[j]]
        par = []
        for i in items:
            p = self.parent[i]
            par.append(idx[p] if p in idx else -1)
        return Plan([self.v[i] for i in items], [self.g[i] for i in items], [self.wl[i] for i in items],
                    [self.wg[i] for i in items], capl, capg, conf, par, ids=[self.ids[i] for i in items],
                    surrogate=self.surrogate)

    # --- fractional bound over "free" items given a partial state
    def frac_bound(self, freemask, reml, remg, work):
        """Upper bound on the value addable from items in freemask: min(global fractional
        knapsack, sum of per-group fractional knapsacks). Work += items scanned."""
        v, wl, wg = self.v, self.wl, self.wg
        scanned = 0
        if self.nbg:
            gb = None
        else:
            gb = 0.0
            cap = remg
            for i in self.gorder:
                scanned += 1
                if not (freemask >> i) & 1:
                    continue
                if wg[i] <= cap:
                    gb += v[i]; cap -= wg[i]
                else:
                    gb += v[i] * cap / wg[i]
                    break
        lb = 0.0
        for g in range(self.G):
            cap = reml[g]
            nb = self.nbl[g]
            for i in self.lorder[g]:
                scanned += 1
                if not (freemask >> i) & 1:
                    continue
                if nb or wl[i] <= cap:
                    lb += v[i]; cap -= wl[i]
                else:
                    lb += v[i] * cap / wl[i]
                    break
        best = lb if gb is None else min(gb, lb)
        if self.surrogate and (self.nu > 0 or any(self.mu)):
            cap = sum(self.mu[g] * reml[g] for g in range(self.G)) + self.nu * remg
            sb = 0.0
            for i in self.sorder:
                scanned += 1
                if not (freemask >> i) & 1:
                    continue
                a = self.agg[i]
                if a <= cap:
                    sb += v[i]; cap -= a
                else:
                    sb += v[i] * max(cap, 0.0) / a
                    break
            best = min(best, sb)
        work.units += scanned
        return best


def check_solution(inst: Instance, sel: int):
    """Exact checker against the TRUE instance. Returns (feasible, value)."""
    if sel is None:
        return False, 0
    items = [i for i in range(inst.n) if (sel >> i) & 1]
    ll = [0] * inst.G
    lg = 0
    for i in items:
        ll[inst.grp[i]] += inst.wl[i]; lg += inst.wg[i]
    ok = lg <= inst.capg and all(ll[g] <= inst.capl_true[g] for g in range(inst.G))
    real = set(inst.conflicts) | {p for p, r in zip(inst.possible, inst.possible_real) if r}
    for (i, j) in real:
        if (sel >> i) & 1 and (sel >> j) & 1:
            ok = False
    for i in items:
        p = inst.parent[i]
        if p >= 0 and not (sel >> p) & 1:
            ok = False
    return ok, sum(inst.v[i] for i in items)


def plan_from_public(pv: PublicView, mode: str, inspector: Inspector | None = None) -> Plan:
    """cons: hidden caps at lo, possible conflicts real (always truly feasible). insp: reveal
    the truth (counted). opt: posterior-median caps, possible conflicts absent (may fail)."""
    if mode == "cons":
        capl, confs = list(pv.cap_lo), list(pv.conflicts) + list(pv.possible)
    elif mode == "insp":
        tru, real = inspector.reveal_all()
        capl, confs = tru, list(pv.conflicts) + real
    elif mode == "opt":   # no inspection: hidden caps at the posterior median, possible conflicts absent
        capl, confs = [(lo + hi) // 2 for lo, hi in zip(pv.cap_lo, pv.cap_hi)], list(pv.conflicts)
    else:
        raise ValueError(mode)
    return Plan(pv.v, pv.grp, pv.wl, pv.wg, capl, pv.capg, confs, pv.parent)


def true_plan(inst: Instance, surrogate=False) -> Plan:
    real = [p for p, r in zip(inst.possible, inst.possible_real) if r]
    return Plan(inst.v, inst.grp, inst.wl, inst.wg, inst.capl_true, inst.capg, list(inst.conflicts) + real, inst.parent,
                surrogate=surrogate)


# --- greedy -----------------------------------------------------------------

def _closure(P, sel, i):
    """i plus its unselected ancestors (parents first)."""
    out = []
    while i >= 0 and not (sel >> i) & 1:
        out.append(i); i = P.parent[i]
    return out[::-1]


def _try_add(P, sel, ll, lg, items, work):
    """Feasibility of adding `items` (closure) to sel; returns (ok, new_ll, new_lg, addmask)."""
    add = 0
    for i in items:
        add |= 1 << i
    work.units += len(items)
    ll2 = list(ll)
    lg2 = lg
    for i in items:
        if P.conf[i] & (sel | add):
            return False, None, None, 0
        ll2[P.g[i]] += P.wl[i]; lg2 += P.wg[i]
        if ll2[P.g[i]] > P.capl[P.g[i]] and not P.nbl[P.g[i]]:
            return False, None, None, 0
    if lg2 > P.capg:
        return False, None, None, 0
    return True, ll2, lg2, add


def greedy(P: Plan, work: Work, start: int = 0):
    """Contract: input plan (+ optional feasible start); output feasible solution (bitmask);
    no optimality guarantee. Work: n*ceil(log2 n) (sort) + items touched by add checks."""
    n = P.n
    work.units += n * max(1, math.ceil(math.log2(max(n, 2))))
    sel = start
    ll, lg = P.loads(sel)
    for i in P.order:
        if (sel >> i) & 1:
            continue
        cl = _closure(P, sel, i)
        ok, ll2, lg2, add = _try_add(P, sel, ll, lg, cl, work)
        if ok:
            sel |= add; ll, lg = ll2, lg2
    return sel


# --- local repair (resumable) -----------------------------------------------

class Repair:
    """Local search from a feasible start: add moves (with requires-closure) and 1-for-1
    swaps (remove j and its selected descendants, add i with closure) taken on first
    improvement. Anytime: never worsens. Resumable: run(budget) may be called repeatedly;
    the trajectory is identical to one uninterrupted run (tested). Work: items touched per
    move evaluation. Finishes at a local optimum (status 'local_opt')."""

    def __init__(self, P: Plan, start: int):
        assert P.feasible(start)
        self.P, self.sel = P, start
        self.work = Work()
        self.done = False
        self.moves = 0
        self._gen = self._steps()

    def _desc(self, j):
        P, sel, out, st = self.P, self.sel, 0, [j]
        while st:
            x = st.pop()
            if (sel >> x) & 1 and not (out >> x) & 1:
                out |= 1 << x
                st.extend(P.children[x])
        return out

    def _steps(self):
        P = self.P
        while True:
            improved = False
            ll, lg = P.loads(self.sel)
            for i in P.order:
                if (self.sel >> i) & 1:
                    continue
                cl = _closure(P, self.sel, i)
                ok, ll2, lg2, add = _try_add(P, self.sel, ll, lg, cl, self.work)
                yield
                if ok:
                    self.sel |= add; improved = True; self.moves += 1
                    break
                gain_add = sum(P.v[x] for x in cl)
                anc = 0
                x = P.parent[i]
                while x >= 0:
                    anc |= 1 << x; x = P.parent[x]
                for j in reversed(P.order):
                    if not (self.sel >> j) & 1:
                        continue
                    rm = self._desc(j)
                    self.work.units += 1
                    addm = 0
                    for x in cl:
                        addm |= 1 << x
                    if rm & (addm | anc):
                        continue
                    rmv = P.value(rm)
                    if rmv >= gain_add:
                        continue
                    base = self.sel & ~rm
                    bl, bg = P.loads(base)
                    ok2, _, _, add2 = _try_add(P, base, bl, bg, cl, self.work)
                    yield
                    if ok2:
                        self.sel = base | add2; improved = True; self.moves += 1
                        break
                if improved:
                    break
            if not improved:
                return

    def run(self, budget):
        while not self.done and self.work.units < budget:
            try:
                next(self._gen)
            except StopIteration:
                self.done = True
        return self.sel


def repair_budget(n):
    return 20 * n * n + 50


# geometric trace grid (work units) at which anytime B&B runs record their certified UB;
# every budget read off a trace is a grid point
TRACE_GRID = tuple(sorted({int(round(1000 * 2 ** (k / 2))) for k in range(0, 21)} | {10_000, 100_000, 1_000_000}))
TRACE_MAX = max(TRACE_GRID)


# --- branch and bound (anytime, resumable, certifying) ----------------------

class BnB:
    """DFS branch-and-bound over P.order (include branch first) with the fractional bound
    min(global, sum of per-group) over free items. Anytime: `log` records (work, value,
    sel) at each incumbent improvement. Resumable: run(budget) may be called repeatedly
    (explicit stack). Certifies optimality when the stack empties (finished=True).
    Work: 1 per node + items scanned by the bound."""

    def __init__(self, P: Plan, incumbent: int = 0):
        assert P.feasible(incumbent)
        self.P = P
        self.work = Work()
        self.inc = incumbent
        self.inc_val = P.value(incumbent)
        self.log = [(-1, self.inc_val, incumbent)]
        self.last_start = -1
        self.finished = False
        self.nodes = 0
        self.root_bound = None
        # state
        self.sel = 0
        self.val = 0
        self.ll = list([0] * P.G)
        self.lg = 0
        self.stack = [[0, 0, False, None]]   # [k, stage, included, node bound]
        self.cps = list(TRACE_GRID)   # budgets at which the certified global UB is recorded
        self.ub_at = {}

    def global_ub(self):
        """Certified upper bound on the plan optimum in the current state: the incumbent,
        the node bounds of frames whose exclude branch is pending (stage 1), and the bound
        of the parent of an unexpanded top frame."""
        if not self.stack:
            return self.inc_val
        ub = self.inc_val
        for fr in self.stack:
            if fr[1] == 1 and fr[3] is not None:
                ub = max(ub, math.floor(fr[3] + 1e-9))
        if self.stack[-1][1] == 0:
            if len(self.stack) >= 2 and self.stack[-2][3] is not None:
                ub = max(ub, math.floor(self.stack[-2][3] + 1e-9))
            elif len(self.stack) == 1:
                ub = max(ub, 10 ** 12)   # nothing expanded yet
        return ub

    def _freemask(self, k):
        P = self.P
        sel, fm = self.sel, 0
        reml = [P.capl[g] - self.ll[g] for g in range(P.G)]
        remg = P.capg - self.lg
        for t in range(k, P.n):
            i = P.order[t]
            if P.conf[i] & sel:
                continue
            p = P.parent[i]
            if p >= 0 and not (sel >> p) & 1 and P.pos[p] < k:
                continue
            if P.wg[i] > remg or (not P.nbl[P.g[i]] and P.wl[i] > reml[P.g[i]]):
                continue
            fm |= 1 << i
        self.work.units += P.n - k
        return fm, reml, remg

    def run(self, budget):
        P = self.P
        order, n = P.order, P.n
        st = self.stack
        cps = self.cps
        while st and self.work.units < budget:
            while cps and self.work.units >= cps[0]:
                self.ub_at[cps.pop(0)] = self.global_ub()
            w0 = self.last_start = self.work.units
            fr = st[-1]
            k, stage = fr[0], fr[1]
            if stage == 0:
                self.nodes += 1
                self.work.units += 1
                if k == n:
                    if self.val > self.inc_val:
                        self.inc, self.inc_val = self.sel, self.val
                        self.log.append((w0, self.val, self.sel))
                    st.pop(); continue
                fm, reml, remg = self._freemask(k)
                if fm == 0:
                    if self.val > self.inc_val:
                        self.inc, self.inc_val = self.sel, self.val
                        self.log.append((w0, self.val, self.sel))
                    st.pop(); continue
                ub = self.val + P.frac_bound(fm, reml, remg, self.work)
                if self.root_bound is None:
                    self.root_bound = ub
                fr[3] = ub
                if math.floor(ub + 1e-9) <= self.inc_val:
                    st.pop(); continue
                i = order[k]
                fr[1] = 1
                if (fm >> i) & 1:
                    # include
                    self.sel |= 1 << i
                    self.val += P.v[i]
                    self.ll[P.g[i]] += P.wl[i]; self.lg += P.wg[i]
                    fr[2] = True
                    st.append([k + 1, 0, False, None])
                else:
                    fr[2] = False
            elif stage == 1:
                i = order[k]
                if fr[2]:
                    self.sel &= ~(1 << i)
                    self.val -= P.v[i]
                    self.ll[P.g[i]] -= P.wl[i]; self.lg -= P.wg[i]
                fr[1] = 2
                st.append([k + 1, 0, False, None])
            else:
                st.pop()
        while cps and st and self.work.units >= cps[0]:   # the state a continued run would record
            self.ub_at[cps.pop(0)] = self.global_ub()
        if not st:
            self.finished = True
        return self.inc

    def at_budget(self, b):
        """(value, sel, finished, charged_work) of an uninterrupted run with budget b,
        reconstructed from the log of a run with budget >= b. The log records the work at
        the START of the iteration that found each incumbent; a run with budget b executes
        exactly the iterations starting with work < b. Charge: total work if it finished
        within b, else b (a budgeted call is charged its budget)."""
        assert b <= self.work.units or self.finished
        best = self.log[0]
        for rec in self.log:
            if rec[0] < b:
                best = rec
        fin = self.finished and self.last_start < b
        return best[1], best[2], fin, (self.work.units if fin else b)

    def ub_at_budget(self, b):
        """Certified global UB of an uninterrupted run with budget b (b must be a TRACE_GRID
        point, or the run finished within b)."""
        if self.finished and self.last_start < b:
            return self.at_budget(b)[0]
        return self.ub_at[b]


def root_bound(P: Plan, work: Work):
    fm = 0
    for i in range(P.n):
        if P.wg[i] <= P.capg and (P.nbl[P.g[i]] or P.wl[i] <= P.capl[P.g[i]]):
            fm |= 1 << i
    work.units += P.n
    return P.frac_bound(fm, list(P.capl), P.capg, work)


# --- beam search --------------------------------------------------------------

def beam(P: Plan, width: int, work: Work):
    """Beam search over P.order, keeping the `width` partial solutions with the best
    value + fractional bound. Output: best complete feasible solution. Work: bound scans."""
    beams = [(0, 0, tuple([0] * P.G), 0)]   # (val, sel, ll, lg)
    for k, i in enumerate(P.order):
        cands = []
        for val, sel, ll, lg in beams:
            cands.append((val, sel, ll, lg))
            p = P.parent[i]
            work.units += 1
            if P.conf[i] & sel or (p >= 0 and not (sel >> p) & 1):
                continue
            g = P.g[i]
            if lg + P.wg[i] > P.capg or (not P.nbl[g] and ll[g] + P.wl[i] > P.capl[g]):
                continue
            ll2 = list(ll); ll2[g] += P.wl[i]
            cands.append((val + P.v[i], sel | (1 << i), tuple(ll2), lg + P.wg[i]))
        if len(cands) > width:
            scored = []
            for c in cands:
                val, sel, ll, lg = c
                fm = 0
                for t in range(k + 1, P.n):
                    j = P.order[t]
                    pj = P.parent[j]
                    if P.conf[j] & sel or (pj >= 0 and P.pos[pj] <= k and not (sel >> pj) & 1):
                        continue
                    fm |= 1 << j
                work.units += P.n - k - 1
                ub = val + P.frac_bound(fm, [P.capl[g] - ll[g] for g in range(P.G)], P.capg - lg, work)
                scored.append((-ub, -val, sel, c))
            scored.sort(key=lambda s: (s[0], s[1], s[2]))
            beams = [s[3] for s in scored[:width]]
        else:
            beams = cands
    return max(beams, key=lambda b: (b[0], -b[1]))[1]


# --- propagation + decomposition ----------------------------------------------

def _ub_given(P, sel_in, excl, work):
    """Upper bound with items sel_in forced in (must be feasible) and excl excluded."""
    ll, lg = P.loads(sel_in)
    reml = [P.capl[g] - ll[g] for g in range(P.G)]
    remg = P.capg - lg
    if remg < 0 or any(reml[g] < 0 and not P.nbl[g] for g in range(P.G)):
        return -1
    fm = 0
    for i in range(P.n):
        if (sel_in >> i) & 1 or (excl >> i) & 1:
            continue
        if P.conf[i] & sel_in:
            continue
        p = P.parent[i]
        if p >= 0 and (excl >> p) & 1:
            continue
        if P.wg[i] > remg or (not P.nbl[P.g[i]] and P.wl[i] > reml[P.g[i]]):
            continue
        fm |= 1 << i
    work.units += P.n
    return P.value(sel_in) + P.frac_bound(fm, reml, remg, work)


def propagate(P: Plan, inc_val: int, work: Work, max_rounds: int = 20):
    """Fix variables to a fixed point. Contract: returns (fixed_in, fixed_out) masks such
    that every solution with value > inc_val satisfies them (so the optimum is
    max(incumbent, best completion)). Rules: requires-closure, conflicts with fixed-in,
    single-item capacity infeasibility, and bound-based fixing (x_i = 0 if the bound with
    i in is <= inc_val; x_i = 1 if the bound with i out is <= inc_val). Work: bound scans."""
    fin, fout = 0, 0
    full = (1 << P.n) - 1
    for _ in range(max_rounds):
        changed = False
        for i in range(P.n):
            if ((fin | fout) >> i) & 1:
                continue
            cl = []
            x = i
            while x >= 0 and not (fin >> x) & 1:
                cl.append(x); x = P.parent[x]
            clm = 0
            for x in cl:
                clm |= 1 << x
            if clm & fout or any(P.conf[x] & (fin | clm) for x in cl):
                fout |= 1 << i; changed = True; continue
            if _ub_given(P, fin | clm, fout, work) < inc_val + 1 - 1e-9:
                fout |= 1 << i; changed = True; continue
            if _ub_given(P, fin, fout | (1 << i), work) < inc_val + 1 - 1e-9:
                # every improving solution contains i (and its closure)
                ll, lg = P.loads(fin | clm)
                if lg <= P.capg and all(ll[g] <= P.capl[g] or P.nbl[g] for g in range(P.G)):
                    fin |= clm; changed = True
                    for x in cl:
                        fout |= P.conf[x] & ~fin
        # descendants of fixed-out are out
        for i in range(P.n):
            p = P.parent[i]
            if p >= 0 and (fout >> p) & 1 and not (fout >> i) & 1 and not (fin >> i) & 1:
                fout |= 1 << i; changed = True
        if not changed:
            break
    fout &= full & ~fin
    return fin, fout


def components(P: Plan, free: list, fin: int, reml, remg):
    """Connected components of free items under conflicts, requires, and binding shared
    capacities (a binding group joins its members; a binding global joins everything)."""
    parent = {i: i for i in free}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    fs = set(free)
    for i in free:
        for j in P.adj[i]:
            if j in fs:
                union(i, j)
        p = P.parent[i]
        if p in fs:
            union(i, p)
    by_g = {}
    for i in free:
        by_g.setdefault(P.g[i], []).append(i)
    glob_binding = sum(P.wg[i] for i in free) > remg
    for g, its in by_g.items():
        if sum(P.wl[i] for i in its) > reml[g]:
            for a in its[1:]:
                union(its[0], a)
    if glob_binding and free:
        for a in free[1:]:
            union(free[0], a)
    comps = {}
    for i in free:
        comps.setdefault(find(i), []).append(i)
    return sorted(comps.values(), key=lambda c: (len(c), c)), glob_binding


class PD:
    """Propagation to fixed point + decomposition into independent components, each
    solved by greedy+repair (seed) and then B&B to a shared budget (smallest component
    first). Exact when every component finishes. Logs let `at_budget(b)` reproduce the
    uninterrupted run at any budget b <= the budget it was run with."""

    def __init__(self, P: Plan, inc: int, inc_work: int):
        self.P = P
        self.inc, self.inc_val = inc, P.value(inc)
        w = Work()
        self.fin, self.fout = propagate(P, self.inc_val, w)
        self.prop_work = w.units
        free = [i for i in range(P.n) if not ((self.fin | self.fout) >> i) & 1]
        ll, lg = P.loads(self.fin)
        reml = [P.capl[g] - ll[g] for g in range(P.G)]
        remg = P.capg - lg
        self.n_free = len(free)
        self.comps, self.glob_binding = components(P, free, self.fin, reml, remg)
        self.subs = []
        seed_work = 0
        for comp in self.comps:
            S = P.restrict(comp, reml, remg, self.fin)
            w = Work()
            s0 = greedy(S, w)
            rp = Repair(S, s0)
            rp.run(repair_budget(S.n))
            seed = rp.sel
            # warm start from the incumbent restricted to the component when it is feasible there
            r = 0
            for t, i in enumerate(comp):
                if (inc >> i) & 1:
                    r |= 1 << t
            w.units += len(comp)
            if S.feasible(r) and S.value(r) > S.value(seed):
                seed = r
            rb = root_bound(S, w)
            seed_work += w.units + rp.work.units
            self.subs.append({"plan": S, "seed": seed, "bnb": BnB(S, seed), "comp": comp, "root_ub": math.floor(rb + 1e-9) + 0})
        self.seed_work = seed_work
        self.base_work = inc_work + self.prop_work + seed_work
        self.ran_to = 0

    def run(self, budget):
        """Run component B&Bs sequentially (smallest first) with a shared search budget."""
        left = budget
        for sub in self.subs:
            if left <= 0:
                break
            bb = sub["bnb"]
            bb.run(left)
            left -= bb.work.units if bb.finished else left
        self.ran_to = budget

    def at_budget(self, b):
        """(sel, value, finished, search_work) of an uninterrupted run with search budget b <= ran_to."""
        assert b <= self.ran_to
        left, sel, fin_all, used = b, self.fin, True, 0
        ub = self.P.value(self.fin)
        for sub in self.subs:
            bb = sub["bnb"]
            if left > 0:
                cval, csel, fin, ch = bb.at_budget(left)
                if fin:
                    cub = cval
                elif left >= bb.work.units:   # the query covers the whole run: bound of its end state
                    cub = min(sub["root_ub"], bb.global_ub())
                else:   # latest recorded certified UB at or before this allocation (valid, maybe loose)
                    pts = [g for g in TRACE_GRID if g <= left and g in bb.ub_at]
                    cub = min(sub["root_ub"], bb.ub_at[pts[-1]]) if pts else sub["root_ub"]
                used += ch
                left -= ch
            else:
                csel, fin, cub = sub["seed"], False, sub["root_ub"]
            fin_all = fin_all and fin
            ub += max(cub, sub["plan"].value(csel))
            sel |= _lift(sub["plan"], csel)
        val = self.P.value(sel)
        if val <= self.inc_val:
            sel, val = self.inc, self.inc_val
        ub = max(ub, self.inc_val)
        if fin_all:
            ub = val
        return sel, val, fin_all, used, ub


def _lift(S: Plan, csel):
    out = 0
    for i in S.items(csel):
        out |= 1 << S.ids[i]
    return out


# ---------------------------------------------------------------------------
# Exact optimum (offline, evaluation only)

EXACT_CAP = 80_000_000   # work units (~5 core-s at 16.4M units/core-s); sizes set so p99 certifies (pilot)


def exact_optimum(inst: Instance, cap_units: int = EXACT_CAP):
    """Oracle for the TRUE instance (offline, evaluation only): greedy+repair incumbent,
    propagation, decomposition, then B&B with the (min of the method bound and the)
    surrogate bound to completion on every component, up to cap_units. Returns
    (sel, value, work, certified, ub). Uncertified instances are KEPT: they are scored
    against the certified upper bound ub (conservative quality), and counted."""
    P = true_plan(inst, surrogate=True)
    w = Work()
    s0 = greedy(P, w)
    rp = Repair(P, s0)
    rp.run(repair_budget(P.n))
    pd = PD(P, rp.sel, w.units + rp.work.units)
    pd.run(cap_units)
    sel, val, fin, used, ub = pd.at_budget(cap_units)
    return sel, val, pd.base_work + used, fin, (val if fin else ub)


# ---------------------------------------------------------------------------
# Reuse

def reuse(P: Plan, cache: int, work: Work):
    """Verify the cached solution on the plan (work: popcount + conflict/requires checks);
    if infeasible, drop the lowest-ratio item implicated in a violation (with its selected
    descendants) until feasible; then local repair to a local optimum. Returns
    (sel, telemetry). Guarantee: the output is feasible for the plan."""
    items = P.items(cache)
    work.units += len(items) + sum(len(P.adj[i]) for i in items)
    verify_work = work.units
    ll, lg = P.loads(cache)
    viol = sum(1 for g in range(P.G) if ll[g] > P.capl[g]) + (lg > P.capg)
    viol += sum(1 for i in items if P.conf[i] & cache) // 2
    viol += sum(1 for i in items if P.parent[i] >= 0 and not (cache >> P.parent[i]) & 1)
    tel = {"cache_feasible": P.feasible(cache), "cache_violations": viol, "cache_value": P.value(cache), "verify_work": verify_work}
    sel = cache
    while not P.feasible(sel):
        ll, lg = P.loads(sel)
        bad = []
        for i in P.items(sel):
            g = P.g[i]
            if (ll[g] > P.capl[g]) or lg > P.capg or P.conf[i] & sel or (P.parent[i] >= 0 and not (sel >> P.parent[i]) & 1):
                bad.append(i)
        work.units += P.n
        j = min(bad, key=lambda i: (P.ratio[i], i))
        st, rm = [j], 0
        while st:
            x = st.pop()
            if (sel >> x) & 1 and not (rm >> x) & 1:
                rm |= 1 << x; st.extend(P.children[x])
        sel &= ~rm
    rp = Repair(P, sel)
    rp.run(repair_budget(P.n))
    work.units += rp.work.units
    tel["reuse_value"] = P.value(rp.sel)
    return rp.sel, tel


# ---------------------------------------------------------------------------
# Arm evaluation

BUDGETS = (1_000, 10_000, 100_000, 1_000_000)   # menu B&B search budgets (work units; TRACE_GRID points)
BEAM_WIDTHS = (4, 16)
MODES = ("cons", "insp", "opt")


def arm_names(budgets=BUDGETS, modes=MODES):
    """The method/budget menu. Propagation + decomposition are always-on preprocessing of
    every B&B arm (PD = greedy+repair incumbent -> propagate -> decompose -> B&B per
    component; RUPD = the same warm-started from the verified/repaired cache)."""
    out = []
    for m in modes:
        out += [f"{m}:G", f"{m}:GR", f"{m}:RV", f"{m}:RU"] + [f"{m}:BM{w}" for w in BEAM_WIDTHS]
        out += [f"{m}:PD0"] + [f"{m}:PD{b}" for b in budgets] + [f"{m}:RUPD{b}" for b in budgets]
    return out


def posterior_feasible_prob(pv: PublicView, sel: int):
    """P(sel is truly feasible | public information), exact under the generator's explicit
    posterior (hidden capacity uniform on the integers of [lo, hi]; each possible conflict
    real with probability 1/2, independently). Known constraints must hold (else 0)."""
    ll = [0] * pv.G
    lg = 0
    for i in range(pv.n):
        if (sel >> i) & 1:
            ll[pv.grp[i]] += pv.wl[i]; lg += pv.wg[i]
            if pv.parent[i] >= 0 and not (sel >> pv.parent[i]) & 1:
                return 0.0
    if lg > pv.capg:
        return 0.0
    for i, j in pv.conflicts:
        if (sel >> i) & 1 and (sel >> j) & 1:
            return 0.0
    p = 1.0
    for g in range(pv.G):
        lo, hi = pv.cap_lo[g], pv.cap_hi[g]
        if ll[g] > hi:
            return 0.0
        if ll[g] > lo:
            p *= (hi - ll[g] + 1) / (hi - lo + 1)
    for i, j in pv.possible:
        if (sel >> i) & 1 and (sel >> j) & 1:
            p *= 0.5
    return p


def _eval_mode(inst, pv, mode, budgets):
    insp = Inspector(inst)
    P = plan_from_public(pv, mode, insp)
    obs = insp.count
    arms, tel, traces = {}, {}, {}
    w = Work(); rb = root_bound(P, w)
    tel["root_bound"] = rb; tel["root_work"] = w.units
    wG = Work(); sG = greedy(P, wG)
    arms["G"] = (sG, {"G": wG.units})
    rp = Repair(P, sG); rp.run(repair_budget(P.n)); sGR = rp.sel
    stGR = {"G": wG.units, "R": rp.work.units}
    arms["GR"] = (sGR, stGR)
    tel.update(g_value=P.value(sG), gr_value=P.value(sGR), gr_moves=rp.moves, gr_work=wG.units + rp.work.units)
    for wd in BEAM_WIDTHS:
        wb = Work(); sb = beam(P, wd, wb)
        arms[f"BM{wd}"] = (sb, {f"BM{wd}": wb.units})
    wR = Work(); sRU, rtel = reuse(P, pv.cache, wR)
    stRU = {"RUv": rtel["verify_work"], "RUr": wR.units - rtel["verify_work"]}
    arms["RU"] = (sRU, stRU)
    arms["RV"] = (pv.cache, {"RUv": rtel["verify_work"]}) if rtel["cache_feasible"] else (sRU, stRU)
    tel.update(rtel)
    tel["ru_work"] = wR.units
    for fam, inc, base in (("PD", sGR, dict(stGR)), ("RUPD", sRU if P.value(sRU) > P.value(sGR) else sGR, dict(stGR, **stRU))):
        pd = PD(P, inc, 0)
        pd.run(TRACE_MAX)
        base[f"{fam}prop"] = pd.prop_work + pd.seed_work
        tr = []
        for b in (0,) + TRACE_GRID:
            sel, _, fin, used, ub = pd.at_budget(b)
            tr.append((b, sel, fin, used, ub))
            if b == 0 and fam == "PD":
                arms["PD0"] = (sel, dict(base))
            if b in budgets:
                arms[f"{fam}{b}"] = (sel, dict(base, **{f"{fam}search": used}))
        traces[fam] = {"base": base, "rows": tr}
        pre = fam.lower()
        tel.update({f"{pre}_nfree": pd.n_free, f"{pre}_ncomp": len(pd.comps),
                    f"{pre}_maxcomp": max([len(c) for c in pd.comps], default=0), f"{pre}_glob_binding": pd.glob_binding,
                    f"{pre}_nfixed_in": bin(pd.fin).count("1"), f"{pre}_prop_work": pd.prop_work + pd.seed_work,
                    f"{pre}0_value": tr[0][1] and P.value(tr[0][1]), f"{pre}0_ub": tr[0][4]})
        for b in budgets:
            row = [r for r in tr if r[0] == b][0]
            tel[f"{pre}{b}_finished"] = row[2]
            tel[f"{pre}{b}_value"] = P.value(row[1])
            tel[f"{pre}{b}_ub"] = row[4]
    if mode == "opt":
        tel["gr_post_feas"] = posterior_feasible_prob(pv, sGR)
        tel["cache_post_feas"] = posterior_feasible_prob(pv, pv.cache)
    return arms, tel, traces, obs


def evaluate_instance(inst: Instance, budgets=BUDGETS, opt=None):
    """Run every arm on one instance (each anytime method ONCE to TRACE_MAX; budgets are
    read off its trace). Returns (record, oracle solution). Record: opt (certified optimum,
    or the certified upper bound when uncertified: conservative), certified, exact_work,
    arms {name: {v, ok, st (mode-qualified stage -> work), obs}}, traces, telemetry per
    mode, public features (+ their work units)."""
    pv = public_view(inst)
    osel, oval, ework, cert, oub = exact_optimum(inst) if opt is None else opt
    feats, fwork = public_features(pv, return_work=True)
    rec = {"opt": oub, "opt_value": oval, "certified": cert, "exact_work": ework, "arms": {}, "tel": {}, "trace": {},
           "pub": feats, "feature_work": fwork,
           "prices": {"c": pv.prices[0], "o": pv.prices[1], "L": pv.prices[2]},
           "episode": inst.meta["episode_seed"], "t": inst.meta["t"], "n_hidden": inst.n_hidden()}
    done = {}
    for mode in MODES:
        alias = inst.n_hidden() == 0 and mode != "cons"
        if alias:   # without hidden parameters every mode is the identical cons computation
            arms, tel, traces, obs = done["cons"]
            sm = "cons"
        else:
            arms, tel, traces, obs = _eval_mode(inst, pv, mode, budgets)
            done[mode] = (arms, tel, traces, obs)
            sm = mode
        rec["tel"][mode] = tel
        for name, (sel, st) in arms.items():
            ok, val = check_solution(inst, sel)
            if mode != "opt":
                assert ok, (mode, name)   # cons plans are conservative, insp plans are exact
            rec["arms"][f"{mode}:{name}"] = {"v": val, "ok": ok, "st": {f"{sm}:{k}": x for k, x in st.items()}, "obs": obs}
        rec["trace"][mode] = {}
        for fam, t in traces.items():
            rows = []
            for b, sel, fin, used, ub in t["rows"]:
                ok, val = check_solution(inst, sel)
                rows.append([b, val, ok, fin, used, ub])
            rec["trace"][mode][fam] = {"base": {f"{sm}:{k}": x for k, x in t["base"].items()}, "rows": rows,
                                       "search_stage": f"{sm}:{fam}search"}
    return rec, osel


def trace_arm(rec, mode, fam, b):
    """A 'virtual arm' read off the trace of (mode, fam) at grid budget b."""
    t = rec["trace"][mode][fam]
    row = [r for r in t["rows"] if r[0] == b][0]
    st = dict(t["base"])
    if b:
        st[t["search_stage"]] = row[4]
    return {"v": row[1], "ok": row[2], "st": st, "obs": rec["arms"][f"{mode}:GR"]["obs"], "fin": row[3], "ub": row[5]}


def utility(rec, arm_or_arms, extra_work=0):
    """U = quality - c*work - o*observations - L*[infeasible or empty commit].
    Arms are names or arm dicts. For a sequence: stage works are charged once (max over
    repeats of the same stage, e.g. a resumed B&B), observations once (max). Commit rule:
    if the sequence inspected (an insp arm), outputs are checked against the revealed truth
    and the best truly feasible one is committed; otherwise the highest-valued output is
    committed (cons outputs are always feasible; an opt output may be infeasible -> fail).
    extra_work: selector charges (features, telemetry, inference) in work units."""
    arms = [arm_or_arms] if isinstance(arm_or_arms, (str, dict)) else list(arm_or_arms)
    st, obs, outs, inspected = {}, 0, [], False
    for a in arms:
        r = rec["arms"][a] if isinstance(a, str) else a
        for k, x in r["st"].items():
            st[k] = max(st.get(k, 0), x)
        obs = max(obs, r["obs"])
        outs.append((r["v"], r["ok"]))
        inspected |= r["obs"] > 0
    if inspected:
        feas = [v for v, ok in outs if ok]
        best, fail = (max(feas), False) if feas else (0, True)
    else:
        best, ok = max(outs, key=lambda t: (t[0], t[1]))
        fail = not ok
    opt = rec["opt"]
    fail = fail or (best == 0 and opt > 0)
    q = 0.0 if fail else ((best / opt) if opt > 0 else 1.0)
    p = rec["prices"]
    return q - p["c"] * (sum(st.values()) + extra_work) - p["o"] * obs - p["L"] * fail


def total_work(rec, arms):
    arms = [arms] if isinstance(arms, (str, dict)) else list(arms)
    st = {}
    for a in arms:
        r = rec["arms"][a] if isinstance(a, str) else a
        for k, x in r["st"].items():
            st[k] = max(st.get(k, 0), x)
    return sum(st.values())


# ---------------------------------------------------------------------------
# Public features (functions of the PublicView only)

def public_features(pv: PublicView, return_work=False):
    """Public features (functions of the PublicView only). With return_work, also the
    deterministic work units of computing them (charged to selectors that use them)."""
    n, G = pv.n, pv.G
    meta = dict(pv.meta)
    sw_l = [0] * G
    for i in range(n):
        sw_l[pv.grp[i]] += pv.wl[i]
    tight_l = [pv.cap_lo[g] / max(1, sw_l[g]) for g in range(G)]
    E = len(pv.conflicts)
    gsz = [0] * G
    for x in pv.grp:
        gsz[x] += 1
    pairs_in = sum(s * (s - 1) // 2 for s in gsz)
    cross = sum(1 for i, j in pv.conflicts if pv.grp[i] != pv.grp[j])
    # public decomposability: components of the conservative interaction graph
    cons = Plan(pv.v, pv.grp, pv.wl, pv.wg, list(pv.cap_lo), pv.capg, list(pv.conflicts) + list(pv.possible), pv.parent)
    comps, _ = components(cons, list(range(n)), 0, list(cons.capl), cons.capg)
    v = np.array(pv.v, float)
    w = (np.array(pv.wl, float) + np.array(pv.wg, float)) / 2
    corr = float(np.corrcoef(v, w)[0, 1]) if n > 2 and v.std() > 0 and w.std() > 0 else 0.0
    n_hidden = sum(pv.hidden_cap) + len(pv.possible)
    cache_items = [i for i in range(n) if pv.cache is not None and (pv.cache >> i) & 1]
    ll = [0] * G
    for i in cache_items:
        ll[pv.grp[i]] += pv.wl[i]
    cset = set(cache_items)
    # value-of-information proxy (public, O(n log n)): root fractional bounds of the
    # conservative plan and of the optimistic plan (hidden caps at hi, possible conflicts absent)
    wk = Work()
    rb_cons = root_bound(cons, wk)
    optp = Plan(pv.v, pv.grp, pv.wl, pv.wg, list(pv.cap_hi), pv.capg, list(pv.conflicts), pv.parent)
    rb_opt = root_bound(optp, wk)
    fwork = 3 * n + 2 * E + 2 * len(pv.possible) + wk.units + n * max(1, math.ceil(math.log2(max(n, 2))))
    cache_pf = posterior_feasible_prob(pv, pv.cache) if pv.cache is not None else 0.0
    feats = {
        "corr_class": dict(meta).get("corr_class", -1), "cache_post_feas": cache_pf,
        "hid_x_L": (n_hidden / (G + n)) * pv.prices[2],
        "rb_cons": rb_cons, "rb_opt": rb_opt, "voi_gap": (rb_opt - rb_cons) / max(rb_opt, 1e-9),
        "n": n, "G": G, "log_n": math.log(n),
        "tight_loc_mean": float(np.mean(tight_l)), "tight_loc_min": float(np.min(tight_l)),
        "tight_glob": pv.capg / max(1, sum(pv.wg)), "glob_binding": float(sum(pv.wg) > pv.capg),
        "conf_density": E / max(1, pairs_in), "cross_conf": cross, "n_conf": E,
        "req_frac": sum(1 for p in pv.parent if p >= 0) / n,
        "n_comp": len(comps), "max_comp": max(len(c) for c in comps), "comp_frac": max(len(c) for c in comps) / n,
        "vw_corr": corr,
        "hidden_caps": sum(pv.hidden_cap), "hidden_cap_frac": sum(pv.hidden_cap) / G, "n_possible": len(pv.possible),
        "n_hidden": n_hidden, "hidden_frac": n_hidden / (G + n),
        "width": meta.get("width", 0.0) if sum(pv.hidden_cap) else 0.0,
        "m": meta.get("m", 0.0), "n_value_changed": meta.get("n_value_changed", 0),
        "n_weight_changed": meta.get("n_weight_changed", 0), "n_cap_changed": meta.get("n_cap_changed", 0),
        "n_conf_changed": meta.get("n_conf_removed", 0) + meta.get("n_conf_added", 0),
        "cache_size": len(cache_items),
        "cache_lo_viol": sum(1 for g in range(G) if ll[g] > pv.cap_lo[g]),
        "cache_hi_viol": sum(1 for g in range(G) if ll[g] > pv.cap_hi[g]),
        "cache_conf_viol": sum(1 for i, j in pv.conflicts if i in cset and j in cset),
        "cache_poss_in": sum(1 for i, j in pv.possible if i in cset and j in cset),
        "t": meta.get("t", 0),
        "log_c": math.log10(pv.prices[0]), "log_o": math.log10(pv.prices[1]), "L": pv.prices[2],
        "log_obs_cost": math.log10(pv.prices[1] * max(n_hidden, 0.1)),
    }
    return (feats, fwork) if return_work else feats


def evaluate_episode(seed: int, knobs=None, budgets=BUDGETS, version=None):
    """Generate an episode, fill each instance's cache with the certified optimum of its
    predecessor, and evaluate instances t = 0..k-1. Returns a list of records."""
    insts, kd = generate_episode(seed, knobs, version)
    prev = exact_optimum(insts[0])
    recs = []
    for inst in insts[1:]:
        inst.cache = prev[0]
        rec, osel = evaluate_instance(inst, budgets)
        rec["knobs"] = kd
        rec["generator"] = version or GENERATOR_VERSION
        recs.append(rec)
        prev = (osel,)
    return recs
