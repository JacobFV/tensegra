"""Probeworld v3 (extended-06 Track B): factor registry, versioned factor extension, split table v3, supplied
factorized state, balanced counterfactual sets and on-manifold interventions.

Pure Python (no torch, no numpy).  Additive over ``campaign04_probeworld`` (``pw``, probeworld-v1) and
``campaign05_probeworld`` (``pw5``): neither module is modified, and every extended-04/05 split, label and trainer
path is unchanged (tested).  Design: research/campaigns/extended-06/design.md (Track B); screening and selection:
research/campaigns/extended-06/trackb-screen.md.

Factors
-------
Inherited (probeworld-v1 semantics, parameter ranges unchanged):
  U unreliable detection    q < 1        (a probe failure is reported "solved" w.p. 1 - q)
  S probe side effect       D_side > 0   (charged when the probe runs on theta != H)
  C correlated failure      corr > 0     (shared latent z shifts P(H) across the k queries)
  E requirement events      p_event > 0  (invalidate candidates, the reduction and per-epoch usage)
Versioned extension (probeworld-v3; see trackb-screen.md section 2 for why):
  D deadline pressure       deadline in {1, 2}: at most `deadline` non-terminal actions per query (a per-query
                            wall-clock budget: build counts; an event does NOT refund steps).  When the budget is
                            spent only terminal actions (commit / commit_infeasible / abstain) remain.  Public.
  T type-dependent exact cost  t_hard in [.5, 2.5]: exact computation cost scales with hidden hardness -- every exact
                            call (b1 or b2) on a hard instance (theta in {F, X}) charges an extra t_hard * c_b2
                            (ledger line "exact_hardness"; like the side effect, the charge is not an observation;
                            reduction does not remove it).  Probe, prop, inspect and use are unchanged.  It raises the value of cheap-first
                            probing, of prop -> b1, of the structure and of abstaining on likely-hard instances, i.e.
                            it pushes AGAINST U and S (which penalize probing) and flips which computation is optimal.
  (Local pilots only, before any metered screening: gen1 had G "b1 times out on unreduced M"; a T surcharge removed
  by propagation was absorbed by prop.  See trackb-screen.md section 2.)
With deadline = 0 and t_hard = 0 every function below is the probeworld-v1 function itself (``env(cfg)`` returns
``pw`` for a v1 ``pw.Config``; a ``Config6`` with D and G off delegates to ``pw``), so v1 labels are reproduced
exactly (tested).

Information state: identical to v1.  The deadline step counter of the current query is stored in the usage bits
>= 6 (``U_STEP`` per non-terminal action); it is a function of the visible history (count of non-terminal actions
since the last terminal action), so labels remain functions of the visible history (tested).
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import asdict, dataclass, fields, replace

from tensegra import campaign04_probeworld as pw
from tensegra import campaign05_probeworld as pw5

VERSION = "probeworld-v3"
GENERATOR_VERSION = "probeworld-v3-gen3"  # design v2 revision 1 ranges; gen2 = v1 ranges + D/T (screened, superseded)
U_STEP = 64  # deadline step counter: usage >> 6 (bits 0-5 are the v1 usage bits)
STEP_SHIFT = 6
assert U_STEP == 1 << STEP_SHIFT and pw.U_INSPECT < U_STEP

FACTORS = ("U", "S", "C", "E", "D", "T")
FACTOR_NAMES = ("unreliable", "side_effect", "correlated", "events", "deadline", "hard_exact_cost")
FACTOR_FIELD = {"U": ("q", 1.0), "S": ("D_side", 0.0), "C": ("corr", 0.0), "E": ("p_event", 0.0),
                "D": ("deadline", 0), "T": ("t_hard", 0.0)}
V1_FACTORS = ("U", "S", "C", "E")
EXT_FACTORS = ("D", "T")
DEADLINE_VALUES = (1, 2)
V3_K = pw.TRAIN_K  # (1, 2, 8): the v1 training horizons; the correlated factor only at k >= 2 (design v2 revision 1)
CORR_RANGE = (0.25, 0.45)  # design v2 revision 1 (v1: [.1, .25] at any k)
P_EVENT_RANGE = (0.3, 0.6)  # design v2 revision 1 (v1: [.1, .3])
T_RANGE = (0.5, 2.5)


@dataclass(frozen=True)
class Config6(pw.Config):
    """probeworld-v3 public configuration: v1 fields + deadline (D) and type-dependent exact cost (T)."""
    deadline: int = 0
    t_hard: float = 0.0

    @property
    def flags(self) -> tuple:  # (U, S, C, E, D, G)
        return (self.q < 1.0, self.D_side > 0.0, self.corr > 0.0, self.p_event > 0.0, self.deadline > 0,
                self.t_hard > 0.0)

    @property
    def extended(self) -> bool:
        return self.deadline > 0 or self.t_hard > 0.0

    def public_vector(self) -> list:
        """v1 public vector (the 4 v1 flags at their v1 positions) + [deadline/2, t_hard, D flag, T flag]."""
        R = self.R
        return ([self.c_probe / R, self.c_b1 / R, self.c_b2 / R, self.c_inspect / R, self.c_prop / R,
                 self.L / R, min(self.C_build / R, 10.0) / 10.0, self.C_execute / R, self.C_return / R,
                 self.C_verify / R, self.D_side / R, self.q, self.corr, self.p_event, self.eta,
                 self.p_conflict, self.k / 8.0, math.log2(self.k) / 3.0, math.log(max(self.rho(), 1e-3)) / 3.0]
                + [float(f) for f in self.flags[:4]] + list(self.prior)
                + [self.deadline / 2.0, self.t_hard, float(self.deadline > 0), float(self.t_hard > 0.0)])


PUBLIC_DIM6 = len(Config6().public_vector())
PUBLIC_EXTRA6 = PUBLIC_DIM6 - pw.PUBLIC_DIM
assert PUBLIC_EXTRA6 == 4


def from_v1(cfg: pw.Config, **kw) -> Config6:
    return Config6(**{**{f.name: getattr(cfg, f.name) for f in fields(pw.Config)}, **kw})


def flags_of(cfg) -> tuple:
    """6 flags for any configuration (a v1 Config has D and G off)."""
    f = tuple(cfg.flags)
    return f + (False,) * (6 - len(f))


def combo_name(flags) -> str:
    return "+".join(n for n, f in zip(FACTOR_NAMES, flags) if f) or "none"


def fam_key(flags) -> str:
    return "".join(x for x, f in zip(FACTORS, flags) if f) or "0"


def flags_from_key(key: str) -> tuple:
    return tuple(x in key for x in FACTORS)


def ablate(cfg: Config6, factor: str) -> Config6:
    field, off = FACTOR_FIELD[factor]
    return replace(cfg, **{field: off})


def restrict(cfg: Config6, keep) -> Config6:
    """The same configuration with every factor not in `keep` switched off."""
    out = cfg
    for x, on in zip(FACTORS, flags_of(cfg)):
        if on and x not in keep:
            out = ablate(out, x)
    return out


# ---------------------------------------------------------------------------------------------------------
# dynamics (delegate to probeworld-v1 unless D or G is active)

def outcome_dist(cfg, a: int, theta: int, reduced: bool) -> list:
    return pw.outcome_dist(cfg, a, theta, reduced)  # D and T change availability / charges, not the outcome model


def hard_surcharge(cfg, a: int, theta: int, reduced: bool) -> float:
    """T: every exact call (b1 or b2) on a hard instance (theta in {F, X}) incurs a hardness surcharge t_hard * c_b2
    (whether or not the instance was reduced: hardness is intrinsic).  Probe, prop, inspect and use are unchanged."""
    t = getattr(cfg, "t_hard", 0.0)
    if t > 0.0 and a in (pw.A_B1, pw.A_B2) and theta in (pw.TF, pw.TX):
        return t * cfg.c_b2
    return 0.0


def update_belief(cfg, b: tuple, a: int, o: int, reduced: bool) -> tuple:
    return pw._bkey([b[t] * sum(p for p, oo in outcome_dist(cfg, a, t, reduced) if oo == o) if b[t] else 0.0
                     for t in range(4)])


def steps_used(state) -> int:
    return state[1][1] >> STEP_SHIFT


def available(cfg, state: tuple) -> tuple:
    acts = pw.available(cfg, state)
    d = getattr(cfg, "deadline", 0)
    if d > 0 and steps_used(state) >= d:
        acts = tuple(a for a in acts if a in pw.TERMINAL)
    return acts


def advance(cfg, state: tuple, a: int, o: int, event: bool, reveal) -> tuple:
    if not getattr(cfg, "extended", False):
        return pw.advance(cfg, state, a, o, event, reveal)
    (i, built, nH, nN), (b, usage, reduced, cand, cand_valid, ev) = state
    if a in pw.TERMINAL:
        return pw.advance(cfg, state, a, o, event, reveal)  # next query: usage (incl. the step counter) resets
    if a != pw.A_BUILD:
        b = update_belief(cfg, b, a, o, reduced)
        usage |= pw.ACTION_BIT[a]
    else:
        built = True
    if cfg.deadline > 0:
        usage += U_STEP
    if a in (pw.A_B1, pw.A_B2, pw.A_USE) and o == pw.O_SOLVED:
        cand, cand_valid = pw.C_EXACT, True
    elif a == pw.A_PROBE and o == pw.O_SOLVED and not (cand == pw.C_EXACT and cand_valid):
        cand, cand_valid = pw.C_PROBE, True
    if a == pw.A_PROP and o == pw.O_REDUCED:
        reduced = True
    if event:  # declared invalidation subset (the deadline step counter is NOT refunded)
        cand_valid = False
        reduced = False
        usage &= ~pw.EPOCH_BITS
        ev = True
    return ((i, built, nH, nN), (b, usage, reduced, cand, cand_valid, ev))


def public_state_from_history(cfg, history) -> tuple:
    s = pw.initial_state(cfg)
    for a, o, event, reveal in history:
        s = advance(cfg, s, a, o, event, reveal)
    return s


class ExactSolver(pw.ExactSolver):
    """probeworld-v3 exact DP (the v1 solver with the v3 outcome model, availability and transition)."""

    def transitions(self, state, a):
        cfg = self.cfg
        if not cfg.extended:
            return pw.ExactSolver.transitions(self, state, a)
        b = state[1][0]
        local = state[1]
        reduced, cand, cand_valid, ev = local[2], local[3], local[4], local[5]
        out = []
        if a in pw.TERMINAL:
            for t in range(4):
                if b[t] <= 0:
                    continue
                if a == pw.A_ABSTAIN:
                    o, r = pw.O_ABSTAINED, 0.0
                else:
                    ok = pw.commit_correct(a, t, cand, cand_valid)
                    o, r = (pw.O_CORRECT, cfg.R) if ok else (pw.O_WRONG, -cfg.L)
                out.append((b[t], o, False, t, advance(cfg, state, a, o, False, t), r))
            return out
        cost = pw.action_cost(cfg, a)
        agg: dict = {}
        for t in range(4):
            if b[t] <= 0:
                continue
            for p, o in outcome_dist(cfg, a, t, reduced):
                if p > 0:
                    agg[o] = agg.get(o, 0.0) + b[t] * p
        pe = cfg.p_event if (cfg.p_event > 0 and not ev) else 0.0
        side = cfg.D_side * (1.0 - b[pw.TH]) if (a == pw.A_PROBE and cfg.D_side > 0) else 0.0
        side += sum(b[t] * hard_surcharge(cfg, a, t, reduced) for t in (pw.TF, pw.TX))
        for o, po in agg.items():
            for event, pev in ((False, 1.0 - pe), (True, pe)):
                if pev <= 0:
                    continue
                out.append((po * pev, o, event, None, advance(cfg, state, a, o, event, None), -cost - side))
        return out

    def q_values(self, state) -> dict:
        q = self._Q.get(state)
        if q is None:
            q = {}
            for a in available(self.cfg, state):
                q[a] = sum(p * (r + self.value(ns)) for p, _, _, _, ns, r in self.transitions(state, a))
            self._Q[state] = q
        return q


class Episode(pw.Episode):
    """probeworld-v3 episode (hidden theta/z only simulate outcomes and charges)."""

    def available(self) -> tuple:
        return available(self.cfg, self.state)

    def step(self, a: int):
        if not self.cfg.extended:
            return pw.Episode.step(self, a)
        if self.done or a not in self.available():
            raise ValueError(f"action {pw.ACTIONS[a]} not available")
        cfg = self.cfg
        theta = self.thetas[self.query]
        t = len(self.history)
        local = self.state[1]
        reveal = None
        event = False
        for item, amt in pw.cost_lines(cfg, a, theta):
            self.ledger.append((t, item, amt))
        hs = hard_surcharge(cfg, a, theta, local[2])
        if hs:
            self.ledger.append((t, "exact_hardness", hs))
        if a in pw.TERMINAL:
            reveal = theta
            if a == pw.A_ABSTAIN:
                o = pw.O_ABSTAINED
            elif pw.commit_correct(a, theta, local[3], local[4]):
                o = pw.O_CORRECT
                self.successes += 1
            else:
                o = pw.O_WRONG
                self.wrong += 1
                self.ledger.append((t, "error_loss", cfg.L))
        else:
            dist = outcome_dist(cfg, a, theta, local[2])
            u, acc, o = self.rng.random(), 0.0, dist[-1][1]
            for p, oo in dist:
                acc += p
                if u < acc:
                    o = oo
                    break
            if cfg.p_event > 0 and not local[5]:
                event = self.rng.random() < cfg.p_event
        rec = (a, o, event, reveal)
        self.history.append(rec)
        self.state = advance(cfg, self.state, a, o, event, reveal)
        self.done = self.state[0][0] >= cfg.k
        return rec


def no_structure_cost(cfg) -> float:
    en = list(cfg.enable)
    en[pw.A_BUILD] = en[pw.A_USE] = False
    c1 = replace(cfg, k=1, corr=0.0, enable=tuple(en))
    return ExactSolver(c1).pi_star_stats()["E_cost"]


def effective_rho(cfg, k=None) -> float:
    k = cfg.k if k is None else k
    return no_structure_cost(cfg) / (cfg.C_build / k + cfg.c_use)


def solver(cfg):
    return ExactSolver(cfg) if isinstance(cfg, Config6) else pw.ExactSolver(cfg)


class _Env:
    """Namespace used by the trainer: ``env(cfg).Episode / ExactSolver / available / advance / effective_rho``."""
    Episode = Episode
    ExactSolver = ExactSolver
    available = staticmethod(available)
    advance = staticmethod(advance)
    effective_rho = staticmethod(effective_rho)


ENV6 = _Env()


def env(cfg):
    """``pw`` itself for a probeworld-v1 Config (bit-identical paths); the v3 namespace for a Config6."""
    return ENV6 if isinstance(cfg, Config6) else pw


# ---------------------------------------------------------------------------------------------------------
# generator

def make_config(cell: tuple, k: int, flags: tuple, seed: int) -> Config6:
    """v3 configuration: v1 prices/prior/U,S parameters from pw.make_config (rng seed*7+1, unchanged); the v3 ranges
    of corr and p_event (design v2 revision 1) and the D / T parameters from a separate stream (seed*11+3), each drawn
    whether or not its factor is on.  Inherited v1 behaviour: pw.make_config draws q / D_side (and its own corr /
    p_event, overwritten here) only for active flags, so toggling a factor at GENERATION shifts eta and p_conflict
    (same marginal distribution).  On-manifold interventions therefore use ``ablate`` (one parameter set to its off
    value; a valid configuration of the same support), never re-generation."""
    flags = tuple(flags) + (False,) * (6 - len(flags))
    base = pw.make_config(cell, k, flags[:4], random.Random(seed * 7 + 1))
    r2 = random.Random(seed * 11 + 3)
    d = r2.choice(DEADLINE_VALUES)
    t = round(r2.uniform(*T_RANGE), 3)
    corr = round(r2.uniform(*CORR_RANGE), 3)
    pe = round(r2.uniform(*P_EVENT_RANGE), 3)
    if flags[2] and k < 2:
        raise ValueError("the correlated factor requires k >= 2 in probeworld-v3")
    return from_v1(base, corr=corr if flags[2] else 0.0, p_event=pe if flags[3] else 0.0,
                   deadline=d if flags[4] else 0, t_hard=t if flags[5] else 0.0)


def draw_params(seed: int, cells, ks, combos) -> tuple:
    """(cell, k, combo) of a stream configuration; a combination with the correlated factor draws k from k >= 2."""
    rng = random.Random(seed)
    cell, k, cmb = rng.choice(cells), rng.choice(ks), rng.choice(combos)
    if k < 2 and cmb is not None and "C" in cmb:
        k = rng.choice(tuple(x for x in ks if x >= 2))
    return cell, k, cmb


def stream_config(base: int, index: int, cells, ks, combos):
    seed = base + index
    cell, k, cmb = draw_params(seed, cells, ks, combos)
    return make_config(cell, k, flags_from_key(cmb), seed), (cell, k, cmb, seed)


def family_flags(key: str) -> tuple:
    return flags_from_key(key)


def all_families(factors=FACTORS, sizes=(2, 3)) -> list:
    return ["".join(c) for n in sizes for c in itertools.combinations(factors, n)]


# ---------------------------------------------------------------------------------------------------------
# seed sub-ranges (probeworld v3; all inside the registered extended-06 Track B range [6.1e9, 6.9e9))

TRACKB_RANGE = (6_100_000_000, 6_900_000_000)
SCREEN_BASE = 6_100_000_000  # screening streams: SCREEN_BASE + SCREEN_STRIDE * family_index + i (exact; no worlds)
SCREEN_STRIDE = 1_000_000
SCREEN_FAMILY_ORDER = tuple(all_families(FACTORS, (1, 2, 3)))  # fixed index per family (singles, pairs, triples)
SUBRANGES = {
    # name: (lo, hi, use)
    "screen": (6_100_000_000, 6_200_000_000, "structural screening streams (exact DP only)"),
    "train": (6_200_000_000, 6_300_000_000, "b6 training pools (all arms share base 6.2e9: common random numbers)"),
    "dev_test": (6_300_000_000, 6_400_000_000, "b6 dev / test_base / test_broad pools + their eval worlds"),
    "hold": (6_400_000_000, 6_500_000_000, "b6 held-out challenge families + eval worlds"),
    "hist": (6_500_000_000, 6_600_000_000, "historical U+E / S+C / U+C challenge sets (v3 generator, fresh) + worlds"),
    "cf": (6_600_000_000, 6_700_000_000, "balanced counterfactual quadruples (per family stream)"),
    "intervene": (6_700_000_000, 6_800_000_000, "on-manifold one-factor intervention sets"),
    "dev_smoke": (6_800_000_000, 6_900_000_000, "tooling tests / smokes (never protocol)"),
}
WORLD_SEED_OFFSET = pw.WORLD_SEED_OFFSET  # 5e7: eval worlds base + 5e7 + 1000 i + r (i < 4000, r < 1000)
MAX_POOL = 4000


def check_subranges():
    rs = sorted(SUBRANGES.values())
    assert rs[0][0] == TRACKB_RANGE[0] and rs[-1][1] == TRACKB_RANGE[1]
    for (lo, hi, _), (lo2, hi2, _) in zip(rs, rs[1:]):
        assert lo < hi <= lo2 < hi2
    assert SCREEN_BASE + SCREEN_STRIDE * len(SCREEN_FAMILY_ORDER) <= SUBRANGES["screen"][1]
    return True


def canon(family: str) -> str:
    """Canonical family key (factor letters in FACTORS order; '0' = no factor)."""
    return fam_key(flags_from_key(family))


def screen_seed(family: str, index: int) -> int:
    assert 0 <= index < SCREEN_STRIDE
    return SCREEN_BASE + SCREEN_STRIDE * SCREEN_FAMILY_ORDER.index(canon(family)) + index


def screen_config(family: str, index: int, cells=pw.TRAIN_CELLS, ks=None):
    ks = V3_K if ks is None else ks
    seed = screen_seed(family, index)
    cell, k, _ = draw_params(seed, cells, ks, (canon(family),))
    return make_config(cell, k, flags_from_key(family), seed)


# ---------------------------------------------------------------------------------------------------------
# decision relevance, interaction and global-shift primitives (exact; used by research/tools/campaign06_bscreen.py)

def s0_q(s) -> dict:
    return s.q_values(pw.initial_state(s.cfg))


def argmax_q(q: dict, bias=None) -> int:
    """Deterministic argmax (ties -> lowest action index, as pi_star) of q (+ an optional per-action bias)."""
    best, ba = None, None
    for a in sorted(q):
        v = q[a] + (bias.get(a, 0.0) if bias else 0.0)
        if best is None or v > best + 1e-12:
            best, ba = v, a
    return ba


def eps_set(q: dict, eps: float = pw.EPS) -> frozenset:
    v = max(q.values())
    return frozenset(a for a, x in q.items() if x >= v - eps)


_UNIFORM = (0.25, 0.25, 0.25, 0.25)


def foreign_advance(fcfg, fst, a, o, e, rev):
    """Advance the foreign (reference) configuration's public state along a visible record; an observation that is
    impossible under the foreign model keeps its previous belief (pw5.evaluate_foreign_policy's documented rule)."""
    try:
        return advance(fcfg, fst, a, o, e, rev)
    except ValueError:
        b = fst[1][0]
        nxt = advance(fcfg, (fst[0], (_UNIFORM,) + fst[1][1:]), a, o, e, rev)
        return (nxt[0], (b,) + nxt[1][1:])


def foreign_policy_value(true_s, foreign_s, bias=None, eps: float = pw.EPS, track=False):
    """Exact expected utility, in true_s's world, of the deterministic policy argmax_a [Q*_foreign(I~, a) + bias(a)]
    where I~ is the foreign configuration's own public state tracked from the shared visible history (the
    'training-support-optimal policy up to a global action bias').  If the foreign choice is unavailable in the true
    world (the deadline hides a non-terminal action), the foreign policy takes its best AVAILABLE action.
    track=True also returns P(the policy takes an eps-suboptimal action at least once) under the true world."""
    cfg, fcfg = true_s.cfg, foreign_s.cfg
    memo: dict = {}

    def choose(st, fst):
        av = available(cfg, st)
        fq = foreign_s.q_values(fst) if fst[0][0] < fcfg.k else {}
        cand = {a: fq[a] for a in av if a in fq}
        if not cand:  # foreign state exhausted or disjoint (cannot happen for one-factor ablations); fall back
            return true_s.pi_star(st)
        return argmax_q(cand, bias)

    def rec(st, fst):
        key = (st, fst)
        v = memo.get(key)
        if v is not None:
            return v
        if st[0][0] >= cfg.k:
            v = (0.0, 0.0)
        else:
            a = choose(st, fst)
            q = true_s.q_values(st)
            bad = q[a] < max(q.values()) - eps
            val = 0.0
            p_ok = 0.0
            for p, o, e, rev, ns, r in true_s.transitions(st, a):
                sub = rec(ns, foreign_advance(fcfg, fst, a, o, e, rev))
                val += p * (r + sub[0])
                p_ok += p * (1.0 - sub[1])
            v = (val, 1.0 if bad else 1.0 - p_ok)
        memo[key] = v
        return v

    val, p_bad = rec(pw.initial_state(cfg), pw.initial_state(fcfg))
    return (val, p_bad) if track else val


def any_decision_change(true_s, foreign_s, eps: float = pw.EPS) -> tuple:
    """Following pi*_true, is there a reachable decision at which the foreign policy (own belief tracking) would
    take an eps-suboptimal action?  Returns (exists, probability that a trajectory of pi*_true meets one)."""
    cfg, fcfg = true_s.cfg, foreign_s.cfg
    memo: dict = {}

    def rec(st, fst):
        key = (st, fst)
        v = memo.get(key)
        if v is not None:
            return v
        if st[0][0] >= cfg.k:
            v = (False, 0.0)
        else:
            q = true_s.q_values(st)
            av = [a for a in q]
            fq = foreign_s.q_values(fst) if fst[0][0] < fcfg.k else {}
            cand = {a: fq[a] for a in av if a in fq}
            fa = argmax_q(cand) if cand else None
            if fa is None or q[fa] < max(q.values()) - eps:
                v = (True, 1.0)
            else:
                a = true_s.pi_star(st)
                ex, pm = False, 0.0
                for p, o, e, rev, ns, r in true_s.transitions(st, a):
                    sub = rec(ns, foreign_advance(fcfg, fst, a, o, e, rev))
                    ex |= sub[0] and p > 0
                    pm += p * sub[1]
                v = (ex, pm)
        memo[key] = v
        return v

    return rec(pw.initial_state(cfg), pw.initial_state(fcfg))


def mobius_additive(qs: dict, full: tuple, order: int) -> dict:
    """Additive prediction of Q_full from the sub-combination Q-vectors qs[subset] (subset = sorted factor tuple),
    keeping inclusion-exclusion terms up to `order` factors (order 1: Q_0 + sum_f (Q_f - Q_0))."""
    acts = qs[full].keys()
    n = len(full)
    pred = {a: 0.0 for a in acts}
    # Moebius coefficients: prediction = sum_{T, |T| <= order} coef(T) Q_T, coef from truncated inclusion-exclusion
    for m in range(order + 1):
        for T in itertools.combinations(full, m):
            coef = sum((-1) ** (j - m) * math.comb(n - m, j - m) for j in range(m, order + 1))
            if coef:
                for a in acts:
                    pred[a] += coef * qs[T].get(a, 0.0)
    return pred


# ---------------------------------------------------------------------------------------------------------
# supplied factorized public state (SUPPLIED-FACTORIZED inputs; LEARNED-FACTORIZED auxiliary targets).
# Design v2 revision 5: the raw factor parameters (q, D_side, corr, p_event, ...) are already model inputs, so the
# factors supplied / learned are DERIVED decision quantities of the public information state: the posterior over the
# hidden type after the history, the probability that a probe (or b1) resolves the query, expected remaining cost
# per strategy (one-query closed forms, not Q*), and the amortized value of building over the remaining horizon.

FACTOR_FEATURES = (
    # posterior over the hidden type after the visible history (exact; includes revealed-type evidence under C)
    "belief_H", "belief_M", "belief_F", "belief_X",
    # probability that a computation resolves the current query
    "p_probe_resolves", "p_probe_false_solved", "p_H_given_solved", "p_b1_resolves",
    # expected factor costs under the current belief / history
    "exp_side_cost_rel", "exp_hard_cost_rel", "event_hazard_active",
    # expected remaining cost of the current query per strategy (price / R; incl. expected error loss)
    "cost_exact_b2_rel", "cost_probe_first_rel", "cost_b1_first_rel", "cost_use_rel", "best_strategy_cost_rel",
    # amortized build value over the remaining horizon (0 once built)
    "build_value_rel", "remaining_queries_rel", "built",
    # validity of computed results
    "candidate_valid", "candidate_exact", "candidate_trust",
    # remaining step budget (deadline factor; 1 when off)
    "steps_left_rel",
)
N_FACTOR_FEATURES = len(FACTOR_FEATURES)


def _clip(x, lo=-5.0, hi=5.0):
    return max(lo, min(hi, x))


def factor_features(cfg, state: tuple) -> list:
    """Derived public decision quantities at a public information state (config + state only; no hidden type).
    Strategy costs are one-query myopic closed forms (documented in trackb-screen.md section 6):
      b2 route        c_b2 + T surcharge, re-run once if an event invalidates the candidate (hazard)
      probe first     c_probe + side cost + P(no 'solved') * b2 route + P(false 'solved') * L + hazard * c_probe
      b1 first        c_b1 + T surcharge + P(b1 times out) * b2 route
      use             amortized build (C_build / remaining queries, 0 once built) + c_use
    build value = remaining queries * (best no-structure cost - c_use) - C_build (0 once built)."""
    (i, built, nH, nN), (b, usage, reduced, cand, cand_valid, ev) = state
    R = cfg.R
    bH, bM, bF, bX = b
    miss = 1.0 - cfg.q
    p_false = (1.0 - bH) * miss
    p_solved = bH + p_false
    p_h_solved = bH / p_solved if p_solved > 0 else 0.0
    p_b1 = bH + bM + (bF if reduced else 0.0)
    hazard = 0.0 if ev else cfg.p_event
    t_h = getattr(cfg, "t_hard", 0.0)
    hard = t_h * cfg.c_b2 * (bF + bX)
    side = cfg.D_side * (1.0 - bH)
    c_b2_route = (cfg.c_b2 + hard) * (1.0 + hazard)
    c_probe = cfg.c_probe + side + (1.0 - p_solved) * c_b2_route + p_false * cfg.L + hazard * cfg.c_probe
    c_b1 = cfg.c_b1 + hard + (1.0 - p_b1) * c_b2_route
    rem = max(cfg.k - i, 1)
    c_use = (0.0 if built else cfg.C_build / rem) + cfg.c_use
    best_ns = min(c_b2_route, c_probe, c_b1)
    build_value = 0.0 if built else rem * (best_ns - cfg.c_use) - cfg.C_build
    d = getattr(cfg, "deadline", 0)
    steps_left = (d - (usage >> STEP_SHIFT)) / 2.0 if d > 0 else 1.0
    cand_ok = cand != pw.C_NONE and cand_valid
    trust = 1.0 if (cand_ok and cand == pw.C_EXACT) else (p_h_solved if (cand_ok and cand == pw.C_PROBE) else 0.0)
    feats = [bH, bM, bF, bX,
             bH, p_false, p_h_solved, p_b1,
             side / R, hard / R, hazard,
             _clip(c_b2_route / R), _clip(c_probe / R), _clip(c_b1 / R), _clip(c_use / R), _clip(min(best_ns, c_use) / R),
             _clip(build_value / R), rem / 8.0, float(built),
             float(cand_ok), float(cand_ok and cand == pw.C_EXACT), trust,
             steps_left]
    assert len(feats) == N_FACTOR_FEATURES
    return feats


# ---------------------------------------------------------------------------------------------------------
# decision types (balanced counterfactual sets): named visible histories, evaluated at the same history in every
# member of a matched set (the history must be possible in every member).

DECISION_TYPES = {
    # name: visible history prefix (records (a, o, event, reveal)); the decision is taken after the prefix
    "first": (),
    "after_probe_solved": ((pw.A_PROBE, pw.O_SOLVED, False, None),),
    "after_probe_failed": ((pw.A_PROBE, pw.O_FAILED, False, None),),
    "after_b1_timeout": ((pw.A_B1, pw.O_TIMEOUT, False, None),),
    # later queries (k >= 2): first decision of query 2 after query 1 was solved exactly and the verifier revealed
    # its type (H, or not-H = M) -- where the correlated factor's relevance arrives (design v2 revision 1)
    "q2_after_H": ((pw.A_B2, pw.O_SOLVED, False, None), (pw.A_COMMIT, pw.O_CORRECT, False, pw.TH)),
    "q2_after_notH": ((pw.A_B2, pw.O_SOLVED, False, None), (pw.A_COMMIT, pw.O_CORRECT, False, pw.TM)),
}


def history_possible(s, history) -> tuple:
    """(possible, state): replay a visible history in solver s's configuration with positive probability."""
    cfg = s.cfg
    st = pw.initial_state(cfg)
    for a, o, e, rev in history:
        if st[0][0] >= cfg.k or a not in available(cfg, st):
            return False, None
        ok = any(p > 0 and oo == o and ee == e and (rev is None or rr == rev)
                 for p, oo, ee, rr, _, _ in s.transitions(st, a))
        if not ok:
            return False, None
        st = advance(cfg, st, a, o, e, rev)
    if st[0][0] >= cfg.k:
        return False, None
    return True, st


def decision_at(s, history, eps: float = pw.EPS):
    ok, st = history_possible(s, history)
    if not ok:
        return None
    q = s.q_values(st)
    opt = eps_set(q, eps)
    return {"state": st, "opt": sorted(opt), "unique": len(opt) == 1, "pi": s.pi_star(st),
            "Q": {int(a): q[a] for a in sorted(q)}}



# ---------------------------------------------------------------------------------------------------------
# split table v3 (b6), design v2 revisions 1, 4, 6.  Held-out challenge families are REGISTERED from the official
# structural screen (trackb-screen.md section 3; registry B-SCREEN) before any model exists; they, and every superset,
# are never in training, dev or selection.  Historical U+E / S+C / U+C are evaluated as challenge sets; under design
# v2 some training arms contain them (B2 / dose), which the evaluation flags per arm (HIST_IN_TRAINING).

SPLIT_TABLE_VERSION = "probeworld-split-v3"
HOLD_FAMILIES = ("SCE", "UCE")  # (primary, secondary): registered from the official screen (trackb-screen.md 3)
HIST_FAMILIES = ("UE", "SC", "UC")  # historical challenge pairs
PAIRS4 = tuple(all_families(V1_FACTORS, (2,)))  # US UC UE SC SE CE
TRAIN_STREAM_BASE = 6_200_000_000  # training type streams (shared by every arm: common random numbers)


def is_superset(key: str, fam: str) -> bool:
    return key != "0" and set(fam) <= set(key)


def excluded_from_training(key: str) -> bool:
    """Never trained on: a held-out family or any combination containing one (its interaction)."""
    return any(is_superset(key, f) for f in HOLD_FAMILIES)


def constituents(fam: str) -> tuple:
    return tuple("".join(c) for c in itertools.combinations(fam, 2))


PRIMARY = HOLD_FAMILIES[0]
# base pair: the pair that is a constituent of no held family (U+S for the two C+E triples)
BASE_PAIR = next(p for p in PAIRS4 if not any(p in constituents(f) for f in HOLD_FAMILIES))
VARIETY_PAIRS = tuple(p for p in PAIRS4 if p not in constituents(PRIMARY))  # dose-0 variety (incl. BASE_PAIR)
# dose order (registered): constituent pairs of the primary, non-historical first, then screening order
DOSE_ORDER = tuple(sorted(constituents(PRIMARY), key=lambda p: (p in HIST_FAMILIES, SCREEN_FAMILY_ORDER.index(p))))


def arm_pairs(arm: str) -> tuple:
    """Pair types sharing the pair slots of a training arm (the none / single / pair shares are fixed)."""
    if arm in ("B0", "B1"):
        return (BASE_PAIR,)
    if arm in ("B2", "B3"):
        return VARIETY_PAIRS
    if arm.startswith("dose"):
        return (BASE_PAIR,) + DOSE_ORDER[:int(arm[4:])]
    raise ValueError(arm)


ARM_SIZE = {"B0": 1, "B1": 2, "B2": 1, "B3": 2, "dose1": 1, "dose2": 1, "dose3": 1}  # x N configurations
ARM_REPLACES = {"B2": "B0", "B3": "B1", "dose1": "B0", "dose2": "B0", "dose3": "B0"}


def arm_composition(arm: str, n: int) -> dict:
    """Target counts per combination type (family key) for an arm of n * ARM_SIZE configurations.
    B0: none 1/6, each single 1/6, the base pair 1/6 (so U and S are on in 1/3 of the configurations, C and E in 1/6).
    Every other arm keeps the none share, the pair share and every per-factor frequency of B0 at its size: its pair
    slots are split equally over arm_pairs(arm) and the singles absorb the difference (replacement)."""
    size = n * ARM_SIZE[arm]
    assert size % 6 == 0, "N must be a multiple of 6"
    sixth = size // 6
    target_f = {f: sixth + (sixth if f in BASE_PAIR else 0) for f in V1_FACTORS}
    pairs = arm_pairs(arm)
    xs = {p: sixth // len(pairs) + (1 if j < sixth % len(pairs) else 0) for j, p in enumerate(pairs)}
    comp = {"0": sixth}
    for f in V1_FACTORS:
        comp[f] = target_f[f] - sum(x for p, x in xs.items() if f in p)
        assert comp[f] >= 0
    comp.update(xs)
    assert sum(comp.values()) == size
    return comp


def factor_frequency(comp: dict) -> dict:
    tot = sum(comp.values())
    return {f: sum(c for k, c in comp.items() if f in k) / tot for f in V1_FACTORS}


TYPE_STREAM_STRIDE = 1_000_000
TRAIN_TYPES = ("0",) + V1_FACTORS + PAIRS4  # every combination type a training arm may use (index = its stream)


def type_seed(key: str, j: int) -> int:
    """Training stream of one combination type: TRAIN_STREAM_BASE + 1e6 * type index + j (shared by every arm)."""
    assert 0 <= j < TYPE_STREAM_STRIDE and not excluded_from_training(key)
    return TRAIN_STREAM_BASE + TYPE_STREAM_STRIDE * TRAIN_TYPES.index(key) + j


def type_config(key: str, j: int) -> Config6:
    seed = type_seed(key, j)
    cell, k, _ = draw_params(seed, pw.TRAIN_CELLS, V3_K, (key,))
    return make_config(cell, k, flags_from_key(key), seed)


FIRST_CLASSES = {"probe": {pw.A_PROBE}, "exact": {pw.A_B1, pw.A_B2}, "gather": {pw.A_INSPECT, pw.A_PROP},
                 "structure": {pw.A_BUILD, pw.A_USE}, "terminal": {pw.A_COMMIT, pw.A_COMMIT_INF, pw.A_ABSTAIN}}


def first_class(s) -> str:
    a = s.pi_star(pw.initial_state(s.cfg))
    return next(c for c, acts in FIRST_CLASSES.items() if a in acts)


def make_classifier():
    """first_class of a configuration, memoized by configuration; the DP table is discarded (bounded memory)."""
    memo = {}

    def classify(cfg):
        c = memo.get(cfg)
        if c is None:
            s = ExactSolver(cfg)
            s.value(pw.initial_state(cfg))
            c = memo[cfg] = first_class(s)
        return c
    classify.memo = memo
    return classify


def build_arm_pools(n: int, arms=("B0", "B1", "B2", "B3", "dose1", "dose2", "dose3"), classify=None, max_tries=40):
    """Deterministic training pools of every arm, as lists of (type, stream index).  B0/B1 take the first counts of
    each type stream (B0 is the first half of B1 type by type).  A replacement arm starts from its reference (B0 or
    B1) and, type by type, drops the LAST surplus configurations and adds configurations of the types it has more of;
    the added configurations are matched to the dropped ones' optimal-first-action classes (for each dropped class,
    the next unused configuration of the added type with that class; after max_tries candidates, the next unused one:
    reported as unmatched).  classify(cfg) -> optimal-first-action class (make_classifier()).  Returns (pools, report)."""
    classify = classify or make_classifier()
    pools, report = {}, {}
    for arm in arms:
        comp = arm_composition(arm, n)
        if arm not in ARM_REPLACES:
            pools[arm] = [(key, j) for key in TRAIN_TYPES if key in comp for j in range(comp[key])]
            report[arm] = {"composition": comp, "factor_frequency": factor_frequency(comp)}
            continue
        ref = ARM_REPLACES[arm]
        if ref not in pools:
            raise ValueError(f"{arm} needs {ref}")
        rcomp = arm_composition(ref, n)
        keep, dropped = [], []
        for key in TRAIN_TYPES:
            have = [x for x in pools[ref] if x[0] == key]
            want = comp.get(key, 0)
            keep += have[:want]
            dropped += have[want:]
        need = []
        for key in TRAIN_TYPES:
            extra = comp.get(key, 0) - rcomp.get(key, 0)
            need += [key] * max(extra, 0)
        dclasses = [classify(type_config(*x)) for x in dropped]
        assert len(need) == len(dropped)
        added, unmatched = [], 0
        taken_by = {}
        for key, cls in zip(need, dclasses):
            taken = taken_by.setdefault(key, set())
            start = rcomp.get(key, 0)
            j, scanned, pick = start, 0, None
            while scanned < max_tries:
                if j not in taken:
                    scanned += 1
                    if classify(type_config(key, j)) == cls:
                        pick = j
                        break
                j += 1
            if pick is None:
                pick = start
                while pick in taken:
                    pick += 1
                unmatched += 1
            taken.add(pick)
            added.append((key, pick))
        pools[arm] = keep + added
        cnt = {}
        for key, _ in pools[arm]:
            cnt[key] = cnt.get(key, 0) + 1
        assert cnt == {k: v for k, v in comp.items() if v}, (cnt, comp)
        mix_ref = _class_mix([classify(type_config(*x)) for x in pools[ref]])
        mix = _class_mix([classify(type_config(*x)) for x in pools[arm]])
        report[arm] = {"composition": comp, "factor_frequency": factor_frequency(comp), "replaces": ref,
                       "dropped": len(dropped), "added": len(added), "unmatched_first_class": unmatched,
                       "first_class_mix": mix, "first_class_mix_ref": mix_ref}
    return pools, report


def _class_mix(classes):
    return {c: classes.count(c) for c in FIRST_CLASSES}


def _splits6():
    sp = {
        # name: (combos (family keys), config seed base, role)
        "b6_dev": (TRAIN_TYPES, 6_300_000_000, "dev"),  # monitoring only; final checkpoints are used
        "b6_test_base": (("0",) + V1_FACTORS + (BASE_PAIR,), 6_310_000_000, "test"),
        "b6_test_pairs": (PAIRS4, 6_320_000_000, "test"),
    }
    for j, f in enumerate(HOLD_FAMILIES):
        sp[f"b6_hold_{f}"] = ((f,), 6_400_000_000 + 10_000_000 * j, "hold")
    for j, f in enumerate(HIST_FAMILIES):
        sp[f"b6_hist_{f}"] = ((f,), 6_500_000_000 + 10_000_000 * j, "hist")
    return sp


SPLITS6 = _splits6()
CF_BASE = {f: 6_600_000_000 + 10_000_000 * j for j, f in enumerate(HOLD_FAMILIES + HIST_FAMILIES)}
B6_EVAL_SPLITS = tuple(s for s, v in SPLITS6.items() if v[2] in ("test", "hold", "hist"))
B6_HOLD_SPLITS = tuple(s for s, v in SPLITS6.items() if v[2] == "hold")
HIST_IN_TRAINING = {arm: sorted(set(arm_pairs(arm)) & set(HIST_FAMILIES)) for arm in ARM_SIZE}


def split6_entry(split: str) -> tuple:
    """(combos, config seed base, role) of a b6 / b6c / b6d split."""
    if split in SPLITS6:
        return SPLITS6[split]
    return SPLITS6C[split] if split in SPLITS6C else SPLITS6D[split]


def split6_config(split: str, index: int) -> Config6:
    combos, base, _ = split6_entry(split)
    assert 0 <= index < MAX_POOL
    seed = base + index
    cell, k, key = draw_params(seed, pw.TRAIN_CELLS, V3_K, tuple(combos))
    return make_config(cell, k, flags_from_key(key), seed)


def world_seed6(split: str, index: int, rep: int) -> int:
    return split6_entry(split)[1] + WORLD_SEED_OFFSET + index * 1000 + rep


def split_table_digest_input():
    return (SPLIT_TABLE_VERSION, GENERATOR_VERSION, HOLD_FAMILIES, HIST_FAMILIES, BASE_PAIR, VARIETY_PAIRS,
            DOSE_ORDER, tuple(sorted(SPLITS6.items())), tuple(sorted(CF_BASE.items())), V3_K, CORR_RANGE,
            P_EVENT_RANGE, TRAIN_TYPES)


def audit_split_table(n: int = 384) -> dict:
    """Generator-parameter audit: no held-out family (or superset) in any training / dev / test pool; every factor
    of every hold is trained singly; per-factor frequency and none/single/pair shares equal across the arms of a
    volume level; seeds inside their registered sub-ranges and pairwise disjoint."""
    out = {}
    trainable = [s for s, v in SPLITS6.items() if v[2] in ("dev", "test")]
    out["holds_absent_from_dev_test"] = all(not excluded_from_training(c) for s in trainable for c in SPLITS6[s][0])
    out["holds_absent_from_arms"] = all(not excluded_from_training(p) for arm in ARM_SIZE for p in arm_pairs(arm))
    out["singles_in_every_arm"] = all(arm_composition(arm, n)[f] > 0 for arm in ARM_SIZE for f in V1_FACTORS)
    ff = {arm: factor_frequency(arm_composition(arm, n)) for arm in ARM_SIZE}
    out["factor_frequency_matched"] = all(abs(ff[arm][f] - ff["B0"][f]) < 1e-12 for arm in ARM_SIZE for f in V1_FACTORS)
    shares = {arm: (arm_composition(arm, n)["0"] / (n * ARM_SIZE[arm]),
                    sum(v for k, v in arm_composition(arm, n).items() if len(k) == 2) / (n * ARM_SIZE[arm]))
              for arm in ARM_SIZE}
    out["none_pair_shares_matched"] = all(shares[a] == shares["B0"] for a in ARM_SIZE)
    out["dose0_variety_has_no_primary_constituent"] = not (set(VARIETY_PAIRS) & set(constituents(PRIMARY)))
    ranges = []
    for s, (_, base, role) in SPLITS6.items():
        ranges.append((base, base + MAX_POOL, s))
        ranges.append((base + WORLD_SEED_OFFSET, base + WORLD_SEED_OFFSET + MAX_POOL * 1000, s + ":worlds"))
    for key in TRAIN_TYPES:
        b = type_seed(key, 0)
        ranges.append((b, b + TYPE_STREAM_STRIDE, "train:" + key))
    for f, b in CF_BASE.items():
        ranges.append((b, b + MAX_POOL, "cf_" + f))
    ranges.sort()
    out["seed_blocks_disjoint"] = all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
    lo, hi = TRACKB_RANGE
    out["inside_trackb_range"] = all(lo <= a[0] and a[1] <= hi for a in ranges)
    sub = {"dev": "dev_test", "test": "dev_test", "hold": "hold", "hist": "hist"}
    out["inside_subranges"] = all(
        SUBRANGES[sub[r]][0] <= b and b + WORLD_SEED_OFFSET + MAX_POOL * 1000 <= SUBRANGES[sub[r]][1]
        for s, (_, b, r) in SPLITS6.items()) and all(
        SUBRANGES["cf"][0] <= b and b + MAX_POOL <= SUBRANGES["cf"][1] for b in CF_BASE.values()) and (
        SUBRANGES["train"][0] <= type_seed(TRAIN_TYPES[0], 0)
        and type_seed(TRAIN_TYPES[-1], 0) + TYPE_STREAM_STRIDE <= SUBRANGES["train"][1])
    out["check_subranges"] = check_subranges()
    out["pass"] = all(out.values())
    return out


# ---------------------------------------------------------------------------------------------------------
# B-FACT-C fresh confirmation split (split set 'b6c'; registry B-FACT-C).  ADDITIVE: SPLITS6, CF_BASE, B6_* and
# split_table_digest_input are unchanged (pinned by tests/test_campaign06_bfactc.py).  Same generator (probeworld-v3),
# eligibility, octet construction and near-miss logic; only the seed bases differ.  The hold base is 6.42e9 (the next
# slot of the b6 hold pattern 6.4e9 + 1e7 j): 6.45e9 is b6_hold_SCE's eval-world block (base + WORLD_SEED_OFFSET).

B6C_VERSION = "probeworld-split-v3-b6c"
SPLITS6C = {"b6c_hold_SCE": (("SCE",), 6_420_000_000, "hold")}
CF_BASE_C = {"SCE": 6_650_000_000}
B6C_EVAL_SPLITS = tuple(SPLITS6C)

# P2-CONFIRM fresh confirmation split (split set 'b6d'; extended-07 registry P2-CONFIRM).  ADDITIVE like b6c: SPLITS6,
# CF_BASE, SPLITS6C, CF_BASE_C and every b6 / b6c output are unchanged (pinned by tests/test_campaign07_b6d.py).  Same
# generator, eligibility, octet construction and near-miss logic; only the seed bases differ.  Hold base 6.43e9 (the
# slot after b6c's 6.42e9; eval worlds 6.48e9, after b6c's 6.47e9 block), octets at 6.66e9 (after b6c's 6.65e9).
# Registered in research/campaigns/extended-07/seed-ranges.json ("ext07 P2 CONFIRM ... (b6d ...)").
B6D_VERSION = "probeworld-split-v3-b6d"
SPLITS6D = {"b6d_hold_SCE": (("SCE",), 6_430_000_000, "hold")}
CF_BASE_D = {"SCE": 6_660_000_000}
B6D_EVAL_SPLITS = tuple(SPLITS6D)


def is_split6(split: str) -> bool:
    return split in SPLITS6 or split in SPLITS6C or split in SPLITS6D


def _pool_blocks(splits: dict, cf: dict, cf_prefix: str) -> list:
    out = []
    for s, (_, base, _) in splits.items():
        out.append((base, base + MAX_POOL, s))
        out.append((base + WORLD_SEED_OFFSET, base + WORLD_SEED_OFFSET + MAX_POOL * 1000, s + ":worlds"))
    for f, b in cf.items():
        out.append((b, b + MAX_POOL, cf_prefix + f))
    return out


def audit_b6c() -> dict:
    """b6c seed blocks: pairwise disjoint and disjoint from every b6 block (pools, eval worlds, training streams,
    counterfactual streams); inside the Track B range; holds (configurations + worlds) inside the 'hold' sub-range and
    octets inside 'cf' (protocol_bases), or all inside 'dev_smoke' (tests / smokes); the b6c families are b6 holds."""
    out = {}
    new = _pool_blocks(SPLITS6C, CF_BASE_C, "b6c_cf_")
    old = _pool_blocks(SPLITS6, CF_BASE, "cf_") + [
        (type_seed(key, 0), type_seed(key, 0) + TYPE_STREAM_STRIDE, "train:" + key) for key in TRAIN_TYPES]
    allb = sorted(new + old)
    out["disjoint"] = all(a[1] <= b[0] for a, b in zip(allb, allb[1:]))
    lo, hi = TRACKB_RANGE
    out["inside_trackb_range"] = all(lo <= a[0] and a[1] <= hi for a in new)

    def inside(name, a, b):
        return SUBRANGES[name][0] <= a and b <= SUBRANGES[name][1]

    holds = [(b, b + WORLD_SEED_OFFSET + MAX_POOL * 1000) for _, b, _ in SPLITS6C.values()]
    cfs = [(b, b + MAX_POOL) for b in CF_BASE_C.values()]
    out["protocol_bases"] = all(inside("hold", *x) for x in holds) and all(inside("cf", *x) for x in cfs)
    out["dev_smoke_bases"] = all(inside("dev_smoke", *x) for x in holds + cfs)
    out["inside_subranges"] = out["protocol_bases"] or out["dev_smoke_bases"]
    out["families_are_holds"] = all(set(c) <= set(HOLD_FAMILIES) and r == "hold" for c, _, r in SPLITS6C.values()) \
        and set(CF_BASE_C) <= set(HOLD_FAMILIES)
    out["b6_audit"] = audit_split_table()["pass"]
    out["pass"] = all(out[k] for k in ("disjoint", "inside_trackb_range", "inside_subranges", "families_are_holds",
                                       "b6_audit"))
    return out


def audit_b6d() -> dict:
    """b6d seed blocks: pairwise disjoint and disjoint from every b6 block (pools, eval worlds, training streams,
    counterfactual streams) AND every b6c block (pool, eval worlds, octets); inside the Track B range; the hold pool
    (configurations + eval worlds) inside the 'hold' sub-range and the octets inside 'cf' (protocol_bases), or all
    inside 'dev_smoke' (tests / smokes); the b6d families are b6 holds; the b6 and b6c audits still pass."""
    out = {}
    new = _pool_blocks(SPLITS6D, CF_BASE_D, "b6d_cf_")
    old = _pool_blocks(SPLITS6, CF_BASE, "cf_") + _pool_blocks(SPLITS6C, CF_BASE_C, "b6c_cf_") + [
        (type_seed(key, 0), type_seed(key, 0) + TYPE_STREAM_STRIDE, "train:" + key) for key in TRAIN_TYPES]
    allb = sorted(new + old)
    out["disjoint"] = all(a[1] <= b[0] for a, b in zip(allb, allb[1:]))
    out["n_blocks_checked"] = len(old)
    lo, hi = TRACKB_RANGE
    out["inside_trackb_range"] = all(lo <= a[0] and a[1] <= hi for a in new)

    def inside(name, a, b):
        return SUBRANGES[name][0] <= a and b <= SUBRANGES[name][1]

    holds = [(b, b + WORLD_SEED_OFFSET + MAX_POOL * 1000) for _, b, _ in SPLITS6D.values()]
    cfs = [(b, b + MAX_POOL) for b in CF_BASE_D.values()]
    out["protocol_bases"] = all(inside("hold", *x) for x in holds) and all(inside("cf", *x) for x in cfs)
    out["dev_smoke_bases"] = all(inside("dev_smoke", *x) for x in holds + cfs)
    out["inside_subranges"] = out["protocol_bases"] or out["dev_smoke_bases"]
    out["families_are_holds"] = all(set(c) <= set(HOLD_FAMILIES) and r == "hold" for c, _, r in SPLITS6D.values()) \
        and set(CF_BASE_D) <= set(HOLD_FAMILIES)
    out["b6_audit"] = audit_split_table()["pass"]
    out["b6c_audit"] = audit_b6c()["pass"]
    out["pass"] = all(out[k] for k in ("disjoint", "inside_trackb_range", "inside_subranges", "families_are_holds",
                                       "b6_audit", "b6c_audit"))
    return out


# ---------------------------------------------------------------------------------------------------------
# balanced counterfactual sets: all sub-combinations of a held-out family at the same prices (none / A / B / A+B for
# a pair), plus a near-miss full-family configuration that does NOT flip, with exact labels per decision type.

FACTOR_WEAK = {"U": ("q", 0.8), "S": ("D_side", 5.0), "C": ("corr", CORR_RANGE[0]), "E": ("p_event", P_EVENT_RANGE[0]),
               "D": ("deadline", 2), "T": ("t_hard", 0.5)}  # weakest ON value inside the generator range
NEAR_MISS_LAMBDAS = (0.5, 1.0)


def weaken(cfg: Config6, factor: str, lam: float):
    field, weak = FACTOR_WEAK[factor]
    cur = getattr(cfg, field)
    if field == "deadline":
        return replace(cfg, deadline=2) if (cur == 1 and lam >= 1.0) else None
    new = round(cur + lam * (weak - cur), 4)
    return None if abs(new - cur) < 1e-9 else replace(cfg, **{field: new})


def _strip(d):
    return None if d is None else {k: v for k, v in d.items() if k != "state"}


def inherited(nm_s, fam, h, level: int = 1) -> bool:
    """At decision type h, the full-family decision of nm is predicted by (is eps-optimal for) the optimal action of
    one of its sub-combinations with `level` factors (1: the single-factor members; |fam| - 1: one-factor ablations)."""
    d = decision_at(nm_s, DECISION_TYPES[h])
    if d is None:
        return False
    for T in itertools.combinations(fam, level):
        sa = ExactSolver(restrict(nm_s.cfg, T))
        da = decision_at(sa, DECISION_TYPES[h])
        if da is not None and da["pi"] in d["opt"]:
            return True
    return False


def counterfactual_set(fam: str, index: int, base=None, near_miss: bool = True) -> dict:
    """One matched set for held-out family `fam`: members = every sub-combination (same prices, prior, k), each with
    exact labels at every decision type whose visible history is possible in that member; per decision type the
    joint-flip flag (no one-factor ablation's optimal action is eps-optimal for the full family) and, for flips, a
    near-miss full-family configuration (factor parameters moved toward their weakest ON value) whose decision is
    inherited from an ablation and differs from the full member's."""
    fam = canon(fam)
    base = CF_BASE[fam] if base is None else base
    seed = base + index
    cell, k, _ = draw_params(seed, pw.TRAIN_CELLS, V3_K, (fam,))
    c = make_config(cell, k, flags_from_key(fam), seed)
    full = tuple(fam)
    members, solvers = {}, {}
    for m in range(len(full) + 1):
        for T in itertools.combinations(full, m):
            key = "".join(T) or "0"
            s = ExactSolver(restrict(c, T))
            solvers[key] = s
            members[key] = {"config": config_dict6(s.cfg),
                            "labels": {h: _strip(decision_at(s, hist)) for h, hist in DECISION_TYPES.items()}}
    out = {"family": fam, "index": index, "seed": seed, "k": k, "members": members, "types": {}}
    for h in DECISION_TYPES:
        dfull = members[fam]["labels"][h]
        singles = [members[f]["labels"][h] for f in full]
        abl = [members["".join(x for x in full if x != f) or "0"]["labels"][h] for f in full]
        if dfull is None or any(d is None for d in singles + abl):
            continue
        # flip (primary, every family size): no single-factor member's optimal action is eps-optimal for the full
        # family (for a pair these are its one-factor ablations); flip_ablations (triples): no one-factor ablation's
        d0 = members["0"]["labels"][h]
        ent = {"flip": all(d["pi"] not in dfull["opt"] for d in singles),
               "flip_ablations": all(d["pi"] not in dfull["opt"] for d in abl),
               "changed_vs_none": d0 is not None and d0["pi"] not in dfull["opt"],
               "unique": dfull["unique"], "near_miss": None}
        if ent["flip"] and near_miss and dfull["unique"]:
            ent["near_miss"] = find_near_miss(solvers[fam], fam, h, dfull)
        out["types"][h] = ent
    return out


def find_near_miss(s_full, fam, h, dfull):
    cands = []
    for f in fam:
        for lam in NEAR_MISS_LAMBDAS:
            c2 = weaken(s_full.cfg, f, lam)
            if c2 is not None:
                cands.append((f"{f}@{lam}", c2))
    c2 = s_full.cfg
    for f in fam:
        c2 = weaken(c2, f, 1.0) or c2
    if c2 != s_full.cfg:
        cands.append(("all@1.0", c2))
    for name, cfg in cands:
        s = ExactSolver(cfg)
        d = decision_at(s, DECISION_TYPES[h])
        if d is None or not d["unique"] or d["pi"] in dfull["opt"]:
            continue
        if inherited(s, fam, h, 1):
            return {"move": name, "config": config_dict6(cfg), "label": _strip(d)}
    return None


def config_dict6(cfg) -> dict:
    d = asdict(cfg)
    for key in ("enable", "prior", "features"):
        d[key] = list(d[key])
    return d


def config_from_dict6(d: dict) -> Config6:
    d = {k: v for k, v in d.items() if k in {f.name for f in fields(Config6)}}
    return Config6(**{**d, "enable": tuple(d["enable"]), "prior": tuple(d["prior"]), "features": tuple(d["features"])})


def one_factor_pairs(cfset: dict):
    """On-manifold interventions inside a matched set: (member without f, member with f, f) for every factor f."""
    fam = cfset["family"]
    out = []
    for key in cfset["members"]:
        for f in fam:
            if f not in key:
                out.append((key, canon((key if key != "0" else "") + f), f))
    return out
