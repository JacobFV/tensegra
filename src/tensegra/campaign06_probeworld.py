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
GENERATOR_VERSION = "probeworld-v3-gen2"  # gen1 (local pilot only) had factor G instead of T
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
V3_K = (1, 2, 4)  # probeworld-v3 horizon support (composition, not horizon, is the Track B question; k = 8 dropped for cost)
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
    return Config6(**{f.name: getattr(cfg, f.name) for f in fields(pw.Config)}, **kw)


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
    """v3 configuration: v1 prices/prior/U,S,C,E parameters from pw.make_config (rng seed*7+1, unchanged), then the
    D and G parameters from a separate stream (seed*11+3), drawn whether or not the factor is on (so switching a
    factor never shifts any other draw)."""
    flags = tuple(flags) + (False,) * (6 - len(flags))
    base = pw.make_config(cell, k, flags[:4], random.Random(seed * 7 + 1))
    r2 = random.Random(seed * 11 + 3)
    d = r2.choice(DEADLINE_VALUES)
    t = round(r2.uniform(*T_RANGE), 3)
    return from_v1(base, deadline=d if flags[4] else 0, t_hard=t if flags[5] else 0.0)


def draw_params(seed: int, cells, ks, combos) -> tuple:
    rng = random.Random(seed)
    return rng.choice(cells), rng.choice(ks), rng.choice(combos)


def stream_config(base: int, index: int, cells, ks, combos):
    seed = base + index
    cell, k, cmb = draw_params(seed, cells, ks, combos)
    return make_config(cell, k, cmb, seed), (cell, k, cmb, seed)


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
    cell, k, _ = draw_params(seed, cells, ks, (None,))
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
# supplied factorized public state (SUPPLIED-FACTORIZED inputs; LEARNED-FACTORIZED auxiliary targets)

FACTOR_FEATURES = (
    # posterior over the hidden type (exact; a function of the visible history)
    "belief_H", "belief_M", "belief_F", "belief_X",
    # detection reliability (U)
    "miss_rate", "p_false_solved", "p_H_given_solved",
    # side-effect severity (S)
    "side_cost_rel", "expected_side_cost_rel",
    # correlation evidence (C)
    "corr", "revealed_H_frac", "revealed_notH_frac",
    # event hazard (E)
    "event_hazard_active", "event_fired_this_query",
    # deadline / remaining steps (D)
    "deadline_active", "steps_left_rel", "last_step",
    # type-dependent exact cost (T)
    "hard_surcharge_rel", "reduced",
    # amortization / remaining horizon
    "remaining_queries_rel", "is_last_query", "built", "amortized_build_rel",
    # candidate validity
    "candidate_valid", "candidate_exact", "candidate_probe_trust",
    # remaining expected cost by strategy (myopic, one query, public closed forms; NOT Q*)
    "cost_exact_b2_rel", "cost_probe_then_b2_rel", "cost_b1_route_rel", "cost_use_rel",
)
N_FACTOR_FEATURES = len(FACTOR_FEATURES)


def factor_features(cfg, state: tuple) -> list:
    """Deterministic public factor values at a public information state (config + state only; no hidden type).
    Price-scale features are divided by R.  The strategy costs are one-query myopic closed forms (documented in
    trackb-screen.md section 6), not DP values."""
    (i, built, nH, nN), (b, usage, reduced, cand, cand_valid, ev) = state
    R = cfg.R
    bH, bM, bF, bX = b
    miss = 1.0 - cfg.q
    p_false = (1.0 - bH) * miss
    p_solved = bH + p_false
    p_h_solved = bH / p_solved if p_solved > 0 else 0.0
    d = getattr(cfg, "deadline", 0)
    t_h = getattr(cfg, "t_hard", 0.0)
    used = usage >> STEP_SHIFT
    steps_left = (d - used) if d > 0 else 3
    rem = max(cfg.k - i, 1)
    amort = 0.0 if built else cfg.C_build / rem
    hazard = 0.0 if ev else cfg.p_event
    # P(b1 times out) under the current belief and reduction state; expected T surcharge of an exact call
    p_to = bF * (0.0 if reduced else 1.0) + bX
    hard = t_h * cfg.c_b2 * (bF + bX)
    c_b2_route = cfg.c_b2 + hard + hazard * cfg.c_b2  # b2 then commit; an event after b2 forces a rerun (expected)
    side = cfg.D_side * (1.0 - bH)
    wrong_probe = p_false * cfg.L  # committing a false 'solved' probe
    c_probe_route = (cfg.c_probe + side + (1.0 - p_solved) * (cfg.c_b2 + hard) + wrong_probe
                     + hazard * cfg.c_probe)
    c_b1_route = cfg.c_b1 + hard + p_to * (cfg.c_b2 + t_h * cfg.c_b2)
    c_use_route = amort + cfg.c_use
    cand_ok = cand != pw.C_NONE and cand_valid
    trust = 1.0 if (cand_ok and cand == pw.C_EXACT) else (p_h_solved if (cand_ok and cand == pw.C_PROBE) else 0.0)
    feats = [bH, bM, bF, bX,
             miss, p_false, p_h_solved,
             cfg.D_side / R, side / R,
             cfg.corr, nH / cfg.k, nN / cfg.k,
             hazard, float(ev),
             float(d > 0), steps_left / 3.0, float(d > 0 and steps_left == 1),
             hard / R, float(reduced),
             rem / 8.0, float(i == cfg.k - 1), float(built), min(amort / R, 10.0) / 10.0,
             float(cand_ok), float(cand_ok and cand == pw.C_EXACT), trust,
             c_b2_route / R, c_probe_route / R, c_b1_route / R, min(c_use_route / R, 10.0)]
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
}


def history_possible(s, history) -> tuple:
    """(possible, state): replay a visible history in solver s's configuration with positive probability."""
    cfg = s.cfg
    st = pw.initial_state(cfg)
    for a, o, e, rev in history:
        if st[0][0] >= cfg.k or a not in available(cfg, st):
            return False, None
        ok = any(abs(p) > 0 and oo == o and ee == e for p, oo, ee, _, _, _ in s.transitions(st, a))
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
