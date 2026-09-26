"""Probeworld (extended-04 Track B): rational probing, switching and recovery with exact labels.

Pure Python (no torch, no numpy).  Spec: research/campaigns/extended-04/probeworld.md.

Family (design v2 revision 10, review F13)
------------------------------------------
* Hidden instance type theta in TYPES = (H, M, F, X):
    H  heuristic-solvable (the cheap probe solves it; exact b1 solves it)
    M  exact-easy        (probe fails; exact b1 solves it)
    F  feasible-hard     (probe fails; exact b1 TIMES OUT unless propagation reduced the instance; b2 solves)
    X  infeasible        (probe fails; exact b1 TIMES OUT; b2 certifies infeasible; propagation may find a conflict)
  A b1 timeout is *unknown*, never infeasible: the posterior keeps mass on F and X.
* Public features (f1, f2) give a declared prior P(theta | features) (FEATURE_PRIORS).
* An episode is k related queries (k in {1, 2, 4, 8}), iid given the features (and, in the correlated
  condition, given a hidden binary latent z shifting P(H)).  After every terminal action the verifier
  reveals theta of that query (public audit feedback).
* Actions per query (each at most once per requirement epoch; see ``available``):
    probe, exact_b1, exact_b2, inspect (noisy easy/hard signal), prop (partial constraint propagation:
    conflict certifies X with prob p_conflict, otherwise the instance is reduced so that b1 solves F),
    build (reusable structure, C_build once per episode), use (structure: C_execute + C_return + C_verify,
    returns a verified answer), commit (irreversible; the best candidate; R if correct, else -L),
    commit_infeasible (irreversible; R iff theta == X, else -L), abstain (0).
* Conditions (flags derived from prices): unreliable detection (q < 1: a probe failure is reported as
  "solved" with prob 1 - q, yielding a wrong candidate), irreversible probe side effect (D_side > 0 charged
  whenever the probe runs on theta != H), correlated heuristic failure (shared latent z), requirement-change
  events (after any non-terminal action, once per query, with prob p_event).  An event invalidates the
  declared subset INVALIDATED_BY_EVENT: solution candidates (become stale; committing a stale candidate is
  wrong), the propagation reduction, and the per-epoch usage of probe/exact/prop/use (they may be re-run).
  Type evidence and the inspect signal stay valid (theta does not change).
* Utility = R * verified successes - sum of costs; every cost is one ledger line charged exactly once.

Labels
------
``ExactSolver`` runs memoized dynamic programming over the finite information-state space (the posterior
is a function of the public information state).  Q*(I, a) for every available action, V*(I), the set of
eps-optimal actions (registered EPS), with continuation policy CONTINUATION (= the optimal policy).
The information state is computed from the visible history only (``public_state_from_history``).
"""
from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, asdict, replace

sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

VERSION = "probeworld-v1"
CONTINUATION = "pi_star_exact_dp_v1"  # the continuation policy every Q* label assumes
EPS = 0.5  # registered epsilon (price units; R = 100 in the generated family)

TYPES = ("H", "M", "F", "X")
TH, TM, TF, TX = range(4)
ACTIONS = ("probe", "exact_b1", "exact_b2", "inspect", "prop", "build", "use",
           "commit", "commit_infeasible", "abstain")
(A_PROBE, A_B1, A_B2, A_INSPECT, A_PROP, A_BUILD, A_USE,
 A_COMMIT, A_COMMIT_INF, A_ABSTAIN) = range(len(ACTIONS))
N_ACTIONS = len(ACTIONS)
TERMINAL = frozenset({A_COMMIT, A_COMMIT_INF, A_ABSTAIN})
OUTCOMES = ("solved", "failed", "timeout", "infeasible", "easy", "hard", "conflict", "reduced",
            "built", "correct", "wrong", "abstained")
(O_SOLVED, O_FAILED, O_TIMEOUT, O_INFEASIBLE, O_EASY, O_HARD, O_CONFLICT, O_REDUCED,
 O_BUILT, O_CORRECT, O_WRONG, O_ABSTAINED) = range(len(OUTCOMES))
N_OUTCOMES = len(OUTCOMES)
# outcomes that are the action's own failure (case b)
FAILURE_OUTCOMES = {(A_PROBE, O_FAILED), (A_B1, O_TIMEOUT), (A_COMMIT, O_WRONG), (A_COMMIT_INF, O_WRONG)}
INVALIDATED_BY_EVENT = ("solution_candidate", "propagation_reduction",
                        "epoch_usage(probe,exact_b1,exact_b2,prop,use)")
STAGES = ("gather", "attempt_cheap", "attempt_exact", "structure", "answer", "abstain")
ACTION_STAGE = {A_INSPECT: 0, A_PROP: 0, A_PROBE: 1, A_B1: 2, A_B2: 2, A_BUILD: 3, A_USE: 3,
                A_COMMIT: 4, A_COMMIT_INF: 4, A_ABSTAIN: 5}

# usage bits (per requirement epoch, except inspect which is per query)
U_PROBE, U_B1, U_B2, U_PROP, U_USE, U_INSPECT = 1, 2, 4, 8, 16, 32
ACTION_BIT = {A_PROBE: U_PROBE, A_B1: U_B1, A_B2: U_B2, A_PROP: U_PROP, A_USE: U_USE, A_INSPECT: U_INSPECT}
EPOCH_BITS = U_PROBE | U_B1 | U_B2 | U_PROP | U_USE
# candidate kinds
C_NONE, C_PROBE, C_EXACT = 0, 1, 2

# declared prior P(theta | f1, f2): f1 = surface regularity (0..2), f2 = constraint density (0..1)
FEATURE_PRIORS = {
    (0, 0): (0.30, 0.35, 0.25, 0.10),
    (0, 1): (0.20, 0.25, 0.30, 0.25),
    (1, 0): (0.55, 0.25, 0.15, 0.05),
    (1, 1): (0.45, 0.20, 0.20, 0.15),
    (2, 0): (0.80, 0.12, 0.06, 0.02),
    (2, 1): (0.65, 0.12, 0.13, 0.10),
}


@dataclass(frozen=True)
class Config:
    """Public episode parameters (everything here is visible to the agent)."""
    prior: tuple = (0.8, 0.2, 0.0, 0.0)
    features: tuple = (-1, -1)  # (-1, -1) = hand-specified prior
    R: float = 100.0
    L: float = 100.0
    c_probe: float = 1.0
    c_b1: float = 20.0
    c_b2: float = 50.0
    c_inspect: float = 2.0
    c_prop: float = 5.0
    C_build: float = 1e9
    C_execute: float = 5.0
    C_return: float = 1.0
    C_verify: float = 1.0
    k: int = 1
    q: float = 1.0  # P(probe failure is detected | theta != H)
    D_side: float = 0.0  # irreversible probe side-effect cost when theta != H
    corr: float = 0.0  # correlated condition: P(H | z) = prior_H +/- corr (0 = off)
    p_event: float = 0.0  # requirement-change event probability (0 = off)
    eta: float = 0.2  # inspect noise
    p_conflict: float = 0.7  # P(prop finds conflict | X)
    enable: tuple = (True,) * N_ACTIONS  # action family switch (for hand-solvable examples)

    @property
    def c_use(self) -> float:
        return self.C_execute + self.C_return + self.C_verify

    @property
    def flags(self) -> tuple:
        return (self.q < 1.0, self.D_side > 0.0, self.corr > 0.0, self.p_event > 0.0)

    def rho(self, k: int | None = None) -> float:
        """rho_k = C_shortcut / (C_build/k + C_execute + C_return + C_verify), C_shortcut = c_b2 (direct
        exact route that resolves every type)."""
        k = self.k if k is None else k
        return self.c_b2 / (self.C_build / k + self.c_use)

    def public_vector(self) -> list:
        """Normalized public inputs for the model (prices / R, flags, k, prior)."""
        R = self.R
        return ([self.c_probe / R, self.c_b1 / R, self.c_b2 / R, self.c_inspect / R, self.c_prop / R,
                 self.L / R, min(self.C_build / R, 10.0) / 10.0, self.C_execute / R, self.C_return / R,
                 self.C_verify / R, self.D_side / R, self.q, self.corr, self.p_event, self.eta,
                 self.p_conflict, self.k / 8.0, math.log2(self.k) / 3.0, math.log(max(self.rho(), 1e-3)) / 3.0]
                + [float(f) for f in self.flags] + list(self.prior))


FLAG_NAMES = ("unreliable", "side_effect", "correlated", "events")
PUBLIC_DIM = len(Config().public_vector())


# ---------------------------------------------------------------------------------------------------------
# outcome model (single source of truth for the simulator and the DP belief update)

def outcome_dist(cfg: Config, a: int, theta: int, reduced: bool) -> list:
    """P(outcome | theta, action, reduced) for non-terminal actions: list of (p, outcome)."""
    if a == A_PROBE:
        if theta == TH:
            return [(1.0, O_SOLVED)]
        return [(cfg.q, O_FAILED), (1.0 - cfg.q, O_SOLVED)] if cfg.q < 1.0 else [(1.0, O_FAILED)]
    if a == A_B1:
        ok = theta in (TH, TM) or (theta == TF and reduced)
        return [(1.0, O_SOLVED if ok else O_TIMEOUT)]
    if a == A_B2 or a == A_USE:
        return [(1.0, O_INFEASIBLE if theta == TX else O_SOLVED)]
    if a == A_INSPECT:
        easy = theta in (TH, TM)
        return [(1.0 - cfg.eta, O_EASY if easy else O_HARD), (cfg.eta, O_HARD if easy else O_EASY)]
    if a == A_PROP:
        if theta == TX:
            return [(cfg.p_conflict, O_CONFLICT), (1.0 - cfg.p_conflict, O_REDUCED)]
        return [(1.0, O_REDUCED)]
    if a == A_BUILD:
        return [(1.0, O_BUILT)]
    raise ValueError(a)


BELIEF_DIGITS = 12


def _bkey(w) -> tuple:
    z = sum(w)
    if z <= 0:
        raise ValueError("impossible observation")
    return tuple(round(x / z, BELIEF_DIGITS) for x in w)


def update_belief(cfg: "Config", b: tuple, a: int, o: int, reduced: bool) -> tuple:
    """Bayes update of the query-type posterior after a non-terminal action's visible outcome."""
    return _bkey([b[t] * sum(p for p, oo in outcome_dist(cfg, a, t, reduced) if oo == o) if b[t] else 0.0
                  for t in range(4)])


def commit_correct(a: int, theta: int, cand: int, cand_valid: bool) -> bool:
    if a == A_COMMIT_INF:
        return theta == TX
    if a == A_COMMIT:
        if not cand_valid:
            return False
        return True if cand == C_EXACT else theta == TH
    return False


# ---------------------------------------------------------------------------------------------------------
# public information state (a function of the visible history only)
#   cross = (i, built, nH, nNotH)   nH/nNotH = revealed counts (tracked only in the correlated condition)
#   local = (belief, usage, reduced, cand, cand_valid, event)
#   belief = posterior over TYPES (rounded to BELIEF_DIGITS), itself computed from the visible history.

def theta_prior(cfg: "Config", nH: int, nN: int) -> tuple:
    p = cfg.prior
    if cfg.corr <= 0.0:
        return _bkey(p)
    pH = p[TH]
    rest = 1.0 - pH
    mix = [0.0] * 4
    zs = []
    for sgn in (1.0, -1.0):
        ph = min(max(pH + sgn * cfg.corr, 0.0), 1.0)
        zs.append((0.5 * (ph ** nH) * ((1.0 - ph) ** nN), ph))
    zsum = sum(w for w, _ in zs)
    for w, ph in zs:
        wz = w / zsum
        for t in range(4):
            mix[t] += wz * (ph if t == TH else ((1.0 - ph) * p[t] / rest if rest > 0 else 0.0))
    return _bkey(mix)


def local0(cfg: "Config", nH: int = 0, nN: int = 0) -> tuple:
    return (theta_prior(cfg, nH, nN), 0, False, C_NONE, False, False)


def initial_state(cfg: "Config") -> tuple:
    return ((0, False, 0, 0), local0(cfg))


def available(cfg: Config, state: tuple) -> tuple:
    (i, built, _, _), (b, usage, reduced, cand, cand_valid, event) = state
    if i >= cfg.k:
        return ()
    en = cfg.enable
    acts = []
    if en[A_PROBE] and not usage & U_PROBE:
        acts.append(A_PROBE)
    if en[A_B1] and not usage & (U_B1 | U_B2):
        acts.append(A_B1)
    if en[A_B2] and not usage & U_B2:
        acts.append(A_B2)
    if en[A_INSPECT] and not usage & U_INSPECT:
        acts.append(A_INSPECT)
    if en[A_PROP] and not usage & U_PROP:
        acts.append(A_PROP)
    if en[A_BUILD] and not built:
        acts.append(A_BUILD)
    if en[A_USE] and built and not usage & U_USE:
        acts.append(A_USE)
    if en[A_COMMIT] and cand != C_NONE:
        acts.append(A_COMMIT)
    if en[A_COMMIT_INF]:
        acts.append(A_COMMIT_INF)
    if en[A_ABSTAIN]:
        acts.append(A_ABSTAIN)
    return tuple(acts)


def advance(cfg: Config, state: tuple, a: int, o: int, event: bool, reveal: int | None) -> tuple:
    """Public transition: (state, visible step record) -> next state.  Uses no hidden information."""
    (i, built, nH, nN), (b, usage, reduced, cand, cand_valid, ev) = state
    if a in TERMINAL:
        if cfg.corr > 0.0:
            nH, nN = (nH + 1, nN) if reveal == TH else (nH, nN + 1)
        return ((i + 1, built, nH, nN), local0(cfg, nH, nN))
    if a != A_BUILD:
        b = update_belief(cfg, b, a, o, reduced)
        usage |= ACTION_BIT[a]
    else:
        built = True
    if a in (A_B1, A_B2, A_USE) and o == O_SOLVED:
        cand, cand_valid = C_EXACT, True
    elif a == A_PROBE and o == O_SOLVED and not (cand == C_EXACT and cand_valid):
        cand, cand_valid = C_PROBE, True
    if a == A_PROP and o == O_REDUCED:
        reduced = True
    if event:  # declared invalidation subset
        cand_valid = False
        reduced = False
        usage &= ~EPOCH_BITS
        ev = True
    return ((i, built, nH, nN), (b, usage, reduced, cand, cand_valid, ev))


def public_state_from_history(cfg: Config, history) -> tuple:
    """Information state from the visible history: a sequence of (action, outcome, event, reveal)."""
    s = initial_state(cfg)
    for a, o, event, reveal in history:
        s = advance(cfg, s, a, o, event, reveal)
    return s


def belief(cfg: Config, state: tuple) -> tuple:
    return state[1][0]


# ---------------------------------------------------------------------------------------------------------
# exact dynamic programming

class ExactSolver:
    """Memoized exact DP over the finite information-state space of one Config."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._V: dict = {}
        self._Q: dict = {}

    def belief(self, state):
        return state[1][0]

    def transitions(self, state, a):
        """List of (prob, outcome, event, reveal, next_state, reward) for action a at state (belief-weighted).
        reward = expected immediate utility contribution of that branch (R/-L for terminals, -cost otherwise)."""
        cfg = self.cfg
        b = self.belief(state)
        (i, built, nH, nN), local = state
        reduced, cand, cand_valid, ev = local[2], local[3], local[4], local[5]
        out = []
        if a in TERMINAL:
            for t in range(4):
                if b[t] <= 0:
                    continue
                if a == A_ABSTAIN:
                    o, r = O_ABSTAINED, 0.0
                else:
                    ok = commit_correct(a, t, cand, cand_valid)
                    o, r = (O_CORRECT, cfg.R) if ok else (O_WRONG, -cfg.L)
                out.append((b[t], o, False, t, advance(cfg, state, a, o, False, t), r))
            return out
        cost = action_cost(cfg, a)
        agg: dict = {}
        for t in range(4):
            if b[t] <= 0:
                continue
            for p, o in outcome_dist(cfg, a, t, reduced):
                if p > 0:
                    agg[o] = agg.get(o, 0.0) + b[t] * p
        pe = cfg.p_event if (cfg.p_event > 0 and not ev) else 0.0
        side = cfg.D_side * (1.0 - b[TH]) if (a == A_PROBE and cfg.D_side > 0) else 0.0
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

    def value(self, state) -> float:
        v = self._V.get(state)
        if v is None:
            if state[0][0] >= self.cfg.k:
                v = 0.0
            else:
                v = max(self.q_values(state).values())
            self._V[state] = v
        return v

    def opt_set(self, state, eps: float = EPS) -> frozenset:
        q = self.q_values(state)
        if not q:
            return frozenset()
        v = max(q.values())
        return frozenset(a for a, x in q.items() if x >= v - eps)

    def exact_opt_set(self, state) -> frozenset:
        return self.opt_set(state, eps=1e-9)

    def labels(self, state) -> dict:
        q = self.q_values(state)
        return {"V": self.value(state), "Q": dict(q), "opt_set": self.opt_set(state), "eps": EPS,
                "continuation": CONTINUATION, "version": VERSION}

    def n_states(self) -> int:
        return len(self._V)

    def pi_star(self, state) -> int:
        """Deterministic representative of the exact-optimal set (lowest action index)."""
        return min(self.exact_opt_set(state))

    def pi_star_stats(self, state=None) -> dict:
        """Exact expectations under the deterministic pi* representative from `state` (default: the root):
        P(builds the structure), E[#probe], E[#exact], E[#inspect+prop], E[total cost incl. error loss]."""
        memo: dict = {}
        cfg = self.cfg

        def rec(st):
            r = memo.get(st)
            if r is not None:
                return r
            if st[0][0] >= cfg.k:
                r = (0.0, 0.0, 0.0, 0.0, 0.0)
            elif st[0][1]:  # already built
                r = (1.0,) + _rest(st)
            else:
                r = _rest(st, with_build=True)
            memo[st] = r
            return r

        def _rest(st, with_build=False):
            a = self.pi_star(st)
            acc = [0.0] * 5
            for p, o, e, rev, ns, rew in self.transitions(st, a):
                sub = rec(ns)
                acc[0] += p * sub[0]
                acc[1] += p * sub[1]
                acc[2] += p * sub[2]
                acc[3] += p * sub[3]
                acc[4] += p * (sub[4] + (-rew if a not in TERMINAL else (cfg.L if o == O_WRONG else 0.0)))
            acc[1] += a == A_PROBE
            acc[2] += a in (A_B1, A_B2)
            acc[3] += a in (A_INSPECT, A_PROP)
            if a == A_BUILD:
                acc[0] = 1.0
            return tuple(acc) if with_build else tuple(acc[1:])

        st = initial_state(cfg) if state is None else state
        p_build, n_probe, n_exact, n_info, cost = rec(st)
        return {"p_build": p_build, "E_probe": n_probe, "E_exact": n_exact, "E_info": n_info, "E_cost": cost}

    def plan_set(self, state, a) -> frozenset:
        """Planned next eps-optimal actions after a: A* at the successor under the most probable
        non-event outcome of a (ties broken by outcome index)."""
        best = None
        for p, o, event, reveal, ns, _ in self.transitions(state, a):
            if event:
                continue
            key = (-p, o, -1 if reveal is None else reveal)
            if best is None or key < best[0]:
                best = (key, ns)
        if best is None or best[1][0][0] >= self.cfg.k:
            return frozenset()
        return self.opt_set(best[1])


def action_cost(cfg: Config, a: int) -> float:
    return {A_PROBE: cfg.c_probe, A_B1: cfg.c_b1, A_B2: cfg.c_b2, A_INSPECT: cfg.c_inspect,
            A_PROP: cfg.c_prop, A_BUILD: cfg.C_build, A_USE: cfg.c_use}.get(a, 0.0)


def no_structure_cost(cfg: Config) -> float:
    """C_shortcut_eff: expected per-query total cost (incl. error loss) of pi* with the structure disabled
    (one query, same prices).  Reported next to the declared rho_k (which uses C_shortcut = c_b2)."""
    en = list(cfg.enable)
    en[A_BUILD] = en[A_USE] = False
    c1 = replace(cfg, k=1, corr=0.0, enable=tuple(en))
    return ExactSolver(c1).pi_star_stats()["E_cost"]


def effective_rho(cfg: Config, k: int | None = None) -> float:
    k = cfg.k if k is None else k
    return no_structure_cost(cfg) / (cfg.C_build / k + cfg.c_use)


def cost_lines(cfg: Config, a: int, theta: int) -> list:
    """Ledger lines (item, amount) charged when action a is taken on hidden theta."""
    if a == A_USE:
        return [("C_execute", cfg.C_execute), ("C_return", cfg.C_return), ("C_verify", cfg.C_verify)]
    lines = []
    c = action_cost(cfg, a)
    if c:
        lines.append((ACTIONS[a], c))
    if a == A_PROBE and cfg.D_side > 0 and theta != TH:
        lines.append(("probe_side_effect", cfg.D_side))
    return lines


CASE_NAMES = {"a": "unjustified", "b": "rational_failed", "c": "invalidated", "d": "direct"}


def case_type(solver: ExactSolver, state, a, o, event, next_state, eps: float = EPS) -> str:
    """Case type of step (I_t, a_t, I_{t+1}); registered precedence a > b > c > d.
    (a) Q*(I_t, a_t) < V*(I_t) - eps
    (b) a_t eps-optimal and its outcome is its own failure (probe failed, b1 timeout, wrong commit)
    (c) a_t eps-optimal, no own failure, and a requirement-change event fired, or the new evidence
        leaves none of the planned next actions (A* at the modal non-event successor) eps-optimal
    (d) otherwise (direct success: continue or terminate)."""
    q = solver.q_values(state)
    if q[a] < solver.value(state) - eps:
        return "a"
    if (a, o) in FAILURE_OUTCOMES:
        return "b"
    if event:
        return "c"
    if next_state[0][0] < solver.cfg.k:
        plan = solver.plan_set(state, a)
        if plan and not (plan & solver.opt_set(next_state, eps)):
            return "c"
    return "d"


DEP_BITS = ("candidate_valid", "candidate_exact", "reduction_valid", "structure_built", "event_this_query")


def dependency_bits(state) -> tuple:
    """Validity/dependency status of computed results (a function of the visible history; L2 target)."""
    (i, built, _, _), (b, usage, reduced, cand, cand_valid, ev) = state
    return (float(cand != C_NONE and cand_valid), float(cand == C_EXACT and cand_valid), float(reduced),
            float(built), float(ev))


def switch_needed(solver: ExactSolver, prev_state, prev_a, state, eps: float = EPS) -> bool:
    """L3 target: none of the actions planned before the previous step remains eps-optimal now (same query)."""
    if prev_state is None or prev_a in TERMINAL or state[0][0] >= solver.cfg.k:
        return False
    plan = solver.plan_set(prev_state, prev_a)
    return bool(plan) and not (plan & solver.opt_set(state, eps))


def stage_set(solver: ExactSolver, state) -> frozenset:
    return frozenset(ACTION_STAGE[a] for a in solver.opt_set(state))


# ---------------------------------------------------------------------------------------------------------
# simulator

class Episode:
    """One episode on hidden worlds.  The hidden theta/z are used ONLY to simulate outcomes and charges;
    the public state is advanced from the visible step record via ``advance``."""

    def __init__(self, cfg: Config, seed: int):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.z = None
        if cfg.corr > 0:
            self.z = 0 if self.rng.random() < 0.5 else 1
        self.thetas = [self._draw_theta() for _ in range(cfg.k)]
        self.state = initial_state(cfg)
        self.history: list = []  # visible: (a, o, event, reveal)
        self.ledger: list = []  # (step, item, amount) ; amounts are costs (positive)
        self.successes = 0
        self.wrong = 0
        self.done = cfg.k == 0

    def _draw_theta(self) -> int:
        p = self.cfg.prior
        if self.z is not None:
            sgn = 1.0 if self.z == 0 else -1.0
            ph = min(max(p[TH] + sgn * self.cfg.corr, 0.0), 1.0)
            rest = 1.0 - p[TH]
            p = [ph] + [(1.0 - ph) * p[t] / rest for t in (TM, TF, TX)]
        u, acc = self.rng.random(), 0.0
        for t in range(4):
            acc += p[t]
            if u < acc:
                return t
        return max(t for t in range(4) if p[t] > 0)

    @property
    def query(self) -> int:
        return self.state[0][0]

    def available(self) -> tuple:
        return available(self.cfg, self.state)

    def step(self, a: int):
        if self.done or a not in self.available():
            raise ValueError(f"action {ACTIONS[a]} not available")
        cfg = self.cfg
        theta = self.thetas[self.query]
        t = len(self.history)
        local = self.state[1]
        reveal = None
        event = False
        for item, amt in cost_lines(cfg, a, theta):
            self.ledger.append((t, item, amt))
        if a in TERMINAL:
            reveal = theta
            if a == A_ABSTAIN:
                o = O_ABSTAINED
            elif commit_correct(a, theta, local[3], local[4]):
                o = O_CORRECT
                self.successes += 1
            else:
                o = O_WRONG
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

    @property
    def total_cost(self) -> float:
        return sum(x for _, _, x in self.ledger)

    @property
    def utility(self) -> float:
        return self.cfg.R * self.successes - self.total_cost


def run_policy(cfg: Config, seed: int, policy) -> Episode:
    """policy(state, available) -> action."""
    ep = Episode(cfg, seed)
    while not ep.done:
        ep.step(policy(ep.state, ep.available()))
    return ep


# ---------------------------------------------------------------------------------------------------------
# generator and splits (by generator parameters only; never by case type)

K_VALUES = (1, 2, 4, 8)
TRAIN_K = (1, 2, 8)
HELDOUT_K = (4,)
# price regions: 3x3 grid over x = cheapness of the heuristic (c_probe / c_b1) and y = amortization demand
# (C_build / c_b2); the centre cell is held out.
X_BINS = ((0.01, 0.05), (0.05, 0.2), (0.2, 0.8))
Y_BINS = ((0.3, 1.0), (1.0, 3.0), (3.0, 10.0))
HELDOUT_CELLS = ((1, 1),)
TRAIN_CELLS = tuple((x, y) for x in range(3) for y in range(3) if (x, y) not in HELDOUT_CELLS)
ALL_FLAG_COMBOS = tuple(tuple(bool(n >> j & 1) for j in range(4)) for n in range(16))
HELDOUT_COMBOS = ((True, False, False, True),   # unreliable + events
                  (False, True, True, False))   # side_effect + correlated
TRAIN_COMBOS = tuple(c for c in ALL_FLAG_COMBOS if sum(c) <= 2 and c not in HELDOUT_COMBOS)
SPLITS = {
    # name: (cells, ks, combos, seed_base)
    "train": (TRAIN_CELLS, TRAIN_K, TRAIN_COMBOS, 4_100_000_000),
    "dev": (TRAIN_CELLS, TRAIN_K, TRAIN_COMBOS, 4_200_000_000),
    "test_iid": (TRAIN_CELLS, TRAIN_K, TRAIN_COMBOS, 4_300_000_000),
    "heldout_price": (HELDOUT_CELLS, TRAIN_K, TRAIN_COMBOS, 4_400_000_000),
    "heldout_k": (TRAIN_CELLS, HELDOUT_K, TRAIN_COMBOS, 4_500_000_000),
    "heldout_comp": (TRAIN_CELLS, TRAIN_K, HELDOUT_COMBOS, 4_600_000_000),
}
# worlds (hidden theta / outcome draws) use a separate offset so config seeds and world seeds never collide
WORLD_SEED_OFFSET = 50_000_000


def _loguniform(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def make_config(cell: tuple, k: int, combo: tuple, rng: random.Random) -> Config:
    features = rng.choice(sorted(FEATURE_PRIORS))
    c_b1 = _loguniform(rng, 8.0, 50.0)
    c_b2 = c_b1 * rng.uniform(1.5, 2.5)
    xr, yr = X_BINS[cell[0]], Y_BINS[cell[1]]
    c_probe = c_b1 * _loguniform(rng, *xr)
    C_build = c_b2 * _loguniform(rng, *yr)
    c_use = c_b2 * rng.uniform(0.1, 0.3)
    parts = [rng.uniform(0.2, 1.0) for _ in range(3)]
    s = sum(parts)
    unreliable, side, corr, events = combo
    return Config(
        prior=FEATURE_PRIORS[features], features=features, R=100.0, L=round(rng.uniform(50.0, 200.0), 3),
        c_probe=round(c_probe, 4), c_b1=round(c_b1, 4), c_b2=round(c_b2, 4),
        c_inspect=round(_loguniform(rng, 0.5, 8.0), 4), c_prop=round(_loguniform(rng, 1.0, 12.0), 4),
        C_build=round(C_build, 4), C_execute=round(c_use * parts[0] / s, 4),
        C_return=round(c_use * parts[1] / s, 4), C_verify=round(c_use * parts[2] / s, 4), k=k,
        q=round(rng.uniform(0.3, 0.8), 3) if unreliable else 1.0,
        D_side=round(rng.uniform(5.0, 40.0), 3) if side else 0.0,
        corr=round(rng.uniform(0.1, 0.25), 3) if corr else 0.0,
        p_event=round(rng.uniform(0.1, 0.3), 3) if events else 0.0,
        eta=round(rng.uniform(0.1, 0.3), 3), p_conflict=round(rng.uniform(0.5, 0.9), 3))


def generator_params(split: str, index: int) -> tuple:
    """(cell, k, combo, config_seed) for config `index` of a split: uniform over the split's parameter set."""
    cells, ks, combos, base = SPLITS[split]
    seed = base + index
    rng = random.Random(seed)
    return rng.choice(cells), rng.choice(ks), rng.choice(combos), seed


def split_config(split: str, index: int) -> Config:
    cell, k, combo, seed = generator_params(split, index)
    return make_config(cell, k, combo, random.Random(seed * 7 + 1))


def world_seed(split: str, index: int, rep: int) -> int:
    return SPLITS[split][3] + WORLD_SEED_OFFSET + index * 1000 + rep


def price_cell(cfg: Config):
    """Recover the price-region cell from public prices (for split audits)."""
    x = cfg.c_probe / cfg.c_b1
    y = cfg.C_build / cfg.c_b2
    cx = next((j for j, (lo, hi) in enumerate(X_BINS) if lo * 0.999 <= x <= hi * 1.001), None)
    cy = next((j for j, (lo, hi) in enumerate(Y_BINS) if lo * 0.999 <= y <= hi * 1.001), None)
    return cx, cy


def split_of_params(cell, k, combo) -> set:
    """All split *families* whose generator-parameter set contains (cell, k, combo) (train/dev/test share one)."""
    fams = set()
    for name, (cells, ks, combos, _) in SPLITS.items():
        if tuple(cell) in cells and k in ks and tuple(combo) in combos:
            fams.add("iid" if name in ("train", "dev", "test_iid") else name)
    return fams


# ---------------------------------------------------------------------------------------------------------
# worked example (task brief): heuristic cost 1, 80% success, exact fallback 100, reliable detection

def worked_example_config(**over) -> Config:
    enable = [False] * N_ACTIONS
    for a in (A_PROBE, A_B1, A_COMMIT, A_ABSTAIN):
        enable[a] = True
    base = dict(prior=(0.8, 0.2, 0.0, 0.0), R=200.0, L=200.0, c_probe=1.0, c_b1=100.0, k=1,
                enable=tuple(enable))
    base.update(over)
    return Config(**base)


def worked_example_table() -> list:
    """Q*(probe-first) vs Q*(exact directly) under the base example and assumption violations."""
    rows = []
    variants = [
        ("base (reliable detection)", {}),
        ("unreliable detection q=0.5", {"q": 0.5}),
        ("unreliable detection q=0.5, L=1000", {"q": 0.5, "L": 1000.0}),
        ("probe side effect D=50", {"D_side": 50.0}),
        ("probe side effect D=500", {"D_side": 500.0}),
        ("heuristic success 30%", {"prior": (0.3, 0.7, 0.0, 0.0)}),
        ("correlated failure (corr=.15), k=4", {"corr": 0.15, "k": 4}),
        ("events p=.3 (candidate invalidated)", {"p_event": 0.3}),
    ]
    for name, over in variants:
        cfg = worked_example_config(**over)
        s = ExactSolver(cfg)
        s0 = initial_state(cfg)
        q = s.q_values(s0)
        # cost-only view of the two strategies (k=1): expected cost of probe-then-exact vs exact
        rows.append({"variant": name, "Q_probe_first": q[A_PROBE], "Q_exact_direct": q[A_B1],
                     "V_star": s.value(s0), "opt_set": sorted(ACTIONS[a] for a in s.opt_set(s0)),
                     "n_states": s.n_states()})
    return rows


def config_dict(cfg: Config) -> dict:
    d = asdict(cfg)
    d["flags"] = dict(zip(FLAG_NAMES, cfg.flags))
    d["rho_k"] = cfg.rho()
    return d


def with_(cfg: Config, **kw) -> Config:
    return replace(cfg, **kw)


# ---------------------------------------------------------------------------------------------------------
# exact-label reports (pure Python)

RHO_SWEEP_BASE = dict(prior=FEATURE_PRIORS[(1, 1)], c_probe=3.0, c_b1=20.0, c_b2=45.0, C_execute=3.0,
                      C_return=1.0, C_verify=1.0, L=150.0)


def rho_sweep_table(C_builds=(10.0, 20.0, 40.0, 60.0, 120.0, 240.0), ks=K_VALUES, **over) -> list:
    """pi* strategy vs rho_k and k at fixed other prices: P(build), E[#probe], E[#exact], cost/query."""
    base = Config(**{**RHO_SWEEP_BASE, **over})
    c_eff = no_structure_cost(base)
    rows = []
    for k in ks:
        for C in C_builds:
            cfg = replace(base, k=k, C_build=C)
            s = ExactSolver(cfg)
            st = s.pi_star_stats()
            rows.append({"k": k, "C_build": C, "rho_k": cfg.rho(), "rho_eff": c_eff / (C / k + cfg.c_use),
                         "p_build": st["p_build"], "E_probe_per_q": st["E_probe"] / k,
                         "E_exact_per_q": st["E_exact"] / k, "cost_per_q": st["E_cost"] / k,
                         "V_per_q": s.value(initial_state(cfg)) / k})
    return rows


def split_label_stats(split: str, n_configs: int = 40, worlds: int = 10) -> dict:
    """Case-type and action mix of pi* on a split's configs; P(build) and direct-success share."""
    cases = {c: 0 for c in "abcd"}
    acts = {a: 0 for a in ACTIONS}
    p_build = []
    n_ep = n_direct = 0
    n_states = 0
    for idx in range(n_configs):
        cfg = split_config(split, idx)
        s = ExactSolver(cfg)
        s.value(initial_state(cfg))
        n_states += s.n_states()
        p_build.append(s.pi_star_stats()["p_build"])
        for w in range(worlds):
            ep = Episode(cfg, world_seed(split, idx, w))
            cs = []
            while not ep.done:
                st = ep.state
                a = s.pi_star(st)
                r = ep.step(a)
                acts[ACTIONS[a]] += 1
                cs.append(case_type(s, st, a, r[1], r[2], ep.state))
            for c in cs:
                cases[c] += 1
            n_ep += 1
            n_direct += all(c == "d" for c in cs)
    return {"split": split, "configs": n_configs, "episodes": n_ep, "direct_success_episodes": n_direct / n_ep,
            "cases_pi_star": cases, "actions_pi_star": acts, "mean_p_build": sum(p_build) / len(p_build),
            "frac_configs_build": sum(p > 0 for p in p_build) / len(p_build), "mean_states": n_states / n_configs}
