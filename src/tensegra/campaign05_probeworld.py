"""Probeworld extensions for extended-05 Track B (B-SPLIT, B-X supplied state, B-LOC / B-X decision records).

Pure Python (no torch, no numpy).  Purely additive over ``campaign04_probeworld`` (imported as ``pw``): the
environment, exact DP, flag semantics, ``make_config`` and every extended-04 split are reused unchanged.  Design:
research/campaigns/extended-05/design.md section 2; tooling note: research/campaigns/extended-05/trackb-tooling.md.

B-SPLIT (registry entry B-SPLIT)
--------------------------------
Flags U (unreliable detection, q < 1), S (irreversible probe side effect, D_side > 0), C (correlated heuristic
failure, corr > 0), E (requirement-change events, p_event > 0).  ``make_config`` samples each flag's parameter
independently of the others, so all 6 pairs are generable; none is degenerate (``composition_interaction`` shows
that the optimal first decision of a pair differs from both single-flag ablations on a non-trivial fraction).

Split table ``probeworld-split-v2`` (design v2 revision 10): every split has a NEW configuration seed base; the
extended-04 table (``pw.SPLITS``, seeds 4.1e9-4.6e9) is untouched and still reproduces B1/B2/F2.

* new confirmation holds (never in training or selection): U+C, S+E  -> splits b5_hold_uc, b5_hold_se.  Pool
  size is fixed by the exact solver before any model run (revision 11): the smallest prefix of the hold's
  configuration stream containing HOLD_TARGET_ELIGIBLE configurations whose first action is uniquely probe
  (``hold_pool_size``); evaluated with 1 world per configuration (the s0 endpoint is configuration-level).
  U+C is also reported on its registered flag-sensitive subset (revision 12: |V*(U+C) - V*(U only)| > 1e-6).
* historical challenge pairs (inspected data): U+E, S+C             -> extended-04 split ``heldout_comp`` (same
  128 configurations B1/F2 were evaluated on; per-pair metrics are reported separately)
* training support (B0, BX2, BO): none, the 4 singles, U+S, C+E     -> b5_train / b5_dev / b5_test_iid
* exposure support (BX1 only): the above + the historical pairs U+E, S+C, with twice as many training
  configurations (more price and k draws within the training cells and k in {1, 2, 8})  -> b5x_train
* transfer axes kept separate: fresh instances (b5_test_iid), new continuous prices (b5_heldout_price, centre
  cell), the unseen intermediate horizon (b5_heldout_k, k = 4), omitted combinations (b5_hold_uc, b5_hold_se).
Configuration seeds 5.1e9-5.8e9 + i and world seeds base + 5e7 + 1000 i + r are disjoint from every extended-04
range (4.1e9-4.65e9, training worlds 8e9+).

B-XC (registry entry B-XC, adaptive confirmation): ``SPLITS5C['b5c_hold_uc']`` = fresh U+C draws of the b5_hold_uc
family, configuration seed base 5.9e9, same sizing rule (>= 60 eligible, >= 20 of them sensitive to the correlated
flag), 1 world per configuration.  Deliberately outside SPLITS5 so split table v2 and its audits are unchanged.
"""
from __future__ import annotations

import itertools
import math
import random

from tensegra import campaign04_probeworld as pw

VERSION = "probeworld-e05-trackb-v1"
SPLIT_TABLE_VERSION = "probeworld-split-v2"  # design v2 revision 10; the extended-04 table (pw.SPLITS) is unchanged

# flag combos (unreliable, side_effect, correlated, events)
NONE = (False, False, False, False)
U, S, C, E = (tuple(j == i for j in range(4)) for i in range(4))


def combo(*flags) -> tuple:
    return tuple(any(f[j] for f in flags) for j in range(4))


SINGLES = (U, S, C, E)
ALL_PAIRS = tuple(combo(a, b) for a, b in itertools.combinations(SINGLES, 2))  # UtS, UC, UE, SC, SE, CE
NEW_HOLD_COMBOS = (combo(U, C), combo(S, E))
HIST_COMBOS = tuple(pw.HELDOUT_COMBOS)  # (U+E, S+C), as in extended-04
assert set(HIST_COMBOS) == {combo(U, E), combo(S, C)}
B5_TRAIN_COMBOS = (NONE,) + SINGLES + (combo(U, S), combo(C, E))
B5X_TRAIN_COMBOS = B5_TRAIN_COMBOS + HIST_COMBOS

SPLITS5 = {
    # name: (cells, ks, combos, seed_base)
    "b5_train": (pw.TRAIN_CELLS, pw.TRAIN_K, B5_TRAIN_COMBOS, 5_100_000_000),
    "b5_dev": (pw.TRAIN_CELLS, pw.TRAIN_K, B5_TRAIN_COMBOS, 5_200_000_000),
    "b5_test_iid": (pw.TRAIN_CELLS, pw.TRAIN_K, B5_TRAIN_COMBOS, 5_300_000_000),
    "b5_heldout_price": (pw.HELDOUT_CELLS, pw.TRAIN_K, B5_TRAIN_COMBOS, 5_400_000_000),
    "b5_heldout_k": (pw.TRAIN_CELLS, pw.HELDOUT_K, B5_TRAIN_COMBOS, 5_500_000_000),
    "b5_hold_uc": (pw.TRAIN_CELLS, pw.TRAIN_K, (combo(U, C),), 5_600_000_000),
    "b5_hold_se": (pw.TRAIN_CELLS, pw.TRAIN_K, (combo(S, E),), 5_700_000_000),
    "b5x_train": (pw.TRAIN_CELLS, pw.TRAIN_K, B5X_TRAIN_COMBOS, 5_800_000_000),
}
B5_TRAINING_SPLITS = ("b5_train", "b5_dev", "b5_test_iid", "b5x_train")  # training support / selection pools
NEW_HOLD_SPLITS = ("b5_hold_uc", "b5_hold_se")
CHALLENGE_SPLIT = "heldout_comp"  # extended-04 configurations (U+E, S+C)
# label pools of the b5 split set: (split, CLI count option); counts are CLI defaults in the trainer
B5_LABEL_SPLITS = (("b5_train", "n_train"), ("b5x_train", "n_train_x"), ("b5_dev", "n_eval"),
                   ("b5_test_iid", "n_eval"), ("b5_heldout_price", "n_eval"), ("b5_heldout_k", "n_eval"),
                   ("b5_hold_uc", "sized"), ("b5_hold_se", "sized"), (CHALLENGE_SPLIT, "n_eval"))
HOLD_TARGET_ELIGIBLE = 60  # s0 uniquely-probe-optimal configurations per new-hold pair (review F11a)
HOLD_MAX_CONFIGS = 4000
HOLD_SHARD = 128  # label shard size for hold pools (bounded memory: ~1.5 KB per DP state)
FLAG_SENSITIVE_TOL = 1e-6
SENSITIVITY_FLAGS = {"b5_hold_uc": (2,)}  # registered: U+C sensitivity to the (partly inert) correlated flag
HOLD_TARGET_SENSITIVE = {"b5_hold_uc": 20}  # >= 20 eligible configs also inside the flag-sensitive U+C subset
B5_EVAL_SPLITS = ("b5_dev", "b5_test_iid", "b5_heldout_price", "b5_heldout_k", "b5_hold_uc", "b5_hold_se",
                  CHALLENGE_SPLIT)
B5_WORLD_OFFSET = 900  # eval world offset for B-X (B1 used 500, F2 700 on heldout_comp)

# B-XC (registry entry B-XC, adaptive confirmation): FRESH U+C configuration draws of the b5_hold_uc family (same cells,
# k and combo; new configuration seed base 5.9e9, never generated before), sized by the same exact-solver rule and
# evaluated with 1 world per configuration at world offset 900.  Kept OUT of SPLITS5 on purpose: split table v2, its
# audits and every b5 output stay bit-identical; split_params / generator_params / split_config / world_seed consult
# SPLITS5C after SPLITS5.
SPLITS5C = {
    "b5c_hold_uc": (pw.TRAIN_CELLS, pw.TRAIN_K, (combo(U, C),), 5_900_000_000),
}
B5C_LABEL_SPLITS = (("b5c_hold_uc", "sized"),)
B5C_EVAL_SPLITS = ("b5c_hold_uc",)
B5C_SOURCE_SPLIT = {"b5c_hold_uc": "b5_hold_uc"}  # fresh draws of this split's generator family
SENSITIVITY_FLAGS["b5c_hold_uc"] = SENSITIVITY_FLAGS["b5_hold_uc"]  # (correlated,)
HOLD_TARGET_SENSITIVE["b5c_hold_uc"] = HOLD_TARGET_SENSITIVE["b5_hold_uc"]  # 20
# split sets of the trainer's --split-set option (b1 = the extended-04 EVAL_SPLITS, handled by the trainer itself)
SPLIT_SETS = {"b5": {"label_splits": B5_LABEL_SPLITS, "eval_splits": B5_EVAL_SPLITS},
              "b5c": {"label_splits": B5C_LABEL_SPLITS, "eval_splits": B5C_EVAL_SPLITS}}


def s0_record(solver: pw.ExactSolver) -> dict:
    """Configuration-level facts at the initial state (the s0 endpoint): eps-optimal set, probe uniquely optimal."""
    cfg = solver.cfg
    st = pw.initial_state(cfg)
    opt = solver.opt_set(st)
    return {"k": cfg.k, "V": solver.value(st), "opt": sorted(opt), "probe_unique": opt == {pw.A_PROBE},
            "probe_eps_opt": pw.A_PROBE in opt, "combo": combo_name(cfg.flags)}


def flag_sensitivity(cfg: pw.Config, V: float, flags=None) -> dict:
    """V* of the configuration with each (or the given) active flag switched off; flag_sensitive = the full
    combination changes the optimal value (|V* - V*_without_flag| > FLAG_SENSITIVE_TOL for every checked flag)."""
    js = [j for j, f in enumerate(cfg.flags) if f] if flags is None else list(flags)
    out = {}
    for j in js:
        c1 = ablate(cfg, j)
        out["V_without_" + pw.FLAG_NAMES[j]] = pw.ExactSolver(c1).value(pw.initial_state(c1))
    out["flag_sensitive"] = all(abs(V - out["V_without_" + pw.FLAG_NAMES[j]]) > FLAG_SENSITIVE_TOL for j in js)
    out["checked_flags"] = [pw.FLAG_NAMES[j] for j in js]
    return out


def hold_pool_size(records, target: int = HOLD_TARGET_ELIGIBLE, target_sensitive: int = 0):
    """Registered sizing rule: smallest n such that records[:n] holds `target` s0-uniquely-probe-optimal configs
    and (U+C) `target_sensitive` of them in the registered flag-sensitive subset."""
    c = cs = 0
    for n, r in enumerate(records, 1):
        c += r["probe_unique"]
        cs += bool(r["probe_unique"] and r.get("flag_sensitive"))
        if c >= target and cs >= target_sensitive:
            return n
    return None


def combo_name(c) -> str:
    return "+".join(n for n, f in zip(pw.FLAG_NAMES, c) if f) or "none"


def family(split: str) -> str:
    if split in ("b5_train", "b5_dev", "b5_test_iid"):
        return "b5_iid"
    return split


def split_params(split: str) -> tuple:
    if split in SPLITS5:
        return SPLITS5[split]
    if split in SPLITS5C:
        return SPLITS5C[split]
    return pw.SPLITS[split]


def generator_params(split: str, index: int) -> tuple:
    """(cell, k, combo, config_seed); for extended-04 split names this is exactly ``pw.generator_params``."""
    if split in pw.SPLITS:
        return pw.generator_params(split, index)
    cells, ks, combos, base = split_params(split)  # SPLITS5, or SPLITS5C (B-XC)
    seed = base + index
    rng = random.Random(seed)
    return rng.choice(cells), rng.choice(ks), rng.choice(combos), seed


def split_config(split: str, index: int) -> pw.Config:
    if split in pw.SPLITS:
        return pw.split_config(split, index)  # extended-04 path unchanged
    cell, k, cmb, seed = generator_params(split, index)
    return pw.make_config(cell, k, cmb, random.Random(seed * 7 + 1))


def world_seed(split: str, index: int, rep: int) -> int:
    return split_params(split)[3] + pw.WORLD_SEED_OFFSET + index * 1000 + rep


# ---------------------------------------------------------------------------------------------------------
# supplied public state (BX2 / BO inputs).  A deterministic function of (public config, public information state);
# the information state is itself a function of the visible history (pw.public_state_from_history).

SUPPLIED_KINDS = ("public", "belief", "bx2")
BX2_FEATURES = (
    "belief_H", "belief_M", "belief_F", "belief_X",
    # unreliable detection
    "miss_rate", "p_false_solved", "p_H_given_solved",
    # side effect
    "D_side", "expected_side_cost",
    # correlated heuristic failure
    "corr", "revealed_H_frac", "revealed_notH_frac",
    # events
    "event_hazard_active", "event_fired_this_query",
    # horizon and structure
    "remaining_queries", "is_last_query", "built", "amortized_build_remaining", "c_use", "c_b2",
    "log_rho_remaining",
)


def supplied_dim(kind: str) -> int:
    return {"public": 0, "belief": 4, "bx2": len(BX2_FEATURES)}[kind]


def supplied_features(cfg: pw.Config, state: tuple, kind: str) -> list:
    """Extra model inputs.  'belief': exact posterior over theta (4).  'bx2': belief + explicit per-flag features
    (detection reliability, event hazard, side-effect cost, correlation evidence, remaining horizon, build/use costs).
    Uses the public config and public information state only."""
    if kind == "public":
        return []
    (i, built, nH, nN), (b, usage, reduced, cand, cand_valid, ev) = state
    b = list(b)
    if kind == "belief":
        return b
    R = cfg.R
    bH = b[pw.TH]
    miss = 1.0 - cfg.q
    p_false = (1.0 - bH) * miss
    p_solved = bH + p_false
    rem = max(cfg.k - i, 1)
    amort = 0.0 if built else cfg.C_build / rem
    denom = amort + cfg.c_use
    feats = b + [
        miss, p_false, bH / p_solved if p_solved > 0 else 0.0,
        cfg.D_side / R, cfg.D_side * (1.0 - bH) / R,
        cfg.corr, nH / cfg.k, nN / cfg.k,
        0.0 if ev else cfg.p_event, float(ev),
        rem / 8.0, float(i == cfg.k - 1), float(built), min(amort / R, 10.0) / 10.0, cfg.c_use / R, cfg.c_b2 / R,
        math.log(max(cfg.c_b2 / denom, 1e-3)) / 3.0,
    ]
    assert len(feats) == len(BX2_FEATURES)
    return feats


# ---------------------------------------------------------------------------------------------------------
# decision records (B-LOC, B-X metrics): exact labels at a visited information state

def probe_margin(q: dict) -> float | None:
    """Q*(probe) - max_{a != probe} Q*(a) (price units); None if probe unavailable or it is the only action."""
    if pw.A_PROBE not in q or len(q) < 2:
        return None
    return q[pw.A_PROBE] - max(v for a, v in q.items() if a != pw.A_PROBE)


def decision_record(solver: pw.ExactSolver, state: tuple, a: int, eps: float = pw.EPS) -> dict:
    q = solver.q_values(state)
    v = max(q.values())
    opt = sorted(b for b, x in q.items() if x >= v - eps)
    return {"i": state[0][0], "a": a, "V": v, "delta": v - q[a], "opt": opt,
            "avail": sorted(q), "Q": [q[b] for b in sorted(q)], "probe_margin": probe_margin(q)}


def q_rank_pairs(avail, qstar, qhat, tol: float = pw.EPS):
    """(n_pairs, n_correct) over available-action pairs whose exact Q* differ by >= tol (price units): a pair is
    correct iff the model's Q-head orders it the same way (ties in Q-head count as incorrect)."""
    n = c = 0
    for x, y in itertools.combinations(range(len(avail)), 2):
        d = qstar[x] - qstar[y]
        if abs(d) < tol:
            continue
        n += 1
        dh = qhat[x] - qhat[y]
        c += (d > 0 and dh > 0) or (d < 0 and dh < 0)
    return n, c


# ---------------------------------------------------------------------------------------------------------
# split audit helpers (pure Python)

def family_params(splits) -> dict:
    fams = {}
    for name in splits:
        cells, ks, combos, _ = split_params(name)
        fams.setdefault(family(name), set()).update(itertools.product(cells, ks, combos))
    return fams


def split_of_params(cell, k, cmb) -> set:
    """All split-table-v2 families whose generator-parameter set contains (cell, k, combo); the BX1 exposure pool is
    reported as its own family."""
    fams = set()
    for name, (cells, ks, combos, _) in SPLITS5.items():
        if tuple(cell) in cells and k in ks and tuple(cmb) in combos:
            fams.add(family(name))
    return fams


def ablate(cfg: pw.Config, flag: int) -> pw.Config:
    """The same configuration with one flag switched off (its parameter set to the 'off' value)."""
    field, off = (("q", 1.0), ("D_side", 0.0), ("corr", 0.0), ("p_event", 0.0))[flag]
    return pw.with_(cfg, **{field: off})


def composition_interaction(cfg: pw.Config) -> dict:
    """Does the pair matter for the first decision?  Exact eps-optimal first-action sets of the pair configuration
    vs each single-flag ablation (same prices, prior, k).  'interaction' = the pair's A*(I0) differs from both
    ablations' (the decision is not inherited from either single condition)."""
    flags = [j for j, f in enumerate(cfg.flags) if f]
    s = pw.ExactSolver(cfg)
    s0 = pw.initial_state(cfg)
    opt = s.opt_set(s0)
    res = {"opt": sorted(opt), "probe_unique": opt == {pw.A_PROBE}, "ablations": {}, "naive_regret": {},
           "V_star": s.value(s0)}
    diff_all = True
    for j in flags:
        c1 = ablate(cfg, j)
        s1 = pw.ExactSolver(c1)
        o1 = s1.opt_set(pw.initial_state(c1))
        res["ablations"]["without_" + pw.FLAG_NAMES[j]] = sorted(o1)
        diff_all &= o1 != opt
        res["naive_regret"]["without_" + pw.FLAG_NAMES[j]] = s.value(s0) - evaluate_foreign_policy(s, s1)
    res["interaction"] = diff_all and len(flags) >= 2
    return res


def evaluate_foreign_policy(solver: pw.ExactSolver, foreign: pw.ExactSolver) -> float:
    """Exact expected utility, in `solver`'s world, of the deterministic pi* of another configuration (`foreign`,
    e.g. the same prices with one flag switched off), which tracks ITS OWN public state from the shared visible
    history.  Where the foreign belief update meets an outcome impossible under its model (e.g. a 'solved' probe
    on a type it has excluded), the foreign policy keeps its previous belief (documented fallback).  Used to
    measure how much a pair requires composing both conditions: regret of acting as if one flag were absent."""
    cfg, fcfg = solver.cfg, foreign.cfg
    memo: dict = {}

    def fadvance(fst, a, o, e, rev):
        try:
            return pw.advance(fcfg, fst, a, o, e, rev)
        except ValueError:  # impossible observation under the foreign model (non-terminal only): keep its belief
            b = fst[1][0]
            nxt = pw.advance(fcfg, (fst[0], (_UNIFORM,) + fst[1][1:]), a, o, e, rev)  # possible under uniform
            return (nxt[0], (b,) + nxt[1][1:])

    def rec(st, fst):
        key = (st, fst)
        v = memo.get(key)
        if v is not None:
            return v
        if st[0][0] >= cfg.k:
            v = 0.0
        else:
            av = pw.available(cfg, st)
            a = foreign.pi_star(fst)
            if a not in av:  # cannot happen (availability depends on bookkeeping only); guard anyway
                a = solver.pi_star(st)
            v = sum(p * (r + rec(ns, fadvance(fst, a, o, e, rev)))
                    for p, o, e, rev, ns, r in solver.transitions(st, a))
        memo[key] = v
        return v

    return rec(pw.initial_state(cfg), pw.initial_state(fcfg))


_UNIFORM = (0.25, 0.25, 0.25, 0.25)
