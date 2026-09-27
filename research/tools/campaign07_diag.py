"""Extended-07 Phase 1: read-only frozen-checkpoint diagnostic runner (design.md, Phase 1).

Every input (checkpoints, train_meta.json, DP label shards, counterfactual octets) is READ ONLY.  Outputs go only to
--out, under versioned names (diag-v1-<protocol>-<model>.jsonl.gz), created exclusively: an existing file is never
overwritten, and --out may not lie under a historical root, a model run dir or a labels dir.

Subcommands
  run      per-decision logs of frozen models on an evaluation pool and/or counterfactual octets
      --labels DIR --pool SPLIT      evaluation pool (sharded DP labels; e.g. b6c_hold_SCE in e06-tbc-labels, or the
                                     b6_hold_SCE development pool in e06-tb-labels)
      --model NAME=RUNDIR            (repeatable) frozen models; kind (LRN / RAWF / SUP / B0) is read from train_meta
      --protocols B A-pistar A-own   B: free-running greedy (each policy acts itself; reproduces eval_b6c exactly).
                                     A-pistar: the SAME fixed pi* histories (exact optimum, lowest-index tie-break, on
                                     the same eval worlds) replayed through every model.  A-own: every model's own
                                     free-run histories replayed through every model of the job.  Replay always
                                     rebuilds the consumer's own recurrent state from the public history (inputs are
                                     rebuilt exactly as in training/eval); states are never transplanted or reset.
      --cf FILE                      (repeatable) counterfactual-only path: every member x decision type (+ near-miss)
                                     of balanced octets (cf_SCE.json, cf_UCE.json); no pool needed
      --ivs IV ...                   immediate-decision interventions (LRN only): at every logged decision, the fusion
                                     input phi is replaced at the SAME reconstructed state and only that decision is
                                     re-read (the recurrent state does not depend on phi, so nothing else changes)
      --rollout IV ...               rollout interventions (LRN only): free-running with phi replaced at EVERY
                                     decision; separate protocol files 'R-<iv>'
      --support-ref NAME=FILE        support reference of an LRN model (subcommand support)
      --phis own exact zero mean     factor-channel EVALUATION CONDITIONS (P2-SCREEN addendum_2), for every model that
                                     reads a factor channel (PHI_KINDS: LRN, S0R1, S1R1, SEP, SUP, CONS-exact,
                                     CONS-pred; R0 arms / RAWF / B0 are skipped, never an error).  'own' = the model's
                                     own input (its prediction; SUP / CONS-exact: the supplied exact factors; CONS-pred:
                                     the predictor's output) = the default output files, unchanged.  exact / zero /
                                     mean replace the fusion input phi at EVERY decision of the whole episode
                                     (free-running protocol B) and at every counterfactual decision (cf), in separate
                                     files diag-v1-<protocol>-<model>@phi=<c>.jsonl.gz (records carry "phi": c).  The
                                     recurrent state never depends on phi, so only the actions (and hence, in B, the
                                     free-running history) change.  'mean' = population mean of the exact targets over
                                     every decision of --phi-mean-bank (P2 history bank; sha256 + values in the header)
  support  support reference of an LRN model's own predictions on training-pool states (free-running greedy and pi*
           histories on --n-configs configurations of the training pool b6_B0): per-coordinate min/max/mean/std,
           prediction-error RMS (the observed error scale), covariance for Mahalanobis distance, a kNN sample, and the
           reference's own distance quantiles.

Interventions (names; groups and dependencies from research/campaigns/extended-07/factor-contract.json (P0a) if
present, else the PROVISIONAL groups below; every file header says which)
  none                  predicted phi (identity; bit-identical to the unmodified model)
  pred                  phi := the prediction passed through the replacement path (identity check)
  exact                 all 23 coordinates replaced by the exact targets
  iso:<G>               group G replaced by exact values, the rest predicted
  dep:<G>               dependency-consistent: group G plus every coordinate computed from it (the contract's
                        replacement_closure) replaced by exact values
  up:<G>                group G plus every coordinate it is computed from (upstream inputs)
  keep:<G>              everything exact except group G (predicted): leave-one-group-predicted
  (duplicates by definition -- p_probe_resolves == belief_H -- are always replaced together; G = G1..G4 of the
  contract; the injection point is the auxiliary output, before the fusion layer shared by the policy and value heads)
  gauss:<s>             exact + s * sigma_c * N(0,1) per coordinate (sigma_c = the LRN's own error RMS on training
                        states; deterministic per decision); needs --support-ref
  gauss_pred:<s>        predicted + s * sigma_c * N(0,1)
  scale_err:<s>         exact + s * (pred - exact)  (structured: s = 0 exact, 1 predicted, 2 doubled error)
  mirror                exact - (pred - exact)      (structured: same error magnitude, opposite sign)
Exact values are never fed into RAWF's untrained (zero-fed) channel, nor into SUP/B0: a non-'none' intervention on a
non-LRN model raises.

Extended-07 Phase 2 checkpoints (additive; historical runs load through the identical constructor call):
  S x R arms (train --shape/--read): kinds S0R0 / S1R0 / S0R1 / S1R1; 'pred' is always logged (in R0 it is the
      disconnected probe's output); interventions only on the arms that READ the prediction (S0R1, S1R1), never on
      R0 (constant channel, like RAWF)
  consumers (train --phi-contract): the model spec must say which factors the consumer reads at evaluation:
      NAME=RUNDIR::exact              the exact factors (kind CONS-exact; 'pred' = the exact inputs, as SUP)
      NAME=RUNDIR::pred=PREDRUNDIR    the predictions of a factor predictor run alongside on the same public history
                                      (kind CONS-pred; state = [consumer h | predictor h]; interventions allowed)

Per-decision record (kind "decision"): protocol, model, history source, cfg id, world seed, public-history id (hash
of the visible record prefix), episode history id, decision index, query index, step in query, decision context,
available actions, exact factor targets (23), predicted factors (LRN: auxiliary head; SUP: the supplied inputs; RAWF /
B0: null), pre-fusion latent (float16 base64 of the trunk output z, 128-d), policy probabilities, greedy action,
history action and its probability (replay), exact Q* per available action, V*, eps-optimal set (eps = 0.5 as bscore),
action-gap regret of the greedy action, interventions (action, probabilities, gap, support distances).
Episode records (kind "episode"): free-run metrics (campaign04 episode_metrics) or replay statistics (log-probability
of the history under the model, minimum action probability, divergences, support flag).
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign04_probeworld_train as T  # noqa: E402  (inserts src/ on sys.path)
from tensegra import campaign04_probeworld as pw  # noqa: E402
from tensegra import campaign06_probeworld as pw6  # noqa: E402

VERSION = "diag-v1"
EPS = pw.EPS
REPO = Path(__file__).resolve().parents[2]
CONTRACT = REPO / "research" / "campaigns" / "extended-07" / "factor-contract.json"
READ_ONLY_ROOTS = [os.path.expanduser("~/structured-latent-dynamics-campaign06")] + \
    [p for p in os.environ.get("DIAG_READ_ONLY_ROOTS", "").split(":") if p]
FEATS = pw6.FACTOR_FEATURES
NF = len(FEATS)
FI = {f: j for j, f in enumerate(FEATS)}
# support-reference worlds: inside the model seed's training band (8e9 + 1e8 seed), disjoint from the sampled training
# worlds (< 256k), the protocol-B2 own-value band (+6e7) and every eval world block (< 7e9)
SUPPORT_WORLD_OFFSET = 70_000_000
HIST_P_LOW, HIST_P_UNSUPPORTED = 0.05, 1e-3

# ------------------------------------------------------------------------------------------------ factor groups

PROVISIONAL_GROUPS = {  # design.md P0a thematic groups (provisional until factor-contract.json is merged)
    "G1": FEATS[0:8],     # posterior / outcome probabilities
    "G2": FEATS[8:11],    # side / event costs
    "G3": FEATS[11:16],   # one-query strategy-cost estimates (myopic closed forms, not Q*)
    "G4": FEATS[16:23],   # amortized build quantities and bookkeeping
}
PROVISIONAL_DEPS = {  # coordinate -> direct inputs among the coordinates (from campaign06_probeworld.factor_features)
    "p_probe_resolves": ["belief_H"], "p_probe_false_solved": ["belief_H"], "p_H_given_solved": ["belief_H"],
    "p_b1_resolves": ["belief_H", "belief_M", "belief_F"],
    "exp_side_cost_rel": ["belief_H"], "exp_hard_cost_rel": ["belief_F", "belief_X"],
    "cost_exact_b2_rel": ["exp_hard_cost_rel", "event_hazard_active"],
    "cost_probe_first_rel": ["exp_side_cost_rel", "p_probe_resolves", "p_probe_false_solved", "cost_exact_b2_rel",
                             "event_hazard_active"],
    "cost_b1_first_rel": ["exp_hard_cost_rel", "p_b1_resolves", "cost_exact_b2_rel"],
    "cost_use_rel": ["remaining_queries_rel", "built"],
    "best_strategy_cost_rel": ["cost_exact_b2_rel", "cost_probe_first_rel", "cost_b1_first_rel", "cost_use_rel"],
    "build_value_rel": ["remaining_queries_rel", "built", "cost_exact_b2_rel", "cost_probe_first_rel",
                        "cost_b1_first_rel"],
    "candidate_exact": ["candidate_valid"], "candidate_trust": ["candidate_valid", "candidate_exact", "p_H_given_solved"],
}
PROB_COORDS = ("belief_H", "belief_M", "belief_F", "belief_X", "p_probe_resolves", "p_probe_false_solved",
               "p_H_given_solved", "p_b1_resolves", "event_hazard_active", "candidate_trust")


class Contract:
    """Factor groups and dependencies: research/campaigns/extended-07/factor-contract.json (P0a, merged) when present,
    else the PROVISIONAL definitions above.  groups: short name (G1..G4; the contract's long names are accepted too)
    -> coordinates; up: coordinate -> its upstream coordinate inputs (transitive); down: coordinate -> the coordinates
    that must be replaced with it for a dependency-consistent vector (the contract's replacement_closure, transitive);
    dups: coordinates identical by definition (p_probe_resolves == belief_H), always replaced together."""

    def __init__(self, path=CONTRACT):
        path = Path(path)
        if path.exists():
            d = json.loads(path.read_text())
            names = {c["index"]: c["name"] for c in d["coordinates"]}
            assert [names[j] for j in range(NF)] == list(FEATS), "factor-contract coordinate order != FACTOR_FEATURES"
            self.long = {}
            self.groups = {}
            for gname, idx in d["groups"].items():
                short = gname.split("_")[0]
                self.groups[short] = [names[j] for j in idx]
                self.long[gname] = short
            direct = {c["name"]: [names[int(x[1:])] for x in c["depends_on"] if x[:1] == "f" and x[1:].isdigit()]
                      for c in d["coordinates"]}
            down = {names[int(k)]: [names[j] for j in v] for k, v in d["replacement_closure"].items()}
            self.dups = {}
            for c in d["coordinates"]:
                if c.get("duplicate_of"):
                    self.dups.setdefault(c["name"], set()).add(c["duplicate_of"])
                    self.dups.setdefault(c["duplicate_of"], set()).add(c["name"])
            self.constant = sorted(d.get("b6_constants", {}))
            self.source = f"factor-contract.json {d.get('version')} sha256:{_sha_file(path)[:16]}"
        else:
            self.groups = {k: list(v) for k, v in PROVISIONAL_GROUPS.items()}
            self.long = {}
            direct = PROVISIONAL_DEPS
            inv = {}
            for c, ins in direct.items():
                for x in ins:
                    inv.setdefault(x, []).append(c)
            down = {c: inv.get(c, []) for c in FEATS}
            self.dups = {"belief_H": {"p_probe_resolves"}, "p_probe_resolves": {"belief_H"}}
            self.constant = []
            self.source = "provisional (campaign07_diag.py)"
        assert sorted(x for g in self.groups.values() for x in g) == sorted(FEATS)
        self.up = {c: _closure([c], direct) for c in FEATS}
        self.down = {c: _closure([c], down) for c in FEATS}

    def group(self, g):
        g = self.long.get(g, g)
        if g not in self.groups:
            raise ValueError(f"unknown factor group {g!r} (have {sorted(self.groups)})")
        return self.groups[g]

    def with_dups(self, coords):
        out = set(coords)
        for c in coords:
            out |= self.dups.get(c, set())
        return _order(out)

    def iso(self, g):
        return self.with_dups(self.group(g))

    def dep(self, g):
        """Dependency-consistent: the group and every coordinate computed from it (replacement_closure)."""
        return self.with_dups(set().union(*(self.down[c] for c in self.group(g))))

    def upstream(self, g):
        """The group and every coordinate it is computed from."""
        return self.with_dups(set().union(*(self.up[c] for c in self.group(g))))

    def keep(self, g):
        """Everything exact except the group (and its duplicates) kept predicted."""
        kept = set(self.with_dups(self.group(g)))
        return [c for c in FEATS if c not in kept]


def _order(cs):
    return sorted(set(cs), key=FI.get)


def _closure(coords, edges):
    out, todo = set(coords), list(coords)
    while todo:
        for d in edges.get(todo.pop(), []):
            if d not in out:
                out.add(d)
                todo.append(d)
    return _order(out)


def load_groups(path=CONTRACT):
    c = Contract(path)
    return c.groups, c.down, c.source


# ------------------------------------------------------------------------------------------------ io guards

def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_out_dir(out, forbidden=()):
    """--out must not be (or be inside) a historical root, a model run dir or a labels dir."""
    o = Path(out).resolve()
    for r in list(READ_ONLY_ROOTS) + [str(x) for x in forbidden]:
        if not r:
            continue
        rp = Path(r).expanduser().resolve()
        if o == rp or rp in o.parents:
            raise SystemExit(f"refusing: output dir {o} is inside read-only input {rp}")
    return o


def out_name(protocol, model):
    return f"{VERSION}-{protocol}-{model}.jsonl.gz"


class Writer:
    """gzip JSONL writer that never overwrites (exclusive create)."""

    def __init__(self, path, header):
        self.path = Path(path)
        self.f = gzip.open(self.path, "xt")  # FileExistsError if present: never overwrite
        self.n = 0
        self.f.write(json.dumps({"_meta": header}) + "\n")

    def write(self, rec):
        self.f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.n += 1

    def close(self):
        self.f.close()


def write_json_new(path, obj):
    txt = json.dumps(obj, indent=1)  # serialize first: no partial file on error
    with open(path, "x") as f:
        f.write(txt)


# ------------------------------------------------------------------------------------------------ models

def run_path(spec):
    """Run dir of a model spec (extended-07 consumers: 'RUNDIR::exact' or 'RUNDIR::pred=PREDICTOR_RUNDIR')."""
    return spec.split("::", 1)[0]


def load_model(run):
    spec = str(run)
    run = Path(run_path(spec))
    phi_spec = spec.split("::", 1)[1] if "::" in spec else None
    meta = json.loads((run / "train_meta.json").read_text())
    arch = meta.get("arch", "flat")
    assert arch in ("flat", "fuse"), f"unsupported arch {arch}"
    model = T.build_model_from_meta(meta)  # historical runs: the identical constructor call
    model.load_state_dict(torch.load(run / "model.pt"))
    model.eval()
    consumer = "consumer" in meta.get("p2", {})
    if phi_spec is not None and not consumer:
        raise SystemExit(f"{spec}: '::' input specs are for extended-07 consumers only")
    if consumer:
        if phi_spec is None:
            raise SystemExit(f"{spec}: a consumer needs '::exact' or '::pred=PREDICTOR_RUNDIR'")
        contract = meta["p2"]["consumer"]["contract"]
        if phi_spec == "exact":
            model._diag_kind = "CONS-exact"
        elif phi_spec.startswith("pred="):
            pred, pmeta = load_model(phi_spec[5:])
            assert getattr(pred, "factor_mode", None) in ("sr", "learned"), "the predictor needs an aux head"
            model = PredictedConsumer(model, pred)
            meta = dict(meta, predictor={"run": phi_spec[5:], "seed": pmeta["seed"],
                                         "model_sha256": _sha_file(Path(phi_spec[5:]) / "model.pt")})
        else:
            raise SystemExit(f"{spec}: unknown consumer input spec {phi_spec!r}")
        meta = dict(meta, consumer_contract=contract)
    return model, meta


class PredictedConsumer(torch.nn.Module):
    """extended-07 consumer evaluated on PREDICTED factors: a SUP-architecture consumer whose fusion input is the
    prediction of a separately trained factor predictor (aux head on its own recurrent encoder) run alongside on the
    same public history.  The recurrent state is [consumer h | predictor h]; the exact factor tail of the input
    (T.supplied(..., 'factors6')) is ignored except by interventions (targets)."""

    def __init__(self, consumer, predictor):
        super().__init__()
        self.cons, self.pred = consumer, predictor
        self.inputs, self.arch, self.factor_mode = "factors6", "fuse", "consumer_pred"
        self.base_dim = consumer.base_dim
        self.state_size = consumer.gru.hidden_size + predictor.gru.hidden_size
        self.fuse, self.pi, self.v, self.q = consumer.fuse, consumer.pi, consumer.v, consumer.q
        self._diag_kind = "CONS-pred"

    def diag_forward(self, x, h):
        Hc = self.cons.gru.hidden_size
        xb = x[:, :self.base_dim]
        p = self.pred
        hp = p.gru(torch.tanh(p.inp(xb[:, :p.base_dim])), h[:, Hc:])
        pred = p.aux(p.trunk(hp))
        c = self.cons
        hc = c.gru(torch.tanh(c.inp(xb)), h[:, :Hc])
        z = c.trunk(hc)
        return torch.cat([hc, hp], 1), z, pred, pred, fuse_out(self, z, pred)


def hidden_size(model):
    return getattr(model, "state_size", None) or model.gru.hidden_size


def model_kind(model):
    if hasattr(model, "_diag_kind"):
        return model._diag_kind
    if getattr(model, "arch", "flat") != "fuse":
        return "B0"
    if model.factor_mode == "sr":
        return f"S{int(model.sr_shape)}R{int(model.sr_read)}"
    if model.factor_mode == "sep":
        return "SEP"
    return {"learned": "LRN", "none": "RAWF", "supplied": "SUP"}[model.factor_mode]


# kinds whose policy READS a learned prediction: the only kinds that accept factor interventions (R0 arms read a
# constant through an untrained channel, like RAWF; SUP / CONS-exact read exact inputs)
READS_PREDICTION = ("LRN", "S0R1", "S1R1", "SEP", "CONS-pred")
# kinds whose policy reads a factor channel at all: the only kinds evaluated under factor-channel conditions (--phis);
# consumer-side INPUT conditions, so unlike interventions they include SUP-type consumers
PHI_KINDS = READS_PREDICTION + ("SUP", "CONS-exact")
PHI_CONDS = ("own", "exact", "zero", "mean")


def bank_phi_mean(path):
    """Population mean of the exact factor targets over every decision of a P2 history bank (campaign07_p2 bank):
    per coordinate math.fsum / n (deterministic)."""
    bank = T.load_bank(path)
    rows = [row for e in bank["episodes"] for row in e["phi"]]
    n = len(rows)
    assert n and all(len(r) == NF for r in rows)
    return {"bank": str(path), "sha256": _sha_file(path), "n_decisions": n, "n_episodes": len(bank["episodes"]),
            "values": [math.fsum(r[j] for r in rows) / n for j in range(NF)]}


def cond_phi(cond, targets, mean):
    """Replacement phi rows (list of lists) of a factor-channel condition for one batch of decisions."""
    if cond == "exact":
        return [list(t) for t in targets]
    if cond == "zero":
        return [[0.0] * NF for _ in targets]
    if cond == "mean":
        assert mean is not None and len(mean) == NF
        return [list(mean) for _ in targets]
    raise ValueError(f"unknown factor-channel condition {cond!r}")


def cond_name(name, cond):
    return f"{name}@phi={cond}"


def forward(model, x, h):
    """Exactly model.step (same ops, same order), exposing the pre-fusion trunk output z and phi.
    Returns (h_new, z_pre, pred, phi, z_fused); pred = auxiliary prediction (LRN) / supplied inputs (SUP) / None."""
    if hasattr(model, "diag_forward"):  # extended-07 consumer on predicted factors
        return model.diag_forward(x, h)
    if getattr(model, "factor_mode", None) == "sep":  # extended-07 separate-predictor READ arm
        return model.sep_forward(x, h)
    fuse = getattr(model, "arch", "flat") == "fuse"
    pre = model.inp(x[:, :model.base_dim] if fuse else x)
    hn = model.gru(torch.tanh(pre), h)
    z = model.trunk(hn)
    if not fuse:
        return hn, z, None, None, z
    if model.factor_mode == "supplied":
        phi = x[:, model.base_dim:]
        pred = phi
    elif model.factor_mode == "learned":
        pred = model.aux(z)
        phi = pred.detach()
    elif model.factor_mode == "sr":  # extended-07 S x R arms: pred is always logged (R0: a disconnected probe)
        pred = model.aux(z)
        phi = pred.detach() if model.sr_read else model.phi_const.expand(z.shape[0], -1)
    else:
        pred, phi = None, torch.zeros(z.shape[0], pw6.N_FACTOR_FEATURES)
    return hn, z, pred, phi, fuse_out(model, z, phi)


def fuse_out(model, z, phi):
    return torch.tanh(model.fuse(torch.cat([z, phi], 1)))


# ------------------------------------------------------------------------------------------------ interventions

class Intervener:
    """Builds replacement phi rows for named interventions on an LRN model (refuses every other kind)."""

    def __init__(self, names, kind, support=None, seed=0):
        self.names = list(names)
        if kind not in READS_PREDICTION and any(n != "none" for n in self.names):
            what = ("RAWF's fusion channel was never trained (it is fed zeros)" if kind == "RAWF" else
                    f"{kind} is not the learned-factor consumer")
            raise ValueError(f"refusing factor injection into {kind}: {what}; interventions are LRN-only")
        self.contract = Contract()
        self.group_source = self.contract.source
        self.support = support
        self.seed = seed
        for n in self.names:
            self.coords(n)  # validate names early
            if n.startswith("gauss") and support is None:
                raise ValueError(f"{n} needs --support-ref (sigma = the model's own training-state error RMS)")

    def coords(self, name):
        """Coordinates set to exact values (for structured / noise interventions: None)."""
        base = name.split(":")[0]
        if base in ("none", "pred"):
            return []
        if base == "exact":
            return list(FEATS)
        if base in ("iso", "dep", "up", "keep"):
            g = name.split(":", 1)[1]
            return {"iso": self.contract.iso, "dep": self.contract.dep, "up": self.contract.upstream,
                    "keep": self.contract.keep}[base](g)
        if base in ("gauss", "gauss_pred", "scale_err", "mirror"):
            if base != "mirror":
                float(name.split(":", 1)[1])
            return None
        raise ValueError(f"unknown intervention {name!r}")

    def phi(self, name, pred, target, key):
        """Replacement phi (list) for one decision.  key: stable decision id (seeds the noise)."""
        base = name.split(":")[0]
        if base == "none":
            return None
        if base == "pred":
            return list(pred)
        cs = self.coords(name)
        if cs is not None:
            ex = set(FI[c] for c in cs)
            return [target[j] if j in ex else pred[j] for j in range(NF)]
        if base == "mirror":
            return [2 * t - p for p, t in zip(pred, target)]
        s = float(name.split(":", 1)[1])
        if base == "scale_err":
            return [t + s * (p - t) for p, t in zip(pred, target)]
        rng = random.Random(f"{self.seed}|{name}|{key}")
        sig = self.support["err_rms"]
        src = target if base == "gauss" else pred
        return [x + s * sd * rng.gauss(0.0, 1.0) for x, sd in zip(src, sig)]


# ------------------------------------------------------------------------------------------------ support distances

class Support:
    def __init__(self, ref):
        import numpy as np
        self.np = np
        self.ref = ref
        self.lo, self.hi = np.array(ref["min"]), np.array(ref["max"])
        self.mean = np.array(ref["mean"])
        self.prec = np.array(ref["precision"])
        self.knn = np.array(ref["knn_sample"])
        self.err_rms = ref["err_rms"]

    def __getitem__(self, k):
        return self.ref[k]

    def dist(self, v):
        np = self.np
        v = np.asarray(v, dtype=float)
        d = v - self.mean
        out_lo, out_hi = v < self.lo - 1e-9, v > self.hi + 1e-9
        n_out = int((out_lo | out_hi).sum())
        maha = float(math.sqrt(max(d @ self.prec @ d, 0.0)))
        knn = float(np.sqrt(((self.knn - v) ** 2).sum(1)).min())
        return {"n_out": n_out, "maha": round(maha, 4), "knn": round(knn, 5),
                "in": n_out == 0 and maha <= self.ref["maha_ref"]["q99"] and knn <= self.ref["knn_ref_loo"]["q99"]}


# ------------------------------------------------------------------------------------------------ helpers

def hist_id(prefix):
    return hashlib.sha1(json.dumps([list(r) for r in prefix], separators=(",", ":")).encode()).hexdigest()[:16]


def context(prefix, st):
    """initial | query2_after_H | query2_after_notH | later_query_start | after_probe_failed | after_probe_solved |
    after_b1_timeout | other (bscore's contexts, with the first decision of a later query split by query index and
    query 1's revealed type)."""
    if not prefix:
        return "initial"
    a, o, _, rev = prefix[-1]
    if a in pw.TERMINAL:
        if st[0][0] == 1:
            return "query2_after_H" if rev == pw.TH else "query2_after_notH"
        return "later_query_start"
    if a == pw.A_PROBE and o == pw.O_FAILED:
        return "after_probe_failed"
    if a == pw.A_PROBE and o == pw.O_SOLVED:
        return "after_probe_solved"
    if a == pw.A_B1 and o == pw.O_TIMEOUT:
        return "after_b1_timeout"
    return "other"


def r6(xs):
    return [round(float(x), 6) for x in xs]


def latent_b64(row_f16):
    return base64.b64encode(row_f16.tobytes()).decode()


def decode_latent(s):
    import numpy as np
    return np.frombuffer(base64.b64decode(s), dtype=np.float16)


def qinfo(q):
    """q: {action: Q*} -> (V, eps-optimal set, sorted avail)."""
    v = max(q.values())
    return v, sorted(a for a, x in q.items() if x >= v - EPS)


class Track:
    """One episode: free-running (Episode + world seed) or forced along a given visible history."""

    def __init__(self, cfg, solver, cfg_id, world_seed=None, history=None, src=None):
        self.cfg, self.s, self.cfg_id, self.ws, self.src = cfg, solver, cfg_id, world_seed, src
        self.env = T._env(cfg)
        self.forced = history is not None
        self.history = [tuple(r) for r in history] if self.forced else None
        if self.forced:
            self.state, self.pos = pw.initial_state(cfg), 0
        else:
            self.ep = self.env.Episode(cfg, world_seed)
        self.prefix = []
        self.infos = []
        self.recs = []

    @property
    def st(self):
        return self.state if self.forced else self.ep.state

    @property
    def done(self):
        return self.pos >= len(self.history) if self.forced else self.ep.done

    def available(self):
        return self.env.available(self.cfg, self.state) if self.forced else self.ep.available()

    def step(self, a):
        st = self.st
        if self.forced:
            r = self.history[self.pos]
            assert r[0] in self.available(), (self.cfg_id, self.pos, r)
            self.state = self.env.advance(self.cfg, self.state, *r)
            self.pos += 1
            info = {"state": st, "a": r[0], "rec": r, "next": self.state}
        else:
            n0, s0 = len(self.ep.ledger), self.ep.successes
            r = self.ep.step(a)
            info = {"state": st, "a": a, "rec": r, "next": self.ep.state, "reward": T.pw_reward(self.ep, n0, s0)}
        self.infos.append(info)
        self.prefix.append(tuple(r))
        return r


@torch.no_grad()
def drive(model, tracks, meta_rec, ivn=None, rollout=None, support=None, latent=True, phi_cond=None, phi_mean=None):
    """Batched rollout mirroring campaign04 run_batch (same inputs, same batch composition and op order, so a free
    greedy run is bit-identical to the historical eval).  Appends per-decision records to track.recs.
    ivn: Intervener (immediate interventions; logged, never acted on).  rollout: intervention name acted on at
    every decision (free-running only).  phi_cond: factor-channel condition (exact / zero / mean) acted on at every
    decision (None / 'own': the model's own phi, bit-identical to the default path)."""
    if phi_cond == "own":
        phi_cond = None
    assert not (phi_cond and rollout), "a factor-channel condition and a rollout intervention are exclusive"
    kind = getattr(model, "inputs", "public")
    B = len(tracks)
    vecs = [t.cfg.public_vector() for t in tracks]
    h = torch.zeros(B, hidden_size(model))
    prev = [None] * B
    ivs = [n for n in (ivn.names if ivn else []) if n != "none"]
    while True:
        act = [i for i in range(B) if not tracks[i].done]
        if not act:
            break
        avails = [tracks[i].available() for i in act]
        sts = [tracks[i].st for i in act]
        x = torch.tensor([T.encode(vecs[i], prev[i], av, st[0][0] / tracks[i].cfg.k) + T.supplied(tracks[i].cfg, st, kind)
                          for i, av, st in zip(act, avails, sts)])
        idx = torch.tensor(act)
        hn, z, pred, phi, zf = forward(model, x, h[idx])
        h = h.index_copy(0, idx, hn)
        mask = torch.zeros(len(act), pw.N_ACTIONS, dtype=torch.bool)
        for j, av in enumerate(avails):
            mask[j, list(av)] = True
        targets = [pw6.factor_features(tracks[i].cfg, st) for i, st in zip(act, sts)]
        pred_l = pred.tolist() if pred is not None else None
        keys = [f"{tracks[i].cfg_id}|{hist_id(tracks[i].prefix)}" for i in act]
        if rollout is not None:
            rp = [ivn.phi(rollout, pred_l[j], targets[j], keys[j]) for j in range(len(act))]
            zf = fuse_out(model, z, torch.tensor(rp, dtype=z.dtype))
        if phi_cond is not None:
            zf = fuse_out(model, z, torch.tensor(cond_phi(phi_cond, targets, phi_mean), dtype=z.dtype))
        logits = model.pi(zf).masked_fill(~mask, -1e9)
        greedy = logits.argmax(-1).tolist()
        probs = torch.softmax(logits, -1).tolist()
        alt = {}
        for n in ivs:
            ph = [ivn.phi(n, pred_l[j], targets[j], keys[j]) for j in range(len(act))]
            lg = model.pi(fuse_out(model, z, torch.tensor(ph, dtype=z.dtype))).masked_fill(~mask, -1e9)
            alt[n] = (ph, lg.argmax(-1).tolist(), torch.softmax(lg, -1).tolist())
        zl = z.to(torch.float16).numpy() if latent else None
        for j, i in enumerate(act):
            t = tracks[i]
            st = sts[j]
            q = t.s.q_values(st)
            a = t.history[t.pos][0] if t.forced else greedy[j]
            rec = decision_record(t, st, q, avails[j], targets[j], pred_l[j] if pred_l else None, probs[j],
                                  greedy[j], a, keys[j], meta_rec)
            if zl is not None:
                rec["z_pre"] = latent_b64(zl[j])
            if support is not None and pred_l is not None:
                rec["support_pred"] = support.dist(pred_l[j])
            if rollout is not None:
                rec["iv_applied"] = {"name": rollout, "mode": "rollout"}
            if phi_cond is not None:
                rec["phi"] = phi_cond
            if alt:
                rec["iv"] = {n: iv_record(alt[n][0][j], alt[n][1][j], alt[n][2][j], q, support) for n in alt}
            t.recs.append(rec)
            r = t.step(a)
            prev[i] = r
    return tracks


def iv_record(phi, a, probs, q, support):
    v, opt = qinfo(q)
    out = {"a": a, "gap": round(v - q[a], 6), "ok": a in opt, "p_opt": round(sum(probs[b] for b in opt), 6),
           "probs": [round(p, 6) for p in probs]}
    if support is not None:
        out["support"] = support.dist(phi)
    return out


def decision_record(t, st, q, avail, target, pred, probs, a_greedy, a, key, meta_rec):
    v, opt = qinfo(q)
    rec = {"kind": "decision", **meta_rec, "src": t.src, "cfg_id": t.cfg_id, "combo": _combo(t.cfg), "k": t.cfg.k,
           "ws": t.ws, "t": len(t.prefix),
           "i": st[0][0], "siq": _step_in_query(t.prefix), "ctx": context(t.prefix, st), "hist_id": key.split("|")[1],
           "avail": sorted(avail), "target": r6(target), "pred": r6(pred) if pred is not None else None,
           "probs": [round(p, 6) for p in probs], "a": a_greedy, "V": round(v, 6),
           "Q": {str(b): round(q[b], 6) for b in sorted(q)}, "opt": opt, "gap": round(v - q[a_greedy], 6),
           "ok": a_greedy in opt, "margin": round(_margin(q), 6)}
    if t.forced:
        rec.update(a_hist=a, p_hist=round(probs[a], 6), gap_hist=round(v - q[a], 6))
    return rec


_COMBO: dict = {}


def _combo(cfg):
    c = _COMBO.get(cfg)
    if c is None:
        c = _COMBO[cfg] = pw6.fam_key(pw6.flags_of(cfg))
    return c


def _step_in_query(prefix):
    n = 0
    for r in reversed(prefix):
        if r[0] in pw.TERMINAL:
            break
        n += 1
    return n


def _margin(q):
    """Q* gap between the best and second-best available action (inf-free: a single action gives 1e9)."""
    xs = sorted(q.values(), reverse=True)
    return xs[0] - xs[1] if len(xs) > 1 else 1e9


def episode_record_forced(t, meta_rec):
    ps = [r["p_hist"] for r in t.recs]
    mn = min(ps) if ps else 1.0
    lp = sum(math.log(max(p, 1e-300)) for p in ps)
    flag = "unsupported" if mn < HIST_P_UNSUPPORTED else ("low" if mn < HIST_P_LOW else "supported")
    return {"kind": "episode", **meta_rec, "src": t.src, "cfg_id": t.cfg_id, "ws": t.ws,
            "ep_hist_id": hist_id(t.history), "n": len(t.recs), "logp_hist": round(lp, 6), "min_p_hist": round(mn, 8),
            "n_diverge": sum(r["a"] != r["a_hist"] for r in t.recs), "support_flag": flag,
            "acc": sum(r["ok"] for r in t.recs) / max(len(t.recs), 1),
            "gap_sum_greedy": round(sum(r["gap"] for r in t.recs), 6)}


# ------------------------------------------------------------------------------------------------ run: pool protocols

def shards(labels, pool, n_configs=None, only=None):
    """(part, [(idx, cfg, solver)], selected idx set) shard by shard; stops after the shard holding the n-th config.
    only: shard indices to process (subsampling by whole shards keeps the historical batch composition)."""
    seen = 0
    for j, part in enumerate(T.pool_parts(labels, pool)):
        if only is not None and j not in only:
            continue
        if n_configs is not None and seen >= n_configs:
            break
        rows = T.load_pool(labels, part)
        sel = {idx for idx, _, _ in rows[:max(0, n_configs - seen)]} if n_configs is not None else {r[0] for r in rows}
        seen += len(rows)
        yield part, rows, sel


def run_pool(a, models, writers, headers, ivns, supports, report):
    protos = a.protocols
    per_model_b = {name: [] for name in models}
    per_model_b.update({cond_name(n, c): [] for n, c in a.phi_runs} if "B" in protos else {})
    for part, rows, sel in shards(a.labels, a.pool, a.n_configs, set(a.shards) if a.shards else None):
        t0 = time.process_time()
        items = T.eval_items(rows, a.pool, a.worlds, a.world_offset)  # (cfg, solver, ws): the historical eval order
        ids = [f"{a.pool}:{idx}" for idx, _, _ in rows for _ in range(a.worlds)]
        keep = [idx in sel for idx, _, _ in rows for _ in range(a.worlds)]
        own = {}
        for name, (model, meta) in models.items():
            # B: free-running greedy over the FULL shard (historical batch composition); only selected configs logged
            tr = [Track(cfg, s, cid, ws, src="self") for (cfg, s, ws), cid in zip(items, ids)]
            drive(model, tr, {"protocol": "B", "model": name}, ivns.get(name), None, supports.get(name), a.latent)
            own[name] = [t.prefix for t in tr]
            rows_m = T.episode_metrics(items, [t.ep for t in tr], [t.infos for t in tr])
            for t, rm, k in zip(tr, rows_m, keep):
                if not k:
                    continue
                per_model_b[name].append(rm)
                if "B" in protos:
                    for r in t.recs:
                        writers[("B", name)].write(r)
                    writers[("B", name)].write({"kind": "episode", "protocol": "B", "model": name, "src": "self",
                                                "cfg_id": t.cfg_id, "ws": t.ws, "ep_hist_id": hist_id(t.prefix),
                                                "n": len(t.recs), **_json_row(rm)})
        if "A-pistar" in protos:
            ref_eps, _ = T.reference_policy_eval(items, "pi_star")
            if "pistar" not in report["hist_written"]:
                report["hist_written"].add("pistar")
            for ep, cid, (cfg, s, ws), k in zip(ref_eps, ids, items, keep):
                if k:
                    writers[("hist", "pistar")].write({"kind": "history", "src": "pistar", "cfg_id": cid, "ws": ws,
                                                       "ep_hist_id": hist_id(ep.history), "history": [list(r) for r in ep.history],
                                                       "U": ep.utility})
            for name, (model, meta) in models.items():
                tr = [Track(cfg, s, cid, ws, history=ep.history, src="pistar")
                      for (cfg, s, ws), cid, ep, k in zip(items, ids, ref_eps, keep) if k]
                emit_forced(model, tr, "A-pistar", name, writers, ivns.get(name), supports.get(name), a.latent)
        if "A-own" in protos:
            for src in models:
                for name, (model, meta) in models.items():
                    tr = [Track(cfg, s, cid, ws, history=hs, src=src)
                          for (cfg, s, ws), cid, hs, k in zip(items, ids, own[src], keep) if k]
                    emit_forced(model, tr, f"A-own_{src}", name, writers, ivns.get(name), supports.get(name), a.latent)
        for name, cond in (a.phi_runs if "B" in protos else []):  # factor-channel conditions: whole free-run episode
            model = models[name][0]
            tr = [Track(cfg, s, cid, ws, src="self") for (cfg, s, ws), cid in zip(items, ids)]
            drive(model, tr, {"protocol": "B", "model": name}, None, None, None, a.latent, cond, a.phi_mean_values)
            rows_m = T.episode_metrics(items, [t.ep for t in tr], [t.infos for t in tr])
            w = writers[("B", cond_name(name, cond))]
            for t, rm, k in zip(tr, rows_m, keep):
                if not k:
                    continue
                per_model_b[cond_name(name, cond)].append(rm)
                for r in t.recs:
                    w.write(r)
                w.write({"kind": "episode", "protocol": "B", "model": name, "phi": cond, "src": "self",
                         "cfg_id": t.cfg_id, "ws": t.ws, "ep_hist_id": hist_id(t.prefix), "n": len(t.recs),
                         **_json_row(rm)})
        for name in a.rollout_models:
            for ivname in a.rollout:
                model = models[name][0]
                tr = [Track(cfg, s, cid, ws, src=f"rollout:{ivname}") for (cfg, s, ws), cid in zip(items, ids)]
                drive(model, tr, {"protocol": f"R-{ivname}", "model": name}, ivns[name], ivname, supports.get(name),
                      a.latent)
                rows_m = T.episode_metrics(items, [t.ep for t in tr], [t.infos for t in tr])
                w = writers[(f"R-{_safe(ivname)}", name)]
                for t, rm, k in zip(tr, rows_m, keep):
                    if k:
                        for r in t.recs:
                            w.write(r)
                        w.write({"kind": "episode", "protocol": f"R-{ivname}", "model": name, "src": t.src,
                                 "cfg_id": t.cfg_id, "ws": t.ws, "ep_hist_id": hist_id(t.prefix), "n": len(t.recs),
                                 **_json_row(rm)})
        report["shards"].append({"part": part, "configs": len(rows), "selected": len(sel),
                                 "cpu_s": round(time.process_time() - t0, 2)})
        print(json.dumps(report["shards"][-1]), flush=True)
        del rows, items
    report["B_summary"] = {name: T.summarize_rows(r) for name, r in per_model_b.items()}


def _json_row(rm):
    return json.loads(json.dumps(rm))


def emit_forced(model, tracks, protocol, name, writers, ivn, support, latent):
    drive(model, tracks, {"protocol": protocol, "model": name}, ivn, None, support, latent)
    w = writers[(protocol, name)]
    for t in tracks:
        for r in t.recs:
            w.write(r)
        w.write(episode_record_forced(t, {"protocol": protocol, "model": name}))


# ------------------------------------------------------------------------------------------------ run: counterfactual

class _LabelQ:
    """Solver stand-in for a counterfactual decision: exact Q* from the octet's labels (keys int)."""

    def __init__(self, qd):
        self.q = {int(k): v for k, v in qd.items()}

    def q_values(self, st):
        return self.q


@torch.no_grad()
def cf_decision(model, cfg, hist, q, cid, meta_rec, ivn, support, latent, phi_cond=None, phi_mean=None):
    """Replay the decision type's visible history through the model (batch 1, exactly campaign04 replay_history /
    model_choice) and record the decision after it."""
    t = Track(cfg, _LabelQ(q), cid, None, history=hist, src="cf")
    kind = getattr(model, "inputs", "public")
    vec = cfg.public_vector()
    h = torch.zeros(1, hidden_size(model))
    prev = None
    st = pw.initial_state(cfg)
    for step in range(len(hist) + 1):
        av = T._env(cfg).available(cfg, st)
        x = torch.tensor([T.encode(vec, prev, av, st[0][0] / cfg.k) + T.supplied(cfg, st, kind)])
        h, z, pred, phi, zf = forward(model, x, h)
        if step == len(hist):
            break
        prev = tuple(hist[step])
        st = T._env(cfg).advance(cfg, st, *prev)
        t.prefix.append(prev)
    mask = torch.zeros(1, pw.N_ACTIONS, dtype=torch.bool)
    mask[0, list(av)] = True
    target = pw6.factor_features(cfg, st)
    if phi_cond not in (None, "own"):  # factor-channel condition at the decision (the state does not depend on phi)
        zf = fuse_out(model, z, torch.tensor(cond_phi(phi_cond, [target], phi_mean), dtype=z.dtype))
    logits = model.pi(zf).masked_fill(~mask, -1e9)
    greedy = int(logits.argmax())
    probs = torch.softmax(logits, -1)[0].tolist()
    pred_l = pred[0].tolist() if pred is not None else None
    t.forced = False  # the decision is the model's own (no history action at the decision)
    key = f"{cid}|{hist_id(t.prefix)}"
    rec = decision_record(t, st, _LabelQ(q).q, av, target, pred_l, probs, greedy, greedy, key, meta_rec)
    assert sorted(int(k) for k in q) == sorted(av), (cid, q, av)
    if phi_cond not in (None, "own"):
        rec["phi"] = phi_cond
    if latent:
        rec["z_pre"] = latent_b64(z.to(torch.float16).numpy()[0])
    if support is not None and pred_l is not None:
        rec["support_pred"] = support.dist(pred_l)
    if ivn is not None:
        ivr = {}
        for n in ivn.names:
            if n == "none":
                continue
            ph = ivn.phi(n, pred_l, target, key)
            lg = model.pi(fuse_out(model, z, torch.tensor([ph], dtype=z.dtype))).masked_fill(~mask, -1e9)
            ivr[n] = iv_record(ph, int(lg.argmax()), torch.softmax(lg, -1)[0].tolist(), _LabelQ(q).q, support)
        if ivr:
            rec["iv"] = ivr
    return rec


def run_cf(a, models, writers, ivns, supports, report):
    runs = [(name, None) for name in models] + list(a.phi_runs)
    for path in a.cf:
        fam_file = Path(path).stem  # cf_SCE
        sets = json.loads(Path(path).read_text())
        if a.octets:
            lo, hi = (int(x) for x in a.octets.split(":"))
            sets = [cs for cs in sets if lo <= cs["index"] < hi]
        if a.n_octets is not None:
            sets = sets[:a.n_octets]
        for name, cond in runs:
            model = models[name][0]
            t0 = time.process_time()
            w = writers[(f"cf-{fam_file[3:]}", name if cond is None else cond_name(name, cond))]
            meta_rec = {"protocol": f"cf-{fam_file[3:]}", "model": name}
            ivn_c, sup_c = (ivns.get(name), supports.get(name)) if cond is None else (None, None)
            for cs in sets:
                fam = cs["family"]
                cfgs = {k: pw6.config_from_dict6(m["config"]) for k, m in cs["members"].items()}
                for h, hist in pw6.DECISION_TYPES.items():
                    if h not in cs["types"]:
                        continue
                    tinfo = cs["types"][h]
                    unit = {"family": fam, "octet": cs["index"], "seed": cs["seed"], "type": h, "flip": tinfo["flip"],
                            "flip_ablations": tinfo.get("flip_ablations"), "unique_full": tinfo["unique"],
                            "has_near_miss": tinfo["near_miss"] is not None}
                    todo = [(k, cfgs[k], m["labels"][h]) for k, m in cs["members"].items() if m["labels"][h] is not None]
                    if tinfo["near_miss"] is not None:
                        nm = tinfo["near_miss"]
                        todo.append(("near_miss", pw6.config_from_dict6(nm["config"]), nm["label"]))
                    for key, cfg, lab in todo:
                        cid = f"{fam_file}:{cs['index']}:{key}"
                        rec = cf_decision(model, cfg, hist, lab["Q"], cid, meta_rec, ivn_c, sup_c, a.latent, cond,
                                          a.phi_mean_values)
                        assert rec["opt"] == sorted(lab["opt"]), (cid, h, rec["opt"], lab["opt"])
                        role = "near_miss" if key == "near_miss" else ("full" if key == fam else "sub")
                        rec["cf"] = {**unit, "member": key, "role": role,
                                     "near_miss_move": tinfo["near_miss"]["move"] if key == "near_miss" else None}
                        w.write(rec)
            report["cf"].append({"file": str(path), "model": name, **({"phi": cond} if cond else {}),
                                 "octets": len(sets),
                                 "cpu_s": round(time.process_time() - t0, 2)})
            print(json.dumps(report["cf"][-1]), flush=True)


# ------------------------------------------------------------------------------------------------ historical check

def historical_check(run, name, summary, pool):
    """Compare the free-running summary with the historical eval file of the run (eval_b6c.json for b6c pools)."""
    fn = "eval_b6c.json" if pool.startswith("b6c_") else "eval.json"
    p = Path(run_path(run)) / fn
    if not p.exists():
        return {"file": str(p), "status": "missing"}
    hist = json.loads(p.read_text())["splits"].get(pool, {}).get("free_running_greedy", {}).get("summary")
    if hist is None:
        return {"file": str(p), "status": "pool absent"}
    diffs = {k: (summary.get(k), v) for k, v in hist.items()
             if isinstance(v, (int, float)) and not (isinstance(summary.get(k), (int, float))
                                                     and abs(summary[k] - v) <= 1e-9 * max(1.0, abs(v)))}
    return {"file": str(p), "status": "identical" if not diffs else "DIFFERENT", "diffs": diffs,
            "n": hist.get("n")}


# ------------------------------------------------------------------------------------------------ commands

def parse_named(xs):
    out = {}
    for x in xs or []:
        k, v = x.split("=", 1)
        assert k not in out, f"duplicate name {k}"
        out[k] = v
    return out


def _safe(s):
    return s.replace(":", "_")


def cmd_run(a):
    torch.set_num_threads(1)
    t_start = time.process_time()
    runs = parse_named(a.model)
    forbidden = [run_path(r) for r in runs.values()] + [r.split("::pred=", 1)[1] for r in runs.values() if "::pred=" in r] \
        + ([a.labels] if a.labels else []) + [str(Path(p).parent) for p in a.cf or []] \
        + ([str(Path(a.phi_mean_bank).parent)] if a.phi_mean_bank else [])
    out = check_out_dir(a.out, forbidden)
    out.mkdir(parents=True, exist_ok=True)
    models = {name: load_model(r) for name, r in runs.items()}
    kinds = {name: model_kind(m) for name, (m, _) in models.items()}
    sup_paths = parse_named(a.support_ref)
    supports = {n: Support(json.loads(Path(p).read_text())) for n, p in sup_paths.items()}
    ivns = {}
    for name, k in kinds.items():  # interventions are applied to LRN models only (never RAWF / SUP / B0)
        if k in READS_PREDICTION and a.ivs:
            ivns[name] = Intervener(a.ivs, k, supports.get(name), a.noise_seed)
    a.rollout = a.rollout or []
    a.rollout_models = [n for n, k in kinds.items() if k in READS_PREDICTION] if a.rollout else []
    for n in a.rollout_models:
        if n not in ivns:
            ivns[n] = Intervener(["none"] + a.rollout, kinds[n], supports.get(n), a.noise_seed)
        for iv in a.rollout:
            ivns[n].coords(iv)
            if iv.startswith("gauss") and supports.get(n) is None:
                raise SystemExit(f"rollout {iv} needs --support-ref for {n}")
    # factor-channel conditions (addendum_2): every non-own condition x every factor-reading model; others skipped
    bad = [c for c in a.phis if c not in PHI_CONDS]
    if bad:
        raise SystemExit(f"unknown --phis {bad} (have {PHI_CONDS})")
    conds = [c for c in PHI_CONDS if c in a.phis and c != "own"]
    a.phi_runs = [(n, c) for n in models for c in conds if kinds[n] in PHI_KINDS]
    phi_skipped = sorted(n for n in models if kinds[n] not in PHI_KINDS) if conds else []
    a.phi_mean_values, phi_mean = None, None
    if "mean" in conds and a.phi_runs:
        if not a.phi_mean_bank:
            raise SystemExit("--phis mean needs --phi-mean-bank (the P2 history bank)")
        phi_mean = bank_phi_mean(a.phi_mean_bank)
        a.phi_mean_values = phi_mean["values"]
    # plan every output file first; refuse if any exists (never overwrite)
    plan = []
    if a.pool:
        protos = [p for p in a.protocols if p in ("B", "A-pistar")] + (
            [f"A-own_{s}" for s in models] if "A-own" in a.protocols else [])
        plan += [(p, n) for p in protos for n in models]
        if "A-pistar" in a.protocols:
            plan.append(("hist", "pistar"))
        plan += [(f"R-{_safe(iv)}", n) for iv in a.rollout for n in a.rollout_models]
        if "B" in a.protocols:
            plan += [("B", cond_name(n, c)) for n, c in a.phi_runs]
    for p in a.cf or []:
        plan += [(f"cf-{Path(p).stem[3:]}", n) for n in models]
        plan += [(f"cf-{Path(p).stem[3:]}", cond_name(n, c)) for n, c in a.phi_runs]
    paths = {k: out / out_name(*k) for k in plan}
    summary_path = out / f"{VERSION}-summary-{a.tag}.json"
    clash = [str(p) for p in list(paths.values()) + [summary_path] if p.exists()]
    if clash:
        raise SystemExit(f"refusing to overwrite existing outputs: {clash}")
    groups, _, gsrc = load_groups()
    header = {"version": VERSION, "tag": a.tag, "labels": a.labels, "pool": a.pool, "cf": a.cf,
              "worlds": a.worlds, "world_offset": a.world_offset, "n_configs": a.n_configs, "shards": a.shards, "n_octets": a.n_octets, "octets": a.octets,
              "eps": EPS, "actions": list(pw.ACTIONS), "features": list(FEATS), "ivs": a.ivs, "ivs_applied_to": sorted(ivns), "rollout": a.rollout,
              "groups": groups, "group_source": gsrc, "latent": "z_pre = trunk output before fusion, float16 base64"
              if a.latent else None, "noise_seed": a.noise_seed,
              "models": {n: {"run": runs[n], "kind": kinds[n], "seed": models[n][1]["seed"],
                             "model_sha256": _sha_file(Path(run_path(runs[n])) / "model.pt"),
                             **({"predictor": models[n][1]["predictor"]} if "predictor" in models[n][1] else {}),
                             **({"consumer_contract": models[n][1]["consumer_contract"]}
                                if "consumer_contract" in models[n][1] else {}),
                             "support_ref": sup_paths.get(n)} for n in models},
              "hist_support_thresholds": {"low": HIST_P_LOW, "unsupported": HIST_P_UNSUPPORTED}}
    if conds:  # absent without --phis conditions: the default header is unchanged
        header["phi_conditions"] = {"conds": ["own"] + conds, "runs": [cond_name(n, c) for n, c in a.phi_runs],
                                    "skipped_models": phi_skipped, "phi_mean": phi_mean,
                                    "note": "phi replaced at every decision (B: whole free-running episode; cf: each "
                                            "decision); 'own' = the default files; condition records carry 'phi'"}
    writers = {}
    for k, p in paths.items():
        base, _, cond = k[1].partition("@phi=")
        writers[k] = Writer(p, {**header, "protocol": k[0], "model": base, **({"phi": cond} if cond else {})})
    report = {"version": VERSION, "tag": a.tag, "shards": [], "cf": [], "hist_written": set(), "files": {}}
    try:
        if a.pool:
            run_pool(a, models, writers, header, ivns, supports, report)
            if a.check_historical:
                report["historical_check"] = {n: historical_check(runs[n], n, report["B_summary"][n], a.pool)
                                              for n in models} if a.n_configs is None and not a.shards else "skipped (subset)"
        if a.cf:
            run_cf(a, models, writers, ivns, supports, report)
    finally:
        for w in writers.values():
            w.close()
    report["hist_written"] = sorted(report["hist_written"])
    report["files"] = {str(w.path.name): w.n for w in writers.values()}
    report["cpu_s_total"] = round(time.process_time() - t_start, 2)
    report["header"] = header
    write_json_new(summary_path, report)
    print(json.dumps({k: v for k, v in report.items() if k in ("files", "cpu_s_total", "historical_check")}, indent=1))


def cmd_support(a):
    """Support reference of an LRN model on training-pool states (b6_B0): its own predictions (free-running greedy
    and pi* histories), exact targets and prediction errors."""
    import numpy as np
    torch.set_num_threads(1)
    t0 = time.process_time()
    runs = parse_named(a.model)
    assert len(runs) == 1
    name, run = next(iter(runs.items()))
    out = check_out_dir(a.out, [run_path(run), a.labels])
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{VERSION}-support-{name}.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    model, meta = load_model(run)
    assert model_kind(model) in READS_PREDICTION, "support references are for learned-factor consumers"
    pool = T.load_pool(a.labels, a.pool)
    rng = random.Random(a.sample_seed)
    idxs = sorted(rng.sample(range(len(pool)), min(a.n_configs, len(pool))))
    items = [(pool[i][1], pool[i][2], T.TRAIN_WORLD_BASE + meta["seed"] * 100_000_000 + SUPPORT_WORLD_OFFSET + i)
             for i in idxs]
    ids = [f"{a.pool}:{pool[i][0]}" for i in idxs]
    del pool
    preds, targs, srcs = [], [], []
    for c0 in range(0, len(items), 64):
        part, pid = items[c0:c0 + 64], ids[c0:c0 + 64]
        tr = [Track(cfg, s, cid, ws, src="self") for (cfg, s, ws), cid in zip(part, pid)]
        drive(model, tr, {"protocol": "support", "model": name}, latent=False)
        ref_eps, _ = T.reference_policy_eval(part, "pi_star")
        tr2 = [Track(cfg, s, cid, ws, history=ep.history, src="pistar") for (cfg, s, ws), cid, ep in zip(part, pid, ref_eps)]
        drive(model, tr2, {"protocol": "support", "model": name}, latent=False)
        for t in tr + tr2:
            for r in t.recs:
                preds.append(r["pred"])
                targs.append(r["target"])
                srcs.append(t.src)
    P, Tg = np.array(preds), np.array(targs)
    cov = np.cov(P.T)
    ridge = 1e-6 * max(np.trace(cov) / NF, 1e-12)
    prec = np.linalg.pinv(cov + ridge * np.eye(NF))
    mean = P.mean(0)
    d = P - mean
    maha = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, prec, d), 0))
    U = np.unique(P, axis=0)  # distinct prediction vectors (pi* and own histories share states, e.g. every s0)
    srng = np.random.default_rng(a.sample_seed)
    ksel = np.sort(srng.choice(len(U), size=min(a.knn_size, len(U)), replace=False))
    K = U[ksel]
    # leave-one-out nearest-neighbour distance of the kNN sample (reference scale for kNN distances)
    nn = []
    for j in range(len(K)):
        dd = np.sqrt(((K - K[j]) ** 2).sum(1))
        dd[j] = np.inf
        nn.append(dd.min())
    nn = np.array(nn)
    q = lambda x: {f"q{p}": float(np.quantile(x, p / 100)) for p in (50, 90, 95, 99)}  # noqa: E731
    ref = {"version": VERSION, "model": name, "run": run, "model_sha256": _sha_file(Path(run_path(run)) / "model.pt"),
           "labels": a.labels, "pool": a.pool, "configs": ids, "n_states": int(len(P)), "n_distinct": int(len(U)),
           "sources": {s: srcs.count(s) for s in sorted(set(srcs))},
           "world_seed_base": T.TRAIN_WORLD_BASE + meta["seed"] * 100_000_000 + SUPPORT_WORLD_OFFSET,
           "features": list(FEATS), "min": P.min(0).tolist(), "max": P.max(0).tolist(), "mean": mean.tolist(),
           "std": P.std(0).tolist(), "target_std": Tg.std(0).tolist(), "target_min": Tg.min(0).tolist(),
           "target_max": Tg.max(0).tolist(), "err_rms": np.sqrt(((P - Tg) ** 2).mean(0)).tolist(),
           "err_mae": np.abs(P - Tg).mean(0).tolist(), "precision": prec.tolist(), "ridge": ridge,
           "maha_ref": q(maha), "knn_sample": K.tolist(), "knn_ref_loo": q(nn), "sample_seed": a.sample_seed,
           "cpu_s": round(time.process_time() - t0, 2)}
    write_json_new(path, ref)
    print(json.dumps({k: ref[k] for k in ("n_states", "sources", "maha_ref", "knn_ref_loo", "cpu_s")}))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--labels", default=None)
    s.add_argument("--pool", default=None)
    s.add_argument("--model", action="append", required=True, metavar="NAME=RUNDIR")
    s.add_argument("--protocols", nargs="*", default=["B", "A-pistar", "A-own"],
                   choices=("B", "A-pistar", "A-own"))
    s.add_argument("--cf", action="append", default=None)
    s.add_argument("--ivs", nargs="*", default=[])
    s.add_argument("--rollout", nargs="*", default=[])
    s.add_argument("--phis", nargs="*", default=["own"], help="factor-channel conditions (own exact zero mean)")
    s.add_argument("--phi-mean-bank", default=None, help="P2 history bank (bank.pkl) for --phis mean")
    s.add_argument("--support-ref", action="append", default=None, metavar="NAME=FILE")
    s.add_argument("--n-configs", type=int, default=None)
    s.add_argument("--shards", type=int, nargs="*", default=None, help="process only these shard indices")
    s.add_argument("--n-octets", type=int, default=None)
    s.add_argument("--octets", default=None, metavar="LO:HI", help="octet index range (subsampling / splitting)")
    s.add_argument("--worlds", type=int, default=1)
    s.add_argument("--world-offset", type=int, default=0, help="b6c eval used --world-offset 0 --worlds 1")
    s.add_argument("--no-latent", dest="latent", action="store_false")
    s.add_argument("--noise-seed", type=int, default=0)
    s.add_argument("--check-historical", action="store_true")
    s.add_argument("--tag", default="run")
    s.add_argument("--out", required=True)
    s = sub.add_parser("support")
    s.add_argument("--labels", required=True)
    s.add_argument("--pool", default="b6_B0")
    s.add_argument("--model", action="append", required=True, metavar="NAME=RUNDIR")
    s.add_argument("--n-configs", type=int, default=96)
    s.add_argument("--knn-size", type=int, default=3000)
    s.add_argument("--sample-seed", type=int, default=0)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "run" and not a.pool and not a.cf:
        p.error("run needs --pool and/or --cf")
    if a.cmd == "run" and a.pool and not a.labels:
        p.error("--pool needs --labels")
    {"run": cmd_run, "support": cmd_support}[a.cmd](a)


if __name__ == "__main__":
    main()
