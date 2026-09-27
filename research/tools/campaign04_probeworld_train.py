"""Extended-04 Track B: probeworld factorization ladder (L0-L4), CPU only.

Subcommands (all deterministic given their arguments; see research/campaigns/extended-04/probeworld.md):

  labels  --out DIR [--n-train N] [--n-eval N]
      Exact DP label tables for every split pool (oracle label compute; charged as offline CPU).
  train   --labels DIR --out DIR --rung L0..L4 --seed S [--updates U] [--batch B]
      One ladder run.  Every rung: identical model (all heads present), identical config/world stream,
      identical optimizer and number of updates/episodes; rungs differ only in which loss weights are > 0.
      --own-value (protocol-B2, default off): add a separate head v_own on the stop-gradient trunk features,
      trained on the realized return-to-go of the model's OWN free-running greedy rollouts at the current
      parameters (continuation OWN_VALUE_CONTINUATION).  Off -> B1 behaviour bit-identical.
  eval    --labels DIR --run DIR [--worlds W]
      Free-running greedy and teacher-forced (pi* histories) evaluation on dev/test_iid/heldout_* pools.
  summarize --runs DIR... --out FILE
      Aggregate eval JSONs across seeds/rungs (+ reference policies).

Extended-05 Track B additions (all default off; defaults reproduce B1/B2/F2 bit for bit, tested):
  labels --split-set b5          B-SPLIT pools (campaign05_probeworld.SPLITS5 + the extended-04 heldout_comp challenge set)
  train  --train-split SPLIT     training pool (default 'train'; b5_train for B0/BX2/BO, b5x_train for BX1)
  train  --inputs public|belief|bx2   supplied public state appended to the inputs (BO: belief; BX2: belief + per-flag
                                 features); a deterministic public computation, labelled supplied
  eval   --split-set b5 --episode-rows   b5 eval splits; per-episode decision records (episodes.jsonl.gz) for the
                                 B-X scorer (campaign05_bx_score.py)
  labels/eval/references --split-set b5c   B-XC: ONLY the fresh sized U+C hold b5c_hold_uc (seed base 5.9e9); models
                                 trained on a b5 labels dir are evaluated with --labels pointing at the b5c labels dir

Extended-06 Track B additions (all default off; b1/b5/b5c paths call the identical functions, tested):
  labels --split-set b6          probeworld-v3 split table v3 (campaign06_probeworld.SPLITS6): shared-seed training
                                 pools (base / broad / dose, 2N or N configurations), dev/test, the registered
                                 held-out challenge families, the historical challenge sets, and the balanced
                                 counterfactual sets cf_<family>.json (exact labels per decision type)
  train  --train-n N             use the first N configurations of --train-split (B0 = N base, B1 = 2N base, ...)
  train  --inputs factors6       SUPPLIED-FACTORIZED public factor values (campaign06_probeworld.FACTOR_FEATURES)
  train  --arch fuse [--factor-mode supplied|learned|none] [--aux-weight W]
                                 factors enter a fusion layer read by every head: supplied (inputs factors6),
                                 learned (auxiliary head predicts the factors from the trunk; the policy reads the
                                 STOP-GRADIENT prediction), none (capacity-matched raw control)
  eval   --split-set b6 [--cf]   b6 eval pools; --cf adds balanced-counterfactual / one-factor intervention records
                                 (cf_eval.json) by replaying the model along each decision type's visible history
  labels --split-set b6c [--parts eval cf]   B-FACT-C fresh confirmation: ONLY b6c_hold_SCE (--n-hold; config base
                                 6.42e9) and fresh S+C+E octets cf_SCE.json (--n-cf; config base 6.65e9); same
                                 generator / eligibility / octet / near-miss code as b6, only the seed bases differ
  eval   --split-set b6c [--cf]  models trained on the b6 labels dir, --labels = the b6c labels dir; evaluates only
                                 b6c_hold_SCE and writes eval_b6c.json / eval_b6c_episodes.jsonl.gz / cf_eval_b6c.json
                                 (never the run's b6 files; campaign06_bprimary --group b6c_hold_SCE reads them)

Extended-07 Phase 2 additions (all default off; every historical invocation runs the unchanged code path, tested
against goldens of the unmodified trainer in tests/test_campaign07_p2.py; research/campaigns/extended-07/p2-infra.md):
  train  --arch fuse --shape {0,1} --read {0,1}   genuine S x R 2x2 (factor mode 'sr'): identical modules / init in all
                                 four arms; S0 = aux reads sg(z); R0 = policy reads phi_const (--phi-const zeros |
                                 train_mean); --clip-mode split (default: separate Adam + clip for aux.*) | global
                                 (historical coupling); --action-rng counter (default for P2) | stream; --init-from /
                                 --init-sha (common initialization); --aux-group-weights G=W
  train  --bank FILE --bank-updates N [--finetune --updates U]   common valid-history bank stage (teacher-forced replay
                                 of stored public histories; imitation + aux), then optional on-policy fine-tuning
  train  --predictor-only [--bank-folds K --bank-exclude-fold k]   factor predictors (aux MSE only) for K-fold OOF
  train  --inputs factors6 --arch fuse --phi-contract exact|oof|mix [--oof F]   controlled consumers

Privileged labels (Q*, A*, stage, dependency, switch, case) enter ONLY training losses; model inputs are the
public config vector plus the visible step record and the public available-action mask.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from tensegra import campaign04_probeworld as pw  # noqa: E402
from tensegra import campaign05_probeworld as pw5  # noqa: E402  (extended-05 Track B; additive)
from tensegra import campaign06_probeworld as pw6  # noqa: E402  (extended-06 Track B; additive)

_env = pw6.env  # pw itself for probeworld-v1 configurations (b1/b5/b5c paths call the identical functions)


def split_cfg(split, idx):
    """Configuration `idx` of a split: split table v3 (b6) or pw5.split_config (b5/b5c and extended-04 names)."""
    return pw6.split6_config(split, idx) if pw6.is_split6(split) else pw5.split_config(split, idx)


def supplied(cfg, state, kind):
    return pw6.factor_features(cfg, state) if kind == "factors6" else pw5.supplied_features(cfg, state, kind)


def supplied_dim(kind):
    return pw6.N_FACTOR_FEATURES if kind == "factors6" else pw5.supplied_dim(kind)


def split_world_seed(split, idx, rep):
    return pw6.world_seed6(split, idx, rep) if pw6.is_split6(split) else pw5.world_seed(split, idx, rep)

RUNGS = ("L0", "L1", "L2", "L3", "L4")
# loss weights per rung (cumulative ladder); every rung keeps the actor-critic RL loss.
RUNG_WEIGHTS = {
    "L0": dict(rl=1.0, imit=0.0, dep=0.0, switch=0.0, q=0.0),
    "L1": dict(rl=1.0, imit=1.0, dep=0.0, switch=0.0, q=0.0),
    "L2": dict(rl=1.0, imit=1.0, dep=0.5, switch=0.0, q=0.0),
    "L3": dict(rl=1.0, imit=1.0, dep=0.5, switch=0.5, q=0.0),
    "L4": dict(rl=1.0, imit=1.0, dep=0.5, switch=0.5, q=1.0),
}
EVAL_SPLITS = ("dev", "test_iid", "heldout_price", "heldout_k", "heldout_comp")
N_STAGES = len(pw.STAGES)
N_DEP = len(pw.DEP_BITS)
STEP_DIM = (pw.N_ACTIONS + 1) + (pw.N_OUTCOMES + 1) + 1 + (4 + 1)  # prev action, outcome, event, reveal
IN_DIM = pw.PUBLIC_DIM + STEP_DIM + pw.N_ACTIONS + 1  # + available mask + queries_done/k
HIDDEN = 128
TRAIN_WORLD_BASE = 8_000_000_000
# protocol-B2 own-greedy rollouts: world seeds TRAIN_WORLD_BASE + 1e8*seed + OWN_WORLD_OFFSET + n.  Inside the
# seed's training band (1e8 wide) but disjoint from the sampled-training worlds (< 256k used per run).
OWN_WORLD_OFFSET = 60_000_000
OWN_VALUE_CONTINUATION = "own_greedy_policy_current_params_mc_v1"


# ------------------------------------------------------------------------------------------------ labels

def build_pool(split: str, n: int):
    pool = []
    for idx in range(n):
        cfg = split_cfg(split, idx)  # == pw.split_config for extended-04 split names
        s = _env(cfg).ExactSolver(cfg)
        s.value(pw.initial_state(cfg))
        pool.append((idx, cfg, s._V, s._Q))
    return pool


def cmd_labels(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    sset = getattr(a, "split_set", "b1")
    if sset == "b6":
        return cmd_labels_b6(a, out)
    if sset == "b6c":
        return cmd_labels_b6c(a, out)
    if sset in pw5.SPLIT_SETS:  # b5 (B-SPLIT pools) or b5c (B-XC: ONLY the fresh sized U+C hold)
        todo = [(s, None if opt == "sized" else getattr(a, opt)) for s, opt in pw5.SPLIT_SETS[sset]["label_splits"]]
    else:
        todo = [("train", a.n_train)] + [(s, a.n_eval) for s in EVAL_SPLITS]
    for split, n in todo:
        if n is None:  # extended-05 new holds: pool sized by the exact solver, stored in shards
            meta[split] = build_hold_shards(out, split, a.hold_target, a.hold_max, a.hold_shard,
                                            getattr(a, "hold_target_sensitive", None))
            print(split, meta[split], flush=True)
            continue
        t0 = time.process_time()
        pool = build_pool(split, n)
        dt = time.process_time() - t0
        with open(out / f"{split}.pkl", "wb") as f:
            pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)
        nst = [len(V) for _, _, V, _ in pool]
        meta[split] = {"n_configs": n, "cpu_s": round(dt, 2), "states_total": sum(nst), "states_max": max(nst),
                       "k_counts": {k: sum(1 for _, c, _, _ in pool if c.k == k) for k in pw.K_VALUES},
                       "combo_counts": _combo_counts(pool)}
        print(split, meta[split], flush=True)
    meta["version"] = pw.VERSION
    if getattr(a, "split_set", "b1") == "b5":  # key absent for the B1 split set (labels_meta unchanged)
        meta["split_set"] = {"name": "b5", "version": pw5.VERSION, "split_table": pw5.SPLIT_TABLE_VERSION,
                             "hold_target_eligible": a.hold_target}
    elif sset == "b5c":  # B-XC labels dir: evaluation pool only (models are trained on the b5 labels dir)
        meta["split_set"] = {"name": "b5c", "registry": "B-XC", "version": pw5.VERSION,
                             "split_table": pw5.SPLIT_TABLE_VERSION, "hold_target_eligible": a.hold_target,
                             "splits": {s: {"config_seed_base": pw5.SPLITS5C[s][3], "fresh_draws_of": src}
                                        for s, src in pw5.B5C_SOURCE_SPLIT.items()}}
    meta["eps"] = pw.EPS
    meta["continuation"] = pw.CONTINUATION
    (out / "labels_meta.json").write_text(json.dumps(meta, indent=1))


def cmd_labels_b6(a, out):
    """extended-06 split table v3 (design v2): training pools of every arm (B0/B1 base, B2/B3 replacement-matched
    variety, dose1-3; campaign06_probeworld.build_arm_pools), dev/test pools, sharded held-out challenge families and
    historical challenge sets (per-configuration s0 facts; optional exact per-factor relevance), and the balanced
    counterfactual octets.  --parts selects label jobs that can run in parallel (train / eval / cf); each part writes
    labels_meta.<part>.json."""
    parts = a.parts or ["train", "eval", "cf"]
    meta = {}
    n = a.n_train
    if "train" in parts:
        t0 = time.process_time()
        classify = pw6.make_classifier()  # DP per candidate, first-action class kept, table discarded
        arms = a.arms or list(pw6.ARM_SIZE)
        pools, report = pw6.build_arm_pools(n, arms=_with_refs(arms), classify=classify)
        meta["classified_configs"] = len(classify.memo)
        for arm in arms:  # arm by arm (bounded memory; shared configurations are re-solved)
            pool = []
            for i, (key, j) in enumerate(pools[arm]):
                cfg = pw6.type_config(key, j)
                s = pw6.ExactSolver(cfg)
                s.value(pw.initial_state(cfg))
                pool.append((i, cfg, s._V, s._Q))
            with open(out / f"b6_{arm}.pkl", "wb") as f:
                pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)
            nst = [len(V) for _, _, V, _ in pool]
            meta[f"b6_{arm}"] = {"n_configs": len(pool), "states_total": sum(nst), "states_max": max(nst),
                                 "k_counts": {k: sum(1 for _, c, _, _ in pool if c.k == k) for k in pw6.V3_K},
                                 "combo_counts": _combo_counts(pool), "stream": [list(x) for x in pools[arm]],
                                 "report": report[arm], "hist_in_training": pw6.HIST_IN_TRAINING[arm]}
            print(f"b6_{arm}", json.dumps({k: v for k, v in meta[f"b6_{arm}"].items() if k != "stream"}), flush=True)
            del pool
        meta["train_cpu_s"] = round(time.process_time() - t0, 2)
    if "eval" in parts:
        todo = [("b6_dev", a.n_eval), ("b6_test_base", a.n_eval), ("b6_test_pairs", a.n_eval)]
        todo += [(s, a.n_hold) for s in pw6.B6_HOLD_SPLITS]
        todo += [(s, a.n_hist) for s, v in pw6.SPLITS6.items() if v[2] == "hist"]
        for split, cnt in todo:
            if cnt <= 0:
                continue
            meta[split] = build_b6_shards(out, split, cnt, a.b6_shard,
                                          relevance=a.hold_relevance and pw6.SPLITS6[split][2] in ("hold", "hist"))
            print(split, json.dumps(meta[split]), flush=True)
    if "cf" in parts:
        for fam in pw6.HOLD_FAMILIES + (pw6.HIST_FAMILIES if a.cf_hist else ()):
            if a.n_cf <= 0:
                continue
            t0 = time.process_time()
            sets = [pw6.counterfactual_set(fam, i) for i in range(a.n_cf)]
            (out / f"cf_{fam}.json").write_text(json.dumps(sets))
            meta[f"cf_{fam}"] = {"n_sets": len(sets), "cpu_s": round(time.process_time() - t0, 2),
                                 "config_seed_base": pw6.CF_BASE[fam], "support": cf_support(sets)}
            print(f"cf_{fam}", json.dumps(meta[f"cf_{fam}"]), flush=True)
    meta["version"] = pw.VERSION
    meta["split_set"] = {"name": "b6", "env": pw6.VERSION, "generator": pw6.GENERATOR_VERSION,
                         "split_table": pw6.SPLIT_TABLE_VERSION, "hold_families": list(pw6.HOLD_FAMILIES),
                         "hist_families": list(pw6.HIST_FAMILIES), "base_pair": pw6.BASE_PAIR,
                         "variety_pairs": list(pw6.VARIETY_PAIRS), "dose_order": list(pw6.DOSE_ORDER), "n": n,
                         "k_support": list(pw6.V3_K), "corr_range": list(pw6.CORR_RANGE),
                         "p_event_range": list(pw6.P_EVENT_RANGE), "audit": pw6.audit_split_table(n), "parts": parts}
    meta["eps"] = pw.EPS
    meta["continuation"] = pw.CONTINUATION
    name = "labels_meta.json" if parts == ["train", "eval", "cf"] else f"labels_meta.{'-'.join(parts)}.json"
    (out / name).write_text(json.dumps(meta, indent=1))


def _split6_entry(split):
    return pw6.SPLITS6[split] if split in pw6.SPLITS6 else pw6.SPLITS6C[split]


def cmd_labels_b6c(a, out):
    """B-FACT-C fresh confirmation labels (split set b6c): ONLY the fresh S+C+E held-out pool b6c_hold_SCE (part eval,
    --n-hold configurations, sharded as b6) and fresh S+C+E octets cf_SCE.json (part cf, --n-cf octets, config seeds
    pw6.CF_BASE_C).  Same generator, eligibility, octet and near-miss code as b6; only the seed bases differ.  Models
    are trained on the b6 labels dir and evaluated with --labels pointing here (eval --split-set b6c)."""
    parts = a.parts or ["eval", "cf"]
    if "train" in parts:
        raise SystemExit("b6c has no training part (train on the b6 labels dir)")
    audit = pw6.audit_b6c()
    assert audit["pass"], audit
    meta = {}
    if "eval" in parts:
        for split in pw6.B6C_EVAL_SPLITS:
            if a.n_hold <= 0:
                continue
            meta[split] = build_b6_shards(out, split, a.n_hold, a.b6_shard,
                                          relevance=a.hold_relevance)
            print(split, json.dumps(meta[split]), flush=True)
    if "cf" in parts:
        for fam, base in pw6.CF_BASE_C.items():
            if a.n_cf <= 0:
                continue
            t0 = time.process_time()
            sets = [pw6.counterfactual_set(fam, i, base=base) for i in range(a.n_cf)]
            (out / f"cf_{fam}.json").write_text(json.dumps(sets))
            meta[f"cf_{fam}"] = {"n_sets": len(sets), "cpu_s": round(time.process_time() - t0, 2),
                                 "config_seed_base": base, "support": cf_support(sets)}
            print(f"cf_{fam}", json.dumps(meta[f"cf_{fam}"]), flush=True)
    meta["version"] = pw.VERSION
    meta["split_set"] = {"name": "b6c", "registry": "B-FACT-C", "b6c_version": pw6.B6C_VERSION, "env": pw6.VERSION,
                         "generator": pw6.GENERATOR_VERSION, "split_table": pw6.SPLIT_TABLE_VERSION,
                         "splits": {s: [list(c), b, r] for s, (c, b, r) in pw6.SPLITS6C.items()},
                         "cf_base": dict(pw6.CF_BASE_C), "k_support": list(pw6.V3_K),
                         "corr_range": list(pw6.CORR_RANGE), "p_event_range": list(pw6.P_EVENT_RANGE),
                         "audit": audit, "parts": parts}
    meta["eps"] = pw.EPS
    meta["continuation"] = pw.CONTINUATION
    (out / f"labels_meta.b6c.{'-'.join(parts)}.json").write_text(json.dumps(meta, indent=1))


def check_b6c_labels(labels_dir, splits, cf):
    """eval --split-set b6c: the labels dir must hold b6c labels built from the CURRENT b6c bases (never a b6 dir)."""
    d = Path(labels_dir)
    metas = [json.loads(p.read_text()) for p in sorted(d.glob("labels_meta*.json"))]
    names = {m.get("split_set", {}).get("name") for m in metas}
    assert names == {"b6c"}, f"{labels_dir} is not a b6c labels dir ({names})"
    for split in splits:
        info = json.loads((d / f"{split}.shards.json").read_text())
        assert info["config_seed_base"] == _split6_entry(split)[1], (split, info["config_seed_base"])
    if cf:
        files = sorted(d.glob("cf_*.json"))
        assert files, "no counterfactual sets in the b6c labels dir"
        for p in files:
            fam = p.stem[3:]
            base = pw6.CF_BASE_C[fam]
            assert all(cs["seed"] == base + cs["index"] for cs in json.loads(p.read_text())), p


def _with_refs(arms):
    need = list(arms)
    for arm in arms:
        ref = pw6.ARM_REPLACES.get(arm)
        if ref and ref not in need:
            need.insert(0, ref)
    order = list(pw6.ARM_SIZE)
    return sorted(need, key=order.index)


def build_b6_shards(out, split, cnt, shard, relevance=False):
    """A b6 evaluation pool stored as shards <split>.shardNNN.pkl + <split>.shards.json (bounded memory; k = 8
    configurations reach ~1.4e5 DP states), with per-configuration s0 facts in <split>_s0.json."""
    t0 = time.process_time()
    shards, cur, recs, nst = [], [], [], []
    for idx in range(cnt):
        cfg = pw6.split6_config(split, idx)
        s = pw6.ExactSolver(cfg)
        s.value(pw.initial_state(cfg))
        recs.append(b6_s0_record(idx, cfg, s._V, s._Q, relevance))
        cur.append((idx, cfg, s._V, s._Q))
        nst.append(len(s._V))
        if len(cur) == shard or idx == cnt - 1:
            name = f"{split}.shard{len(shards):03d}"
            with open(out / f"{name}.pkl", "wb") as f:
                pickle.dump(cur, f, protocol=pickle.HIGHEST_PROTOCOL)
            shards.append(name)
            cur = []
    info = {"n_configs": cnt, "shards": shards, "config_seed_base": _split6_entry(split)[1],
            "combos": list(_split6_entry(split)[0])}
    (out / f"{split}.shards.json").write_text(json.dumps(info, indent=1))
    (out / f"{split}_s0.json").write_text(json.dumps(recs))
    info.update(cpu_s=round(time.process_time() - t0, 2), states_total=sum(nst), states_max=max(nst),
                k_counts={k: sum(r["k"] == k for r in recs) for k in pw6.V3_K}, support=b6_support(recs))
    return info


def b6_s0_record(idx, cfg, V, Q, relevance=True):
    """Per-configuration facts of an evaluation pool: s0 eps-optimal set, eligibility (V* > eps) and (relevance=True)
    per-factor exact relevance (first decision changes / ignoring the factor costs > eps) via one-factor ablations."""
    s = pw6.ExactSolver(cfg)
    s._V, s._Q = V, Q
    s0 = pw.initial_state(cfg)
    q = s.q_values(s0)
    opt = pw6.eps_set(q)
    v = s.value(s0)
    rel = {}
    for f, on in zip(pw6.FACTORS, pw6.flags_of(cfg)):
        if not on or not relevance:
            continue
        sa = pw6.ExactSolver(pw6.ablate(cfg, f))
        pa = sa.pi_star(pw.initial_state(sa.cfg))
        ex, _ = pw6.any_decision_change(s, sa)
        rel[f] = {"first": pa not in opt, "any": ex, "regret": round(v - pw6.foreign_policy_value(s, sa), 6)}
    rec = {"idx": idx, "k": cfg.k, "family": pw6.fam_key(pw6.flags_of(cfg)), "V": v, "eligible": v > pw.EPS,
           "opt": sorted(opt), "unique": len(opt) == 1, "pi": s.pi_star(s0)}
    if relevance:
        rec.update(rel=rel, joint_flip=len(rel) >= 2 and all(r["first"] for r in rel.values()),
                   all_any_relevant=len(rel) >= 2 and all(r["any"] for r in rel.values()),
                   all_regret_relevant=len(rel) >= 2 and all(r["regret"] > pw.EPS for r in rel.values()))
    return rec


def b6_support(recs):
    out = {"n": len(recs), "eligible": sum(r["eligible"] for r in recs), "unique_first": sum(r["unique"] for r in recs),
           "first_action": {pw.ACTIONS[a]: sum(r["pi"] == a for r in recs) for a in range(pw.N_ACTIONS)
                            if any(r["pi"] == a for r in recs)}}
    if recs and "rel" in recs[0]:
        out.update(joint_flip=sum(r["joint_flip"] for r in recs), all_any_relevant=sum(r["all_any_relevant"] for r in recs),
                   all_regret_relevant=sum(r["all_regret_relevant"] for r in recs))
    return out


def cf_support(sets):
    out = {}
    for h in pw6.DECISION_TYPES:
        ents = [s["types"][h] for s in sets if h in s["types"]]
        out[h] = {"sets": len(ents), "unique_full": sum(e["unique"] for e in ents),
                  "flip": sum(e["flip"] for e in ents), "flip_unique": sum(e["flip"] and e["unique"] for e in ents),
                  "near_miss": sum(e["near_miss"] is not None for e in ents)}
    return out


def _combo_counts(pool):
    d = {}
    for _, c, _, _ in pool:
        key = pw6.combo_name(c.flags)
        d[key] = d.get(key, 0) + 1
    return d


def build_hold_shards(out, split, target, max_n, shard, sens_target=None):
    """extended-05 (design v2 revision 11): label a new-hold pool whose size is fixed by the exact solver -- the
    smallest prefix of the split's configuration stream with `target` configurations whose first action is uniquely
    probe (pw5.hold_pool_size).  Stored as shards <split>.shardNNN.pkl (bounded memory) + <split>.shards.json, and
    the per-configuration s0 facts (incl. registered flag sensitivity) in <split>_s0.json."""
    t0 = time.process_time()
    records, cur, shards, idx, nst = [], [], [], 0, []
    sens_flags = pw5.SENSITIVITY_FLAGS.get(split)
    t_sens = pw5.HOLD_TARGET_SENSITIVE.get(split, 0) if sens_target is None else (sens_target if sens_flags else 0)
    n_final = None
    while idx < max_n and n_final is None:
        cfg = pw5.split_config(split, idx)
        s = pw.ExactSolver(cfg)
        s.value(pw.initial_state(cfg))
        rec = {"idx": idx, **pw5.s0_record(s)}
        if sens_flags:
            rec.update(pw5.flag_sensitivity(cfg, rec["V"], sens_flags))
        records.append(rec)
        cur.append((idx, cfg, s._V, s._Q))
        nst.append(len(s._V))
        idx += 1
        n_final = pw5.hold_pool_size(records, target, t_sens)
        if len(cur) == shard or n_final is not None or idx == max_n:
            name = f"{split}.shard{len(shards):03d}"
            with open(out / f"{name}.pkl", "wb") as f:
                pickle.dump(cur, f, protocol=pickle.HIGHEST_PROTOCOL)
            shards.append(name)
            cur = []
    info = {"n_configs": idx, "sizing_rule": f"smallest prefix with >= {target} s0-uniquely-probe-optimal configs"
            + (f", >= {t_sens} of them flag-sensitive" if t_sens else ""),
            "target_eligible": target, "target_eligible_flag_sensitive": t_sens, "eligible": sum(r["probe_unique"] for r in records),
            "reached_target": n_final is not None, "shards": shards}
    (out / f"{split}.shards.json").write_text(json.dumps(info, indent=1))
    (out / f"{split}_s0.json").write_text(json.dumps(records))
    ks = {k: sum(1 for r in records if r["k"] == k) for k in pw.K_VALUES}
    info.update(cpu_s=round(time.process_time() - t0, 2), states_total=sum(nst), states_max=max(nst), k_counts=ks,
                eligible_by_k={k: sum(1 for r in records if r["k"] == k and r["probe_unique"]) for k in pw.K_VALUES})
    if sens_flags:
        info["flag_sensitive"] = sum(r["flag_sensitive"] for r in records)
        info["flag_sensitive_eligible"] = sum(r["flag_sensitive"] and r["probe_unique"] for r in records)
    return info


def pool_parts(labels_dir, split):
    """Label files of a split: [split] for an ordinary pool, the shard names for a sized (sharded) hold pool."""
    p = Path(labels_dir) / f"{split}.shards.json"
    if not (Path(labels_dir) / f"{split}.pkl").exists() and p.exists():
        return json.loads(p.read_text())["shards"]
    return [split]


def load_pool(labels_dir, split):
    parts = pool_parts(labels_dir, split)
    if parts != [split]:  # sharded hold pool: concatenate (use pool_parts + load_pool per shard to bound memory)
        return [x for part in parts for x in load_pool(labels_dir, part)]
    with open(Path(labels_dir) / f"{split}.pkl", "rb") as f:
        pool = pickle.load(f)
    out = []
    for idx, cfg, V, Q in pool:
        s = _env(cfg).ExactSolver(cfg)
        s._V, s._Q = V, Q
        out.append((idx, cfg, s))
    return out


# ------------------------------------------------------------------------------------------------ model

# extended-05 BX3: per-flag modular encoders (design v2 revision 14).  Column indices into the public vector
# (campaign04_probeworld.Config.public_vector): D_side 10, q 11, corr 12, p_event 13, flags 19..22.
FLAG_PARAM_COLS = (11, 10, 12, 13)  # unreliable -> q (module sees 1 - q), side_effect, correlated, events
FLAG_BIT_COL0 = 19
MODULAR_WIDTH = 16
B1_PARAMS = 129_189  # B0/L1 ProbeNet at hidden 128 (the parameter count BX3 is matched to)


class ProbeNet(nn.Module):
    """GRU over visible-history tokens + public prices.  All heads exist in every rung (matched capacity)."""

    def __init__(self, hidden=HIDDEN, own_value=False, inputs="public", arch="flat", public_extra=0,
                 factor_mode=None, shape=None, read=None):
        super().__init__()
        self.inputs = inputs  # extended-05: supplied public state appended to the inputs ('public' = none)
        self.arch = arch  # extended-05: 'flat' (B1) | 'modular' (BX3); extended-06: 'fuse'
        self.base_dim = IN_DIM + public_extra  # extended-06: probeworld-v3 public vector has public_extra more entries
        fuse_in = arch == "fuse"  # extended-06: supplied factors go to the fusion layer, not the GRU input
        self.inp = nn.Linear(self.base_dim + (0 if fuse_in else supplied_dim(inputs)), hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, pw.N_ACTIONS)
        self.v = nn.Linear(hidden, 1)
        self.q = nn.Linear(hidden, pw.N_ACTIONS)
        self.stage = nn.Linear(hidden, N_STAGES)
        self.dep = nn.Linear(hidden, N_DEP)
        self.switch = nn.Linear(hidden, 1)
        self.case = nn.Linear(hidden, 4)
        if own_value:  # protocol-B2; created LAST so every B1 parameter gets the identical initialization
            self.v_own = nn.Linear(hidden, 1)
        if arch == "modular":  # BX3: one gated encoder per flag over (its own parameter, the full public input)
            self.flag_mods = nn.ModuleList(
                nn.Sequential(nn.Linear(IN_DIM + 1, MODULAR_WIDTH), nn.Tanh(), nn.Linear(MODULAR_WIDTH, hidden))
                for _ in range(4))
        if arch == "fuse":  # extended-06 factorized arms (created last: every other parameter keeps its init order)
            self.factor_mode = factor_mode or ("supplied" if inputs == "factors6" else "none")
            nf = pw6.N_FACTOR_FEATURES
            if self.factor_mode in ("learned", "sr"):  # extended-07 'sr': the identical module set, same order
                self.aux = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, nf))
            self.fuse = nn.Linear(hidden + nf, hidden)
            if self.factor_mode == "sr":  # extended-07 genuine S x R 2x2 (gradient-flow.md 6)
                self.sr_shape, self.sr_read = bool(shape), bool(read)
                # R0 constant fusion input: NOT part of the state_dict (non-persistent), so the initial parameters
                # of all four arms hash identically; its value is recorded in train_meta['fuse']['phi_const']
                self.register_buffer("phi_const", torch.zeros(nf), persistent=False)

    def step(self, x, h):
        pre = self.inp(x[:, :self.base_dim] if getattr(self, "arch", "flat") == "fuse" else x)
        if getattr(self, "arch", "flat") == "modular":
            base = x[:, :IN_DIM]
            for j, mod in enumerate(self.flag_mods):
                par = base[:, FLAG_PARAM_COLS[j]:FLAG_PARAM_COLS[j] + 1]
                if j == 0:
                    par = 1.0 - par  # detection miss rate (0 when reliable)
                gate = base[:, FLAG_BIT_COL0 + j:FLAG_BIT_COL0 + j + 1]
                pre = pre + gate * mod(torch.cat([par, base], 1))
        h = self.gru(torch.tanh(pre), h)
        z = self.trunk(h)
        if getattr(self, "arch", "flat") == "fuse":
            z = self.fuse_step(x, z)
        return h, z

    def fuse_step(self, x, z):
        """extended-06: heads read tanh(W [z; phi]).  phi = supplied public factors (from the input tail), the
        STOP-GRADIENT auxiliary prediction (learned; the aux head is trained only by its factor loss), or zeros
        (capacity-matched raw control).  The auxiliary prediction of the last step is kept in self.last_aux."""
        if self.factor_mode == "supplied":
            phi = x[:, self.base_dim:]
        elif self.factor_mode == "learned":
            self.last_aux = self.aux(z)
            phi = self.last_aux.detach()
        elif self.factor_mode == "sr":  # S: aux reads z (S1) or z.detach() (S0); R: policy reads sg(pred) or a constant
            self.last_aux = self.aux(z if self.sr_shape else z.detach())
            phi = self.last_aux.detach() if self.sr_read else self.phi_const.expand(z.shape[0], -1)
        else:
            phi = torch.zeros(z.shape[0], pw6.N_FACTOR_FEATURES)
        return torch.tanh(self.fuse(torch.cat([z, phi], 1)))


def matched_hidden(arch="modular", inputs="public", target=B1_PARAMS):
    """Hidden width whose ProbeNet parameter count is closest to `target` (BX3 matched capacity, disclosed)."""
    best = None
    with torch.random.fork_rng(devices=[]):  # never perturbs the caller's torch RNG (initialization matching)
        for hdim in range(64, 160):
            d = abs(n_params(ProbeNet(hdim, inputs=inputs, arch=arch)) - target)
            if best is None or d < best[0]:
                best = (d, hdim)
    return best[1]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def encode(cfg_vec, prev, avail, qfrac):
    """cfg_vec: public list; prev: (a, o, event, reveal) or None; avail: tuple of action ids."""
    x = list(cfg_vec)
    a1 = [0.0] * (pw.N_ACTIONS + 1)
    o1 = [0.0] * (pw.N_OUTCOMES + 1)
    r1 = [0.0] * 5
    ev = 0.0
    if prev is None:
        a1[-1] = o1[-1] = 1.0
        r1[-1] = 1.0
    else:
        a, o, e, rev = prev
        a1[a] = 1.0
        o1[o] = 1.0
        ev = float(e)
        r1[rev if rev is not None else 4] = 1.0
    m = [0.0] * pw.N_ACTIONS
    for a in avail:
        m[a] = 1.0
    return x + a1 + o1 + [ev] + r1 + m + [qfrac]


CASE_IDX = {"a": 0, "b": 1, "c": 2, "d": 3}


def run_batch(model, items, mode, rng=None, need_labels=True):
    """Roll out a batch.  items: list of (cfg, solver, world_seed).  mode: 'sample' | 'greedy' | 'teacher'.
    Returns per-step tensors and per-episode records."""
    B = len(items)
    eps = [_env(cfg).Episode(cfg, ws) for cfg, _, ws in items]
    vecs = [cfg.public_vector() for cfg, _, _ in items]
    h = torch.zeros(B, model.gru.hidden_size)
    prev = [None] * B
    prev_state = [None] * B
    prev_act = [None] * B
    steps = []  # per step: dict of tensors over active episodes
    ep_steps = [[] for _ in range(B)]  # per-episode step info (python)
    while True:
        act = [i for i in range(B) if not eps[i].done]
        if not act:
            break
        avails = [eps[i].available() for i in act]
        kind = getattr(model, "inputs", "public")
        x = torch.tensor([encode(vecs[i], prev[i], av, eps[i].query / eps[i].cfg.k)
                          + supplied(eps[i].cfg, eps[i].state, kind) for i, av in zip(act, avails)])
        idx = torch.tensor(act)
        hn, z = model.step(x, h[idx])
        aux_rec = None
        if getattr(model, "factor_mode", None) in ("learned", "sr"):  # extended-06/07 auxiliary factor targets
            aux_rec = {"aux": model.last_aux,
                       "aux_t": torch.tensor([pw6.factor_features(eps[i].cfg, eps[i].state) for i in act])}
        h = h.index_copy(0, idx, hn)
        mask = torch.zeros(len(act), pw.N_ACTIONS, dtype=torch.bool)
        for j, av in enumerate(avails):
            mask[j, list(av)] = True
        logits = model.pi(z).masked_fill(~mask, -1e9)
        logp_all = Fn.log_softmax(logits, -1)
        if mode == "sample":
            with torch.no_grad():
                probs = logp_all.exp()
                if isinstance(rng, CounterRNG):  # extended-07: u keyed by (seed, episode world seed, step)
                    u = torch.tensor([rng.u(items[i][2], len(ep_steps[i])) for i in act]).unsqueeze(1)
                else:
                    u = torch.tensor([rng.random() for _ in act]).unsqueeze(1)
                choice = (probs.cumsum(-1) < u).sum(-1).clamp(max=pw.N_ACTIONS - 1)
            chosen = []
            for j, c in enumerate(choice.tolist()):
                if not mask[j, c]:  # numerical edge: fall back to last available
                    c = max(avails[j])
                chosen.append(c)
        elif mode == "greedy":
            chosen = logits.argmax(-1).tolist()
        else:  # teacher: deterministic representative of pi*
            chosen = [items[i][1].pi_star(eps[i].state) for i in act]
        rec = {"idx": idx, "mask": mask, "logp_all": logp_all, "z": z, "chosen": torch.tensor(chosen)}
        if aux_rec is not None:
            rec.update(aux_rec)
        if need_labels:
            Qt = torch.zeros(len(act), pw.N_ACTIONS)
            opt = torch.zeros(len(act), pw.N_ACTIONS, dtype=torch.bool)
            stg = torch.zeros(len(act), N_STAGES, dtype=torch.bool)
            dep = torch.zeros(len(act), N_DEP)
            sw = torch.zeros(len(act))
            swm = torch.zeros(len(act))
            vstar = torch.zeros(len(act))
            for j, i in enumerate(act):
                s = items[i][1]
                st = eps[i].state
                q = s.q_values(st)
                for a_, v_ in q.items():
                    Qt[j, a_] = v_
                for a_ in s.opt_set(st):
                    opt[j, a_] = True
                for g in pw.stage_set(s, st):
                    stg[j, g] = True
                dep[j] = torch.tensor(pw.dependency_bits(st))
                vstar[j] = s.value(st)
                if prev_state[i] is not None and prev_act[i] not in pw.TERMINAL:
                    swm[j] = 1.0
                    sw[j] = float(pw.switch_needed(s, prev_state[i], prev_act[i], st))
            rec.update(Q=Qt, opt=opt, stage=stg, dep=dep, sw=sw, swm=swm, vstar=vstar)
        for j, i in enumerate(act):
            ep = eps[i]
            st = ep.state
            a_ = chosen[j]
            nledger = len(ep.ledger)
            succ0 = ep.successes
            r = ep.step(a_)
            reward = pw_reward(ep, nledger, succ0)
            info = {"state": st, "a": a_, "rec": r, "next": ep.state, "reward": reward, "step_row": len(steps),
                    "row": j}
            ep_steps[i].append(info)
            prev[i] = r
            prev_state[i], prev_act[i] = st, a_
        steps.append(rec)
    return eps, steps, ep_steps


def pw_reward(ep, nledger, succ0):
    return ep.cfg.R * (ep.successes - succ0) - sum(x for _, _, x in ep.ledger[nledger:])


def attach_case_labels(items, ep_steps, steps):
    """Case type of each step (the agent's own action) and prev-case targets for the next step."""
    for i, info_list in enumerate(ep_steps):
        s = items[i][1]
        for n, info in enumerate(info_list):
            a, o, e, _ = info["rec"]
            info["case"] = pw.case_type(s, info["state"], a, o, e, info["next"])
    for rec in steps:
        rec["prev_case"] = torch.full((len(rec["idx"]),), -1, dtype=torch.long)
    for i, info_list in enumerate(ep_steps):
        for n in range(1, len(info_list)):
            prev_info, info = info_list[n - 1], info_list[n]
            if prev_info["a"] in pw.TERMINAL:
                continue
            steps[info["step_row"]]["prev_case"][info["row"]] = CASE_IDX[prev_info["case"]]


def losses(model, items, eps, steps, ep_steps, w):
    R = 100.0
    # returns-to-go (scaled by 1/R)
    G = {}
    for i, info_list in enumerate(ep_steps):
        g = 0.0
        for info in reversed(info_list):
            g += info["reward"]
            G[(info["step_row"], info["row"])] = g / R
    out = {}
    pl, vl, il, dl, sl, ql, ent = [], [], [], [], [], [], []
    for t, rec in enumerate(steps):
        n = len(rec["idx"])
        Gt = torch.tensor([G[(t, j)] for j in range(n)])
        z = rec["z"]
        v = model.v(z).squeeze(-1)
        logp_all = rec["logp_all"]
        lp = logp_all.gather(1, rec["chosen"].unsqueeze(1)).squeeze(1)
        adv = (Gt - v).detach()
        pl.append(-(lp * adv))
        vl.append((v - Gt) ** 2)
        p = logp_all.exp()
        ent.append(-(p * logp_all.masked_fill(~rec["mask"], 0.0)).sum(-1))
        # L1: set-valued CE  -log sum_{a in A*} pi(a)
        il.append(-torch.logsumexp(logp_all.masked_fill(~rec["opt"], -1e9), -1))
        # L2: stage set-CE + dependency BCE
        slog = Fn.log_softmax(model.stage(z), -1)
        stage_ce = -torch.logsumexp(slog.masked_fill(~rec["stage"], -1e9), -1)
        dep_bce = Fn.binary_cross_entropy_with_logits(model.dep(z), rec["dep"], reduction="none").mean(-1)
        dl.append(stage_ce + dep_bce)
        # L3: switch BCE (masked) + prev-case CE (masked)
        swl = Fn.binary_cross_entropy_with_logits(model.switch(z).squeeze(-1), rec["sw"], reduction="none") * rec["swm"]
        pc = rec["prev_case"]
        cm = (pc >= 0).float()
        cl = Fn.cross_entropy(model.case(z), pc.clamp(min=0), reduction="none") * cm
        sl.append(swl + cl)
        # L4: Q* regression over available actions (scaled 1/R), Huber
        qd = Fn.smooth_l1_loss(model.q(z), rec["Q"] / R, reduction="none") * rec["mask"].float()
        ql.append(qd.sum(-1) / rec["mask"].float().sum(-1))
    cat = lambda xs: torch.cat(xs).mean()
    out = {"rl": cat(pl) + 0.5 * cat(vl) - 0.01 * cat(ent), "imit": cat(il), "dep": cat(dl), "switch": cat(sl),
           "q": cat(ql), "value_mse": cat(vl).detach(), "entropy": cat(ent).detach()}
    total = sum(w[k] * out[k] for k in ("rl", "imit", "dep", "switch", "q"))
    if "aux" in steps[0]:  # extended-06 LEARNED-FACTORIZED: factor regression (MSE, mean over factors and steps)
        out["aux"] = aux_mse(steps, w)
        total = total + w.get("aux", 0.0) * out["aux"]
    return total, out


def aux_mse(steps, w):
    """Factor regression: mean over the 23 coordinates and all steps (extended-06).  extended-07 optional per-group
    coordinate weights w['aux_cw'] (a 23-vector; mean over coordinates of w_c * err_c^2); absent = historical."""
    cw = w.get("aux_cw")
    if cw is None:
        return torch.cat([((rec["aux"] - rec["aux_t"]) ** 2).mean(-1) for rec in steps]).mean()
    return torch.cat([(((rec["aux"] - rec["aux_t"]) ** 2) * cw).mean(-1) for rec in steps]).mean()


class CounterRNG:
    """extended-07 counter-based action sampling: the uniform for decision t of the episode with world seed ws is
    random.Random('e07-act:<seed>:<ws>:<t>').random().  Episodes are identified by their (unique) training world seed,
    so every arm of a seed sees the same uniform at the same (episode, step) whatever happened in other episodes."""

    def __init__(self, seed):
        self.seed = seed

    def u(self, ws, t):
        return random.Random(f"e07-act:{self.seed}:{ws}:{t}").random()


def cmd_train(a):
    torch.set_num_threads(1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.process_time()
    pool = load_pool(a.labels, getattr(a, "train_split", "train"))
    if getattr(a, "train_n", None):  # extended-06: the first N configurations (B0 = N of the 2N base pool)
        if a.train_n > len(pool):
            raise SystemExit(f"--train-n {a.train_n} > pool size {len(pool)}")
        pool = pool[:a.train_n]
    t_load = time.process_time() - t_start
    p2 = p2_setup(a)  # extended-07 Phase 2 options (None under every historical invocation)
    w = RUNG_WEIGHTS[a.rung]
    if getattr(a, "factor_mode", None) in ("learned", "sr"):
        w = dict(w, aux=a.aux_weight)
    if p2 is not None and p2["aux_cw"] is not None:
        w = dict(w, aux_cw=torch.tensor(p2["aux_cw"]))
    public_extra = pw6.PUBLIC_EXTRA6 if isinstance(pool[0][1], pw6.Config6) else 0
    arch = getattr(a, "arch", "flat")
    if arch == "modular" and getattr(a, "match_params", True):  # BX3: hidden width matched to B0's parameters
        a.hidden = matched_hidden(arch, getattr(a, "inputs", "public"))
    torch.manual_seed(1000 + a.seed)  # same init across rungs for a given seed
    model = ProbeNet(a.hidden, own_value=a.own_value, inputs=getattr(a, "inputs", "public"), arch=arch,
                     public_extra=public_extra, factor_mode=getattr(a, "factor_mode", None),
                     **({} if p2 is None else p2["sr_kwargs"]))
    if p2 is not None:
        p2_init(a, p2, model)  # --init-from copy, init sha256, frozen trunk, R0 constant
    # B1 parameters (everything except v_own): optimizer and gradient clipping see exactly these, so v_own can
    # change neither the Adam state nor the clip coefficient of the policy/trunk.
    main_params = [p for n, p in model.named_parameters() if not n.startswith("v_own.")]
    if p2 is None or p2["clip_mode"] == "global":
        if p2 is not None:
            main_params = [p for p in main_params if p.requires_grad]
        opt = torch.optim.Adam(main_params, lr=a.lr)
        opt_groups = [(opt, main_params, "all")]
    else:  # extended-07 split: actor group A and aux predictor group X, separate Adam + separate clip
        grp_a = [p for n, p in model.named_parameters()
                 if not n.startswith("v_own.") and not n.startswith("aux.") and p.requires_grad]
        grp_x = [p for n, p in model.named_parameters() if n.startswith("aux.")]
        opt_groups = [(torch.optim.Adam(grp_a, lr=a.lr), grp_a, "A")]
        if grp_x:
            opt_groups.append((torch.optim.Adam(grp_x, lr=a.lr), grp_x, "X"))
        opt = None
    own = None
    if a.own_value:
        own = {"opt": torch.optim.Adam(model.v_own.parameters(), lr=a.lr), "rng": random.Random(11_000 + a.seed),
               "worlds": 0, "steps": 0, "cpu_s": 0.0, "log": []}
    data_rng = random.Random(7_000 + a.seed)  # same config/world stream across rungs for a given seed
    act_rng = random.Random(9_000 + a.seed)
    if p2 is not None and p2["action_rng"] == "counter":
        act_rng = CounterRNG(a.seed)
    log = []
    world_counter = 0
    if p2 is not None and p2["bank"] is not None:  # extended-07 stage 1: common valid-history bank
        bank_stage(a, p2, model, w, opt_groups, log, t_start)
    t_onpolicy = time.process_time()
    for upd in range(a.updates):
        items = []
        for _ in range(a.batch):
            idx, cfg, s = pool[data_rng.randrange(len(pool))]
            # training worlds: a seed range disjoint from every eval split's world seeds (< 5e9)
            items.append((cfg, s, TRAIN_WORLD_BASE + a.seed * 100_000_000 + world_counter))
            world_counter += 1
        eps, steps, ep_steps = run_batch(model, items, "sample", act_rng)
        attach_case_labels(items, ep_steps, steps)
        total, parts = losses(model, items, eps, steps, ep_steps, w)
        if p2 is None:
            opt.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(main_params, 1.0)
            opt.step()
        else:
            gn = p2_step(opt_groups, total)
        if own is not None and (upd + 1) % a.own_every == 0:
            own_value_update(model, pool, a, own, upd)
        if upd % a.log_every == 0 or upd == a.updates - 1:
            U = sum(e.utility for e in eps) / len(eps)
            Vs = sum(s.value(pw.initial_state(c)) for c, s, _ in items) / len(items)
            row = {"update": upd, "U": U, "V_star": Vs, "regret": Vs - U,
                   **{k: float(v.detach()) for k, v in parts.items()}, "cpu_s": time.process_time() - t_start}
            if p2 is not None:
                row.update(stage="onpolicy", **gn)
            log.append(row)
            print(json.dumps(row), flush=True)
    torch.save(model.state_dict(), out / "model.pt")
    meta = {"rung": a.rung, "seed": a.seed, "weights": w, "updates": a.updates, "batch": a.batch,
            "episodes": a.updates * a.batch, "lr": a.lr, "hidden": a.hidden, "params": n_params(model),
            "in_dim": IN_DIM, "cpu_s_total": time.process_time() - t_start, "cpu_s_label_load": t_load,
            "version": pw.VERSION, "continuation": pw.CONTINUATION, "eps": pw.EPS}
    if getattr(a, "train_split", "train") != "train":  # extended-05 keys: absent under defaults
        meta["train_split"] = a.train_split
    if getattr(a, "inputs", "public") != "public":
        meta["inputs"] = a.inputs
        meta["in_dim"] = IN_DIM + supplied_dim(a.inputs)
        meta["supplied_state"] = {"kind": a.inputs, "features": (list(pw6.FACTOR_FEATURES) if a.inputs == "factors6"
                                  else list(pw5.BX2_FEATURES[:pw5.supplied_dim(a.inputs)])),
                                  "label": "supplied (deterministic public computation), not learned"}

    if arch != "flat":
        meta["arch"] = arch
        meta["modular"] = {"width": MODULAR_WIDTH, "flag_param_cols": FLAG_PARAM_COLS, "flag_bit_col0": FLAG_BIT_COL0,
                           "params_target": B1_PARAMS, "hidden_matched": a.hidden,
                           "form": "tanh(W x + sum_f flag_f * MLP_f([param_f, x])) -> GRU; same public inputs"}
    if public_extra:  # extended-06 keys: absent for probeworld-v1 pools
        meta["public_extra"] = public_extra
        meta["env"] = pw6.VERSION
        meta["in_dim"] = IN_DIM + public_extra + supplied_dim(getattr(a, "inputs", "public"))
        meta["train_n"] = len(pool)
        meta["train_combo_counts"] = {k: v for k, v in sorted(_combo_counts([(i, c, None, None) for i, c, _ in pool]).items())}
    if arch == "fuse":
        meta["fuse"] = {"factor_mode": model.factor_mode, "features": list(pw6.FACTOR_FEATURES),
                        "aux_weight": w.get("aux", 0.0), "stop_gradient": model.factor_mode == "learned",
                        "form": "heads read tanh(W [z; phi]); phi = supplied factors | sg(aux(z)) | 0"}
    if p2 is not None:  # extended-07 keys: absent under every historical invocation
        p2_meta(a, p2, model, meta, t_onpolicy)
    if own is not None:  # key absent when off (B1 train_meta unchanged)
        meta["own_value"] = {"continuation": OWN_VALUE_CONTINUATION, "every": a.own_every,
                             "episodes_per_collection": a.own_episodes, "collections": own["steps"],
                             "greedy_episodes_total": own["worlds"], "stop_gradient": True,
                             "loss": "MSE(v_own(z.detach()), G_own/R), one Adam step per collection", "lr": a.lr,
                             "world_seed_base": TRAIN_WORLD_BASE + a.seed * 100_000_000 + OWN_WORLD_OFFSET,
                             "config_rng_seed": 11_000 + a.seed, "pool": "train", "cpu_s": own["cpu_s"]}
        (out / "own_value_log.json").write_text(json.dumps(own["log"]))
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    (out / "train_log.json").write_text(json.dumps(log))


# ------------------------------------------------------------------------------------------------ extended-07 Phase 2
# Genuine S x R 2x2 (gradient-flow.md 6), common valid-history bank (design Phase 2 / brief 12) and controlled
# consumers (brief 10).  Every option defaults off; p2_setup returns None for every historical invocation, and the
# historical code path is then executed unchanged (tests/test_campaign07_p2.py goldens).

P2_OPTS = ("shape", "read", "clip_mode", "action_rng", "init_from", "init_sha", "bank", "aux_group_weights",
           "phi_contract", "frozen_trunk", "predictor_only", "bank_updates", "finetune")
CONTRACT_JSON = Path(__file__).resolve().parents[2] / "research" / "campaigns" / "extended-07" / "factor-contract.json"
BANK_RNG_OFFSET = 7_500  # bank-episode sampling stream random.Random(7_500 + seed): common to every arm of a seed


def state_sha(sd):
    """sha256 over the sorted (name, raw bytes) of a state_dict: identical tensors <=> identical hash."""
    import hashlib
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(str(tuple(sd[k].shape)).encode())
        h.update(sd[k].detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def contract_groups(path=CONTRACT_JSON):
    """Short group name (G1..G4) -> coordinate indices, from factor-contract.json (P0a)."""
    d = json.loads(Path(path).read_text())
    return {g.split("_")[0]: list(idx) for g, idx in d["groups"].items()}


def aux_coord_weights(spec):
    """['G1=1', 'G4=0.25', ...] -> 23 coordinate weights (unnamed groups keep weight 1)."""
    groups = contract_groups()
    cw = [1.0] * pw6.N_FACTOR_FEATURES
    for item in spec:
        g, v = item.split("=")
        if g not in groups:
            raise SystemExit(f"--aux-group-weights: unknown group {g!r} (have {sorted(groups)})")
        for j in groups[g]:
            cw[j] = float(v)
    return cw


def load_bank(path):
    with open(path, "rb") as f:
        bank = pickle.load(f)
    assert bank["meta"]["format"] == BANK_FORMAT, bank["meta"].get("format")
    return bank


BANK_FORMAT = "e07-history-bank-v1"


def bank_folds(bank, K):
    """Deterministic K-fold assignment of the bank's TRAINING CONFIGURATIONS (never episodes): within each condition
    combination, configurations in index order go to fold j % K (stratified by combination; seed-free, so every seed
    and arm uses the same folds).  Returns {cfg_idx: fold}."""
    by = {}
    for e in bank["episodes"]:
        by.setdefault(e["combo"], set()).add(e["cfg_idx"])
    out = {}
    for combo in sorted(by):
        for j, idx in enumerate(sorted(by[combo])):
            out[idx] = j % K
    return out


def p2_setup(a):
    def unset(k):
        v = getattr(a, k, None)
        return v is None if k in ("shape", "read") else v in (None, False, 0, [])  # --shape 0 is a setting
    if all(unset(k) for k in P2_OPTS):
        return None
    sr = getattr(a, "shape", None) is not None or getattr(a, "read", None) is not None
    if sr:
        if a.shape is None or a.read is None:
            raise SystemExit("--shape and --read go together (the S x R 2x2)")
        if getattr(a, "arch", "flat") != "fuse" or getattr(a, "factor_mode", None) not in (None, "sr"):
            raise SystemExit("--shape/--read need --arch fuse (factor mode 'sr' is implied)")
        a.factor_mode = "sr"
    p2 = {"sr": sr, "sr_kwargs": {"shape": a.shape, "read": a.read} if sr else {},
          "clip_mode": a.clip_mode or ("split" if sr else "global"), "action_rng": a.action_rng or "counter",
          "aux_cw": aux_coord_weights(a.aux_group_weights) if a.aux_group_weights else None,
          "bank": None, "phi_contract": a.phi_contract, "oof": None, "cpu_s_bank": 0.0,
          "noise_seed": a.noise_seed, "mix_p": a.mix_p}
    if a.bank:
        p2["bank"] = load_bank(a.bank)
        p2["bank_sha256"] = _sha_path(a.bank)
        eps = p2["bank"]["episodes"]
        if a.bank_exclude_fold is not None or a.bank_only_fold is not None:
            folds = bank_folds(p2["bank"], a.bank_folds)
            eps = [e for e in eps if (a.bank_exclude_fold is None or folds[e["cfg_idx"]] != a.bank_exclude_fold)
                   and (a.bank_only_fold is None or folds[e["cfg_idx"]] == a.bank_only_fold)]
        if a.bank_combos:
            eps = [e for e in eps if e["combo"] in set(a.bank_combos)]
        if not eps:
            raise SystemExit("no bank episodes left after the fold / combination filters")
        p2["bank_eps"] = eps
        if a.updates > 0 and not a.finetune:
            raise SystemExit("--bank with --updates > 0 needs --finetune (on-policy fine-tuning is separately flagged)")
    elif a.bank_updates or a.predictor_only or a.phi_contract:
        raise SystemExit("--bank-updates / --predictor-only / --phi-contract need --bank")
    if a.predictor_only and (a.updates > 0 or getattr(a, "factor_mode", None) not in ("sr", "learned")):
        raise SystemExit("--predictor-only: bank stage only (--updates 0), on a model with an aux head")
    if a.phi_contract:
        if getattr(a, "inputs", "public") != "factors6" or getattr(a, "arch", "flat") != "fuse":
            raise SystemExit("--phi-contract: consumers use the SUP architecture (--inputs factors6 --arch fuse)")
        if a.phi_contract in ("oof", "mix"):
            if not a.oof:
                raise SystemExit(f"--phi-contract {a.phi_contract} needs --oof FILE (campaign07_p2.py oof)")
            p2["oof"] = torch.load(a.oof)
            p2["oof_sha256"] = _sha_path(a.oof)
            assert p2["oof"]["bank_sha256"] == p2["bank_sha256"], "oof file was made from a different bank"
            if a.updates > 0:
                raise SystemExit("on-policy fine-tuning of oof / mix consumers is not defined (it would read exact)")
    return p2


def _sha_path(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def p2_init(a, p2, model):
    """Common initialization: the model was constructed under torch.manual_seed(1000 + seed) with the identical module
    list in all four S x R arms; --init-from copies a saved reference initial state_dict (strict: same keys and
    shapes), --init-sha asserts its hash.  Then the optional frozen trunk and the R0 constant."""
    p2["init_sha256_constructed"] = state_sha(model.state_dict())
    if a.init_from:
        ref = torch.load(a.init_from)
        own = model.state_dict()
        if set(ref) != set(own) or any(tuple(ref[k].shape) != tuple(own[k].shape) for k in own):
            raise SystemExit(f"--init-from {a.init_from}: keys/shapes differ from this arm's model")
        model.load_state_dict(ref)
    p2["init_sha256"] = state_sha(model.state_dict())
    if a.init_sha and a.init_sha != p2["init_sha256"]:
        raise SystemExit(f"--init-sha mismatch: {p2['init_sha256']} != {a.init_sha}")
    if a.frozen_trunk:  # consumer variant: inp/gru/trunk loaded from a trained run and frozen
        src = torch.load(Path(a.frozen_trunk) / "model.pt")
        own = model.state_dict()
        keys = [k for k in own if k.split(".")[0] in ("inp", "gru", "trunk")]
        for k in keys:
            if tuple(src[k].shape) != tuple(own[k].shape):
                raise SystemExit(f"--frozen-trunk: shape mismatch at {k}")
        model.load_state_dict({**own, **{k: src[k] for k in keys}})
        for n, p in model.named_parameters():
            if n.split(".")[0] in ("inp", "gru", "trunk"):
                p.requires_grad_(False)
        p2["frozen_trunk_sha256"] = _sha_path(Path(a.frozen_trunk) / "model.pt")
    if p2["sr"]:
        if a.phi_const == "train_mean":  # sensitivity option: mean exact factor vector over all bank decisions
            if p2["bank"] is None:
                raise SystemExit("--phi-const train_mean needs --bank (mean of the bank's exact factor labels)")
            rows = [r for e in p2["bank"]["episodes"] for r in e["phi"]]
            model.phi_const.copy_(torch.tensor(rows, dtype=torch.float64).mean(0).float())
        p2["phi_const"] = model.phi_const.tolist()


def p2_step(opt_groups, total):
    """One optimizer step.  'all' (global): one Adam + one clip over every parameter (historical coupling).  Split:
    group A (actor: everything except aux.*) and group X (aux.*) each have their own Adam and their own
    clip_grad_norm_(., 1.0), so a disconnected auxiliary loss changes neither A's clip coefficient nor A's moments."""
    for o, _, _ in opt_groups:
        o.zero_grad()
    total.backward()
    gn = {}
    for _, ps, name in opt_groups:
        gn[f"gnorm_{name}"] = float(nn.utils.clip_grad_norm_(ps, 1.0))
    for o, _, _ in opt_groups:
        o.step()
    return gn


def replay_bank_batch(model, sel, phis=None):
    """Teacher-forced replay of stored public histories (the bank's own actions, never the model's): inputs are
    rebuilt exactly as run_batch builds them (public vector, previous visible record, public mask, queries_done / k),
    plus, for the SUP-architecture consumers, the contract's factor vector at the input tail.  Returns per-step records
    like run_batch (idx, mask, logp_all, z, opt, G, aux, aux_t)."""
    B = len(sel)
    h = torch.zeros(B, model.gru.hidden_size)
    prev = [None] * B
    tail = getattr(model, "inputs", "public") == "factors6"
    if tail and phis is None:
        raise ValueError("a factors6 consumer needs the contract's factor vectors")
    assert getattr(model, "inputs", "public") in ("public", "factors6")
    has_aux = getattr(model, "factor_mode", None) in ("learned", "sr")
    steps = []
    for t in range(max(len(e["hist"]) for e in sel)):
        act = [i for i in range(B) if t < len(sel[i]["hist"])]
        x = torch.tensor([encode(sel[i]["vec"], prev[i], sel[i]["avail"][t], sel[i]["q"][t] / sel[i]["k"])
                          + (list(phis[i][t]) if tail else []) for i in act])
        idx = torch.tensor(act)
        hn, z = model.step(x, h[idx])
        h = h.index_copy(0, idx, hn)
        mask = torch.zeros(len(act), pw.N_ACTIONS, dtype=torch.bool)
        opt = torch.zeros(len(act), pw.N_ACTIONS, dtype=torch.bool)
        for j, i in enumerate(act):
            mask[j, list(sel[i]["avail"][t])] = True
            opt[j, list(sel[i]["opt"][t])] = True
        logits = model.pi(z).masked_fill(~mask, -1e9)
        rec = {"idx": idx, "mask": mask, "logp_all": Fn.log_softmax(logits, -1), "z": z, "opt": opt,
               "G": torch.tensor([sel[i]["G"][t] for i in act])}
        if has_aux:
            rec["aux"] = model.last_aux
            rec["aux_t"] = torch.tensor([sel[i]["phi"][t] for i in act])
        steps.append(rec)
        for i in act:
            prev[i] = tuple(sel[i]["hist"][t])
    return steps


def bank_losses(model, steps, w, value_weight=0.0, predictor_only=False):
    """Bank stage objective: set-valued imitation of the exact eps-optimal set at every stored decision (RL is off:
    the stored actions are not the model's) + the auxiliary factor MSE (models with an aux head) + optionally the
    value MSE toward the behaviour source's return-to-go.  predictor_only: the aux MSE alone (K-fold predictors)."""
    cat = lambda xs: torch.cat(xs).mean()  # noqa: E731
    il = [-torch.logsumexp(rec["logp_all"].masked_fill(~rec["opt"], -1e9), -1) for rec in steps]
    out = {"imit": cat(il)}
    total = 0.0 if predictor_only else w["imit"] * out["imit"]
    if value_weight and not predictor_only:
        out["value_mse"] = cat([(model.v(rec["z"]).squeeze(-1) - rec["G"]) ** 2 for rec in steps])
        total = total + value_weight * out["value_mse"]
    if "aux" in steps[0]:
        out["aux"] = aux_mse(steps, w)
        total = total + w.get("aux", 0.0) * out["aux"]
    return total, out


def contract_phis(p2, sel, seed, upd):
    """Consumer input contract for a sampled batch: exact labels | out-of-fold predictions | declared mixture (each
    sampled episode is noisy with probability mix_p: exact + sigma_c * N(0, 1) per decision and coordinate, sigma_c =
    the out-of-fold predictor RMS error of coordinate c; deterministic in (noise seed, seed, update, slot))."""
    c = p2["phi_contract"]
    if c is None:
        return None
    if c == "exact":
        return [e["phi"] for e in sel]
    if c == "oof":
        return [p2["oof"]["preds"][e["id"]] for e in sel]
    sig = p2["oof"]["err_rms"]
    out = []
    for slot, e in enumerate(sel):
        r = random.Random(f"e07-mix:{p2['noise_seed']}:{seed}:{upd}:{slot}")
        if r.random() < p2["mix_p"]:
            out.append([[x + s * r.gauss(0.0, 1.0) for x, s in zip(row, sig)] for row in e["phi"]])
        else:
            out.append(e["phi"])
    return out


def bank_stage(a, p2, model, w, opt_groups, log, t_start):
    t0 = time.process_time()
    eps = p2["bank_eps"]
    rng = random.Random(BANK_RNG_OFFSET + a.seed)  # identical episode stream for every arm of a seed
    n_dec = 0
    for bu in range(a.bank_updates):
        sel = [eps[rng.randrange(len(eps))] for _ in range(a.batch)]
        steps = replay_bank_batch(model, sel, contract_phis(p2, sel, a.seed, bu))
        n_dec += sum(len(e["hist"]) for e in sel)
        total, parts = bank_losses(model, steps, w, a.bank_value_weight, a.predictor_only)
        gn = p2_step(opt_groups, total)
        if bu % a.log_every == 0 or bu == a.bank_updates - 1:
            row = {"stage": "bank", "update": bu, **{k: float(v.detach()) for k, v in parts.items()}, **gn,
                   "cpu_s": time.process_time() - t_start}
            log.append(row)
            print(json.dumps(row), flush=True)
    p2["cpu_s_bank"] = time.process_time() - t0
    p2["bank_decisions_seen"] = n_dec


def p2_meta(a, p2, model, meta, t_onpolicy):
    info = {"clip_mode": p2["clip_mode"], "action_rng": p2["action_rng"], "init_from": a.init_from,
            "init_sha256": p2["init_sha256"], "init_sha256_constructed": p2["init_sha256_constructed"],
            "aux_coord_weights": p2["aux_cw"], "cpu_s_bank": p2["cpu_s_bank"],
            "cpu_s_onpolicy": time.process_time() - t_onpolicy, "onpolicy_updates": a.updates}
    if p2["bank"] is not None:
        info["bank"] = {"path": a.bank, "sha256": p2["bank_sha256"], "updates": a.bank_updates, "batch": a.batch,
                        "episodes_available": len(p2["bank_eps"]), "episodes_total": len(p2["bank"]["episodes"]),
                        "folds": a.bank_folds, "exclude_fold": a.bank_exclude_fold, "only_fold": a.bank_only_fold,
                        "combos": a.bank_combos, "value_weight": a.bank_value_weight,
                        "predictor_only": bool(a.predictor_only), "finetune": bool(a.finetune),
                        "decisions_seen": p2.get("bank_decisions_seen"), "sampling_rng": BANK_RNG_OFFSET + a.seed,
                        "loss": ("aux MSE only" if a.predictor_only else "set-valued imitation"
                                 + (" + aux MSE" if hasattr(model, "aux") else "")
                                 + (f" + {a.bank_value_weight} value MSE" if a.bank_value_weight else "")),
                        "bank_meta": {k: v for k, v in p2["bank"]["meta"].items() if k != "episode_index"}}
    if p2["phi_contract"]:
        info["consumer"] = {"contract": p2["phi_contract"], "oof": a.oof, "oof_sha256": p2.get("oof_sha256"),
                            "mix_p": a.mix_p if p2["phi_contract"] == "mix" else None,
                            "noise_seed": a.noise_seed if p2["phi_contract"] == "mix" else None,
                            "frozen_trunk": a.frozen_trunk, "frozen_trunk_sha256": p2.get("frozen_trunk_sha256"),
                            "arch": "SUP architecture (factors enter the fusion layer only)"}
    elif a.frozen_trunk:
        info["frozen_trunk"] = a.frozen_trunk
    meta["p2"] = info
    if p2["sr"]:
        meta["fuse"].update(shape=int(model.sr_shape), read=int(model.sr_read), phi_const_mode=a.phi_const,
                            phi_const=p2["phi_const"], stop_gradient=True, arm=f"S{int(model.sr_shape)}R{int(model.sr_read)}",
                            form="pred = aux(z if S1 else sg(z)); heads read tanh(W [z; sg(pred) if R1 else phi_const])")


def build_model_from_meta(meta):
    """ProbeNet for a saved run (historical runs: the identical constructor call as before; extended-07 'sr' runs:
    shape/read and the R0 constant restored from train_meta)."""
    fz = meta.get("fuse", {})
    kw = {"shape": fz["shape"], "read": fz["read"]} if fz.get("factor_mode") == "sr" else {}
    model = ProbeNet(meta["hidden"], own_value="own_value" in meta, inputs=meta.get("inputs", "public"),
                     arch=meta.get("arch", "flat"), public_extra=meta.get("public_extra", 0),
                     factor_mode=fz.get("factor_mode"), **kw)
    if kw:
        model.phi_const.copy_(torch.tensor(fz["phi_const"]))
    return model


def own_returns(ep_steps, R=100.0):
    """{(step_row, row): realized return-to-go / R} of the rollout's own continuation."""
    G = {}
    for info_list in ep_steps:
        g = 0.0
        for info in reversed(info_list):
            g += info["reward"]
            G[(info["step_row"], info["row"])] = g / R
    return G


def own_value_update(model, pool, a, own, upd):
    """protocol-B2: collect a.own_episodes free-running GREEDY episodes of the current model on train-pool configs
    (own config RNG and world-seed range; the B1 data/action RNGs are untouched) and take one Adam step of v_own
    toward their realized return-to-go.  Stop-gradient: v_own sees z.detach(), so the trunk/policy never receive
    its gradient.  Labels are not computed (need_labels=False): the target is the model's own return only."""
    t0 = time.process_time()
    items = []
    for _ in range(a.own_episodes):
        idx, cfg, s = pool[own["rng"].randrange(len(pool))]
        items.append((cfg, s, TRAIN_WORLD_BASE + a.seed * 100_000_000 + OWN_WORLD_OFFSET + own["worlds"]))
        own["worlds"] += 1
    with torch.no_grad():
        eps, steps, ep_steps = run_batch(model, items, "greedy", need_labels=False)
    G = own_returns(ep_steps)
    preds, targets = [], []
    for t, rec in enumerate(steps):
        preds.append(model.v_own(rec["z"].detach()).squeeze(-1))
        targets.append(torch.tensor([G[(t, j)] for j in range(len(rec["idx"]))]))
    loss = ((torch.cat(preds) - torch.cat(targets)) ** 2).mean()
    own["opt"].zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.v_own.parameters(), 1.0)
    own["opt"].step()
    own["steps"] += 1
    own["cpu_s"] += time.process_time() - t0
    if upd % a.log_every < a.own_every or upd == a.updates - 1:
        U = sum(e.utility for e in eps) / len(eps)
        Vs = sum(s.value(pw.initial_state(c)) for c, s, _ in items) / len(items)
        row = {"update": upd, "own_greedy_U": U, "own_greedy_regret": Vs - U, "v_own_mse": float(loss.detach()),
               "cpu_s_own": own["cpu_s"]}
        own["log"].append(row)
        print(json.dumps(row), flush=True)


# ------------------------------------------------------------------------------------------------ eval

def episode_metrics(items, eps, ep_steps, steps=None, model=None):
    rows = []
    for i, ep in enumerate(eps):
        cfg, s, _ = items[i]
        s0 = pw.initial_state(cfg)
        gap = 0.0
        cases = {"a": 0, "b": 0, "c": 0, "d": 0}
        sw = {f"{j}_{c}": 0 for j in ("just", "unjust") for c in "abcd"}
        built = 0
        prev = None
        first = {"first_probe": None, "probe_eps_opt": None, "probe_unique_opt": None}
        for info in ep_steps[i]:
            st, a = info["state"], info["a"]
            q = s.q_values(st)
            if first["first_probe"] is None:  # B-H4 (protocol-B1): first decision of the episode
                v0 = s.value(st)
                opt = {b for b, qb in q.items() if qb >= v0 - pw.EPS}
                first = {"first_probe": int(a == pw.A_PROBE), "probe_eps_opt": int(pw.A_PROBE in opt),
                         "probe_unique_opt": int(opt == {pw.A_PROBE})}
            gap += s.value(st) - q[a]
            case = info.get("case")
            if case is None:
                ao = info["rec"]
                case = pw.case_type(s, st, a, ao[1], ao[2], info["next"])
                info["case"] = case
            cases[case] += 1
            if a == pw.A_BUILD:
                built = 1
            if prev is not None and prev["a"] not in pw.TERMINAL:
                if pw.ACTION_STAGE[a] != pw.ACTION_STAGE[prev["a"]]:
                    j = "just" if q[a] >= s.value(st) - pw.EPS else "unjust"
                    sw[f"{j}_{prev['case']}"] += 1
            prev = info
        rows.append({"cfg_k": cfg.k, "rho": cfg.rho(), "rho_eff": _rho_eff(cfg), "flags": list(cfg.flags), "U": ep.utility,
                     "V_star": s.value(s0), "gap_regret": gap, "success": ep.successes / cfg.k,
                     "cost": ep.total_cost, "wrong": ep.wrong, "built": built, "steps": len(ep_steps[i]),
                     "cases": cases, "switches": sw, **first})
    return rows


_RHO_EFF: dict = {}


def _rho_eff(cfg):
    r = _RHO_EFF.get(cfg)
    if r is None:
        r = _RHO_EFF[cfg] = _env(cfg).effective_rho(cfg)
    return r


def value_calibration(model_rows):
    """(pred, target) pairs -> binned reliability (10 quantile bins) and MAE, in units of R."""
    if not model_rows:
        return {}
    pairs = sorted(model_rows)
    nb = 10
    bins = []
    for b in range(nb):
        chunk = pairs[b * len(pairs) // nb:(b + 1) * len(pairs) // nb]
        if not chunk:
            continue
        mp = sum(p for p, _ in chunk) / len(chunk)
        mt = sum(t for _, t in chunk) / len(chunk)
        bins.append({"n": len(chunk), "pred": mp, "target": mt})
    ece = sum(b["n"] * abs(b["pred"] - b["target"]) for b in bins) / len(pairs)
    mae = sum(abs(p - t) for p, t in pairs) / len(pairs)
    return {"reliability_error": ece, "mae": mae, "bins": bins}


def summarize_rows(rows):
    n = len(rows)
    if n == 0:
        return {}
    mean = lambda f: sum(f(r) for r in rows) / n
    def se(f):
        m = mean(f)
        return math.sqrt(sum((f(r) - m) ** 2 for r in rows) / max(n - 1, 1) / n)
    out = {"n": n, "U": mean(lambda r: r["U"]), "V_star": mean(lambda r: r["V_star"]),
           "regret": mean(lambda r: r["V_star"] - r["U"]), "regret_se": se(lambda r: r["V_star"] - r["U"]),
           "gap_regret": mean(lambda r: r["gap_regret"]), "gap_regret_se": se(lambda r: r["gap_regret"]),
           "success": mean(lambda r: r["success"]), "cost": mean(lambda r: r["cost"]),
           "wrong_per_ep": mean(lambda r: r["wrong"]), "build_rate": mean(lambda r: r["built"]),
           "steps": mean(lambda r: r["steps"])}
    def cond_rate(sel):
        sub = [r for r in rows if r.get("first_probe") is not None and sel(r)]
        return (sum(r["first_probe"] for r in sub) / len(sub)) if sub else None, len(sub)
    out["first_probe_rate_when_not_opt"], out["n_probe_not_opt"] = cond_rate(lambda r: not r["probe_eps_opt"])
    out["first_probe_rate_when_unique_opt"], out["n_probe_unique_opt"] = cond_rate(lambda r: r["probe_unique_opt"])
    for c in "abcd":
        out[f"case_{c}"] = mean(lambda r: r["cases"][c])
    for key in rows[0]["switches"]:
        out[f"switch_{key}"] = mean(lambda r: r["switches"][key])
    return out


RHO_BINS = (0.0, 0.5, 1.0, 2.0, 4.0, 1e9)


def rho_curve(rows, ref_rows):
    """Structure-build rate vs rho_k bin (declared and effective) and k (agent vs pi* on the same worlds)."""
    out = {}
    for key in ("rho", "rho_eff"):
        curve = []
        for lo, hi in zip(RHO_BINS[:-1], RHO_BINS[1:]):
            sel = [j for j, r in enumerate(rows) if lo <= r[key] < hi]
            if sel:
                curve.append({"lo": lo, "hi": hi, "n": len(sel),
                              "build_rate": sum(rows[j]["built"] for j in sel) / len(sel),
                              "build_rate_pi_star": sum(ref_rows[j]["built"] for j in sel) / len(sel),
                              "gap_regret": sum(rows[j]["gap_regret"] for j in sel) / len(sel)})
        out["by_" + key] = curve
    by_k = []
    for k in pw.K_VALUES:
        sel = [j for j, r in enumerate(rows) if r["cfg_k"] == k]
        if sel:
            by_k.append({"k": k, "n": len(sel), "build_rate": sum(rows[j]["built"] for j in sel) / len(sel),
                         "build_rate_pi_star": sum(ref_rows[j]["built"] for j in sel) / len(sel)})
    out["by_k"] = by_k
    return out


EVAL_WORLD_OFFSET = 500  # B1 default; Phase F uses a fresh offset (--world-offset)


def eval_items(pool, split, worlds, offset=None):
    base = EVAL_WORLD_OFFSET if offset is None else offset
    return [(cfg, s, split_world_seed(split, idx, base + r)) for idx, cfg, s in pool for r in range(worlds)]


def reference_policy_eval(items, name):
    eps, ep_steps = [], []
    for cfg, s, ws in items:
        ep = _env(cfg).Episode(cfg, ws)
        infos = []
        while not ep.done:
            st = ep.state
            a = REFERENCE_POLICIES[name](s, st, ep.available())
            n0, s0 = len(ep.ledger), ep.successes
            r = ep.step(a)
            infos.append({"state": st, "a": a, "rec": r, "next": ep.state, "reward": pw_reward(ep, n0, s0)})
        eps.append(ep)
        ep_steps.append(infos)
    return eps, ep_steps


def _ref_exact(s, st, av):
    """Fixed rule: exact b2 then commit (or commit_infeasible)."""
    if pw.A_COMMIT in av and st[1][4]:
        return pw.A_COMMIT
    if pw.A_B2 in av:
        return pw.A_B2
    return pw.A_COMMIT_INF if st[1][0][pw.TX] > 0.5 else pw.A_ABSTAIN


def _ref_probe_first(s, st, av):
    """Fixed rule: probe first; if it reports solved commit, else exact b2 then commit."""
    if pw.A_COMMIT in av and st[1][4]:
        return pw.A_COMMIT
    if pw.A_PROBE in av and not st[1][5]:
        return pw.A_PROBE
    return _ref_exact(s, st, av)


REFERENCE_POLICIES = {"pi_star": lambda s, st, av: s.pi_star(st), "fixed_exact_b2": _ref_exact,
                      "fixed_probe_first": _ref_probe_first}


TIE_TOL = 1e-6  # B-LOC deployment class: an eps-optimal action's logit within TIE_TOL of the chosen one's


def decision_rows(items, eps, ep_steps, steps, model, R=100.0):
    """extended-05: per-episode decision records of a rollout (exact labels from the solver in float64; model
    Q-head and policy logits over the available actions).  Used by --episode-rows and by B-LOC."""
    out = []
    heads = []
    for rec in steps:
        logits = model.pi(rec["z"]).masked_fill(~rec["mask"], -1e9)
        heads.append((logits, model.q(rec["z"]) * R))
    for i, info_list in enumerate(ep_steps):
        cfg, s, ws = items[i]
        decs = []
        siq = 0
        for info in info_list:
            t, j, a = info["step_row"], info["row"], info["a"]
            d = pw5.decision_record(s, info["state"], a)
            d["step_in_query"] = siq
            siq = 0 if a in pw.TERMINAL else siq + 1
            lg, qh = heads[t]
            lgl = [float(lg[j, b]) for b in d["avail"]]
            d["qhat"] = [round(float(qh[j, b]), 4) for b in d["avail"]]
            d["logits"] = [round(x, 4) for x in lgl]
            la = float(lg[j, a])
            d["tie_opt"] = any(b != a and float(lg[j, b]) >= la - TIE_TOL for b in d["opt"])
            d["belief"] = list(info["state"][1][0])
            d["Q"] = [round(x, 6) for x in d["Q"]]
            d["rec"] = list(info["rec"])
            decs.append(d)
        ep = eps[i]
        out.append({"world_seed": ws, "k": cfg.k, "combo": pw6.combo_name(cfg.flags), "U": ep.utility,
                    "V_star": s.value(pw.initial_state(cfg)), "success": ep.successes / cfg.k, "wrong": ep.wrong,
                    "cost": ep.total_cost, "decisions": decs})
    return out


@torch.no_grad()
def replay_history(model, cfg, history, upto=None):
    """extended-05 (B-LOC / BO diagnostic): drive `model` along a GIVEN visible history (the history's own
    actions, not the model's) and return, for each step t < upto (default len(history) + 1 while the episode is
    not over), (available actions, masked policy logits, Q-head * R).  Inputs are rebuilt exactly as in run_batch
    from the public config, the previous visible record, the public mask and the model's supplied-state kind."""
    kind = getattr(model, "inputs", "public")
    vec = cfg.public_vector()
    h = torch.zeros(1, model.gru.hidden_size)
    st = pw.initial_state(cfg)
    prev = None
    out = []
    n = len(history) + 1 if upto is None else upto
    for t in range(n):
        if st[0][0] >= cfg.k:
            break
        av = _env(cfg).available(cfg, st)
        x = torch.tensor([encode(vec, prev, av, st[0][0] / cfg.k) + supplied(cfg, st, kind)])
        h, z = model.step(x, h)
        mask = torch.zeros(1, pw.N_ACTIONS, dtype=torch.bool)
        mask[0, list(av)] = True
        logits = model.pi(z).masked_fill(~mask, -1e9)[0]
        out.append((av, logits, model.q(z)[0] * 100.0))
        if t >= len(history):
            break
        a, o, e, rev = history[t]
        st = _env(cfg).advance(cfg, st, a, o, e, rev)
        prev = history[t]
    return out


@torch.no_grad()
def model_eval(model, items, mode, chunk=256, ext=None):
    eps_all, steps_rows, ep_steps_all, calib_v, calib_vstar, calib_q = [], [], [], [], [], []
    has_own = hasattr(model, "v_own")
    b2 = {"v_own_own": [], "v_own_vstar": [], "vstar_own": [], "commit": [], "episode": []}
    tf = {"n": 0, "argmax_in_opt": 0, "q_gap": 0.0}
    for c0 in range(0, len(items), chunk):
        part = items[c0:c0 + chunk]
        eps, steps, ep_steps = run_batch(model, part, mode, need_labels=True)
        # value head: predicted return-to-go vs realized (own policy) and vs V* (pi*)
        heads = []
        for rec in steps:
            logits = model.pi(rec["z"]).masked_fill(~rec["mask"], -1e9)
            heads.append((model.v(rec["z"]).squeeze(-1), model.q(rec["z"]), logits.argmax(-1)))
        vown = [model.v_own(rec["z"]).squeeze(-1) for rec in steps] if has_own else None
        if ext is not None:  # extended-05 --episode-rows (read-only: no RNG, no state touched)
            ext += decision_rows(part, eps, ep_steps, steps, model)
        for i, info_list in enumerate(ep_steps):
            if has_own:
                b2_episode_records(b2, info_list, steps, heads, vown)
            g = 0.0
            for info in reversed(info_list):
                g += info["reward"]
                t, j = info["step_row"], info["row"]
                rec = steps[t]
                vhat, qhat, am = heads[t]
                calib_v.append((float(vhat[j]), g / 100.0))
                calib_vstar.append((float(vhat[j]), float(rec["vstar"][j]) / 100.0))
                a = info["a"]
                calib_q.append((float(qhat[j, a]), float(rec["Q"][j, a]) / 100.0))
                if has_own:
                    b2["v_own_own"].append((float(vown[t][j]), g / 100.0))
                    b2["v_own_vstar"].append((float(vown[t][j]), float(rec["vstar"][j]) / 100.0))
                    b2["vstar_own"].append((float(rec["vstar"][j]) / 100.0, g / 100.0))
                if mode == "teacher":
                    amj = int(am[j])
                    tf["n"] += 1
                    tf["argmax_in_opt"] += int(bool(rec["opt"][j, amj]))
                    tf["q_gap"] += float(rec["vstar"][j] - rec["Q"][j, amj])
        eps_all += eps
        ep_steps_all += ep_steps
    rows = episode_metrics(items, eps_all, ep_steps_all)
    res = {"summary": summarize_rows(rows),
           "value_calibration_own_return": value_calibration(calib_v),
           "value_vs_vstar": value_calibration(calib_vstar),
           "q_head_vs_qstar_taken": value_calibration(calib_q)}
    if has_own:  # protocol-B2 keys; absent for B1 models (eval.json unchanged)
        res["v_own_calibration_own_return"] = value_calibration(b2["v_own_own"])
        res["v_own_vs_vstar"] = value_calibration(b2["v_own_vstar"])
        res["vstar_as_predictor_own_return"] = value_calibration(b2["vstar_own"])  # noise-floor reference
        res["failure_prediction"] = failure_prediction(b2)
    if mode == "teacher":
        res["teacher_forced"] = {"n_steps": tf["n"], "argmax_in_opt_rate": tf["argmax_in_opt"] / max(tf["n"], 1),
                                 "mean_q_gap": tf["q_gap"] / max(tf["n"], 1)}
    return res, rows


B2_PREDICTORS = ("v_own", "v", "q_head_taken", "qstar_taken", "vstar")


def b2_episode_records(b2, info_list, steps, heads, vown):
    """Failure-prediction records (protocol-B2).  Commit level: every commit / commit_infeasible the model takes;
    positive = wrong (outcome O_WRONG); predictors read at the pre-commit state.  Episode level: predictors at the
    episode's first step; positive = the episode has >= 1 wrong commit."""
    def preds(info):
        t, j, a = info["step_row"], info["row"], info["a"]
        rec, (vhat, qhat, _) = steps[t], heads[t]
        return {"v_own": float(vown[t][j]), "v": float(vhat[j]), "q_head_taken": float(qhat[j, a]),
                "qstar_taken": float(rec["Q"][j, a]) / 100.0, "vstar": float(rec["vstar"][j]) / 100.0}
    wrong_any = 0
    for info in info_list:
        if info["a"] in (pw.A_COMMIT, pw.A_COMMIT_INF):
            wrong = int(info["rec"][1] == pw.O_WRONG)
            wrong_any |= wrong
            b2["commit"].append({"wrong": wrong, **preds(info)})
    if info_list:
        b2["episode"].append({"wrong": wrong_any, **preds(info_list[0])})


def auroc(pos_scores, neg_scores):
    """P(score_pos > score_neg) + .5 P(tie); None if a class is empty."""
    if not pos_scores or not neg_scores:
        return None
    neg = sorted(neg_scores)
    tot = 0.0
    for x in pos_scores:
        lo, hi = bisect.bisect_left(neg, x), bisect.bisect_right(neg, x)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(pos_scores) * len(neg))


def failure_prediction(b2):
    """AUROC of LOW predicted value for wrong commits (score = -prediction).  Raw records are returned under
    '_records' (cmd_eval moves them to failure_records.json for the scorer's pooled statistics)."""
    out = {"_records": {}}
    for level in ("commit", "episode"):
        recs = b2[level]
        pos = [r for r in recs if r["wrong"]]
        neg = [r for r in recs if not r["wrong"]]
        out[level] = {"n_pos": len(pos), "n_neg": len(neg),
                      "auroc": {k: auroc([-r[k] for r in pos], [-r[k] for r in neg]) for k in B2_PREDICTORS}}
        out["_records"][level] = recs
    return out


def cmd_eval(a):
    torch.set_num_threads(1)
    run = Path(a.run)
    meta = json.loads((run / "train_meta.json").read_text())
    model = build_model_from_meta(meta)  # extended-07: identical constructor call for every historical run
    model.load_state_dict(torch.load(run / "model.pt"))
    model.eval()
    t0 = time.process_time()
    result = {"rung": meta["rung"], "seed": meta["seed"], "splits": {}}
    failure_records = {}
    ep_rows = [] if getattr(a, "episode_rows", False) else None
    splits = a.splits
    if splits is None:
        splits = default_eval_splits(a)
    if getattr(a, "split_set", "b1") == "b6c":  # B-FACT-C: own output names (never overwrite a run's b6 eval files)
        assert all(s in pw6.SPLITS6C for s in splits), splits
        check_b6c_labels(a.labels, splits, getattr(a, "cf", False))
        a.out_name = getattr(a, "out_name", None) or "eval_b6c.json"
        a.cf_out = getattr(a, "cf_out", None) or "cf_eval_b6c.json"
        result["split_set"] = "b6c"
    split_worlds = dict(x.split("=") for x in (getattr(a, "split_worlds", None) or []))
    for split in splits:
        worlds = int(split_worlds.get(split, a.worlds))
        parts = pool_parts(a.labels, split)
        if parts != [split]:  # extended-05 sized hold pool (sharded labels)
            result["splits"][split] = eval_sharded(model, a, split, parts, worlds, ep_rows)
            print(split, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in
                                     result["splits"][split]["free_running_greedy"]["summary"].items()}), flush=True)
            continue
        pool = load_pool(a.labels, split)
        items = eval_items(pool, split, worlds, getattr(a, 'world_offset', None))
        ref_eps, ref_steps = reference_policy_eval(items, "pi_star")
        ref_rows = episode_metrics(items, ref_eps, ref_steps)
        ext = [] if ep_rows is not None else None
        free, rows = model_eval(model, items, "greedy", ext=ext)
        if ext is not None:
            annotate_rows(ext, rows, ref_rows, ref_eps, pool, split, worlds, load_s0(a.labels, split))
            ep_rows += ext
        free["rho_curve"] = rho_curve(rows, ref_rows)
        teach, _ = model_eval(model, items, "teacher")
        by_combo = {}
        for r in rows:
            key = pw6.combo_name(r["flags"])
            by_combo.setdefault(key, []).append(r)
        free["by_condition"] = {k: summarize_rows(v) for k, v in by_combo.items()}
        for mode, res in (("free_running_greedy", free), ("teacher_forced", teach)):
            if "failure_prediction" in res:
                failure_records.setdefault(split, {})[mode] = res["failure_prediction"].pop("_records")
        result["splits"][split] = {"free_running_greedy": free, "teacher_forced": teach}
        print(split, json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                 for k, v in free["summary"].items()}), flush=True)
    if getattr(a, "cf", False):  # extended-06 balanced counterfactual / intervention records
        cf_eval(model, a, run)
    result["cpu_s"] = time.process_time() - t0
    out_name = getattr(a, "out_name", None) or "eval.json"
    if ep_rows is not None:  # extended-05; file absent under defaults
        import gzip
        weights = meta.get("weights", {})
        head = {"_meta": {"rung": meta["rung"], "seed": meta["seed"], "inputs": meta.get("inputs", "public"),
                          "arch": meta.get("arch", "flat"), "params": meta.get("params"),
                          "train_split": meta.get("train_split", "train"), "q_head_trained": weights.get("q", 0) > 0,
                          "world_offset": getattr(a, "world_offset", None), "eps": pw.EPS, "actions": pw.ACTIONS}}
        if result.get("split_set") == "b6c":  # key absent for every other split set (headers unchanged)
            head["_meta"]["split_set"] = "b6c"
        with gzip.open(run / out_name.replace(".json", "_episodes.jsonl.gz"), "wt") as f:
            f.write(json.dumps(head) + "\n")
            for r in ep_rows:
                f.write(json.dumps(r) + "\n")
    (run / out_name).write_text(json.dumps(result, indent=1))
    if failure_records:  # protocol-B2 only
        (run / "failure_records.json").write_text(json.dumps(failure_records))


def default_eval_splits(a):
    """Eval/references splits when --splits is not given: extended-04 EVAL_SPLITS (b1), the b5 eval splits, or the
    B-XC fresh U+C hold only (b5c)."""
    sset = getattr(a, "split_set", "b1")
    if sset == "b6":
        return list(pw6.B6_EVAL_SPLITS)
    if sset == "b6c":
        return list(pw6.B6C_EVAL_SPLITS)
    return list(pw5.SPLIT_SETS[sset]["eval_splits"]) if sset in pw5.SPLIT_SETS else list(EVAL_SPLITS)


def eps_u(ep):
    return ep.utility


def load_s0(labels_dir, split):
    """extended-05: per-configuration s0 facts of a sized hold pool ({} for ordinary pools)."""
    p = Path(labels_dir) / f"{split}_s0.json"
    return {r["idx"]: r for r in json.loads(p.read_text())} if p.exists() else {}


def annotate_rows(ext, rows, ref_rows, ref_eps, pool, split, worlds, s0map):
    """extended-05 --episode-rows: add split/config/world ids, pi* reference outcomes on the same world, first-decision
    probe facts and (sized holds) the registered flag-sensitivity of the configuration."""
    for j, (r, rr, e) in enumerate(zip(rows, ref_rows, ext)):
        idx = pool[j // worlds][0]
        e.update(split=split, cfg_idx=idx, rep=j % worlds, rho=r["rho"], rho_eff=r["rho_eff"],
                 gap_regret=r["gap_regret"], built=r["built"], pi_star_success=rr["success"],
                 pi_star_U=eps_u(ref_eps[j]), pi_star_built=rr["built"], first_probe=r["first_probe"],
                 probe_eps_opt=r["probe_eps_opt"], probe_unique_opt=r["probe_unique_opt"])
        if idx in s0map and "flag_sensitive" in s0map[idx]:
            e["flag_sensitive"] = s0map[idx]["flag_sensitive"]
        if idx in s0map and "family" in s0map[idx]:  # extended-06 b6 evaluation pools
            e.update(family=s0map[idx]["family"], eligible=s0map[idx]["eligible"])
            for key in ("rel", "joint_flip", "all_any_relevant", "all_regret_relevant"):
                if key in s0map[idx]:
                    e[key] = s0map[idx][key]


def eval_sharded(model, a, split, parts, worlds, ep_rows):
    """extended-05: greedy evaluation of a sharded (sized) hold pool, one shard in memory at a time.  Reports the
    free-running summary / rho curve / by-condition over all shards; teacher-forced and value-calibration keys are
    not computed for sharded pools (the B-X endpoints come from the episode rows)."""
    rows_all, ref_all = [], []
    s0map = load_s0(a.labels, split)
    for part in parts:
        pool = load_pool(a.labels, part)
        items = eval_items(pool, split, worlds, getattr(a, "world_offset", None))
        ref_eps, ref_steps = reference_policy_eval(items, "pi_star")
        ref_rows = episode_metrics(items, ref_eps, ref_steps)
        ext = [] if ep_rows is not None else None
        _, rows = model_eval(model, items, "greedy", ext=ext)
        if ext is not None:
            annotate_rows(ext, rows, ref_rows, ref_eps, pool, split, worlds, s0map)
            ep_rows += ext
        rows_all += rows
        ref_all += ref_rows
        del pool, items
    by_combo = {}
    for r in rows_all:
        by_combo.setdefault(pw6.combo_name(r["flags"]), []).append(r)
    free = {"summary": summarize_rows(rows_all), "rho_curve": rho_curve(rows_all, ref_all),
            "by_condition": {k: summarize_rows(v) for k, v in by_combo.items()},
            "sharded": {"shards": len(parts), "worlds": worlds,
                        "omitted": ["teacher_forced", "value_calibration_own_return", "value_vs_vstar",
                                    "q_head_vs_qstar_taken"]}}
    return {"free_running_greedy": free}


@torch.no_grad()
def model_choice(model, cfg, history):
    """The model's greedy choice after replaying a visible history (None if the history ends the episode)."""
    steps = replay_history(model, cfg, list(history), upto=len(history) + 1)
    if len(steps) <= len(history):
        return None
    av, logits, _ = steps[len(history)]
    return int(logits.argmax())


def cf_eval(model, a, run):
    """extended-06: for every balanced counterfactual set in the labels dir (cf_<family>.json), the model's greedy
    choice at every member x decision type (and the near-miss) after replaying the type's visible history; correct =
    eps-optimal under the member's exact labels.  Writes <run>/cf_eval.json (scored by campaign06_bscore.py)."""
    out = {"_meta": {"eps": pw.EPS, "decision_types": list(pw6.DECISION_TYPES)}, "families": {}}
    for p in sorted(Path(a.labels).glob("cf_*.json")):
        fam = p.stem[3:]
        rows = []
        for cs in json.loads(p.read_text()):
            ent = {"index": cs["index"], "k": cs["k"], "types": {}}
            cfgs = {key: pw6.config_from_dict6(m["config"]) for key, m in cs["members"].items()}
            for h, hist in pw6.DECISION_TYPES.items():
                if h not in cs["types"]:
                    continue
                t = cs["types"][h]
                mem = {}
                for key, m in cs["members"].items():
                    lab = m["labels"][h]
                    if lab is None:
                        continue
                    ch = model_choice(model, cfgs[key], hist)
                    mem[key] = {"a": ch, "ok": ch in lab["opt"], "opt": lab["opt"], "unique": lab["unique"]}
                nm = None
                if t["near_miss"] is not None:
                    ch = model_choice(model, pw6.config_from_dict6(t["near_miss"]["config"]), hist)
                    nm = {"a": ch, "ok": ch in t["near_miss"]["label"]["opt"], "opt": t["near_miss"]["label"]["opt"]}
                ent["types"][h] = {"flip": t["flip"], "unique": t["unique"], "members": mem, "near_miss": nm}
            rows.append(ent)
        out["families"][fam] = rows
        print("cf", fam, len(rows), flush=True)
    (run / (getattr(a, "cf_out", None) or "cf_eval.json")).write_text(json.dumps(out))


def cmd_references(a):
    """Reference policies (pi*, fixed rules) on the same eval worlds."""
    res = {}
    splits = a.splits
    if splits is None:
        splits = default_eval_splits(a)
    split_worlds = dict(x.split("=") for x in (getattr(a, "split_worlds", None) or []))
    for split in splits:
        acc = {name: [] for name in REFERENCE_POLICIES}
        for part in pool_parts(a.labels, split):  # [split] unless a sized (sharded) extended-05 hold pool
            pool = load_pool(a.labels, part)
            items = eval_items(pool, split, int(split_worlds.get(split, a.worlds)), getattr(a, 'world_offset', None))
            for name in REFERENCE_POLICIES:
                eps, steps = reference_policy_eval(items, name)
                acc[name] += episode_metrics(items, eps, steps)
            del pool, items
        res[split] = {}
        for name in REFERENCE_POLICIES:
            rows = acc[name]
            res[split][name] = summarize_rows(rows)
            if name == "pi_star":
                res[split][name]["rho_curve"] = rho_curve(rows, rows)
        print(split, {n: round(v["regret"], 2) for n, v in res[split].items()}, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


def cmd_summarize(a):
    rows = {}
    for d in a.runs:
        p = Path(d) / "eval.json"
        if not p.exists():
            continue
        e = json.loads(p.read_text())
        tm = json.loads((Path(d) / "train_meta.json").read_text())
        rows.setdefault(e["rung"], []).append((e, tm))
    table = {}
    keys = ("regret", "gap_regret", "success", "cost", "U", "V_star", "build_rate", "case_a", "case_b", "case_c",
            "switch_just_b", "switch_unjust_b", "switch_just_c", "switch_unjust_c", "switch_unjust_d")
    for rung, lst in sorted(rows.items()):
        table[rung] = {"seeds": [tm["seed"] for _, tm in lst], "params": lst[0][1]["params"],
                       "train_cpu_s": [tm["cpu_s_total"] for _, tm in lst], "splits": {}}
        for split in lst[0][0]["splits"]:
            per = [e["splits"][split] for e, _ in lst]
            ent = {}
            for key in keys:
                vals = [p["free_running_greedy"]["summary"][key] for p in per]
                ent[key] = {"mean": sum(vals) / len(vals), "per_seed": vals}
            tfv = [p["teacher_forced"]["teacher_forced"]["argmax_in_opt_rate"] for p in per]
            ent["tf_argmax_in_opt"] = {"mean": sum(tfv) / len(tfv), "per_seed": tfv}
            cal = [p["free_running_greedy"]["value_calibration_own_return"]["reliability_error"] for p in per]
            ent["value_reliability_error"] = {"mean": sum(cal) / len(cal), "per_seed": cal}
            curves = [p["free_running_greedy"]["rho_curve"]["by_rho_eff"] for p in per]
            ent["rho_eff_curve"] = [{"lo": c0["lo"], "hi": c0["hi"], "n": c0["n"],
                                     "build_rate": sum(c[j]["build_rate"] for c in curves) / len(curves),
                                     "build_rate_pi_star": c0["build_rate_pi_star"]}
                                    for j, c0 in enumerate(curves[0]) if all(len(c) == len(curves[0]) for c in curves)]
            table[rung]["splits"][split] = ent
    Path(a.out).write_text(json.dumps(table, indent=1))
    for rung, t in table.items():
        print(rung, {s: round(v["gap_regret"]["mean"], 2) for s, v in t["splits"].items()})


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("labels")
    s.add_argument("--out", required=True)
    s.add_argument("--n-train", type=int, default=384)
    s.add_argument("--n-eval", type=int, default=128)
    s.add_argument("--split-set", choices=("b1", "b5", "b5c", "b6", "b6c"), default="b1",
                   help="extended-05: b5 = B-SPLIT pools; b5c = B-XC fresh U+C hold only (b5c_hold_uc); "
                        "extended-06: b6 = probeworld-v3 split table v3; b6c = B-FACT-C fresh S+C+E hold + octets only")
    s.add_argument("--n-hold", type=int, default=400, help="b6: configurations per held-out challenge family")
    s.add_argument("--n-hist", type=int, default=200, help="b6: configurations per historical challenge set")
    s.add_argument("--n-cf", type=int, default=160, help="b6: balanced counterfactual octets per held-out family")
    s.add_argument("--parts", nargs="+", choices=("train", "eval", "cf"), default=None,
                   help="b6: label parts to build (default all; parts can run as parallel jobs into one directory)")
    s.add_argument("--arms", nargs="+", default=None, help="b6: training arms to write (default all)")
    s.add_argument("--hold-relevance", action="store_true",
                   help="b6: exact per-factor relevance records for held-out / historical configurations")
    s.add_argument("--b6-shard", type=int, default=24, help="b6: evaluation pool shard size")
    s.add_argument("--cf-hist", action="store_true", help="b6: also build counterfactual sets for the historical pairs")
    s.add_argument("--n-train-x", type=int, default=768, help="b5: exposure (BX1) training pool size")
    s.add_argument("--hold-target", type=int, default=pw5.HOLD_TARGET_ELIGIBLE,
                   help="b5: new-hold pool = smallest prefix with this many s0-uniquely-probe-optimal configs")
    s.add_argument("--hold-max", type=int, default=pw5.HOLD_MAX_CONFIGS)
    s.add_argument("--hold-target-sensitive", type=int, default=None,
                   help="b5: U+C flag-sensitive eligible target (default registered pw5.HOLD_TARGET_SENSITIVE)")
    s.add_argument("--hold-shard", type=int, default=pw5.HOLD_SHARD)
    s = sub.add_parser("train")
    s.add_argument("--labels", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--rung", choices=RUNGS, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--updates", type=int, default=4000)
    s.add_argument("--batch", type=int, default=64)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--hidden", type=int, default=HIDDEN)
    s.add_argument("--log-every", type=int, default=100)
    s.add_argument("--own-value", action="store_true", help="protocol-B2 v_own head (default off: B1 identical)")
    s.add_argument("--own-every", type=int, default=1, help="collect own greedy rollouts every N updates")
    s.add_argument("--own-episodes", type=int, default=64, help="greedy episodes per collection")
    s.add_argument("--train-split", default="train", help="extended-05: training pool (b5_train / b5x_train)")
    s.add_argument("--inputs", choices=pw5.SUPPLIED_KINDS + ("factors6",), default="public",
                   help="extended-05: supplied public state (belief = BO; bx2 = belief + per-flag features)")
    s.add_argument("--arch", choices=("flat", "modular", "fuse"), default="flat",
                   help="extended-05: modular = BX3 per-flag gated encoders (hidden width matched to B0's parameters); "
                        "extended-06: fuse = factorized arms (heads read a fusion of the trunk and the factors)")
    s.add_argument("--train-n", type=int, default=None, help="extended-06: use the first N configurations of the pool")
    s.add_argument("--factor-mode", choices=("supplied", "learned", "none"), default=None,
                   help="extended-06 --arch fuse: supplied (needs --inputs factors6) | learned | none (raw control)")
    s.add_argument("--aux-weight", type=float, default=1.0, help="extended-06 learned factor loss weight")
    # extended-07 Phase 2 (all default off; gradient-flow.md 6, research/campaigns/extended-07/p2-infra.md)
    s.add_argument("--shape", type=int, choices=(0, 1), default=None,
                   help="extended-07 S x R: S1 = the aux loss updates the trunk/encoder, S0 = aux reads sg(z)")
    s.add_argument("--read", type=int, choices=(0, 1), default=None,
                   help="extended-07 S x R: R1 = the policy reads sg(pred), R0 = the constant phi_const")
    s.add_argument("--phi-const", choices=("zeros", "train_mean"), default="zeros",
                   help="extended-07 R0 constant (default zeros = the RAWF control; train_mean needs --bank)")
    s.add_argument("--clip-mode", choices=("split", "global"), default=None,
                   help="extended-07: split = separate Adam + clip for the aux head (default for S x R); global = "
                        "historical coupling (one Adam, one clip over everything)")
    s.add_argument("--action-rng", choices=("stream", "counter"), default=None,
                   help="extended-07: counter = u(seed, episode world seed, step) (default for every P2 run)")
    s.add_argument("--init-from", default=None, help="extended-07: reference initial state_dict (strict copy)")
    s.add_argument("--init-sha", default=None, help="extended-07: assert the initial state_dict sha256")
    s.add_argument("--aux-group-weights", nargs="*", default=None, metavar="G=W",
                   help="extended-07: per-group aux loss weights (factor-contract groups G1..G4; default all 1)")
    s.add_argument("--bank", default=None, help="extended-07: common valid-history bank (campaign07_p2.py bank)")
    s.add_argument("--bank-updates", type=int, default=0, help="extended-07: bank-stage updates (batch = --batch)")
    s.add_argument("--bank-value-weight", type=float, default=0.0,
                   help="extended-07: value MSE toward the behaviour return-to-go in the bank stage (default off)")
    s.add_argument("--bank-folds", type=int, default=5, help="extended-07: K of the configuration folds")
    s.add_argument("--bank-exclude-fold", type=int, default=None, help="extended-07: train on the other folds")
    s.add_argument("--bank-only-fold", type=int, default=None, help="extended-07: train on this fold only")
    s.add_argument("--bank-combos", nargs="*", default=None, help="extended-07: restrict to these combinations")
    s.add_argument("--finetune", action="store_true",
                   help="extended-07: allow --updates > 0 on-policy fine-tuning after the bank stage")
    s.add_argument("--predictor-only", action="store_true",
                   help="extended-07: bank stage trains the aux MSE only (K-fold / full factor predictors)")
    s.add_argument("--phi-contract", choices=("exact", "oof", "mix"), default=None,
                   help="extended-07 consumer input contract (SUP architecture; bank stage)")
    s.add_argument("--oof", default=None, help="extended-07: out-of-fold prediction file (campaign07_p2.py oof)")
    s.add_argument("--mix-p", type=float, default=0.5, help="extended-07: noisy-episode probability (mix)")
    s.add_argument("--noise-seed", type=int, default=0, help="extended-07: mix-contract noise seed")
    s.add_argument("--frozen-trunk", default=None,
                   help="extended-07 consumer variant: load inp/gru/trunk from RUN and freeze them")
    s = sub.add_parser("eval")
    s.add_argument("--labels", required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--worlds", type=int, default=4)
    s.add_argument("--splits", nargs="+", default=None, help="default: B1 eval splits (or b5 eval splits)")
    s.add_argument("--world-offset", type=int, default=None, help="eval world-seed offset (default 500 = B1)")
    s.add_argument("--split-set", choices=("b1", "b5", "b5c", "b6", "b6c"), default="b1")
    s.add_argument("--episode-rows", action="store_true", help="extended-05: write <out>_episodes.jsonl.gz")
    s.add_argument("--cf", action="store_true", help="extended-06: balanced counterfactual records (cf_eval.json)")
    s.add_argument("--cf-out", default=None, help="extended-06: counterfactual record file name (cf_eval.json)")
    s.add_argument("--out-name", default=None, help="extended-05: eval file name inside the run dir (eval.json)")
    s.add_argument("--split-worlds", nargs="*", default=None, metavar="SPLIT=N",
                   help="extended-05: per-split world count override (sized holds: 1 world per configuration)")
    s = sub.add_parser("references")
    s.add_argument("--labels", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--worlds", type=int, default=4)
    s.add_argument("--splits", nargs="+", default=None)
    s.add_argument("--world-offset", type=int, default=None, help="eval world-seed offset (default 500 = B1)")
    s.add_argument("--split-set", choices=("b1", "b5", "b5c", "b6", "b6c"), default="b1")
    s.add_argument("--split-worlds", nargs="*", default=None, metavar="SPLIT=N")
    s = sub.add_parser("summarize")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    {"labels": cmd_labels, "train": cmd_train, "eval": cmd_eval, "references": cmd_references,
     "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
