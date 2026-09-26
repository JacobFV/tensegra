"""extended-03 P1 metrics/analysis (protocol-P1-metrics.md): synthetic rows, the evaluator-only
step audit on real depworld episodes, and a tiny end-to-end sealed evaluation on CPU."""
from dataclasses import asdict
from functools import partial
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEV = 2_060_000_000  # development seeds only (never sealed)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"research/tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = _load("campaign03_p1_analysis")
C = _load("campaign03_p1_configs")


# --- synthetic row builders ---------------------------------------------------------------

def step(i, kind, need="select", open_=0, reusable=(), pre=None, outcome="success", **extra):
    row = {"step": i, "kind": kind, "key": f"k{i}", "need": need, "outcome": outcome, "work": 0}
    if need is not None:
        row.update(open=open_, reusable=list(reusable), applicable_pre=list(reusable if pre is None else pre),
                   applicable_any=len(reusable))
    row.update(extra)
    return row


def attempt(kind, key, status, deps, reason=None):
    return {"action_kind": kind, "action_key": key, "outcome_status": status, "reason": reason,
            "dependency_versions_at_attempt": deps, "arguments": {}}


def row(seed, success=True, steps=(), attempts=(), uses=(), event=None, work=100, utility=None, targets=None):
    steps = list(steps)
    history = [{"action": {"kind": s["kind"], "arguments": {"target": (targets or {}).get(s["step"])}},
                "feedback": {"status": s["outcome"]}} for s in steps]
    return {"seed": seed, "truncated": False, "outcome": {
        "verified_success": success, "utility": float(success) - .05 if utility is None else utility, "cost": .05,
        "work_units": work, "steps": len(steps), "observations": 0, "travel_distance": 0, "compute_units": 0,
        "calls": 0, "history": history, "reuse_audit": list(uses),
        "p1_audit": {"version": "p1-audit-v1", "steps": steps, "attempts": list(attempts), "event": event}}}


def use_audit(applicable=True, dependency=True, foreign=False):
    return {"applicable_hidden": applicable, "dependency_match_hidden": dependency, "foreign": foreign}


REUSE_STEPS = [step(1, "retrieve", reusable=["h"]),
               step(2, "use_return", reusable=["h"], use={"handle": "h", "as": "select", "reusable": True})]
RECOMPUTE_STEPS = [step(1, "call", reusable=["h"], call_primitive="constrained_subset"),
                   step(2, "use_return", reusable=["h"], use={"handle": "h", "as": "select", "reusable": True})]


# --- per-episode reductions -----------------------------------------------------------------

def test_correct_reuse_windows():
    m = A.episode_metrics(row(1, steps=REUSE_STEPS))
    assert (m["reuse_opportunities"], m["reuse_correct"]) == (1, 1)
    # A solver call for the needed primitive before the use: recomputed, not correct reuse.
    m = A.episode_metrics(row(1, steps=RECOMPUTE_STEPS))
    assert (m["reuse_opportunities"], m["reuse_correct"], m["reuse_recomputed"]) == (1, 0, 1)
    # A call of another primitive does not count against the window.
    other = [step(1, "call", reusable=["h"], call_primitive="csp")] + REUSE_STEPS[1:]
    assert A.episode_metrics(row(1, steps=other))["reuse_correct"] == 1
    # No reusable record anywhere in the window: no opportunity, even if a record is used.
    none = [step(1, "use_return", use={"handle": "h", "as": "select", "reusable": False})]
    assert A.episode_metrics(row(1, steps=none))["reuse_opportunities"] == 0
    # A rejected use of a reusable record is not a correct reuse.
    rejected = [step(1, "use_return", reusable=["h"], outcome="rejected",
                     use={"handle": "h", "as": "select", "reusable": True})]
    assert A.episode_metrics(row(1, steps=rejected))["reuse_correct"] == 0
    # Two windows (new stage opening) are two opportunities; per-stage split.
    two = REUSE_STEPS + [step(3, "think", need="assign", open_=2, reusable=["g"])]
    m = A.episode_metrics(row(1, steps=two))
    assert (m["reuse_opportunities"], m["reuse_correct"], m["reuse_opportunities_assign"]) == (2, 1, 1)
    # Same need, new stage-open step (e.g. revocation) = new window.
    reopened = REUSE_STEPS + [step(3, "think", need="select", open_=3, reusable=["h2"])]
    assert A.episode_metrics(row(1, steps=reopened))["reuse_opportunities"] == 2


def test_identical_retry():
    d0, d1 = {"pending": ["a"]}, {"pending": ["b"]}
    attempts = [attempt("commit_pending", "c", "rejected", d0, "capacity"),
                attempt("commit_pending", "c", "rejected", d0, "capacity"),   # identical retry
                attempt("commit_pending", "c", "rejected", d1, "capacity"),   # deps changed: not identical
                attempt("move", "m", "success", {"p": 0}),
                attempt("move", "m", "success", {"p": 0}),                     # last outcome success
                attempt("call", "x", "timeout", {"d": 1}),
                attempt("call", "x", "timeout", {"d": 1})]                    # calls: all-kinds variant only
    m = A.episode_metrics(row(1, attempts=attempts))
    assert (m["retry_rejections"], m["retry_identical"]) == (3, 1)
    assert (m["retry_attempts"], m["retry_identical_attempts"]) == (5, 1)
    assert (m["retry_rejections_all"], m["retry_identical_all"]) == (5, 2)
    assert m["retry_rejections_commit_pending"] == 3 and m["retry_rejections_move"] == 0
    md = [attempt("commit_pending", "c", "rejected", d0, "missing_dependency")] * 2
    m = A.episode_metrics(row(1, attempts=md))
    assert m["retry_identical"] == 1 and m["retry_rejections_no_missing_dependency"] == 0


def test_invalid_and_stale_use():
    uses = [use_audit(), use_audit(False, True, True), use_audit(False, False), use_audit(False, False, True)]
    m = A.episode_metrics(row(1, uses=uses))
    assert (m["audited_uses"], m["invalid_uses"], m["stale_uses"]) == (4, 3, 2)
    assert (m["invalid_foreign_uses"], m["invalid_own_uses"]) == (2, 1)
    assert m["invalid_use_episode"] == 1 and m["stale_use_episode"] == 1


def test_revision_quality():
    event = {"step": 2, "kind": "slot_closed", "revoked": ["assignment"], "kept": ["selection"],
             "kept_completable": {"selection": True, "assignment": None}, "work_at_event": 40}
    base = [step(1, "commit_pending"), step(2, "commit_assignment", need="assign")]
    good = base + [step(3, "call", need="assign", call_primitive="csp")]
    m = A.episode_metrics(row(1, steps=good, event=event, work=60))
    assert (m["event_episodes"], m["over_revision"], m["revision_good"], m["post_event_work"]) == (1, 0, 1, 20)
    # Recomputing the still-valid selection after the event: over-revision.
    bad = base + [step(3, "call", need="assign", call_primitive="constrained_subset")]
    assert A.episode_metrics(row(1, steps=bad, event=event))["over_revision"] == 1
    # Uncommitting the still-valid selection: over-revision.
    unc = base + [step(3, "uncommit", need="assign")]
    m = A.episode_metrics(row(1, steps=unc, event=event, targets={3: "select"}))
    assert m["over_revision"] == 1 and m["revision_good"] == 0
    # ... unless the kept selection could no longer be completed (hidden check at the event).
    stuck = {**event, "kept_completable": {"selection": False, "assignment": None}}
    assert A.episode_metrics(row(1, steps=unc, event=stuck, targets={3: "select"}))["over_revision"] == 0
    # Failure after the event is never good revision.
    assert A.episode_metrics(row(1, False, steps=good, event=event))["revision_good"] == 0
    assert A.episode_metrics(row(1, steps=base))["event_episodes"] == 0


def test_aggregation_is_pooled_within_condition():
    rows = [A.episode_metrics(row(1, steps=REUSE_STEPS, work=100)),
            A.episode_metrics(row(2, False, steps=RECOMPUTE_STEPS, work=300))]
    agg = A.aggregate(rows)
    assert agg["success"] == .5 and agg["work_per_success"] == 400 and agg["correct_reuse_rate"] == .5
    assert agg["invalid_reuse_rate"] is None and agg["supports"]["episodes"] == 2
    assert A.group_mean({"a": .2, "b": None, "c": .4}, ("a", "b", "c")) == (pytest.approx(.3), ["a", "c"])


# --- end-to-end on a synthetic evaluation directory -----------------------------------------------

def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def synthetic_eval(tmp_path, spec_fn, n=10):
    """spec_fn(identity, condition, i) -> row kwargs. Writes a complete evaluator-shaped directory."""
    prices = {"action_price": .001, "observation_price": .001, "travel_price": .001, "work_price": 2e-4,
              "compute_price": 1e-4}
    arms = [("rl" if e == "rl" else "boot", a, r) for e in ("boot", "rl") for a in A.ARMS for r in A.LINEAGES]
    results = []
    for c in A.CONDITIONS:
        _write(tmp_path / c / "worlds.jsonl.gz", [{"seed": s, "spec": prices} for s in range(n)])
        for ident in arms + [("ref", x) for x in A.REFERENCES]:
            rows = [row(i, **spec_fn(ident, c, i)) for i in range(n)]
            if ident[0] == "ref":
                name, kind, extra = f"reference-{ident[1]}", "supplied_schedule", {}
            else:
                name, kind = f"p1-{ident[0]}-{ident[1]}-r{ident[2]}", "learned"
                extra = {"checkpoint_binding": {"name": name, "endpoint": ident[0], "arm": ident[1], "lineage": ident[2]}}
            art = tmp_path / c / f"{name}.jsonl.gz"
            _write(art, rows)
            results.append({"condition": c, "arm": name, "kind": kind, "artifact": f"{c}/{name}.jsonl.gz",
                            "artifact_sha256": hashlib.sha256(art.read_bytes()).hexdigest(), **extra})
    (tmp_path / "summary.json").write_text(json.dumps({"config_sha256": "x", "results": results, "checkpoints": []}))
    return tmp_path


def test_rules_end_to_end_synthetic(tmp_path):
    def spec(ident, c, i):
        kw = {}
        if ident[0] == "ref":
            return {"utility": .90}
        endpoint, arm, r = ident
        success = True
        if arm == "x1" and c == "iid_f2":
            success = i < 9                                  # .9 in iid_f2, 1.0 in iid_f0 -> mean .95
        if arm == "x1" and c == A.HELDOUT:
            success = i < 8                                  # ratio .8/.95 < .9 -> R5 not supported
        kw["success"] = success
        kw["utility"] = .94 if success else -.05
        if arm == "x2":                                      # reuse emerges: boot recomputes, RL reuses
            kw["steps"] = RECOMPUTE_STEPS if endpoint == "boot" else REUSE_STEPS
            kw["work"] = 200 if endpoint == "boot" else 120
        if arm == "x3":                                      # X3 invalid uses in 2 of 3 lineages
            kw["uses"] = [use_audit(), use_audit(r == 2)]
        elif arm in ("x1", "x4"):
            kw["uses"] = [use_audit(), use_audit()]
        d = {"p": 1}
        kw["attempts"] = [attempt("commit_pending", "c", "rejected", d, "capacity"),
                          attempt("commit_pending", "c", "rejected", {"p": 2}, "capacity")]
        return kw
    result = A.analyze(synthetic_eval(tmp_path, spec), resamples=50)
    R = result["rules"]
    assert R["R1"]["aggregate"]["success_lineages_passing"] == 3 and R["R1"]["supported"]
    assert R["R1"]["per_condition_differs"] == {"iid_f2": False}          # aggregate and per-condition differ
    assert R["R2"]["supported"] and R["R2"]["aggregate"]["lineages"][0]["reuse_gain"] == 1.0
    ci = R["R2"]["aggregate"]["lineages"][0]["reuse_gain_ci95"]
    assert ci == [1.0, 1.0]
    assert R["R3"]["aggregate"]["pairs_passing"] == 2 and R["R3"]["supported"]
    assert R["R3"]["aggregate"]["invalid_reuse"][0]["diff"] == pytest.approx(.5)
    assert not R["R4"]["supported"] and R["R4"]["aggregate"]["retry_pairs_passing"] == 0
    assert not R["R5"]["supported"] and R["R5"]["aggregate"]["lineages"][0]["ratio"] == pytest.approx(.8 / .95)
    assert result["missing"] == [] and "## R2" in A.markdown(result)
    # Tampered artifact is refused.
    art = tmp_path / "iid_f0" / "reference-dep_reuse.jsonl.gz"
    art.write_bytes(gzip.compress(b"{}\n"))
    with pytest.raises(ValueError):
        A.analyze(tmp_path, resamples=0)


def test_r4_success_branch_and_missing_arms(tmp_path):
    def spec(ident, c, i):
        if ident[0] != "ref" and ident[1] == "x4":
            return {"success": i < 9}                        # X4 success .9 vs X1 1.0 everywhere
        return {}
    R = A.analyze(synthetic_eval(tmp_path, spec, n=10), resamples=0)["rules"]
    assert R["R4"]["aggregate"]["lineage_mean_success_gap_x1_minus_x4"] == pytest.approx(.1)
    assert R["R4"]["supported"] and R["R4"]["aggregate"]["success_pass"]


def test_sealed_conditions_and_checkpoint_discovery(tmp_path):
    assert tuple(n for n, _ in C.SEALED) == A.CONDITIONS
    assert all(set(g) <= set(A.CONDITIONS) for g in A.GROUPS.values())
    cfg = C.sealed([], ["dep_reuse"], examples=3)
    assert [c["seed_start"] for c in cfg["conditions"]][:2] == [110_000_000, 110_100_000]
    for endpoint in ("boot", "rl"):
        for arm in C.ARMS:
            for r in range(3):
                job = tmp_path / f"p1-{endpoint}-{arm}-r{r}"
                (job / "checkpoints").mkdir(parents=True)
                ck = job / "checkpoints" / "round-0-slot-5-attempt-5.pt"
                ck.write_bytes(f"{endpoint}{arm}{r}".encode())
                (job / "state.json").write_text(json.dumps({"status": "completed", "finalist": {
                    "checkpoint": "checkpoints/round-0-slot-5-attempt-5.pt",
                    "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest()}}))
    rows = C.endpoint_checkpoints(tmp_path)
    assert len(rows) == 24 and {(x["endpoint"], x["arm"], x["lineage"]) for x in rows} == {
        (e, a, r) for e in ("boot", "rl") for a in C.ARMS for r in range(3)}
    assert all(A.NAME_PATTERN.match(x["name"]) for x in rows)
    assert {x["feature_version"] for x in rows if x["arm"] == "x3"} == {"d1-noapp"}
    (tmp_path / "p1-rl-x4-r2" / "checkpoints" / "round-0-slot-5-attempt-5.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        C.endpoint_checkpoints(tmp_path)


# --- real depworld episodes (evaluator-only audit) --------------------------------------------------

def test_audit_is_logging_only_and_reference_calibration():
    """The audited world reproduces DepWorkshop exactly; on the registered definition the
    dep_reuse teacher reuses at every opportunity and dep_recompute never does."""
    from tensegra.campaign02_protocol import execute
    from tensegra.campaign02_references import make_reference, run_episode
    from tensegra.campaign03_depworld import DepWorkshop, depworld_executor, generate_depworld
    from tensegra.campaign03_p1_audit import P1AuditedDepWorkshop
    ex = partial(depworld_executor, execute_call=execute)
    worlds = dict(C.SEALED)
    totals = {}
    for cname in ("foreign4", "events_train_kinds_p1"):
        for mode in ("dep_reuse", "dep_recompute", "dep_naive_reuse"):
            for i in range(6):
                seed = DEV + i
                spec = generate_depworld(seed, **worlds[cname])
                a = run_episode(P1AuditedDepWorkshop(spec, executor=ex, address_seed=seed), make_reference(mode), 1.0)
                b = run_episode(DepWorkshop(spec, executor=ex, address_seed=seed), make_reference(mode), 1.0)
                a.pop("controller_cpu_seconds"), b.pop("controller_cpu_seconds")
                for k in ("episode_cpu_seconds", "episode_wall_seconds", "solver_cpu_seconds"):
                    a.pop(k), b.pop(k)
                audit = a.pop("p1_audit")
                assert a == b                                       # identical trajectory and outcome
                assert len(audit["steps"]) == len(a["history"]) and audit["version"] == "p1-audit-v1"
                m = A.episode_metrics({"seed": seed, "outcome": {**a, "p1_audit": audit}})
                t = totals.setdefault(mode, [0, 0, 0, 0])
                t[0] += m["reuse_opportunities"]; t[1] += m["reuse_correct"]
                t[2] += m["invalid_uses"]; t[3] += m["retry_identical"]
    assert totals["dep_reuse"][0] > 0 and totals["dep_reuse"][0] == totals["dep_reuse"][1]
    assert totals["dep_recompute"][0] > 0 and totals["dep_recompute"][1] == 0
    assert totals["dep_reuse"][2] == 0 and totals["dep_naive_reuse"][2] > 0
    assert all(t[3] == 0 for t in totals.values())


def test_tiny_sealed_evaluation_end_to_end(tmp_path):
    """Random-init d1 checkpoint + two references through campaign02_evaluate.py on CPU."""
    import torch
    from tensegra.campaign02_policy import CandidatePolicy, PolicyConfig
    from tensegra.campaign02_training import TrainConfig
    torch.manual_seed(0)
    model = CandidatePolicy(PolicyConfig(82, 153, width=8, feature_version="d1"))
    ck = tmp_path / "ck.pt"
    torch.save({"config": asdict(TrainConfig(width=8, max_steps=24)), "policy_config": asdict(model.config),
                "model": model.state_dict(), "updates": 0, "presentations": 0}, ck)
    sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    binding = {"name": "p1-rl-x1-r0", "path": str(ck), "sha256": sha, "arm": "x1", "lineage": 0, "endpoint": "rl",
               "feature_version": "d1"}
    cfg = C.sealed([binding], ["dep_reuse", "dep_recompute"], examples=2, start=DEV)
    cfg["conditions"] = [c for c in cfg["conditions"] if c["name"] in ("iid_f2", "events_train_kinds_p1")]
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    out = tmp_path / "eval"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    subprocess.run([sys.executable, str(ROOT / "research/tools/campaign02_evaluate.py"), str(tmp_path / "cfg.json"),
                    "--output", str(out), "--device", "cpu"], check=True, env=env, timeout=600)
    summary = json.loads((out / "summary.json").read_text())
    assert "p1_audit" in summary["sources"]
    learned = [r for r in summary["results"] if r["kind"] == "learned"]
    assert learned[0]["checkpoint_binding"]["endpoint"] == "rl" and "path" not in learned[0]["checkpoint_binding"]
    result = A.analyze(out, resamples=10)
    table = result["table"]
    assert set(table) == {"x1-rl-r0", "dep_reuse", "dep_recompute"}
    assert table["dep_recompute"]["iid_f2"]["supports"]["audit"] == 2
    assert table["dep_recompute"]["events_train_kinds_p1"].get("correct_reuse_rate") in (0.0, None)
    assert table["x1-rl-r0"]["iid_f2"]["supports"]["audit"] == 2
    assert set(result["rules"]) == {"R1", "R2", "R3", "R4", "R5"}
