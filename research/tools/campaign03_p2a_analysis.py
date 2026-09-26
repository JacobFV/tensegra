"""extended-03 P2a screening analysis: no-progress metrics and readings S0-S3.

Registered design: research/campaigns/extended-03/protocol-P2a.md (non-sealed,
non-confirmatory screening). This tool reads a campaign02_evaluate.py output
directory built by ``campaign03_p2a_configs.py screening`` and recomputes every
metric from raw per-episode rows. P1 metric code (success, utility, work per
success, correct reuse, invalid use) is reused from campaign03_p1_analysis.py
unchanged. Standard library only (no torch); deterministic.

    python research/tools/campaign03_p2a_analysis.py <screening-eval-dir> --output <dir> \\
        --deployment-c0 <c0-deployment.json> --deployment-c1 <c1-deployment.json>

No-progress metrics (protocol-P2a.md "Metrics"), computed from the environment's
exact action/feedback ``history`` so they apply to learned and reference rows
alike. Operationalization, fixed here before any P2a output exists:

- **Public decision state** after each step = (draft, commitments, retrieved set,
  position), reconstructed from the history:
  * draft: every problem draft's constraints, finish_by bound, exclusions and the
    requirement versions it was last built against (the environment's
    ``depends_on``; re-editing a draft after an event refreshes it), foreign drafts
    starting from their public snapshot; plus the pending item choices and pending
    slot choices (the draft of the selection/assignment before commit);
  * commitments: selection_id, assignment_id (commits, use_return, uncommit, event
    revocations) and verified;
  * retrieved set: record handles retrieved so far;
  * position: current location (start from the world spec).
  ``tests/test_campaign03_p2a_analysis.py`` checks the reconstruction against the
  environment's own public observations.
- **Accepted** action = feedback status ``success``.
- **Idempotent repeat** at step t: accepted, its action_key equals the action_key
  of the most recent earlier accepted action, and the decision state after t
  equals the state before t.
- **Short cycle** at step t: accepted, and the state after t equals a state seen
  after one of the 6 preceding steps (or the episode start, when t <= 6), where
  no step from that earlier occurrence through t is a solver call, a *new*
  inspection (first inspection of a target in the episode) or a step at which a
  public event fired (new public information, like an inspection). The equal state may
  be the immediately preceding one, i.e. an accepted no-effect action closes a
  cycle of length 1.
- **No-progress step** = idempotent repeat or short cycle. Rates are per decision
  (Σ flagged steps / Σ steps, pooled within a condition); per-episode counts and
  the fraction of episodes with any no-progress step are reported alongside.
- **Steps to cap**: an episode that ends without verified success after
  len(history) >= its step cap, the cap being the world's ``step_limit`` (and
  the learned decision cap ``max_steps`` if smaller). P1's ``truncated`` row field
  is never used (it is never set for references and is not the protocol quantity).

Readings (protocol-P2a.md, decided before running; they are not claims):
- S0 replay: C0 final development success < bootstrap development success - 0.20
  (from the deployment-rule JSON, which carries the run's development rows).
- S1 preservation: C1 final greedy IID-group success >= bootstrap - 0.02 AND C1
  final greedy IID-group no-progress rate <= bootstrap + 0.02.
- S2 improvement: C1 final IID-group utility >= bootstrap + 0.01, OR work per
  success <= 0.9 x bootstrap with success >= bootstrap - 0.02.
- S3 deployment vs learning: deployed vs final for each arm; whether the deployed
  checkpoint helps only where the final one fails.
Group values are equal-weighted means over iid_f0 and iid_f2 (P1 convention).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

ANALYSIS_VERSION = "p2a-analysis-v1"
HERE = Path(__file__).resolve().parent
CYCLE_WINDOW = 6
IID = ("iid_f0", "iid_f2")
SCREENING_CONDITIONS = ("iid_f0", "iid_f2", "events_train_kinds_p1", "foreign4")
ROLES = ("bootstrap", "c0_final", "c0_deployed", "c1_final", "c1_deployed", "p1_rl")
THRESHOLDS = {"s0_collapse": 0.20, "s1_success_margin": 0.02, "s1_no_progress_margin": 0.02,
              "s2_utility_gain": 0.01, "s2_work_ratio": 0.9, "s2_success_margin": 0.02, "s3_help": 0.02}


def _load_p1():
    spec = importlib.util.spec_from_file_location("campaign03_p1_analysis", HERE / "campaign03_p1_analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P1 = _load_p1()


# ---------------------------------------------------------------------------
# Public decision state and no-progress flags (per episode)
# ---------------------------------------------------------------------------

def action_key(action):
    """campaign03_depworld.action_key on a history row's action (canonical kind + arguments)."""
    value = {"kind": action["kind"], "arguments": dict(action.get("arguments") or {})}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


# Mirrors campaign03_depworld (asserted equal in tests/test_campaign03_p2a_analysis.py).
REQUIREMENTS = ("capacity", "funds", "incompatible", "slots", "deadline", "map")
READS = {"constrained_subset": ("capacity", "funds", "incompatible"), "csp": ("slots",), "shortest_path": ("map",)}
DRAFT_PRIMITIVE = {"start_subset": "constrained_subset", "start_assign": "csp", "build_route": "shortest_path"}


def _deps_key(depends_on):
    return json.dumps(depends_on, sort_keys=True)


def initial_drafts(spec):
    """Public drafts present at the start (foreign problems problem_0..), from the world spec."""
    drafts = {}
    for i, f in enumerate((spec or {}).get("foreign", ())):
        p = f["snapshot"]["problem"]
        drafts[f"problem_{i}"] = (f["primitive"], tuple(sorted(p.get("constraints", ()))), p.get("finish_by"),
                                  tuple(sorted(p.get("excluded", ()))), _deps_key(f["depends_on"]))
    return drafts


class DecisionState:
    """Mutable reconstruction of the public decision state from history rows.

    draft: per problem (primitive, constraints, finish_by bound, exclusions, the dependency
    versions it was last built against -- as the environment's ``depends_on``), plus pending
    item and slot choices; commitments: selection_id, assignment_id, verified; retrieved
    handles; position. Requirement versions are tracked only to rebuild draft ``depends_on``.
    """

    def __init__(self, start=None, drafts=None):
        self.drafts = dict(drafts or {})
        self.pending_items: tuple = ()
        self.pending_slots: dict = {}
        self.selection_id = self.assignment_id = None
        self.verified = False
        self.retrieved: set = set()
        self.position = start
        self.versions = {n: 1 for n in REQUIREMENTS}

    def key(self):
        return (tuple(sorted(self.drafts.items())), tuple(sorted(self.pending_items)),
                tuple(sorted(self.pending_slots.items())), self.selection_id, self.assignment_id, self.verified,
                tuple(sorted(self.retrieved)), self.position)

    def _deps(self, primitive, selection=None):
        out = {"requirements_version": {n: self.versions[n] for n in READS[primitive]}}
        if primitive == "csp":
            out["selection_id"] = selection
        return out

    def apply(self, event):
        action, feedback = event["action"], event["feedback"]
        kind, args = action["kind"], action.get("arguments") or {}
        ok = feedback.get("status") == "success"
        if ok and kind in DRAFT_PRIMITIVE:
            primitive = DRAFT_PRIMITIVE[kind]
            self.drafts[feedback.get("problem", args.get("handle"))] = (
                primitive, (), None, (), _deps_key(self._deps(primitive, self.selection_id)))
        elif ok and kind == "add_constraint":
            name = args["problem"]
            c = args["constraint"]
            if name not in self.drafts:  # only without a world spec (foreign drafts unknown)
                self.drafts[name] = ("csp" if c in ("finish_by", "conflicts") else "constrained_subset",
                                     (), None, (), "{}")
            primitive, constraints, bound, excluded, deps = self.drafts[name]
            if c == "finish_by":
                bound = args.get("bound")
            elif c == "exclude":
                excluded = tuple(sorted(set(excluded) | {args.get("item")}))
            elif primitive == "constrained_subset" or c == "conflicts":
                constraints = tuple(sorted(set(constraints) | {c}))
            if primitive == "csp":
                deps = _deps_key(self._deps("csp", json.loads(deps).get("selection_id")))
            else:
                deps = _deps_key(self._deps(primitive))
            self.drafts[name] = (primitive, constraints, bound, excluded, deps)
        elif ok and kind == "choose_item":
            self.pending_items = tuple(feedback.get("pending", ()))
        elif ok and kind == "choose_slot":
            self.pending_slots[args["item"]] = args["slot"]
        elif ok and kind == "retrieve":
            self.retrieved.add(args["handle"])
        elif ok and kind == "uncommit":
            if args.get("target") == "select":
                self.selection_id = self.assignment_id = None
                self.pending_slots = {}
            else:
                self.assignment_id = None
        if ok and "selection_id" in feedback:  # commit_pending or use_return(select)
            self.selection_id = feedback["selection_id"]
            self.pending_slots = {}
        if ok and "assignment_id" in feedback:  # commit_assignment or use_return(assign)
            self.assignment_id = feedback["assignment_id"]
        if ok and "position" in feedback:  # move or use_return(route)
            self.position = feedback["position"]
        if ok and feedback.get("verified"):
            self.verified = True
        public_event = feedback.get("event")  # fires after the action (environment order)
        if public_event:
            self.versions[public_event["affected"]] += 1
            revoked = public_event.get("revoked") or ()
            if "selection" in revoked:
                self.selection_id = None
                self.pending_slots = {}
            if "assignment" in revoked:
                self.assignment_id = None


def no_progress_flags(history, start=None, drafts=None, window=CYCLE_WINDOW):
    """Per step: (idempotent_repeat, short_cycle). See the module docstring."""
    state = DecisionState(start, drafts)
    states = [state.key()]  # states[j] = decision state after j steps
    progress_mark = []      # progress_mark[t-1]: step t is a call, a new inspection or an event step
    inspected = set()
    last_accepted_key = None
    flags = []
    for t, event in enumerate(history, start=1):
        action, feedback = event["action"], event["feedback"]
        kind, args = action["kind"], action.get("arguments") or {}
        key = action_key(action)
        accepted = feedback.get("status") == "success"
        new_inspection = kind == "inspect" and args.get("target") not in inspected
        if kind == "inspect":
            inspected.add(args.get("target"))
        progress_mark.append(kind == "call" or new_inspection or bool(feedback.get("event")))
        state.apply(event)
        after = state.key()
        idempotent = accepted and key == last_accepted_key and after == states[t - 1]
        cycle = False
        if accepted:
            for j in range(t - 1, max(0, t - window) - 1, -1):
                # a call / new inspection / event at steps j+1..t breaks the cycle
                if progress_mark[j]:
                    break
                if states[j] == after:
                    cycle = True
                    break
        states.append(after)
        if accepted:
            last_accepted_key = key
        flags.append((bool(idempotent), bool(cycle)))
    return flags


def episode_no_progress(row, spec=None, decision_cap=None):
    """No-progress numerators/denominators for one raw evaluator row."""
    o = row["outcome"]
    history = o.get("history", [])
    flags = no_progress_flags(history, (spec or {}).get("start"), initial_drafts(spec))
    idem = sum(a for a, _ in flags)
    cycle = sum(b for _, b in flags)
    either = sum(a or b for a, b in flags)
    cap = (spec or {}).get("step_limit", 96)
    if decision_cap is not None:
        cap = min(cap, decision_cap)
    capped = int(not o["verified_success"] and len(history) >= cap)
    return {"decisions": len(history), "idempotent_repeats": idem, "short_cycles": cycle, "no_progress": either,
            "no_progress_episode": int(either > 0), "steps_to_cap": capped,
            "accepted": sum(h["feedback"].get("status") == "success" for h in history)}


NP_RATES = {
    "idempotent_repeat_rate": ("idempotent_repeats", "decisions"),
    "short_cycle_rate": ("short_cycles", "decisions"),
    "no_progress_rate": ("no_progress", "decisions"),
    "no_progress_per_episode": ("no_progress", "episodes"),
    "no_progress_episode_rate": ("no_progress_episode", "episodes"),
    "steps_to_cap_rate": ("steps_to_cap", "episodes"),
    "decisions_per_episode": ("decisions", "episodes"),
}
P1_KEEP = ("success", "utility", "cost", "work_per_episode", "work_per_success", "steps", "correct_reuse_rate",
           "invalid_reuse_rate", "stale_use_rate", "invalid_reuse_episode_rate", "reuse_opportunities_per_episode",
           "identical_retry_rate")


def aggregate(metrics):
    base = P1.aggregate([{k: v for k, v in m.items() if k not in NP_FIELDS} for m in metrics])
    totals = defaultdict(float)
    for m in metrics:
        for k in NP_FIELDS:
            totals[k] += m.get(k, 0)
    totals["episodes"] = len(metrics)
    out = {k: base.get(k) for k in P1_KEEP}
    out.update({name: P1._ratio(totals[a], totals[b]) for name, (a, b) in NP_RATES.items()})
    out["no_progress_counts"] = [m["no_progress"] for m in metrics]
    out["supports"] = {**base["supports"], "decisions": totals["decisions"]}
    return out


NP_FIELDS = ("decisions", "idempotent_repeats", "short_cycles", "no_progress", "no_progress_episode",
             "steps_to_cap", "accepted")


# ---------------------------------------------------------------------------
# Loading the screening evaluation
# ---------------------------------------------------------------------------

def result_role(result):
    if result["kind"] == "supplied_schedule":
        return result["arm"].removeprefix("reference-")
    binding = result.get("checkpoint_binding") or {}
    return binding.get("role", result["arm"])


def split_condition(name, result):
    """(base condition, mode). Sampled conditions are named '<base>-sampled'."""
    mode = result.get("policy_mode", "greedy")
    base = name[:-len("-sampled")] if name.endswith("-sampled") else name
    return base, mode


def load(eval_dir: Path, allow_partial=False, verify=True):
    path = eval_dir / "summary.json"
    if not path.exists():
        if not allow_partial:
            raise FileNotFoundError("summary.json missing (evaluation incomplete); use --allow-partial for smoke only")
        path = eval_dir / "summary.partial.json"
    summary = json.loads(path.read_text())
    specs, table_rows = {}, {}
    for result in summary["results"]:
        c = result["condition"]
        if c not in specs:
            specs[c] = {w["seed"]: w["spec"] for w in P1.read_rows(eval_dir / c / "worlds.jsonl.gz")}
        artifact = eval_dir / result["artifact"]
        if verify and P1.sha256(artifact) != result["artifact_sha256"]:
            raise ValueError(f"artifact hash mismatch: {artifact}")
        cap = (result.get("training_config") or {}).get("max_steps")
        base, mode = split_condition(c, result)
        metrics = []
        for row in P1.read_rows(artifact):
            spec = specs[c][row["seed"]]
            prices = {k: spec[k] for k in ("action_price", "observation_price", "travel_price", "work_price",
                                           "compute_price")}
            metrics.append({**P1.episode_metrics(row, prices), **episode_no_progress(row, spec, cap)})
        table_rows[(result_role(result), mode, base)] = metrics
    return summary, table_rows


def group_mean(table, role, mode, metric, group=IID):
    values = {c: (table.get((role, mode, c)) or {}).get(metric) for c in group}
    return P1.group_mean(values, group)[0]


# ---------------------------------------------------------------------------
# Readings
# ---------------------------------------------------------------------------

def readings(table, deployment_c0=None, deployment_c1=None):
    T = THRESHOLDS
    g = lambda role, metric, mode="greedy": group_mean(table, role, mode, metric)
    out = {}
    # S0 (replay): development data from the deployment-rule JSON of C0.
    if deployment_c0 is not None:
        boot = deployment_c0["bootstrap_development"]["success"]
        final = deployment_c0["final"]["success"]
        out["S0"] = {"bootstrap_development_success": boot, "c0_final_development_success": final,
                     "threshold": boot - T["s0_collapse"], "collapse_reproduced": final < boot - T["s0_collapse"],
                     "note": "If False, GPU/numerical nondeterminism enters the explanation and is reported."}
    else:
        out["S0"] = {"missing": "pass --deployment-c0"}
    # S1 (preservation), greedy IID group.
    bs, cs = g("bootstrap", "success"), g("c1_final", "success")
    bn, cn = g("bootstrap", "no_progress_rate"), g("c1_final", "no_progress_rate")
    s1_success = None if bs is None or cs is None else cs >= bs - T["s1_success_margin"]
    s1_np = None if bn is None or cn is None else cn <= bn + T["s1_no_progress_margin"]
    out["S1"] = {"bootstrap_success": bs, "c1_final_success": cs, "success_clause": s1_success,
                 "bootstrap_no_progress_rate": bn, "c1_final_no_progress_rate": cn, "no_progress_clause": s1_np,
                 "holds": bool(s1_success and s1_np) if None not in (s1_success, s1_np) else None}
    # S2 (improvement), greedy IID group.
    bu, cu = g("bootstrap", "utility"), g("c1_final", "utility")
    bw, cw = g("bootstrap", "work_per_success"), g("c1_final", "work_per_success")
    util_clause = None if bu is None or cu is None else cu >= bu + T["s2_utility_gain"]
    work_clause = (None if bw is None or cw is None or bs is None or cs is None
                   else cw <= T["s2_work_ratio"] * bw and cs >= bs - T["s2_success_margin"])
    out["S2"] = {"bootstrap_utility": bu, "c1_final_utility": cu, "utility_clause": util_clause,
                 "bootstrap_work_per_success": bw, "c1_final_work_per_success": cw, "work_clause": work_clause,
                 "holds": bool(util_clause or work_clause) if (util_clause, work_clause) != (None, None) else None,
                 "scope": "S2 alone does not justify a claim of reward-driven improvement from one lineage."}
    # S3 (deployment vs learning), each arm, greedy IID group and per condition.
    s3 = {}
    for arm, deployment in (("c0", deployment_c0), ("c1", deployment_c1)):
        final_s, dep_s = g(f"{arm}_final", "success"), g(f"{arm}_deployed", "success")
        final_u, dep_u = g(f"{arm}_final", "utility"), g(f"{arm}_deployed", "utility")
        final_fails = None if final_s is None or bs is None else final_s < bs - T["s1_success_margin"]
        helps = None if final_s is None or dep_s is None else dep_s - final_s > T["s3_help"]
        if final_fails is None or helps is None:
            verdict = None
        elif final_fails and helps:
            verdict = "deployment rule, not stable learning: the deployed checkpoint helps where the final one fails"
        elif final_fails:
            verdict = "final fails and deployment does not recover it"
        elif helps:
            verdict = "deployed better than a non-failing final"
        else:
            verdict = "final preserved; deployment adds nothing"
        s3[arm] = {"final_success": final_s, "deployed_success": dep_s, "final_utility": final_u,
                   "deployed_utility": dep_u, "final_fails": final_fails, "deployed_helps": helps, "reading": verdict,
                   "deployed_checkpoint": (deployment or {}).get("deployed", {}).get("label"),
                   "per_condition": {c: {"final_success": (table.get((f"{arm}_final", "greedy", c)) or {}).get("success"),
                                         "deployed_success": (table.get((f"{arm}_deployed", "greedy", c)) or {}).get("success")}
                                     for c in SCREENING_CONDITIONS}}
    out["S3"] = s3
    # Descriptive replay check on new worlds: C0 final vs P1's X1-rl-r2 endpoint.
    out["replay_on_screening_worlds"] = {m: {"c0_final": g("c0_final", m), "p1_rl": g("p1_rl", m)}
                                         for m in ("success", "utility", "no_progress_rate")}
    return out


def analyze(eval_dir: Path, deployment_c0=None, deployment_c1=None, allow_partial=False):
    summary, rows = load(eval_dir, allow_partial)
    table = {key: aggregate(ms) for key, ms in rows.items()}
    return {"analysis_version": ANALYSIS_VERSION, "evaluation_dir": str(eval_dir),
            "evaluation_config_sha256": summary.get("config_sha256"), "checkpoints": summary.get("checkpoints"),
            "partial": not (eval_dir / "summary.json").exists(), "thresholds": THRESHOLDS,
            "cycle_window": CYCLE_WINDOW, "iid_group": list(IID),
            "readings": readings(table, deployment_c0, deployment_c1),
            "groups": {f"{role}/{mode}": {m: group_mean(table, role, mode, m) for m in
                                          ("success", "utility", "work_per_success", "no_progress_rate",
                                           "idempotent_repeat_rate", "short_cycle_rate", "steps_to_cap_rate",
                                           "correct_reuse_rate", "invalid_reuse_rate")}
                       for role, mode in sorted({(r, m) for r, m, _ in table})},
            "table": {f"{role}/{mode}/{c}": v for (role, mode, c), v in sorted(table.items())}}


def _f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


KEY = ("success", "utility", "work_per_success", "correct_reuse_rate", "invalid_reuse_rate", "idempotent_repeat_rate",
       "short_cycle_rate", "no_progress_rate", "no_progress_per_episode", "steps_to_cap_rate")


def markdown(result):
    R = result["readings"]
    lines = [f"# P2a screening analysis ({result['analysis_version']})", "",
             f"Evaluation: `{result['evaluation_dir']}` (config sha256 `{result['evaluation_config_sha256']}`)"
             + (" **PARTIAL**" if result["partial"] else ""), "",
             "Non-sealed, non-confirmatory screening (protocol-P2a.md). Readings are not claims.", "",
             "## Readings", "", "| Reading | Holds | Detail |", "|---|---|---|"]
    s0 = R["S0"]
    lines.append(f"| S0 replay (C0 collapses) | {_f(s0.get('collapse_reproduced'))} | boot dev "
                 f"{_f(s0.get('bootstrap_development_success'))}, C0 final dev {_f(s0.get('c0_final_development_success'))} |")
    s1 = R["S1"]
    lines.append(f"| S1 preservation | {_f(s1['holds'])} | success {_f(s1['c1_final_success'])} vs boot "
                 f"{_f(s1['bootstrap_success'])}; no-progress {_f(s1['c1_final_no_progress_rate'])} vs boot "
                 f"{_f(s1['bootstrap_no_progress_rate'])} |")
    s2 = R["S2"]
    lines.append(f"| S2 improvement | {_f(s2['holds'])} | utility {_f(s2['c1_final_utility'])} vs boot "
                 f"{_f(s2['bootstrap_utility'])}; work/success {_f(s2['c1_final_work_per_success'], 1)} vs boot "
                 f"{_f(s2['bootstrap_work_per_success'], 1)} |")
    for arm, s3 in R["S3"].items():
        lines.append(f"| S3 {arm} | - | final {_f(s3['final_success'])}, deployed {_f(s3['deployed_success'])}: "
                     f"{s3['reading'] or 'n/a'} |")
    lines += ["", "## Groups (IID: iid_f0, iid_f2)", "",
              "| policy/mode | success | utility | work/success | no-progress | idempotent | short-cycle | to cap |",
              "|---|---|---|---|---|---|---|---|"]
    for name, g in result["groups"].items():
        lines.append(f"| {name} | {_f(g['success'])} | {_f(g['utility'])} | {_f(g['work_per_success'], 1)} | "
                     f"{_f(g['no_progress_rate'])} | {_f(g['idempotent_repeat_rate'])} | {_f(g['short_cycle_rate'])} | "
                     f"{_f(g['steps_to_cap_rate'])} |")
    lines += ["", "## Per condition", "", "| policy/mode/condition | " + " | ".join(KEY) + " |",
              "|---" * (len(KEY) + 1) + "|"]
    for name, agg in result["table"].items():
        lines.append(f"| {name} | " + " | ".join(_f(agg.get(k), 1 if k == "work_per_success" else 3) for k in KEY) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("evaluation", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deployment-c0", type=Path, help="deployment-rule JSON for C0 (S0, S3)")
    p.add_argument("--deployment-c1", type=Path, help="deployment-rule JSON for C1 (S3)")
    p.add_argument("--allow-partial", action="store_true", help="smoke only: read summary.partial.json")
    a = p.parse_args()
    d0 = json.loads(a.deployment_c0.read_text()) if a.deployment_c0 else None
    d1 = json.loads(a.deployment_c1.read_text()) if a.deployment_c1 else None
    result = analyze(a.evaluation, d0, d1, a.allow_partial)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "p2a-analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (a.output / "p2a-analysis.md").write_text(markdown(result))
    print(json.dumps({k: (v.get("holds", v.get("collapse_reproduced")) if isinstance(v, dict) and k != "S3" else None)
                      for k, v in result["readings"].items() if k.startswith("S")}))


if __name__ == "__main__":
    main()
