"""Extended-05 B-LOC: localize the composition failure of existing probeworld models (registry entry B-LOC).

No training.  For every greedy decision of each model on the historical challenge pairs (extended-04 split
``heldout_comp``: unreliable+events and side_effect+correlated, reported SEPARATELY) this records the configuration,
the visible history, the exact belief, Q*/V*, the eps-optimal set A*, the chosen action (= the policy argmax under
greedy deployment), the model's Q-head values (meaningful only when the Q head was trained, i.e. rung L4), the
policy logits and delta_t = V*(I_t) - Q*(I_t, a_t).  It then locates each episode's FIRST CONSEQUENTIAL ERROR
(delta_t > eps, eps = 0.5 registered) and classifies it (precedence as listed):

  deployment        an eps-optimal action's logit ties the chosen one's (|diff| <= 1e-6; greedy tie-break).  The
                    action mask is the public availability mask, so an eps-optimal action is never masked
                    (A* is a subset of the available set by construction; asserted).  Near-vacuous by construction
                    (review F15): ties have measure zero; kept so that the classification is exhaustive.
  ranking           Q head trained, and the Q-head argmax is eps-optimal, but the policy picks otherwise.
  belief_dependent  the BO models (L1 recipe + exact-belief input, trained on the extended-04 split, design v2
                    revision 15; --bo-runs), each driven along the SAME visible history, pick an eps-optimal action at
                    that state (>= 2/3 of the BO models).  Needs the BO diagnostic arm; without it the class is not
                    assigned.  Expected rare at s0, where the exact belief equals the declared prior (review F14).
  value_estimate    Q head trained and its argmax is not eps-optimal (sub-flag `misorders_chosen`:
                    Q-hat(chosen) >= max over A* of Q-hat).
  unlocalized       no trained Q head (L1) and no BO resolution.

Realized regret V*(I_0) - U and accumulated gap regret sum_t delta_t are reported separately (equal in
expectation under the registered utility and termination; per trajectory they differ).  Exact labels come from the
pure-Python DP (pickled label tables = ExactSolver memo, verified equal to a fresh DP by the B1 audit).

World seeds default to each model's original evaluation worlds (B1 runs 'b-train-*': offset 500; F2 runs
'f-btrain-*': offset 700), so the first-decision probe rates reproduce eval.json exactly (checked and reported).

Usage (remote, torch):
  python research/tools/campaign05_bloc.py --labels LABELS_DIR --runs RUN... --out bloc.json \
      [--records bloc_records.jsonl.gz] [--bo-runs BO_RUN...] [--split heldout_comp] [--worlds 4] [--offset N]
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tensegra import campaign04_probeworld as pw  # noqa: E402
from tensegra import campaign05_probeworld as pw5  # noqa: E402

_spec = importlib.util.spec_from_file_location("pwtrain_bloc", REPO / "research/tools/campaign04_probeworld_train.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

CLASSES = ("deployment", "ranking", "belief_dependent", "value_estimate", "unlocalized")


def load_model(run: Path):
    meta = json.loads((run / "train_meta.json").read_text())
    model = T.ProbeNet(meta["hidden"], own_value="own_value" in meta, inputs=meta.get("inputs", "public"),
                       arch=meta.get("arch", "flat"))
    model.load_state_dict(torch.load(run / "model.pt"))
    model.eval()
    return model, meta


def default_offset(run: Path) -> int:
    return 700 if "f-btrain" in str(run) else T.EVAL_WORLD_OFFSET


def classify(d: dict, q_trained: bool, bo: dict | None) -> tuple:
    """Class of a consequential error record d (see module docstring) and diagnostic flags."""
    a, opt, avail, qhat = d["a"], set(d["opt"]), d["avail"], d["qhat"]
    assert opt <= set(avail)
    qa = avail[max(range(len(avail)), key=lambda j: qhat[j])]
    flags = {"q_head_argmax": qa, "q_head_argmax_eps_opt": qa in opt if q_trained else None,
             "misorders_chosen": (qhat[avail.index(a)] >= max(qhat[avail.index(b)] for b in opt)) if q_trained
             else None, "tie_opt": d["tie_opt"],
             "bo_argmax": bo["argmax"] if bo else None, "bo_eps_opt": bo["eps_opt"] if bo else None}
    if d["tie_opt"]:
        return "deployment", flags
    if q_trained and qa in opt:
        return "ranking", flags
    if bo and bo["eps_opt"]:
        return "belief_dependent", flags
    if q_trained:
        return "value_estimate", flags
    return "unlocalized", flags


def bo_at(bo_models, cfg, decisions, t):
    """Each BO model driven along the visible history decisions[:t]; its argmax at step t.  eps_opt = at least 2/3
    of the BO models pick an eps-optimal action."""
    hist = [tuple(d["rec"]) for d in decisions[:t]]
    am = []
    for bo_model in bo_models:
        av, logits, _ = T.replay_history(bo_model, cfg, hist, upto=t + 1)[t]
        assert tuple(av) == tuple(decisions[t]["avail"])
        am.append(int(logits.argmax()))
    hits = sum(x in decisions[t]["opt"] for x in am)
    return {"argmax": am, "eps_opt": 3 * hits >= 2 * len(am), "hits": hits}


def run_model(run: Path, pool, split, worlds, offset, bo_models=None, chunk=256):
    model, meta = load_model(run)
    q_trained = meta.get("weights", {}).get("q", 0) > 0
    items = T.eval_items(pool, split, worlds, offset)
    cfg_of = {id(cfg): idx for idx, cfg, _ in pool}
    rows = []
    with torch.no_grad():
        for c0 in range(0, len(items), chunk):
            part = items[c0:c0 + chunk]
            eps, steps, ep_steps = T.run_batch(model, part, "greedy", need_labels=False)
            rows += T.decision_rows(part, eps, ep_steps, steps, model)
    for j, r in enumerate(rows):
        cfg = items[j][0]
        r["cfg_idx"] = cfg_of[id(cfg)]
        r["rep"] = j % worlds
        decs = r["decisions"]
        r["gap_regret"] = sum(d["delta"] for d in decs)
        r["realized_regret"] = r["V_star"] - r["U"]
        t_err = next((t for t, d in enumerate(decs) if d["delta"] > pw.EPS), None)
        r["first_error_t"] = t_err
        r["n_consequential"] = sum(d["delta"] > pw.EPS for d in decs)
        if t_err is not None:
            d = decs[t_err]
            bo = bo_at(bo_models, cfg, decs, t_err) if bo_models else None
            cls, flags = classify(d, q_trained, bo)
            rep_opt = min(d["opt"], key=lambda b: -d["Q"][d["avail"].index(b)])
            r["first_error"] = {"class": cls, **flags, "t": t_err, "query": d["i"],
                                "step_in_query": t_err - max((u + 1 for u in range(t_err) if decs[u]["a"] in pw.TERMINAL),
                                                             default=0),
                                "chosen": pw.ACTIONS[d["a"]], "best": pw.ACTIONS[rep_opt],
                                "opt": [pw.ACTIONS[b] for b in d["opt"]], "delta": d["delta"],
                                "first_decision_of_episode": t_err == 0,
                                "belief_moved": d["belief"] != list(pw.local0(cfg, *_revealed(cfg, decs, t_err))[0])}
        fd = decs[0]
        opt0 = set(fd["opt"])
        r["first_probe"] = int(fd["a"] == pw.A_PROBE)
        r["probe_eps_opt"] = int(pw.A_PROBE in opt0)
        r["probe_unique_opt"] = int(opt0 == {pw.A_PROBE})
    return rows, meta, q_trained


def _revealed(cfg, decs, t):
    """(nH, nNotH) revealed before decision t (tracked only in the correlated condition, as in the public state)."""
    if cfg.corr <= 0:
        return 0, 0
    nH = sum(1 for d in decs[:t] if d["a"] in pw.TERMINAL and d["rec"][3] == pw.TH)
    nN = sum(1 for d in decs[:t] if d["a"] in pw.TERMINAL and d["rec"][3] != pw.TH)
    return nH, nN


def summarize(rows, q_trained):
    n = len(rows)
    err = [r for r in rows if r["first_error_t"] is not None]
    cls = {c: sum(1 for r in err if r["first_error"]["class"] == c) for c in CLASSES}
    patterns = {}
    for r in err:
        key = f'{r["first_error"]["chosen"]} (best {r["first_error"]["best"]})'
        patterns[key] = patterns.get(key, 0) + 1
    uo = [r for r in rows if r["probe_unique_opt"]]
    no = [r for r in rows if not r["probe_eps_opt"]]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    out = {
        "episodes": n, "configs": len({r["cfg_idx"] for r in rows}),
        "episodes_with_consequential_error": len(err),
        "first_error_class_counts": cls,
        "first_error_class_fraction": {c: v / len(err) for c, v in cls.items()} if err else None,
        "first_error_patterns": dict(sorted(patterns.items(), key=lambda kv: -kv[1])),
        "first_error_at_first_decision": sum(1 for r in err if r["first_error"]["first_decision_of_episode"]),
        "first_error_query_index": _hist([r["first_error"]["query"] for r in err]),
        "first_error_step_in_query": _hist([r["first_error"]["step_in_query"] for r in err]),
        "first_error_belief_moved": sum(1 for r in err if r["first_error"]["belief_moved"]),
        "consequential_errors_total": sum(r["n_consequential"] for r in rows),
        "realized_regret_mean": mean([r["realized_regret"] for r in rows]),
        "gap_regret_mean": mean([r["gap_regret"] for r in rows]),
        "gap_regret_first_error_mean": sum(r["first_error"]["delta"] for r in err) / n if n else None,
        "gap_regret_consequential_mean": sum(d["delta"] for r in rows for d in r["decisions"]
                                             if d["delta"] > pw.EPS) / n if n else None,
        "realized_regret_mean_no_error_episodes": mean([r["realized_regret"] for r in rows
                                                        if r["first_error_t"] is None]),
        "realized_regret_mean_error_episodes": mean([r["realized_regret"] for r in err]),
        "first_probe_rate_when_unique_opt": mean([r["first_probe"] for r in uo]), "n_probe_unique_opt": len(uo),
        "first_probe_rate_when_not_opt": mean([r["first_probe"] for r in no]), "n_probe_not_opt": len(no),
        "q_head_trained": q_trained,
    }
    if q_trained and err:
        out["q_head_argmax_eps_opt_at_first_error"] = mean([float(r["first_error"]["q_head_argmax_eps_opt"])
                                                            for r in err])
        out["misorders_chosen_at_first_error"] = mean([float(r["first_error"]["misorders_chosen"]) for r in err])
    bo = [r for r in err if r["first_error"]["bo_eps_opt"] is not None]
    out["bo_available"] = bool(bo)
    if bo:
        out["bo_resolves_first_error"] = mean([float(r["first_error"]["bo_eps_opt"]) for r in bo])
    return out


def _hist(xs):
    h = {}
    for x in xs:
        h[str(x)] = h.get(str(x), 0) + 1
    return dict(sorted(h.items(), key=lambda kv: int(kv[0])))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--records", default=None, help="optional per-decision records (jsonl.gz)")
    p.add_argument("--bo-runs", nargs="*", default=None,
                   help="BO diagnostic models (inputs=belief, extended-04 split); enables belief_dependent")
    p.add_argument("--split", default=pw5.CHALLENGE_SPLIT)
    p.add_argument("--worlds", type=int, default=4)
    p.add_argument("--offset", type=int, default=None, help="world offset (default: per model, B1 500 / F2 700)")
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    t0 = time.process_time()
    pool = T.load_pool(a.labels, a.split)
    t_load = time.process_time() - t0
    bo_models = []
    for r in a.bo_runs or []:
        m, bo_meta = load_model(Path(r))
        assert bo_meta.get("inputs") == "belief", "BO run must be trained with --inputs belief"
        bo_models.append(m)
    result = {"tool": "campaign05_bloc", "version": pw5.VERSION, "split": a.split, "eps": pw.EPS,
              "labels": str(a.labels), "bo_runs": a.bo_runs, "classes": CLASSES, "models": {}}
    rec_f = gzip.open(a.records, "wt") if a.records else None
    for run in map(Path, a.runs):
        t1 = time.process_time()
        offset = default_offset(run) if a.offset is None else a.offset
        rows, meta, q_trained = run_model(run, pool, a.split, a.worlds, offset, bo_models)
        name = run.parent.name if run.name == "run" else run.name
        ent = {"rung": meta["rung"], "seed": meta["seed"], "world_offset": offset,
               "inputs": meta.get("inputs", "public"), "train_split": meta.get("train_split", "train"),
               "all": summarize(rows, q_trained), "by_pair": {}}
        for pair in sorted({r["combo"] for r in rows}):
            ent["by_pair"][pair] = summarize([r for r in rows if r["combo"] == pair], q_trained)
        ev = run / "eval.json"
        if ev.exists() and a.split == pw5.CHALLENGE_SPLIT and a.worlds == 4:
            ref = json.loads(ev.read_text())["splits"].get(a.split, {}).get("free_running_greedy", {}).get("summary")
            if ref:
                ent["reproduces_eval_json"] = {
                    k: (ref[k], ent["all"][k], abs((ref[k] or 0) - (ent["all"][k] or 0)) < 1e-9)
                    for k in ("first_probe_rate_when_unique_opt", "n_probe_unique_opt",
                              "first_probe_rate_when_not_opt", "n_probe_not_opt")}
                ent["reproduces_eval_json"]["gap_regret"] = (ref["gap_regret"], ent["all"]["gap_regret_mean"],
                                                             abs(ref["gap_regret"] - ent["all"]["gap_regret_mean"]) < 1e-6)
        ent["cpu_s"] = time.process_time() - t1
        result["models"][name] = ent
        print(name, json.dumps({k: ent["all"][k] for k in ("episodes_with_consequential_error",
                                                            "first_error_class_counts", "realized_regret_mean",
                                                            "gap_regret_mean")}), flush=True)
        if rec_f:
            for r in rows:
                rec_f.write(json.dumps({"model": name, **r}) + "\n")
    if rec_f:
        rec_f.close()
    # pooled over models of the same rung
    pooled = {}
    for name, ent in result["models"].items():
        pooled.setdefault(ent["rung"], []).append(ent)
    result["by_rung"] = {}
    for rung, ents in pooled.items():
        agg = {}
        for pair in sorted({p for e in ents for p in e["by_pair"]}):
            cc = {c: sum(e["by_pair"][pair]["first_error_class_counts"][c] for e in ents if pair in e["by_pair"])
                  for c in CLASSES}
            agg[pair] = {"models": len(ents), "first_error_class_counts": cc,
                         "episodes_with_consequential_error": sum(e["by_pair"][pair]["episodes_with_consequential_error"]
                                                                  for e in ents if pair in e["by_pair"]),
                         "realized_regret_per_model": [e["by_pair"][pair]["realized_regret_mean"] for e in ents],
                         "gap_regret_per_model": [e["by_pair"][pair]["gap_regret_mean"] for e in ents],
                         "unique_opt_probe_rate_per_model": [e["by_pair"][pair]["first_probe_rate_when_unique_opt"]
                                                             for e in ents],
                         "n_probe_unique_opt_per_model": [e["by_pair"][pair]["n_probe_unique_opt"] for e in ents]}
        result["by_rung"][rung] = agg
    result["cpu_s"] = time.process_time() - t0
    result["cpu_s_label_load"] = t_load
    Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
