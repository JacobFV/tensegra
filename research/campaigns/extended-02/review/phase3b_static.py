#!/usr/bin/env python3
"""Phase-3b independent static checks: training-config diffs and encoder source facts.

Stdlib only, written fresh for this audit. Run from the repository root:
    python3 research/campaigns/extended-02/review/phase3b_static.py > /tmp/phase3b-static.json
"""
import ast
import json
import sys
import time

CFG = "configs/campaign02/"
SRC = "src/topoformer/campaign02_modular.py"
T0 = time.process_time()


def flatten(x, p=""):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(flatten(v, f"{p}.{k}"))
        return out
    if isinstance(x, list):
        out = {}
        for i, v in enumerate(x):
            out.update(flatten(v, f"{p}[{i}]"))
        return out
    return {p: x}


def diff(a, b):
    A, B = flatten(json.load(open(CFG + a))), flatten(json.load(open(CFG + b)))
    miss = object()
    return {k: [A.get(k, "<absent>"), B.get(k, "<absent>")] for k in sorted(set(A) | set(B))
            if A.get(k, miss) != B.get(k, miss)}


def classify(d, allowed_prefixes):
    return sorted({k for k in d if not any(k.startswith(p) for p in allowed_prefixes)})


def config_checks():
    out = {}
    for kind in ("boot", "rl-single"):
        for r in range(3):
            d1720 = diff(f"e17-{kind}-r{r}.json", f"e20-{kind}-r{r}.json")
            d2022 = diff(f"e20-{kind}-r{r}.json", f"e22-{kind}-r{r}.json")
            # initial_checkpoints differ legitimately (each experiment bootstraps its own teacher-imitation start)
            ok1720 = [".address_namespace", ".policy.feature_version", ".initial_checkpoints"]
            other1720 = [k for k in classify(d1720, ok1720) if not (k.startswith(".world_mix[") and k.endswith(".same_type_distractors"))]
            std_vals = {tuple(v) for k, v in d1720.items() if k.endswith(".same_type_distractors")}
            ckpt_paths_1720 = sorted({v[1] for k, v in d1720.items() if k.startswith(".initial_checkpoints") and k.endswith(".path")})
            out[f"e17->e20 {kind} r{r}"] = {
                "n_diffs": len(d1720), "unexpected_keys": other1720,
                "feature_version": d1720.get(".policy.feature_version"),
                "same_type_distractors_values": sorted(map(list, std_vals)),
                "n_world_mix_entries_changed": sum(k.endswith(".same_type_distractors") for k in d1720),
                "initial_checkpoint_sources": ckpt_paths_1720}
            ok2022 = [".address_namespace", ".policy.feature_version", ".initial_checkpoints"]
            out[f"e20->e22 {kind} r{r}"] = {
                "n_diffs": len(d2022), "unexpected_keys": classify(d2022, ok2022),
                "feature_version": d2022.get(".policy.feature_version"),
                "initial_checkpoint_sources": sorted({v[1] for k, v in d2022.items() if k.endswith(".path")})}
    for r in range(3):
        d = diff(f"e15-halving-plain-r{r}.json", f"e21-halving-robust-r{r}.json")
        e21 = json.load(open(CFG + f"e21-halving-robust-r{r}.json"))
        e13 = json.load(open(CFG + f"e13-robust-fitness-pbt-r{r}.json"))
        mix = e21["development_world_mix"]
        out[f"e15-plain->e21 r{r}"] = {
            "n_diffs": len(d), "unexpected_keys": classify(d, [".development_world_mix"]),
            "dev_mix_entries": len(mix),
            "robustness_entries": [m for m in mix if m.get("work_limit") == 0 or m.get("categories") == 5],
            "dev_mix_equals_e13": mix == e13.get("development_world_mix")}
    return out


def src_checks():
    text = open(SRC).read()
    tree = ast.parse(text)
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    def strings(fn):
        return sorted({n.value for n in ast.walk(funcs[fn]) if isinstance(n, ast.Constant) and isinstance(n.value, str)})

    def attrs(fn):
        return sorted({n.attr for n in ast.walk(funcs[fn]) if isinstance(n, ast.Attribute)})

    def calls(fn):
        return sorted({(n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "?"))
                       for n in ast.walk(funcs[fn]) if isinstance(n, ast.Call)})
    enc = {}
    for fn in ("encode_action", "encode_action_m2", "encode_observation", "encode_observation_m3", "encode_action_m3",
               "encode_public", "_action_stage"):
        enc[fn] = {"string_constants": strings(fn), "attributes": attrs(fn), "calls": calls(fn),
                   "mentions_prior": "prior" in strings(fn) or "prior" in attrs(fn),
                   "touches_private_state": any(a.startswith("_") for a in attrs(fn))}
    # m3 counter update: find the step() method of ModularWorkshop and quote its counter logic
    step_src = ast.get_source_segment(text, funcs["step"])
    counter_lines = [ln.strip() for ln in step_src.splitlines() if "_stage_rejections" in ln or "_stage_calls" in ln
                     or "len(self._completed) != before" in ln or "action.kind in" in ln or "self._feedback.get" in ln]
    obs_src = ast.get_source_segment(text, funcs["observe"])
    catalog_src = ast.get_source_segment(text, funcs["action_catalog"])
    init_src = text  # ModularWorkshop.__init__ is shadowed by later __init__ defs in funcs; search whole module
    return {"encoders": enc, "step_counter_lines": counter_lines,
            "observe_exposes_counters": "self._stage_rejections, self._stage_calls" in obs_src,
            "observe_strips_payload_only": 'if k != "payload"' in obs_src,
            "prior_records_named": "problem=f\"prior_{i}\"" in init_src,
            "draft_handle_pattern": "problem_{len(o.problems)}" in catalog_src}


def main():
    out = {"configs": config_checks(), "source": src_checks()}
    out["cpu_seconds"] = time.process_time() - T0
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
