"""Artifact-only summaries of the latent-grounding study (no torch dependency).

Accuracy differences are treatment minus control, expressed as fractions.
Sample SD describes seed variability; it is not a confidence interval.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def describe(values):
    values = list(values)
    if not values or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in values):
        raise ValueError("expected nonempty finite numerical observations")
    return {"n": len(values), "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def validate_finite(value, path="row"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            validate_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            validate_finite(item, f"{path}[{i}]")


def paired_difference(treatment, control):
    """Pair seed-indexed observations; never replace missing seeds with zeros."""
    shared = sorted(treatment.keys() & control.keys())
    differences = [{"seed": seed, "difference": treatment[seed] - control[seed]} for seed in shared]
    return {"pairs": differences,
            "summary": describe(x["difference"] for x in differences) if differences else None,
            "missing_treatment": sorted(control.keys() - treatment.keys()),
            "missing_control": sorted(treatment.keys() - control.keys())}


METRICS = ("task_accuracy", "grounding_accuracy", "grounding_entropy", "pre_step_next_node_mass",
           "structural_next_node_mass", "relation_attention_mass", "clean_next_attention_mass",
           "exact_path_completion", "exact_attention_path_completion", "exact_pre_step_grounding", "null_mass",
           "key_grounding_accuracy", "distractor_null_accuracy", "oracle_task_accuracy",
           "oracle_exact_path_completion", "attention_oracle_task_accuracy",
           "attention_oracle_exact_path_completion", "mean_distinct_path_nodes")
CONDITION_FIELDS = ("condition", "depth", "nodes", "distractors", "composition", "corruption")


def condition_key(evaluation):
    return json.dumps({k: evaluation[k] for k in CONDITION_FIELDS}, sort_keys=True)


def summarize(rows, expected_seeds=None):
    if not rows:
        raise ValueError("no experiment rows")
    seeds = sorted(set(row["seed"] for row in rows)) if expected_seeds is None else sorted(expected_seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate expected seeds")
    configs, sources, seen = set(), set(), set()
    groups, by_variant = defaultdict(dict), defaultdict(set)
    diagnostics, examples = defaultdict(list), []
    for row in rows:
        validate_finite(row)
        variant, seed = row["variant"], row["seed"]
        if seed not in seeds:
            raise ValueError(f"unexpected seed {seed}")
        identity = (variant, seed)
        if identity in seen:
            raise ValueError(f"duplicate run {identity}")
        seen.add(identity)
        configs.add(row["config_hash"])
        sources.add(json.dumps(row["source"], sort_keys=True))
        if not row["source"] or not row["config_hash"] or not row["training"]["schedule_hash"]:
            raise ValueError("missing provenance or training schedule")
        if not row["evaluations"]:
            raise ValueError(f"run has no evaluations: {identity}")
        by_variant[variant].add(seed)
        evaluated = set()
        for evaluation in row["evaluations"]:
            key = condition_key(evaluation)
            if key in evaluated:
                raise ValueError(f"duplicate evaluation {identity} {key}")
            evaluated.add(key)
            for metric in METRICS:
                value = evaluation[metric]
                if metric == "distractor_null_accuracy" and evaluation["distractors"] == 0 and value is None:
                    continue
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                    raise ValueError(f"invalid metric {metric}")
                if metric not in {"grounding_entropy", "mean_distinct_path_nodes"} and not -1e-6 <= value <= 1 + 1e-6:
                    raise ValueError(f"probability out of range: {metric}")
            if not evaluation["data_hash"]:
                raise ValueError("evaluation lacks data hash")
            groups[(key, variant)][seed] = (evaluation, row)
            if [step["step"] for step in evaluation["step_diagnostics"]] != list(range(evaluation["depth"])):
                raise ValueError("incomplete or duplicate step diagnostics")
            for step in evaluation["step_diagnostics"]:
                diagnostics[(key, variant, step["step"])].append(step)
            examples.append({"variant": variant, "seed": seed,
                             "condition": json.loads(key), "trajectory": evaluation["example"]})
    if len(configs) != 1 or len(sources) != 1:
        raise ValueError("mixed configuration or source provenance; analyze runs separately")
    incomplete = {variant: sorted(set(seeds) - actual) for variant, actual in by_variant.items()
                  if actual != set(seeds)}
    if incomplete:
        raise ValueError(f"missing expected seeds: {incomplete}")
    conditions = sorted({key for key, _ in groups})
    variants = sorted(by_variant)
    aggregates = []
    for key in conditions:
        for variant in variants:
            observations = groups.get((key, variant), {})
            if set(observations) != set(seeds):
                raise ValueError(f"incomplete condition/seed grid for {variant}: {key}")
            aggregates.append({"condition": json.loads(key), "variant": variant,
                               "seeds": seeds,
                               "metrics": {metric: (describe(obs[0][metric] for obs in observations.values())
                                          if observations[seeds[0]][0][metric] is not None else None)
                                           for metric in METRICS}})
    comparisons = []
    # Every soft variant is explicit; do not combine random and identity initialization.
    control_variants = {"none", "graph_input", "known", "hard", "frozen", "permuted",
                        "none_keyed", "graph_input_keyed"}
    treatments = [v for v in variants if v not in control_variants]
    controls = [v for v in variants if v in control_variants]
    for key in conditions:
        for treatment in treatments:
            for control in controls:
                if treatment.endswith("_keyed") != control.endswith("_keyed"):
                    continue  # Supplementary content-key priors form a separate comparison family.
                left, right = groups[(key, treatment)], groups[(key, control)]
                common_hash_verified = True
                for seed in seeds:
                    a, arun = left[seed]
                    b, brun = right[seed]
                    if a["data_hash"] != b["data_hash"] or arun["training"]["schedule_hash"] != brun["training"]["schedule_hash"]:
                        raise ValueError(f"unpaired data/schedule for {treatment}/{control} seed {seed}")
                    ah = arun["training"].get("initial_parameter_hashes", {})
                    bh = brun["training"].get("initial_parameter_hashes", {})
                    common = ah.keys() & bh.keys()
                    common_hash_verified &= bool(common) and all(ah[k] == bh[k] for k in common)
                comparisons.append({"condition": json.loads(key), "treatment": treatment, "control": control,
                                    "initial_common_parameters_match": common_hash_verified,
                                    **paired_difference({s: x[0]["task_accuracy"] for s, x in left.items()},
                                                        {s: x[0]["task_accuracy"] for s, x in right.items()})})
    step_aggregates = []
    for (key, variant, step), entries in sorted(diagnostics.items()):
        metrics = sorted(set.intersection(*(set(e) for e in entries)) - {"step"})
        numeric = {metric: describe(e[metric] for e in entries)
                   for metric in metrics if all(isinstance(e[metric], (int, float)) and not isinstance(e[metric], bool) for e in entries)}
        step_aggregates.append({"condition": json.loads(key), "variant": variant, "step": step, "metrics": numeric})
    return {"schema_version": 1, "seeds": seeds, "variants": variants,
            "config_hash": next(iter(configs)), "source": json.loads(next(iter(sources))),
            "run_count": len(rows), "aggregates": aggregates, "paired_task_differences": comparisons,
            "step_diagnostics": step_aggregates, "examples": examples,
            "runs": [{k: row[k] for k in ("variant", "seed", "training", "resources", "model")} for row in rows]}


def fmt(summary):
    if summary is None:
        return "N/A"
    return f'{summary["mean"]:.4f}' + (f' ± {summary["sd"]:.4f}' if summary["sd"] is not None else " (one seed)")


def render_report(summary):
    lines = ["# Stage 3 artifact report", "",
             f'{summary["run_count"]} runs; seeds {summary["seeds"]}. Values are seed means ± sample SD. No significance claim is made.', "",
             "Task accuracy is primary. Exact path completion means all pre-step and post-step grounding argmaxes match their intended nodes, including the destination. Exact attention path completion separately requires every step’s mean-head attention argmax to select the clean next-node token. Exact pre-step grounding instead checks current-node grounding argmaxes and excludes the destination. Both are independent of answer accuracy. Pre-step next-node mass measures grounding before retrieval; structural next-node mass propagates grounding through the supplied relation.", "",
             "Computation is recurrent and externally scheduled by the supplied relation path. Deeper paths receive more computation. The deterministic exact-routing oracle is an algorithmic control, not a learned model. `known` uses a fixed identity matcher; `graph_input` uses the same fixed cosine soft identity matcher, a privileged prior rather than exact identity routing. `hard` argmax blocks gradients to its identity-initialized grounder. These controls are not parameter-identical learning mechanisms. In `graph_input`, relation-message values already contain successor information, so attending the current entity can retrieve the next value. Its next-token attention and attention-path diagnostics are therefore not mechanism-equivalent to direct structural-attention routing; low values do not establish failed message passing. Identity-initialized soft grounding and randomly initialized soft grounding must be interpreted separately. A high task score alone does not establish graph use or traversal. Identity generalization here concerns unseen random continuous keys and opaque numeric IDs; it is not a lexical-language test.", "",
             "## Accuracy and grounding", "",
             "| Condition | Depth | Nodes | Distractors | Composition | Corruption | Variant | Task accuracy | Grounding accuracy | Complete grounding trajectory | Complete attention trajectory | Exact pre-step grounding | Clean-next-token attention |", 
             "|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for group in summary["aggregates"]:
        c, m = group["condition"], group["metrics"]
        lines.append(f'| {c["condition"]} | {c["depth"]} | {c["nodes"]} | {c["distractors"]} | {c["composition"]} | {c["corruption"]} | {group["variant"]} | ' +
                     " | ".join(fmt(m[k]) for k in ("task_accuracy", "grounding_accuracy", "exact_path_completion", "exact_attention_path_completion", "exact_pre_step_grounding", "clean_next_attention_mass")) + " |")
    if any(variant.endswith("_keyed") for variant in summary["variants"]):
        lines += ["", "The `_keyed` supplement gives all three variants a shared, graph-free cosine identity content bias (β=8). Its soft structural strength starts at 16 to compete with attraction to the current identity. This is a separate comparison family, not a matched comparison to the primary runs. Paired effects compare `soft_keyed` only with `none_keyed` and `graph_input_keyed`; source/configuration mixing remains prohibited. The graph-input attention diagnostic caveat also applies to `graph_input_keyed`."]
    lines += ["", "## Exact routing controls", "", "Direct graph traversal and exact one-hot induced attention route the supplied graph. Under corruption their clean-answer accuracy may fall. Distinct path nodes quantify revisiting; instruction depth is not the number of unique entities visited.", "", "| Condition | Depth | Nodes | Direct oracle accuracy | Attention oracle accuracy | Attention oracle exact path | Distinct path nodes |", "|---|---:|---:|---:|---:|---:|---:|"]
    first_variant = summary["variants"][0]
    for group in summary["aggregates"]:
        if group["variant"] != first_variant:
            continue
        c, m = group["condition"], group["metrics"]
        lines.append(f'| {c["condition"]} | {c["depth"]} | {c["nodes"]} | ' + " | ".join(fmt(m[k]) for k in ("oracle_task_accuracy", "attention_oracle_task_accuracy", "attention_oracle_exact_path_completion", "mean_distinct_path_nodes")) + " |")
    lines += ["", "## Paired task-accuracy differences", "",
              "Treatment minus control; positive favors the soft treatment. Pairs have identical evaluation data and training schedules. Initialization matching is reported separately because grounding ablations can deliberately change parameters.", "",
              "| Condition | Depth | Nodes | Treatment | Control | Difference | Common initialization verified |", "|---|---:|---:|---|---|---:|---|"]
    for result in summary["paired_task_differences"]:
        c = result["condition"]
        lines.append(f'| {c["condition"]} | {c["depth"]} | {c["nodes"]} | {result["treatment"]} | {result["control"]} | {fmt(result["summary"])} | {result["initial_common_parameters_match"]} |')
    lines += ["", "## Null capacity and binding diagnostics", "",
              "Null mass and distractor-null accuracy diagnose forced binding; key accuracy diagnoses memory alignment. Entropy uses natural logarithms. These diagnostics do not replace task performance.", "",
              "| Condition | Depth | Nodes | Variant | Entropy | Null mass | Distractor null accuracy | Key grounding accuracy | Pre-step next-node mass | Structural next-node mass | Supplied-relation attention |", "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in summary["aggregates"]:
        c, m = group["condition"], group["metrics"]
        lines.append(f'| {c["condition"]} | {c["depth"]} | {c["nodes"]} | {group["variant"]} | ' + " | ".join(fmt(m[k]) for k in ("grounding_entropy", "null_mass", "distractor_null_accuracy", "key_grounding_accuracy", "pre_step_next_node_mass", "structural_next_node_mass", "relation_attention_mass")) + " |")
    lines += ["", "## Example trajectories", "", "Examples are predetermined first evaluation examples from the first seed, not selected for success. Node numbers index the current runtime graph and are not stable training labels."]
    for example in summary["examples"]:
        if example["seed"] != summary["seeds"][0] or example["condition"]["depth"] not in (4, 16, 32):
            continue
        if example["variant"] not in ("soft", "random_init", "known", "frozen", "soft_keyed", "graph_input_keyed"):
            continue
        lines += ["", f'### {example["variant"]}: {example["condition"]}', "",
                  "| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |", "|---:|---|---|---|---:|---:|---:|"]
        for row in example["trajectory"]:
            lines.append(f'| {row["step"]} | {row["gold_node"]} | {row["grounded_node"]} | {row["next_node"]} | {row["max_probability"]:.4f} | {row["entropy"]:.4f} | {row["next_attention_mass"]:.4f} |')
    lines += ["", "## Resource and model records", "", "Raw per-run records below preserve resource units, model parameters, learned strengths and temperatures. Resource comparisons include the runner’s stated instrumentation; no asymptotic scaling law is inferred."]
    resource_keys = ("seconds", "train_seconds", "inference_batch_seconds", "peak_rss_mib")
    lines += ["", "Process RSS is a high-water mark, not per-model allocated memory. Plain inference timing excludes diagnostic extraction; total run time includes evaluation and instrumentation.", "", "| Variant | Total seconds | Training seconds | Plain inference batch seconds | Process peak RSS MiB |", "|---|---:|---:|---:|---:|"]
    for variant in summary["variants"]:
        runs = [run for run in summary["runs"] if run["variant"] == variant]
        values = [describe(run["resources"][k] for run in runs) if all(k in run["resources"] for run in runs) else None for k in resource_keys]
        lines.append("| " + variant + " | " + " | ".join(fmt(value) for value in values) + " |")
    for run in summary["runs"]:
        lines += ["", f'### {run["variant"]}, seed {run["seed"]}', "", "```json", json.dumps({k: run[k] for k in ("resources", "model")}, indent=2, sort_keys=True), "```"]
    lines += ["", "## Provenance", "", "```json", json.dumps({"source": summary["source"], "config_hash": summary["config_hash"], "analysis_artifact": summary.get("artifact")}, indent=2, sort_keys=True), "```", ""]
    return "\n".join(lines)


def write_plots(summary, directory):
    """Optional static scientific figures; no renderer required for analysis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"available": False, "files": []}
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    core = {"soft", "random_init", "known", "hard", "none", "graph_input", "frozen", "permuted",
            "soft_keyed", "none_keyed", "graph_input_keyed", "soft_strength4"}
    for axis in ("depth", "nodes", "corruption", "distractors"):
        panels = defaultdict(lambda: defaultdict(list))
        fixed = [key for key in CONDITION_FIELDS if key not in ("condition", axis)]
        for row in summary["aggregates"]:
            if row["variant"] not in core:
                continue
            c = row["condition"]
            key = json.dumps({k: c[k] for k in fixed}, sort_keys=True)
            panels[key][row["variant"]].append((c[axis], row["metrics"]))
        for panel, series in sorted(panels.items()):
            if max(len(values) for values in series.values()) < 2:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            for variant, points in sorted(series.items()):
                points.sort(key=lambda p: p[0])
                for ax, metric, title in zip(axes, ("task_accuracy", "exact_path_completion"),
                                              ("Task accuracy", "Complete grounding trajectory")):
                    ax.errorbar([p[0] for p in points], [p[1][metric]["mean"] for p in points],
                                yerr=[p[1][metric]["sd"] or 0 for p in points], marker="o", capsize=2, label=variant)
                    ax.set(xlabel=axis, ylabel=title, ylim=(-.02, 1.02))
                    ax.grid(alpha=.2)
            axes[-1].legend(fontsize=7)
            fig.suptitle(panel, fontsize=8)
            stem = f'{axis}-{hashlib.sha256(panel.encode()).hexdigest()[:8]}'
            for ext in ("png", "svg"):
                path = directory / f"{stem}.{ext}"
                fig.savefig(path, dpi=160)
                files.append(str(path.name))
            plt.close(fig)
    # Layer diagnostics: each point describes current-state grounding before routing.
    panels = defaultdict(lambda: defaultdict(list))
    for row in summary["step_diagnostics"]:
        c = row["condition"]
        if c["depth"] not in (16, 32) or c["corruption"] or c["composition"] != "train" or row["variant"] not in core:
            continue
        panels[json.dumps(c, sort_keys=True)][row["variant"]].append(row)
    for panel, series in sorted(panels.items()):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for variant, rows in sorted(series.items()):
            rows.sort(key=lambda row: row["step"])
            for ax, metric in zip(axes, ("grounding_accuracy", "clean_next_attention_mass")):
                ax.plot([r["step"] for r in rows], [r["metrics"][metric]["mean"] for r in rows], label=variant)
                ax.set(xlabel="Relation instruction / recurrent step (zero based)", ylabel=metric, ylim=(-.02, 1.02))
                ax.grid(alpha=.2)
        axes[-1].legend(fontsize=7)
        fig.suptitle(panel, fontsize=8)
        stem = f'diagnostics-{hashlib.sha256(panel.encode()).hexdigest()[:8]}'
        for ext in ("png", "svg"):
            path = directory / f"{stem}.{ext}"
            fig.savefig(path, dpi=160)
            files.append(str(path.name))
        plt.close(fig)
    return {"available": True, "files": files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
    summary = summarize(rows, expected_seeds=args.seeds)
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary["artifact"] = {"metrics_sha256": hashlib.sha256(args.metrics.read_bytes()).hexdigest(),
                           "analyzer_file": "src/topoformer/grounding_analysis.py",
                           "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    summary["plots"] = {"available": False, "files": []} if args.no_plots else write_plots(summary, args.outdir / "figures")
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    report = render_report(summary)
    if summary["plots"]["files"]:
        report += "\n## Figures\n\n" + "\n".join(f'- [{name}](figures/{name})' for name in summary["plots"]["files"]) + "\n"
    (args.outdir / "report.md").write_text(report)
    print(json.dumps({"runs": summary["run_count"], "output": str(args.outdir), "plots": len(summary["plots"]["files"])}))


if __name__ == "__main__":
    main()
