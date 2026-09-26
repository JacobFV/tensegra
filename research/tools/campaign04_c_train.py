"""Train Track C appraisal/controller variants for one base (CPU; extended-04 trackc.md §4-5).

    CUDA_VISIBLE_DEVICES= python research/tools/campaign04_c_train.py \
        --labels results/c-labels-p1-boot-x1-r0 --base p1-boot-x1-r0 \
        --variants main shuffled drop:policy ... --seeds 0 1 2 --output results/c-train-p1-boot-x1-r0

Variants:
  main          telemetry -> GRU(64) -> heads; margin m and tau registered on the dev split
  shuffled      shuffled-target control (targets permuted across positions/points; inputs intact)
  drop:<group>  retrained with telemetry group <group> zeroed in training and evaluation (necessity)

The dev split (20% of label worlds, hash of the world seed) is used for the registered
margin m, tau and the dev metrics only; evaluation worlds are never touched here.
Writes <variant>-s<seed>.pt (state dict, spec, registration, dev metrics) + .json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign04_c_configs as C  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, nargs="+", required=True, help="label directories or .pt files")
    p.add_argument("--base", required=True)
    p.add_argument("--variants", nargs="+", default=["main"])
    p.add_argument("--seeds", type=int, nargs="+", default=list(C.META_SEEDS))
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--dev-fraction", type=float, default=0.2)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args(argv)
    import random
    import torch
    from tensegra.campaign04_meta import META_VERSION, MetaSpec, parameter_count
    from tensegra.campaign04_meta_train import Tensors, dev_metrics, merge, register, split, train
    torch.set_num_threads(a.threads)
    for v in a.variants:
        if v not in ("main", "shuffled") and not (v.startswith("drop:") and v[5:] in C.TELEMETRY_GROUPS):
            raise SystemExit(f"unknown variant {v!r}")
    files = []
    for x in a.labels:
        files += sorted(x.glob("labels-c*.pt")) if x.is_dir() else [x]
    if not files:
        raise SystemExit("no label chunks")
    wall0, cpu0 = time.perf_counter(), time.process_time()
    packs = [torch.load(f, weights_only=False) for f in files]
    for pk in packs:
        if pk["meta"].get("base") != a.base:
            raise SystemExit(f"label chunk for {pk['meta'].get('base')} given for base {a.base}")
    data = merge(packs)
    train_eps, dev_eps = split(data, a.dev_fraction)
    a.output.mkdir(parents=True, exist_ok=True)
    label_hashes = {str(f): C.file_hash(f) for f in files}
    base = C.base(a.base)
    actor_params = packs[0]["meta"]["actor"]["parameters"]
    written = []
    for variant in a.variants:
        for seed in a.seeds:
            out = a.output / f"{variant.replace(':', '-')}-s{seed}"
            if out.with_suffix(".pt").exists():
                raise SystemExit(f"refusing to overwrite {out}.pt")
            t0, c0 = time.perf_counter(), time.process_time()
            tensors = Tensors(data)
            if variant == "shuffled":
                from tensegra.campaign04_meta import digest_seed
                tensors.shuffle_targets(train_eps, random.Random(digest_seed("c-shuffle-targets", seed)))
            drop = (variant[5:],) if variant.startswith("drop:") else ()
            spec = MetaSpec(drop_groups=drop)
            model, history = train(tensors, train_eps, spec, seed=seed, epochs=a.epochs, lr=a.lr, batch=a.batch)
            # dev metrics/registration always against the TRUE dev targets
            metrics, rows = dev_metrics(model, Tensors(data), data, dev_eps)
            reg = register(rows)
            record = {"meta_version": META_VERSION, "variant": variant, "seed": seed, "base": a.base,
                      "base_sha256": base["sha256"], "base_type": base["base_type"], "spec": spec.__dict__,
                      "parameters": parameter_count(model), "actor_parameters": actor_params,
                      "epochs": a.epochs, "lr": a.lr, "batch": a.batch, "train_loss": history,
                      "train_episodes": len(train_eps), "dev_episodes": len(dev_eps),
                      "points": int(len(data["pt_episode"])), "registration": reg, "dev": metrics,
                      "label_files": label_hashes, "sources": C.source_hashes(),
                      "stop_gradient": "inputs are telemetry numbers (actor distribution computed under no_grad); "
                                       "no actor parameter is reachable from any loss",
                      "wall_seconds": time.perf_counter() - t0, "process_cpu_seconds": time.process_time() - c0}
            torch.save({"state_dict": model.state_dict(), "spec": spec.__dict__, "record": record},
                       out.with_suffix(".pt"))
            out.with_suffix(".json").write_text(json.dumps(record, indent=2, default=str))
            written.append(out.name)
            print(json.dumps({"model": out.name, "m": reg["margin"], "tau": reg["tau"],
                              "dev_gain_m": reg["margin_dev_gain"], "ece": metrics["ece_success_steps"],
                              "adv_mae": metrics["adv_mae_points"], "loss": history[-1],
                              "wall": record["wall_seconds"]}, default=str), flush=True)
    print(json.dumps({"written": written, "wall_seconds": time.perf_counter() - wall0,
                      "process_cpu_seconds": time.process_time() - cpu0}))


def load_model(path):
    """(model, record) from a campaign04_c_train output file."""
    import torch
    from tensegra.campaign04_meta import MetaSpec, make_model
    blob = torch.load(path, weights_only=False)
    spec = dict(blob["spec"])
    spec["drop_groups"] = tuple(spec.get("drop_groups", ()))
    model = make_model(MetaSpec(**spec))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob["record"]


if __name__ == "__main__":
    main()
