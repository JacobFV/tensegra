#!/usr/bin/env python3
"""Bounded instrumentation of frozen Stage 6 checkpoints; never trains."""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=2, choices=(1, 2))
    parser.add_argument('--examples', type=int, default=2)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.source) / 'src'))
    import torch
    from topoformer import thinking_study as study
    from topoformer.thinking_runtime import ProtectedSession
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    rows = []
    manifests = {}
    profiling_start = time.perf_counter()
    for seed in (0, 1, 2):
        folder = Path(args.results) / f'stage6-seed{seed}'
        manifest = json.loads((folder / 'manifest.json').read_text())
        manifests[str(seed)] = {key: manifest.get(key) for key in ('source_hash', 'git_commit', 'config_hash')}
        config = study.ThinkingStudyConfig(**manifest['config'])
        checkpoint = folder / f'local-seed{seed}-step{config.steps}.pt'
        checkpoint_sha = sha(checkpoint)
        expected = manifest.get('checkpoints', {}).get(checkpoint.name, {}).get('file_sha256')
        if expected is not None and expected != checkpoint_sha:
            raise ValueError(f'Checkpoint hash mismatch: {checkpoint}')
        model = study.build_model(config, seed)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        model.eval()
        for depth in (4, 32):
            for index in range(args.examples):
                condition = dict(depth=depth, condition='depth', kwargs={})
                episode = study.evaluation_episode(config, seed, index, condition)
                for mode in ('free', 'oracle_minimal'):
                    cap = config.eval_max_microsteps if mode == 'free' else len(episode.public.frames) + len(episode.gold.trace)
                    local_config = replace(config, max_microsteps=cap)
                    # Main oracle_minimal preserves the evaluation model's hard cap;
                    # the driver supplies the shorter privileged unroll separately.
                    model.config = replace(model.config, max_microsteps=config.eval_max_microsteps)
                    totals = defaultdict(lambda: dict(calls=0, wall_seconds=0., process_cpu_seconds=0.))
                    original_step = model.step
                    original_execute = ProtectedSession.execute
                    original_inject = study.inject_runtime_events
                    def wrapper(name, function):
                        def timed(*a, **k):
                            wall = time.perf_counter(); cpu = time.process_time()
                            try:
                                return function(*a, **k)
                            finally:
                                elapsed_cpu = time.process_time() - cpu
                                elapsed_wall = time.perf_counter() - wall
                                row = totals[name]
                                row['calls'] += 1
                                row['wall_seconds'] += elapsed_wall
                                row['process_cpu_seconds'] += elapsed_cpu
                        return timed
                    model.step = wrapper('neural_cell', original_step)
                    ProtectedSession.execute = wrapper('symbolic_session', original_execute)
                    study.inject_runtime_events = wrapper('event_reintegration_adapter', original_inject)
                    wall = time.perf_counter(); cpu = time.process_time()
                    try:
                        with torch.no_grad():
                            result = study.rollout(model, episode.public, local_config, 'local',
                                gold=episode.gold if mode == 'oracle_minimal' else None,
                                teacher_forcing=mode == 'oracle_minimal')
                    finally:
                        total_cpu = time.process_time() - cpu
                        total_wall = time.perf_counter() - wall
                        model.step = original_step
                        ProtectedSession.execute = original_execute
                        study.inject_runtime_events = original_inject
                    for name in ('neural_cell', 'symbolic_session', 'event_reintegration_adapter'):
                        _ = totals[name]
                    totals['other_rollout_and_instrumentation'] = dict(calls=1,
                        wall_seconds=total_wall-sum(v['wall_seconds'] for v in totals.values()),
                        process_cpu_seconds=total_cpu-sum(v['process_cpu_seconds'] for v in totals.values()))
                    events = [event for entry in result['trace'] for event in entry['events']]
                    rows.append(dict(seed=seed, depth=depth, example=index, mode=mode, compute_cap=cap,
                        checkpoint=checkpoint.name, checkpoint_sha256=checkpoint_sha,
                        public_hash=study.digest(asdict(episode.public)), data_hash=study.digest(asdict(episode)),
                        microsteps=result['microsteps'], executed_events=sum(e['status']=='executed' for e in events),
                        duplicate_events=sum(e['status']=='duplicate' for e in events),
                        proposed_execution_events=len(events),
                        task_correct=result['numeric_prediction']==episode.gold.result and result['answer_prediction']==episode.gold.answer,
                        exact_result_correct=result['semantic_result']==episode.gold.result,
                        post_event_pass=result['post_event_pass'],
                        total_wall_seconds=total_wall,total_process_cpu_seconds=total_cpu,categories=dict(totals)))
    wall = time.perf_counter(); cpu = time.process_time()
    for _ in range(10000):
        w = time.perf_counter(); c = time.process_time()
        _ = time.process_time()-c; _ = time.perf_counter()-w
    overhead = dict(iterations=10000,wall_seconds=time.perf_counter()-wall,process_cpu_seconds=time.process_time()-cpu)
    output = dict(schema_version=1,source_manifests=manifests,script_sha256=sha(__file__),
        frozen_source_sha256={str(p.relative_to(Path(args.source))):sha(p) for p in sorted((Path(args.source)/'src'/'topoformer').glob('thinking*.py'))},
        platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,threads=args.threads,
        examples_per_seed_depth_mode=args.examples,elapsed_wall_seconds=time.perf_counter()-profiling_start,
        timer_calibration=overhead,
        caveats=[
            'Concurrent training on the host; this is cost decomposition, not a throughput benchmark.',
            'Process CPU time includes CPU consumed by Torch worker threads; wall time includes scheduling delays.',
            'Three timed categories are disjoint; other includes initialization, tensor adapters, audit/loss computation and timer overhead.',
            'Neural cell excludes event encoding; event reintegration adapter includes learned encoding, injection and encoder reconstruction diagnostics.',
            'Oracle rows execute privileged actions and compute auxiliary audits, so their other time is not comparable to free policy overhead.',
            'One measurement per example, no warmup exclusion; module/kernel cold starts remain included.'
        ],rows=rows)
    target = Path(args.output); target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(output=str(target),rows=len(rows),elapsed_wall_seconds=output['elapsed_wall_seconds'])))

if __name__ == '__main__':
    main()
