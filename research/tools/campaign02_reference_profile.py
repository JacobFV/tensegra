"""CPU-only registered reference profile; no launch on import.

Example config: {"episodes":32,"seed_start":200000,"cpu_limit_seconds":120,
"conditions":[{"name":"easy","categories":2,"choices":2,"locations":5,
"step_limit":64}],"modes":["cheap","always_tool","cheap_first"]}.
Every mode receives identical generated worlds. This executable refuses to
replace existing outputs. Root coordinator owns launch and resource scheduling.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from contextlib import nullcontext
from functools import partial
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import time

from topoformer import campaign02_references, campaign02_world, campaign02_protocol
from topoformer.campaign02_references import ReferencePolicy, make_reference, run_episode
from topoformer.campaign02_world import Workshop, generate_world, protocol_executor


def cpu_seconds(owner=None):
    parts = [resource.getrusage(k) for k in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN)]
    total = sum(p.ru_utime+p.ru_stime for p in parts)
    process = getattr(owner,'_process',None)
    if process is not None and process.is_alive():
        # Linux coordinator accounting only; never enters actor observations.
        # Child is not yet included in RUSAGE_CHILDREN until joined.
        try:
            fields = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()
            total += (int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')
        except FileNotFoundError:
            pass  # Final close/join makes completed child usage authoritative.
    return total


def profile(config, output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    wall, cpu = time.monotonic(), cpu_seconds()
    records, cells = [], []
    modes = config.get('modes',['cheap','always_tool','cheap_first'])
    count = config.get('episodes',32)
    if count < 1 or count > 1024:
        raise ValueError('bounded episode count required')
    limit = config.get('cpu_limit_seconds',120)
    stopped = False
    executor_kind = config.get('executor', 'isolated')
    if executor_kind not in ('isolated','persistent'):
        raise ValueError('executor must be isolated or persistent')
    owner = campaign02_protocol.BoundedSolver() if executor_kind=='persistent' else None
    context = owner if owner is not None else nullcontext()
    executor = partial(protocol_executor,execute_call=owner.execute) if owner else protocol_executor
    executor_stats = {'kind':executor_kind}
    with context:
        for ci, condition in enumerate(config['conditions']):
            kwargs = {k:v for k,v in condition.items() if k!='name'}
            if kwargs.get('categories',2)*kwargs.get('choices',3)>20:
                raise ValueError('maximum20 inventory items')
            kwargs.setdefault('step_limit',64)
            if 'call_budgets' in kwargs:
                kwargs['call_budgets'] = tuple(kwargs['call_budgets'])
            for i in range(count):
                if cpu_seconds(owner)-cpu >= limit:
                    stopped = True
                    break
                seed_offset = 0 if config.get('shared_condition_seeds',False) else ci*10000
                seed = config.get('seed_start',200000)+seed_offset+i
                spec = generate_world(seed,**kwargs)
                semantic_hash = hashlib.sha256(json.dumps(asdict(spec),sort_keys=True).encode()).hexdigest()
                for mode in modes:
                    address_seed = config.get('address_seed_start',70000000)+seed_offset+i
                    result = run_episode(Workshop(spec,executor,address_seed=address_seed),make_reference(mode),
                                         model_compute_tariff=config.get('model_compute_tariff',0.0))
                    records.append(dict(condition=condition['name'],seed=seed,address_seed=address_seed,mode=mode,
                                        world_sha256=semantic_hash,instance_key=str(seed),**result))
            if stopped:
                break
    if owner is not None:
        executor_stats.update(startup_wall_seconds=owner.startup_wall_seconds,
                              startup_child_cpu_seconds=owner.startup_child_cpu_seconds,
                              call_wall_seconds=owner.call_wall_seconds,restarts=owner.restarts)
    for condition in config['conditions']:
        for mode in modes:
            rows = [r for r in records if r['condition']==condition['name'] and r['mode']==mode]
            if not rows:
                continue
            summary = dict(condition=condition['name'],mode=mode,n=len(rows),
                           successes=sum(r['verified_success'] for r in rows))
            for key in ('utility','cost','work_units','steps','observations','travel_distance',
                        'controller_cpu_seconds','solver_cpu_seconds','episode_wall_seconds'):
                summary['mean_'+key] = sum(r.get(key,0) for r in rows)/len(rows)
            summary['solver_calls'] = sum(sum(t['action']['kind']=='call' for t in r['trace']) for r in rows)
            cells.append(summary)
    paired = []
    by_cell = {}
    for row in records:
        by_cell.setdefault((row['condition'],row['mode']),{})[row['instance_key']] = row
    keys = list(by_cell)
    for ai,a in enumerate(keys):
        for b in keys[ai+1:]:
            common = sorted(by_cell[a].keys() & by_cell[b].keys())
            if not common:
                continue
            counts = {'both_correct':0,'a_only':0,'b_only':0,'both_wrong':0}
            for key in common:
                x,y = bool(by_cell[a][key]['verified_success']),bool(by_cell[b][key]['verified_success'])
                counts['both_correct' if x and y else 'a_only' if x else 'b_only' if y else 'both_wrong'] += 1
            paired.append(dict(a=list(a),b=list(b),paired_instances=len(common),outcomes=counts,
                               success_difference_b_minus_a=(counts['b_only']-counts['a_only'])/len(common)))
    sources = {Path(inspect.getfile(module)).name: hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
               for module in (campaign02_references,campaign02_world,campaign02_protocol)}
    sources[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with gzip.open(output/'episodes.jsonl.gz','wt') as file:
        for row in records:
            file.write(json.dumps({k:v for k,v in row.items() if k!='trace'},separators=(',',':'))+'\n')
    result = dict(config=config,executor=executor_stats,source_sha256=sources,cells=cells,
                  paired_success=paired,unique_seed_instances=len({r['seed'] for r in records}),
                  support_note='Same seed across policies/conditions is paired support, not independent episodes. '
                  'Shared-condition seeds equate hidden worlds only when generator parameters agree; public catalogue metadata may differ.',
                  stopped_cpu_limit=stopped,
                  cpu_core_seconds=cpu_seconds()-cpu,wall_seconds=time.monotonic()-wall,
                  accounting='self+reaped children CPU; includes generation, policy, solver, serialization')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    path=Path(args.config)
    config=json.loads(path.read_text())
    result=profile(config,args.output)
    result['config_file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    Path(args.output,'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('cells','cpu_core_seconds','wall_seconds','stopped_cpu_limit')},indent=2))
