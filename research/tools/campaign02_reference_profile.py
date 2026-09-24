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
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import resource
import time

from topoformer import campaign02_references, campaign02_world, campaign02_protocol
from topoformer.campaign02_references import ReferencePolicy, run_episode
from topoformer.campaign02_world import Workshop, generate_world, protocol_executor


def cpu_seconds():
    parts = [resource.getrusage(k) for k in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN)]
    return sum(p.ru_utime+p.ru_stime for p in parts)


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
    for ci, condition in enumerate(config['conditions']):
        kwargs = {k:v for k,v in condition.items() if k!='name'}
        if kwargs.get('categories',2)*kwargs.get('choices',3)>20:
            raise ValueError('maximum20 inventory items')
        kwargs.setdefault('step_limit',64)
        for i in range(count):
            if cpu_seconds()-cpu >= limit:
                stopped = True
                break
            seed = config.get('seed_start',200000)+ci*10000+i
            spec = generate_world(seed,**kwargs)
            semantic_hash = hashlib.sha256(json.dumps(asdict(spec),sort_keys=True).encode()).hexdigest()
            for mode in modes:
                result = run_episode(Workshop(spec,protocol_executor),ReferencePolicy(mode))
                records.append(dict(condition=condition['name'],seed=seed,mode=mode,
                                    world_sha256=semantic_hash,**result))
        if stopped:
            break
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
    sources = {Path(inspect.getfile(module)).name: hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
               for module in (campaign02_references,campaign02_world,campaign02_protocol)}
    with gzip.open(output/'episodes.jsonl.gz','wt') as file:
        for row in records:
            file.write(json.dumps({k:v for k,v in row.items() if k!='trace'},separators=(',',':'))+'\n')
    result = dict(config=config,source_sha256=sources,cells=cells,stopped_cpu_limit=stopped,
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
