"""Replay a registered frozen supervised teacher stream, without Torch/models.

Solver and generator work is real and must be launched/accounted by coordinator.
Imports frozen trusted repository modules from a temporary package so spawned
bounded workers can import the same source. No candidate gets filesystem access.
"""
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial
import argparse
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def cpu(owner=None):
    usage=[resource.getrusage(k) for k in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN)]
    result=sum(x.ru_utime+x.ru_stime for x in usage)
    process=getattr(owner,'_process',None)
    if process is not None and process.is_alive():
        try:
            fields=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()
            result+=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')
        except FileNotFoundError:pass
    return result


@contextmanager
def frozen_modules(source):
    names=('campaign02_world','campaign02_protocol','campaign02_references',
           'campaign02_training','campaign02_population')
    with tempfile.TemporaryDirectory(prefix='topoformer-teacher-replay-') as directory:
        pkg=Path(directory)/'_frozen_teacher_replay';pkg.mkdir();(pkg/'__init__.py').write_text('')
        hashes={}
        for name in names:
            code=subprocess.check_output(['git','show',source+':src/topoformer/'+name+'.py'])
            hashes[name]=hashlib.sha256(code).hexdigest()
            if name not in ('campaign02_training','campaign02_population'):
                (pkg/(name+'.py')).write_bytes(code)
        sys.path.insert(0,directory)
        try:
            modules=[importlib.import_module('_frozen_teacher_replay.'+n) for n in names[:3]]
            yield *modules,hashes
        finally:
            sys.path.remove(directory)
            for name in list(sys.modules):
                if name.startswith('_frozen_teacher_replay'):sys.modules.pop(name,None)


def replay(config_path, source, output, limit=120, expected_data_hash=None):
    cpu_start,wall=cpu(),time.monotonic()
    config=json.loads(Path(config_path).read_text());output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if config['mode']!='single' or config['rounds']!=1 or config.get('methods')!=['supervised']:
        raise ValueError('This replay supports the declared single-round supervised E05 stream only')
    slots=config['population_size'];tranche=config['updates_per_slot']*config['train']['batch_size']
    count=slots*tranche;start=config['training_seed_start'];max_steps=config['train']['max_steps']
    if count!=4800 or start!=310000000:
        raise ValueError('Expected registered E05 4800-episode stream starting310000000')
    history=hashlib.sha256(hashlib.sha256().hexdigest().encode())
    kinds=Counter();budgets=Counter();lasts=Counter();subtypes=Counter();mixtures=Counter()
    work=success=presentations=episodes=0;checkpoints=[];exhausted=False
    with frozen_modules(source) as (world,protocol,refs,hashes):
        with protocol.BoundedSolver() as solver,gzip.open(output/'teacher-episodes.jsonl.gz','wt') as file:
            execute=partial(world.protocol_executor,execute_call=solver.execute)
            for i in range(count):
                if cpu(solver)-cpu_start>=limit:exhausted=True;break
                if i and i%tranche==0:
                    checkpoints.append({'episodes':i,'data_hash':history.hexdigest()})
                    history=hashlib.sha256(history.hexdigest().encode())
                seed=start+i
                mix=int(digest({'world_seed':seed,'role':'public-world-mixture'})[:16],16)%len(config['world_mix'])
                mixtures[mix]+=1
                address=int(digest({'namespace':config['address_namespace'],'world_seed':seed,'role':'record-addresses'})[:16],16)
                env=world.Workshop(world.generate_world(seed,**config['world_mix'][mix]),execute,address_seed=address)
                teacher=refs.make_reference(config['teacher']);o=env.observe();steps=[];call_states=[]
                for step in range(max_steps):
                    if o.done:break
                    catalog=world.action_catalog(o);action=teacher.choose(o,catalog);index=catalog.index(action)
                    kinds[action.kind]+=1
                    steps.append({'observation_hash':digest(o.to_dict()),'action':asdict(action),'index':index})
                    if action.kind=='call':
                        target=action.arguments['budget'];budgets[target]+=1
                        p=o.problems[action.arguments['problem']]
                        records=[r for r in o.records if r.get('problem')==action.arguments['problem'] and r.get('problem_snapshot')==p]
                        last=records[-1] if records else None
                        label='initial' if not last else 'after_timeout' if last.get('status')=='timeout' else 'other_repeat'
                        subtypes[label]+=1;lasts[str(last.get('budget',0) if last else 0)+' -> '+str(target)]+=1
                        alternatives=[a.arguments['budget'] for a in catalog if a.kind=='call' and a.arguments['problem']==action.arguments['problem']]
                        features=world.encode_action(o,action)
                        call_states.append({'step':step,'primitive':p['primitive'],'requested':target,'available':alternatives,
                            'last_budget':last.get('budget') if last else None,'last_status':last.get('status') if last else None,
                            'budget_feature':features[20],'last_budget_feature':features[52],'timeout_feature':features[51]})
                    o=env.step(action)
                outcome=env.evaluate();success+=outcome['verified_success'];work+=outcome['work_units']
                presentations+=len(steps);episodes+=1
                history.update(json.dumps(steps,sort_keys=True).encode())
                file.write(json.dumps({'seed':seed,'address_seed':address,'mixture':mix,'steps':steps,
                    'call_states':call_states,'verified_success':outcome['verified_success'],
                    'work_units':outcome['work_units'],'truncated':not o.done},separators=(',',':'))+'\n')
            checkpoints.append({'episodes':episodes,'data_hash':history.hexdigest()})
        solver_stats={'startup_wall_seconds':solver.startup_wall_seconds,'startup_child_cpu_seconds':solver.startup_child_cpu_seconds,
                      'call_wall_seconds':solver.call_wall_seconds,'restarts':solver.restarts}
    summary={'source':source,'source_hashes':hashes,'protocol_sha256':hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        'episodes':episodes,'expected_episodes':count,'presentations':presentations,'successes':success,'solver_work_units':work,
        'action_counts':dict(kinds),'call_budget_counts':dict(budgets),'call_subtypes':dict(subtypes),
        'last_to_requested_budget':dict(lasts),'mixture_counts':dict(mixtures),'checkpoint_data_hashes':checkpoints,
        'final_data_hash':history.hexdigest(),'expected_data_hash':expected_data_hash,
        'data_hash_match':None if expected_data_hash is None else expected_data_hash==history.hexdigest(),
        'stopped_cpu_limit':exhausted,'executor':solver_stats,'cpu_core_seconds':cpu()-cpu_start,'wall_seconds':time.monotonic()-wall,
        'scope':'Exact deterministic teacher-stream replay, no gradient learning/neural inference. Hash match required before calling this actual training exposure.',
        'trajectory_sha256':hashlib.sha256((output/'teacher-episodes.jsonl.gz').read_bytes()).hexdigest()}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--source',default='a9337b71');p.add_argument('--output',required=True)
    p.add_argument('--cpu-limit',type=float,default=120);p.add_argument('--expected-data-hash')
    a=p.parse_args();result=replay(a.config,a.source,a.output,a.cpu_limit,a.expected_data_hash)
    print(json.dumps(result,indent=2))
