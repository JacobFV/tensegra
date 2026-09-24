"""Cheap public codec shape profile. No neural forward or real solver execution."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

from topoformer.campaign02_memory import encode_memory,decode_memory,normalized_document,MemoryCapacityError,MemoryLimits
from topoformer.campaign02_world import Workshop,generate_world,Action,action_catalog
from topoformer.campaign02_references import make_reference


def main(path):
    start=time.process_time()
    rows=[]
    def measure(label,env,step):
        obs=env.observe()
        tick=time.process_time()
        try:
            memory=encode_memory(obs)
            assert decode_memory(memory)==normalized_document(obs)
            row={"label":label,"step":step,"status":"supported",**memory.stats,
                 "dense_feature_float32_bytes":len(memory.rows)*memory.stats['row_dim']*4}
        except MemoryCapacityError as error:
            row={"label":label,"step":step,**error.record()}
        row['codec_cpu_seconds']=time.process_time()-tick
        rows.append(row)
    for categories,choices in ((2,3),(3,4)):
        env=Workshop(generate_world(71,categories=categories,choices=choices,step_limit=64))
        teacher=make_reference('cheap_first_fallback_v2')
        for step in range(65):
            measure(f'public_teacher_{categories}x{choices}_no_solver',env,step)
            if env.observe().done:break
            env.step(teacher.choose(env.observe()))
    # Explicit mechanical executor: every call returns unknown, consumes0work,
    # makes no correctness claim. This profiles repeated public snapshots only.
    def unknown_executor(primitive,problem,budget):
        return {'status':'unknown','payload':None,'work_units':0,'certificate_valid':False}
    env=Workshop(generate_world(72,categories=8,choices=8,step_limit=256),executor=unknown_executor)
    step=0;measure('large_repeated_call_memory_mechanical',env,step)
    for item in env.observe().item_inventory:
        env.step(Action('inspect',{'target':item['handle']}));step+=1
    env.step(Action('start_subset',{'handle':'draft'}));step+=1
    for constraint in ('capacity','funds','incompatibility'):
        env.step(Action('add_constraint',{'problem':'draft','constraint':constraint}));step+=1
    measure('large_repeated_call_memory_mechanical',env,step)
    for index in range(24):
        env.step(Action('call',{'problem':'draft','budget':16}));step+=1
        if index in (0,3,7,11,15,23):measure('large_repeated_call_memory_mechanical',env,step)
    report={'kind':'CPU shape and exact-roundtrip diagnostic, not model performance',
        'default_limits':asdict(MemoryLimits()),'rows':rows,'process_cpu_seconds':time.process_time()-start,
        'source_hashes':{name:hashlib.sha256((Path('src/topoformer')/name).read_bytes()).hexdigest()
            for name in ('campaign02_memory.py','campaign02_world.py','campaign02_references.py')},
        'limits':'Teacher profile has no executor and may abstain; stress executor is deliberately unknown-only mechanical fixture. Neither certifies tool solvability.'}
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(report,indent=2))
    print(json.dumps({'cpu_seconds':report['process_cpu_seconds'],'measurements':len(rows),
        'max_supported_rows':max(r.get('nodes',0) for r in rows),
        'overflow_count':sum(r['status']!='supported' for r in rows)}))

if __name__=='__main__':main(sys.argv[1])
