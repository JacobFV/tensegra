"""Independent archived-trajectory audit; no solver execution or model inference."""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types


def audit(root, source):
    start=time.process_time()
    path=Path(root)
    summary=json.loads((path/'summary.json').read_text())
    config=summary['config']
    code=subprocess.check_output(['git','show',source+':src/topoformer/campaign02_world.py'])
    assert hashlib.sha256(code).hexdigest()==summary['source_sha256']['campaign02_world.py']
    mod=types.ModuleType('_frozen_audit_world');sys.modules[mod.__name__]=mod
    exec(compile(code,source+':campaign02_world.py','exec'),mod.__dict__)
    specs={};hidden={};cells={};failures={};address={};seen=set();validated_success=0
    with gzip.open(path/'episodes.jsonl.gz','rt') as stream:
        for line in stream:
            r=json.loads(line);seed=r['seed'];condition=r['condition'];mode=r['mode']
            key=(condition,seed)
            if key not in specs:
                cfg=next(c for c in config['conditions'] if c['name']==condition)
                kwargs={k:v for k,v in cfg.items() if k!='name'}
                if 'call_budgets' in kwargs:kwargs['call_budgets']=tuple(kwargs['call_budgets'])
                spec=mod.generate_world(seed,**kwargs);specs[key]=spec
                values=asdict(spec)
                expected=hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()
                assert expected==r['world_sha256']
                for field in ('work_limit','include_remaining_budget'):values.pop(field)
                encoded=json.dumps(values,sort_keys=True)
                assert seed not in hidden or hidden[seed]==encoded
                hidden[seed]=encoded
            spec=specs[key]
            assert (condition,mode,seed) not in seen
            seen.add((condition,mode,seed));assert int(r['instance_key'])==seed
            assert seed not in address or address[seed]==r['address_seed'];address[seed]=r['address_seed']
            c=cells.setdefault((condition,mode),{})
            c[seed]=r['verified_success']
            h=r['history'];assert r['steps']==len(h)
            assert r['work_units']==h[-1]['work'];assert r['work_units']<=spec.work_limit
            observations=sum(t['action']['kind']=='inspect' for t in h)
            assert observations==r['observations']
            cost=len(h)*spec.action_price+observations*spec.observation_price+r['work_units']*spec.work_price+r['travel_distance']*spec.travel_price+r['compute_units']*spec.compute_price
            assert abs(cost-r['cost'])<1e-12
            assert abs(float(r['verified_success'])-cost-r['utility'])<1e-12
            assert all(x['correct'] for x in r['reductions'])
            if r['verified_success']:
                selected=next(t['feedback']['selected'] for t in reversed(h) if 'selected' in t['feedback'])
                inventory={x.handle:x for x in spec.items};chosen=[inventory[x] for x in selected]
                assert len(set(selected))==len(selected)
                assert sorted(x.category for x in chosen)==sorted(spec.categories)
                assert sum(x.weight for x in chosen)<=spec.capacity and sum(x.price for x in chosen)<=spec.funds
                assert not any(a in selected and b in selected for a,b in spec.incompatible)
                assert h[-1]['action']['kind']=='verify' and h[-1]['feedback']['verified']
                assert any(t['feedback'].get('position')==spec.destination and t['feedback'].get('delivered') for t in h)
                validated_success+=1
            else:
                f=failures.setdefault((condition,mode),[])
                calls=[t for t in h if t['action']['kind']=='call']
                f.append(dict(seed=seed,work=r['work_units'],last=h[-1]['action']['kind'],
                              budgets=[t['action']['arguments']['budget'] for t in calls],
                              statuses=[t['feedback']['status'] for t in calls],
                              subset_selected=any('selected' in t['feedback'] for t in h),
                              reduction_primitives=[x['primitive'] for x in r['reductions']],
                              any_valid_incumbent=any(x['certificate_valid'] for x in r['reductions'])))
    checks=[]
    for cell in summary['cells']:
        values=cells[(cell['condition'],cell['mode'])]
        assert len(values)==cell['n']==config['episodes']
        assert sum(values.values())==cell['successes']
        checks.append({k:cell[k] for k in ('condition','mode','n','successes')})
    pairs=[]
    keys=list(cells)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            common=cells[a].keys()&cells[b].keys();cnt=Counter()
            for seed in common:
                x,y=cells[a][seed],cells[b][seed]
                cnt['both_correct' if x and y else 'a_only' if x else 'b_only' if y else 'both_wrong']+=1
            outcomes={k:cnt[k] for k in ('both_correct','a_only','b_only','both_wrong')}
            recorded=next(v for v in summary['paired_success'] if v['a']==list(a) and v['b']==list(b))
            assert outcomes==recorded['outcomes'] and len(common)==recorded['paired_instances']
            pairs.append(dict(a=list(a),b=list(b),n=len(common),outcomes=outcomes))
    return dict(source=source,archive_sha256=hashlib.sha256((path/'episodes.jsonl.gz').read_bytes()).hexdigest(),
                summary_sha256=hashlib.sha256((path/'summary.json').read_bytes()).hexdigest(),
                rows=len(seen),unique_instances=len(hidden),independently_validated_successful_selections=validated_success,
                hidden_world_comparability='identical after removing public work_limit and include_remaining_budget fields',
                all_recorded_reductions_correct=True,cells=checks,paired_success=pairs,
                failures={str(k):v for k,v in failures.items()},cpu_seconds=time.process_time()-start,
                limitations=['no policy/solver inference replay','status/certificate/reduction correctness fields are recorded, not fully independently reexecuted',
                             'success subset constraints and final verified/destination events are independently checked; complete path replay omitted'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=audit(args.root,args.source)
    Path(args.output).write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps({k:result[k] for k in ('rows','unique_instances','independently_validated_successful_selections','cells','cpu_seconds')},indent=2))
