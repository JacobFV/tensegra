"""Recompute Track C frozen gates from raw prediction classes and targets."""
import hashlib
import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
config=json.loads((root/'config.json').read_text())
manifest=json.loads((root/'manifest.json').read_text())
fields=('value','type','operation','argument0','argument1','provenance')
runs=[]
aggregate={}
for init in config['seeds']:
    initial_hashes={r['initial_hash'] for r in manifest['runs'] if r['seed']==init}
    assert len(initial_hashes)==1,'paired initialization mismatch'
    for mode in config['modes']:
        records=[json.loads(l) for l in (root/f'seed{init}-{mode}.jsonl').read_text().splitlines()]
        rows=[r for r in records if 'split' in r]
        for row in rows:
            n=len(row['targets']['value'])
            assert n==config['eval_size']
            matches={f:[a==b for a,b in zip(row['predictions'][f],row['targets'][f])] for f in fields}
            for f in fields:assert row['counts'][f]==dict(correct=sum(matches[f]),total=n)
            assert row['counts']['joint']==dict(correct=sum(all(matches[f][i] for f in fields) for i in range(n)),total=n)
        selected={(r['seed'],r['distractors']):r for r in rows if r['split']=='validation' and r['delay']==16 and r['intervention']=='none'}
        expected={(seed,d) for seed in config['validation_seeds'] for d in config['eval_distractors']}
        assert set(selected)==expected
        minima={f:1. for f in (*fields,'argument1_required')}
        for row in selected.values():
            for f in fields:minima[f]=min(minima[f],row['counts'][f]['correct']/row['counts'][f]['total'])
            indices=[i for i,op in enumerate(row['targets']['operation']) if op!=3]
            assert indices
            acc=sum(row['predictions']['argument1'][i]==row['targets']['argument1'][i] for i in indices)/len(indices)
            minima['argument1_required']=min(minima['argument1_required'],acc)
        passed=all(v>(.99 if f in ('type','operation') else .98) for f,v in minima.items())
        runs.append(dict(initialization=init,mode=mode,gate_c=passed,min_validation16_accuracy=minima))
        for row in rows:
            if row['intervention']!='none':continue
            for field in fields+('joint',):
                key=(mode,row['split'],row['delay'],field)
                count=aggregate.setdefault(key,[0,0])
                count[0]+=row['counts'][field]['correct'];count[1]+=row['counts'][field]['total']
result=dict(runs=runs,arm_gates={mode:all(r['gate_c'] for r in runs if r['mode']==mode) for mode in config['modes']},pooled=[dict(mode=k[0],split=k[1],delay=k[2],field=k[3],correct=v[0],total=v[1],accuracy=v[0]/v[1]) for k,v in aggregate.items()],source_hashes=manifest['source'],config_hash=manifest['config'],durable_path=str(root),total_run_seconds=sum(r['elapsed'] for r in manifest['runs']),optimizer_steps=sum(r['optimizer_steps'] for r in manifest['runs']),examples_seen=sum(r['examples_seen'] for r in manifest['runs']))
(root/'gate-audit.json').write_text(json.dumps(result,indent=2))
checksums={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.name!='artifact-sha256.json'}
(root/'artifact-sha256.json').write_text(json.dumps(checksums,indent=2))
print(json.dumps({k:result[k] for k in ('arm_gates','total_run_seconds','optimizer_steps','examples_seen')},indent=2))
