"""Audit saved return predictions and prespecified retention gate (stdlib only)."""
import argparse
import gzip
import json
from pathlib import Path

FIELDS=('value','type','operation','argument0','argument1','provenance')

def audit(root):
    config=json.loads((root/'config.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    errors=[];rows_checked=0;examples=0;gate_results=[]
    for run in manifest['runs']:
        name=f"{run['seed']}-{run['encoding']}-{run['availability']}"
        path=root/(name+'.jsonl')
        if path.exists(): rows=[json.loads(x) for x in path.read_text().splitlines()]
        else:
            with gzip.open(str(path)+'.gz','rt') as stream: rows=[json.loads(x) for x in stream]
        evaluations=[r for r in rows if r['phase']=='eval']
        for r in evaluations:
            matches={k:[a==b for a,b in zip(r['predictions'][k],r['targets'][k])] for k in FIELDS}
            matches['identity_joint']=[all(x) for x in zip(*(matches[k] for k in FIELDS[3:]))]
            matches['scalar_joint']=[all(x) for x in zip(*(matches[k] for k in FIELDS[:3]))]
            matches['joint']=[all(x) for x in zip(*(matches[k] for k in FIELDS))]
            expected={k:dict(correct=sum(v),total=len(v)) for k,v in matches.items()}
            required=[v for v,t in zip(matches['argument1'],r['targets']['argument1']) if t!=6]
            expected['argument1_required']=dict(correct=sum(required),total=len(required))
            if expected!=r['counts']:errors.append([name,'counts',r['split'],r['steps'],r['intervention']])
            rows_checked+=1;examples+=len(matches['joint'])
        passed=True;failed=[]
        for seed in config['validation_seeds']:
            for d in config['eval_distractors']:
                selected=[r for r in evaluations if r['split']=='validation' and r['seed']==seed and r['distractors']==d and r['steps']==16 and r['intervention']=='none']
                if len(selected)!=1:errors.append([name,'missing_gate_cell']);passed=False;continue
                counts=selected[0]['counts']
                for k in FIELDS:
                    c=counts['argument1_required' if k=='argument1' else k]
                    if counts['joint']['total']<512 or not c['total'] or c['correct']/c['total']<=(.99 if k in ('type','operation') else .98):
                        passed=False;failed.append([seed,d,k])
        if passed!=run['retention_gate']:errors.append([name,'gate'])
        gate_results.append(dict(run=name,passed=passed,failed_fields=failed))
    return dict(rows=rows_checked,example_evaluations=examples,errors=errors,gates=gate_results,audit_passed=bool(rows_checked) and not errors)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();r=audit(a.root)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='gates'}));raise SystemExit(0 if r['audit_passed'] else 1)
