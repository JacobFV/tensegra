"""Independent final attention pairing and coordinate-restoration checks."""
import argparse,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic();arms=['soft4','soft8','context','message','hard'];allrows={};initial=[];checks=[]
for arm in arms:
 root=a.root/f'{arm}-{a.seed}';cfg=json.loads((root/'config.json').read_text());m=json.loads((root/'manifest.json').read_text());raw=np.load(root/f"eval-{cfg['steps']:05d}.npz");rows=json.loads((root/f"eval-{cfg['steps']:05d}.json").read_text())['rows'];initial.append(m['initial_shared_tensor_sha256']);assert len(rows)==(20 if arm=='soft4' else 18)
 indexed={}
 for i,r in enumerate(rows):
  c=r['condition'];group=c['data_group'];key=json.dumps(c,sort_keys=True);indexed[key]=(raw,i,r)
  base=next(j for j,x in enumerate(rows)if x['condition']=={k:v for k,v in c.items()if k in ('nodes','depth','data_group','composition')})
  for field in ['gold','start','relation','successor']:assert np.array_equal(raw[f'c{i}_{field}'],raw[f'c{base}_{field}'])
  if c.get('node_permutation'):
   for field in ['pred','route','task']:assert np.array_equal(raw[f'c{i}_{field}'],raw[f'c{base}_{field}'])
  checks.append(dict(arm=arm,condition=c,task_correct=int(raw[f'c{i}_task'].sum()),examples=len(raw[f'c{i}_task'])))
 allrows[arm]=indexed
assert len(set(initial))==1
for arm in arms[1:]:
 for key,(raw,i,r)in allrows[arm].items():
  reference,j,_=allrows['soft4'][key]
  for field in ['gold','start','relation','successor']:assert np.array_equal(raw[f'c{i}_{field}'],reference[f'c{j}_{field}'])
out=dict(seed=a.seed,cpu_audit_wall_seconds=time.monotonic()-start,paired_final_cells=len(checks),initial_shared_tensor_sha256=initial[0],checks=checks,scope='Final clean targets and public graph identifiers paired across arms/interventions; coordinate-restored outputs identical for permutation control. Same1024 event draws are reused across arms; cross-seed identity requires the completed matrix audit.');a.output.write_text(json.dumps(out,indent=2)+'\n');print(dict(seed=a.seed,cells=len(checks),seconds=out['cpu_audit_wall_seconds']))
