"""Independently verify public-prior raw-pair intervention and provenance."""
import argparse,gzip,hashlib,json
from pathlib import Path

def load(path):
 with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return json.load(f)
def audit(root):
 errors=[];pairs=frames=empty_frames=0;provenance=[]
 for mode in ('protected','recurrent'):
  for seed in (0,1,2):
   run=root/f'{mode}-{seed}';manifest=load(run/'manifest.json');old=load(root.parents[1]/'stage8/belief-idmatched'/f'{mode}-{seed}'/'manifest.json')
   if manifest['checkpoint_sha256']!=old['checkpoint_sha256']:errors.append([run.name,'checkpoint'])
   provenance.append(dict(run=run.name,checkpoint_sha256=manifest['checkpoint_sha256'],source=manifest['source_sha256'],config_sha256=manifest['config_sha256']))
   for file in run.glob('supplied_empty_prior-*.json.gz'):
    a=load(file);b=load(run/file.name.replace('supplied_empty_prior','learned_prior'))
    for key in ('target','empty_ledger','event_seed','candidates','condition','event_sha256'):
     if a[key]!=b[key]:errors.append([str(file),key])
    n=a['candidates']
    for pa,pb,mask in zip(a['posterior'],b['posterior'],a['empty_ledger']):
     for x,y,empty in zip(pa,pb,mask):
      frames+=1;empty_frames+=empty
      expected=[1/n]*n+[0.] if empty else y
      if x!=expected:errors.append([str(file),'nonempty changed or prior incorrect'])
    pairs+=1
 return dict(pairs=pairs,frames=frames,empty_frames=empty_frames,provenance=provenance,errors=errors,passed=pairs==432 and not errors)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.root);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='provenance'}));raise SystemExit(0 if r['passed'] else 1)
