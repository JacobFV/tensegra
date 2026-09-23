"""Rehash immutable remote continuation/backbone/readout files without model loading."""
import argparse,gzip,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
with gzip.open(a.manifest,'rt')as f:x=json.load(f)
expected={}
for r in x['runs']:
 s=r['backbone_seed']
 for arm in r['arms']:expected[f"topoformer-stage11/returns/main/{s}-{arm['arm']}.pt"]=arm['checkpoint_sha256']
 expected[f'topoformer-stage9/returns/readout-main/{s}-backbone.pt']=r['initial_checkpoint_sha256']
 expected[f'topoformer-stage10/returns/main/{s}-shared_pool-fit.pt']=r['frozen_ridge_sha256']
program='import hashlib,json,pathlib\nexpected='+repr(expected)+'\nrows=[]\nfor p,want in expected.items():\n f=pathlib.Path.home()/p;got=hashlib.file_digest(f.open("rb"),"sha256").hexdigest();assert got==want;rows.append(dict(path=str(f),sha256=got))\nprint(json.dumps(dict(rows=rows,files_verified=len(rows))))\n'
r=subprocess.run(['ssh','gb10-direct','python3','-'],input=program,text=True,check=True,capture_output=True);out=json.loads(r.stdout);a.output.write_text(json.dumps(out,indent=2)+'\n');print(out['files_verified'])
