import gzip,json,hashlib,time
from pathlib import Path
import torch
t=time.monotonic();torch.set_num_threads(2);r=Path('/home/brandonin/topoformer-campaign01-semantics/results/s18-reference-v2');m=json.load(gzip.open(r/'manifest.json.gz','rt'));out=[]
for c in m['results'][0]['curves']:
 p=Path(c['checkpoint']['path']);h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==c['checkpoint']['sha256'];x=torch.load(p,map_location='cpu',weights_only=True);d=hashlib.sha256()
 for name,v in sorted(x['model'].items()):
  v=v.detach().cpu().contiguous();d.update(name.encode());d.update(str(v.dtype).encode());d.update(str(tuple(v.shape)).encode());d.update(v.numpy().tobytes())
 assert d.hexdigest()==c['model_state_sha256'];assert x['added_update']==c['added_update'] and x['update']==24576+c['added_update'];assert {int(s['step']) for s in x['optimizer']['state'].values()}=={x['update']};out.append(dict(update=x['update'],checkpoint_sha256=h,tensor_sha256=d.hexdigest()));del x
result=dict(checkpoints=out,cpu_audit_wall_seconds=time.monotonic()-t,scope='CPU saved checkpoint bytes/tensor fingerprint/AdamW steps; historical trajectory audits reused; no model construction or inference.')
Path('/tmp/s18-reference-state-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
