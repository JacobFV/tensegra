"""CPU-only immutable semantic-parent state sanity audit (no model inference)."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();torch.set_num_threads(2)
sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest();assert sha==a.sha256
x=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert x['update']==16384
steps={int(s['step']) for s in x['optimizer']['state'].values()};assert steps=={16384}
assert sum(x['visits'])==131072 and len(x['visits'])==8192 and min(x['visits'])==max(x['visits'])==16
assert sorted(x['order'])==list(range(8192)) and 0<=x['position']<=8192
out={'checkpoint_sha256':sha,'update':x['update'],'optimizer_steps':sorted(steps),'presentations':sum(x['visits']),'unique_visited':len(x['visits']),'visits_each':16,'position':x['position'],'generator_sha256':hashlib.sha256(x['generator'].numpy().tobytes()).hexdigest(),'schedule_sha256':hashlib.sha256(x['schedule'].numpy().tobytes()).hexdigest(),'cpu_audit_wall_seconds':time.monotonic()-t,'scope':'Checkpoint bytes, optimizer steps, complete sixteen-epoch coverage, sampling-state availability; no model inference or RNG-history reconstruction.'}
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
