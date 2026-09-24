from pathlib import Path
import argparse,gzip,hashlib,json,time,torch
from topoformer.campaign_semantics_s16 import tensor_hash
p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);tick=time.monotonic();path=a.manifest;m=json.load(gzip.open(path,'rt'));checked={}
for r in m['artifacts']:
 p=Path(r['directory'])/'model-u24576.pt';assert hashlib.sha256(p.read_bytes()).hexdigest()==r['checkpoint_sha256'];state=torch.load(p,map_location='cpu',weights_only=True)['model'];assert tensor_hash(state)==r['tensor_before_sha256']==r['tensor_after_sha256'];assert tensor_hash({'thresholds':torch.tensor(r['thresholds'])})==r['threshold_tensor_sha256'];checked[f"{r['seed']}-{r['arm']}"]=r['tensor_before_sha256'];del state
out=dict(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),saved_checkpoint_tensor_hashes=checked,threshold_tensor_hashes_verified=True,cpu_audit_wall_seconds=time.monotonic()-tick,scope='CPU saved tensor bytes only; no actor construction or model forward.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
