"""Independent CPU R09 binary-phase readout replay from immutable features."""
import torch,gzip,json,hashlib,time,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);a=p.parse_args();start=time.monotonic();torch.set_num_threads(2);root=Path(a.root);m=json.load(gzip.open(root/'manifest.json.gz','rt'));c=m['config'];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(c['feature_path'])==c['feature_sha256'];assert sha(c['reference_path'])==c['reference_sha256'];assert sha(c['reference_manifest'])==c['reference_manifest_sha256'];assert sha(c['checkpoint'])==c['checkpoint_sha256']
with gzip.open(c['feature_path'],'rb')as f:cache=torch.load(f,map_location='cpu',weights_only=True)
with gzip.open(root/'logits.pt.gz','rb')as f:logits=torch.load(f,map_location='cpu',weights_only=True)
reference=torch.load(c['reference_path'],map_location='cpu',weights_only=True);heads={}
for r in m['fits']:
 path=root/(r['arm']+'.pt');assert sha(path)==r['checkpoint_sha256'];h=torch.load(path,map_location='cpu',weights_only=True);heads[r['arm']]=h
 if r['arm']=='linear':
  for k,v in reference['state'].items():assert torch.equal(v,h['state'][k])
  assert torch.equal(reference['mean'],h['mean'])and torch.equal(reference['scale'],h['scale'])
linear=lambda x,s,prefix:torch.nn.functional.linear(x,s[prefix+'weight'],s[prefix+'bias']);n=0;maxdiff=0
with torch.no_grad():
 for key,b in cache.items():
  for delay,x in b['features'].items():
   for arm,h in heads.items():
    z=(x-h['mean'])/h['scale'];s=h['state']
    got=linear(z,s,'') if arm=='linear' else linear(z,s,'ingestion.' if delay==0 else 'recurrent.')
    expected=logits[f'{key}/{delay}/{arm}'];assert torch.equal(got.argmax(-1),expected.argmax(-1));maxdiff=max(maxdiff,float((got-expected).abs().max()));n+=1
out=dict(cells=n,all_argmax_equal=True,linear_R06_weights_exact=True,weight_cache_hashes_verified=True,max_logit_difference=maxdiff,cpu_audit_wall_seconds=time.monotonic()-start);Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(out)
