"""Independent paired nuisance-augmentation comparison and sample-stream replay."""
import argparse,json,hashlib,time,torch,numpy as np
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('static',type=Path);p.add_argument('rekey',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();torch.set_num_threads(2);load=lambda p:torch.load(p,map_location='cpu',weights_only=False);ms=json.loads((a.static/'summary.json').read_text());mr=json.loads((a.rekey/'summary.json').read_text())
for key in('config','data','initial_state_sha256','parameter_count','optimizer_presentations','unique_base_events','sample_index_stream_sha256'):assert ms[key]==mr[key],key
assert ms['presented_key_stream_sha256']!=mr['presented_key_stream_sha256'];v=load(a.static/'visitation.pt');assert torch.equal(v,load(a.rekey/'visitation.pt'));cfg=ms['config'];g=torch.Generator().manual_seed(cfg['sample_seed']);h=hashlib.sha256();visits=torch.zeros_like(v)
for _ in range(cfg['updates']):
 idx=torch.randint(len(v),(cfg['batch_size'],),generator=g);h.update(idx.numpy().tobytes());visits+=torch.bincount(idx,minlength=len(v))
assert h.hexdigest()==ms['sample_index_stream_sha256']and torch.equal(visits,v)
rng=np.random.default_rng(530000999);ix=rng.integers(0,4096,(4000,4096));results=[]
for endpoint in('endpoint','selected'):
 for split in('validation','validation-rekey','fresh_validation','fresh_validation-rekey'):
  s=load(a.static/f'{endpoint}-{split}.pt');r=load(a.rekey/f'{endpoint}-{split}.pt');assert all(torch.equal(x,r['labels'][k])for k,x in s['labels'].items());assert all(torch.equal(x,r['public'][k])for k,x in s['public'].items())
  for field in('answer','value'):
   target=s['labels']['task'if field=='answer'else'value'];sc=(s['logits'][field].argmax(-1)==target).numpy();rc=(r['logits'][field].argmax(-1)==target).numpy();diff=rc.astype(float)-sc;results.append(dict(endpoint=endpoint,split=split,field=field,static=int(sc.sum()),rekey=int(rc.sum()),both_correct=int((sc&rc).sum()),static_only=int((sc&~rc).sum()),rekey_only=int((~sc&rc).sum()),both_wrong=int((~sc&~rc).sum()),difference=float(diff.mean()),paired_event_bootstrap95=np.quantile(diff[ix].mean(1),[.025,.975]).tolist()))
out=dict(initialization_data_and_samples_paired=True,sample_stream_replayed=True,results=results,cpu_audit_wall_seconds=time.monotonic()-t,scope='One inspected development initialization, same semantic pool/exposure, fresh nuisance views. Primary fixed4000 and selected secondary remain separate. Augmentation benefit does not alone identify memorization as the unique causal mechanism.');a.output.write_text(json.dumps(out,indent=2)+'\n');print([r for r in results if r['endpoint']=='endpoint'and r['split']=='fresh_validation'])
