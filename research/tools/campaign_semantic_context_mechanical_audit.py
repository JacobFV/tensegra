import argparse,collections,gzip,hashlib,json,re,time,importlib.util
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('remote',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);t=time.monotonic();r=a.repo;remote=a.remote;cache=remote/'data/s15-shape-v2/train_mixed.jsonl.gz';plan=json.loads((r/'research/campaigns/extended-01/semantics/S18-compute-plan.json').read_text());assert hashlib.sha256(cache.read_bytes()).hexdigest()==plan['cache_sha256'];lengths=[len(re.findall(r'\w+|[^\w\s]',json.loads(line)['text'])) for line in gzip.open(cache,'rt')];rng=torch.Generator().manual_seed(15115);visits=[0]*4096;lengthcounts=collections.Counter();seq=hashlib.sha256();totalcontext=totalunit=0
for epoch in range(8):
 order=torch.randperm(4096,generator=rng).tolist()
 for start in range(0,4096,8):
  ids=order[start:start+8];seq.update(json.dumps(ids).encode());n=max(lengths[i] for i in ids);lengthcounts[n]+=1
  for i in ids:visits[i]+=1
  w=1024
  def block(s,n):
   memory=n+1
   linear=(3*s+s+s+s+2*memory+2*s+2*s)*w*w
   attention=2*s*s*w+2*s*memory*w
   return linear+attention
  unit=4*block(8,n);context=2*unit+8*block(n,n)+(n+128)*w*w+128*n*w;totalunit+=8*unit;totalcontext+=8*context
assert visits==[8]*4096 and seq.hexdigest()==plan['sequence_sha256'];assert {str(k):v for k,v in lengthcounts.items()}==plan['padded_batch_length_counts'];assert {str(k):v for k,v in collections.Counter(lengths).items()}==plan['public_length_counts'];best=min(range(2,31),key=lambda k:(abs(k*totalunit-totalcontext),k));assert best==plan['control_microsteps']==10 and 2*totalunit==plan['original_macs'] and best*totalunit==plan['control_macs'] and totalcontext==plan['context_macs'];assert best*totalunit/totalcontext==plan['control_vs_context_ratio']
p=r/'research/campaigns/extended-01/semantics/S18-parent-mapping-audit.py';spec=importlib.util.spec_from_file_location('mapping',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);mapping=m.run(remote/'results/s11-lr-decay-n8192-196k-dev201/model-u24576.pt',remote/'data/s01/audit.json');assert mapping==json.loads((r/'research/campaigns/extended-01/semantics/S18-parent-mapping.json').read_text())
out=dict(independent_public_lengths_schedule_and_MAC_totals=True,control_microsteps=best,control_vs_context_ratio=best*totalunit/totalcontext,mapping=mapping,cpu_audit_wall_seconds=time.monotonic()-t,scope='CPU public text/analytic multiply-accumulate bookkeeping and saved metadata only; no acquired behavior, timing/FLOP equality, threshold choice or training authorization.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
