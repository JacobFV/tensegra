"""Independent C02 endpoint counts, calibration selection and static replay."""
import argparse,json,hashlib,time,torch
from pathlib import Path
from campaign_composition_audit import count,paired
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();torch.set_num_threads(2);m=json.loads((a.root/'summary.json').read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False);best=(-1,0);cells=0;parity=0
for rec in m['curves']:
 raw=load(a.root/f"calibration-{rec['step']}.pt");got=count(raw['logits']['answer'].argmax(-1),raw['labels']['task']);assert got==rec['calibration']['answer'];best=max(best,(got['correct'],-rec['step']))
 if m['arm']=='static'and m['config'].get('historical_directory'):
  old=load(Path(m['config']['historical_directory'])/f"calibration-{m['config']['learning_rate']}-{rec['step']}.pt")
  for group in('labels','logits'):
   for k,v in raw[group].items():assert torch.equal(v,old[group][k])
  parity+=1
assert -best[1]==m['selected_step']
def verify(raw,record):
 global cells
 out=raw['logits'];labels=raw['labels'];pub=raw['public'];assert count(out['answer'].argmax(-1),labels['task'])==record['answer'];assert count(out['value'].argmax(-1),labels['value'])==record['value'];op=out['primitive'].argmax(-1);ptr=out['pointers'].argmax(-1).clone();ptr[op==3,2]=(pub['keys']==0).all(-1).long().argmax(-1)[op==3];assert count((op==labels['primitive'])&(ptr==labels['targets']).all(-1),torch.ones_like(op,dtype=torch.bool))==record['lowering']['full_semantic'];cells+=1
for endpoint,report in m['results'].items():
 for split in('validation','fresh_validation'):
  original=load(a.root/f'{endpoint}-{split}.pt');verify(original,report[split]);changed=load(a.root/f'{endpoint}-{split}-rekey.pt');verify(changed,report[split+'_rekey']);assert all(torch.equal(v,changed['labels'][k])for k,v in original['labels'].items())
  for ref in ('instruction_destinations','instruction_arguments','query_destination'):
   oldref=original['public'][ref].reshape(len(original['labels']['task']),-1,32);newref=changed['public'][ref].reshape(len(changed['labels']['task']),-1,32);oldlinks=(oldref[:,:,None]==original['public']['keys'][:,None]).all(-1);newlinks=(newref[:,:,None]==changed['public']['keys'][:,None]).all(-1);assert torch.equal(oldlinks,newlinks) and (oldlinks.sum(-1)==1).all()
  assert torch.equal((original['public']['keys']==0).all(-1),(changed['public']['keys']==0).all(-1))
  for k,v in original['public'].items():
   if k not in('keys','instruction_destinations','instruction_arguments','query_destination'):assert torch.equal(v,changed['public'][k])
 for kind,rec in report['controls'].items():
  raw=load(a.root/f'{endpoint}-control-{kind}.pt');verify(raw,rec['supplied']);assert paired(raw['logits']['answer'].argmax(-1),raw['original_labels']['task'],raw['labels']['task'])==rec['paired']
for name,h in m['checkpoint_sha256'].items():assert hashlib.sha256((a.root/f'{name}.pt').read_bytes()).hexdigest()==h
if m['arm']=='static'and m['config'].get('historical_directory'):
 old=load(Path(m['config']['historical_directory'])/'selected.pt');new=load(a.root/'selected.pt');assert all(torch.equal(v,new[k])for k,v in old.items())
visits=load(a.root/'visitation.pt');assert int(visits.sum())==m['optimizer_presentations'];assert int((visits>0).sum())==m['unique_base_events']
out=dict(arm=m['arm'],calibration_checkpoints=len(m['curves']),cells=cells,selected_step=m['selected_step'],historical_calibration_exact=parity,checkpoint_hashes_verified=3,optimizer_presentations=int(visits.sum()),results={e:{s:r[s]['answer']for s in('validation','validation_rekey','fresh_validation','fresh_validation_rekey')}for e,r in m['results'].items()},cpu_audit_wall_seconds=time.monotonic()-t,scope='Fixed primary endpoint and calibration-selected secondary verified separately; fresh population remains development. Rekey labels/non-key fields unchanged.');a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
