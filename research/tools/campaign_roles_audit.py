"""Independent C03 endpoint counts, calibration selection and static replay."""
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
 bi=torch.arange(len(op));matches=(pub['instruction_destinations']==pub['query_destination'][:,None]).all(-1);assert (matches.sum(-1)==1).all();ins=matches.long().argmax(-1);trueop=pub['instruction_cues'][bi,ins].argmax(-1);refs=torch.cat((pub['query_destination'][:,None],pub['instruction_arguments'][bi,ins]),1);links=(refs[:,:,None]==pub['keys'][:,None]).all(-1);assert (links.sum(-1)==1).all();ptrgold=links.long().argmax(-1);assert torch.equal(trueop,labels['primitive'])and torch.equal(ptrgold,labels['targets'])
 x=pub['operand_values'][bi,ptrgold[:,1]];y=pub['operand_values'][bi,ptrgold[:,2]];value=torch.where(trueop==0,x+y,torch.where(trueop==1,x-y,torch.where(trueop==2,x*y,torch.where(trueop==3,-x,(x<y).float()))));assert torch.equal((value*2+16).long(),labels['value']);assert torch.equal(((value>pub['query'][:,0])^pub['query'][:,1].bool()).long(),labels['task'])
for endpoint,report in m['results'].items():
 for split in('validation','fresh_validation'):
  original=load(a.root/f'{endpoint}-{split}.pt');verify(original,report[split]);changed=load(a.root/f'{endpoint}-{split}-reversed.pt');verify(changed,report[split+'_reversed']['supplied']);assert paired(changed['logits']['answer'].argmax(-1),original['labels']['task'],changed['labels']['task'])==report[split+'_reversed']['paired']
  op=original['public']['instruction_cues'].argmax(-1);args=original['public']['instruction_arguments'];want=torch.where((op!=3)[...,None,None],args.flip(-2),args);assert torch.equal(want,changed['public']['instruction_arguments'])
  for k,v in original['public'].items():
   if k!='instruction_arguments':assert torch.equal(v,changed['public'][k])

 for kind,rec in report['controls'].items():
  raw=load(a.root/f'{endpoint}-control-{kind}.pt');verify(raw,rec['supplied']);assert paired(raw['logits']['answer'].argmax(-1),raw['original_labels']['task'],raw['labels']['task'])==rec['paired']
for name,h in m['checkpoint_sha256'].items():assert hashlib.sha256((a.root/f'{name}.pt').read_bytes()).hexdigest()==h
if m['arm']=='static'and m['config'].get('historical_directory'):
 old=load(Path(m['config']['historical_directory'])/'selected.pt');new=load(a.root/'selected.pt');assert all(torch.equal(v,new[k])for k,v in old.items())
visits=load(a.root/'visitation.pt')['base'];assert int(visits.sum())==m['optimizer_presentations'];assert int((visits>0).sum())==m['unique_base_events']
out=dict(arm=m['arm'],calibration_checkpoints=len(m['curves']),cells=cells,selected_step=m['selected_step'],historical_calibration_exact=parity,checkpoint_hashes_verified=3,optimizer_presentations=int(visits.sum()),results={e:{s:(r[s]['supplied']['answer']if s.endswith('_reversed')else r[s]['answer'])for s in('validation','validation_reversed','fresh_validation','fresh_validation_reversed')}for e,r in m['results'].items()},cpu_audit_wall_seconds=time.monotonic()-t,scope='Fixed primary endpoint and calibration-selected secondary verified separately; fresh population remains development. All binary public instruction roles reversed; non-role fields unchanged; actual ordered arithmetic targets verified.');a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
