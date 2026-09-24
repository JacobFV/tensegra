"""Independent C04 public arithmetic, causal gates, scalar and overlap replay."""
import argparse,collections,gzip,hashlib,json,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--replicate',type=int,required=True);p.add_argument('--aggregate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic();torch.set_num_threads(2);base=a.root/f'c04-confirmation-{a.replicate}';path=lambda arm:Path(str(base)+'-'+arm);load=lambda p:torch.load(p,map_location='cpu',weights_only=False);ms={arm:json.loads((path(arm)/'summary.json').read_text()) for arm in ('hybrid','n1_static','n1_roles','n2_rekey')};agg=json.loads(a.aggregate.read_text())['lineages'][str(a.replicate)];h=ms['hybrid'];n=h['config']['data']['validation']['count'];assert n==4096;bi=torch.arange(n)
def public_labels(pub):
 b=torch.arange(len(pub['query']));which=(pub['instruction_destinations']==pub['query_destination'][:,None]).all(-1);assert (which.sum(-1)==1).all();ix=which.long().argmax(-1);op=pub['instruction_cues'][b,ix].argmax(-1);refs=torch.cat((pub['query_destination'][:,None],pub['instruction_arguments'][b,ix]),1);links=(refs[:,:,None]==pub['keys'][:,None]).all(-1);assert (links.sum(-1)==1).all();ptr=links.long().argmax(-1);value,typ=arithmetic(pub,op,ptr)
 return dict(primitive=op,targets=ptr,value=(value*2+16).long(),task=((value>pub['query'][:,0])^pub['query'][:,1].bool()).long(),type=typ)
def arithmetic(pub,op,ptr):
 b=torch.arange(len(op));x=pub['operand_values'][b,ptr[:,1]];y=pub['operand_values'][b,ptr[:,2]];xt=pub['operand_types'][b,ptr[:,1]];yt=pub['operand_types'][b,ptr[:,2]];value=torch.where(op==0,x+y,torch.where(op==1,x-y,torch.where(op==2,x*y,torch.where(op==3,-x,(x<y).float()))));typ=torch.where(op==4,2,torch.where(op==3,xt,torch.maximum(xt,yt)));return value,typ
publics={};labels={};neural=[]
for arm in ('n1_static','n1_roles','n2_rekey'):
 for policy in ('endpoint','selected'):
  for view in ('clean','reversed'):
   x=load(path(arm)/(f'{policy}-validation'+('-reversed' if view=='reversed' else '')+'.pt'));pub=x['public'];gold=public_labels(pub)
   for k in ('primitive','targets','value','task'):assert torch.equal(gold[k],x['labels'][k]),(arm,policy,view,k)
   if view not in publics:publics[view]=pub;labels[view]=gold
   else:assert all(torch.equal(v,publics[view][k]) for k,v in pub.items())
   op=x['logits']['primitive'].argmax(-1);ptr=x['logits']['pointers'].argmax(-1).clone();ptr[op==3,2]=(pub['keys']==0).all(-1).long().argmax(-1)[op==3];full=(op==gold['primitive'])&(ptr==gold['targets']).all(-1);rec=ms[arm]['results'][policy]['validation'+('_reversed' if view=='reversed' else '')];rec=rec.get('supplied',rec);assert int(full.sum())==rec['lowering']['full_semantic']['correct'];neural.append(dict(arm=arm,policy=policy,view=view,answer=int((x['logits']['answer'].argmax(-1)==gold['task']).sum()),value=int((x['logits']['value'].argmax(-1)==gold['value']).sum()),full_proposal=int(full.sum())))
clean=publics['clean'];reverse=publics['reversed'];op=clean['instruction_cues'].argmax(-1);assert torch.equal(torch.where((op!=3)[...,None,None],clean['instruction_arguments'].flip(-2),clean['instruction_arguments']),reverse['instruction_arguments'])
for k in clean:
 if k!='instruction_arguments':assert torch.equal(clean[k],reverse[k])
hybrid=[];events=0
for view in ('clean','reversed'):
 pub=publics[view];gold=labels[view]
 for dist in (2,8):
  x=load(path('hybrid')/f'{view}-{dist}.pt');assert torch.equal(x['original'],gold['task']);assert all(torch.equal(x['labels'][k],gold[k]) for k in ('primitive','targets','value','task'));proposal=x['proposal'];op=proposal['primitive'];ptr=proposal['canonical_pointers'];full=(op==gold['primitive'])&(ptr==gold['targets']).all(-1);assert torch.equal(full,x['full_proposal']);v,typ=arithmetic(pub,op,ptr);accepted=torch.tensor([not s for s in x['reasons']]);ids=accepted.nonzero().flatten();assert accepted.all()  # Observed lineage has no refusal; never drop a refused event.
  assert (pub['operand_types'][bi,ptr[:,0]]==-1).all() and (pub['keys'][bi,ptr[:,0]].abs().sum(-1)>0).all();assert (pub['operand_types'][bi,ptr[:,1]]>=0).all();assert ((op==3)|(pub['operand_types'][bi,ptr[:,2]]>=0)).all();assert (v.abs()<=8).all() and (v*2==(v*2).round()).all()
  event=x['supplied_public']['event'];assert torch.equal(event['values'][:,0],v);assert torch.equal(event['types'][:,0],typ);assert torch.equal(event['operations'][:,0],op);assert torch.equal(event['provenance'][:,0],pub['keys'][bi,ptr[:,0]]);assert torch.equal(event['arguments'][:,0,0],pub['keys'][bi,ptr[:,1]]);assert torch.equal(event['arguments'][:,0,1],torch.where((op==3)[:,None],torch.zeros_like(pub['keys'][bi,ptr[:,2]]),pub['keys'][bi,ptr[:,2]]));assert torch.equal(x['result']['values'],v);assert torch.equal(x['result']['supplied'],((v>pub['query'][:,0])^pub['query'][:,1].bool()).long());events+=n
  for delay,c in x['result']['cells'].items():
   pred=c['predictions'];scores=c['scores'];scalar=scores.argmax(-1);assert torch.isfinite(scores).all();rec=next(z for z in agg['hybrid_cells'] if z.get('delay')==delay and z['view']==view and z['distractors']==dist);actual=int((scalar==(2*v+16).long()).sum());requested=int((scalar==gold['value']).sum());assert rec['scalar']['correct_actual_return']==actual and rec['scalar']['correct_requested_value']==requested;assert rec['joint_correct']==int((full&(pred==gold['task'])).sum());assert rec['joint_pass']==(rec['joint_correct']>=4015);hybrid.append(dict(view=view,distractors=dist,delay=delay,answer=int((pred==gold['task']).sum()),joint=rec['joint_correct'],scalar_actual=actual,scalar_requested=requested,refused=int((pred<0).sum())))
causal=[]
for view in ('clean','reversed'):
 controls={k:load(path('hybrid')/f'{view}-causal-{k}.pt') for k in ('correct','drop','wrong','swap')};truth=controls['correct']['original'];ncausal=len(truth);assert ncausal==h['config']['causal_size'];assert all(torch.equal(z['original'],truth) for z in controls.values())
 for kind,x in controls.items():
  s=x['result']['supplied'];pub=x['supplied_public']
  if kind=='drop':assert (s==-1).all()
  else:assert torch.equal(s,((pub['event']['values'][:,0]>pub['query'][:,0])^pub['query'][:,1].bool()).long())
 for method,delays in [('workspace',(0,1,16)),('supplied_copy',(None,))]:
  pred=lambda k:controls[k]['exact_copy'] if method=='supplied_copy' else controls[k]['result']['cells'][delay]['predictions']
  for delay in delays:
   acc=lambda z:float((z==truth).double().mean());good=acc(pred('correct'));drop=acc(pred('drop'));query=acc(controls['correct']['query_only']);copy=acc(controls['correct']['exact_copy']);gain=(good-query)/(copy-query) if copy>query else None;changed={}
   for kind in ('wrong','swap'):
    s=controls[kind]['result']['supplied'];mask=(s>=0)&(s!=truth);changed[kind]=dict(correct=int(((pred(kind)==s)&mask).sum()),total=int(mask.sum()))
   passed=good-drop>=.15 and gain is not None and gain>=.8 and all(z['total']>=256 and z['correct']/z['total']>=.9 for z in changed.values());rec=next(z for z in next(v for v in h['causal'] if v['view']==view)['checks'] if z['path']==method and z['delay']==delay);assert rec['passed']==passed and rec['changed_supplied']==changed and rec['normalized_gain']==gain and rec['clean_minus_drop']==good-drop;causal.append(dict(view=view,path=method,delay=delay,attempted=ncausal,changed_supplied=changed,clean_minus_drop=good-drop,normalized_gain=gain,passed=passed))
out=dict(replicate=a.replicate,actual_runtime_events=events,neural=neural,hybrid=hybrid,causal=causal,public_labels_and_ordered_proposals_verified=True,scalar_actual_vs_requested_verified=True,all_reported_refusals_preserved=True,cpu_audit_wall_seconds=time.monotonic()-tick,scope='Independent saved public instruction/key resolution, ordered arithmetic/types/provenance, full proposals, scalar actual/requested counts and causal gates; no model inference. All current lineage proposals executed; no refusal filtering or repair. Whole matrix remains pending.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(dict(seconds=out['cpu_audit_wall_seconds'],runtime_events=events,neural_cells=len(neural),hybrid_cells=len(hybrid),causal_cells=len(causal)))
