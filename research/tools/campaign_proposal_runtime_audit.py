"""P01 runtime event fidelity for actual proposed operations; no gold reset."""
import torch,json,time,argparse
from pathlib import Path
from topoformer.campaign_composition import make_lowering_batch,execute_proposal
start=time.monotonic();torch.set_num_threads(2);base=Path.home()/'topoformer-campaign-01/composition';result=[]
for seed in [302,303,304]:
 p=base/f'p01-confirmation-{seed}';m=json.loads((p/'summary.json').read_text());spec=m['config']['data']['validation'];data=make_lowering_batch(spec['seed'],spec['count']);raw=torch.load(p/'validation.pt',map_location='cpu',weights_only=False);op=raw['logits']['primitive'].argmax(-1);ptr=raw['logits']['pointers'].argmax(-1).clone();null=raw['public']['keys'].abs().sum(-1).argmin(-1);ptr[op==3,2]=null[op==3];counts={'correct_proposals':0,'correct_event_tensor_matches':0,'incorrect_proposals':0,'incorrect_refused':0};errors=[]
 for i,row in enumerate(data['public']):
  got=execute_proposal(row,int(op[i]),ptr[i].tolist());correct=int(op[i])==int(raw['labels']['primitive'][i]) and torch.equal(ptr[i],raw['labels']['targets'][i])
  if correct:
   counts['correct_proposals']+=1;assert got['status']=='executed'
   for k,v in got['event'].items():assert torch.equal(v,data['reference']['public']['event'][k][i:i+1]),(seed,i,k)
   counts['correct_event_tensor_matches']+=1
  else:
   counts['incorrect_proposals']+=1;counts['incorrect_refused']+=got['status']=='refused';errors.append({'index':i,'primitive':int(op[i]),'pointers':ptr[i].tolist(),'status':got['status'],'reason':got['reason']})
 result.append(dict(seed=seed,counts=counts,errors=errors))
out=dict(results=result,cpu_audit_wall_seconds=time.monotonic()-start,scope='CPU execution of actual public predicted proposals; correct-event identity checked against generator reference solely as audit target.')
Path('/tmp/p01-runtime-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
