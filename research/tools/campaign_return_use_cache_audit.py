"""CPU replay of actual R05 actor features and fitted comparator outputs."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with gzip.open(p,'rb') as f:return torch.load(f,map_location='cpu',weights_only=False)
def run(root):
 start=time.monotonic();torch.set_num_threads(2);root=Path(root);m=json.load(gzip.open(root/'manifest.json.gz','rt'));cache=load(root/'cache.pt.gz');logits=load(root/'logits.pt.gz');rows=json.load(gzip.open(root/'predictions.json.gz','rt'));lookup={(r['arm'],r['key'],r['delay']):r for r in rows}
 saved=torch.load(m['config']['readout'],map_location='cpu',weights_only=True);assert sha(m['config']['readout'])==m['config']['readout_sha256'];assert sha(m['config']['checkpoint'])==m['config']['checkpoint_sha256']
 accessor=nn.Linear(1024,33);accessor.load_state_dict(saved['state']);heads={}
 for f in m['fits']:
  p=root/(f['arm']+'.pt');assert sha(p)==f['checkpoint_sha256'];h=nn.Sequential(nn.Linear(35,1024),nn.GELU(),nn.Linear(1024,2));h.load_state_dict(torch.load(p,map_location='cpu',weights_only=True));heads[f['arm']]=h.eval()
 max_score_error=max_logit_error=0.;checked=0
 with torch.no_grad():
  for key,b in cache.items():
   q=b['query'];v=(b['original_value_labels'].float()-16)/2;gold=((v>q[:,0])^q[:,1].bool()).long();assert torch.equal(gold,b['original_targets'])
   if b['supplied_value_labels'] is not None:
    v=(b['supplied_value_labels'].float()-16)/2;sg=((v>q[:,0])^q[:,1].bool()).long();assert torch.equal(sg,b['supplied_targets'])
   for d,x in b['features'].items():
    score=accessor((x-saved['mean'])/saved['scale']);max_score_error=max(max_score_error,float((score-b['scores'][d]).abs().max()));assert torch.equal(score.argmax(-1),b['scores'][d].argmax(-1))
    query=q.clone();query[:,0]/=8
    for arm,h in heads.items():
     if arm=='learned':value=b['scores'][d].softmax(-1)
     elif arm=='query_only' or b['supplied_value_labels'] is None:value=torch.zeros_like(score)
     else:value=F.one_hot(b['supplied_value_labels'],33).float()
     got=h(torch.cat((value,query),1));expected=logits[f'{key}/{d}/{arm}'];max_logit_error=max(max_logit_error,float((got-expected).abs().max()));pred=got.argmax(-1)
     assert pred.tolist()==lookup[arm,key,d]['predictions'];checked+=1
 return dict(cells=checked,accessor_max_abs_logit_error=max_score_error,consumer_max_abs_logit_error=max_logit_error,all_argmax_equal=True,original_and_supplied_targets_rederived=True,checkpoint_bytes_verified=True,cpu_audit_wall_seconds=time.monotonic()-start,scope='Frozen cached workspace/accessor/consumer replay; does not independently rerun backbone or event generator.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);a=p.parse_args();r=run(a.root);Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
