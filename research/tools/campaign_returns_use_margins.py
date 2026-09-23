"""Describe decision precision by true query margin, without changing task gates."""
import argparse,gzip,io,json
from collections import defaultdict
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
cache=torch.load(io.BytesIO(gzip.decompress((a.results/'cache.pt.gz').read_bytes())),map_location='cpu',weights_only=True)
rows=json.load(gzip.open(a.results/'predictions.json.gz','rt'));out=[]
for row in rows:
 if not(row['key'].startswith('validation/') or row['key'].startswith('test/') or row['key'].startswith('balanced/')):continue
 batch=cache[row['key']];margin=(batch['original_value_labels'].float()-16)/2-batch['query'][:,0];groups=defaultdict(lambda:dict(correct=0,total=0))
 for i,(pred,target) in enumerate(zip(row['predictions'],row['original_targets'])):
  key=str(float(margin[i]));groups[key]['total']+=1;groups[key]['correct']+=int(pred==target)
 scalar=batch['scores'][row['delay']].argmax(-1);truth=batch['original_value_labels'];wrong=scalar!=truth
 predicted_value=(scalar.float()-16)/2
 exact_decision=((predicted_value>batch['query'][:,0]) ^ batch['query'][:,1].bool()).long()
 task_correct=torch.tensor(row['predictions'])==torch.tensor(row['original_targets'])
 out.append(dict(arm=row['arm'],key=row['key'],delay=row['delay'],signed_value_minus_threshold=dict(groups),scalar_wrong_total=int(wrong.sum()),task_correct_on_scalar_wrong=int((task_correct&wrong).sum()),exact_argmax_comparison_correct_on_scalar_wrong=int(((exact_decision==batch['original_targets'])&wrong).sum())))
(a.results/'decision-margin-groups.json').write_text(json.dumps(out,indent=2)+'\n')
