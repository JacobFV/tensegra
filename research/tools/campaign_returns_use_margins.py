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
 out.append(dict(arm=row['arm'],key=row['key'],delay=row['delay'],signed_value_minus_threshold=dict(groups)))
(a.results/'decision-margin-groups.json').write_text(json.dumps(out,indent=2)+'\n')
