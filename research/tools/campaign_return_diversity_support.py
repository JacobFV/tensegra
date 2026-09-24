import argparse,json,time,hashlib
from pathlib import Path
import torch
from topoformer.retention_data import make_batch
from topoformer.campaign_returns_balanced_diversity import nested_indices
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
started=time.monotonic();torch.set_num_threads(2)
c=json.loads(Path(a.config).read_text());b=make_batch(c['pool_seed'],c['pool_size'],distractors=2)
i=nested_indices(b,max(c['fit_sizes']),c['strata_seed'])
strata=[(t,y)for t,ys in ((0,range(0,33,2)),(1,range(33)),(2,(16,18)))for y in ys]
out={'config_sha256':hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),'pool_size':len(b['targets']['value']),'pool_support':{f'{t}/{y}':int(((b['targets']['type']==t)&(b['targets']['value']==y)).sum())for t,y in strata},'prefixes':{}}
for n in c['fit_sizes']:
 ids=i[:n];out['prefixes'][str(n)]={'distinct_indices':len(ids.unique()),'strata_support':{f'{t}/{y}':int(((b['targets']['type'][ids]==t)&(b['targets']['value'][ids]==y)).sum())for t,y in strata}}
out['seconds']=time.monotonic()-started
Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(out['seconds'])
