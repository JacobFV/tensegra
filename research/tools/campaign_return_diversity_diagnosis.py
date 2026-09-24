"""CPU-only localization from immutable R10 arrays; no fitting or selection."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory)
raw=(root/'predictions.json.gz').read_bytes();rows=json.loads(gzip.decompress(raw))
manifest=json.loads(gzip.decompress((root/'manifest.json.gz').read_bytes()))
out={'predictions_sha256':hashlib.sha256(raw).hexdigest(),'late_cells':[],'curves':{}}
for r in rows:
    if not(r['head'].startswith('balanced_') and r['split'] in ('train','validation_grid') and r['target_delay']==16):continue
    n=int(r['head'].split('_')[1]) if r['split']=='train' else len(r['targets']['value'])
    total=Counter();wrong=Counter();distance=Counter();types=Counter();operations=Counter()
    for i in range(n):
        typ,val=r['targets']['type'][i],r['targets']['value'][i];pred=r['predictions']['value'][i];total[typ,val]+=1
        if pred!=val:
            wrong[typ,val]+=1;distance[pred-val]+=1;types[typ]+=1;operations[r['targets']['operation'][i]]+=1
    out['late_cells'].append(dict(head=r['head'],split=r['split'],total=n,errors=sum(wrong.values()),
        error_type=dict(types),error_operation=dict(operations),error_halfunit_distance=dict(distance),
        strata=[dict(type=k[0],value=(k[1]-16)/2,wrong=wrong[k],total=total[k])for k in sorted(total)]))
for f in manifest['fits']:
    out['curves'][f['arm']]={'loss_300_step_means':[sum(f['losses'][i:i+300])/len(f['losses'][i:i+300]) for i in range(0,len(f['losses']),300)],
         'calibration_minima':[dict(step=c['step'],correct=min(x['correct']for x in c['cells']),denominator=1024)for c in f['curve']]}
(root/'failure-localization.json').write_text(json.dumps(out,indent=2)+'\n')
