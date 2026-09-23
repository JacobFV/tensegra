#!/usr/bin/env python3
"""Field-conditioned scalar errors from frozen raw probe predictions."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();d=Path(a.directory)
source=d/'predictions.json'
rows=json.loads(source.read_text()) if source.exists() else json.loads(gzip.decompress((d/'predictions.json.gz').read_bytes()))
events=json.loads((d/'events.json').read_text()); result=[]
for row in rows:
 if row['split']!='test': continue
 target,pred=row['targets'],row['predictions'];errors=[(p-t)/2 for p,t in zip(pred,target)]
 item={k:row[k] for k in ('run','split','boundary','family')};item['confusion']=[[a,b,c] for (a,b),c in sorted(Counter(zip(target,pred)).items())];item['groups']={}
 labels={'value':target,'type':events['test']['targets']['type'],'operation':events['test']['targets']['operation'],
         'sign':[int(v>16)-int(v<16) for v in target],'magnitude':[abs(v-16)/2 for v in target]}
 for name,groups in labels.items():
  item['groups'][name]={str(label):{'total':sum(v==label for v in groups),'correct':sum(v==label and err==0 for v,err in zip(groups,errors)),
    'absolute_error_sum':sum(abs(err) for v,err in zip(groups,errors) if v==label),'signed_error_sum':sum(err for v,err in zip(groups,errors) if v==label)} for label in sorted(set(groups))}
 result.append(item)
(d/'scalar-groups.json.gz').write_bytes(gzip.compress(json.dumps(result,separators=(',',':')).encode(),mtime=0))
