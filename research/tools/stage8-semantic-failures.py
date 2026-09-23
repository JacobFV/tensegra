"""Deterministic first-record failure examples, not representative sampling."""
import argparse
import base64
import gzip
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('directories',nargs='+',type=Path)
args=parser.parse_args()
rows=[]
for directory in args.directories:
    path=directory/'calibrated-predictions-semantic-seed0-p100000.jsonl.gz'
    seen=set()
    with gzip.open(path,'rt') as stream:
        for line in stream:
            row=json.loads(line);surface=(row['language'],row['renamed'])
            if surface in seen:continue
            seen.add(surface)
            p,g=row['prediction'],row['target']
            n,_,relations=g['edges']['shape']
            pb=base64.b64decode(p['edges']['packed_b64']);gb=base64.b64decode(g['edges']['packed_b64'])
            false_positive=[];false_negative=[]
            for index in range(n*n*relations):
                i,jr=divmod(index,n*relations);j,r=divmod(jr,relations)
                pred=bool(pb[index//8]&(1<<(index%8))) and p['presence'][i] and p['presence'][j]
                gold=bool(gb[index//8]&(1<<(index%8)))
                dest=false_positive if pred and not gold else false_negative if gold and not pred else None
                if dest is not None and len(dest)<5:dest.append([i,j,r])
            fields={field:[dict(node=i,predicted=p[field][i],gold=g[field][i]) for i in range(n)
                           if g['presence'][i] and p[field][i]!=g[field][i]][:5]
                    for field in ('kind','value','copy')}
            rows.append(dict(corpus=directory.name,selection='first deterministic evaluation record for each surface; seed0,100000 presentations',
                             language=row['language'],renamed=row['renamed'],public_text=row['public_text'],graph_sha256=row['graph_sha256'],
                             metrics=row['metrics'],field_errors=fields,first_false_positive_edges=false_positive,
                             first_false_negative_edges=false_negative,edge_tuple_schema=['source node index','target node index','typed relation index']))
print(json.dumps(rows,indent=2))
