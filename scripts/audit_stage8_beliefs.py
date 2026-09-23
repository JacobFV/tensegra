"""Reconstruct belief aggregates from full exported posteriors (stdlib only)."""
import argparse
import gzip
import json
import math
from pathlib import Path


def audit(root):
    errors=[]; cells=0; samples=0
    for directory in sorted(root.glob('*-*')):
        if not (directory/'metrics.json').exists(): continue
        rows=json.loads((directory/'metrics.json').read_text())
        for row in rows:
            path=directory/f"raw-{row['split']}-{row['candidates']}-{row['condition']}.json.gz"
            data=json.load(gzip.open(path,'rt'))
            p,q=data['posterior'],data['target']; n=len(p)
            if n!=row['count']: errors.append([str(path),'count'])
            cells+=1; samples+=n
            for t,reported in enumerate(row['frames']):
                support=mass=l1=entropy=0.; bins=[[] for _ in range(10)]
                for predictions,targets in zip(p,q):
                    a,b=predictions[t],targets[t]
                    if abs(sum(a)-1)>2e-5 or abs(sum(b)-1)>2e-5: errors.append([str(path),'normalization'])
                    j=max(range(len(a)),key=a.__getitem__)
                    support+=b[j]>0; mass+=sum(x for x,y in zip(a,b) if y==0)
                    l1+=sum(abs(x-y) for x,y in zip(a,b))
                    entropy-=sum(x*math.log(max(x,1e-12)) for x in a)
                    bins[min(9,int(a[j]*10))].append((a[j],b[j]))
                calculated=dict(support_accuracy=support/n,impossible_mass=mass/n,posterior_l1=l1/n,entropy=entropy/n)
                for name,value in calculated.items():
                    if abs(value-reported[name])>3e-6: errors.append([str(path),t,name,value,reported[name]])
                for values,saved in zip(bins,reported['calibration']):
                    if len(values)!=saved['count']: errors.append([str(path),t,'bin_count'])
                    if values:
                        for idx,name in [(0,'confidence'),(1,'accuracy')]:
                            value=sum(v[idx] for v in values)/len(values)
                            if abs(value-saved[name])>3e-6: errors.append([str(path),t,name])
    return dict(cells=cells,episodes=samples,errors=errors,passed=cells>0 and not errors)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args(); result=audit(args.root)
    if args.output: args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));raise SystemExit(0 if result['passed'] else 1)
