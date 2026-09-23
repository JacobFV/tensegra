"""Independent stdlib regeneration of public primitive strata and ID comparison."""
import gzip,json,random,argparse
from pathlib import Path

def primitive_values(count,seed,n):
 values=[]
 for index in range(count):
  rng=random.Random(seed*1000003+index);keys=[]
  while len(keys)<n+3:
   key=tuple(rng.choice((-1.,1.)) for _ in range(16))
   if key not in keys:keys.append(key)
  records=[]
  while len(records)<n:
   a,b=rng.sample(range(2,min(len(keys),max(5,n//2+2))),2);record=(rng.randrange(2),rng.randrange(2),a,b)
   if record not in records:records.append(record)
  rng.shuffle(records);gold=rng.randrange(n);values.append(records[gold][0])
 return values

def load(path):
 with gzip.open(path,'rt') as f:return json.load(f)
def audit(root):
 saved=json.loads((root/'primitive-strata.json').read_text());errors=[];checked=0;paired=0
 for folder in sorted(root.glob('*-[012]')):
  for file in sorted(folder.glob('supplied_empty_prior-*.json.gz')):
   raw=load(file);values=primitive_values(len(raw['posterior']),raw['event_seed'],raw['candidates'])
   for value in (0,1):
    inds=[i for i,v in enumerate(values) if v==value];correct=0;err=[]
    for i in inds:
     ps,qs=raw['posterior'][i],raw['target'][i];j=max(range(len(ps[-1])),key=ps[-1].__getitem__);correct+=qs[-1][j]>0
     err.extend(sum(abs(a-b) for a,b in zip(p,q)) for p,q in zip(ps,qs))
    row=next(r for r in saved if r['model']==folder.name and r['condition']==raw['condition'] and r['primitive_value']==value)
    if row['count']!=len(inds) or row['final_correct']!=correct or abs(row['mean_l1']-sum(err)/len(err))>2e-6:errors.append([folder.name,raw['condition'],value])
    checked+=1
   if raw['condition']=='distinct_equal_seen_id':
    other=load(folder/file.name.replace('distinct_equal_seen_id','distinct_equal_unseen_id'))
    if raw['target']!=other['target'] or raw['event_seed']!=other['event_seed'] or raw['empty_ledger']!=other['empty_ledger']:errors.append([folder.name,'paired_target'])
    paired+=1
 return dict(strata=checked,seen_unseen_pairs=paired,errors=errors,passed=checked==48 and paired==6 and not errors)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.root);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));raise SystemExit(0 if r['passed'] else 1)
