"""Standard-library reconstruction of historical belief errors; no model inference."""
import gzip,json,hashlib,argparse
from pathlib import Path

def audit(root):
 rows=[]
 for folder in sorted(root.glob('*-[012]')):
  for path in sorted(folder.glob('raw-*.json.gz')):
   with gzip.open(path,'rt') as f: data=json.load(f)
   errors=[]; null=[]; impossible=[]; support=[]; ambiguous=0; no_match=0; signed=[]
   for ps,qs in zip(data['posterior'],data['target']):
    for p,q in zip(ps,qs):
     errors.append(sum(abs(a-b) for a,b in zip(p,q)))
     impossible.append(sum(a for a,b in zip(p,q) if b==0))
     support.append(q[max(range(len(p)),key=p.__getitem__)]>0)
     ambiguous+=sum(x>0 for x in q)>1; no_match+=q[-1]==1
     signed.append(p[-1]-q[-1])
    p,q=ps[0],qs[0]; null.append({'u':p[-1],'l1':sum(abs(a-b) for a,b in zip(p,q)), 'candidate_range':max(p[:-1])-min(p[:-1])})
   mean=lambda xs:sum(xs)/len(xs)
   rows.append(dict(arm=folder.name,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),episodes=len(data['posterior']),frames=len(errors),mean_l1=mean(errors),mean_impossible=mean(impossible),support_correct=sum(support),ambiguous_frames=ambiguous,no_match_frames=no_match,null_signed_error=mean(signed),max_l1=max(errors),initial_null=mean([x['u'] for x in null]),initial_l1=mean([x['l1'] for x in null]),max_initial_l1_minus_2u=max(abs(x['l1']-2*x['u']) for x in null),initial_candidate_range=max(x['candidate_range'] for x in null)))
 return {'kind':'archived_predictions_not_inference','rows':rows,'cells':len(rows),'episode_evaluations':sum(r['episodes'] for r in rows)}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('research/results/stage8/belief-idmatched'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.write_text(json.dumps(audit(a.root),indent=2))
