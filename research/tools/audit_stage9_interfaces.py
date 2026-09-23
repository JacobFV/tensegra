"""Independent stdlib reconstruction of Stage9 frozen-interface predictions."""
import argparse, gzip, hashlib, json
from pathlib import Path

def read(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return json.load(f)

def return_probes(root):
    path=next((root/name for name in ('predictions.json.gz','predictions.json','results.json') if (root/name).exists()),None)
    rows=[r for r in read(path) if 'metrics' in r]
    errors=[];selections={}
    for i,r in enumerate(rows):
        p,q=r['predictions'],r['targets'];n=len(q)
        if len(p)!=n or not n:errors.append([i,'length']);continue
        distances=[(a-b)/2 for a,b in zip(p,q)]
        m=dict(correct=sum(a==b for a,b in zip(p,q)),total=n,mae=sum(abs(d) for d in distances)/n,signed_error=sum(distances)/n,within_half=sum(abs(d)<=.5 for d in distances))
        for k,v in m.items():
            if abs(v-r['metrics'][k])>2e-6:errors.append([i,k,v,r['metrics'][k]])
        if 'selection' in r:
            grid=r['selection']['validation_grid'];winner=max(grid,key=lambda v:v['correct'])
            if winner['alpha']!=r['alpha']:errors.append([i,'validation_selection'])
            if r['split']=='validation' and winner['correct']!=m['correct']:errors.append([i,'selected_validation_count'])
            key=(r['run'],r['boundary'],r['family']);s=json.dumps(r['selection'],sort_keys=True)
            if key in selections and selections[key]!=s:errors.append([i,'split_selection_mismatch'])
            selections[key]=s
            if r['selection']['parameters']!=(1025*33 if r['family']=='categorical_ridge' else 1025):errors.append([i,'probe_parameter_count'])
    return dict(kind='archived prediction reconstruction, not inference',rows=len(rows),probe_selections=len(selections),errors=errors,passed=bool(rows) and not errors)

def belief(root):
    errors=[];cells=[]
    for metrics in sorted(root.glob('*/metrics.json')):
      if metrics.parent.name=='profile':continue
      for r in read(metrics):
        path=metrics.parent/r['raw'];raw=read(path);p,q=raw['posterior'],raw['target'];n=len(p);T=len(p[0]);l1=mass=correct=0.;frame_errors=[]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=r['raw_sha256']:errors.append([str(path),'sha'])
        for t in range(T):
            sm=sl=0.;sc=amb=nt=0;null_signed=null_mae=null_brier=0.;tp=fp=0;bins=[[] for _ in range(10)]
            for a,b in zip(p,q):
                a,b=a[t],b[t];j=max(range(len(a)),key=a.__getitem__);d=sum(abs(x-y) for x,y in zip(a,b));im=sum(x for x,y in zip(a,b) if y==0)
                sl+=d;sm+=im;sc+=b[j]>0;amb+=sum(v>0 for v in b)>1;nt+=b[-1];err=a[-1]-b[-1];null_signed+=err;null_mae+=abs(err);null_brier+=err*err
                tp+=j==len(a)-1 and b[-1]==1;fp+=j==len(a)-1 and b[-1]==0;bins[min(9,int(a[j]*10))].append((a[j],b[j]))
            values=dict(posterior_l1=sl/n,impossible_mass=sm/n,support_correct=sc,ambiguous=amb,null_targets=nt,null_signed_error=null_signed/n,null_mae=null_mae/n,null_brier=null_brier/n,no_match_true_positive=tp,no_match_false_positive=fp)
            for k,v in values.items():
                if abs(v-r['frames'][t][k])>3e-6:errors.append([str(path),t,k])
            for entries,saved in zip(bins,r['frames'][t]['calibration']):
                if len(entries)!=saved['count']:errors.append([str(path),t,'bin_count'])
                if entries:
                    for k,idx in [('confidence',0),('expected_correctness',1)]:
                        if abs(sum(e[idx] for e in entries)/len(entries)-saved[k])>3e-6:errors.append([str(path),t,k])
            l1+=sl;mass+=sm;correct=sc
        passed=n>=512 and correct/n>(.98 if r['candidates']==8 else .95) and l1/(n*T)<.05 and mass/(n*T)<.01
        if passed!=r['passed']:errors.append([str(path),'gate'])
        cells.append(dict(mode=r['mode'],seed=r['seed'],arm=r['arm'],split=r['split'],candidates=r['candidates'],condition=r['condition'],count=n,passed=passed,mean_l1=l1/(n*T),mean_impossible=mass/(n*T)))
    expected={(m,s,a,sp,n,c) for m in ('protected','recurrent') for s in (0,1,2) for a in ('learned_prior','supplied_empty_prior') for sp in ('validation','test') for n in (8,16,32) for c in ('clean','reorder','duplicate','long_duplicate','contradiction','retract','partial','empty','full_retract','distinct_equal','candidate_permutation','id_rename')}
    if root.name=='idlocal':expected={(m,s,a,'development',8,c) for m in ('protected','recurrent') for s in (0,1,2) for a in ('learned_prior','supplied_empty_prior') for c in ('clean','distinct_equal_seen_id','distinct_equal_unseen_id','id_permute_seen')}
    actual={(r['mode'],r['seed'],r['arm'],r['split'],r['candidates'],r['condition']) for r in cells}
    return dict(cells=cells,count=len(cells),complete_matrix=actual==expected,missing_cells=sorted(expected-actual),errors=errors,passed=bool(cells) and not errors)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['belief','returns']);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=belief(a.root) if a.kind=='belief' else return_probes(a.root);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('cells','missing_cells')}));raise SystemExit(0 if r['passed'] else 1)
