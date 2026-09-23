"""Independent stdlib reconstruction of Stage10 fixed-node edge acquisition."""
import gzip, hashlib, itertools, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'research/results/stage10/semantic-edge'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def threshold(pairs):
    pairs=sorted(pairs); err=sum(not y for _,y in pairs)
    best=(err,pairs[0][0]-1)
    for score,group in itertools.groupby(pairs,key=lambda x:x[0]):
        ys=[y for _,y in group];err+=sum(ys)-(len(ys)-sum(ys))
        if err<best[0]:best=(err,score)
    return best
x=load(BASE/'exposure.json.gz');checks=0;rows=[]
assert hashlib.sha256(json.dumps(x['config'],sort_keys=True).encode()).hexdigest()==x['config_sha256']
for name,h in x['source_sha256'].items(): assert sha(BASE/'frozen-source'/name)==h
assert [r['seed'] for r in x['runs']]==[20,21,22]
for run in x['runs']:
    assert run['width']==1024 and run['actual_unique_graphs']==8
    assert [c['update'] for c in run['curves']]==[0,25,100,300,600,1200]
    for c in run['curves']:
        p=BASE/Path(c['score_file']).name;assert sha(p)==c['score_sha256'];s=load(p)
        assert len(s['graphs'])==8
        for r,t in enumerate(c['thresholds']):
            pairs=[(a[r],bool(b[r])) for g in s['graphs'] for a,b in zip(g['scores'],g['truth'])]
            errors,cut=threshold(pairs)
            assert errors==t['train_errors'] and abs(cut-t['threshold'])<1e-6
            assert sum(y for _,y in pairs)==t['positive'] and sum(not y for _,y in pairs)==t['negative']
        for policy in ['raw','calibrated']:
            exact=errors=sloterrors=0
            for g,out in zip(s['graphs'],c[policy]):
                n=g['nodes'];assert out['seed']==g['graph_seed']
                gold={(i//n,i%n,r) for i,a in enumerate(g['truth']) for r,y in enumerate(a) if y}
                pred={(i//n,i%n,r) for i,a in enumerate(g['scores']) for r,v in enumerate(a) if v>(0 if policy=='raw' else c['thresholds'][r]['threshold'])}
                assert gold==set(map(tuple,out['gold_edges'])) and pred==set(map(tuple,out['predicted_edges']))
                e=len(gold^pred); se=sum(a!=b for a,b in zip(out['predicted_slots'],out['gold_slots']))
                assert e==out['edge_errors'] and se==out['slot_errors_on_gold_edges']
                assert (e==0 and se==0)==out['exact_graph']
                exact+=e==0 and se==0; errors+=e;sloterrors+=se;checks+=1
            rows.append(dict(seed=run['seed'],update=c['update'],policy=policy,exact=exact,total=8,edge_errors=errors,slot_errors=sloterrors))
out=dict(audit='independent archived-score and threshold reconstruction; not inference',source_files_verified=len(x['source_sha256']),score_files_verified=18,graph_decisions_verified=checks,rows=rows,final_raw_gate=all(r['exact']==8 for r in rows if r['update']==1200 and r['policy']=='raw'),final_calibrated_gate=all(r['exact']==8 for r in rows if r['update']==1200 and r['policy']=='calibrated'),interpretation='Privileged eight-graph fixed-set acquisition; no heldout semantics or programmable attention claim.')
p=ROOT/'research/results/stage10/audits/semantic-edge-acquisition.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
