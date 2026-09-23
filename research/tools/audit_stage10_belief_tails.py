"""Independent archived counterfactual tail reconstruction, no model inference."""
import gzip,json
from pathlib import Path
root=Path(__file__).resolve().parents[2];p=root/'research/results/stage10/beliefs/main'
def load(f):
 with gzip.open(f,'rt')as h:return json.load(h)
x=load(p/'invariance-tails.json.gz');checked=0
for row in x['rows']:
 def raw(condition):return load(p/row['run']/f'raw-{row["split"]}-{row["candidates"]}-{condition}.json.gz')
 a=raw('clean');b=raw(row['condition']);assert a['target']==b['target'];ds=[];ends=[];changes=0
 outcomes=dict(correct_correct=0,correct_wrong=0,wrong_correct=0,wrong_wrong=0)
 for i,(ep,eq,target) in enumerate(zip(a['posterior'],b['posterior'],a['target'])):
  d=max(abs(v-w)for f,g in zip(ep,eq)for v,w in zip(f,g));last=max(abs(v-w)for v,w in zip(ep[-1],eq[-1]));ds.append(d);ends.append(last)
  j=max(range(len(ep[-1])),key=ep[-1].__getitem__);k=max(range(len(eq[-1])),key=eq[-1].__getitem__);changes+=j!=k
  ca=target[-1][j]>0;cb=target[-1][k]>0;outcomes[('correct'if ca else'wrong')+'_'+('correct'if cb else'wrong')]+=1
  saved=row['episodes'][i];assert saved['max_probability_delta']==d and saved['final_probability_delta']==last and saved['final_argmax_changed']==(j!=k) and saved['clean_final_correct']==ca and saved['renamed_final_correct']==cb
 assert max(ds)==row['max']and max(ends)==row['final_max']and changes==row['final_argmax_changes']and outcomes==row['final_outcomes']
 for key,t in [('above_01',.01),('above_05',.05),('above_1',.1)]:assert sum(d>t for d in ds)==row[key]
 for key,q in [('p95',.95),('p99',.99)]:
  z=sorted(ds);pos=(len(z)-1)*q;lo=int(pos);v=z[lo]+(z[min(lo+1,len(z)-1)]-z[lo])*(pos-lo);assert abs(v-row[key])<1e-12
 checked+=1
assert checked==108
out=dict(counterfactual_cells_verified=checked,paired_episode_evaluations=checked*512,scope='All per-episode deltas, final argmax/outcome tables, tail quantiles and support counts; no new inference or probability calibration claim.')
(root/'research/results/stage10/audits/belief-counterfactual-tails.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
public=json.loads((p/'invariance-failure-events.json').read_text())
for e in public['examples']:
 for condition,field in [('clean','clean'),(e['condition'],'renamed')]:
  r=load(p/e['run']/f'raw-{e["split"]}-{e["candidates"]}-{condition}.json.gz')
  assert r['posterior'][e['episode']][e['frame']]==e[field] and r['target'][e['episode']][e['frame']]==e['target']
 inventory=set(json.loads((p/e['run']/'training-id-inventory.json').read_text()));assert [h for h in e['renamed_handles']if h not in inventory]==e['unseen_handles']
 assert max(abs(a-b)for a,b in zip(e['clean'],e['renamed']))==e['probability_delta']
out=dict(examples_verified=len(public['examples']),checks=['raw probabilities and targets','ID inventory membership','maximum probability differences'],limitation='Public feature regeneration not independently repeated; source compiler remains versioned.')
(root/'research/results/stage10/audits/belief-public-tail-check.json').write_text(json.dumps(out,indent=2)+'\n')
