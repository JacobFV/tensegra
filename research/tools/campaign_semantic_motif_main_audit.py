"""Independent compact-graph, public target and inherited-state S14 main audit."""
import argparse,ast,collections,gzip,hashlib,json,re,time
from pathlib import Path
from audit_stage11_semantic_text import same,edge_set

def metrics(p,g):
 P=set(p['present']);G=set(g['present']);pe={tuple(e) for e in p['edges'] if e[0] in P and e[1] in P};ge={tuple(e) for e in g['edges']};ps={(i,j):s for i,j,s in p['slots']};gs={(i,j):s for i,j,s in g['slots']};C={i for i,v in enumerate(g['copy']) if v>=0}
 def count(a,b):
  tp=len(a&b);n=len(a);m=len(b);return dict(true_positive=tp,predicted_count=n,gold_count=m,precision=tp/n if n else 0.,recall=tp/m if m else 0.,f1=2*tp/(n+m) if n+m else 1.)
 exact=0 not in g['value'] and P==G and all(p['kind'][i]==g['kind'][i] for i in G) and pe==ge and all(p['value'][i]==v for i,v in enumerate(g['value']) if v>=0) and all(p['copy'][i]==g['copy'][i] for i in C) and all(ps.get((i,j),-1)==gs.get((i,j),-1) for i,j,r in ge)
 return dict(node=count(P,G),typed_edge=count(pe,ge),ordered_edge=count({(i,j,r,ps[i,j]) for i,j,r in pe if ps.get((i,j),-1)>=0},{(i,j,r,gs[i,j]) for i,j,r in ge if gs.get((i,j),-1)>=0}),node_type_accuracy=sum(p['kind'][i]==g['kind'][i] for i in G)/len(G),identity_copy_accuracy=sum(p['copy'][i]==g['copy'][i] for i in C)/len(C) if C else 1.,entity_equivalence=sum((p['copy'][i]==p['copy'][j])==(g['copy'][i]==g['copy'][j]) for i in C for j in C)/len(C)**2 if C else 1.,semantic_equivalence=float(exact))
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic();base=a.repo/'research/results/campaign-01/semantics';root=base/'s14-motif-main';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.load(gzip.open(p,'rt'));m=load(root/'manifest.json.gz');cfg=m['config'];targets=load(root/'targets.json.gz');freq=load(root/'frequency.json.gz');analysis=json.loads((root/'analysis.json').read_text())
assert cfg['job']=='main' and len(cfg['runs'])==6;assert targets['cache_sha256']==cfg['cache_sha256']==sha(base/'s14-arity3-cache/diagnostic.jsonl.gz');assert sha(root/'targets.json.gz')==m['targets_sha256']==analysis['targets_sha256'];assert sha(root/'frequency.json.gz')==m['frequency_sha256'];assert sha(root/'manifest.json.gz')==analysis['manifest_sha256']
for n,h in cfg['main_source_sha256'].items():assert sha(a.repo/'src/topoformer'/n)==h
assert m['source_sha256']==cfg['main_source_sha256']['campaign_semantics_motif_inference.py']
rows=[json.loads(s) for s in gzip.open(base/'s14-arity3-cache/diagnostic.jsonl.gz','rt')];assert len(rows)==len(targets['rows'])==1024
meta=json.loads((base/'s14-arity3-cache/audit.json').read_text());vocab=meta['config']['value_vocabulary'];const={}
for n in ast.parse((a.repo/'src/topoformer/thinking_language.py').read_text()).body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('KINDS','ROLES'):const[n.targets[0].id]=ast.literal_eval(n.value)
for row,g in zip(rows,targets['rows']):
 assert row['semantic_sha256']==g['semantic_sha256'];tok=re.findall(r'\w+|[^\w\s]',row['text']);assert hashlib.sha256(row['text'].encode()).hexdigest()==row['public_sha256'];gold=dict(capacity=128,present=list(range(len(row['nodes']))),kind=[0]*128,value=[-1]*128,copy=[-1]*128,edges=[],slots=[])
 for i,(kind,value) in enumerate(row['nodes']):
  gold['kind'][i]=const['KINDS'].index(kind)
  if kind in ('ident','entity'):gold['copy'][i]=tok.index(value)
  else:gold['value'][i]=vocab.index(json.dumps(value,sort_keys=True))
 for i,j,r,s in row['edges']:
  gold['edges'].append([i,j,const['ROLES'].index(r)])
  gold['slots'].append([i,j,-1 if s is None else s])
 gold['slots']=[list(t) for t in set(map(tuple,gold['slots']))]
 assert all(gold[k]==g['target'][k] for k in ('capacity','present','kind','value','copy'));assert sorted(gold['edges'])==sorted(g['target']['edges']) and sorted(gold['slots'])==sorted(g['target']['slots'])
baseline=load(base/'s01-frequency.json.gz');assert sha(base/'s01-frequency.json.gz')==cfg['baseline_sha256']==freq['baseline_sha256'];bp=baseline['prediction'];fp=freq['prediction'];assert fp['present']==[i for i,v in enumerate(bp['presence']) if v];assert all(fp[k]==bp[k] for k in ('kind','value','copy'));assert set(map(tuple,fp['edges']))==edge_set(bp);assert all(s==bp['slots'][i][j] for i,j,s in fp['slots'])
results=[]
for rec in m['artifacts']:
 path=root/rec['artifact'];assert sha(path)==rec['sha256'];x=load(path);end=x['endpoint'];pr=base/f's12-{end["arm"]}-{end["seed"]}';pm=load(pr/'manifest.json.gz');primary=load(pr/'evaluation-u24576.json.gz');assert end['checkpoint_sha256']==pm['curves'][-1]['checkpoint_sha256'];assert end['primary_sha256']==sha(pr/'evaluation-u24576.json.gz') and end['parent_manifest_sha256']==sha(pr/'manifest.json.gz');assert end['thresholds']==primary['thresholds'];assert x['targets_sha256']==m['targets_sha256'] and len(x['rows'])==1024;agg=next(y for y in analysis['results'] if (y['seed'],y['arm'])==(end['seed'],end['arm']));assert agg['artifact_sha256']==rec['sha256'];modes={'raw':[],'calibrated':[]}
 for r,g,row in zip(x['rows'],targets['rows'],rows):
  assert r['semantic_sha256']==g['semantic_sha256'] and r['seed']==row['seed'] and r['tree_shape_sha256']==row['tree_shape_sha256'] and r['node_count']==len(row['nodes']);raw=r['raw'];cal={**raw,'edges':r['calibrated_edges'],'slots':raw['slots']+r['calibrated_extra_slots']}
  for mode,pred in [('raw',raw),('calibrated',cal)]:
   got=metrics(pred,g['target']);same(got,r[mode+'_metrics']);modes[mode].append(got)
 for mode,values in modes.items():
  expected=dict(exact=sum(z['semantic_equivalence'] for z in values)/1024,copy=sum(z['identity_copy_accuracy'] for z in values)/1024,node_type=sum(z['node_type_accuracy'] for z in values)/1024)
  for kind in ('node','typed_edge','ordered_edge'):
   c={k:sum(z[kind][k] for z in values) for k in ('true_positive','predicted_count','gold_count')};c['f1']=2*c['true_positive']/(c['predicted_count']+c['gold_count']);expected[kind]=c
  same(expected,agg[mode]);assert expected['exact']==x['summary'][mode+'_exact']==0
 results.append(dict(seed=end['seed'],arm=end['arm'],raw=agg['raw'],calibrated=agg['calibrated']))
assert sorted((r['seed'],r['arm']) for r in results)==[(s,a) for s in (701,702,703) for a in ('constant','decay')]
for r,g in zip(freq['rows'],targets['rows']):assert r['semantic_sha256']==g['semantic_sha256'];same(metrics(freq['prediction'],g['target']),r['metrics'])
occ=json.loads((root/'s14-motif-main-v1.occupancy.json').read_text());assert occ['exit_code']==0 and not occ['timed_out'] and occ['primary_matrix']==cfg['primary_matrix'];assert occ['config_sha256']==sha(a.repo/'configs/campaign-s14-motif-main-frozen-v1.json')
out=dict(public_targets_reconstructed=1024,policy_graphs_reconstructed=12288,frequency_graphs_reconstructed=1024,all_six_inherited_thresholds_checkpoints_verified=True,results=results,full_occupancy_seconds=occ['process_occupancy_seconds'],input_sha256={str(p.relative_to(a.repo)):sha(p) for p in root.iterdir() if p.is_file()},cpu_audit_wall_seconds=time.monotonic()-tick,scope='All six frozen endpoints score zero exact on the fixed fresh arity3/two-motif population. Partial component metrics remain; no broad semantic-collapse or depth/language conclusion follows. Public inputs/targets independently reconstructed from previously audited fixed cache; no GPU or model inference.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ('results','input_sha256')})
