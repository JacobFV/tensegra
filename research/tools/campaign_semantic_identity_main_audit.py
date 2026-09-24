"""S16 independent compact predictions, copy restoration and paired summaries."""
import argparse,gzip,hashlib,json,re,struct,time
from pathlib import Path
from audit_stage11_semantic_text import edge_set,same
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--semantics',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();base=a.repo/'research/results/campaign-01/semantics';root=base/'s16-main';outdir=root/'s16-normalized-main';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.load(gzip.open(p,'rt'));m=load(outdir/'manifest.json.gz');c=m['config'];summary=json.loads((root/'paired-summary.json').read_text());profile=json.loads((a.review/'S16-profile-state-audit.json').read_text());inventory=json.loads((root/'archive-sha256.json').read_text())
for path,h in inventory.items():assert sha(root/path)==h
config=a.repo/'research/campaigns/extended-01/semantics/S16-main-frozen-v1.json';assert c==json.loads(config.read_text()) and sha(config)=='1e95311a823bf753d7496852a954d9c2d39e212687970adb420641e2cba60496'
for name,h in c['source_sha256'].items():assert sha(a.repo/'src/topoformer'/name)==h
cache=a.semantics/'research/results/campaign-01/semantics/s01-data/reserved_confirmation.jsonl.gz';assert sha(cache)==c['cache_sha256'];public=[json.loads(line)['text'] for line in gzip.open(cache,'rt')];assert len(public)==1024
occ=json.loads((root/'s16-main-launch.occupancy.json').read_text());outer=json.loads((root/'outer-command-accounting.json').read_text());started=json.loads((root/'s16-main-launch.started.json').read_text());done=json.loads((outdir/'completion.json').read_text());clock=dict(line.split('=') for line in (root/'s16-main-launch.outer.txt').read_text().splitlines());assert occ['exit_code']==outer['exit_code']==int(clock['exit_code'])==0 and not occ['timed_out'] and occ['error'] is None;assert not occ['gpu_processes'].strip() and done['success'];assert occ['config_sha256']==started['config_sha256']==sha(config);assert occ['wrapper_sha256']==c['source_sha256']['campaign_semantics_s16_launch.py'];assert occ['cap_seconds']==480;assert float(clock['elapsed_seconds'])==outer['outer_seconds']==114.73 and outer['conservative_charge_seconds']==114.74;assert hashlib.sha256(outer['command'].encode()).hexdigest()==outer['command_sha256'];assert m['process_seconds']<=done['process_seconds']<=occ['process_occupancy_seconds']<=114.74

def metric(p,g):
 P=set(p['present']);G={i for i,v in enumerate(g['presence']) if v};pe={tuple(e) for e in p['edges'] if e[0] in P and e[1] in P};ge=edge_set(g);slots={(i,j):v for i,j,v in p['slots']};C={i for i,v in enumerate(g['copy']) if v>=0}
 def count(x,y):
  tp=len(x&y);n=len(x);z=len(y);return dict(true_positive=tp,predicted_count=n,gold_count=z,precision=tp/n if n else 0.,recall=tp/z if z else 0.,f1=2*tp/(n+z) if n+z else 1.)
 exact=0 not in g['value'] and P==G and pe==ge and all(p['kind'][i]==g['kind'][i] for i in G) and all(p['value'][i]==v for i,v in enumerate(g['value']) if v>=0) and all(p['copy'][i]==g['copy'][i] for i in C) and all(slots.get((i,j),-1)==g['slots'][i][j] for i,j,r in ge)
 return dict(node=count(P,G),typed_edge=count(pe,ge),ordered_edge=count({(i,j,r,slots[i,j]) for i,j,r in pe if slots.get((i,j),-1)>=0},{(i,j,r,g['slots'][i][j]) for i,j,r in ge if g['slots'][i][j]>=0}),node_type_accuracy=sum(p['kind'][i]==g['kind'][i] for i in G)/len(G),identity_copy_accuracy=sum(p['copy'][i]==g['copy'][i] for i in C)/len(C) if C else 1.,entity_equivalence=sum((p['copy'][i]==p['copy'][j])==(g['copy'][i]==g['copy'][j]) for i in C for j in C)/len(C)**2 if C else 1.,semantic_equivalence=float(exact))
def aggregate(values):
 n=len(values);out=dict(examples=n,exact=sum(int(v['semantic_equivalence']) for v in values))
 for k in ('node_type_accuracy','identity_copy_accuracy','entity_equivalence'):out[k]=sum(v[k] for v in values)/n
 for k in ('node','typed_edge','ordered_edge'):
  counts={key:sum(v[k][key] for v in values) for key in ('true_positive','predicted_count','gold_count')};out[k]={**counts,'micro_f1':2*counts['true_positive']/max(1,counts['predicted_count']+counts['gold_count'])}
 return out
results=[];cells=0;restored=0;ids=None
for record,binding in zip(m['artifacts'],c['primary_matrix']):
 seed,arm=record['seed'],record['arm'];assert (seed,arm)==(binding['seed'],binding['arm']);assert record['checkpoint_sha256']==binding['checkpoint_sha256'];assert record['tensor_before_sha256']==record['tensor_after_sha256']==profile['saved_checkpoint_tensor_hashes'][f'{seed}-{arm}'];assert record['examples']==1024
 h=hashlib.sha256();h.update(json.dumps(['thresholds','torch.float32',[13]]).encode());h.update(struct.pack('<13f',*record['thresholds']));assert h.hexdigest()==record['threshold_tensor_sha256']
 primary_path=a.semantics/f'research/results/campaign-01/semantics/s12-{arm}-{seed}/evaluation-u24576.json.gz';assert sha(primary_path)==record['primary_sha256'];primary=load(primary_path);assert primary['thresholds']==record['thresholds'];path=outdir/record['artifact'];assert sha(path)==record['sha256'];data=load(path);rows=data['rows'];assert len(rows)==1024
 for k,v in data['endpoint'].items():
  if k!='seconds':assert record[k]==v
 if ids is None:ids=[r['semantic_sha256'] for r in rows]
 assert ids==[r['semantic_sha256'] for r in rows]==[r['semantic_sha256'] for r in primary['rows']]
 hashed=hashlib.sha256();policies={k:[] for k in ('raw','calibrated')}
 for i,(r,old,text) in enumerate(zip(rows,primary['rows'],public)):
  assert r['index']==i;tok=re.findall(r'\w+|[^\w\s]',text);assert r['raw']['capacity']==128
  predicted=r['raw']['copy'];assert all(j==-1 or (0<=j<len(tok) and tok.index(tok[j])==j) for j in predicted);assert r['copied_original_public_tokens']==[None if j==-1 else tok[j] for j in predicted];restored+=len(predicted)
  packed={k:r[k] for k in ('raw','calibrated_edges','calibrated_extra_slots','copied_original_public_tokens')};hashed.update(json.dumps(packed,sort_keys=True).encode());cal={**r['raw'],'edges':r['calibrated_edges'],'slots':r['raw']['slots']+r['calibrated_extra_slots']}
  for policy,pred in [('raw',r['raw']),('calibrated',cal)]:
   got=metric(pred,old['target']);same(got,r['metrics'][policy]);assert r['original_metrics'][policy]==old[policy+'_metrics'];repair=bool(got['semantic_equivalence'] and not old[policy+'_metrics']['semantic_equivalence']);regress=bool(old[policy+'_metrics']['semantic_equivalence'] and not got['semantic_equivalence']);assert r['transitions'][policy]==dict(repair=repair,regression=regress);policies[policy].append(got);cells+=1
 assert hashed.hexdigest()==record['output_sha256'];result=dict(seed=seed,arm=arm,artifact=record['artifact'],policies={})
 for policy,values in policies.items():
  before=[r[policy+'_metrics'] for r in primary['rows']];repairs=sum(not x['semantic_equivalence'] and bool(y['semantic_equivalence']) for x,y in zip(before,values));regressions=sum(bool(x['semantic_equivalence']) and not y['semantic_equivalence'] for x,y in zip(before,values));result['policies'][policy]=dict(original=aggregate(before),normalized=aggregate([r['metrics'][policy] for r in rows]),repairs=repairs,regressions=regressions,net_exact_change=repairs-regressions);assert record['summary'][policy]['exact']==sum(v['semantic_equivalence'] for v in values) and record['summary'][policy]['repairs']==repairs and record['summary'][policy]['regressions']==regressions
 results.append(result)
assert results==summary['endpoints'];assert len(results)==6 and summary['distinct_semantic_instances']==1024 and summary['model_instance_cells']==6144
paths=[root/path for path in inventory]+[a.repo/'research/campaigns/extended-01/semantics/S16-results.md',a.repo/'research/campaigns/extended-01/semantics/S16-analyze.py'];out=dict(policy_graphs_reconstructed=cells,copy_indices_restored_verified=restored,all_output_stream_hashes_exact=True,all_frozen_checkpoint_tensor_threshold_and_reference_bindings=True,all_paired_components_repairs_regressions_verified=True,endpoints=[dict(seed=r['seed'],arm=r['arm'],policies={p:{k:v for k,v in x.items() if k in ('repairs','regressions','net_exact_change')}|dict(original_exact=x['original']['exact'],normalized_exact=x['normalized']['exact']) for p,x in r['policies'].items()}) for r in results],full_outer_seconds=114.73,conservative_charge_seconds=114.74,input_sha256={str(p.relative_to(a.repo)):sha(p) for p in paths},cpu_audit_wall_seconds=time.monotonic()-t,scope='Six frozen models/same1024instances, not6144independent samples. Calibrated decay exactness declines66→48/131→103/139→120, constants0. Programmed public-input rename equality is separate from acquired invariance and semantic correctness; no normalized-renamed forwards or historical gate promotion.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ('input_sha256','endpoints')})
