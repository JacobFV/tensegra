"""CPU-only frozen all-six lexical-renaming outcome reconstruction."""
import argparse,gzip,hashlib,json,re,time
from pathlib import Path
from audit_stage11_semantic_text import components,same
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();base=a.repo/'research/results/campaign-01/semantics';root=base/'s12-renamed-inference';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.load(gzip.open(p,'rt'));m=load(root/'manifest.json.gz');cfg=m['config'];cache=base/'s12-rename-case-audit/renamed_confirmation.jsonl.gz';assert sha(cache)==cfg['cache_sha256'];rows=[json.loads(s) for s in gzip.open(cache,'rt')];assert len(rows)==1024;data=json.loads((base/'s12-rename-case-audit/audit.json').read_text());assert data['renamed_cache_sha256']==cfg['cache_sha256'];assert len(data['mapping'])==len(set(data['mapping'].values()))==11
for n,h in cfg['main_source_sha256'].items():assert sha(a.repo/'src/topoformer'/n)==h
assert m['source_sha256']==cfg['main_source_sha256']['campaign_semantics_confirmation_rename.py'];results=[]
for rec in m['artifacts']:
 path=root/rec['artifact'];assert sha(path)==rec['sha256'];x=load(path);pr=base/f's12-{x["arm"]}-{x["seed"]}';pm=load(pr/'manifest.json.gz');original=load(pr/'evaluation-u24576.json.gz');assert x['checkpoint_sha256']==pm['curves'][-1]['checkpoint_sha256'];assert x['primary_sha256']==sha(pr/'evaluation-u24576.json.gz');assert x['thresholds']==original['thresholds'];assert x['renamed_cache_sha256']==cfg['cache_sha256'];assert len(x['rows'])==1024;metrics={'raw':[],'calibrated':[]};tables={k:{f'{i}->{j}':0 for i in (0,1) for j in (0,1)} for k in metrics}
 for r,old,row in zip(x['rows'],original['rows'],rows):
  assert r['seed']==old['seed']==row['seed'] and r['semantic_sha256']==old['semantic_sha256']==row['semantic_sha256'] and r['graph_sha256']==row['graph_sha256'];assert r['target']==old['target'];tokens=re.findall(r'\w+|[^\w\s]',row['text']);assert hashlib.sha256(row['text'].encode()).hexdigest()==row['public_sha256']
  for (kind,value),copy in zip(row['nodes'],r['target']['copy']):
   if kind in ('ident','entity'):assert copy==tokens.index(value)
  for mode,pred in [('raw',r['raw']),('calibrated',{**r['raw'],'edges':r['calibrated_edges']})]:
   got=components(pred,r['target']);same(got,r[mode+'_metrics']);metrics[mode].append(got);tables[mode][f'{int(old[mode+"_metrics"]["semantic_equivalence"])}->{int(got["semantic_equivalence"])}']+=1
 def agg(v):
  result=dict(exact=int(sum(z['semantic_equivalence'] for z in v)),copy=sum(z['identity_copy_accuracy'] for z in v)/1024,node_type=sum(z['node_type_accuracy'] for z in v)/1024)
  for kind in ('node','typed_edge','ordered_edge'):
   c={k:sum(z[kind][k] for z in v) for k in ('true_positive','predicted_count','gold_count')};c['f1']=2*c['true_positive']/(c['predicted_count']+c['gold_count']);result[kind]=c
  return result
 results.append(dict(seed=x['seed'],arm=x['arm'],raw=agg(metrics['raw']),calibrated=agg(metrics['calibrated']),paired_original_to_renamed=tables))
assert sorted((r['seed'],r['arm']) for r in results)==[(s,a) for s in (701,702,703) for a in ('constant','decay')];occ=json.loads((root/'s12-renamed-main-v1.occupancy.json').read_text());assert occ['exit_code']==0 and not occ['timed_out'] and occ['primary_matrix']==cfg['primary_matrix'];assert occ['config_sha256']==sha(a.repo/'configs/campaign-s12-renamed-main-frozen-v1.json')
out=dict(examples_per_endpoint=1024,graph_metrics_reconstructed=12288,all_targets_equal_original_indexed_targets=True,novel_public_first_copy_positions_verified=True,original_thresholds_and_checkpoint_bindings_verified=True,results=results,full_occupancy_seconds=occ['process_occupancy_seconds'],input_sha256={str(p.relative_to(a.repo)):sha(p) for p in root.iterdir() if p.is_file()},cpu_audit_wall_seconds=time.monotonic()-t,scope='Frozen-model lexical substitution diagnostic with case-preserving unseen strings, equal graph/index/copy targets and inherited TRAIN thresholds. Novel lexical hashes change actor inputs; loss is not a claim of alpha-renaming equivariance or broad semantic impossibility.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(dict(seconds=out['cpu_audit_wall_seconds'],results=[(r['seed'],r['arm'],r['raw']['exact'],r['calibrated']['exact']) for r in results]))
