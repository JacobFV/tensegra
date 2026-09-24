"""Independent fixed-pair S15 curve aggregation, paired events and bootstrap."""
import argparse,collections,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();root=a.repo/'research/results/campaign-01/semantics';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.load(gzip.open(p,'rt'));summary=json.loads((root/'s15-paired-analysis.json').read_text());dev=[json.loads(s) for s in gzip.open(root/'s15-shape-cache-v2/development.jsonl.gz','rt')];assert sha(root/'s15-shape-cache-v2/development.jsonl.gz')==summary['cache_sha256'];assert sha(a.repo/'research/campaigns/extended-01/semantics/S15-analyze.py')==summary['source_sha256'];endpoints={};initial={};ms={}
for arm in ('control','mixed'):
 receipt=json.loads((a.review/f'S15-{arm}-main-audit.json').read_text())
 for path,h in receipt['input_sha256'].items():assert sha(a.repo/path)==h
 run=root/f's15-{arm}-main-v3';m=load(run/'manifest.json.gz');ms[arm]=m;assert sha(run/'manifest.json.gz')==summary['manifest_sha256'][arm]
 expected=[]
 for point in m['curves']:
  e=load(run/point['evaluation']['artifact']);rows=e['rows'];assert [r['semantic_sha256'] for r in rows]==[r['semantic_sha256'] for r in dev];cells={}
  if point['added_update']==0:initial[arm]=hashlib.sha256(json.dumps([rows,e['thresholds'],e['train_rows']],sort_keys=True).encode()).hexdigest()
  for arity in (3,4):
   for facts in (3,4):
    chosen=[r for r,d in zip(rows,dev) if (d['arity'],d['facts'])==(arity,facts)];assert len(chosen)==512;cell={}
    for policy in ('raw','calibrated'):
     values=[r[policy+'_metrics'] for r in chosen];edges={}
     for kind in ('typed_edge','ordered_edge'):
      tp=sum(v[kind]['true_positive'] for v in values);n=sum(v[kind]['predicted_count'] for v in values);g=sum(v[kind]['gold_count'] for v in values);edges[kind]=dict(true_positive=tp,predicted=n,gold=g,precision=tp/n if n else 0.,recall=tp/g if g else 0.,f1=2*tp/(n+g) if n+g else 1.)
     cell[policy]=dict(examples=512,exact=sum(v['semantic_equivalence'] for v in values),exact_copy=sum(v['identity_copy_accuracy']==1 for v in values),mean_copy=sum(v['identity_copy_accuracy'] for v in values)/512,edges=edges)
    cells[f'{arity}x{facts}']=cell
  expected.append(dict(added_update=point['added_update'],cells=cells))
  if point['added_update']==4096:endpoints[arm]=rows
 assert summary['results'][arm]==dict(curves=expected,**{k:m[k] for k in ('optimizer_tokens','optimizer_nodes','optimizer_edges','training_seconds')})
assert initial['control']==initial['mixed']
for k in ('initial_state_sha256','inherited_optimizer_steps','construction_sequence_sha256','common_pair_sequence_sha256','common_presentations','visits'):assert ms['control'][k]==ms['mixed'][k]
assert ms['control']['pair_sequence_sha256']!=ms['mixed']['pair_sequence_sha256']
paired={};rng=np.random.default_rng(15015)
for arity in (3,4):
 for facts in (3,4):
  key=f'{arity}x{facts}';paired[key]={}
  for policy in ('raw','calibrated'):
   selected=[(x,y) for x,y,d in zip(endpoints['control'],endpoints['mixed'],dev) if (d['arity'],d['facts'])==(arity,facts)];table=collections.Counter((int(x[policy+'_metrics']['semantic_equivalence']),int(y[policy+'_metrics']['semantic_equivalence'])) for x,y in selected);delta=np.array([b-a for a,b in [(int(x[policy+'_metrics']['semantic_equivalence']),int(y[policy+'_metrics']['semantic_equivalence'])) for x,y in selected]])
   # Equivalent independent event-multiplicity implementation of frozen draws.
   indices=rng.integers(0,512,size=(10000,512));samples=np.array([np.bincount(row,minlength=512)@delta/512 for row in indices]);interval=100*np.quantile(samples,[.025,.975]);r=summary['paired'][key][policy];assert r['transitions']=={f'{x}->{y}':n for (x,y),n in table.items()};assert r['mixed_minus_control']==int(delta.sum());assert r['difference_percentage_points']==100*float(delta.mean());assert np.array_equal(r['paired_event_bootstrap_95_percentage_points'],interval);paired[key][policy]=r
final=summary['results']['mixed']['curves'][-1]['cells'];retain=sum(final[f'4x{f}']['calibrated']['exact'] for f in (3,4));retain_delta=sum(paired[f'4x{f}']['calibrated']['mixed_minus_control'] for f in (3,4));criteria=dict(acquisition=final['3x3']['calibrated']['exact']>=52 and paired['3x3']['calibrated']['mixed_minus_control']>=26,retention=retain>=103 and retain_delta>=-51,recombination=final['3x4']['calibrated']['exact']>=52 and paired['3x4']['calibrated']['mixed_minus_control']>=26,retention_exact=retain,retention_delta=retain_delta);assert criteria==summary['criteria'];confirm=all(criteria[k] for k in ('acquisition','retention','recombination'));extend=criteria['acquisition'] and criteria['retention'] and not criteria['recombination'];assert summary['confirmation_eligible']==confirm and summary['one_paired_extension_eligible']==extend
paths=[root/'s15-paired-analysis.json',a.repo/'research/campaigns/extended-01/semantics/S15-analyze.py'];report=a.repo/'research/campaigns/extended-01/semantics/S15-report.md'
if report.exists():paths.append(report)
out=dict(all_curve_components_verified=True,paired_initial_predictions_exact=True,paired_initial_sampler_common_query_stream_equal=True,bootstrap_samples_per_cell_policy=10000,bootstrap_rng_seed=15015,independent_event_multiplicity_intervals_exact=True,paired=paired,criteria=criteria,confirmation_eligible=confirm,one_paired_extension_eligible=extend,input_sha256={str(p.relative_to(a.repo)):sha(p) for p in paths},cpu_audit_wall_seconds=time.monotonic()-t,scope='One S14-informed development pair. Marginal paired-event uncertainty, not seed uncertainty. Acquisition, retention and recombination all fail; neither fixed successor is eligible. Partial component acquisition remains distinct from complete-graph success.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ('paired','input_sha256')})
