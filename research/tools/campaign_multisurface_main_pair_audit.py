"""Independent S13 paired fixed-endpoint summary and calibration support."""
import argparse,base64,collections,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();root=a.repo/'research/results/campaign-01/semantics';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.load(gzip.open(p,'rt'));summary=json.loads((root/'s13-paired-analysis.json').read_text());ms={};evaluations={};calrows=0
for arm in ('english','mixed'):
 run=root/f's13-{arm}-main';m=load(run/'manifest.json.gz');ms[arm]=m;assert sha(run/'manifest.json.gz')==summary['manifest_sha256'][arm];sr=summary['results'][arm];assert sr['optimizer_tokens']==m['optimizer_tokens'] and sr['training_seconds']==m['training_seconds'];assert [c['added_update'] for c in sr['curves']]==[0,1024,2048,4096]
 for point,reported in zip(m['curves'],sr['curves']):
  update=point['added_update'];surface={}
  for language in ('english','spanish'):
   d=load(run/point[language]['artifact']);evaluations[arm,update,language]=d;surface[language]=d;records=reported['surfaces'][language]
   if language=='english':
    assert d['thresholds']==[r['threshold'] for r in d['calibration']];cal=np.load(run/d['calibration_data_artifact']);assert len(d['train_rows'])==128
    for i,tr in enumerate(d['train_rows']):
     lo,hi=cal['offsets'][i:i+2];pairs=cal['pairs'][lo:hi];presence=np.asarray(tr['raw']['presence']);expected=np.argwhere(presence[:,None]&presence[None,:]);assert np.array_equal(pairs,expected);edge=tr['target']['edges'];gold=np.unpackbits(np.frombuffer(base64.b64decode(edge['packed_b64']),dtype=np.uint8),bitorder='little')[:np.prod(edge['shape'])].reshape(edge['shape']);assert np.array_equal(cal['targets'][lo:hi],gold[pairs[:,0],pairs[:,1]]);calrows+=1
   else:assert d['thresholds']==surface['english']['thresholds']
   for policy in ('raw','calibrated'):
    metrics=[r[policy+'_metrics'] for r in d['rows']];r=records[policy];assert r['exact']==sum(v['semantic_equivalence'] for v in metrics);assert r['exact_copy']==sum(v['identity_copy_accuracy']==1 for v in metrics);assert abs(r['mean_copy']-np.mean([v['identity_copy_accuracy'] for v in metrics]))<1e-12
    for kind in ('typed_edge','ordered_edge'):
     tp=sum(v[kind]['true_positive'] for v in metrics);n=sum(v[kind]['predicted_count'] for v in metrics);g=sum(v[kind]['gold_count'] for v in metrics);assert r['edges'][kind]==dict(true_positive=tp,predicted=n,gold=g,precision=tp/n if n else 0.,recall=tp/g if g else 0.,f1=2*tp/(n+g) if n+g else 1.)
    sizes=collections.defaultdict(lambda:[0,0])
    for row in d['rows']:n=sum(row['target']['presence']);sizes[n][0]+=1;sizes[n][1]+=row[policy+'_metrics']['semantic_equivalence']
    assert r['by_node_count']=={str(n):dict(examples=v[0],exact=v[1]) for n,v in sizes.items()}
    if language=='english':assert point[language]['dev_'+policy]['exact']==r['exact'] and point[language]['dev_'+policy]['copy']==r['mean_copy']
  for en,es in zip(surface['english']['rows'],surface['spanish']['rows']):assert en['semantic_sha256']==es['semantic_sha256']
  for policy in ('raw','calibrated'):
   pairs=collections.Counter(f"{int(en[policy+'_metrics']['semantic_equivalence'])}->{int(es[policy+'_metrics']['semantic_equivalence'])}" for en,es in zip(surface['english']['rows'],surface['spanish']['rows']));assert dict(pairs)==reported['english_to_spanish'][policy]
for key in ('initial_state_sha256','inherited_presentations','inherited_optimizer_steps','renderer_phases','first_batch_replay','added_visits','construction_sequence_sha256','pair_sequence_sha256'):assert ms['english'][key]==ms['mixed'][key]
for lang in ('english','spanish'):
 initial=[evaluations[arm,0,lang] for arm in ('english','mixed')];assert initial[0]['rows']==initial[1]['rows'] and initial[0]['thresholds']==initial[1]['thresholds']
 for policy in ('raw','calibrated'):
  rows=[evaluations[arm,4096,lang]['rows'] for arm in ('english','mixed')];assert [r['semantic_sha256'] for r in rows[0]]==[r['semantic_sha256'] for r in rows[1]];assert [r['target'] for r in rows[0]]==[r['target'] for r in rows[1]];pairs=collections.Counter(f"{int(x[policy+'_metrics']['semantic_equivalence'])}->{int(y[policy+'_metrics']['semantic_equivalence'])}" for x,y in zip(*rows));assert summary['paired_endpoint'][lang][policy]==dict(transitions=dict(pairs),mixed_minus_english=pairs['0->1']-pairs['1->0'])
final=summary['results']['mixed']['curves'][-1]['surfaces'];criteria=dict(spanish_at_least_52=final['spanish']['calibrated']['exact']>=52,spanish_gain_at_least_26=summary['paired_endpoint']['spanish']['calibrated']['mixed_minus_english']>=26,english_loss_at_most_26=summary['paired_endpoint']['english']['calibrated']['mixed_minus_english']>=-26);assert criteria==summary['promotion_criteria'] and all(criteria.values())==summary['promotion_pass'];gain=final['spanish']['calibrated']['exact']-summary['results']['mixed']['curves'][2]['surfaces']['spanish']['calibrated']['exact'];assert gain==summary['spanish_2048_to_4096_gain']==0
out=dict(calibration_graph_target_masks_verified=calrows,all_bilingual_curve_components_and_size_counts_verified=True,paired_initial_predictions_and_sampler_hashes_equal=True,paired_endpoint=summary['paired_endpoint'],promotion_criteria=criteria,promotion_pass=False,spanish_late_gain=0,input_sha256={str((root/'s13-paired-analysis.json').relative_to(a.repo)):sha(root/'s13-paired-analysis.json')},cpu_audit_wall_seconds=time.monotonic()-t,scope='One fixed development pair: both Spanish exact0/512, mixed English18 calibrated versus English-only135. All three criteria fail and late Spanish gain0; no extension, confirmation or general semantic claim follows.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
