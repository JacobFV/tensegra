"""S15 fixed paired development decision; CPU archived outputs, no fitting."""
import collections,gzip,hashlib,json,math
from pathlib import Path
import numpy as np

def decisions(acquisition_exact,acquisition_delta,retention_exact,retention_delta,recombination_exact,recombination_delta):
 acquisition=acquisition_exact>=52 and acquisition_delta>=26
 retention=retention_exact>=103 and retention_delta>=-51
 recombination=recombination_exact>=52 and recombination_delta>=26
 return dict(criteria=dict(acquisition=acquisition,retention=retention,recombination=recombination,retention_exact=retention_exact,retention_delta=retention_delta),confirmation_eligible=acquisition and retention and recombination,one_paired_extension_eligible=acquisition and retention and not recombination)

def bootstrap_samples(delta,rng):
 delta=np.asarray(delta);assert delta.shape==(512,)
 return delta[rng.integers(0,512,size=(10000,512))].mean(axis=1)

def validate_visits(visits):
 assert len(visits)==4096 and set(visits)=={8} and sum(visits)==32768

def main():
 root=Path('research/results/campaign-01/semantics');load=lambda p:json.load(gzip.open(p,'rt'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 cache=[json.loads(line) for line in gzip.open(root/'s15-shape-cache-v2/development.jsonl.gz','rt')];by_identity={r['semantic_sha256']:r for r in cache};assert len(by_identity)==2048
 manifests={};results={};endpoint={};manifest_hashes={}
 for arm in ('control','mixed'):
  folder=root/f's15-{arm}-main-v3';m=load(folder/'manifest.json.gz');manifests[arm]=m;manifest_hashes[arm]=sha(folder/'manifest.json.gz')
  validate_visits(m['visits'])
  assert m['added_presentations']==32768 and m['common_presentations']==16384 and m['initial_calibration_replay_exact']
  config_path=Path(f'configs/campaign-s15-{arm}-main-frozen-v3.json');config=json.loads(config_path.read_text())
  expected_config={'control':'576d7e31c3b42395cb762c49c86fc11da7da62de949fe6e68a53a29108b7388c','mixed':'42216efd35879f66df125c2cacde8e334d78140917a7e75aac746ccf36ab672b'}
  assert sha(config_path)==expected_config[arm] and m['config']==config
  receipt=json.loads((folder/f's15-{arm}-main-v3.occupancy.json').read_text())
  assert receipt['exit_code']==0 and not receipt['timed_out'] and receipt['config_sha256']==expected_config[arm] and receipt['cap_seconds']==1100
  assert receipt['wrapper_sha256']==config['launch_source_sha256']['campaign_semantics_shape_launch.py']
  assert receipt['primary_matrix']['parent_checkpoint_sha256']==config['parent_checkpoint_sha256']=='3799ade595500a083b9a558b6a5bc97c5cbc1b8bdce6e7e4b86d97b937ea373e'
  assert m['inherited_optimizer_steps']==[24576.] and m['inherited_presentations']==196608
  for group in ('source_sha256','launch_source_sha256'):
   for name,expected in config[group].items():assert sha(Path('src/topoformer')/name)==expected,name
  assert sha(root/'s15-shape-cache-v2/audit.json')==config['audit_sha256']==receipt['primary_matrix']['audit_sha256']
  for split,expected in config['cache_sha256'].items():assert sha(root/'s15-shape-cache-v2'/(split+'.jsonl.gz'))==expected==receipt['primary_matrix']['cache_sha256'][split]
  assert sha(root/'s01-data/train.jsonl.gz')==config['calibration_cache_sha256']
  assert sha(root/'s15-bindings/original-data-audit.json')==config['data_audit_sha256']
  assert sha(root/'s11-lr-decay-n8192-196k-dev201/evaluation-u24576.json.gz')==config['parent_evaluation_sha256']
  assert [c['added_update'] for c in m['curves']]==[0,1024,2048,4096]
  assert all(math.isfinite(v) for p in m['losses'] for v in p['parts'].values())
  curves=[]
  for curve in m['curves']:
   p=folder/curve['evaluation']['artifact'];assert sha(p)==curve['evaluation']['sha256'];d=load(p);assert len(d['rows'])==2048 and {r['semantic_sha256'] for r in d['rows']}==set(by_identity)
   cells={}
   for a in (3,4):
    for f in (3,4):
     key=f'{a}x{f}';rows=[r for r in d['rows'] if (by_identity[r['semantic_sha256']]['arity'],by_identity[r['semantic_sha256']]['facts'])==(a,f)];assert len(rows)==512
     cells[key]={}
     for policy in ('raw','calibrated'):
      metrics=[r[policy+'_metrics'] for r in rows];edges={}
      for kind in ('typed_edge','ordered_edge'):
       tp=sum(v[kind]['true_positive'] for v in metrics);pred=sum(v[kind]['predicted_count'] for v in metrics);gold=sum(v[kind]['gold_count'] for v in metrics)
       edges[kind]=dict(true_positive=tp,predicted=pred,gold=gold,precision=tp/pred if pred else 0.,recall=tp/gold if gold else 0.,f1=2*tp/(pred+gold) if pred+gold else 1.)
      cells[key][policy]=dict(examples=512,exact=sum(v['semantic_equivalence'] for v in metrics),exact_copy=sum(v['identity_copy_accuracy']==1 for v in metrics),mean_copy=sum(v['identity_copy_accuracy'] for v in metrics)/512,edges=edges)
   curves.append(dict(added_update=curve['added_update'],cells=cells))
   if curve['added_update']==4096:endpoint[arm]=d['rows']
  results[arm]=dict(curves=curves,optimizer_tokens=m['optimizer_tokens'],optimizer_nodes=m['optimizer_nodes'],optimizer_edges=m['optimizer_edges'],training_seconds=m['training_seconds'])
 for field in ('initial_state_sha256','inherited_optimizer_steps','construction_sequence_sha256','common_pair_sequence_sha256','common_presentations','visits'):
  assert manifests['control'][field]==manifests['mixed'][field],field
 paired={};rng=np.random.default_rng(15015)
 for a in (3,4):
  for f in (3,4):
   key=f'{a}x{f}';paired[key]={}
   for policy in ('raw','calibrated'):
    transitions=collections.Counter();deltas=[]
    for x,y in zip(endpoint['control'],endpoint['mixed']):
     assert x['semantic_sha256']==y['semantic_sha256'];c=by_identity[x['semantic_sha256']]
     if (c['arity'],c['facts'])!=(a,f):continue
     u=int(x[policy+'_metrics']['semantic_equivalence']);v=int(y[policy+'_metrics']['semantic_equivalence']);transitions[f'{u}->{v}']+=1;deltas.append(v-u)
    delta=np.asarray(deltas);samples=bootstrap_samples(delta,rng)
    paired[key][policy]=dict(transitions=dict(transitions),mixed_minus_control=int(delta.sum()),difference_percentage_points=100*float(delta.mean()),paired_event_bootstrap_95_percentage_points=(100*np.quantile(samples,[.025,.975])).tolist())
 final=results['mixed']['curves'][-1]['cells']
 retention_count=sum(final[f'4x{f}']['calibrated']['exact'] for f in (3,4));retention_delta=sum(paired[f'4x{f}']['calibrated']['mixed_minus_control'] for f in (3,4))
 decision=decisions(final['3x3']['calibrated']['exact'],paired['3x3']['calibrated']['mixed_minus_control'],retention_count,retention_delta,final['3x4']['calibrated']['exact'],paired['3x4']['calibrated']['mixed_minus_control'])
 report=dict(source_sha256=sha(Path(__file__)),cache_sha256=sha(root/'s15-shape-cache-v2/development.jsonl.gz'),manifest_sha256=manifest_hashes,results=results,paired=paired,**decision,scope='S14-informed single-parent development pair; trained3x3 acquisition vs untrained3x4 combination; marginal per-cell/policy paired event bootstrap conditional on this fixed training pair, not training-seed uncertainty; independently verified parent checkpoint bytes bound by frozen config and durable preflight receipt; no automatic successor')
 (root/'s15-paired-analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('results','paired')},indent=2))

if __name__=='__main__':main()
