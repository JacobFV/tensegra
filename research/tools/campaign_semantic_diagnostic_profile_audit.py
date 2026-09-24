"""CPU-only all-six diagnostic profile contracts, compact metrics and timing."""
import argparse,base64,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from audit_stage11_semantic_text import components,same
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();load=lambda p:json.load(gzip.open(p,'rt'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def packed(g):
 n=g['capacity'];present=np.zeros(n,bool);present[g['present']]=True;edges=np.zeros((n,n,13),bool);slots=np.full((n,n),-1,int)
 for i,j,r in g['edges']:edges[i,j,r]=True
 for i,j,s in g['slots']:slots[i,j]=s
 return dict(presence=present.tolist(),kind=g['kind'],value=g['value'],copy=g['copy'],slots=slots.tolist(),edges=dict(shape=[n,n,13],packed_b64=base64.b64encode(np.packbits(edges.flatten(),bitorder='little')).decode(),bitorder='little'))
root=a.root/'s14-motif-profile';m=load(root/'manifest.json.gz');targets=load(root/'targets.json.gz');frequency=load(root/'frequency.json.gz');assert sha(root/'targets.json.gz')==m['targets_sha256'] and sha(root/'frequency.json.gz')==m['frequency_sha256'];assert len(targets['rows'])==8
count=0
for rec in m['artifacts']:
 p=root/rec['artifact'];assert sha(p)==rec['sha256'];x=load(p);end=x['endpoint'];primaryroot=a.root/f's12-{end["arm"]}-{end["seed"]}';pm=load(primaryroot/'manifest.json.gz');primary=load(primaryroot/'evaluation-u24576.json.gz');assert end['checkpoint_sha256']==pm['curves'][-1]['checkpoint_sha256'] and end['primary_sha256']==sha(primaryroot/'evaluation-u24576.json.gz') and end['parent_manifest_sha256']==sha(primaryroot/'manifest.json.gz');assert end['thresholds']==primary['thresholds'];assert x['targets_sha256']==m['targets_sha256'] and len(x['rows'])==8
 for row,gold in zip(x['rows'],targets['rows']):
  assert row['semantic_sha256']==gold['semantic_sha256'];raw=row['raw'];cal={**raw,'edges':row['calibrated_edges'],'slots':raw['slots']+row['calibrated_extra_slots']}
  for mode,pred in [('raw',raw),('calibrated',cal)]:same(components(packed(pred),packed(gold['target'])),row[mode+'_metrics'])
  count+=1
for row,gold in zip(frequency['rows'],targets['rows']):assert row['semantic_sha256']==gold['semantic_sha256'];same(components(packed(frequency['prediction']),packed(gold['target'])),row['metrics'])
profiles={}
for lane,directory,filename in [('s14','s14-motif-profile','manifest.json.gz'),('rename','s12-renamed-profile','profile.json.gz')]:
 root=a.root/directory;x=load(root/filename);cfg=x['config'];assert cfg['job']=='profile' and len(cfg['runs'])==6
 for n,h in cfg['profile_source_sha256'].items():assert sha(a.source_root/n)==h
 occ=json.loads(next(root.glob('*.occupancy.json')).read_text());assert occ['exit_code']==0 and not occ['timed_out'];rows=x['artifacts'] if lane=='s14' else x['rows'];assert sorted((r['seed'],r['arm']) for r in rows)==[(s,a) for s in (701,702,703) for a in ('constant','decay')]
 if lane=='rename':
  assert x['examples']==48
  for row in rows:
   assert row['examples']==8 and len(row['output_sha256'])==64;pm=load(a.root/f's12-{row["arm"]}-{row["seed"]}/manifest.json.gz');assert row['checkpoint_sha256']==pm['curves'][-1]['checkpoint_sha256']
 loops=sum(r['seconds'] for r in rows);fixed=occ['process_occupancy_seconds']-loops;projected=fixed+128*loops;assert projected<420
 profiles[lane]=dict(full_seconds=occ['process_occupancy_seconds'],preflight_seconds=occ['preflight_seconds'],six_endpoint_loop_seconds=loops,fixed_remainder_seconds=fixed,conservative_main_projection=projected,cap_seconds=420,supported_for_separate_release=True)
out=dict(compact_policy_graphs_reconstructed=count,frequency_graphs_reconstructed=8,all_six_threshold_checkpoint_bindings_verified=True,profiles=profiles,cpu_audit_wall_seconds=time.monotonic()-t,scope='Mechanical first8 only, no efficacy selection. Loop scales128x including model loads; fixed setup/preflight once, not entirewall128x. Rename profile archives timing/output digests only, no raw graph metrics available for replay.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
