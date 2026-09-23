"""Paired confirmation intervals and immutable checkpoint bytes for A06."""
import argparse,json,hashlib,subprocess,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();allraw={};receipts=[];summaries=[]
for seed in (601,602,603):
 initial=[]
 for arm in ('soft','hard','context','none'):
  root=a.root/str(seed)/arm;cfg=json.loads((root/'config.json').read_text());m=json.loads((root/'manifest.json').read_text());initial.append(m['initial_tensor_sha256']);raw=dict(np.load(root/'eval-01000.npz'));allraw[seed,arm]=raw
  path=f'/home/brandonin/topoformer-campaign01/attention/a06/{seed}/{arm}/checkpoint.pt';got=subprocess.check_output(['ssh','gb10-direct','sha256sum',path],text=True).split()[0];assert got==m['checkpoint_sha256'];receipts.append(dict(seed=seed,arm=arm,path=path,sha256=got))
  for ci in range(9):
   base=allraw[601,'soft']
   for field in ('gold','start','relation','successor','oracle_terminal'):
    assert np.array_equal(raw[f'c{ci}_{field}'],base[f'c{ci}_{field}'])
  for ci in (16,17,18):
   group=cfg['conditions'][ci]['data_group'];baseidx=next(i for i,c in enumerate(cfg['conditions'])if c.get('data_group')==group and not any(k.endswith('override')for k in c));assert np.array_equal(raw[f'c{ci}_gold'],raw[f'c{baseidx}_gold'])
  if arm=='context':
   for ci,baseidx in ((19,0),(20,6),(21,8)):assert np.array_equal(raw[f'c{ci}_gold'],raw[f'c{baseidx}_gold'])
  summaries.append(dict(seed=seed,arm=arm,raw_joint=int(raw['c8_task'].sum()),content16_joint=int(raw['c18_task'].sum()),both16_joint=int(raw['c21_task'].sum())if arm=='context'else None,examples=1024))
 assert len(set(initial))==1
rng=np.random.default_rng(96301);index=rng.integers(0,1024,(4000,1024));intervals=[]
for arm in ('soft','hard','context'):
 for contrast in ('content_vs_raw', 'both_vs_content')if arm=='context'else ('content_vs_raw',):
  before,after=(8,18)if contrast=='content_vs_raw'else(18,21);differences=np.stack([allraw[s,arm][f'c{after}_task'].astype(float)-allraw[s,arm][f'c{before}_task']for s in(601,602,603)])
  means=differences.mean(0);draws=means[index].mean(1);intervals.append(dict(arm=arm,contrast=contrast,per_seed=differences.mean(1).tolist(),mean=means.mean(),event_bootstrap_95=np.quantile(draws,[.025,.975]).tolist()))
# Strongest methods share the same final events: paired equality is exact here.
for seed in(601,602,603):assert np.array_equal(allraw[seed,'soft']['c18_task'],allraw[seed,'hard']['c18_task'])and np.array_equal(allraw[seed,'soft']['c18_task'],allraw[seed,'context']['c21_task'])
out=dict(checkpoints=receipts,summary=summaries,paired_intervals=intervals,unique_primary_events=1024,seeds=3,bootstrap_replicates=4000,scope='Shared event resampling across all seeds, preserving within-event dependence. Seed variability reported separately; three seed/event replicates are not three independent datasets. No unique soft-bias advantage.',cpu_audit_wall_seconds=time.monotonic()-t);a.output.write_text(json.dumps(out,indent=2)+'\n');print(summaries)
