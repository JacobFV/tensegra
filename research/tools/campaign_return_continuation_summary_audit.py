"""Independent fixed-endpoint R11 counts and development decision."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();rows=json.load(gzip.open(a.root/'predictions.json.gz','rt'));s=json.loads((a.root/'summary.json').read_text());idx={(r['head'],r['split'],r['target_delay'],r['distractors']):r for r in rows};eq={}
for k,r in idx.items():
 e={f:np.asarray(r['predictions'][f])==r['targets'][f] for f in r['targets']};e['nonvalue_joint']=np.logical_and.reduce([v for f,v in e.items() if f!='value']);e['joint']=np.logical_and.reduce([e[f] for f in r['targets']]);eq[k]=e
def verify(c,mask=None):
 k=tuple(c[f] for f in ('head','split','target_delay','distractors'));e=eq[k];mask=slice(None) if mask is None else mask;assert c['total']==len(e['value'][mask]);assert c['correct']=={f:int(v[mask].sum()) for f,v in e.items()};return k
assert len(s['cells'])==len(idx)==108 and len(s['strata'])==936 and len(s['paired'])==72
for c in s['cells']:verify(c)
minima={}
for c in s['strata']:
 k=tuple(c[f] for f in ('head','split','target_delay','distractors'));r=idx[k];mask=(np.asarray(r['targets']['type'])==c['type'])&(np.asarray(r['targets']['value'])==c['value_label']);assert mask.sum()==128;verify(c,mask);group=k[:2];minima[group]=min(minima.get(group,128),c['correct']['value'])
for c in s['paired']:
 k=c['split'],c['delay'],c['distractors'];ref=eq[('reference',)+k]['value'];arm=eq[(c['arm'],)+k]['value'];v=np.bincount(2*ref.astype(int)+arm.astype(int),minlength=4);assert c['outcomes']=={f'{i}->{j}':int(v[2*i+j]) for i in (0,1) for j in (0,1)}
gates={}
for arm in ('reference','constant','decay'):
 mixture=min(int(v['value'].sum())/len(v['value']) for k,v in eq.items() if k[:2]==(arm,'calibration'));grid=minima[arm,'calibration_grid']/128;train=int(eq[arm,'train',16,2]['value'].sum());gates[arm]=dict(mixture_minimum=mixture,grid_minimum=grid,training16_correct=train,advance=mixture>=.98 and grid>=122/128 and train>=65405)
assert gates==s['gates'];assert not any(gates[x]['advance'] for x in ('constant','decay'));assert s['selected'] is None;assert s['lower_lr_specific_gain']==(minima['decay','calibration_grid']-minima['constant','calibration_grid']>=4);assert hashlib.sha256((a.root/'predictions.json.gz').read_bytes()).hexdigest()==s['predictions_sha256']
out=dict(cells_verified=108,strata_verified=936,paired_reference_tables_verified=72,gates=gates,selected=None,lower_lr_specific_gain=s['lower_lr_specific_gain'],validation_grid_minima={a:minima[a,'validation_grid'] for a in ('reference','constant','decay')},input_sha256={str(Path('research/results/campaign-01/returns/r11-development')/name):hashlib.sha256((a.root/name).read_bytes()).hexdigest() for name in ('summary.json','predictions.json.gz')},cpu_audit_wall_seconds=time.monotonic()-t,scope='Both fixed development endpoints fail worst-calibration122/128. Decay improves fitting and reused validation but cannot override failure. Difference from constant supports the registered relative-LR comparison, not broad capacity/information-loss diagnosis. No confirmation or further optimization search.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
