"""Independent array reconstruction of the frozen R10 development summary."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np

def audit(root):
 t=time.monotonic(); rows=json.load(gzip.open(root/'predictions.json.gz','rt'));summary=json.loads((root/'summary.json').read_text());index={};arrays={}
 for r in rows:
  key=(r['head'],r['split'],r['target_delay'],r['distractors']);assert key not in index;index[key]=r
  fields=tuple(r['targets']);eq={f:np.asarray(r['predictions'][f])==r['targets'][f] for f in fields};eq['nonvalue_joint']=np.logical_and.reduce([eq[f] for f in fields if f!='value']);eq['joint']=np.logical_and.reduce([eq[f] for f in fields]);arrays[key]=eq
 def check(c,mask=None):
  key=tuple(c[f] for f in ('head','split','target_delay','distractors'));eq=arrays[key];mask=slice(None) if mask is None else mask
  assert c['correct']=={f:int(v[mask].sum()) for f,v in eq.items()};assert c['total']==len(eq['value'][mask]);return key
 assert len(summary['cells'])==len(index)==108
 for c in summary['cells']:
  key=check(c)
  for label,mask in [('in_pool',slice(0,int(c['head'].split('_')[-1]) if c['head']!='unchanged' else 0)),('heldout_prefix_tail',slice(int(c['head'].split('_')[-1]) if c['head']!='unchanged' else 0,None))]:
   if label in c:check(dict(c,**c[label]),mask)
 assert len(summary['strata'])==18*52
 minima={}
 for c in summary['strata']:
  key=tuple(c[f] for f in ('head','split','target_delay','distractors'));r=index[key];mask=(np.asarray(r['targets']['type'])==c['type'])&(np.asarray(r['targets']['value'])==c['value_label']);check(c,mask);assert int(mask.sum())==128
  k=(c['head'],c['split']);minima[k]=min(minima.get(k,128),c['correct']['value'])
 assert len(summary['paired'])==108
 for c in summary['paired']:
  a,b=[arrays[(h,c['split'],c['delay'],c['distractors'])][c['field']] for h in c['arms']];v=np.bincount(2*a.astype(int)+b.astype(int),minlength=4);assert c['outcomes']=={f'{i}->{j}':int(v[2*i+j]) for i in (0,1) for j in (0,1)}
 small,large='balanced_16384','balanced_65536';lo=minima[small,'calibration_grid'];hi=minima[large,'calibration_grid'];mix=all(int(v['value'].sum())>=1004 for k,v in arrays.items() if k[0]==large and k[1]=='calibration')
 gate=dict(mixture_pass=mix,calibration_grid_minima={small:lo/128,large:hi/128},minimum_at_least_95pct=hi>=122,minimum_gain_at_least_4_of_128=hi-lo>=4);gate['advance']=mix and hi>=122 and hi-lo>=4;assert gate==summary['development_gate'];assert not gate['advance']
 rawhash=hashlib.sha256((root/'predictions.json.gz').read_bytes()).hexdigest();assert rawhash==summary['predictions_sha256']
 return dict(cells_verified=108,strata_verified=936,paired_tables_verified=108,development_gate=gate,grid_minimum_counts={f'{a}/{s}':n for (a,s),n in minima.items()},original_mixture_validation_minima={a:min(int(v['value'].sum()) for k,v in arrays.items() if k[0]==a and k[1]=='validation') for a in (small,large)},input_sha256={str(Path('research/results/campaign-01/returns/r10-development')/p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()},scope='Development screen fails. Larger context diversity improves balanced tails but does not meet the fixed advancement rule; no confirmation or broad precision/representation claim follows.',cpu_audit_wall_seconds=time.monotonic()-t)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();o=audit(a.root);a.output.write_text(json.dumps(o,indent=2)+'\n');print(o)
