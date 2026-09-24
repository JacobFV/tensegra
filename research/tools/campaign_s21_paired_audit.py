"""Independent fixed S21 development gates and descriptive shared-event intervals."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
for n in ('raw','analysis','output'):p.add_argument(n,type=Path)
a=p.parse_args();start=time.monotonic();raw=json.loads(a.raw.read_text());given=json.loads(a.analysis.read_text())
assert raw['status']=='pass' and raw['graphs']==29696 and given['manifest_sha256']==raw['manifest_sha256']
flags=raw['flags'];cells=('2x4','3x3','3x4','4x3','4x4','5x3','5x4');steps=('0','1024','2048','4096');assert set(flags)=={'original','broad'}
for report in raw['reports']:
 assert report['summary']==given['curves'][report['arm']][str(report['update'])][report['population']]
for arm in flags:
 assert set(flags[arm])==set(steps)
 for step in steps:
  assert set(flags[arm][step])=={'train','development'} and set(flags[arm][step]['development'])==set(cells)
identities={c:sorted(flags['original']['4096']['development'][c]) for c in cells};assert all(len(x)==512 for x in identities.values()) and len(set().union(*(set(x) for x in identities.values())))==3584
for arm in flags:
 for step in steps:
  for cell in cells:
   v=flags[arm][step]['development'][cell];assert sorted(v)==identities[cell] and all(type(x)is bool for x in v.values())
rng=np.random.default_rng(21021);intervals={};paired={}
for cell in cells:
 draws=rng.integers(0,512,size=(10000,512));weights=np.bincount((draws+512*np.arange(10000)[:,None]).ravel(),minlength=5120000).reshape(10000,512).astype(float)
 for step in steps:
  old=np.array([flags['original'][step]['development'][cell][i] for i in identities[cell]],dtype=float);new=np.array([flags['broad'][step]['development'][cell][i] for i in identities[cell]],dtype=float)
  transitions={f'{i}->{j}':int(((old==i)&(new==j)).sum()) for i in (0,1) for j in (0,1)};key=cell+'/'+step;delta=int(new.sum()-old.sum());g=given['paired'][key]
  assert g['transitions_original_to_broad']==transitions and g['delta_complete']==delta
  comp={k:given['curves']['broad'][step]['development']['cells'][cell]['exact_components'][k]-given['curves']['original'][step]['development']['cells'][cell]['exact_components'][k] for k in ('presence','kind','value','copy','edges','slots')};assert comp==g['component_delta']
  paired[key]=dict(delta_complete=delta,transitions_original_to_broad=transitions,component_delta=comp)
  if step=='4096':
   intervals[cell]=(np.quantile(weights@(new-old)/512,[.025,.975])*100).tolist();assert np.allclose(intervals[cell],given['endpoint_delta_pp_conditional_ci95'][cell],rtol=0,atol=1e-10)
counts={arm:{cell:sum(flags[arm]['4096']['development'][cell].values()) for cell in cells} for arm in flags};assert counts==given['endpoint_counts'];old=counts['original'];new=counts['broad'];loss=sum(old[c]-new[c] for c in ('3x3','4x3','4x4'));held=new['3x4']>=52 and new['3x4']-old['3x4']>=26;retention=loss<=76;acquisition=all(new[c]>=256 for c in ('2x4','5x3','5x4'))
decisions=dict(heldout_gain=held,old_known_loss_complete=loss,old_known_retention=retention,new_motif_acquisition=acquisition,advance=held and retention and acquisition,automatic_extension=False);assert decisions==given['decisions']
out=dict(status='pass',seconds=time.monotonic()-start,endpoint_counts=counts,decisions=decisions,paired=paired,endpoint_delta_pp_conditional_ci95=intervals,analysis_sha256=hashlib.sha256(a.analysis.read_bytes()).hexdigest(),scope='All independently scored curves/components/metrics/failure IDs,28pairedcelltables and7descriptive event CIs. One fixed development pair; not training-seed uncertainty or sealed confirmation.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ('paired',)})
