"""Independent S18 paired endpoint gates and shared-event bootstrap from audited flags."""
import argparse,collections,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--review',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic();base=a.repo/'research/results/campaign-01/semantics';load=lambda p:json.load(gzip.open(p,'rt'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();summary=json.loads(a.analysis.read_text());cache={r['semantic_sha256']:r for r in map(json.loads,gzip.open(base/'s15-shape-cache-v2/development.jsonl.gz','rt'))};cells=('3x3','3x4','4x3','4x4');steps=(0,1024,2048,4096);policies=('raw','historical','matched');flags={};audits={}
for arm,label in [('context','context'),('workspace_control','control')]:
 path=a.review/f'S18-main-{label}-raw-audit.json';d=json.loads(path.read_text());audits[str(path)]=sha(path)
 for step in steps:
  z=d['results'][str(step)]
  for cell in cells:
   assert z['matched'][cell]['event_ids']==z['historical'][cell]['event_ids'];assert z['matched'][cell]['raw']==z['historical'][cell]['raw'];ids=z['matched'][cell]['event_ids'];expected=sorted(k for k,r in cache.items() if f"{r['arity']}x{r['facts']}"==cell);assert ids==expected and len(ids)==512
   flags[(arm,step,'raw',cell)]=z['matched'][cell]['raw'];flags[(arm,step,'matched',cell)]=z['matched'][cell]['calibrated'];flags[(arm,step,'historical',cell)]=z['historical'][cell]['calibrated']
for step in steps:
 old=load(base/f's15-mixed-main-v3/evaluation-u{24576+step}.json.gz');new=load(base/'s17-calibration-main-v1/mixed/evaluation-u28672.json.gz' if step==4096 else a.reference/f'u{24576+step}/evaluation-u{24576+step}.json.gz');O={r['semantic_sha256']:r for r in old['rows']};N={r['semantic_sha256']:r for r in new['rows']};assert O.keys()==N.keys()==cache.keys()
 for cell in cells:
  ids=sorted(k for k,r in cache.items() if f"{r['arity']}x{r['facts']}"==cell)
  for policy,rows,key in [('raw',O,'raw_metrics'),('historical',O,'calibrated_metrics'),('matched',N,'calibrated_metrics')]:flags[('original',step,policy,cell)]=[int(rows[i][key]['semantic_equivalence']) for i in ids]
counts={arm:{cell:sum(flags[(arm,4096,'matched',cell)]) for cell in cells} for arm in ('original','context','workspace_control')};c=counts['context'];z=[counts['original'],counts['workspace_control']];acq=c['3x3']>=52 and min(c['3x3']-v['3x3'] for v in z)>=26;ret=c['4x3']+c['4x4']>=103 and min(c['4x3']+c['4x4']-v['4x3']-v['4x4'] for v in z)>=-51;rec=c['3x4']>=52 and min(c['3x4']-v['3x4'] for v in z)>=26;decisions=dict(acquisition=acq,retention=ret,recombination=rec,development_criteria_met=acq and ret,recombination_criteria_met=acq and ret and rec,automatic_extension=False);assert decisions==summary['decisions'];rng=np.random.default_rng(18018);verified=0;intervals={}
for cell in cells:
 draws=rng.integers(0,512,size=(10000,512))
 for step in steps:
  for policy in policies:
   for comparator in ('original','workspace_control'):
    context=flags[('context',step,policy,cell)];other=flags[(comparator,step,policy,cell)];key=f'{cell}/{step}/{policy}/context-minus-{comparator}';row=summary['paired'][key];transition=dict(collections.Counter(f'{x}->{y}' for x,y in zip(other,context)));assert row['transitions_comparator_to_context']==transition and row['delta_complete']==sum(context)-sum(other);verified+=1
    if step==4096:
     delta=np.asarray(context)-np.asarray(other);ci=np.percentile(delta[draws].mean(axis=1)*100,[2.5,97.5]);assert np.array_equal(ci,summary['endpoint_delta_percentage_points_ci95'][key]);intervals[key]=ci.tolist()
for (arm,step,policy,cell),v in flags.items():assert summary['results'][arm][str(step)][policy][cell]['complete']==sum(v) and summary['results'][arm][str(step)][policy][cell]['examples']==512
out=dict(endpoint_matched_counts=counts,decisions=decisions,paired_transitions_verified=verified,endpoint_shared_event_CIs_verified=len(intervals),all144_curve_policy_cell_complete_counts_exact=True,endpoint_delta_percentage_points_ci95=intervals,input_sha256={str(a.analysis)[str(a.analysis).index('research/'):]:sha(a.analysis)},audited_flag_receipt_sha256=audits,cpu_audit_wall_seconds=time.monotonic()-tick,scope='Fixed4096 matched policy only; exact both-control threshold and retention gates.10000 shared event-index draws percell acrosspolicies/comparators, one training lineage conditional descriptive intervals, no seed-uncertainty/confirmation or automaticextension. Uses independently raw-reconstructed new-arm flags and previously audited original reference records.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ('endpoint_delta_percentage_points_ci95','audited_flag_receipt_sha256')})
