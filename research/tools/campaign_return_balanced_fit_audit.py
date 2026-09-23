"""Independent R06 population-only return repair audit."""
import argparse,gzip,json,hashlib,subprocess,time
from pathlib import Path
from collections import Counter
from campaign_return_diversity_audit import counts,groups,FIELDS
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--source-ref',required=True);p.add_argument('--output',required=True);a=p.parse_args();start=time.monotonic();root=Path(a.root);m=json.load(gzip.open(root/'manifest.json.gz','rt'));c=m['config'];raw=(root/'predictions.json.gz').read_bytes();assert hashlib.sha256(raw).hexdigest()==m['predictions_sha256'];rows=json.loads(gzip.decompress(raw));idx={}
for name,h in m['source'].items():assert hashlib.sha256(subprocess.check_output(['git','show',a.source_ref+':src/topoformer/'+name])).hexdigest()==h
for r in rows:
 k=(r['split'],r['distractors'],r['target_delay'],r['head']);assert k not in idx;idx[k]=r;assert counts(r)==r['counts'];assert groups(r)==r['groups']
expected={(s,d,t,h)for s in ['train_original','train_balanced',*c['data'],*c['grids']]for d in ([2]if s.startswith('train_')else [8]if s in c['grids']else c['eval_distractors'])for t in (c['grid_delays']if s in c['grids']else c['delays'])for h in ('unchanged','original','balanced')};assert set(idx)==expected
for (s,d,t,h),r in idx.items():
 base=idx[s,d,t,'unchanged'];assert r['targets']==base['targets'];assert r['event_sha256']==base['event_sha256'];assert all(r['predictions'][f]==base['predictions'][f]for f in FIELDS[1:])
 if s in c['grids']:assert len(Counter(zip(r['targets']['type'],r['targets']['value'])))==52 and set(Counter(zip(r['targets']['type'],r['targets']['value'])).values())=={64}
sets=[]
for f in m['fits']:
 n=c['fit_events'];assert f['events']==n;ids=f['training_event_hashes'];assert len(set(ids))==n;sets.append(set(ids));bits=[(b>>j)&1 for b in bytes.fromhex(f['visited_row_bits_little_endian'])for j in range(8)][:f['rows']];assert sum(bits)==f['unique_rows_sampled'];assert len({i%n for i,b in enumerate(bits)if b})==f['unique_events_sampled'];assert f['optimizer_presentations']==c['ce_updates']*c['ce_batch_size']
 gold=idx['train_'+f['arm'],2,0,'unchanged']['targets']
 for field,cs in f['support_by_field'].items():assert cs==[gold[field].count(j)for j in range(len(cs))]
 if f['arm']=='balanced':assert set(Counter(zip(gold['type'],gold['value'])).values())<={n//52,n//52+1}
assert len(sets[0]&sets[1])==m['fit_population_overlap']
summary={}
for h in ('original','balanced'):
 summary[h]={}
 for s in ('calibration','validation','calibration_grid','validation_grid'):
  selected=[r for r in rows if r['head']==h and r['split']==s]
  if s.endswith('grid'):
   vals=[]
   for r in selected:
    strata={k:[]for k in zip(r['targets']['type'],r['targets']['value'])}
    for typ,y,pred in zip(r['targets']['type'],r['targets']['value'],r['predictions']['value']):strata[typ,y].append(pred==y)
    vals.extend(sum(x)for x in strata.values())
   summary[h][s]=dict(minimum_stratum_correct=min(vals),support=64)
  else:summary[h][s]=dict(minimum_correct=min(r['counts']['value']['correct']for r in selected),support=selected[0]['counts']['value']['total'])
pass_cal=summary['balanced']['calibration_grid']['minimum_stratum_correct']>summary['original']['calibration_grid']['minimum_stratum_correct'] and summary['balanced']['calibration']['minimum_correct']/summary['balanced']['calibration']['support']>=.98
out=dict(rows_verified=len(rows),summary=summary,calibration_advancement=pass_cal,fit_overlap=m['fit_population_overlap'],cpu_audit_wall_seconds=time.monotonic()-start,scope='One frozen-backbone development population intervention; unchanged nonvalue outputs. No uniform-value gate or confirmation inferred.')
Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(out)
