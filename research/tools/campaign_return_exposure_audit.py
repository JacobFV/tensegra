"""Independent frozen-feature exposure endpoint and selection audit."""
import argparse,gzip,json,time,hashlib,subprocess
from pathlib import Path
from campaign_returns_audit import counts,groups,FIELDS
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--reference',type=Path,required=True);p.add_argument('--source-ref',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic();m=json.load(gzip.open(a.root/'manifest.json.gz','rt')) if (a.root/'manifest.json.gz').exists() else json.loads((a.root/'manifest.json').read_text());c=m['config'];ref=json.load(gzip.open(a.reference/'manifest.json.gz','rt'));rows=json.load(gzip.open(a.root/'predictions.json.gz','rt'));old=json.load(gzip.open(a.reference/'predictions.json.gz','rt'))
assert hashlib.sha256((a.root/'predictions.json.gz').read_bytes()).hexdigest()==m['predictions_sha256'];assert hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()==m['config_sha256']
for name,want in m['source'].items():assert hashlib.sha256(subprocess.check_output(['git','show',f'{a.source_ref}:src/topoformer/{name}'])).hexdigest()==want
idx={};prior={(r['split'],r['distractors'],r['target_delay']):r for r in old if r['head']=='ce_16384'}
for r in rows:
 key=r['split'],r['distractors'],r['target_delay'];assert key+(r['head'],)not in idx;idx[key+(r['head'],)]=r
 assert counts(r)==r['counts']and groups(r)==r['groups'];baseline=prior[key];assert r['targets']==baseline['targets']and r['event_sha256']==baseline['event_sha256']
 assert all(r['predictions'][f]==baseline['predictions'][f]for f in FIELDS[1:])
 if r['head']=='ce_900':assert r['predictions']==baseline['predictions']
assert set(idx)=={key+(f'ce_{step}',)for key in prior for step in c['endpoints']}
for e in m['endpoints']:
 scores=[]
 for cell in e['cells']:
  r=idx['calibration',cell['distractors'],cell['delay'],f"ce_{e['step']}"];assert r['counts']['value']=={'correct':cell['correct'],'total':cell['total']};scores.append(cell['correct'])
 assert e['selection_score']==[min(scores),sum(scores)]
 bits=[(b>>i)&1 for b in bytes.fromhex(e['visited_row_bits_little_endian'])for i in range(8)][:m['fit_rows']]
 assert sum(bits)==e['unique_rows_sampled'];assert len({i%m['fit_unique_events']for i,b in enumerate(bits)if b})==e['unique_events_sampled'];assert e['optimizer_presentations']==e['step']*c['batch_size']
assert m['endpoints'][0]['visited_row_bits_little_endian']==ref['fits'][-1]['visited_row_bits_little_endian'];winner=max(m['endpoints'],key=lambda e:e['selection_score']);assert winner['step']==m['selected_step']==900
out=dict(rows_verified=len(rows),selected_step=m['selected_step'],endpoint_summary=[dict(step=e['step'],calibration_score=e['selection_score'],validation_minimum=min(r['counts']['value']['correct']for r in rows if r['head']==f"ce_{e['step']}"and r['split']=='validation'))for e in m['endpoints']],cpu_audit_wall_seconds=time.monotonic()-start,scope='Raw endpoint metrics, calibration selection,900 prediction replay and visitation bitsets; separate weight/cache replay verifies producer parameter report. Adaptive development, no confirmation gate.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
