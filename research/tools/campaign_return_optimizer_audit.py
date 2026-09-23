"""Independent R08 endpoint selection and all-field archived counts."""
import argparse,gzip,json,hashlib,subprocess,time
from pathlib import Path
from campaign_return_diversity_audit import counts,groups,FIELDS
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();m=json.load(gzip.open(a.root/'manifest.json.gz'));blob=(a.root/'predictions.json.gz').read_bytes();assert hashlib.sha256(blob).hexdigest()==m['predictions_sha256'];rows=json.loads(gzip.decompress(blob));assert len(rows)==210
for f,h in m['source'].items():assert hashlib.sha256(subprocess.check_output(['git','show','5a01b0c:src/topoformer/'+f])).hexdigest()==h
scores={};summary={};trained=[f for f in m['fits']if f['trained_here']];assert len({f['visited_row_bits_little_endian']for f in trained})==1
for row in rows:assert counts(row)==row['counts']and groups(row)==row['groups'];assert row['target_delay']!=32
for fit in m['fits']:
 arm=fit['arm'];cells=[r for r in rows if r['head']==arm];grid=[];mixture=[]
 for row in cells:
  if row['split']=='calibration':mixture.append(row['counts']['value']['correct']/row['counts']['value']['total'])
  if row['split']=='calibration_grid':
   for typ,val in set(zip(row['targets']['type'],row['targets']['value'])):
    ids=[i for i,(tt,vv)in enumerate(zip(row['targets']['type'],row['targets']['value']))if(tt,vv)==(typ,val)];correct=sum(row['predictions']['value'][i]==val for i in ids);grid.append(correct/len(ids));record=next(c for c in fit['curve'][-1]['grid']if c['delay']==row['target_delay']and c['type']==typ and c['label']==val);assert record['correct']==correct and record['total']==len(ids)
 scores[arm]=(min(grid),min(mixture));summary[arm]=dict(calibration_grid_min=min(grid),calibration_mixture_min=min(mixture),validation_mixture_min=min(r['counts']['value']['correct']/r['counts']['value']['total']for r in cells if r['split']=='validation'))
selected=max(trained,key=lambda f:scores[f['arm']])['arm'];assert selected==m['selected'];out=dict(rows_verified=len(rows),selected=selected,summary=summary,paired_training_rows=True,cpu_audit_wall_seconds=time.monotonic()-t,scope='Fixed endpoint selection from calibration only; inspected validation populations remain development.');a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
