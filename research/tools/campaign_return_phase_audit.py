"""Independent R09 counts and matched linear endpoint reconstruction."""
import argparse,gzip,json,hashlib,subprocess,time
from pathlib import Path
from campaign_return_diversity_audit import counts,groups,FIELDS
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('reference');p.add_argument('--output',required=True);a=p.parse_args();start=time.monotonic();root=Path(a.root);m=json.load(gzip.open(root/'manifest.json.gz','rt'));raw=(root/'predictions.json.gz').read_bytes();assert hashlib.sha256(raw).hexdigest()==m['predictions_sha256'];rows=json.loads(gzip.decompress(raw));old=json.load(gzip.open(Path(a.reference)/'predictions.json.gz','rt'));old={(r['split'],r['distractors'],r['target_delay']):r for r in old if r['head']=='balanced'};idx={};summary={}
for f,h in m['source'].items():assert hashlib.sha256(subprocess.check_output(['git','show','fd9f191:src/topoformer/'+f])).hexdigest()==h
for r in rows:
 assert counts(r)==r['counts']and groups(r)==r['groups'];k=(r['split'],r['distractors'],r['target_delay']);base=old[k];assert r['targets']==base['targets'];assert all(r['predictions'][f]==base['predictions'][f] for f in FIELDS[1:]);idx[(r['head'],*k)]=r
 if r['head']=='linear':assert r['predictions']==base['predictions']
assert len(idx)==84
for arm in ('linear','phase_linear'):
 summary[arm]={}
 for split in ('train_balanced','calibration','validation','calibration_grid','validation_grid'):
  cells=[r for r in rows if r['head']==arm and r['split']==split]
  s={'min_correct':min(r['counts']['value']['correct']for r in cells),'support':cells[0]['counts']['value']['total']}
  if split.endswith('grid'):
   strata=[]
   for r in cells:
    for t,v in set(zip(r['targets']['type'],r['targets']['value'])):
     ids=[i for i,(tt,vv)in enumerate(zip(r['targets']['type'],r['targets']['value']))if(tt,vv)==(t,v)];strata.append({'type':t,'value':(v-16)/2,'delay':r['target_delay'],'correct':sum(r['predictions']['value'][i]==v for i in ids),'total':len(ids)})
   s['worst_strata']=sorted(strata,key=lambda x:x['correct'])[:5]
  summary[arm][split]=s
assert m['fits'][0]['visited_row_bits_little_endian']==m['fits'][1]['visited_row_bits_little_endian'];assert m['fits'][0]['phase_presentations']==m['fits'][1]['phase_presentations'];assert sum(m['fits'][1]['phase_presentations'])==m['fits'][1]['optimizer_presentations']
result={'rows_verified':len(rows),'linear_predictions_equal_R06':True,'matched_visitation':True,'summary':summary,'cpu_audit_wall_seconds':time.monotonic()-start,'scope':'Fixed900-step development endpoints; no evidence that failure proves information absence.'};Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(result)
