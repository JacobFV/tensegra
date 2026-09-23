"""Independent nested-data return readout raw metric and exposure audit."""
import argparse,gzip,json,hashlib,subprocess,time
from pathlib import Path
from collections import Counter
from campaign_returns_audit import counts,groups,FIELDS

def audit(root,ref):
 t=time.monotonic();m=json.load(gzip.open(root/'manifest.json.gz','rt'));c=m['config'];rows=json.load(gzip.open(root/'predictions.json.gz','rt'))
 assert hashlib.sha256((root/'predictions.json.gz').read_bytes()).hexdigest()==m['predictions_sha256']
 assert hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()==m['config_sha256']
 for name,want in m['source'].items():assert hashlib.sha256(subprocess.check_output(['git','show',f'{ref}:src/topoformer/{name}'])).hexdigest()==want
 arms=['unchanged']+[f'ce_{n}'for n in c['pool_sizes']];idx={}
 for r in rows:
  k=r['split'],r['distractors'],r['target_delay'],r['head'];assert k not in idx;idx[k]=r
  assert counts(r)==r['counts']and groups(r)==r['groups']
  assert r['counts']['value']['total']==c['data'][r['split']]['size']
  if r['split']=='train'and r['head']!='unchanged':
   n=int(r['head'].split('_')[1]);ok=[a==b for a,b in zip(r['predictions']['value'],r['targets']['value'])]
   assert r['in_pool_value']==dict(correct=sum(ok[:n]),total=n)
   if n<len(ok):assert r['heldout_prefix_tail_value']==dict(correct=sum(ok[n:]),total=len(ok)-n)
 expected={(s,k,d,a)for s in c['data']for k in ([2]if s=='train'else c['eval_distractors'])for d in c['delays']for a in arms};assert set(idx)==expected
 for (s,k,d,a),r in idx.items():
  base=idx[s,2,0,'unchanged'];assert r['targets']==base['targets']and r['event_sha256']==base['event_sha256']
  assert all(r['predictions'][f]==idx[s,k,d,'unchanged']['predictions'][f]for f in FIELDS[1:])
 maxhash=m['fits'][-1]['training_event_hashes'];exposure=[]
 for f in m['fits']:
  n=f['events'];assert f['training_event_hashes']==maxhash[:n];assert len(set(f['training_event_hashes']))==n
  bits=[(b>>i)&1 for b in bytes.fromhex(f['visited_row_bits_little_endian'])for i in range(8)][:f['rows']]
  assert sum(bits)==f['unique_rows_sampled'];assert len({i%n for i,b in enumerate(bits)if b})==f['unique_events_sampled']
  assert f['rows']==n*len(c['delays']);assert f['optimizer_presentations']==c['ce_updates']*c['ce_batch_size']
  gold=idx['train',2,0,'unchanged']['targets']
  for field,cs in f['support_by_field'].items():assert cs==[gold[field][:n].count(j)for j in range(len(cs))]
  assert f['label_counts']==f['support_by_field']['value']and min(f['label_counts'])>0
  exposure.append(dict(events=n,rows=f['rows'],visited_rows=sum(bits),visited_events=f['unique_events_sampled'],validation_minimum=min(r['counts']['value']['correct']for r in rows if r['split']=='validation'and r['head']==f'ce_{n}')))
 return dict(rows_verified=len(rows),source_ref=ref,exposure=exposure,cpu_audit_wall_seconds=time.monotonic()-t,scope='Raw metrics, nested populations, visitation bitsets and source hashes. Frozen feature/weight replay remains separate; development only.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--source-ref',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();o=audit(a.root,a.source_ref);a.output.write_text(json.dumps(o,indent=2)+'\n');print(o)
