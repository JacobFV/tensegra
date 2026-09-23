"""CPU-only cross-run event identity check for R01 exploratory/confirmation pools."""
import argparse,gzip,hashlib,io,json,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);started=time.monotonic();sets={};records=[]
for name in ('r01-profile','r01-development','r01-confirmation/10','r01-confirmation/11','r01-confirmation/12'):
 root=a.root/name;m=json.loads((root/'manifest.json').read_text());packed=(root/'features.pt.gz').read_bytes();assert hashlib.sha256(packed).hexdigest()==m['feature_cache']['sha256'];cache=torch.load(io.BytesIO(gzip.decompress(packed)),map_location='cpu',weights_only=True)
 for split,spec in m['config']['data'].items():
  ids=cache[split+'/2']['event_row_hashes'];assert len(ids)==len(set(ids))==spec['size'];sets[name+'/'+split]=set(ids)
  records.append(dict(population=name+'/'+split,events=len(ids),sorted_identity_sha256=hashlib.sha256(''.join(sorted(ids)).encode()).hexdigest()))
 del cache,packed
for name,ids in sets.items():
 for other,there in sets.items():
  if name<other:assert not(ids&there),(name,other)
out=dict(cpu_audit_wall_seconds=time.monotonic()-started,populations=records,distinct_events=sum(len(s)for s in sets.values()),all_populations_pairwise_disjoint=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(dict(cpu_seconds=out['cpu_audit_wall_seconds'],populations=len(sets),events=out['distinct_events']))
