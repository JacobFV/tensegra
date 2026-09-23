"""Reconstruct underlying public episodes and count unique construction hashes."""
import hashlib,json,argparse
from pathlib import Path
from topoformer.belief_contracts import contract_episodes
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();groups=[];sets={}
for name,bases,ns in [('main',[9100000,9200000],[8,16,32]),('idlocal',[9300000],[8])]:
 hashes=[]
 for base in bases:
  for seed in range(3):
   for n in ns:
    for i,e in enumerate(contract_episodes(512,base+seed,n,'clean')):
     # Opaque nonce values remain part of public construction identity; candidate order does not.
     payload={'keys':e['keys'],'records':sorted(e['records'])}
     h=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();hashes.append(h)
     groups.append(dict(group=name,seed=base+seed,candidates=n,index=i,construction_sha256=h))
 sets[name]=set(hashes)
a.out.write_text(json.dumps(dict(definition='exact public nonce-key/candidate-record construction, candidate-order invariant; not alpha-equivalence',main_unique=len(sets['main']),idlocal_unique=len(sets['idlocal']),cross_overlap=len(sets['main']&sets['idlocal']),rows=groups),indent=2))
