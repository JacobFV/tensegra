import json,time,hashlib
from pathlib import Path
import torch
from topoformer import campaign_semantics_s18 as s
torch.set_num_threads(2)
calls=[]
def sentinel(*a,**k):calls.append(1);raise AssertionError('actor construction forbidden')
s.S18Actor=sentinel
c=json.loads(Path('configs/campaign-s18-profile-frozen-v1.json').read_text());tick=time.monotonic();b,vocab,train,matched,historical,dev=s.prepare(c)
assert len(train)==4096 and len(matched)==len(historical)==128 and len(dev)==64 and not calls
out=dict(actual_prepare_seconds=time.monotonic()-tick,actor_constructions=len(calls),forwards=0,train=len(train),matched=len(matched),historical=len(historical),dev=len(dev),config_sha256=hashlib.sha256(Path('configs/campaign-s18-profile-frozen-v1.json').read_bytes()).hexdigest())
Path('/tmp/s18-review-prepare-result.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
