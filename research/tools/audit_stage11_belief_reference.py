"""Verify frozen belief reference only; no actor inference or training."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2];doc=json.loads((root/'research/stages/stage-11/stage11-belief-reference.json').read_text());checked=[]
def walk(x):
 if isinstance(x,dict):
  if 'path'in x and 'sha256'in x:
   assert hashlib.sha256((root/x['path']).read_bytes()).hexdigest()==x['sha256'];checked.append(x['path'])
  for v in x.values():walk(v)
 elif isinstance(x,list):
  for v in x:walk(v)
walk(doc)
assert doc['constructor']==dict(arm='ledger_only',width=1024,inner=2048)
assert not doc['stage11_activity']['new_training'] and not doc['stage11_activity']['new_model_inference']
assert not any(doc['authorization'].values())
prior=json.loads((root/'research/results/stage10/audits/belief-checkpoint-hashes.json').read_text())
checkpoints=[]
def checkpoint_walk(x):
 if isinstance(x,dict):
  if 'location'in x and 'sha256'in x:
   key=x['location'].split('/main/')[1];assert prior[key]==x['sha256'];checkpoints.append(key)
  for v in x.values():checkpoint_walk(v)
 elif isinstance(x,list):
  for v in x:checkpoint_walk(v)
checkpoint_walk(doc)
out=dict(local_file_hashes_verified=len(checked),checkpoint_hashes_linked_to_prior_independent_audit=len(checkpoints),new_inference=False,new_remote_hashing=False,passed=True)
(root/'research/results/stage11/audits/belief-reference.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
