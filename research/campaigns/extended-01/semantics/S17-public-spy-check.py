"""CPU synthetic forward spy: unchanged base algorithm, observed phases only."""
import gzip,json,re,tempfile
from pathlib import Path
import numpy as np
import torch
from topoformer.campaign_semantics import Dataset,calibrated_evaluation
from topoformer.campaign_semantics_data import load_cache
from topoformer.campaign_semantics_recalibrate import timed_evaluation
from topoformer.thinking_language import ActorInput,KINDS,ROLES

torch.set_num_threads(2)
class Spy:
 def __init__(self):self.calls=[];self.training=True
 def eval(self):self.training=False;return self
 def train(self):self.training=True;return self
 def __call__(self,public):
  assert isinstance(public,ActorInput) and not self.training
  self.calls.append(public.text);n=128;t=len(re.findall(r'\w+|[^\w\s]',public.text,re.UNICODE))
  presence=torch.full((n,),-1.);presence[:8]=1.
  return dict(presence=presence,kind=torch.zeros(n,len(KINDS)),value=torch.zeros(n,4),copy=torch.zeros(n,t),edges=(torch.arange(n*n*len(ROLES)).reshape(n,n,len(ROLES))%7-3).float(),slots=torch.zeros(n,n,33))
rows=load_cache('data/s15-shape-v2/train_mixed.jsonl.gz');vocab=['<unknown>','"parent"','"unify"','null'];cal=Dataset(rows[:3],vocab);dev=Dataset(rows[3:5],vocab)
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);a=root/'plain';b=root/'observed';a.mkdir();b.mkdir();plain=Spy();observed=Spy();original_save=np.savez_compressed
 calibrated_evaluation(plain,cal,dev,a,1,3,2);meta,timing=timed_evaluation(observed,cal,dev,b,1,3,2)
 assert np.savez_compressed is original_save and plain.calls==observed.calls and len(plain.calls)==5
 x=json.load(gzip.open(a/'evaluation-u1.json.gz','rt'));y=json.load(gzip.open(b/'evaluation-u1.json.gz','rt'))
 for k in ('thresholds','calibration','calibration_records','train_metrics','train_rows','rows'):assert x[k]==y[k],k
 with np.load(a/'calibration-u1.npz') as x,np.load(b/'calibration-u1.npz') as y:
  for k in x.files:assert np.array_equal(x[k],y[k]),k
 assert all(v>=0 for v in timing.values()) and abs(sum(v for k,v in timing.items() if k!='total_seconds')-timing['total_seconds'])<1e-9
 print(json.dumps(dict(public_only_forward_calls=5,synthetic_spy_not_model=True,raw_target_threshold_metrics_exact_replay=True,calibration_arrays_exact=True,timing_additive=True,save_hook_restored=True,cuda_visible=torch.cuda.is_available())))
