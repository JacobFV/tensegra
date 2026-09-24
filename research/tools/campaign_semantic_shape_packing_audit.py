"""CPU actual target/FP32/BF16 packing against independent expected bytes; no actor."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import torch
from topoformer.campaign_semantics_data import target
from topoformer.semantic_curriculum import encode_text
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('expected',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();torch.set_num_threads(2);base=a.repo/'research/results/campaign-01/semantics';vocab=json.loads((base/'s14-arity3-cache/audit.json').read_text())['config']['value_vocabulary'];expected=json.loads(a.expected.read_text());seen=set()
def digest(arrays):
 h=hashlib.sha256()
 for name,tensor in sorted(arrays.items()):v=tensor.numpy();h.update(name.encode());h.update(str((str(v.dtype),v.shape)).encode());h.update(v.tobytes())
 return h.hexdigest()
for path in sorted((base/'s15-shape-cache-v2').glob('*.jsonl.gz')):
 for line in gzip.open(path,'rt'):
  r=json.loads(line);key=str(r['seed'])
  if key in seen:continue
  seen.add(key);want=expected[key];gold=target(r,vocab);assert digest(gold)==want['target_sha256'];features,length=encode_text(r['text']);assert length==r['tokens'] and features.shape==(1,length,68);assert hashlib.sha256(features.numpy().tobytes()).hexdigest()==want['fp32_sha256'];assert hashlib.sha256(features.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()).hexdigest()==want['bf16_sha256'];assert int(gold['copy'].max())<length
assert seen==set(expected);out=dict(examples=len(seen),actual_target_bytes_equal_independent_labels=True,actual_fp32_and_bf16_feature_bytes_equal=True,copy_positions_within_full_public_inventory=True,expected_sha256=hashlib.sha256(a.expected.read_bytes()).hexdigest(),cpu_audit_wall_seconds=time.monotonic()-t,scope='Actual CPU data target and public encoder only; no model construction, forward, loss or confirmation inference.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
