import json,hashlib,time,argparse
from pathlib import Path
import torch
from topoformer.campaign_attention_selector import generate,targets,oracle_successors
parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();torch.set_num_threads(2);t=time.monotonic();root=args.root;cfg=json.loads((root/'config.json').read_text());m=json.loads((root/'manifest.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(root/'checkpoint.pt')==m['checkpoint_sha256'] and sha(root/'training-state.pt')==m['training_state_sha256'];x=torch.load(root/'training-state.pt',map_location='cpu',weights_only=False);model=torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=True);assert x['step']==cfg['steps'] and {int(v['step']) for v in x['optimizer']['state'].values()}=={cfg['steps']};assert all(torch.equal(v,x['model'][k]) for k,v in model.items());tensor=hashlib.sha256(b''.join(v.numpy().tobytes() for v in model.values())).hexdigest();assert tensor==m['final_tensor_sha256'];assert x['cpu_rng'].numel() and len(x['cuda_rng'])==1
for ci,c in enumerate(cfg['conditions']):
 h={k:hashlib.sha256() for k in ['keys','attributes','instructions','adjacency']}
 for offset in range(0,cfg['examples'],cfg['batch']):
  b=generate(min(cfg['batch'],cfg['examples']-offset),c['nodes'],c['depth'],groups=c['groups'],seed=cfg['data_seed']+ci*100000+offset,device='cpu',balanced=True)
  for k,v in h.items():v.update(getattr(b,k).numpy().tobytes())
 assert all(v.hexdigest()==m['confirmation']['public_sha256'][f'c{ci}_input_'+k] for k,v in h.items())
out=dict(checkpoint_sha256=m['checkpoint_sha256'],training_state_sha256=m['training_state_sha256'],model_state_matches_checkpoint=True,optimizer_steps=cfg['steps'],presentations=cfg['steps']*16,cpu_cuda_rng_archived=True,full_public_input_hashes_independently_regenerated=True,record_order_scope='CUDA-generated order receipt/source verified; no GPU replay attempted',cpu_audit_wall_seconds=time.monotonic()-t);args.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
