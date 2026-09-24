"""CPU S15 profile states and independent per-presentation query replay."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import torch
from topoformer.campaign_semantics_data import load_cache,target
from topoformer.semantic_curriculum import sampled_pairs
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();torch.set_num_threads(2);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:torch.load(p,map_location='cpu',weights_only=True);ms={arm:json.load(gzip.open(a.root/f'results/s15-{arm}-profile-v2/manifest.json.gz','rt')) for arm in ('control','mixed')};c=ms['control']['config'];assert sha(a.root/c['parent_checkpoint'])==c['parent_checkpoint_sha256'];parent=load(a.root/c['parent_checkpoint']);vocab=json.loads((a.root/c['data_audit']).read_text())['value_vocabulary'];rows={arm:load_cache(a.root/c['data_dir']/f'train_{arm}.jsonl.gz') for arm in ms};finals={};hashes={}
def eq(a,b):
 if torch.is_tensor(a):assert torch.equal(a,b)
 elif isinstance(a,dict):assert a.keys()==b.keys();[eq(v,b[k]) for k,v in a.items()]
 elif isinstance(a,(list,tuple)):assert len(a)==len(b);[eq(x,y) for x,y in zip(a,b)]
 else:assert a==b
for arm,m in ms.items():
 for point in m['curves']:
  path=a.root/f'results/s15-{arm}-profile-v2/model-u{point["update"]}.pt';assert sha(path)==point['checkpoint_sha256'];x=load(path);assert x['update']==point['update'] and x['added_update']==point['added_update'];assert {int(v['step']) for v in x['optimizer']['state'].values()}=={point['update']};assert x['presentation_number']==point['added_update']*8;assert sum(x['visits'])==point['added_update']*8;hashes[f'{arm}/{point["update"]}']=sha(path)
  if point['added_update']==0:
   eq(x['model'],parent['model']);eq(x['optimizer'],parent['optimizer']);schedule=torch.Generator().manual_seed(15115);order=torch.randperm(4096,generator=schedule).tolist();assert x['order']==order and x['position']==0;eq(x['schedule'],schedule.get_state())
 finals[arm]=x
 schedule=torch.Generator().manual_seed(15115);order=torch.randperm(4096,generator=schedule).tolist();visits=[0]*4096;sequence=hashlib.sha256();pairs=hashlib.sha256();common=hashlib.sha256();common_count=0;tokens=nodes=edges=0
 for step in range(20):
  ids=order[step*8:step*8+8];sequence.update(json.dumps(ids).encode())
  for offset,index in enumerate(ids):
   number=step*8+offset;row=rows[arm][index];gold=target(row,vocab);query=sampled_pairs(gold,torch.Generator().manual_seed(150150000+number),128);pairs.update(query.numpy().tobytes());visits[index]+=1;tokens+=row['tokens'];nodes+=len(row['nodes']);edges+=len(row['edges'])
   if rows['control'][index]['alpha_sha256']==rows['mixed'][index]['alpha_sha256']:common.update(number.to_bytes(8,'little'));common.update(query.numpy().tobytes());common_count+=1
 assert sequence.hexdigest()==m['construction_sequence_sha256'] and pairs.hexdigest()==m['pair_sequence_sha256'] and common.hexdigest()==m['common_pair_sequence_sha256'];assert common_count==m['common_presentations']==74;assert visits==m['visits']==x['visits'];assert (tokens,nodes,edges)==(m['optimizer_tokens'],m['optimizer_nodes'],m['optimizer_edges']);assert x['order']==order and x['position']==160;eq(x['schedule'],schedule.get_state())
for k in ('schedule','order','position','visits','presentation_number'):eq(finals['control'][k],finals['mixed'][k])
out=dict(checkpoint_sha256=hashes,initial_model_adamw_exact_parent=True,new_shared_sampler_seed15115_verified=True,all320_presentation_negative_queries_replayed=True,common_presentations_identical=74,complete_token_node_edge_exposure_replayed=True,final_steps=24596,added_presentations_per_arm=160,cpu_audit_wall_seconds=time.monotonic()-t,scope='CPU saved state and public target/query sampling only; no model construction or forward. Different graphs consume different pairs with presentation-isolated RNG, common queries remain exact.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
