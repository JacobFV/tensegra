"""Independent visited arithmetic-signature audit, CPU data replay only."""
import argparse,collections,gzip,hashlib,json,time
from pathlib import Path
import torch
from topoformer.campaign_composition_study import get_data
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--replicate',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic();torch.set_num_threads(2);base=a.root/f'c04-confirmation-{a.replicate}';path=lambda arm:Path(str(base)+'-'+arm);load=lambda p:torch.load(p,map_location='cpu',weights_only=True);ms={arm:json.loads((path(arm)/'summary.json').read_text()) for arm in ('n1_static','n1_roles','n2_rekey')};cfg=ms['n1_static']['config'];pools={False:{},True:{}}
def signature(pub,with_query,reverse):
 n=len(pub['query']);b=torch.arange(n);selected=(pub['instruction_destinations']==pub['query_destination'][:,None]).all(-1);assert (selected.sum(-1)==1).all();j=selected.long().argmax(-1);op=pub['instruction_cues'][b,j].argmax(-1);refs=pub['instruction_arguments'][b,j].clone()
 if reverse:refs=torch.where((op!=3)[:,None,None],refs.flip(1),refs)
 links=(refs[:,:,None]==pub['keys'][:,None]).all(-1);assert (links.sum(-1)==1).all();ptr=links.long().argmax(-1);out=[]
 for i,o in enumerate(op.tolist()):
  pieces=[str(o)]
  for k in (0,1):
   if k==1 and o==3:pieces.append('absent');continue
   q=int(ptr[i,k]);typ=int(pub['operand_types'][i,q]);value=float(pub['operand_values'][i,q]);pieces.append(str(typ)+':'+(str(int(value)) if typ==0 else value.hex()))
  if with_query:pieces.extend([float(pub['query'][i,0]).hex(),str(int(pub['query'][i,1]))])
  out.append('|'.join(pieces))
 return out
for split,spec in cfg['data'].items():
 _,pub,_=get_data(spec)
 for withquery in (False,True):
  for view in ('clean','reversed'):pools[withquery][split+'/'+view]=signature(pub,withquery,view=='reversed')
 del pub
results=[]
for arm,m in ms.items():
 saved=load(path(arm)/'visitation.pt');visits=torch.zeros_like(saved['base']);swaps=torch.zeros_like(visits);g=torch.Generator().manual_seed(cfg['sample_seed']);vg=torch.Generator().manual_seed(cfg['role_seed']);h=hashlib.sha256();vh=hashlib.sha256()
 for _ in range(cfg['updates']):
  ix=torch.randint(len(visits),(cfg['batch_size'],),generator=g);h.update(ix.numpy().tobytes());visits+=torch.bincount(ix,minlength=len(visits));coin=torch.rand(cfg['batch_size'],generator=vg)<.5 if arm=='n1_roles' else torch.zeros(cfg['batch_size'],dtype=torch.bool);vh.update(coin.numpy().tobytes());swaps+=torch.bincount(ix[coin],minlength=len(visits))
 assert torch.equal(visits,saved['base']) and torch.equal(swaps,saved['swapped']);assert h.hexdigest()==m['sample_index_stream_sha256'] and vh.hexdigest()==m['presentation_swap_coin_stream_sha256'];assert int(visits.sum())==m['optimizer_presentations']==1024000
 archived=json.load(gzip.open(path(arm)/'numeric-overlap.json.gz'))
 for withquery,kind in [(False,'operation_operands'),(True,'operation_operands_query')]:
  audit=archived['audits'][kind];counts=collections.Counter()
  for view,weights in [('clean',visits-swaps),('reversed',swaps)]:
   for key,weight in zip(pools[withquery]['train/'+view],weights.tolist()):
    if weight:counts[key]+=weight
  assert dict(counts)==audit['actual_training_signature_counts'];assert len(counts)==audit['actual_training_distinct_signatures'] and sum(counts.values())==audit['actual_training_presentations']
  for name,keys in pools[withquery].items():
   freq=collections.Counter(keys);cell=audit['populations'][name];common=set(freq)&set(counts);events=sum(freq[k] for k in common);assert dict(freq)==cell['signature_counts'];assert cell['events']==len(keys) and cell['distinct_signatures']==len(freq);assert cell['training_overlap_distinct']==len(common) and cell['training_overlap_events']==events
   if name.startswith('validation/'):results.append(dict(arm=arm,signature=kind,population=name,events=len(keys),overlap_events=events,overlap_fraction=events/len(keys)))
out=dict(replicate=a.replicate,sample_and_view_stream_replayed=True,all_population_and_visited_signatures_reconstructed=True,validation_overlap=results,cpu_audit_wall_seconds=time.monotonic()-tick,scope='CPU fixed public-data generation, independent typed ordered operand/query signatures and visited multiplicities. No model or forward. Finite arithmetic overlap disclosed; inherited training overlap unmeasured; remaining lineages pending.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(dict(seconds=out['cpu_audit_wall_seconds'],validation_cells=len(results)))
