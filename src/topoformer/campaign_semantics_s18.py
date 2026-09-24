"""S18 matched continuation/reused-reference evaluation. No new actor mechanism."""
import argparse,collections,gzip,json,resource,time
from pathlib import Path
import torch
from torch import nn
from .campaign_semantics import Dataset,digest,write_gzip,calibrated_evaluation
from . import campaign_semantics as evaluation_base
from .campaign_semantics_data import load_cache
from .campaign_semantics_continue import next_indices
from .campaign_semantics_lr import override_learning_rate
from .campaign_semantics_shape_train import presentation_seed
from .campaign_semantics_s18_actor import S18Actor
from .campaign_semantics_s18_freeze import verify
from .semantic_curriculum import sampled_pairs,curriculum_weights
from .semantic_text_acquisition import corrected_losses
from .thinking_language import state_hash
import hashlib


def selected_rows(rows,selection):
 indices=[item['index'] for item in selection]
 if len(indices)!=128 or indices!=sorted(set(indices)):raise ValueError('fixed128TRAIN indices required')
 chosen=[]
 for item in selection:
  row=rows[item['index']]
  if any(row[k]!=item[k] for k in ('seed','semantic_sha256','alpha_sha256')):raise ValueError('selection mismatch')
  chosen.append(row)
 if collections.Counter((r['arity'],r['facts']) for r in chosen)!={(3,3):64,(4,3):32,(4,4):32}:raise ValueError('calibration mixture changed')
 return chosen


def prepare(c):
 b=verify(c,Path(__file__).parent,c['proposed_cap_seconds'])
 vocab=json.loads(Path(b['data_audit']['path']).read_text())['value_vocabulary']
 rows=load_cache(b['caches']['train_mixed']['path']);dev=load_cache(b['caches']['development']['path'])
 if len(rows)!=4096 or len(dev)!=2048:raise ValueError('support changed')
 chosen=selected_rows(rows,json.loads(Path(b['selection']['path']).read_text())['mixed'])
 dev=[r for a in (3,4) for f in (3,4) for r in [x for x in dev if (x['arity'],x['facts'])==(a,f)][:c['dev_per_cell']]]
 return b,vocab,Dataset(rows,vocab),Dataset(chosen,vocab),Dataset(load_cache(b['historical_train']['path'])[:128],vocab),Dataset(dev,vocab)


def evaluate(model,cal,dev,folder,update):
 folder.mkdir(parents=True,exist_ok=False);before=state_hash(model);tick=time.monotonic();marks={}
 class ObservedDev:
  def __len__(self):return len(dev)
  def __getitem__(self,index):
   marks.setdefault('dev',time.monotonic());return dev[index]
 original_save=evaluation_base.np.savez_compressed
 def observed_save(*args,**kwargs):
  marks.setdefault('export',time.monotonic());return original_save(*args,**kwargs)
 evaluation_base.np.savez_compressed=observed_save
 try:result=calibrated_evaluation(model,cal,ObservedDev(),folder,update,128,len(dev))
 finally:evaluation_base.np.savez_compressed=original_save
 end=time.monotonic()
 if state_hash(model)!=before:raise ValueError('evaluation mutated model')
 result['whole_evaluation_seconds']=time.monotonic()-tick
 result['timing']=dict(calibration_trainpanel_seconds=marks['dev']-tick,
  dev_seconds=marks['export']-marks['dev'],export_seconds=end-marks['export'],post_state_hash_seconds=time.monotonic()-end)
 return result


def make_model(vocab,arm,c):
 torch.manual_seed(201)
 return S18Actor(arm=arm,control_microsteps=10 if arm=='workspace_control' else None,
  value_count=len(vocab),width=1024,capacity=128,workspace_rows=8,autocast_dtype='bfloat16').to(c['device'])


def reference(c,b,vocab,matched,dev,out):
 records=[]
 for entry in b['checkpoints'][:3]:
  tick=time.monotonic();checkpoint=torch.load(entry['checkpoint']['path'],map_location='cpu',weights_only=True)
  if checkpoint['added_update']!=entry['added_update'] or checkpoint['update']!=entry['update']:raise ValueError('frozen curve exposure mismatch')
  model=make_model(vocab,'original',c);model.load_state_dict(checkpoint['model']);del checkpoint
  before=state_hash(model);folder=out/f"u{entry['update']}"
  result=evaluate(model,matched,dev,folder,entry['update'])
  new=json.load(gzip.open(folder/result['artifact']));old=json.load(gzip.open(entry['historical_evaluation']['path']))
  if len(new['rows'])!=len(old['rows']) or any(a['raw']!=z['raw'] or a['target']!=z['target'] for a,z in zip(new['rows'],old['rows'])):raise ValueError('reused baseline raw/target replay mismatch')
  records.append(dict(added_update=entry['added_update'],checkpoint=entry['checkpoint'],model_state_sha256=before,evaluation=result,seconds=time.monotonic()-tick,raw_target_replay_exact=True))
  del model;print(json.dumps(dict(event='reference_curve',added_update=entry['added_update'],seconds=records[-1]['seconds'])),flush=True)
 return dict(no_optimizer_or_training=True,curves=records,endpoint_reused=b['matched_endpoint'],historical_training_seconds=484.141407571)


def optimizer_update(model,optimizer,batch,queries,update):
 """Exact inherited S15 update; public-only forward precedes supervised losses."""
 optimizer.zero_grad()
 outputs=model.forward_batch([p for p,_,_ in batch],pairs=queries)
 weights=curriculum_weights(update*8,1000,2000)
 parts=[corrected_losses(o,g,q) for o,(_,g,_),q in zip(outputs,batch,queries)]
 loss=torch.stack([sum(weights[k]*v for k,v in item.items()) for item in parts]).mean()
 if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
 loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
 return loss,parts


def train_arm(c,b,vocab,train,matched,historical,dev,out,arm):
 out.mkdir();tick=time.monotonic();parent=torch.load(b['parent']['path'],map_location='cpu',weights_only=True)
 if parent['update']!=24576 or sum(parent['visits'])!=196608:raise ValueError('parent exposure mismatch')
 model=make_model(vocab,arm,c);model.load_state_dict(parent['model'])
 if state_hash(model)!=b['original_initial_state_sha256']:raise ValueError('parent model mismatch')
 if any(isinstance(m,nn.Dropout) and m.p for m in model.modules()):raise ValueError('stochastic actor')
 optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5);optimizer.load_state_dict(parent['optimizer']);lr=override_learning_rate(optimizer,1e-5);del parent
 steps=sorted({float(s['step']) for s in optimizer.state.values()})
 if steps!=[24576.]:raise ValueError('AdamW step mismatch')
 setup_seconds=time.monotonic()-tick
 schedule=torch.Generator().manual_seed(15115);order=torch.randperm(4096,generator=schedule).tolist();position=0
 visits=[0]*4096;sequence=hashlib.sha256();pair_sequence=hashlib.sha256();tokens=nodes=edges=0;training_seconds=0.;curves=[];losses=[]
 for step in range(c['added_updates']+1):
  update=24576+step
  if step in c['checkpoints']:
   primary=evaluate(model,matched,dev,out/f'matched-u{update}',update)
   secondary=evaluate(model,historical,dev,out/f'historical-u{update}',update)
   # Same checkpoint must produce identical raw graphs regardless of fitted policy.
   a=json.load(gzip.open(out/f'matched-u{update}'/primary['artifact']));z=json.load(gzip.open(out/f'historical-u{update}'/secondary['artifact']))
   if any(x['raw']!=y['raw'] or x['target']!=y['target'] for x,y in zip(a['rows'],z['rows'])):raise ValueError('policy evaluation changed raw outputs')
   curve=dict(added_update=step,update=update,matched=primary,historical=secondary,added_presentations=sum(visits),model_state_sha256=state_hash(model))
   save_tick=time.monotonic();path=out/f'model-u{update}.pt'
   torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),schedule=schedule.get_state(),order=order,position=position,visits=visits,update=update,added_update=step,presentation_number=step*8),path)
   curve.update(checkpoint_sha256=digest(path),checkpoint_export_seconds=time.monotonic()-save_tick);curves.append(curve)
   print(json.dumps(dict(event='evaluation_complete',arm=arm,added_update=step,matched_seconds=primary['whole_evaluation_seconds'],historical_seconds=secondary['whole_evaluation_seconds'])),flush=True)
  if step==c['added_updates']:break
  indices,order,position=next_indices(order,position,schedule,8);batch=[train[i] for i in indices];queries=[]
  for offset,(index,(_,gold,row)) in enumerate(zip(indices,batch)):
   query=sampled_pairs(gold,torch.Generator().manual_seed(presentation_seed(step*8+offset)),128);queries.append(query);pair_sequence.update(query.numpy().tobytes())
   visits[index]+=1;tokens+=row['tokens'];nodes+=len(row['nodes']);edges+=len(row['edges'])
  sequence.update(json.dumps(indices).encode());torch.cuda.synchronize();tick=time.monotonic()
  loss,parts=optimizer_update(model,optimizer,batch,queries,update)
  torch.cuda.synchronize();training_seconds+=time.monotonic()-tick
  if (step+1)%128==0:
   losses.append(dict(added_update=step+1,training_seconds=training_seconds,loss=float(loss.detach())))
   print(json.dumps(dict(event='progress',arm=arm,added_update=step+1,training_seconds=training_seconds)),flush=True)
 streams=dict(construction_sequence_sha256=sequence.hexdigest(),pair_sequence_sha256=pair_sequence.hexdigest(),optimizer_tokens=tokens,optimizer_nodes=nodes,optimizer_edges=edges)
 if c['job']=='main' and (set(visits)!={8} or streams!=b['expected_streams']):raise ValueError('matched exposure/query stream differs')
 result=dict(arm=arm,setup_seconds=setup_seconds,training_seconds=training_seconds,curves=curves,losses=losses,visits=visits,learning_rate_override=lr,inherited_optimizer_steps=steps,final_state_sha256=state_hash(model),**streams)
 write_gzip(out/'manifest.json.gz',result);del model,optimizer
 return result


def run(c):
 start=time.monotonic();torch.set_num_threads(2);b,vocab,train,matched,historical,dev=prepare(c);preflight_seconds=time.monotonic()-start
 out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=False);torch.cuda.reset_peak_memory_stats()
 if c['job']=='reference':results=[reference(c,b,vocab,matched,dev,out)]
 else:results=[train_arm(c,b,vocab,train,matched,historical,dev,out/arm,arm) for arm in c['arms']]
 manifest=dict(config=c,results=results,preflight_seconds=preflight_seconds,process_seconds=time.monotonic()-start,peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 write_gzip(out/'manifest.json.gz',manifest)
 (out/'completion.json').write_text(json.dumps(dict(success=True,seconds=time.monotonic()-start))+'\n');return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();run(json.loads(Path(a.config).read_text()))
