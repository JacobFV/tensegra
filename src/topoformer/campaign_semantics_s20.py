"""S20 fixed paired continuations and scratch record replication; final-only evaluation.

Training functions receive TRAIN only. Confirmation contents are loaded only after
an arm returns its fixed endpoint. Historical learners/update/evaluation stay intact.
"""
import argparse,collections,gzip,hashlib,json,resource,time
from pathlib import Path
import torch
from torch import nn
from .campaign_semantics import Dataset,write_gzip,digest
from .campaign_semantics_data import load_cache,target
from .campaign_semantics_continue import next_indices
from .campaign_semantics_shape_train import presentation_seed
from .campaign_semantics_lr import override_learning_rate
from .campaign_semantics_s18 import selected_rows,optimizer_update,evaluate as workspace_evaluate
from .campaign_semantics_s18_actor import S18Actor
from .campaign_semantics_s19 import field_losses,padded_records,learning_rate,evaluate as record_evaluate,worst_case_timing
from .campaign_semantics_s19_actor import TypedRecordActor
from .campaign_semantics_s19_codec import encode_row
from .campaign_semantics_s20_freeze import verify,STREAM,PAIRS
from .semantic_curriculum import sampled_pairs
from .semantic_scaling import tokens
from .thinking_language import ActorInput,state_hash


def prepare_training(c):
 """No development/confirmation load and no actor construction."""
 vocab=json.loads(Path(c['inputs']['vocabulary_audit']['path']).read_text())['value_vocabulary']
 rows=load_cache(c['inputs']['train']['path']);selection=json.loads(Path(c['inputs']['selection']['path']).read_text())['mixed']
 if len(rows)!=4096:raise ValueError('TRAIN population changed')
 panel=selected_rows(rows,selection)
 return vocab,rows,panel


def prepare_record_rows(rows,vocab,*,evaluation=False):
 result=[]
 for row in rows:
  x=dict(public=ActorInput(row['text'],()),records=encode_row(row,vocab),row=row)
  if evaluation:x['gold']=target(row,vocab)
  result.append(x)
 return result


def load_final_evaluation(c,completed_updates):
 if completed_updates!=c['updates']:raise ValueError('evaluation before fixed endpoint')
 label='development' if c['job']=='profile' else 'confirmation'
 rows=load_cache(c['inputs'][label]['path'])
 if len(rows)!=2048 or collections.Counter((r['arity'],r['facts']) for r in rows)!={(3,3):512,(3,4):512,(4,3):512,(4,4):512}:raise ValueError('evaluation population changed')
 if len({r['semantic_sha256'] for r in rows})!=2048:raise ValueError('duplicate evaluation identities')
 selected=[r for a in (3,4) for f in (3,4) for r in [x for x in rows if (x['arity'],x['facts'])==(a,f)][:c['evaluation_per_cell']]]
 return label,selected


def new_stream():
 generator=torch.Generator().manual_seed(15115)
 return generator,torch.randperm(4096,generator=generator).tolist(),0


def initialize(c,seed,arm,vocab):
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 if arm=='record':
  model=TypedRecordActor(value_count=len(vocab),width=1024,heads=8,node_capacity=128,max_records=160,max_slot=32,autocast_dtype='bfloat16').to(c['device'])
  optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
  history=dict(inherited_presentations=0,inherited_optimizer_steps=[],parent_checkpoint_sha256=None,initial_optimizer_states=len(optimizer.state))
 else:
  binding=c['parents'][str(seed)];parent=torch.load(binding['checkpoint']['path'],map_location='cpu',weights_only=True)
  if parent['update']!=24576 or len(parent['visits'])!=8192 or set(parent['visits'])!={24}:raise ValueError('parent exposure mismatch')
  model=S18Actor(arm=arm,value_count=len(vocab),width=1024,capacity=128,workspace_rows=8,autocast_dtype='bfloat16').to(c['device']);model.load_state_dict(parent['model'])
  if state_hash(model)!=binding['model_state_sha256']:raise ValueError('parent state mismatch')
  optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5);optimizer.load_state_dict(parent['optimizer']);override_learning_rate(optimizer,1e-5)
  steps=sorted({float(s['step']) for s in optimizer.state.values()})
  if steps!=[24576.] or len(optimizer.state)!=98:raise ValueError('parent AdamW states mismatch')
  history=dict(inherited_presentations=196608,inherited_optimizer_steps=steps,parent_checkpoint_sha256=binding['checkpoint']['sha256'],initial_optimizer_states=len(optimizer.state));del parent
 if any(isinstance(m,nn.Dropout) and m.p for m in model.modules()):raise ValueError('unexpected stochastic layer')
 return model,optimizer,history


def record_update(model,optimizer,batch,update):
 """Unchanged S19 TF update and true4096 schedule, even during20-update profile."""
 records=padded_records([x['records'] for x in batch],model.features.weight.device)
 for group in optimizer.param_groups:group['lr']=learning_rate(update)
 optimizer.zero_grad();outputs=model.teacher_forced([x['public'] for x in batch],records);loss,sums,counts,correct=field_losses(outputs,records)
 if not torch.isfinite(loss):raise FloatingPointError('nonfinite record loss')
 loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
 return loss


def train_endpoint(c,seed,arm,vocab,rows,out):
 """Only TRAIN rows enter. No evaluation callback, corpus, or target is accepted."""
 out.mkdir(parents=True,exist_ok=False);tick=time.monotonic();model,optimizer,history=initialize(c,seed,arm,vocab);initial=state_hash(model);setup_seconds=time.monotonic()-tick
 prepared=prepare_record_rows(rows,vocab) if arm=='record' else Dataset(rows,vocab)
 schedule,order,position=new_stream();visits=[0]*4096;sequence=hashlib.sha256();pairs_hash=hashlib.sha256();training_seconds=0.;losses=[];token_count=node_count=edge_count=record_count=0;stress=None
 if c['job']=='profile' and arm=='record':
  ids=sorted(range(len(prepared)),key=lambda i:(-len(tokens(prepared[i]['public'])),i))[:32]
  stress=worst_case_timing(model,[prepared[i]['public'] for i in ids]);stress['train_indices']=ids;t=time.monotonic();path=out/'worst-case-profile.json.gz';write_gzip(path,stress);stress={k:v for k,v in stress.items() if k!='records'};stress.update(artifact=path.name,sha256=digest(path),export_seconds=time.monotonic()-t)
 for added in range(c['updates']):
  indices,order,position=next_indices(order,position,schedule,8);batch=[prepared[i] for i in indices];queries=[];sequence.update(json.dumps(indices).encode())
  for offset,index in enumerate(indices):
   row=rows[index];visits[index]+=1;token_count+=row['tokens'];node_count+=len(row['nodes']);edge_count+=len(row['edges'])
   if arm=='record':record_count+=len(batch[offset]['records'])
   else:
    q=sampled_pairs(batch[offset][1],torch.Generator().manual_seed(presentation_seed(added*8+offset)),128);queries.append(q);pairs_hash.update(q.numpy().tobytes())
  torch.cuda.synchronize();t=time.monotonic()
  loss=record_update(model,optimizer,batch,added+1) if arm=='record' else optimizer_update(model,optimizer,batch,queries,24576+added)[0]
  torch.cuda.synchronize();training_seconds+=time.monotonic()-t
  if (added+1)%128==0 or added+1==c['updates']:
   row=dict(seed=seed,arm=arm,added_update=added+1,loss=float(loss.detach()),learning_rate=optimizer.param_groups[0]['lr'],training_seconds=training_seconds);losses.append(row);print(json.dumps(dict(event='progress',**row)),flush=True)
  del loss
 streams=dict(construction_sequence_sha256=sequence.hexdigest(),pair_sequence_sha256=pairs_hash.hexdigest() if arm!='record' else None,optimizer_tokens=token_count,optimizer_nodes=node_count,optimizer_edges=edge_count,optimizer_records=record_count if arm=='record' else None)
 if c['job']=='main':
  if set(visits)!={8} or streams['construction_sequence_sha256']!=STREAM or (token_count,node_count,edge_count)!=(1720320,988008,2208616):raise ValueError('fixed matched exposure changed')
  if (arm=='record' and record_count!=3229392) or (arm!='record' and pairs_hash.hexdigest()!=PAIRS):raise ValueError('record/query stream changed')
 update=c['updates']+(0 if arm=='record' else 24576);t=time.monotonic();checkpoint=out/f'model-u{update}.pt'
 torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),schedule=schedule.get_state(),order=order,position=position,visits=visits,update=update,added_update=c['updates'],presentation_number=c['updates']*8,seed=seed,arm=arm),checkpoint)
 result=dict(seed=seed,arm=arm,update=update,added_update=c['updates'],parameters=sum(p.numel() for p in model.parameters()),initial_state_sha256=initial,final_state_sha256=state_hash(model),setup_seconds=setup_seconds,training_seconds=training_seconds,visits=visits,added_presentations=sum(visits),checkpoint=checkpoint.name,checkpoint_sha256=digest(checkpoint),checkpoint_export_seconds=time.monotonic()-t,worst_case_timing=stress,losses=losses,**history,**streams)
 del optimizer
 return model,result


def evaluate_endpoint(c,model,result,vocab,panel,rows,out,label):
 tick=time.monotonic();before=state_hash(model)
 if result['arm']=='record':
  training=record_evaluate(model,prepare_record_rows(panel,vocab,evaluation=True),len(vocab),out,'train',result['update'],32)
  evaluation=record_evaluate(model,prepare_record_rows(rows,vocab,evaluation=True),len(vocab),out,label,result['update'],32)
  result.update(train=training,evaluation=evaluation,evaluation_population=label)
 else:
  folder=out/f"matched-u{result['update']}";evaluation=workspace_evaluate(model,Dataset(panel,vocab),Dataset(rows,vocab),folder,result['update'])
  result.update(matched=evaluation,matched_directory=folder.name,evaluation_population=label)
 if state_hash(model)!=before or before!=result['final_state_sha256']:raise ValueError('endpoint evaluation changed state')
 result['evaluation_seconds']=time.monotonic()-tick;write_gzip(out/'manifest.json.gz',result)
 return result


def run(c):
 start=time.monotonic();torch.set_num_threads(2);receipt=verify(c,Path(__file__).parent,c['cap_seconds']);vocab,rows,panel=prepare_training(c);preflight_seconds=time.monotonic()-start
 out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=False);torch.cuda.reset_peak_memory_stats();results=[]
 for seed in c['seeds']:
  for arm in c['arms']:
   folder=out/f'seed-{seed}'/arm;model,result=train_endpoint(c,seed,arm,vocab,rows,folder)
   # This is the sole evaluation-cache parser call, after fixed endpoint training.
   label,evaluation_rows=load_final_evaluation(c,result['added_update'])
   result=evaluate_endpoint(c,model,result,vocab,panel,evaluation_rows,folder,label);results.append(result);del model,evaluation_rows
   print(json.dumps(dict(event='endpoint_complete',seed=seed,arm=arm,added_update=result['added_update'],training_seconds=result['training_seconds'],evaluation_seconds=result['evaluation_seconds'])),flush=True)
 manifest=dict(config=c,preflight=receipt,preflight_seconds=preflight_seconds,results=results,process_seconds=time.monotonic()-start,peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 write_gzip(out/'manifest.json.gz',manifest);return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();run(json.loads(Path(a.config).read_text()))
