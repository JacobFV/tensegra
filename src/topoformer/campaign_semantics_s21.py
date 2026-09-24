"""S21 paired corpus intervention; unchanged S19 sequential actor and objective."""
import argparse,collections,hashlib,json,resource,time
from pathlib import Path
import torch
from torch import nn
from .campaign_semantics import write_gzip,digest
from .campaign_semantics_data import load_cache,target
from .campaign_semantics_continue import next_indices
from .campaign_semantics_s19 import field_losses,padded_records,learning_rate,evaluate,worst_case_timing
from .campaign_semantics_s19_actor import TypedRecordActor
from .campaign_semantics_s19_codec import encode_row
from .campaign_semantics_s21_freeze import verify,CELLS
from .semantic_scaling import tokens
from .thinking_language import ActorInput,state_hash

def prepare(c):
 """Only the explicitly frozen TRAIN/DEV inputs; no sealed confirmation path."""
 vocab=json.loads(Path(c['inputs']['vocabulary_audit']['path']).read_text())['value_vocabulary']
 panels=json.loads(Path(c['inputs']['panels']['path']).read_text());audit=json.loads(Path(c['inputs']['audit']['path']).read_text())
 dev=load_cache(c['inputs']['development']['path'])
 if len(dev)!=3584 or collections.Counter((r['arity'],r['facts']) for r in dev)!=dict.fromkeys(CELLS,512):raise ValueError('seven DEV512 cells required')
 chosen=[r for cell in CELLS for r in [r for r in dev if (r['arity'],r['facts'])==cell][:c['dev_per_cell']]]
 def one(row,evaluation=False):
  x=dict(row=row,public=ActorInput(row['text'],()),records=encode_row(row,vocab))
  if evaluation:x['gold']=target(row,vocab)
  return x
 development=[one(r,True) for r in chosen];arms={}
 for arm in c['arms']:
  rows=load_cache(c['inputs']['train_'+arm]['path'])
  if len(rows)!=4096 or len({r['semantic_sha256'] for r in rows})!=4096:raise ValueError('TRAIN4096 unique required')
  entries=panels[arm]
  if len(entries)!=128 or len({r['index'] for r in entries})!=128:raise ValueError('TRAIN128 panel required')
  train=[one(r) for r in rows];panel=[]
  for entry in entries:
   row=rows[entry['index']]
   if any(row[k]!=entry[k] for k in ('seed','semantic_sha256','alpha_sha256')):raise ValueError('panel identity mismatch')
   panel.append(one(row,True))
  expected={(3,3):64,(4,3):32,(4,4):32} if arm=='original' else {(3,3):32,(4,3):16,(4,4):16,(2,4):16,(5,3):24,(5,4):24}
  if collections.Counter((x['row']['arity'],x['row']['facts']) for x in panel)!=expected:raise ValueError('panel proportions changed')
  exposure=dict(tokens=sum(len(tokens(x['public'])) for x in train),nodes=sum(len(x['row']['nodes']) for x in train),edges=sum(len(x['row']['edges']) for x in train),records=sum(len(x['records']) for x in train))
  if arm=='broad' and any(exposure[k]!=audit['summaries']['train_broad'][k] for k in exposure):raise ValueError('audited broad exposure mismatch')
  if exposure!=c['expected_epoch_exposures'][arm]:raise ValueError('frozen data-derived exposure mismatch')
  arms[arm]=(train,panel)
 return vocab,arms,development

def train_arm(c,arm,vocab,train,panel,dev,out):
 out.mkdir(parents=True,exist_ok=False);torch.manual_seed(2101);torch.cuda.manual_seed_all(2101);tick=time.monotonic()
 model=TypedRecordActor(value_count=len(vocab),width=1024,heads=8,node_capacity=128,max_records=160,max_slot=32,autocast_dtype='bfloat16').to(c['device']);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01);initial=state_hash(model);setup_seconds=time.monotonic()-tick
 schedule=torch.Generator().manual_seed(15115);order=torch.randperm(4096,generator=schedule).tolist();position=0;visits=[0]*4096;stream=hashlib.sha256();curves=[];losses=[];training_seconds=0.;exposure=dict.fromkeys(('tokens','nodes','edges','records'),0);stress=None
 if c['job']=='profile':
  # Include longest public inputs across TRAIN and all profiled DEV, no gold lengths.
  public=[x['public'] for x in train+dev];indices=sorted(range(len(public)),key=lambda i:(-len(tokens(public[i])),i))[:32];stress=worst_case_timing(model,[public[i] for i in indices]);stress['public_pool_indices']=indices;tick=time.monotonic();path=out/'worst-case-profile.json.gz';write_gzip(path,stress);stress={k:v for k,v in stress.items() if k!='records'};stress.update(artifact=path.name,sha256=digest(path),export_seconds=time.monotonic()-tick)
 for added in range(c['updates']+1):
  if added in c['checkpoints']:
   tr=evaluate(model,panel,len(vocab),out,'train',added,32);dv=evaluate(model,dev,len(vocab),out,'development',added,32);tick=time.monotonic();path=out/f'model-u{added}.pt';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),schedule=schedule.get_state(),order=order,position=position,visits=visits,update=added,seed=2101,arm=arm),path)
   curves.append(dict(update=added,train=tr,development=dv,model_state_sha256=state_hash(model),checkpoint=path.name,checkpoint_sha256=digest(path),checkpoint_export_seconds=time.monotonic()-tick,presentations=sum(visits),exposure=dict(exposure)));print(json.dumps(dict(event='evaluation_complete',arm=arm,update=added)),flush=True)
  if added==c['updates']:break
  indices,order,position=next_indices(order,position,schedule,8);stream.update(json.dumps(indices).encode());batch=[train[i] for i in indices];records=padded_records([x['records'] for x in batch],model.features.weight.device)
  for i,x in zip(indices,batch):
   visits[i]+=1;exposure['tokens']+=len(tokens(x['public']));exposure['nodes']+=len(x['row']['nodes']);exposure['edges']+=len(x['row']['edges']);exposure['records']+=len(x['records'])
  for group in optimizer.param_groups:group['lr']=learning_rate(added+1)
  torch.cuda.synchronize();tick=time.monotonic();optimizer.zero_grad();outputs=model.teacher_forced([x['public'] for x in batch],records);loss,sums,counts,correct=field_losses(outputs,records)
  if not torch.isfinite(loss):raise FloatingPointError('nonfinite S21 loss')
  loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();torch.cuda.synchronize();training_seconds+=time.monotonic()-tick
  if (added+1)%128==0 or added+1==c['updates']:
   row=dict(arm=arm,update=added+1,learning_rate=learning_rate(added+1),loss=float(loss.detach()),field_loss_sums={k:float(v.detach()) for k,v in sums.items()},field_counts=counts,training_seconds=training_seconds);losses.append(row);print(json.dumps(dict(event='progress',**row)),flush=True)
  del outputs,loss,sums,correct,records
 if c['job']=='main':
  if len(visits)!=4096 or set(visits)!={8} or stream.hexdigest()!=c['expected_construction_sequence_sha256']:raise ValueError('fixed construction stream changed')
  if exposure!={k:v*8 for k,v in c['expected_epoch_exposures'][arm].items()}:raise ValueError('eight-epoch data-derived exposure changed')
 result=dict(arm=arm,seed=2101,parameters=sum(p.numel() for p in model.parameters()),initial_state_sha256=initial,final_state_sha256=state_hash(model),inherited_presentations=0,added_presentations=sum(visits),visits=visits,construction_sequence_sha256=stream.hexdigest(),exposure=exposure,training_seconds=training_seconds,setup_seconds=setup_seconds,worst_case_timing=stress,curves=curves,losses=losses)
 write_gzip(out/'manifest.json.gz',result);return result

def run(c):
 started=time.monotonic();torch.set_num_threads(2);verify(c,Path(__file__).parent,c['cap_seconds']);vocab,arms,dev=prepare(c);preflight_seconds=time.monotonic()-started;out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=False);torch.cuda.reset_peak_memory_stats();results=[]
 for arm in c['arms']:
  train,panel=arms[arm];results.append(train_arm(c,arm,vocab,train,panel,dev,out/arm))
 if results[0]['initial_state_sha256']!=results[1]['initial_state_sha256']:raise ValueError('paired scratch initialization differs')
 m=dict(config=c,preflight_seconds=preflight_seconds,results=results,process_seconds=time.monotonic()-started,peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);write_gzip(out/'manifest.json.gz',m);return m
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();run(json.loads(Path(a.config).read_text()))
