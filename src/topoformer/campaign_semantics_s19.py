"""S19 scratch public-text typed-record acquisition; no semantic parser at inference."""
import argparse,collections,gzip,hashlib,json,math,resource,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .campaign_semantics import write_gzip,digest
from .campaign_semantics_data import load_cache,target
from .campaign_semantics_continue import next_indices
from .campaign_semantics_s19_actor import TypedRecordActor
from .campaign_semantics_s19_codec import BOS,NODE,EDGE,EOS,PAD,KINDS,ROLES,CodecError,encode_row,canonicalize_public_copies,canonical_edge_order,records_to_targets
from .campaign_semantics_s19_freeze import verify
from .semantic_curriculum import pack_graph
from .semantic_scaling import tokens,metrics
from .thinking_language import ActorInput,state_hash
FIELDS=('type','kind','value','copy','source','target','role','slot')


def learning_rate(update):
 if not 1<=update<=4096:raise ValueError('update outside frozen schedule')
 if update<=128:return 3e-4*update/128
 return 3e-5+.5*(3e-4-3e-5)*(1+math.cos(math.pi*(update-128)/(4096-128)))


def field_losses(logits,records):
 tag=records[:,:,0];node=tag.eq(NODE);edge=tag.eq(EDGE);lexical=records[:,:,1].eq(KINDS.index('ident'))|records[:,:,1].eq(KINDS.index('entity'))
 supports={'type':tag.ne(PAD),'kind':node,'value':node&~lexical,'copy':node&lexical,'source':edge,'target':edge,'role':edge,'slot':edge}
 labels={'type':tag-1,'kind':records[:,:,1],'value':records[:,:,2],'copy':records[:,:,3],'source':records[:,:,1],'target':records[:,:,2],'role':records[:,:,3],'slot':records[:,:,4]+1}
 sums={};counts={};correct={};means=[]
 for name in FIELDS:
  active=supports[name];count=int(active.sum());counts[name]=count
  if count:
   selected=logits[name][active].float();gold=labels[name][active];sums[name]=F.cross_entropy(selected,gold,reduction='sum');correct[name]=(selected.argmax(-1)==gold).sum();means.append(sums[name]/count)
  else:
   # Public inputs are nonempty; copy position0 is always finite.
   sums[name]=logits[name][...,0].float().sum()*0;correct[name]=torch.zeros((),dtype=torch.long,device=records.device);means.append(sums[name])
 return torch.stack(means).sum()/8,sums,counts,correct


def padded_records(rows,device):
 result=torch.full((len(rows),max(len(r) for r in rows),5),-1,dtype=torch.long,device=device);result[:,:,0]=PAD
 for i,r in enumerate(rows):result[i,:len(r)]=torch.tensor(r,dtype=torch.long,device=device)
 return result


def prepared_data(c):
 verify(c,Path(__file__).parent,c['cap_seconds'])
 b=c['inputs'];vocab=json.loads(Path(b['vocabulary_audit']['path']).read_text())['value_vocabulary'];train=load_cache(b['train']['path']);dev=load_cache(b['development']['path']);selection=json.loads(Path(b['selection']['path']).read_text())['mixed']
 if len(train)!=4096 or len(dev)!=2048 or len(selection)!=128:raise ValueError('fixed population size changed')
 indices=[s['index'] for s in selection]
 if indices!=sorted(set(indices)):raise ValueError('TRAIN panel indices changed')
 for s in selection:
  if any(train[s['index']][k]!=s[k] for k in ('seed','semantic_sha256','alpha_sha256')):raise ValueError('TRAIN panel identity changed')
 if collections.Counter((train[i]['arity'],train[i]['facts']) for i in indices)!={(3,3):64,(4,3):32,(4,4):32}:raise ValueError('TRAIN panel mixture changed')
 chosen=[r for a in (3,4) for f in (3,4) for r in [x for x in dev if (x['arity'],x['facts'])==(a,f)][:c['dev_per_cell']]]
 def prepare(row):return dict(public=ActorInput(row['text'],()),records=encode_row(row,vocab),row=row)
 # Training needs serialized labels; dense target tensors are evaluation-only.
 prepared=[prepare(r) for r in train];development=[prepare(r) for r in chosen];panel=[prepared[i] for i in indices]
 for example in panel+development:example['gold']=target(example['row'],vocab)
 return vocab,prepared,panel,development


def component_exact(pred,gold):
 return dict(presence=bool(torch.equal(pred['presence'],gold['presence'])),kind=bool((pred['kind'][gold['presence']]==gold['kind'][gold['presence']]).all()),value=bool((pred['value'][gold['value']>=0]==gold['value'][gold['value']>=0]).all()),copy=bool((pred['copy'][gold['copy']>=0]==gold['copy'][gold['copy']>=0]).all()),edges=bool(torch.equal(pred['edges'],gold['edges'])),slots=bool((pred['slots'][gold['edges'].any(-1)]==gold['slots'][gold['edges'].any(-1)]).all()))


def decode_evaluation(public,generated,gold,vocab_size):
 """Generated public-only records; malformed results never receive partial credit."""
 raw=generated['records'];reason=generated['status'];canonical=None;pred=None
 try:
  if reason!='eos':raise CodecError(reason)
  public_tokens=tokens(public);canonical=canonicalize_public_copies(raw,public_tokens)
  pred=records_to_targets(canonical,token_count=len(public_tokens),vocab_size=vocab_size,capacity=128,max_records=160)
  m=metrics(pred,gold);components=component_exact(pred,gold);exact=all(components.values())
  if exact!=bool(m['semantic_equivalence']):raise ValueError('codec/historical exact mismatch')
  return dict(valid=True,reason=None,complete=exact,canonical_edge_order=canonical_edge_order(raw),records=raw,canonical_records=canonical,prediction=pack_graph(pred),target=pack_graph(gold),metrics=m,exact_components=components)
 except CodecError as error:
  return dict(valid=False,reason=str(error),complete=False,records=raw,canonical_records=canonical,prediction=None,target=pack_graph(gold),metrics=None,exact_components={k:False for k in ('presence','kind','value','copy','edges','slots')})


def evaluate(model,examples,vocab_size,out,label,update,batch_size=32):
 before=state_hash(model);was_training=model.training;model.eval();tick=time.monotonic();tf_sums={k:0. for k in FIELDS};tf_counts={k:0 for k in FIELDS};tf_correct={k:0 for k in FIELDS};tf_seconds=free_seconds=0.;rows=[];cell_counts=collections.Counter();cell_exact=collections.Counter();invalid=collections.Counter()
 with torch.no_grad():
  for start in range(0,len(examples),batch_size):
   batch=examples[start:start+batch_size];publics=[x['public'] for x in batch];records=padded_records([x['records'] for x in batch],model.features.weight.device)
   torch.cuda.synchronize();t=time.monotonic();outputs=model.teacher_forced(publics,records);_,sums,counts,correct=field_losses(outputs,records);torch.cuda.synchronize();tf_seconds+=time.monotonic()-t
   for key in FIELDS:tf_sums[key]+=float(sums[key]);tf_counts[key]+=counts[key];tf_correct[key]+=int(correct[key])
   del outputs,records
   torch.cuda.synchronize();t=time.monotonic();generated=model.greedy(publics);torch.cuda.synchronize();free_seconds+=time.monotonic()-t
   if len(generated)!=len(batch):raise ValueError('generated batch count mismatch')
   for example,g in zip(batch,generated):
    r=example['row'];item=decode_evaluation(example['public'],g,example['gold'],vocab_size);cell=f"{r['arity']}x{r['facts']}";item.update(seed=r['seed'],semantic_sha256=r['semantic_sha256'],graph_sha256=r['graph_sha256'],cell=cell);rows.append(item);cell_counts[cell]+=1;cell_exact[cell]+=int(item['complete'])
    if not item['valid']:invalid[item['reason']]+=1
 model.train(was_training)
 if state_hash(model)!=before:raise ValueError('evaluation changed model')
 tf={k:dict(loss_sum=tf_sums[k],count=tf_counts[k],correct=tf_correct[k],mean_loss=tf_sums[k]/max(1,tf_counts[k]),accuracy=tf_correct[k]/max(1,tf_counts[k])) for k in FIELDS}
 data=dict(update=update,population=label,teacher_forced=tf,teacher_forced_loss=sum(v['mean_loss'] for v in tf.values())/8,rows=rows,scope='Teacher-forced gold-prefix diagnostics separate from public-only free-running records. Invalid output receives no partial graph credit. TRAIN panel is training-exposed; no calibration.')
 t=time.monotonic();artifact=out/f'{label}-u{update}.json.gz';write_gzip(artifact,data);export_seconds=time.monotonic()-t
 return dict(artifact=artifact.name,sha256=digest(artifact),examples=len(rows),complete=sum(int(r['complete']) for r in rows),valid=sum(int(r['valid']) for r in rows),cells={k:dict(examples=n,complete=cell_exact[k]) for k,n in cell_counts.items()},invalid_reasons=dict(invalid),teacher_forced=tf,teacher_forced_loss=data['teacher_forced_loss'],teacher_forced_seconds=tf_seconds,free_running_seconds=free_seconds,export_seconds=export_seconds,total_seconds=time.monotonic()-tick)


def worst_case_timing(model,publics):
 """Timing stress only:160 cached own-prediction steps, ignoring stop/validity."""
 was_training=model.training;model.eval();torch.cuda.synchronize();tick=time.monotonic()
 with torch.no_grad():
  stress_records=[[] for _ in publics];cache=model.begin(publics);previous=torch.full((len(publics),5),-1,dtype=torch.long,device=cache.memory.device);previous[:,0]=BOS
  for step in range(160):
   logits,cache=model._step(previous,cache,validate=False)
   keys=tuple(logits);chosen=torch.stack([logits[k].argmax(-1) for k in keys],-1).cpu().tolist();rows=[]
   for values in chosen:
    choice=dict(zip(keys,values));tag=choice['type']+1
    if tag==NODE:
     kind=choice['kind'];lexical=kind in (KINDS.index('ident'),KINDS.index('entity'));rows.append([NODE,kind,-1 if lexical else choice['value'],choice['copy'] if lexical else -1,-1])
    elif tag==EDGE:rows.append([EDGE,choice['source'],choice['target'],choice['role'],choice['slot']-1])
    else:rows.append([EOS,-1,-1,-1,-1])
   for saved,row in zip(stress_records,rows):saved.append(row)
   previous=torch.tensor(rows,dtype=torch.long,device=cache.memory.device)
 torch.cuda.synchronize();seconds=time.monotonic()-tick;model.train(was_training)
 return dict(seconds=seconds,batch_size=len(publics),steps=160,records=stress_records,scope='Timing-only own-prediction rollout ignores EOS/invalid stop, no gold prefix/count. Not an evaluated prediction or quality result.')


def run(c):
 started=time.monotonic();torch.set_num_threads(2);vocab,train,panel,dev=prepared_data(c);preflight_seconds=time.monotonic()-started;out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=False)
 torch.manual_seed(1901);torch.cuda.manual_seed_all(1901);tick=time.monotonic();model=TypedRecordActor(value_count=len(vocab),width=1024,heads=8,node_capacity=128,max_records=160,max_slot=32,autocast_dtype='bfloat16').to(c['device']);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01);setup_seconds=time.monotonic()-tick;initial=state_hash(model)
 generator=torch.Generator().manual_seed(c['schedule_seed']);order=torch.randperm(4096,generator=generator).tolist();position=0;visits=[0]*4096;sequence=hashlib.sha256();curves=[];losses=[];training_seconds=0.;optimizer_records=optimizer_tokens=optimizer_nodes=optimizer_edges=0;torch.cuda.reset_peak_memory_stats();stress=None
 if c['job']=='profile':
  # Deterministic public-length stress: first32 longest token sequences, ties byTRAINindex.
  ids=sorted(range(len(train)),key=lambda i:(-len(tokens(train[i]['public'])),i))[:32];stress=worst_case_timing(model,[train[i]['public'] for i in ids]);stress['train_indices']=ids;tick=time.monotonic();stress_path=out/'worst-case-profile.json.gz';write_gzip(stress_path,stress);stress_export=time.monotonic()-tick;stress={k:v for k,v in stress.items() if k!='records'};stress.update(artifact=stress_path.name,sha256=digest(stress_path),export_seconds=stress_export)
 for added in range(c['updates']+1):
  if added in c['checkpoints']:
   tr=evaluate(model,panel,len(vocab),out,'train',added,32);dv=evaluate(model,dev,len(vocab),out,'development',added,32);tick=time.monotonic();path=out/f'model-u{added}.pt';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),schedule=generator.get_state(),order=order,position=position,visits=visits,update=added),path)
   curve=dict(update=added,train=tr,development=dv,model_state_sha256=state_hash(model),checkpoint_sha256=digest(path),checkpoint_export_seconds=time.monotonic()-tick,presentations=sum(visits),optimizer_records=optimizer_records,optimizer_tokens=optimizer_tokens);curves.append(curve);print(json.dumps(dict(event='evaluation_complete',update=added,train_seconds=tr['total_seconds'],dev_seconds=dv['total_seconds'])),flush=True)
  if added==c['updates']:break
  indices,order,position=next_indices(order,position,generator,8);batch=[train[i] for i in indices];records=padded_records([x['records'] for x in batch],model.features.weight.device);sequence.update(json.dumps(indices).encode())
  for i,x in zip(indices,batch):visits[i]+=1;optimizer_records+=len(x['records']);optimizer_tokens+=len(tokens(x['public']));optimizer_nodes+=len(x['row']['nodes']);optimizer_edges+=len(x['row']['edges'])
  lr=learning_rate(added+1)
  for group in optimizer.param_groups:group['lr']=lr
  torch.cuda.synchronize();tick=time.monotonic();optimizer.zero_grad();outputs=model.teacher_forced([x['public'] for x in batch],records);loss,sums,counts,correct=field_losses(outputs,records)
  if not torch.isfinite(loss):raise FloatingPointError('nonfinite S19 loss')
  loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();torch.cuda.synchronize();training_seconds+=time.monotonic()-tick
  if (added+1)%128==0 or added+1==c['updates']:
   log=dict(update=added+1,learning_rate=lr,loss=float(loss.detach()),field_loss_sums={k:float(v.detach()) for k,v in sums.items()},field_counts=counts,training_seconds=training_seconds);losses.append(log);print(json.dumps(dict(event='progress',**log)),flush=True)
  del outputs,loss,sums,correct,records
 if c['job']=='main':
  if len(visits)!=4096 or set(visits)!={8}:raise ValueError('incomplete eight-epoch exposure')
  if sequence.hexdigest()!='85054bf25e3a3e2a5a9a932b441d6413fd0d7ad827df79e1583268aaddf5acd9' or (optimizer_tokens,optimizer_nodes,optimizer_edges,optimizer_records)!=(1720320,988008,2208616,3229392):raise ValueError('matched construction stream/exposure changed')
 manifest=dict(config=c,parameters=sum(p.numel() for p in model.parameters()),initial_state_sha256=initial,final_state_sha256=state_hash(model),inherited_presentations=0,added_presentations=sum(visits),visits=visits,construction_sequence_sha256=sequence.hexdigest(),optimizer_records=optimizer_records,optimizer_tokens=optimizer_tokens,optimizer_nodes=optimizer_nodes,optimizer_edges=optimizer_edges,training_seconds=training_seconds,preflight_seconds=preflight_seconds,setup_seconds=setup_seconds,worst_case_timing=stress,curves=curves,losses=losses,process_seconds=time.monotonic()-started,peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 write_gzip(out/'manifest.json.gz',manifest);return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');args=p.parse_args();run(json.loads(Path(args.config).read_text()))
