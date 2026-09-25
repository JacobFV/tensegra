"""S22 frozen-checkpoint privileged NODE-boundary diagnostics; no training."""
import argparse,collections,gzip,json,resource,time
from pathlib import Path
import torch
from .campaign_semantics import digest,write_gzip
from .campaign_semantics_data import load_cache,target
from .campaign_semantics_s19 import decode_evaluation,worst_case_timing
from .campaign_semantics_s19_actor import TypedRecordActor
from .campaign_semantics_s19_codec import encode_row,NODE
from .campaign_semantics_s22_controller import decode,NodeCount,NodeKinds,NodePrefix
from .campaign_semantics_s22_freeze import verify,CELLS,POLICIES
from .semantic_scaling import tokens
from .thinking_language import ActorInput,state_hash

def boundary(node_records,policy):
 """Typed narrow metadata only; controller never receives graph rows or targets."""
 if any(len(r)!=5 or r[0]!=NODE for r in node_records):raise ValueError('NODE-only boundary required')
 if policy=='oracle_node_count':return NodeCount(len(node_records))
 if policy=='oracle_node_kinds':return NodeKinds(tuple(r[1] for r in node_records))
 if policy=='oracle_node_prefix':return NodePrefix(tuple(tuple(r) for r in node_records))
 raise ValueError('unregistered policy')
def prepare(c):
 vocab=json.loads(Path(c['inputs']['vocabulary_audit']['path']).read_text())['value_vocabulary'];rows=load_cache(c['inputs']['development']['path'])
 if len(rows)!=3584 or collections.Counter((r['arity'],r['facts']) for r in rows)!=dict.fromkeys(CELLS,512):raise ValueError('fixed seven DEV512 cells')
 chosen=[r for cell in CELLS for r in [r for r in rows if (r['arity'],r['facts'])==cell][:c['dev_per_cell']]];examples=[]
 for row in chosen:
  records=encode_row(row,vocab);node_records=[]
  for record in records:
   if record[0]!=NODE:break
   node_records.append(tuple(record))
  # Full graph targets are kept in evaluator storage, outside controller arguments.
  examples.append(dict(row=row,public=ActorInput(row['text'],()),nodes=tuple(node_records),gold=target(row,vocab)))
 return vocab,examples

def evaluate_policy(model,examples,vocab,policy,reference,out):
 before=state_hash(model);rows=[];decode_seconds=0.;tick=time.monotonic()
 with torch.no_grad():
  for start in range(0,len(examples),32):
   batch=examples[start:start+32];publics=[x['public'] for x in batch];boundaries=[boundary(x['nodes'],policy) for x in batch]
   torch.cuda.synchronize();t=time.monotonic();generated=decode(model,publics,policy=policy,boundaries=boundaries);torch.cuda.synchronize();decode_seconds+=time.monotonic()-t
   if len(generated)!=len(batch):raise ValueError('controller batch size')
   for example,pred in zip(batch,generated):
    row=example['row'];item=decode_evaluation(example['public'],pred,example['gold'],len(vocab));event=row['semantic_sha256'];old=reference[event]
    if item['target']!=old['target'] or old['seed']!=row['seed'] or old['graph_sha256']!=row['graph_sha256']:raise ValueError('public reference target/identity mismatch')
    comp=item['exact_components'];item.update(seed=row['seed'],semantic_sha256=event,graph_sha256=row['graph_sha256'],cell=f"{row['arity']}x{row['facts']}",policy=policy,controller_stats=pred['controller_stats'],exact_nodes=all(comp[k] for k in ('presence','kind','value','copy')),exact_relations=comp['edges'] and comp['slots'],public_reference={k:old[k] for k in ('valid','complete','reason','exact_components')});rows.append(item)
 if state_hash(model)!=before:raise ValueError('diagnostic mutated frozen model')
 stats=collections.Counter()
 for row in rows:
  for key,value in row['controller_stats'].items():
   if isinstance(value,(int,float)):stats[key]+=value
 cells={}
 for cell in CELLS:
  rr=[r for r in rows if r['cell']==f'{cell[0]}x{cell[1]}'];cells[f'{cell[0]}x{cell[1]}']=dict(examples=len(rr),valid=sum(r['valid'] for r in rr),complete=sum(r['complete'] for r in rr),exact_nodes=sum(r['exact_nodes'] for r in rr),exact_relations=sum(r['exact_relations'] for r in rr))
 doc=dict(policy=policy,rows=rows,scope='Privileged NODE-boundary diagnostic; supplied information explicitly named by policy. Edges/slots/termination predicted; no learned-public acquisition or historical-gate claim. No training/calibration.');path=out/(policy+'.json.gz');t=time.monotonic();write_gzip(path,doc);export_seconds=time.monotonic()-t
 return dict(policy=policy,artifact=path.name,sha256=digest(path),examples=len(rows),valid=sum(r['valid'] for r in rows),complete=sum(r['complete'] for r in rows),cells=cells,invalid_reasons=dict(collections.Counter(r['reason'] for r in rows if not r['valid'])),controller_stats=dict(stats),decode_seconds=decode_seconds,export_seconds=export_seconds,total_seconds=time.monotonic()-tick,model_state_sha256=before)

def run(c):
 start=time.monotonic();torch.set_num_threads(2);verify(c,Path(__file__).parent,c['cap_seconds']);vocab,examples=prepare(c);preflight_seconds=time.monotonic()-start;out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=False);results=[];torch.cuda.reset_peak_memory_stats()
 for arm in c['arms']:
  bind=c['checkpoints'][arm];tick=time.monotonic();checkpoint=torch.load(bind['checkpoint']['path'],map_location='cpu',weights_only=True)
  if checkpoint['update']!=4096 or checkpoint['seed']!=2101 or checkpoint['arm']!=arm or len(checkpoint['visits'])!=4096 or set(checkpoint['visits'])!={8}:raise ValueError('wrong frozen endpoint')
  model=TypedRecordActor(value_count=len(vocab),width=1024,heads=8,node_capacity=128,max_records=160,max_slot=32,autocast_dtype='bfloat16').to(c['device']);model.load_state_dict(checkpoint['model']);model.eval();del checkpoint
  if state_hash(model)!=bind['model_state_sha256']:raise ValueError('frozen model state mismatch')
  setup_seconds=time.monotonic()-tick;folder=out/arm;folder.mkdir();ref=json.load(gzip.open(bind['public_reference']['path'],'rt'));reference={r['semantic_sha256']:r for r in ref['rows']}
  if len(reference)!=len(ref['rows']) or len(reference)!=3584:raise ValueError('public reference population')
  if len(reference)!=3584 or not all(x['row']['semantic_sha256'] in reference for x in examples):raise ValueError('reference event coverage')
  stress=None
  if c['job']=='profile':
   ids=sorted(range(len(examples)),key=lambda i:(-len(tokens(examples[i]['public'])),i))[:32];stress=worst_case_timing(model,[examples[i]['public'] for i in ids]);stress['example_indices']=ids;t=time.monotonic();path=folder/'worst-case-profile.json.gz';write_gzip(path,stress);stress={k:v for k,v in stress.items() if k!='records'};stress.update(artifact=path.name,sha256=digest(path),export_seconds=time.monotonic()-t)
  policies=[evaluate_policy(model,examples,vocab,p,reference,folder) for p in POLICIES]
  if state_hash(model)!=bind['model_state_sha256'] or digest(bind['checkpoint']['path'])!=bind['checkpoint']['sha256']:raise ValueError('checkpoint/state changed')
  result=dict(arm=arm,checkpoint_sha256=bind['checkpoint']['sha256'],model_state_sha256=state_hash(model),parameters=sum(p.numel() for p in model.parameters()),setup_seconds=setup_seconds,worst_case_timing=stress,policies=policies,no_optimizer_or_training=True);write_gzip(folder/'manifest.json.gz',result);results.append(result);del model
  print(json.dumps(dict(event='arm_complete',arm=arm,policy_count=len(policies))),flush=True)
 manifest=dict(config=c,results=results,no_optimizer_or_training=True,preflight_seconds=preflight_seconds,process_seconds=time.monotonic()-start,peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);write_gzip(out/'manifest.json.gz',manifest);return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();run(json.loads(Path(a.config).read_text()))
