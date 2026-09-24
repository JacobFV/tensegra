"""Closed complete S20 main categorical-record audit; CPU and no actor calls."""
import argparse,collections,gzip,hashlib,json,math,time
from pathlib import Path
import torch
from campaign_s19_raw_audit import score
from topoformer.campaign_semantics_data import load_cache
from topoformer.campaign_semantics_s19_codec import encode_row,NODE,KINDS

p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('receipt',type=Path);p.add_argument('data_root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();start=time.monotonic();torch.set_num_threads(2)
load=lambda p:json.load(gzip.open(p,'rt'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
m=load(a.run/'manifest.json.gz');c=m['config'];receipt=json.loads(a.receipt.read_text());assert receipt['exit_code']==0 and not receipt['timed_out'] and receipt['gpu_processes']=='';assert receipt['config_sha256']=='4e648a9a274a7cabcbb84c0b2e6e6a2dcbf37925e2146b0758702086245f6c8b';assert hashlib.sha256((json.dumps(c,indent=2)+'\n').encode()).hexdigest()==receipt['config_sha256'];assert [(r['seed'],r['arm']) for r in m['results']]==[(s,arm) for s in (701,702,703) for arm in ('original','context','record')];assert all(r['added_update']==4096 for r in m['results'])
# Only after full closed-main guard: parse TRAIN and confirmation.
paths={k:a.data_root/v['path'] for k,v in c['inputs'].items()}
for k,q in paths.items():assert sha(q)==c['inputs'][k]['sha256']
v=json.loads(paths['vocabulary_audit'].read_text())['value_vocabulary'];train=load_cache(paths['train']);selection=json.loads(paths['selection'].read_text())['mixed'];panel=[train[r['index']] for r in selection];evaluation=load_cache(paths['confirmation']);evaluation=[r for arity in (3,4) for facts in (3,4) for r in evaluation if (r['arity'],r['facts'])==(arity,facts)];assert len(panel)==128 and len(evaluation)==2048 and len({r['semantic_sha256'] for r in evaluation})==2048
flags={};reports=[];bindings={'manifest.json.gz':sha(a.run/'manifest.json.gz')};graphs=0
for result in m['results']:
 if result['arm']!='record':continue
 seed=result['seed'];folder=a.run/f'seed-{seed}'/'record';assert result==load(folder/'manifest.json.gz');assert result['update']==4096 and result['evaluation_population']=='confirmation';flags[str(seed)]={'record':{'categorical':{}}}
 for label,pop in [('train',panel),('evaluation',evaluation)]:
  entry=result[label];path=folder/entry['artifact'];assert sha(path)==entry['sha256'];bindings[str(path.relative_to(a.run))]=sha(path);d=load(path);assert d['update']==4096 and d['population']==('train' if label=='train' else 'confirmation');assert len(d['rows'])==len(pop)==entry['examples'];supports=collections.Counter();invalid=collections.Counter();cells={};valid=exact=0
  for raw,row in zip(d['rows'],pop,strict=True):
   vi,ex,reason=score(raw,row,v);graphs+=1;valid+=vi;exact+=ex;cell=f"{row['arity']}x{row['facts']}";assert raw['cell']==cell;cc=cells.setdefault(cell,dict(examples=0,complete=0));cc['examples']+=1;cc['complete']+=int(ex)
   if not vi:invalid[reason]+=1
   if label=='evaluation':flags[str(seed)]['record']['categorical'].setdefault(cell,{})[row['semantic_sha256']]=bool(ex)
   rec=encode_row(row,v);n=len(row['nodes']);lex=sum(x[0]==NODE and x[1] in (KINDS.index('ident'),KINDS.index('entity')) for x in rec);ed=len(row['edges']);supports.update(type=len(rec),kind=n,value=n-lex,copy=lex,source=ed,target=ed,role=ed,slot=ed)
  assert entry['valid']==valid and entry['complete']==exact and entry['invalid_reasons']==dict(invalid) and entry['cells']==cells and entry['teacher_forced']==d['teacher_forced']
  for k,x in d['teacher_forced'].items():assert x['count']==supports[k] and 0<=x['correct']<=x['count'] and math.isfinite(x['loss_sum']) and math.isclose(x['mean_loss'],x['loss_sum']/x['count']) and math.isclose(x['accuracy'],x['correct']/x['count'])
  assert math.isclose(d['teacher_forced_loss'],sum(x['mean_loss'] for x in d['teacher_forced'].values())/8) and d['teacher_forced_loss']==entry['teacher_forced_loss'];reports.append(dict(seed=seed,population=label,cells=cells,supports=dict(supports),invalid=dict(invalid)))
r=dict(status='pass',seconds=time.monotonic()-start,graphs=graphs,flags=flags,reports=reports,artifact_sha256=bindings,scope='All3record endpoints strict generated-record/publiccopy/target/metric reconstruction andTFsupports; no actor/inference. TF logitsums remain source-bound aggregates, not new forward.')
a.output.write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ('flags','reports','artifact_sha256')})
