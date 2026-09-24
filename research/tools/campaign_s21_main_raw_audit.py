"""Closed complete S21 raw codec/target/metric/TF support audit, CPU only."""
import argparse,collections,gzip,hashlib,json,math,os,time
from pathlib import Path
import torch
from campaign_s19_raw_audit import score
from topoformer.campaign_semantics_s21 import prepare
from topoformer.campaign_semantics_s19_codec import NODE,KINDS
p=argparse.ArgumentParser()
for n in ('run','receipt','data_root','output'):p.add_argument(n,type=Path)
a=p.parse_args();start=time.monotonic();torch.set_num_threads(2)
load=lambda p:json.load(gzip.open(p,'rt'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def summarize(rows):
 valid=[r for r in rows if r['valid']]
 return dict(examples=len(rows),valid=len(valid),complete=sum(r['complete'] for r in rows),exact_components={k:sum(r['exact_components'][k] for r in rows) for k in ('presence','kind','value','copy','edges','slots')},invalid_reasons=dict(collections.Counter(r['reason'] for r in rows if not r['valid'])),macro_graph_f1_invalid_zero={k:sum(r['metrics'][k]['f1'] for r in valid)/max(1,len(rows)) for k in ('node','typed_edge','ordered_edge')},canonical_edge_order_count=sum(r.get('canonical_edge_order',False) for r in rows),failure_semantic_ids=sorted(r['semantic_sha256'] for r in rows if not r['complete']))
receipt=json.loads(a.receipt.read_text());assert receipt['exit_code']==0 and not receipt['timed_out'] and receipt.get('error') is None and receipt['gpu_processes']==''
assert receipt['config_sha256']=='5a13908ec22e9ad47350d7d85433b46291dada4552a474300fcef57871f30db3'
m=load(a.run/'manifest.json.gz');c=m['config'];assert hashlib.sha256((json.dumps(c,indent=2)+'\n').encode()).hexdigest()==receipt['config_sha256']
assert c['job']=='main' and c['updates']==4096 and c['dev_per_cell']==512 and c['checkpoints']==[0,1024,2048,4096]
assert [r['arm'] for r in m['results']]==['original','broad'] and all(r['added_presentations']==32768 and [v['update'] for v in r['curves']]==[0,1024,2048,4096] for r in m['results'])
for x in c['inputs'].values():assert sha(a.data_root/x['path'])==x['sha256']
# Complete matrix/receipt guards precede every experimental prediction read.
os.chdir(a.data_root);vocab,arms,dev=prepare(c);assert len(dev)==3584;reports=[];flags={};bindings={};graphs=0
for result in m['results']:
 arm=result['arm'];folder=a.run/arm;assert result==load(folder/'manifest.json.gz');flags[arm]={};train,panel=arms[arm]
 for curve in result['curves']:
  step=curve['update'];flags[arm][str(step)]={}
  for label,pop in [('train',panel),('development',dev)]:
   entry=curve[label];path=folder/entry['artifact'];assert sha(path)==entry['sha256'];bindings[str(path.relative_to(a.run))]=sha(path);d=load(path)
   assert d['population']==label and d['update']==step and len(d['rows'])==len(pop)==entry['examples'];supports=collections.Counter();invalid=collections.Counter();cells={};valid=exact=0;ff={}
   for raw,x in zip(d['rows'],pop,strict=True):
    row=x['row'];vi,ex,reason=score(raw,row,vocab);graphs+=1;valid+=vi;exact+=ex;cell=f"{row['arity']}x{row['facts']}";assert raw['cell']==cell;cc=cells.setdefault(cell,dict(examples=0,complete=0));cc['examples']+=1;cc['complete']+=int(ex)
    assert row['semantic_sha256'] not in ff.setdefault(cell,{});ff[cell][row['semantic_sha256']]=bool(ex)
    if not vi:invalid[reason]+=1
    rec=x['records'];n=len(row['nodes']);lex=sum(v[0]==NODE and v[1] in (KINDS.index('ident'),KINDS.index('entity')) for v in rec);ed=len(row['edges']);supports.update(type=len(rec),kind=n,value=n-lex,copy=lex,source=ed,target=ed,role=ed,slot=ed)
   assert entry['valid']==valid and entry['complete']==exact and entry['invalid_reasons']==dict(invalid) and entry['cells']==cells and entry['teacher_forced']==d['teacher_forced']
   for k,x in d['teacher_forced'].items():assert x['count']==supports[k] and 0<=x['correct']<=x['count'] and math.isfinite(x['loss_sum']) and math.isclose(x['mean_loss'],x['loss_sum']/x['count']) and math.isclose(x['accuracy'],x['correct']/x['count'])
   assert math.isclose(d['teacher_forced_loss'],sum(x['mean_loss'] for x in d['teacher_forced'].values())/8) and d['teacher_forced_loss']==entry['teacher_forced_loss']
   summary=summarize(d['rows']);summary.update(cells={cell:summarize([r for r in d['rows'] if r['cell']==cell]) for cell in cells},teacher_forced=d['teacher_forced'],teacher_forced_loss=d['teacher_forced_loss'])
   flags[arm][str(step)][label]=ff;reports.append(dict(arm=arm,update=step,population=label,summary=summary,supports=dict(supports)))
assert graphs==29696
out=dict(status='pass',seconds=time.monotonic()-start,graphs=graphs,reports=reports,flags=flags,artifact_sha256=bindings,manifest_sha256=sha(a.run/'manifest.json.gz'),scope='All two arms/four checkpoints/TRAIN128+DEV7x512 generated records, public copy canonicalization, strict malformed failure, targets/components/metrics and eight-field supports. No new actor calls or sealed confirmation. TF logit sums remain source-bound.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ('flags','reports','artifact_sha256')})
