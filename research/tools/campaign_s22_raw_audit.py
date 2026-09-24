"""CPU-only complete S22 archived record and privileged-boundary audit."""
import argparse,collections,gzip,hashlib,json,time
from pathlib import Path
import torch
from campaign_s19_raw_audit import score,read,digest
from topoformer.campaign_semantics_data import load_cache
from topoformer.campaign_semantics_s19_codec import encode_row,NODE
p=argparse.ArgumentParser()
for k in ('run','receipt','base','output'):p.add_argument(k,type=Path)
a=p.parse_args();start=time.monotonic();torch.set_num_threads(2)
m=read(a.run/'manifest.json.gz');c=m['config'];receipt=json.loads(a.receipt.read_text())
assert receipt['exit_code']==0 and not receipt['timed_out'] and not receipt['gpu_processes']
assert hashlib.sha256((json.dumps(c,indent=2)+'\n').encode()).hexdigest()==receipt['config_sha256']
assert c['no_optimizer_or_training'] and m['no_optimizer_or_training']
assert c['arms']==['original','broad'] and c['policies']==['oracle_node_count','oracle_node_kinds','oracle_node_prefix']
for v in c['inputs'].values():assert digest(a.base/v['path'])==v['sha256']
vocab=json.loads((a.base/c['inputs']['vocabulary_audit']['path']).read_text())['value_vocabulary'];dev=load_cache(a.base/c['inputs']['development']['path'])
cells=[(2,4),(3,3),(3,4),(4,3),(4,4),(5,3),(5,4)];chosen=[r for cell in cells for r in [r for r in dev if (r['arity'],r['facts'])==cell][:c['dev_per_cell']]]
assert [r['arm'] for r in m['results']]==c['arms'];reports=[];graphs=0;flags={};summaries={};baseline={}
for result in m['results']:
 arm=result['arm'];bind=c['checkpoints'][arm];folder=a.run/arm;assert read(folder/'manifest.json.gz')==result
 assert result['no_optimizer_or_training'] and result['parameters']==62677315 and result['model_state_sha256']==bind['model_state_sha256'] and result['checkpoint_sha256']==bind['checkpoint']['sha256']
 for key in ('checkpoint','manifest','public_reference'):assert digest(a.base/bind[key]['path'])==bind[key]['sha256']
 reference={r['semantic_sha256']:r for r in read(a.base/bind['public_reference']['path'])['rows']};flags[arm]={};summaries[arm]={};baseline[arm]={}
 for cell in cells:
  label=f'{cell[0]}x{cell[1]}';rr=[r for r in chosen if (r['arity'],r['facts'])==cell];baseline[arm][label]={r['semantic_sha256']:reference[r['semantic_sha256']]['complete'] for r in rr}
 assert [e['policy'] for e in result['policies']]==c['policies']
 for entry in result['policies']:
  policy=entry['policy'];path=folder/entry['artifact'];assert digest(path)==entry['sha256'];d=read(path);assert d['policy']==policy and len(d['rows'])==len(chosen)==entry['examples']
  stats=collections.Counter();valid=exact=0;invalid=collections.Counter();cc={};ff={}
  for raw,row in zip(d['rows'],chosen,strict=True):
   vi,ex,reason=score(raw,row,vocab);graphs+=1;valid+=vi;exact+=ex
   if not vi:invalid[reason]+=1
   old=reference[row['semantic_sha256']];assert raw['target']==old['target'] and raw['public_reference']=={k:old[k] for k in ('valid','complete','reason','exact_components')}
   nodes=[list(r) for r in encode_row(row,vocab) if r[0]==NODE];n=len(nodes);records=raw['records'];s=raw['controller_stats']
   assert s['policy']==policy and s['supplied_node_count']==s['forced_node_tags']==n
   assert len(records)>n and all(r[0]==1 for r in records[:n]) and all(r[0] in (2,3) for r in records[n:])
   assert s['node_suppression_steps']==len(records)-n and 0<=s['node_suppression_replacements']<=s['node_suppression_steps'] and 0<=s['tag_replacements']<=n
   kinds=policy!='oracle_node_count';prefix=policy=='oracle_node_prefix'
   assert s['forced_node_kinds']==(n if kinds else 0) and s['supplied_prefix_records']==(n if prefix else 0)
   assert 0<=s['kind_replacements']<=s['forced_node_kinds']
   for k in ('prefix_value_replacements','prefix_copy_replacements'):assert 0<=s[k]<=(n if prefix else 0)
   if kinds:assert [r[1] for r in records[:n]]==[r[1] for r in nodes]
   if prefix:assert records[:n]==nodes
   stats.update({k:v for k,v in s.items() if k!='policy'});comp=raw['exact_components'];assert raw['exact_nodes']==all(comp[k] for k in ('presence','kind','value','copy')) and raw['exact_relations']==(comp['edges'] and comp['slots'])
   cell=f"{row['arity']}x{row['facts']}";ff.setdefault(cell,{})[row['semantic_sha256']]=bool(ex);v=cc.setdefault(cell,dict(examples=0,valid=0,complete=0,exact_nodes=0,exact_relations=0))
   for k in v:v[k]+=1 if k=='examples' else int(raw[k])
  assert entry['valid']==valid and entry['complete']==exact and entry['invalid_reasons']==dict(invalid) and entry['cells']==cc and entry['controller_stats']==dict(stats)
  flags[arm][policy]=ff
  summaries[arm][policy]=dict(examples=len(chosen),valid=valid,complete=exact,cells=cc,controller_stats=dict(stats),invalid_reasons=dict(invalid))
  assert entry['model_state_sha256']==bind['model_state_sha256'];reports.append(dict(arm=arm,policy=policy,examples=len(chosen),artifact_sha256=digest(path),decode_seconds=entry['decode_seconds'],export_seconds=entry['export_seconds'],total_seconds=entry['total_seconds']))
assert graphs==len(chosen)*6
out=dict(status='PASS',graphs=graphs,policies=6,cpu_wall_seconds=time.monotonic()-start,manifest_sha256=digest(a.run/'manifest.json.gz'),receipt_sha256=digest(a.receipt),reports=reports,flags=flags,baseline=baseline,summaries=summaries,scope='Every generated record/strict invalid/target/component/metric, supplied NODE boundary, aggregate counter and checkpoint file binding. No actor or forward. Replacement counters only source-bound plus bounds, since unforced logits are not archived.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('reports','flags','baseline','summaries')}))
