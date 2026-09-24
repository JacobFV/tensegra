"""Independent final S21 prefix/edge descriptors from closed, raw-audited rows."""
import argparse,base64,collections,gzip,hashlib,json,re,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
for n in ('run','public','localization','events','output'):p.add_argument(n,type=Path)
a=p.parse_args();start=time.monotonic();load=lambda p:json.load(gzip.open(p,'rt'));given=json.loads(a.localization.read_text());events=load(a.events)
assert given['config_sha256']=='5a13908ec22e9ad47350d7d85433b46291dada4552a474300fcef57871f30db3'
assert given['manifest_sha256']==hashlib.sha256((a.run/'manifest.json.gz').read_bytes()).hexdigest()
assert given['event_artifact']['sha256']==hashlib.sha256(a.events.read_bytes()).hexdigest()
public={r['semantic_sha256']:r for r in map(json.loads,gzip.open(a.public,'rt'))};docs={arm:load(a.run/arm/'development-u4096.json.gz')['rows'] for arm in ('original','broad')};templates={}
for r in docs['original']:
 seq=tuple(k for k,v in zip(r['target']['kind'],r['target']['presence']) if v and k!=1);assert templates.setdefault(r['cell'],seq)==seq
def first(x,y):
 return next((i for i in range(max(len(x),len(y))) if i>=len(x) or i>=len(y) or x[i]!=y[i]),None)
reconstructed={};count=0
for arm,rows in docs.items():
 saved={r['semantic_sha256']:r for r in events[arm]};assert len(saved)==3584;reconstructed[arm]=[]
 for r in rows:
  tokens=re.findall(r'\w+|[^\w\s]',public[r['semantic_sha256']]['text']);records=[list(x) for x in r['records']]
  for rec in records:
   if rec[0]==1 and 0<=rec[3]<len(tokens):rec[3]=tokens.index(tokens[rec[3]])
  nodes=[]
  for rec in records:
   if rec[0]!=1:break
   nodes.append(rec[1:4])
  t=r['target'];gold=[[t[k][i] for k in ('kind','value','copy')] for i,v in enumerate(t['presence']) if v];idx=first(nodes,gold);fields=[]
  if idx is not None:
   fields=['missing_node'] if idx>=len(nodes) else ['extra_node'] if idx>=len(gold) else [k for j,k in enumerate(('kind','value','copy')) if nodes[idx][j]!=gold[idx][j]]
  nlate=sum(x[0]==1 for x in records[len(nodes):]);edge=t['edges'];bits=np.unpackbits(np.frombuffer(base64.b64decode(edge['packed_b64']),dtype=np.uint8),bitorder='little')[:np.prod(edge['shape'])].reshape(edge['shape']);ge=[(int(i),int(j),int(k),t['slots'][i][j]) for i,j,k in np.argwhere(bits)];pe=[tuple(x[1:]) for x in records if x[0]==2];missing=sorted(set(ge)-set(pe));extra=sorted(set(pe)-set(ge));wrongslot=sum(any(y[:3]==x[:3] and y[3]!=x[3] for y in pe) for x in missing)
  kinds=tuple(x[0] for x in nodes);skeleton=tuple(x for x in kinds if x!=1);goldk=tuple(x[0] for x in gold);category='node:'+','.join(fields) if idx is not None else 'node:nonleading_node' if nlate else 'edge_or_slot' if missing or extra else 'invalid_termination_or_record' if not r['valid'] else 'other_requires_audit' if not r['complete'] else 'complete'
  v=dict(semantic_sha256=r['semantic_sha256'],cell=r['cell'],valid=r['valid'],reason=r['reason'],complete=r['complete'],category=category,generated_leading_nodes=len(nodes),nonleading_node_records=nlate,gold_nodes=len(gold),first_node_index=idx,first_node_fields=fields,first_node_predicted=None if idx is None or idx>=len(nodes) else nodes[idx],first_node_gold=None if idx is None or idx>=len(gold) else gold[idx],kind_sequence_correct=kinds==goldk,entity_removed_layout_correct=skeleton==tuple(x for x in goldk if x!=1),matching_layouts=[k for k,tpl in templates.items() if tpl==skeleton],eos_present=any(x[0]==3 for x in records),missing_edge_count=len(missing),extra_edge_count=len(extra),slot_only_missing_count=wrongslot,first_missing_edge=list(missing[0]) if missing else None,first_extra_edge=list(extra[0]) if extra else None,first_positional_edge_difference=first(pe,ge),duplicate_edge_count=len(pe)-len(set(pe)),leading_nodes_correct=idx is None,all_nodes_correct=idx is None and nlate==0,edges_correct_after_nodes=idx is None and nlate==0 and not missing and not extra,raw_record_count=len(records))
  assert v==saved[r['semantic_sha256']],(arm,r['semantic_sha256']);reconstructed[arm].append(v);count+=1
 for cell,summary in given['summaries'][arm].items():
  rr=[x for x in reconstructed[arm] if x['cell']==cell];assert summary['examples']==len(rr)==512
  assert summary['counts']=={k:sum(r[k] for r in rr) for k in summary['counts']}
  assert summary['categories']==dict(collections.Counter(r['category'] for r in rr));assert summary['invalid_reasons']==dict(collections.Counter(r['reason'] for r in rr if not r['valid']))
  for key,values in [('first_node_fields',[','.join(r['first_node_fields']) or 'none' for r in rr]),('first_node_indices',[str(r['first_node_index']) for r in rr]),('matching_layouts',[','.join(r['matching_layouts']) or 'other' for r in rr]),('generated_leading_node_counts',[str(r['generated_leading_nodes']) for r in rr])]:assert summary[key]==dict(collections.Counter(values))
  assert summary['edge_error_counts_after_correct_nodes']=={k:sum(r[field] for r in rr if r['all_nodes_correct']) for k,field in [('missing','missing_edge_count'),('extra','extra_edge_count'),('slot_only_missing','slot_only_missing_count')]}
left={r['semantic_sha256']:r for r in reconstructed['original']};right={r['semantic_sha256']:r for r in reconstructed['broad']};assert set(left)==set(right)==set(public)
for cell,g in given['paired'].items():
 ids=[i for i,r in left.items() if r['cell']==cell];assert g['complete_transitions']=={f'{i}->{j}':sum(left[k]['complete']==bool(i) and right[k]['complete']==bool(j) for k in ids) for i in (0,1) for j in (0,1)}
 assert g['category_transitions']==dict(collections.Counter(left[k]['category']+' -> '+right[k]['category'] for k in ids));assert g['layout_transitions']==dict(collections.Counter(str(left[k]['entity_removed_layout_correct'])+' -> '+str(right[k]['entity_removed_layout_correct']) for k in ids))
out=dict(status='pass',events=count,seconds=time.monotonic()-start,localization_sha256=hashlib.sha256(a.localization.read_bytes()).hexdigest(),events_sha256=hashlib.sha256(a.events.read_bytes()).hexdigest(),scope='All7168 final leading-prefix/copy/kind/layout/edge/slot descriptors, cell summaries and paired transitions independently reconstructed. Descriptive canonical-position failures only; no graph repair, length causality or universal learnability claim.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
