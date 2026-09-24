"""Prospective posthoc localization: closed paired S21 DEV matrix only; no model."""
import argparse,base64,collections,gzip,hashlib,json,time
from pathlib import Path
CELLS=('2x4','3x3','3x4','4x3','4x4','5x3','5x4');STEPS=(0,1024,2048,4096);ARMS=('original','broad')
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.load(gzip.open(path,'rt'))
def indexed(rows):
 out={r['semantic_sha256']:r for r in rows};require(len(out)==len(rows),'duplicate semantic identity');return out
def target_edges(t):
 e=t['edges'];require(e['bitorder']=='little','edge bit order');n,m,r=e['shape'];buf=base64.b64decode(e['packed_b64']);out=[]
 bits=int.from_bytes(buf,'little')
 while bits:
  lowest=bits&-bits;position=lowest.bit_length()-1;bits^=lowest
  require(position<n*m*r,'nonzero padding edge bit');pair,k=divmod(position,r);i,j=divmod(pair,m);out.append((i,j,k,t['slots'][i][j]))
 return out
def first_difference(a,b):
 for i in range(max(len(a),len(b))):
  if i>=len(a) or i>=len(b) or a[i]!=b[i]:return i
 return None
def canonical_records(records,tokens):
 result=[]
 for raw in records:
  require(len(raw)==5 and all(type(x)is int for x in raw),'malformed raw record container');x=list(raw)
  if x[0]==1 and 0<=x[3]<len(tokens):x[3]=tokens.index(tokens[x[3]])
  result.append(x)
 return result
def describe(row,tokens,templates):
 records=canonical_records(row['records'],tokens);pn=[]
 for tag,a,b,c,d in records:
  if tag!=1:break
  pn.append((a,b,c))
 t=row['target'];n=sum(t['presence']);gn=list(zip(t['kind'][:n],t['value'][:n],t['copy'][:n]));f=first_difference(pn,gn);fields=[]
 if f is not None:
  if f>=len(pn):fields=['missing_node']
  elif f>=len(gn):fields=['extra_node']
  else:fields=[name for j,name in enumerate(('kind','value','copy')) if pn[f][j]!=gn[f][j]]
 pe=[tuple(x[1:]) for x in records if x[0]==2];ge=target_edges(t);missing=sorted(set(ge)-set(pe));extra=sorted(set(pe)-set(ge));slot_only=sum(any(x[:3]==y[:3] and x[3]!=y[3] for y in pe) for x in missing)
 kindseq=tuple(x[0] for x in pn);skeleton=tuple(x for x in kindseq if x!=1);goldsk=tuple(x[0] for x in gn if x[0]!=1)
 if f is not None:category='node:'+','.join(fields)
 elif missing or extra:category='edge_or_slot'
 elif not row['valid']:category='invalid_termination_or_record'
 elif not row['complete']:category='other_requires_audit'
 else:category='complete'
 return dict(semantic_sha256=row['semantic_sha256'],cell=row['cell'],valid=row['valid'],reason=row['reason'],complete=row['complete'],category=category,generated_nodes=len(pn),gold_nodes=n,first_node_index=f,first_node_fields=fields,first_node_predicted=None if f is None or f>=len(pn) else pn[f],first_node_gold=None if f is None or f>=len(gn) else gn[f],kind_sequence_correct=kindseq==tuple(x[0] for x in gn),entity_removed_layout_correct=skeleton==goldsk,matching_layouts=[k for k,v in templates.items() if skeleton==v],eos_present=any(x[0]==3 for x in records),missing_edge_count=len(missing),extra_edge_count=len(extra),slot_only_missing_count=slot_only,first_missing_edge=missing[0] if missing else None,first_extra_edge=extra[0] if extra else None,first_positional_edge_difference=first_difference(pe,ge),duplicate_edge_count=len(pe)-len(set(pe)),all_nodes_correct=f is None,edges_correct_after_nodes=f is None and not missing and not extra,raw_record_count=len(records))
def cell_summary(rows):
 return dict(examples=len(rows),counts={k:sum(bool(r[k]) for r in rows) for k in ('valid','complete','all_nodes_correct','kind_sequence_correct','entity_removed_layout_correct','eos_present','edges_correct_after_nodes')},categories=dict(collections.Counter(r['category'] for r in rows)),invalid_reasons=dict(collections.Counter(r['reason'] for r in rows if not r['valid'])),first_node_fields=dict(collections.Counter(','.join(r['first_node_fields']) or 'none' for r in rows)),first_node_indices=dict(collections.Counter(str(r['first_node_index']) for r in rows)),matching_layouts=dict(collections.Counter(','.join(r['matching_layouts']) or 'other' for r in rows)),generated_node_counts=dict(collections.Counter(str(r['generated_nodes']) for r in rows)),edge_error_counts_after_correct_nodes=dict(missing=sum(r['missing_edge_count'] for r in rows if r['all_nodes_correct']),extra=sum(r['extra_edge_count'] for r in rows if r['all_nodes_correct']),slot_only_missing=sum(r['slot_only_missing_count'] for r in rows if r['all_nodes_correct'])))
def matrix_guard(manifest,receipt,config,config_sha):
 require(receipt['exit_code']==0 and not receipt['timed_out'] and receipt.get('error') is None,'closed successful process required')
 require(receipt['config_sha256']==config_sha and manifest['config']==config,'config mismatch')
 require(config['job']=='main' and config['budget_status']=='frozen' and config['updates']==4096 and config['checkpoints']==list(STEPS) and config['arms']==list(ARMS) and config['dev_per_cell']==512 and config['width']==1024,'full fixed main contract')
 require([x['arm'] for x in manifest['results']]==list(ARMS),'complete paired arms required')
 require(len({x['initial_state_sha256'] for x in manifest['results']})==1,'paired initial state')
 for arm in manifest['results']:
  require(arm['seed']==2101 and arm['inherited_presentations']==0 and arm['added_presentations']==32768 and len(arm['visits'])==4096 and set(arm['visits'])=={8},'completed prescribed exposure')
  require([x['update'] for x in arm['curves']]==list(STEPS),'complete curve matrix required')
  require(arm['curves'][-1]['model_state_sha256']==arm['final_state_sha256'],'fixed endpoint state')
def main(a):
 tick=time.monotonic();c=json.loads(a.config.read_text());require(sha(a.config)==a.config_sha256,'prospective frozen config hash');m=load(a.main/'manifest.json.gz');receipt=json.loads(a.receipt.read_text());matrix_guard(m,receipt,c,a.config_sha256)
 require(sha(a.development)==c['inputs']['development']['sha256'],'public DEV cache');public=indexed([json.loads(x) for x in gzip.open(a.development,'rt')]);require(len(public)==3584,'public population')
 import re
 tokens={k:re.findall(r'\w+|[^\w\s]',r['text']) for k,r in public.items()};inventory={};documents={};identity=None
 # Load both arms/all checkpoints/all populations before performing or writing any localization.
 for arm in m['results']:
  name=arm['arm']
  for curve in arm['curves']:
   for pop in ('train','development'):
    entry=curve[pop];path=a.main/name/entry['artifact'];require(sha(path)==entry['sha256'],'curve artifact hash');d=load(path);rows=d['rows'];ix=indexed(rows)
    require(d['update']==curve['update'] and d['population']==pop and len(rows)==(3584 if pop=='development' else 128),'complete population')
    require(d['teacher_forced']==entry['teacher_forced'] and sum(r['complete'] for r in rows)==entry['complete'] and sum(r['valid'] for r in rows)==entry['valid'],'manifest score binding')
    counts=collections.Counter(r['cell'] for r in rows);expected=dict.fromkeys(CELLS,512) if pop=='development' else ({'3x3':64,'4x3':32,'4x4':32} if name=='original' else {'3x3':32,'4x3':16,'4x4':16,'2x4':16,'5x3':24,'5x4':24});require(counts==expected,'cell support')
    if pop=='development':
     current={k:(r['seed'],r['graph_sha256'],r['cell'],r['target']) for k,r in ix.items()};require(set(ix)==set(public),'same public events')
     if identity is None:identity=current
     else:require(current==identity,'paired targets or events differ')
    inventory[f'{name}/{curve["update"]}/{pop}']=dict(path=str(path),sha256=sha(path));documents[name,curve['update'],pop]=d
 templates={}
 for r in documents['original',4096,'development']['rows']:
  t=r['target'];seq=tuple(k for k,p in zip(t['kind'],t['presence']) if p and k!=1)
  if r['cell'] in templates:require(templates[r['cell']]==seq,'cell has multiple skeletons')
  else:templates[r['cell']]=seq
 events={};summaries={};teacher={}
 for arm in ARMS:
  d=documents[arm,4096,'development'];events[arm]=[describe(r,tokens[r['semantic_sha256']],templates) for r in d['rows']];summaries[arm]={cell:cell_summary([r for r in events[arm] if r['cell']==cell]) for cell in CELLS};teacher[arm]=dict(scope='Gold-prefix field prediction; not free-running graph accuracy.',fields=d['teacher_forced'],loss=d['teacher_forced_loss'])
 before=indexed(events['original']);after=indexed(events['broad']);paired={}
 for cell in CELLS:
  ids=[k for k,v in before.items() if v['cell']==cell];paired[cell]=dict(complete_transitions={f'{i}->{j}':sum(before[k]['complete']==bool(i) and after[k]['complete']==bool(j) for k in ids) for i in (0,1) for j in (0,1)},category_transitions=dict(collections.Counter(before[k]['category']+' -> '+after[k]['category'] for k in ids)),layout_transitions=dict(collections.Counter(str(before[k]['entity_removed_layout_correct'])+' -> '+str(after[k]['entity_removed_layout_correct']) for k in ids)))
 a.output.parent.mkdir(parents=True,exist_ok=True);eventpath=a.output.with_name(a.output.stem+'-events.json.gz')
 with gzip.GzipFile(filename=str(eventpath),mode='wb',mtime=0) as f:f.write(json.dumps(events,separators=(',',':')).encode())
 out=dict(scope='Posthoc fixed-endpoint DEV localization only, after both complete paired arms. Canonical position/layout diagnostics do not establish causal mechanism or graph-isomorphism failure. Invalid prefixes retained; no partial graph credit or new thresholds.',source_sha256=sha(Path(__file__)),config_sha256=a.config_sha256,manifest_sha256=sha(a.main/'manifest.json.gz'),receipt_sha256=sha(a.receipt),artifact_inventory=inventory,summaries=summaries,teacher_forced_separate=teacher,paired=paired,event_artifact=dict(path=str(eventpath),sha256=sha(eventpath)),cpu_wall_seconds=time.monotonic()-tick);a.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('main','receipt','config','development','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--config-sha256',required=True);main(p.parse_args())
