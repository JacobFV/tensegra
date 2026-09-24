"""Read-only final DEV prefix localization; no model, confirmation or recipe selection."""
import collections,gzip,hashlib,json,re,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.load(gzip.open(p,'rt'))
def edit(a,b):
 d=list(range(len(b)+1))
 for i,x in enumerate(a):
  q=[i+1]
  for j,y in enumerate(b):q.append(min(q[-1]+1,d[j+1]+1,d[j]+(x!=y)))
  d=q
 return d[-1]
def main():
 start=time.monotonic();base=ROOT/'research/results/campaign-01/semantics';raw=base/'s19-main/s19-main-v1/development-u4096.json.gz';cache=base/'s15-shape-cache-v2/development.jsonl.gz';train=base/'s15-shape-cache-v2/train_mixed.jsonl.gz'
 rows=load(raw)['rows'];public={r['semantic_sha256']:r for r in map(json.loads,gzip.open(cache,'rt'))};training=list(map(json.loads,gzip.open(train,'rt')))
 kinds=('scope','entity','token','str','num','ident','nil','pred','rel','node','tuple','list','record','app')
 skeletons={};fulltemplates=collections.defaultdict(set)
 for r in training:
  cell=f"{r['arity']}x{r['facts']}";seq=tuple(kinds.index(k) for k,v in r['nodes']);skeletons[cell]=tuple(x for x in seq if x!=1);fulltemplates[cell].add(seq)
 gold34=next(r for r in rows if r['cell']=='3x4')['target'];skeletons['3x4']=tuple(k for k,p in zip(gold34['kind'],gold34['presence']) if p and k!=1)
 out={};details=[]
 for cell in ('3x3','3x4','4x3','4x4'):
  rs=[r for r in rows if r['cell']==cell];c=collections.Counter();hist=collections.defaultdict(collections.Counter)
  for r in rs:
   pub=public[r['semantic_sha256']];tok=re.findall(r'\w+|[^\w\s]',pub['text']);records=r['records'];nodes=[]
   for rec in records:
    if rec[0]!=1:break
    k,v,copy=rec[1:4]
    if 0<=copy<len(tok):copy=tok.index(tok[copy])
    nodes.append((k,v,copy))
   gold=r['target'];n=sum(gold['presence']);gn=list(zip(gold['kind'][:n],gold['value'][:n],gold['copy'][:n]));seq=tuple(x[0] for x in nodes);sk=tuple(x for x in seq if x!=1)
   c['examples']+=1;c['valid']+=r['valid'];c['complete']+=r['complete'];c['eos_present']+=any(x[0]==3 for x in records);c['node_count_correct']+=len(nodes)==n;c['all_node_records_correct']+=nodes==gn;c['kind_sequence_correct']+=seq==tuple(x[0] for x in gn)
   c['nodes_correct_but_graph_wrong']+=(nodes==gn and not r['complete']);c['nodes_correct_but_invalid']+=(nodes==gn and not r['valid'])
   if nodes==gn and r['valid']:c['nodes_correct_edges_wrong']+=not r['exact_components']['edges'];c['nodes_correct_slots_wrong']+=not r['exact_components']['slots']
   first=next((i for i in range(max(len(nodes),len(gn))) if i>=len(nodes) or i>=len(gn) or nodes[i]!=gn[i]),None)
   mismatch=[]
   if first is not None:
    if first>=len(nodes):mismatch=['missing_node']
    elif first>=len(gn):mismatch=['extra_node']
    else:mismatch=[name for j,name in enumerate(('kind','value','copy')) if nodes[first][j]!=gn[first][j]]
   hist['first_mismatch_kind_pair'][str((None if first is None or first>=len(nodes) else nodes[first][0],None if first is None or first>=len(gn) else gn[first][0]))]+=1;hist['first_mismatch_index'][str(first)]+=1;hist['first_mismatch_fields']['+'.join(mismatch) or 'none']+=1;hist['generated_node_count'][str(len(nodes))]+=1;hist['gold_node_count'][str(n)]+=1;hist['reason'][r['reason'] or 'valid']+=1;hist['public_tokens'][str(len(tok))]+=1
   matching=[k for k,s in skeletons.items() if sk==s];hist['entity_removed_kind_template'][','.join(matching) or 'other']+=1
   full=[k for k,s in fulltemplates.items() if seq in s];hist['full_kind_template_membership'][','.join(full) or 'none']+=1
   dist={k:edit(sk,s) for k,s in skeletons.items()};hist['distance_4x3_minus_3x4'][str(dist['4x3']-dist['3x4'])]+=1
   # Generated-only semantic structural signature: outgoing argument cardinality of generated pred nodes.
   arg=collections.Counter(x[1] for x in records if x[0]==2 and x[3]==3);hist['pred_argument_counts'][str([arg[i] for i,x in enumerate(nodes) if x[0]==7])]+=1
   details.append(dict(semantic_sha256=r['semantic_sha256'],cell=cell,valid=r['valid'],complete=r['complete'],reason=r['reason'],generated_nodes=len(nodes),gold_nodes=n,all_nodes_correct=nodes==gn,first_mismatch=first,first_fields=mismatch,first_predicted=None if first is None or first>=len(nodes) else nodes[first],first_gold=None if first is None or first>=len(gn) else gn[first],entity_removed_template=matching,template_edit_distances=dist))
  out[cell]=dict(counts=dict(c),histograms={k:dict(v) for k,v in hist.items()})
 result=dict(scope='Final inspected DEV only, all generated prefixes including invalids. Canonical-position diagnostics do not establish graph-isomorphism failure causes. Entity-removal is a diagnostic signature only, never relaxed scoring.',source_sha256=sha(Path(__file__)),inputs={str(p.relative_to(ROOT)):sha(p) for p in (raw,cache,train)},template_kind_sequences={k:[kinds[i] for i in v] for k,v in skeletons.items()},cells=out,cpu_wall_seconds=time.monotonic()-start)
 events=base/'s19-heldout-localization-events.json.gz'
 with gzip.GzipFile(filename=str(events),mode='wb',mtime=0) as f:f.write(json.dumps(details,separators=(',',':')).encode())
 result['event_artifact']=dict(path=str(events.relative_to(ROOT)),sha256=sha(events),count=len(details))
 result['cpu_wall_seconds']=time.monotonic()-start
 dest=base/'s19-heldout-localization.json';dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(cpu_wall_seconds=result['cpu_wall_seconds'],cells=out),indent=2))
if __name__=='__main__':main()
