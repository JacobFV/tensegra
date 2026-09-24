"""Deterministic CPU S15 split reservation and cache construction, no model import."""
from pathlib import Path
support_source=Path('research/campaigns/extended-01/semantics/S15-support-audit.py')
# Reuse the exact audited stdlib graph/key/exclusion helpers, stopping before generation.
exec(compile(support_source.read_text().split('tick=time.monotonic();')[0],str(support_source),'exec'))
import random
out=base/'s15-shape-cache';assert not out.exists();tick=time.monotonic();seen=set(excluded);texts={};records={};stats={}

def record(arity,seed):
 e=build_tcn_example('unification',seed,difficulty=1/3 if arity==3 else .5,languages=('english',));g=e.privileged.graph;ids={n.id:i for i,n in enumerate(g.nodes)};nodes=[[n.kind,n.value] for n in g.nodes];edges=[[ids[x.source],ids[x.target],x.role,x.slot] for x in g.edges];identity=alpha(nodes,edges)
 if identity in seen:return None
 facts=sum(r=='item' for i,j,r,s in edges);text=e.public[0].text;tokens=re.findall(r'\w+|[^\w\s]',text,re.UNICODE);forms={};copies={}
 for k,v in nodes:
  if k in ('ident','entity'):
   form=get_language('english').render(Ident(v));assert form in tokens and (form not in forms or forms[form]==v);forms[form]=v;copies[v]=tokens.index(form)
  else:assert json.dumps(v,sort_keys=True) in ['"parent"','"unify"','null']
 assert text not in texts or texts[text]==identity
 slots=collections.defaultdict(set)
 for i,j,r,s in edges:
  if s is not None:slots[i,j].add(s)
 assert all(len(v)==1 and 0<=min(v)<=max(v)<=3 for v in slots.values())
 shape,depth=shape_and_depth(nodes,edges);assert depth==3 and len(nodes)<=128
 return dict(seed=seed,text=text,graph_sha256=g.digest(),semantic_sha256=hashlib.sha256(key(g).encode()).hexdigest(),alpha_sha256=identity,public_sha256=hashlib.sha256(text.encode()).hexdigest(),nodes=nodes,edges=edges,tokens=len(tokens),tree_shape_sha256=shape,structural_depth=depth,english_first_copy_positions=copies,arity=arity,facts=facts)

def accept(row):
 seen.add(row['alpha_sha256']);texts[row['text']]=row['alpha_sha256']
for split,start,quota in [('development',916000001,256),('confirmation',917000001,512)]:
 selected=[]
 for arity in (3,4):
  cells=collections.Counter()
  for attempt in range(10000):
   row=record(arity,start+(arity-2)*100000+attempt)
   if row is None or cells[row['facts']]>=quota:continue
   accept(row);selected.append(row);cells[row['facts']]+=1
   if cells[3]==cells[4]==quota:break
  assert cells[3]==cells[4]==quota,(split,arity,dict(cells))
  stats[f'{split}-{arity}']=dict(attempts=attempt+1,selected=dict(cells))
 records[split]=selected
 print(split,len(selected),flush=True)
candidates=json.load(gzip.open(base/'s15-support-audit/accepted-seeds.json.gz','rt'));selected=collections.defaultdict(list)
for candidate in candidates:
 cell=(candidate['arity'],candidate['facts'])
 if cell not in ((3,3),(4,3),(4,4)) or len(selected[cell])>=2048:continue
 row=record(candidate['arity'],candidate['seed'])
 if row is not None:accept(row);selected[cell].append(row)
assert all(len(selected[c])==2048 for c in ((3,3),(4,3),(4,4))),{str(c):len(v) for c,v in selected.items()}
common=selected[4,3][:1024]+selected[4,4][:1024];control=selected[4,3][1024:]+selected[4,4][1024:];mixed=selected[3,3]
order=list(range(4096));random.Random(15015).shuffle(order)
records['train_control']=[(common+control)[i] for i in order];records['train_mixed']=[(common+mixed)[i] for i in order]
out.mkdir();hashes={};summaries={}
for name,rows in records.items():
 p=out/(name+'.jsonl.gz')
 with gzip.GzipFile(filename=str(p),mode='wb',mtime=0) as f:
  for row in rows:f.write((json.dumps(row,separators=(',',':'))+'\n').encode())
 hashes[name]=sha(p);summaries[name]=dict(rows=len(rows),cells=dict(collections.Counter(f"{r['arity']}x{r['facts']}" for r in rows)),motifs=dict(collections.Counter(r['tree_shape_sha256'] for r in rows)),tokens=sum(r['tokens'] for r in rows),nodes=sum(len(r['nodes']) for r in rows))
report=dict(source_sha256=sha(__file__),support_source_sha256=sha(support_source),protocol_sha256=sha('research/campaigns/extended-01/semantics/S15-protocol.md'),support_audit_sha256=sha(base/'s15-support-audit/audit.json'),cache_sha256=hashes,summaries=summaries,reservation_statistics=stats,excluded_sources=sources,train_common_count=2048,common_index_alignment=all(a['alpha_sha256']==b['alpha_sha256'] for a,b in zip(records['train_control'],records['train_mixed']) if a['alpha_sha256'] in {r['alpha_sha256'] for r in common}),public_text_target_consistency=True,alpha_disjoint_except_intended_common=True,independent_packing_audit_pending=True,no_model_import_or_inference=True,cpu_seconds=time.monotonic()-tick)
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summaries),flush=True)
