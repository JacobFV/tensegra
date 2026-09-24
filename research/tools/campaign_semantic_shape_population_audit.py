"""Independent S15 v2 populations, ordering, generator and numerical target support."""
import argparse,ast,collections,gzip,hashlib,json,math,random,re,sys,time,types
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--packing',type=Path,required=True);a=p.parse_args();t=time.monotonic();root=a.repo/'research/results/campaign-01/semantics';cache=root/'s15-shape-cache-v2';meta=json.loads((cache/'audit.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:[json.loads(s) for s in gzip.open(p,'rt')];package=types.ModuleType('topoformer');package.__path__=[str(a.repo/'src/topoformer')];sys.modules['topoformer']=package
from topoformer.tcn_data import build_tcn_example,verify_vendor_manifest
assert verify_vendor_manifest();constants={}
for n in ast.parse((a.repo/'src/topoformer/thinking_language.py').read_text()).body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('KINDS','ROLES'):constants[n.targets[0].id]=ast.literal_eval(n.value)
vocab=json.loads((root/'s14-arity3-cache/audit.json').read_text())['config']['value_vocabulary']
def alpha(nodes,edges):
 ids={};labels=[]
 for k,v in nodes:
  if k in ('ident','entity'):ids.setdefault(v,len(ids));v=['identity',ids[v]]
  labels.append([k,v])
 return hashlib.sha256(json.dumps([labels,sorted(edges)],sort_keys=True,separators=(',',':')).encode()).hexdigest()
def shape(nodes,edges):
 kept=[i for i,(k,v) in enumerate(nodes) if k not in ('scope','entity')];ix={i:j for j,i in enumerate(kept)};struct=[e for e in edges if e[2] in ('argument','item') or e[2].startswith('field:')];parents={e[1]:e[0] for e in struct};depth=0
 for i in kept:
  seen=set()
  while i in parents:assert i not in seen;seen.add(i);i=parents[i]
  depth=max(depth,len(seen))
 payload=[[nodes[i][0] for i in kept],[(ix[i],ix[j],r,s) for i,j,r,s in struct]];return hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest(),depth
old=[]
for name,info in meta['excluded_sources'].items():
 path=Path(name) if name.startswith('/') else a.repo/name;assert sha(path)==info['sha256'];v=read(path);assert len(v)==info['rows'];old+=v
excluded={alpha(r['nodes'],r['edges']) for r in old};data={k:read(cache/(k+'.jsonl.gz')) for k in meta['cache_sha256']}
for k,h in meta['cache_sha256'].items():assert sha(cache/(k+'.jsonl.gz'))==h
sets={k:{alpha(r['nodes'],r['edges']) for r in rows} for k,rows in data.items()}
for k,rows in data.items():assert len(sets[k])==len(rows);assert not sets[k]&excluded
for k,one in sets.items():
 for j,two in sets.items():
  if k<j:assert len(one&two)==(2048 if {k,j}=={'train_control','train_mixed'} else 0)
assert data['confirmation']==read(root/'s15-shape-cache/confirmation.jsonl.gz');assert data['development'][:1024]==read(root/'s15-shape-cache/development.jsonl.gz')
# Independently replay the bounded candidate generator, including rejects and order.
support=json.loads((root/'s15-support-audit/audit.json').read_text());accepted=json.load(gzip.open(root/'s15-support-audit/accepted-seeds.json.gz','rt'));observed=[];seen=set();generated={};stats=collections.defaultdict(collections.Counter)
for arity in (3,4):
 for seed in range(915000001+(arity-2)*100000,915000001+(arity-2)*100000+10000):
  e=build_tcn_example('unification',seed,difficulty=1/3 if arity==3 else .5,languages=('english',));g=e.privileged.graph;ids={n.id:i for i,n in enumerate(g.nodes)};nodes=[[n.kind,n.value] for n in g.nodes];edges=[[ids[x.source],ids[x.target],x.role,x.slot] for x in g.edges];key=alpha(nodes,edges);facts=sum(r=='item' for i,j,r,s in edges);cell=f'{arity}x{facts}';stats[cell]['attempts']+=1
  if key in excluded:stats[cell]['historical_overlap']+=1;continue
  if key in seen:stats[cell]['duplicate']+=1;continue
  seen.add(key);s,d=shape(nodes,edges);assert d==3;stats[cell]['unique']+=1;stats[cell]['nodes_total']+=len(nodes);stats[cell]['tokens_total']+=len(re.findall(r'\w+|[^\w\s]',e.public[0].text));observed.append((seed,arity,facts,key,s));generated[seed]=(nodes,edges,e.public[0].text,g.digest())
assert observed==[(r['seed'],r['arity'],r['facts'],r['alpha_sha256'],r['tree_shape_sha256']) for r in accepted];assert {k:dict(v) for k,v in stats.items()}==support['cells']
reserved=sets['development']|sets['confirmation'];selected=collections.defaultdict(list)
for seed,arity,facts,key,s in observed:
 if (arity,facts) not in ((3,3),(4,3),(4,4)) or len(selected[arity,facts])>=2048 or key in reserved:continue
 selected[arity,facts].append(seed)
common=selected[4,3][:1024]+selected[4,4][:1024];replacement=selected[4,3][1024:]+selected[4,4][1024:];order=list(range(4096));random.Random(15015).shuffle(order)
assert [r['seed'] for r in data['train_control']]==[(common+replacement)[i] for i in order];assert [r['seed'] for r in data['train_mixed']]==[(common+selected[3,3])[i] for i in order]
# Reconstruct first-qualifying DEV/confirmation reservations, including skipped candidates.
reservation_seen=set(excluded);reservation_checks=0
for split,start,quota,expected_rows in [('development',916000001,256,data['development'][:1024]),('confirmation',917000001,512,data['confirmation']),('development-extension',918000001,256,data['development'][1024:])]:
 picked=[]
 for arity in (3,4):
  counts=collections.Counter()
  for attempt in range(10000):
   seed=start+(arity-2)*100000+attempt;e=build_tcn_example('unification',seed,difficulty=1/3 if arity==3 else .5,languages=('english',));g=e.privileged.graph;ids={n.id:i for i,n in enumerate(g.nodes)};nodes=[[n.kind,n.value] for n in g.nodes];edges=[[ids[z.source],ids[z.target],z.role,z.slot] for z in g.edges];key=alpha(nodes,edges);facts=sum(role=='item' for i,j,role,slot in edges);reservation_checks+=1
   if key in reservation_seen or counts[facts]>=quota:continue
   reservation_seen.add(key);picked.append(seed);counts[facts]+=1;generated[seed]=(nodes,edges,e.public[0].text,g.digest())
   if counts[3]==counts[4]==quota:break
  assert counts[3]==counts[4]==quota
 assert picked==[r['seed'] for r in expected_rows]
lexical={};features_seen={};public_seen={};packing={};summary={}
def digest(arrays):
 h=hashlib.sha256()
 for name,v in sorted(arrays.items()):h.update(name.encode());h.update(str((str(v.dtype),v.shape)).encode());h.update(v.tobytes())
 return h.hexdigest()
for split,rows in data.items():
 for r in rows:
  if str(r['seed']) in packing:continue
  if r['seed'] in generated:nodes,edges,text,gh=generated[r['seed']]
  else:
   e=build_tcn_example('unification',r['seed'],difficulty=1/3 if r['arity']==3 else .5,languages=('english',));g=e.privileged.graph;ids={n.id:i for i,n in enumerate(g.nodes)};nodes=[[n.kind,n.value] for n in g.nodes];edges=[[ids[z.source],ids[z.target],z.role,z.slot] for z in g.edges];text=e.public[0].text;gh=g.digest()
  assert nodes==r['nodes'] and edges==r['edges'] and text==r['text'] and gh==r['graph_sha256'];assert alpha(nodes,edges)==r['alpha_sha256'];s,d=shape(nodes,edges);assert s==r['tree_shape_sha256'] and d==r['structural_depth']==3;assert sum(role=='item' for i,j,role,slot in edges)==r['facts'];arities=collections.Counter(i for i,j,role,slot in edges if role=='argument' and nodes[i]==['pred','parent']);assert set(arities.values())=={r['arity']}
  tok=re.findall(r'\w+|[^\w\s]',text);assert len(tok)==r['tokens'] and hashlib.sha256(text.encode()).hexdigest()==r['public_sha256'];assert len(nodes)<=128;target=dict(presence=np.arange(128)<len(nodes),kind=np.zeros(128,np.int64),value=np.full(128,-1,np.int64),copy=np.full(128,-1,np.int64),slots=np.full((128,128),-1,np.int64),edges=np.zeros((128,128,13),bool));copy={}
  for i,(k,v) in enumerate(nodes):
   target['kind'][i]=constants['KINDS'].index(k)
   if k in ('ident','entity'):target['copy'][i]=tok.index(v);copy[v]=tok.index(v)
   else:target['value'][i]=vocab.index(json.dumps(v,sort_keys=True))
  assert copy==r['english_first_copy_positions'] and len(set(copy.values()))==len(copy)
  for i,j,role,slot in edges:
   target['edges'][i,j,constants['ROLES'].index(role)]=True
   if slot is not None:assert 0<=slot<=3 and target['slots'][i,j] in (-1,slot);target['slots'][i,j]=slot
  f=[]
  for i,v in enumerate(tok):
   bits=hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True).encode()).digest()[:8];assert bits not in lexical or lexical[bits]==v;lexical[bits]=v;f.append([float((b>>bit)&1) for b in bits for bit in range(8)]+[i/len(tok),math.sin(i),math.cos(i),0.])
  f=np.asarray(f,np.float32)[None];u=f.view(np.uint32);bf=((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint16);td=digest(target)
  for precision,v in [('fp32',f),('bf16',bf)]:
   h=precision+hashlib.sha256(v.tobytes()).hexdigest();assert h not in features_seen or features_seen[h]==td;features_seen[h]=td
  assert text not in public_seen or public_seen[text]==td;public_seen[text]=td;packing[str(r['seed'])]=dict(target_sha256=td,fp32_sha256=hashlib.sha256(f.tobytes()).hexdigest(),bf16_sha256=hashlib.sha256(bf.tobytes()).hexdigest())
 summary[split]=dict(rows=len(rows),cells=dict(collections.Counter(f"{r['arity']}x{r['facts']}" for r in rows)),motifs=dict(collections.Counter(r['tree_shape_sha256'] for r in rows)),tokens=sum(r['tokens'] for r in rows),nodes=sum(len(r['nodes']) for r in rows));assert summary[split]==meta['summaries'][split]
assert set(summary['train_mixed']['cells'])=={'4x3','4x4','3x3'} and len(summary['train_control']['motifs'])==2 and len(summary['train_mixed']['motifs'])==3;assert set(lexical.values()).issubset({v for r in old for v in re.findall(r'\w+|[^\w\s]',r['text'])});assert 'torch' not in sys.modules
a.packing.write_text(json.dumps(packing,separators=(',',':'))+'\n');result=dict(bounded_support_candidates_replayed=20000,accepted_candidates=len(observed),reservation_candidates_replayed=reservation_checks,unique_packed_examples=len(packing),alpha_exclusion_and_split_disjointness=True,intended_common_examples=2048,common_order_verified=True,training_selection_replayed=True,confirmation_v1_bytes_unchanged=sha(cache/'confirmation.jsonl.gz')==sha(root/'s15-shape-cache/confirmation.jsonl.gz'),all_generator_targets_replayed=True,fp32_bf16_and_public_target_conflicts=0,lexical_types=len(lexical),summaries=summary,packing_expected_sha256=sha(a.packing),input_sha256={str(p.relative_to(a.repo)):sha(p) for p in cache.iterdir() if p.is_file()},cpu_audit_wall_seconds=time.monotonic()-t,scope='CPU generator/public/target support only; no actor/import/forward. Mixed trained arity3/threefacts is acquisition; heldout arity3/fourfacts is factor recombination at unchanged depth3. Counts and tokens/nodes differ explicitly; prospective profile/source/negative-RNG review remains separate.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ('input_sha256','summaries')})
