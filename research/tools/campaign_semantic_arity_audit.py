"""Independent S14 replay, structural support, surface-target and feature audit."""
import argparse,ast,collections,gzip,hashlib,json,math,re,sys,time,types
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic();root=a.repo/'research/results/campaign-01/semantics';cache=root/'s14-arity3-cache';meta=json.loads((cache/'audit.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:[json.loads(s) for s in gzip.open(p,'rt')]
package=types.ModuleType('topoformer');package.__path__=[str(a.repo/'src/topoformer')];sys.modules['topoformer']=package
from topoformer.tcn_data import build_tcn_example,verify_vendor_manifest
assert verify_vendor_manifest();tree=ast.parse((a.repo/'src/topoformer/semantic_scaling.py').read_text());func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='semantic_key');ns={'json':json};exec(compile(ast.Module(body=[func],type_ignores=[]),'<historical key>','exec'),ns)
constants={}
for n in ast.parse((a.repo/'src/topoformer/thinking_language.py').read_text()).body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('KINDS','ROLES'):constants[n.targets[0].id]=ast.literal_eval(n.value)
def alpha(row):
 ids={};nodes=[]
 for k,v in row['nodes']:
  if k in ('ident','entity'):ids.setdefault(v,len(ids));v=['identity',ids[v]]
  nodes.append([k,v])
 return json.dumps([nodes,sorted(row['edges'],key=lambda e:(e[0],e[1],e[2],-1 if e[3] is None else e[3]))],sort_keys=True)
def shape(row):
 kept=[i for i,(k,v) in enumerate(row['nodes']) if k not in ('scope','entity')];ix={i:j for j,i in enumerate(kept)};edges=[e for e in row['edges'] if e[2] in ('argument','item') or e[2].startswith('field:')];parents={e[1]:e[0] for e in edges};depths=[]
 for i in kept:
  seen=set()
  while i in parents:assert i not in seen;seen.add(i);i=parents[i]
  depths.append(len(seen))
 payload=[[row['nodes'][i][0] for i in kept],[(ix[i],ix[j],rel,slot) for i,j,rel,slot in edges]]
 return sha(json.dumps(payload,separators=(',',':')).encode()),max(depths)
old=[];oldtrain=[]
for split,h in meta['config']['historical_cache_sha256'].items():
 f=root/'s01-data'/(split+'.jsonl.gz');assert sha(f.read_bytes())==h;rs=read(f);old+=rs
 if split=='train':oldtrain=rs
s13=Path(meta['config']['s13_data_dir']);s13meta=json.loads((s13/'audit.json').read_text())
for split,h in s13meta['cache_sha256'].items():
 f=s13/(split+'.jsonl.gz');assert sha(f.read_bytes())==h;old+=read(f)
excluded={alpha(r) for r in old};known_shapes={shape(r)[0] for r in oldtrain};rows=read(cache/'diagnostic.jsonl.gz');assert sha((cache/'diagnostic.jsonl.gz').read_bytes())==meta['cache_sha256'];assert len(rows)==1024
seen=set();shapes=collections.Counter();feature_seen={};public_seen={};alltokens={};targets=[];maxarity=collections.Counter()
def support(r):
 tok=re.findall(r'\w+|[^\w\s]',r['text'],re.UNICODE);assert len(tok)==r['tokens'] and sha(r['text'].encode())==r['public_sha256'];pos={v:tok.index(v) for k,v in r['nodes'] if k in ('ident','entity')};assert len(set(pos.values()))==len(pos);slots={};labels=[]
 for k,v in r['nodes']:
  assert k in constants['KINDS'];labels.append([constants['KINDS'].index(k),-1 if k in ('ident','entity') else meta['config']['value_vocabulary'].index(json.dumps(v,sort_keys=True)),pos[v] if k in ('ident','entity') else -1])
 for i,j,rel,slot in r['edges']:
  assert rel in constants['ROLES'] and 0<=i<len(labels) and 0<=j<len(labels);value=-1 if slot is None else slot;assert -1<=value<32;assert (i,j) not in slots or slots[i,j]==value;slots[i,j]=value
 assert len(labels)<=128;target=sha(json.dumps([labels,sorted(r['edges'],key=lambda e:(e[0],e[1],e[2],-1 if e[3] is None else e[3]))],separators=(',',':')).encode());assert r['text'] not in public_seen or public_seen[r['text']]==target;public_seen[r['text']]=target
 f=[]
 for i,tokword in enumerate(tok):
  bits=hashlib.sha256(json.dumps(tokword,ensure_ascii=False,sort_keys=True).encode()).digest()[:8];assert bits not in alltokens or alltokens[bits]==tokword;alltokens[bits]=tokword;f.append([float((b>>n)&1) for b in bits for n in range(8)]+[i/len(tok),math.sin(i),math.cos(i),0.])
 f=np.asarray(f,dtype=np.float32)[None];u=f.view(np.uint32);bf=((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint16)
 for precision,arr in [('float32',f),('bfloat16',bf)]:
  h=precision+sha(arr.tobytes());assert h not in feature_seen or feature_seen[h]==target;feature_seen[h]=target
 return pos,target
# Old English and S13 English surfaces check feature identifiability across all excluded populations too.
for r in old:support(r)
for r in rows:
 key=alpha(r);assert key not in excluded and key not in seen;seen.add(key);s,d=shape(r);assert s==r['tree_shape_sha256'] and s not in known_shapes and d==r['structural_depth']==3;shapes[s]+=1;pos,target=support(r);assert pos==r['english_first_copy_positions'];targets.append(target)
 arities=collections.Counter(i for i,j,rel,slot in r['edges'] if rel=='argument' and r['nodes'][i]==['pred','parent']);assert set(arities.values())=={3};maxarity[3]+=1
 e=build_tcn_example('unification',r['seed'],difficulty=1/3,languages=('english',));g=e.privileged.graph;ids={n.id:i for i,n in enumerate(g.nodes)};assert [[n.kind,n.value] for n in g.nodes]==r['nodes'];assert [[ids[x.source],ids[x.target],x.role,x.slot] for x in g.edges]==r['edges'];assert e.public[0].text==r['text'];assert g.digest()==r['graph_sha256'];assert sha(ns['semantic_key'](g).encode())==r['semantic_sha256']
assert dict(shapes)==meta['tree_shape_counts'];assert 'torch' not in sys.modules
byseed={r['seed']:r for r in rows};previous=set();duplicates=0
for seed in range(meta['config']['first_seed'],max(byseed)+1):
 if seed in byseed:
  identity=byseed[seed]['semantic_sha256'];assert identity not in previous;previous.add(identity)
 else:
  graph=build_tcn_example('unification',seed,difficulty=1/3,languages=('english',)).privileged.graph
  assert sha(ns['semantic_key'](graph).encode()) in previous;duplicates+=1
assert duplicates==meta['duplicates']==15
assert max(byseed)-meta['config']['first_seed']+1==meta['attempts']==1039
out=dict(bounded_attempts_reconstructed=1039,rejected_previous_duplicates_replayed=15,cache_sha256=meta['cache_sha256'],examples=1024,historical_key_and_generator_replays=1024,alpha_overlap_all_excluded=0,unique_without_roots=1024,shapes=dict(shapes),depth=3,all_parent_predicate_arities=3,independent_copy_targets=True,single_slot_pairs=True,feature_precisions=['float32','bfloat16'],feature_and_lexical_conflicts=0,target_sequence_sha256=sha(''.join(targets).encode()),torch_imported=False,cpu_audit_wall_seconds=time.monotonic()-tick,scope='Pinned-generator replay plus independent compact target/copy, alpha-without-roots exclusion, topology/depth and NumPy FP32/BF16 support checks. Features checked against excluded English populations. No actor/model/forward; smaller unseen arity motifs, not deeper or language transfer.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
