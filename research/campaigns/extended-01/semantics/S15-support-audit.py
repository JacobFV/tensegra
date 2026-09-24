"""Bounded CPU support audit before S15 population counts are frozen."""
import ast,collections,gzip,hashlib,json,re,sys,time,types
from pathlib import Path
package=types.ModuleType('topoformer');package.__path__=[str(Path('src/topoformer').resolve())];sys.modules['topoformer']=package
from topoformer.tcn_data import build_tcn_example,verify_vendor_manifest,SOURCE_COMMIT
from topoformer._vendor.tcn_language.languages import get_language
from topoformer._vendor.tcn_language._structure import Ident
helper=Path('research/campaigns/extended-01/semantics/S14-build-arity3.py')
exec(compile(ast.Module(body=[n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(helper),'exec'))
def alpha(nodes,edges):
 identities={};result=[]
 for kind,value in nodes:
  if kind in ('ident','entity'):
   identities.setdefault(value,len(identities));value=['identity',identities[value]]
  result.append((kind,value))
 return hashlib.sha256(json.dumps([result,sorted(edges)],sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert verify_vendor_manifest()
base=Path('research/results/campaign-01/semantics');excluded=set();sources={}
paths=list((base/'s01-data').glob('*.jsonl.gz'))+list(Path('/home/brandonin/Documents/topoformer/.worktrees/campaign/research/results/campaign-01/semantics/s13-multisurface-cache').glob('*.jsonl.gz'))+[base/'s14-arity3-cache/diagnostic.jsonl.gz']
for p in paths:
 count=0
 for line in gzip.open(p,'rt'):
  row=json.loads(line);excluded.add(alpha(row['nodes'],row['edges']));count+=1
 sources[str(p)]=dict(sha256=sha(p),rows=count)
tick=time.monotonic();seen=set();texts={};cells=collections.defaultdict(collections.Counter);curves=[];accepted=[]
for arity,difficulty,first in [(3,1/3,915100001),(4,.5,915200001)]:
 for attempt in range(10000):
  seed=first+attempt;e=build_tcn_example('unification',seed,difficulty=difficulty,languages=('english',));g=e.privileged.graph;ids={n.id:i for i,n in enumerate(g.nodes)};nodes=[[n.kind,n.value] for n in g.nodes];edges=[[ids[x.source],ids[x.target],x.role,x.slot] for x in g.edges]
  facts=sum(r=='item' for i,j,r,s in edges);cell=f'{arity}x{facts}';c=cells[cell];c['attempts']+=1;identity=alpha(nodes,edges)
  if identity in excluded:c['historical_overlap']+=1;continue
  if identity in seen:c['duplicate']+=1;continue
  seen.add(identity);text=e.public[0].text;tokens=re.findall(r'\w+|[^\w\s]',text,re.UNICODE);forms={}
  for k,v in nodes:
   if k in ('ident','entity'):
    form=get_language('english').render(Ident(v));assert form in tokens and (form not in forms or forms[form]==v);forms[form]=v
   else:assert json.dumps(v,sort_keys=True) in ['"parent"','"unify"','null']
  assert text not in texts or texts[text]==identity;texts[text]=identity
  slots=collections.defaultdict(set)
  for i,j,r,s in edges:
   if s is not None:slots[i,j].add(s)
  assert all(len(v)==1 and 0<=min(v)<=max(v)<=3 for v in slots.values())
  shape,depth=shape_and_depth(nodes,edges);assert depth==3 and len(nodes)<=128
  c['unique']+=1;c['nodes_total']+=len(nodes);c['tokens_total']+=len(tokens)
  accepted.append(dict(seed=seed,arity=arity,facts=facts,alpha_sha256=identity,semantic_sha256=hashlib.sha256(key(g).encode()).hexdigest(),tree_shape_sha256=shape))
  if (attempt+1)%1000==0:curves.append(dict(arity=arity,attempts=attempt+1,cells={k:dict(v) for k,v in cells.items()}))
 print(arity,{k:dict(v) for k,v in cells.items()},flush=True)
out=base/'s15-support-audit';out.mkdir(exist_ok=False)
with gzip.GzipFile(filename=str(out/'accepted-seeds.json.gz'),mode='wb',mtime=0) as f:f.write(json.dumps(accepted,separators=(',',':')).encode())
report=dict(source_sha256=sha(__file__),helper_sha256=sha(helper),vendor_commit=SOURCE_COMMIT,excluded_sources=sources,excluded_alpha_unique=len(excluded),attempts_per_arity=10000,cells={k:dict(v) for k,v in cells.items()},motifs={k:sorted({r['tree_shape_sha256'] for r in accepted if f"{r['arity']}x{r['facts']}"==k}) for k in cells},curves=curves,accepted_seed_sha256=sha(out/'accepted-seeds.json.gz'),finite_space_exhaustion_proven=False,public_text_target_consistency=True,visible_injective_copy_support=True,slot_support_maximum=3,max_depth=3,no_model_import_or_inference=True,cpu_seconds=time.monotonic()-tick)
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['cells']))
