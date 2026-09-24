"""Versioned bounded CPU diagnostic builder; pinned generator remains unchanged."""
import ast,collections,gzip,hashlib,json,re,sys,time,types
from pathlib import Path
package=types.ModuleType('topoformer');package.__path__=[str(Path('src/topoformer').resolve())];sys.modules['topoformer']=package
from topoformer.tcn_data import build_tcn_example,verify_vendor_manifest,SOURCE_COMMIT
from topoformer._vendor.tcn_language.languages import get_language
from topoformer._vendor.tcn_language._structure import Ident

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def shape_and_depth(nodes,edges):
    kinds=[k for k,v in nodes];kept=[i for i,k in enumerate(kinds) if k not in ('entity','scope')];index={i:j for j,i in enumerate(kept)}
    edges=[(i,j,r,s) for i,j,r,s in edges if r in ('argument','item') or r.startswith('field:')]
    tree=json.dumps([[kinds[i] for i in kept],[(index[i],index[j],r,s) for i,j,r,s in edges]],separators=(',',':'))
    incoming={j:i for i,j,_,_ in edges};depths=[]
    for i in kept:
        depth=0
        while i in incoming:depth+=1;i=incoming[i]
        depths.append(depth)
    return hashlib.sha256(tree.encode()).hexdigest(),max(depths)

def key(graph):
    # Exact stdlib implementation of historical semantic_scaling.semantic_key.
    ids={n.id:i for i,n in enumerate(graph.nodes)};identities={};nodes=[]
    for n in graph.nodes:
        value=n.value
        if n.kind in ('ident','entity'):
            identities.setdefault(value,len(identities));value=['identity',identities[value]]
        nodes.append((n.kind,value))
    return json.dumps([nodes,sorted((ids[e.source],ids[e.target],e.role,e.slot) for e in graph.edges),[ids[r] for r in graph.roots]],sort_keys=True,separators=(',',':'))

config_path=Path(sys.argv[1]);c=json.loads(config_path.read_text());tick=time.monotonic()
assert c['generation_status']=='root_authorized_cpu_only' and c['source_commit']==SOURCE_COMMIT and verify_vendor_manifest()
assert (c['difficulty'],c['requested_unique'],c['max_attempts'])==(1/3,1024,10000)
out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=False)
constants={}
for node in ast.parse(Path('src/topoformer/thinking_language.py').read_text()).body:
    if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('KINDS','ROLES'):
        constants[node.targets[0].id]=ast.literal_eval(node.value)
excluded=set();excluded_sources={};known_shapes=set()
for split,expected in c['historical_cache_sha256'].items():
    p=Path(c['data_dir'])/(split+'.jsonl.gz');assert sha(p)==expected
    rows=[json.loads(line) for line in gzip.open(p,'rt')];excluded.update(r['semantic_sha256'] for r in rows);excluded_sources[str(p)]=dict(sha256=sha(p),rows=len(rows))
    if split=='train':known_shapes.update(shape_and_depth(r['nodes'],r['edges'])[0] for r in rows)
s13=Path(c['s13_data_dir']);audit=json.loads((s13/'audit.json').read_text())
for split,expected in audit['cache_sha256'].items():
    p=s13/(split+'.jsonl.gz');assert sha(p)==expected
    rows=[json.loads(line) for line in gzip.open(p,'rt')];excluded.update(r['semantic_sha256'] for r in rows);excluded_sources[str(p)]=dict(sha256=sha(p),rows=len(rows))
records=[];seen=set();seen_public={};duplicates=0;overlap=0;curve=[]
for attempt in range(c['max_attempts']):
    seed=c['first_seed']+attempt;e=build_tcn_example('unification',seed,difficulty=c['difficulty'],languages=('english',));g=e.privileged.graph
    semantic=key(g);identity=hashlib.sha256(semantic.encode()).hexdigest()
    if identity in excluded:overlap+=1;continue
    if semantic in seen:duplicates+=1;continue
    seen.add(semantic);positions={n.id:i for i,n in enumerate(g.nodes)};nodes=[[n.kind,n.value] for n in g.nodes];edges=[[positions[x.source],positions[x.target],x.role,x.slot] for x in g.edges]
    shape,depth=shape_and_depth(nodes,edges);assert shape not in known_shapes and depth==3
    text=e.public[0].text;tokens=re.findall(r'\w+|[^\w\s]',text,re.UNICODE)
    assert len(nodes)<=128 and all(k in constants['KINDS'] for k,v in nodes) and all(r in constants['ROLES'] for i,j,r,s in edges)
    copies={};forms={}
    for kind,value in nodes:
        if kind in ('ident','entity'):
            form=get_language('english').render(Ident(value));assert form in tokens
            assert form not in forms or forms[form]==value
            forms[form]=value;copies[value]=tokens.index(form)
        else:assert json.dumps(value,sort_keys=True) in c['value_vocabulary'][1:]
    slots=collections.defaultdict(set)
    for i,j,r,s in edges:
        if s is not None:slots[i,j].add(s)
    assert all(len(s)==1 and 0<=min(s)<=max(s)<32 for s in slots.values())
    target_json=json.dumps([nodes,edges],separators=(',',':'))
    assert text not in seen_public or seen_public[text]==target_json
    seen_public[text]=target_json
    records.append(dict(seed=seed,text=text,graph_sha256=g.digest(),semantic_sha256=identity,public_sha256=hashlib.sha256(text.encode()).hexdigest(),nodes=nodes,edges=edges,tokens=len(tokens),tree_shape_sha256=shape,structural_depth=depth,english_first_copy_positions=copies))
    if len(records)%128==0:curve.append(dict(attempts=attempt+1,accepted=len(records),duplicates=duplicates))
    if len(records)==c['requested_unique']:break
path=out/'diagnostic.jsonl.gz'
with gzip.GzipFile(filename=str(path),mode='wb',mtime=0) as f:
    for r in records:f.write((json.dumps(r,separators=(',',':'))+'\n').encode())
paths=[Path(__file__),config_path,Path('src/topoformer/tcn_data.py'),Path('src/topoformer/semantic_graph.py'),Path('src/topoformer/semantic_scaling.py'),Path('src/topoformer/thinking_language.py'),Path('src/topoformer/_vendor/tcn_language/context.py'),Path('src/topoformer/_vendor/tcn_language/lessons/unification.py')]
result=dict(config=c,config_sha256=sha(config_path),source_sha256={str(p):sha(p) for p in paths},excluded_sources=excluded_sources,excluded_unique_semantics=len(excluded),actual_unique=len(records),attempts=attempt+1,duplicates=duplicates,historical_or_s13_overlap=overlap,attempt_curve=curve,requested_reached=len(records)==1024,finite_space_exhaustion_proven=False,restricted_population_warning=len(records)<512,node_counts=dict(collections.Counter(len(r['nodes']) for r in records)),token_counts=dict(collections.Counter(r['tokens'] for r in records)),tree_shape_counts=dict(collections.Counter(r['tree_shape_sha256'] for r in records)),depths=dict(collections.Counter(r['structural_depth'] for r in records)),all_motifs_unseen_in_historical_train=True,all_targets_within_fixed_schema_and_capacity=True,all_copy_identities_visibly_injective=True,all_single_slot_pairs=True,public_target_consistency_passed=True,no_model_import_or_inference=True,cache_sha256=sha(path),cpu_seconds=time.monotonic()-tick)
(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('actual_unique','attempts','duplicates','historical_or_s13_overlap','tree_shape_counts','cpu_seconds')}))
