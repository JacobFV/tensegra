"""Bounded CPU-only source feasibility: no cache generation or actor import."""
import collections, hashlib, json, re, time, sys, types
from pathlib import Path
# Avoid package __init__ importing torch attention; this diagnostic uses only stdlib data/compiler code.
if 'topoformer' not in sys.modules:
    package=types.ModuleType('topoformer');package.__path__=[str(Path('src/topoformer').resolve())]
    sys.modules['topoformer']=package
from topoformer.tcn_data import build_tcn_example, verify_vendor_manifest, SOURCE_COMMIT
from topoformer._vendor.tcn_language.context import GenerationContext
from topoformer._vendor.tcn_language.languages import get_language
from topoformer._vendor.tcn_language._structure import Ident

start=time.monotonic()
assert verify_vendor_manifest()
historical=json.loads(Path('research/results/campaign-01/semantics/s07-diversity-audit.json').read_text())
known=set(historical['splits']['train']['tree_shape_counts'])
vocab={'"parent"','"unify"','null'}
records=[]
for difficulty in (0., 1/3, .5, 1.):
    arity=GenerationContext(difficulty=difficulty).at(2,5,default=2)
    for seed in range(914000001,914000017):
        ex=build_tcn_example('unification',seed,difficulty=difficulty,languages=('english',))
        graph=ex.privileged.graph; ids={n.id:i for i,n in enumerate(graph.nodes)}
        kinds=[n.kind for n in graph.nodes]
        structural=[(ids[e.source],ids[e.target],e.role,e.slot) for e in graph.edges if e.role in ('argument','item') or e.role.startswith('field:')]
        kept=[i for i,k in enumerate(kinds) if k not in ('scope','entity')];index={i:j for j,i in enumerate(kept)}
        tree=json.dumps([[kinds[i] for i in kept],[(index[i],index[j],r,s) for i,j,r,s in structural]],separators=(',',':'))
        shape=hashlib.sha256(tree.encode()).hexdigest()
        incoming={j:i for i,j,_,_ in structural};depths=[]
        for i in kept:
            depth=0
            while i in incoming:depth+=1;i=incoming[i]
            depths.append(depth)
        tokens=re.findall(r'\w+|[^\w\s]',ex.public[0].text,re.UNICODE)
        assert len(graph.nodes)<=128
        assert all(json.dumps(n.value,sort_keys=True) in vocab for n in graph.nodes if n.kind not in ('ident','entity'))
        assert all(get_language('english').render(Ident(n.value)) in tokens for n in graph.nodes if n.kind in ('ident','entity'))
        slots=collections.defaultdict(set)
        for e in graph.edges:
            if e.slot is not None:slots[e.source,e.target].add(e.slot)
        assert all(len(s)==1 and min(s)>=0 and max(s)<32 for s in slots.values())
        records.append(dict(seed=seed,difficulty=difficulty,arity=arity,tree_shape_sha256=shape,unseen_tree_shape=shape not in known,depth=max(depths),nodes=len(graph.nodes),tokens=len(tokens),max_slot=max(max(s) for s in slots.values()),graph_sha256=graph.digest(),surface_sha256=ex.audit['surface_sha256']['english']))
summary=[]
for arity in (2,3,4,5):
    rows=[r for r in records if r['arity']==arity]
    summary.append(dict(arity=arity,examples=len(rows),tree_shapes=len({r['tree_shape_sha256'] for r in rows}),all_unseen_tree_shapes=all(r['unseen_tree_shape'] for r in rows),depths=sorted({r['depth'] for r in rows}),nodes=[min(r['nodes'] for r in rows),max(r['nodes'] for r in rows)],tokens=[min(r['tokens'] for r in rows),max(r['tokens'] for r in rows)],max_slot=max(r['max_slot'] for r in rows)))
paths=[Path(__file__),Path('src/topoformer/tcn_data.py'),Path('src/topoformer/semantic_graph.py'),Path('src/topoformer/_vendor/tcn_language/context.py'),Path('src/topoformer/_vendor/tcn_language/lessons/unification.py')]
result=dict(scope='64 bounded CPU construction fixtures only; no actor import/inference/training, no main corpus/cache',source_commit=SOURCE_COMMIT,vendor_manifest_valid=True,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},same_frozen_value_vocabulary=True,all_english_identifiers_visibly_copyable=True,all_single_slot_pairs=True,summary=summary,records=records,cpu_seconds=time.monotonic()-start)
output=Path('research/results/campaign-01/semantics/s14-motif-feasibility.json');output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
