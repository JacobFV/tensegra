"""CPU-only regeneration of declared semantic split identities; no model."""
import hashlib,json,sys
from pathlib import Path
from topoformer import semantic_scaling as base
from topoformer.semantic_text_acquisition import fixed_corpus
train=fixed_corpus();train_keys={base.semantic_key(e.privileged.graph)for e in train};tokens={t for e in train for t in base.tokens(base.surface_input(e,'english')[0])}
audit=json.loads(Path(sys.argv[1]).read_text());seen=set();novel=set()
for row in audit['rows']:
 e=base.build_tcn_example('unification',row['seed'],difficulty=.5);g=e.privileged.graph;key=base.semantic_key(g);public,_=base.surface_input(e,'english')
 assert key not in train_keys and key not in seen;seen.add(key)
 assert g.digest()==row['graph_sha256']and hashlib.sha256(key.encode()).hexdigest()==row['semantic_sha256']and hashlib.sha256(public.text.encode()).hexdigest()==row['public_sha256']
 novelty=set(base.tokens(public))-tokens;assert sorted(novelty)==row['novel_public_tokens'];novel|=novelty
assert len(train_keys)==8 and len(seen)==512
print(json.dumps(dict(cpu_only=True,model_loaded=False,training_constructions=8,fresh_constructions=512,alpha_overlap=0,all_graph_semantic_and_public_hashes_verified=True,novel_tokens=sorted(novel)),indent=2))
