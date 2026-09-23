"""Prospective labels-only audit. Never loads a model or predictions."""
import hashlib,json,collections,argparse
from pathlib import Path
from topoformer.semantic_text_acquisition import fixed_corpus
from topoformer import semantic_scaling as base
from topoformer.semantic_contracts import single_slot_labels
from topoformer.semantic_curriculum import encode_text
from topoformer.thinking_language import lexical_bits
p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args()
train=fixed_corpus();seen={base.semantic_key(x.privileged.graph) for x in train};vocab=base.value_vocabulary(train)
train_tokens={t for e in train for t in base.tokens(base.surface_input(e,'english')[0])}
rows=[];rejected=collections.Counter();texts={};hashes={};maximum=collections.Counter();unknown=collections.Counter();novel=collections.Counter()
for attempt in range(10000):
    seed=11100000+attempt;e=base.build_tcn_example('unification',seed,difficulty=.5);g=e.privileged.graph;key=base.semantic_key(g)
    if key in seen:rejected['duplicate_or_train_alpha_key']+=1;continue
    seen.add(key);public,_=base.surface_input(e,'english');tok=base.tokens(public)
    missing={json.dumps(n.value,sort_keys=True) for n in g.nodes if n.kind not in ('ident','entity')}-set(vocab)
    if missing:
        rejected['unknown_finite_value']+=1;unknown.update(missing);continue
    try:
        single_slot_labels(g);gold=base.targets(g,public,128,vocab,'english');encoded,length=encode_text(public.text)
        if encoded.shape[1]!=len(tok) or length!=len(tok):raise ValueError('truncation')
        if public.text in texts and texts[public.text]!=key:raise ValueError('incompatible_public')
        for token in tok:
            bits=tuple(lexical_bits(token))
            if bits in hashes and hashes[bits]!=token:raise ValueError('hash_collision')
            hashes[bits]=token
    except ValueError as ex:rejected[str(ex)]+=1;continue
    texts[public.text]=key;maximum['nodes']=max(maximum['nodes'],len(g.nodes));maximum['tokens']=max(maximum['tokens'],len(tok));maximum['slot']=max(maximum['slot'],max((x.slot for x in g.edges if x.slot is not None),default=-1))
    novel.update(set(tok)-train_tokens)
    rows.append(dict(novel_public_tokens=sorted(set(tok)-train_tokens),seed=seed,graph_sha256=g.digest(),semantic_sha256=hashlib.sha256(key.encode()).hexdigest(),public_sha256=hashlib.sha256(public.text.encode()).hexdigest(),nodes=len(g.nodes),tokens=len(tok),audit=e.audit))
    if len(rows)==512:break
Path(a.output).write_text(json.dumps(dict(scope='Labels-only prospective audit; no model loaded or predictions inspected. Fresh alpha keys relative to eight Stage11 training examples only; historical Stage8 overlap not asserted absent.',start_seed=11100000,max_attempts=10000,attempts=attempt+1,accepted=len(rows),rejection_counts=dict(rejected),unknown_values=dict(unknown),fixed_value_vocabulary=vocab,novel_token_example_counts=dict(novel),maxima=dict(maximum),rows=rows),indent=2)+'\n')
