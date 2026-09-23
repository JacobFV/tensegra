"""Read-only public observability audit of the frozen Stage 8 construction pool."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
from topoformer import semantic_scaling as base
from topoformer.semantic_curriculum import partition_corpus
from topoformer.thinking_language import lexical_bits


def audit(database, exclusions, manifest, limit=10064):
    db=sqlite3.connect(f'file:{database}?mode=ro',uri=True)
    rows=db.execute('select lesson,seed from examples order by id').fetchall()[:limit]
    vocab=json.loads(Path(manifest).read_text())['value_vocabulary']
    public={}; tokens={}; counter=Counter(); lessons=Counter(); examples={}; maxima=Counter()
    slot_collisions=[]; public_collisions=[]; hash_collisions=[]; missing=[]; unknown=[]
    for index,(lesson,seed) in enumerate(rows):
        example=base.build_tcn_example(lesson,seed,difficulty=.5); graph=example.privileged.graph
        lessons[lesson]+=1; key=base.semantic_key(graph)
        maxima['nodes']=max(maxima['nodes'],len(graph.nodes)); maxima['edges']=max(maxima['edges'],len(graph.edges))
        pairlabels=defaultdict(set)
        for edge in graph.edges:
            pairlabels[edge.source,edge.target].add((edge.role,edge.slot))
            if edge.slot is not None: maxima['slot']=max(maxima['slot'],edge.slot)
        for pair,labels in pairlabels.items():
            if len(labels)>1:
                counter['multi_label_pairs']+=1
                if len({slot for _,slot in labels})>1:
                    counter['conflicting_pair_slots']+=1
                    if len(slot_collisions)<10: slot_collisions.append(dict(index=index,pair=pair,labels=sorted(labels,key=str)))
        for language in ('english','spanish','symbols'):
            inp,_=base.surface_input(example,language); tok=base.tokens(inp)
            maxima['tokens']=max(maxima['tokens'],len(tok)); counter['surfaces']+=1
            digest=hashlib.sha256(inp.text.encode()).hexdigest()
            previous=public.setdefault(digest,(key,index,language))
            if previous[0]!=key:
                counter['incompatible_same_public']+=1
                if len(public_collisions)<10: public_collisions.append(dict(first=previous[1:],second=[index,language]))
            for token in set(tok):
                bits=tuple(lexical_bits(token)); old=tokens.setdefault(bits,token)
                if old!=token:
                    counter['hash_collisions']+=1
                    if len(hash_collisions)<10: hash_collisions.append([old,token])
            for node in graph.nodes:
                if node.kind in ('ident','entity'):
                    if not any(t in base.identifier_forms(str(node.value),language) for t in tok):
                        counter['uncopyable_identities']+=1
                        if len(missing)<10: missing.append([index,language,node.value])
                elif json.dumps(node.value,sort_keys=True) not in vocab:
                    counter['unknown_values']+=1
                    if len(unknown)<10: unknown.append([index,node.value])
        examples[index]=example
    class Corpus:
        def __len__(self):return len(rows)
        def __getitem__(self,index):return examples[index]
    excl=json.loads(Path(exclusions).read_text())['semantic_keys']
    evaluation,training,split=partition_corpus(Corpus(),64,10000,excl)
    return dict(kind='archived construction regeneration; no model inference or training',graphs=len(rows),lessons=dict(lessons),counts=dict(counter),maxima=dict(maxima),
      examples=dict(slot_collisions=slot_collisions,public_collisions=public_collisions,hash_collisions=hash_collisions,uncopyable=missing,unknown_values=unknown),
      split=split,eval_lessons=dict(Counter(rows[i][0] for i in evaluation)),train_lessons=dict(Counter(rows[i][0] for i in training)),
      capacities=dict(nodes=128,ordered_slots=32,public_tokens='no truncation in actor encoding'),
      limitations=['Finite enumerated pool cannot prove absence of collisions in the full generator.','Identical text is compared under alpha-normalized canonical graph equivalence; compiler traversal is not unrestricted graph isomorphism.','Hidden metadata is not a target; this audit detects incompatible text targets only when observed.'],
      database_sha256=hashlib.sha256(Path(database).read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('database');p.add_argument('exclusions');p.add_argument('manifest');p.add_argument('output');a=p.parse_args()
    Path(a.output).write_text(json.dumps(audit(a.database,a.exclusions,a.manifest),indent=2)+'\n')
