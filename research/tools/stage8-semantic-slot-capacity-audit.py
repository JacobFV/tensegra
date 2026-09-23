"""Gold-only audit of additive slot-head label expressivity; no model outcomes."""
import itertools
import json
import sqlite3
from pathlib import Path
from topoformer.semantic_scaling import build_tcn_example
from topoformer.semantic_curriculum import partition_corpus


def witness(graph):
    slots={(e.source,e.target):e.slot for e in graph.edges if e.slot is not None}
    for (source_a,target_a), (source_b,target_b) in itertools.combinations(slots,2):
        slot=slots[source_a,target_a]
        if (source_a!=source_b and target_a!=target_b and slots[source_b,target_b]==slot
            and (source_a,target_b) not in slots and (source_b,target_a) not in slots):
            return dict(sources=[source_a,source_b],targets=[target_a,target_b],
                        labels=[[slot,-1],[-1,slot]],graph_sha256=graph.digest())
    return None


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('database');parser.add_argument('exclusions');parser.add_argument('output')
    args=parser.parse_args()
    class ReadCorpus:
        db=sqlite3.connect(f'file:{args.database}?mode=ro',uri=True)
        def __len__(self): return self.db.execute('select count(*) from examples').fetchone()[0]
        def __getitem__(self,index):
            lesson,seed=self.db.execute('select lesson,seed from examples where id=?',(index+1,)).fetchone()
            return build_tcn_example(lesson,seed,difficulty=.5)
    corpus=ReadCorpus(); exclusion=json.loads(Path(args.exclusions).read_text())['semantic_keys']
    _,training,_=partition_corpus(corpus,64,10000,exclusion)
    groups={}
    for index in training[:1000]:
        e=corpus[index]; result=witness(e.privileged.graph)
        group=groups.setdefault(e.audit['lesson'],dict(graphs=0,graphs_with_witness=0,first_witness=None))
        group['graphs']+=1
        if result:
            group['graphs_with_witness']+=1
            if group['first_witness'] is None: group['first_witness']=result
    Path(args.output).write_text(json.dumps(dict(training_graphs=1000,by_lesson=groups,
      theorem='For logits a[source,class]+b[target,class], a 2x2 XOR of two class labels cannot satisfy all four strict argmax inequalities.',
      scope='Slot loss supervises no-slot on sampled nonedge pairs. This establishes an objective/head expressivity mismatch, not impossibility of exact graph decoding after an independent edge mask.',
      recipe_changed=False),indent=2)+'\n')
