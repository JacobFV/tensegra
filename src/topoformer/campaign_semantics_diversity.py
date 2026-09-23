"""Separate alpha/equality diversity from ordered semantic-tree shapes, CPU only."""
import argparse,collections,gzip,hashlib,json
from pathlib import Path


def run(data):
    out={};seen={}
    for split in ('train','development'):
        skeletons=collections.Counter();full=collections.Counter();eq=collections.Counter();nodes=collections.Counter();depths=collections.Counter()
        for line in gzip.open(Path(data)/(split+'.jsonl.gz'),'rt'):
            r=json.loads(line);kinds=[k for k,v in r['nodes']];edges=r['edges'];nodes[len(kinds)]+=1
            structural=[(i,j,role,s) for i,j,role,s in edges if role in ('argument','item') or role.startswith('field:')]
            kept=[i for i,k in enumerate(kinds) if k not in ('entity','scope')];index={i:j for j,i in enumerate(kept)}
            tree=json.dumps([[kinds[i] for i in kept],[(index[i],index[j],role,s) for i,j,role,s in structural]],separators=(',',':'))
            skeletons[hashlib.sha256(tree.encode()).hexdigest()]+=1
            full[hashlib.sha256(json.dumps([kinds,edges],separators=(',',':')).encode()).hexdigest()]+=1
            ids={};sequence=[]
            for k,v in r['nodes']:
                if k=='ident':
                    if v not in ids:ids[v]=len(ids)
                    sequence.append(ids[v])
            eq[tuple(sequence)]+=1
            incoming={j:i for i,j,_,_ in structural};maximum=0
            for i in kept:
                visited=set();depth=0
                while i in incoming:
                    if i in visited:raise ValueError('structural cycle')
                    visited.add(i);depth+=1;i=incoming[i]
                maximum=max(maximum,depth)
            depths[maximum]+=1
        seen[split]=set(skeletons)
        out[split]=dict(examples=sum(nodes.values()),exact_indexed_kind_edge_slot_skeletons=len(full),ordered_tree_shapes=len(skeletons),tree_shape_counts=dict(skeletons),alpha_equality_patterns=len(eq),nodes=dict(nodes),semantic_structural_depth=dict(depths),data_sha256=hashlib.sha256((Path(data)/(split+'.jsonl.gz')).read_bytes()).hexdigest())
    return dict(scope='TRAIN/DEV only; exact full graph skeleton retains entity interleaving/reference topology. Ordered tree shape excludes scope/entity nodes and all bookkeeping edges; keeps node kinds and argument/item/field slots but no values.',splits=out,development_tree_shapes_unseen_in_train=len(seen['development']-seen['train']),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('output');a=p.parse_args();Path(a.output).write_text(json.dumps(run(a.data),indent=2)+'\n')
