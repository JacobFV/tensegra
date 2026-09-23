"""CPU-only audit of identity-copy targets versus ordered English occurrences."""
import argparse, collections, gzip, hashlib, json, re
from pathlib import Path


def run(data):
    result={}
    for split in ('train','development'):
        counts=collections.Counter();examples=[];alignment=hashlib.sha256()
        for line in gzip.open(Path(data)/(split+'.jsonl.gz'),'rt'):
            row=json.loads(line);tok=re.findall(r'\w+|[^\w\s]',row['text'],re.UNICODE)
            ident=[(i,v) for i,(k,v) in enumerate(row['nodes']) if k=='ident'];names={v for _,v in ident}
            visible=[(j,t) for j,t in enumerate(tok) if t in names]
            counts['graphs']+=1;counts['identifier_occurrences']+=len(ident)
            consistent=[v for _,v in ident]==[t for _,t in visible]
            counts['exact_ordered_surface_alignment_graphs']+=consistent
            if not consistent:examples.append(dict(seed=row['seed'],canonical=ident,visible=visible))
            multiplicity=collections.Counter(v for _,v in ident)
            counts['graphs_with_repeated_identities']+=any(n>1 for n in multiplicity.values())
            counts['identities']+=len(multiplicity)
            counts['occurrences_sharing_identity_target']+=sum(n for n in multiplicity.values() if n>1)
            counts['distinct_occurrence_pairs_sharing_identity_target']+=sum(n*(n-1)//2 for n in multiplicity.values())
            if consistent:
                mapping=[(node,position) for (node,_),(position,_) in zip(ident,visible)]
                counts['occurrence_target_differs_from_identity_first']+=sum(position!=next(j for j,t in visible if t==row['nodes'][node][1]) for node,position in mapping)
                alignment.update(json.dumps([row['semantic_sha256'],mapping],separators=(',',':')).encode())
        result[split]=dict(counts=dict(counts),alignment_sha256=alignment.hexdigest(),counterexamples=examples[:5],data_sha256=hashlib.sha256((Path(data)/(split+'.jsonl.gz')).read_bytes()).hexdigest())
    return dict(scope='Inspected English unification TRAIN/DEV only; no model predictions or confirmation input',copy_contract='Cross entropy to first visible identity position; not a union over occurrences. Decoder canonicalizes repeated visible tokens to first position.',interpretation='Copy supervision identifies entities, not occurrence-specific node spans. Ordered graph labels remain distinct. Alignment existence does not establish that adding its loss improves acquisition.',splits=result,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('output');a=p.parse_args();Path(a.output).write_text(json.dumps(run(a.data),indent=2)+'\n')
