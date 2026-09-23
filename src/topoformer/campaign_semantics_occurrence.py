"""Privileged occurrence targets; never an actor input or deployment dependency."""
import re
import torch


def occurrence_targets(row, gold):
    """Align canonical ident occurrences to checked English surface traversal.

    Other fields and entity-copy targets retain the historical contract. Reject
    rather than guess if this narrow renderer's occurrence order differs.
    """
    tokens=re.findall(r'\w+|[^\w\s]',row['text'],re.UNICODE)
    occurrences=[(i,v) for i,(k,v) in enumerate(row['nodes']) if k=='ident']
    names={v for _,v in occurrences}
    visible=[(i,t) for i,t in enumerate(tokens) if t in names]
    if [v for _,v in occurrences]!=[v for _,v in visible]:
        raise ValueError('English occurrence alignment contract violated')
    copy=gold['copy'].clone()
    for (node,_),(position,_) in zip(occurrences,visible):copy[node]=position
    return {**gold,'copy':copy}
