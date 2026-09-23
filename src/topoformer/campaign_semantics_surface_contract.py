"""Prospective dataset-only checks for surface-local semantic copy targets.

Privileged graph/renderer alignment is used to validate supervision, never as
actor inference input. This module does not alter historical caches or decode.
"""
from __future__ import annotations
import hashlib
import json
from .semantic_scaling import identifier_forms,tokens


class SurfaceCopyContractError(ValueError):
    pass


def validate_surface_copy_contract(graph,public,language,*,target=None):
    """Reject uncopyable or non-injective surface identities; never repair them."""
    if public.options:
        raise SurfaceCopyContractError('This graph-cache contract supports text-only ActorInput')
    tok=tokens(public)
    identities=sorted({str(n.value) for n in graph.nodes if n.kind in ('ident','entity')})
    positions={value:tuple(i for i,token in enumerate(tok) if token in identifier_forms(value,language)) for value in identities}
    for value,support in positions.items():
        if not support:raise SurfaceCopyContractError(f'Uncopyable identity: {value!r}')
    owner={}
    for value,support in positions.items():
        for position in support:
            if position in owner and owner[position]!=value:
                raise SurfaceCopyContractError(f'Surface identity collision at token {position}: {owner[position]!r} and {value!r}')
            owner[position]=value
    first={value:support[0] for value,support in positions.items()}
    if target is not None:
        for i,node in enumerate(graph.nodes):
            if node.kind in ('ident','entity'):
                if int(target['copy'][i])!=first[str(node.value)] or int(target['value'][i])!=-1:
                    raise SurfaceCopyContractError('Copy target violates surface-local pointer/value separation')
    return dict(language=language,first_copy=first,identity_token_support={k:list(v) for k,v in positions.items()},identity_count=len(identities),token_count=len(tok),contract='surface-local pointers/equality; no hidden canonical-name export')


def register_surface_target(graph,public,language,seen,*,target=None):
    """Reject identical actor text with different representable graph targets.

    The fingerprint intentionally anchors identity to public token positions,
    not hidden English aliases. Exact canonical-name export is a different task.
    """
    contract=validate_surface_copy_contract(graph,public,language,target=target)
    ids={n.id:i for i,n in enumerate(graph.nodes)}
    nodes=[(n.kind,None,contract['first_copy'][str(n.value)]) if n.kind in ('ident','entity') else (n.kind,n.value,None) for n in graph.nodes]
    edges=sorted((ids[e.source],ids[e.target],e.role,-1 if e.slot is None else e.slot) for e in graph.edges)
    signature=hashlib.sha256(json.dumps(dict(nodes=nodes,edges=edges),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if public.text in seen and seen[public.text]!=signature:
        raise SurfaceCopyContractError('Identical public text has incompatible surface-local graph targets')
    seen[public.text]=signature
    return dict(contract,target_sha256=signature)
