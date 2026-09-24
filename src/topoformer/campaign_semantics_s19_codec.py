"""S19 typed record codec for canonical *scored* graph attributes.

No parser or inference-time graph metadata is used. Node count comes from NODE
records. Provenance/node-id dataclass bookkeeping is deliberately not encoded.
"""
from __future__ import annotations
import json
from numbers import Integral

BOS, NODE, EDGE, EOS, PAD = range(5)
MAX_RECORDS = 160
CAPACITY = 128
MAX_SLOT = 32
KINDS = ('scope','entity','token','str','num','ident','nil','pred','rel','node','tuple','list','record','app')
ROLES = ('contains','declares','refers_to','argument','item','binds','binding_scope',
         'field:query','field:substitution','field:pattern','field:fact','field:facts','field:scene')
EMPTY = (-1, -1, -1, -1)


class CodecError(ValueError):
    """Invalid generated graph; must count as an exact-graph failure."""


def _record(row):
    if len(row) != 5 or any(isinstance(x, bool) or not isinstance(x, Integral) for x in row):
        raise CodecError('record must contain five integers')
    return tuple(int(x) for x in row)


def decode_records(records, *, token_count, vocab_size, capacity=CAPACITY,
                   max_records=MAX_RECORDS, require_canonical=False):
    """Validate generated records; return lossless integer nodes and edge list.

    EOS is mandatory. Optional PAD rows follow EOS only. BOS is input-only.
    Edge emission order is not graph semantics (canonical ordering is optional).
    Multiple relations/slots on a node pair are retained. Exact duplicate edges
    are rejected. No target node count or gold graph is accepted by this API.
    """
    nodes, edges, seen = [], [], set()
    ended = False
    for position, raw in enumerate(records):
        tag, a, b, c, d = _record(raw)
        if ended:
            if (tag, a, b, c, d) != (PAD, *EMPTY):
                raise CodecError('only PAD allowed after EOS')
            continue
        if position >= max_records:
            raise CodecError('record capacity overflow')
        if tag == EOS:
            if (a,b,c,d) != EMPTY: raise CodecError('EOS has payload')
            ended = True
        elif tag == NODE:
            if edges: raise CodecError('NODE after EDGE')
            if len(nodes) >= capacity: raise CodecError('node capacity overflow')
            if not 0 <= a < len(KINDS) or d != -1: raise CodecError('invalid NODE kind/payload')
            if KINDS[a] in ('ident','entity'):
                if b != -1 or not 0 <= c < token_count: raise CodecError('invalid public-copy node')
            elif not 0 <= b < vocab_size or c != -1:
                raise CodecError('invalid finite-value node')
            nodes.append((a,b,c))
        elif tag == EDGE:
            if not (0 <= a < len(nodes) and 0 <= b < len(nodes)):
                raise CodecError('undefined node reference')
            if not 0 <= c < len(ROLES) or not -1 <= d < MAX_SLOT:
                raise CodecError('invalid relation/slot')
            edge = (a,b,c,d)
            if edge in seen: raise CodecError('duplicate edge record')
            if require_canonical and edges and edge < edges[-1]:
                raise CodecError('noncanonical edge order')
            seen.add(edge); edges.append(edge)
        else:
            raise CodecError('BOS/PAD/unknown tag in generated sequence')
    if not ended: raise CodecError('missing EOS')
    return {'nodes': nodes, 'edges': edges}


def canonical_edge_order(records):
    """Serialization diagnostic only; argument order is encoded by slot labels."""
    edges = [tuple(_record(row)[1:]) for row in records if _record(row)[0] == EDGE]
    return edges == sorted(edges)


def encode_row(row, vocab, *, capacity=CAPACITY, max_records=MAX_RECORDS):
    """Training-only lowering of existing compact targets to records.

    Public identity supervision exactly matches the inherited English renderer
    alias policy and first visible occurrence. No private identity enters input.
    """
    from .semantic_scaling import tokens, identifier_forms
    from .thinking_language import ActorInput
    public_tokens = tokens(ActorInput(row['text'], ()))
    records = []
    for kind, value in row['nodes']:
        try: k = KINDS.index(kind)
        except ValueError as e: raise CodecError('unknown node kind') from e
        if kind in ('ident','entity'):
            positions = [i for i,t in enumerate(public_tokens) if t in identifier_forms(str(value),'english')]
            if not positions: raise CodecError('uncopyable identity')
            records.append((NODE,k,-1,positions[0],-1))
        else:
            try: value_id = vocab.index(json.dumps(value,sort_keys=True))
            except ValueError as e: raise CodecError('unknown finite value') from e
            records.append((NODE,k,value_id,-1,-1))
    edge_records = []
    for source,target,role,slot in row['edges']:
        try: relation = ROLES.index(role)
        except ValueError as e: raise CodecError('unknown relation') from e
        edge_records.append((EDGE,source,target,relation,-1 if slot is None else slot))
    records.extend(sorted(edge_records)); records.append((EOS,*EMPTY))
    decode_records(records,token_count=len(public_tokens),vocab_size=len(vocab),
                   capacity=capacity,max_records=max_records,require_canonical=True)
    return records


def canonicalize_public_copies(records, public_tokens):
    """First identical public token; no gold identity or alias substitution."""
    first = {}
    for i,t in enumerate(public_tokens): first.setdefault(t,i)
    result = []
    for raw in records:
        tag,a,b,c,d = _record(raw)
        if tag == NODE and c >= 0:
            if c >= len(public_tokens): raise CodecError('copy index outside public input')
            c = first[public_tokens[c]]
        result.append((tag,a,b,c,d))
    return result


def records_to_targets(records, *, token_count, vocab_size, capacity=CAPACITY,
                       max_records=MAX_RECORDS, require_canonical=False):
    """Historical tensor adapter. Reject unsupported pair-slot multiplicity.

    Presence is padded to capacity from GENERATED nodes. Callers must retain
    CodecError as exact failure, never score a partially decoded graph instead.
    """
    import torch
    graph = decode_records(records,token_count=token_count,vocab_size=vocab_size,
        capacity=capacity,max_records=max_records,require_canonical=require_canonical)
    n = len(graph['nodes'])
    out = dict(presence=torch.arange(capacity)<n,kind=torch.zeros(capacity,dtype=torch.long),
        value=torch.full((capacity,),-1,dtype=torch.long),copy=torch.full((capacity,),-1,dtype=torch.long),
        edges=torch.zeros(capacity,capacity,len(ROLES),dtype=torch.bool),
        slots=torch.full((capacity,capacity),-1,dtype=torch.long))
    for i,(kind,value,copy) in enumerate(graph['nodes']):
        out['kind'][i]=kind; out['value'][i]=value; out['copy'][i]=copy
    pair_slots = {}
    for i,j,role,slot in graph['edges']:
        if (i,j) in pair_slots and pair_slots[i,j] != slot:
            raise CodecError('historical metric cannot express multiple pair-slot labels')
        pair_slots[i,j]=slot; out['edges'][i,j,role]=True; out['slots'][i,j]=slot
    return out


def teacher_forcing_inputs(records):
    """BOS + previous target rows; caller handles padding/masking explicitly."""
    return [(BOS,*EMPTY)] + list(records[:-1])
