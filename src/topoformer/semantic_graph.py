"""Canonical typed graphs compiled from TCN TermJSON (never actor inputs).

This is a structural semantic compiler, not a general TCN rewrite engine.
Identifier identity is scoped to the input construction; lexical scope creation
beyond that construction is intentionally not inferred from arbitrary heads.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

COMPILER_VERSION = 'topoformer-termgraph-v1'


@dataclass(frozen=True)
class SemanticNode:
    id: str
    kind: str
    value: str | int | float | bool | None = None
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticEdge:
    source: str
    target: str
    role: str
    slot: int | None = None


@dataclass(frozen=True)
class SemanticGraph:
    nodes: tuple[SemanticNode, ...]
    edges: tuple[SemanticEdge, ...]
    roots: tuple[str, ...]
    compiler_version: str = COMPILER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def compile_term(term: Mapping[str, Any]) -> SemanticGraph:
    """Compile ordered occurrences plus explicit identity, scope and bind edges.

    Named field slots use lexicographic order, matching TCN's Rec/App algebra.
    Node IDs encode canonical structural paths, not rendering or random seeds.
    Shared entities differ from identifier occurrences, preserving repeated uses.
    """
    nodes = [SemanticNode('scope', 'scope', provenance=('construction',))]
    edges: list[SemanticEdge] = []
    entities: dict[str, str] = {}

    def visit(t: Mapping[str, Any], path: str) -> str:
        if not isinstance(t, Mapping):
            raise ValueError('A TermJSON node must be a mapping')
        kind = t.get('t')
        if kind in {'token', 'str', 'num', 'ident', 'nil'}:
            value = t.get('v')
            if not isinstance(value, (str, int, float, bool, type(None))) or (isinstance(value, float) and not math.isfinite(value)):
                raise ValueError('Term atom must be a finite JSON scalar')
            if kind == 'num' and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                raise ValueError('num requires a number')
            if kind in {'str','ident'} and not isinstance(value, str):
                raise ValueError(f'{kind} requires a string')
            if kind == 'nil' and value is not None:
                raise ValueError('nil requires null')
            children = []
        elif kind in {'pred', 'rel', 'node'}:
            value = t['head']
            children = [('argument', i, x) for i, x in enumerate(t['args'])]
        elif kind in {'tuple', 'list'}:
            value = None
            children = [('item', i, x) for i, x in enumerate(t['items'])]
        elif kind in {'record', 'app'}:
            value = t['fn'] if kind == 'app' else None
            fields = t['args'] if kind == 'app' else t['fields']
            children = [(f'field:{key}', i, fields[key]) for i, key in enumerate(sorted(fields))]
        else:
            raise ValueError(f'Unsupported TermJSON type: {kind!r}')
        if kind in {'pred','rel','node','app'} and not isinstance(value,str):
            raise ValueError('Term head must be a string')
        nodes.append(SemanticNode(path, kind, value, (path,)))
        edges.append(SemanticEdge('scope', path, 'contains'))
        if kind == 'ident':
            if value not in entities:
                entity = 'entity:' + hashlib.sha256(value.encode()).hexdigest()
                entities[value] = entity
                nodes.append(SemanticNode(entity, 'entity', value, ('scope',)))
                edges.append(SemanticEdge('scope', entity, 'declares'))
            edges.append(SemanticEdge(path, entities[value], 'refers_to'))
        child_ids = []
        for role, slot, child in children:
            child_id = visit(child, f'{path}/{slot}')
            edges.append(SemanticEdge(path, child_id, role, slot))
            child_ids.append(child_id)
        if kind == 'pred' and value == 'bind' and len(child_ids) == 2:
            edges.append(SemanticEdge(child_ids[0], child_ids[1], 'binds'))
            edges.append(SemanticEdge(path, 'scope', 'binding_scope'))
        return path

    root = visit(term, 'root')
    return SemanticGraph(tuple(nodes), tuple(edges), (root,))
