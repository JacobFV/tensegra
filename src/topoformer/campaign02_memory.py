"""Versioned lossless, alpha-normalized public JSON memory; no Torch or solvers.

This codec supplies syntax, public identity equality, and exact scalar bytes. It
never solves a constraint, follows a route, or computes a teacher preference.
Rows retain parent/order/reference indices so even ordinary keyed readers receive
relations rather than an unordered bag. Capacity overflow raises; never truncates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import math
import struct
from typing import Any

VERSION = "public-tree-v1"
# Fixed schema vocabulary is only a compact representation. Unknown literal
# strings/field names are losslessly represented by ordered UTF-8 byte chunks.
TOKENS = tuple(sorted(set("""
observation actions version state_version goal categories capacity funds destination
call_budgets include_remaining_budget item_inventory handle category known_items
known_edges incompatible records retrieved problems pending selected position delivered
verified done remaining_steps remaining_travel remaining_work prices feedback action
work travel compute status return primitive problem problem_snapshot items handles
constraints max_cost forbidden_pairs n edges start kind source payload work_units
budget call_position certificate_valid feasible_incumbent arguments target item
constraint as weight price record stale reason path edge selected verified
workshop-v1 success timeout infeasible invalid_input unknown unavailable_resource
ready rejected obstacle incomplete computation shortest_path constrained_subset
lookup filter intersection arithmetic compare csp subset route map capacity funds
incompatibility inspect build start_subset add_constraint build_route choose_item
commit_pending use_return call retrieve commit_subset move deliver verify think abstain
stale_result no_valid_feasible_payload not_success_result retrieve_required
stale_problem assembly_requires_workshop no_selection wrong_destination
missing_edge travel_budget category_coverage duplicate_item invalid_handles
unknown_item compatibility valid return_type
""".split())))
TOKEN_INDEX = {value: i for i, value in enumerate(TOKENS)}
KINDS = ("dict", "list", "null", "bool", "int", "float", "string", "handle", "bytes", "keybytes")
KIND_INDEX = {value: i for i, value in enumerate(KINDS)}
# All index fields use exact binary fractions; supported indices are <=65535.
INDEX_SCALE = 65536.0
KIND_START = 0
KEY_MODE_START = len(KINDS)  # none, vocabulary field, handle field, UTF8 field
KEY_TOKEN_START = KEY_MODE_START + 4
VALUE_TOKEN_START = KEY_TOKEN_START + len(TOKENS)
INDEX_START = VALUE_TOKEN_START + len(TOKENS)
# index,parent+1,ordinal,depth,handle+1,keyhandle+1,byte_count,numeric_aux1,aux2
BYTE_START = INDEX_START + 9
ROW_DIM = BYTE_START + 8


@dataclass(frozen=True)
class MemoryLimits:
    max_nodes: int = 8192
    max_handles: int = 512
    max_depth: int = 64
    max_string_bytes: int = 4096

    def __post_init__(self):
        if not 1 <= self.max_nodes <= 65535 or not 1 <= self.max_handles <= 65535:
            raise ValueError("Node and handle capacities must be in [1,65535]")
        if not 1 <= self.max_depth <= 65535 or not 1 <= self.max_string_bytes <= 65535:
            raise ValueError("Invalid depth/string capacity")


class MemoryCapacityError(ValueError):
    def __init__(self, message, *, limit=None, observed=None):
        super().__init__(message)
        self.limit, self.observed = limit, observed

    def record(self):
        return {"status": "unsupported_public_memory", "reason": str(self),
                "limit": self.limit, "observed": self.observed}


@dataclass
class PublicMemory:
    rows: list[list[float]]
    edges: list[tuple[int, int, int]]
    action_links: list[list[int]]
    action_roots: list[int]
    stats: dict[str, Any]

    @property
    def edge_index(self):
        return [(a, b) for a, b, _ in self.edges]

    @property
    def edge_type(self):
        return [r for _, _, r in self.edges]


def _plain(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("Public JSON requires string mapping keys")
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported public JSON type: {type(value).__name__}")


class _Handle:
    def __init__(self, index):
        self.index = index


# Locations carrying references, as opposed to semantically meaningful strings.
REFERENCE_FIELDS = {"handle", "item", "source", "target", "return"}
REFERENCE_LISTS = {"pending", "selected", "handles", "incompatible"}
HANDLE_MAPS = {"known_items", "retrieved", "problems"}


def _normalize(observation, actions, limits):
    obs, actions = _plain(observation), _plain(actions)
    handles: dict[str, int] = {}
    def intern(name):
        if not isinstance(name, str):
            raise ValueError("Opaque handle must be a string")
        if name not in handles:
            if len(handles) >= limits.max_handles:
                raise MemoryCapacityError("Opaque handle capacity exceeded", limit=limits.max_handles, observed=len(handles)+1)
            handles[name] = len(handles)
        return handles[name]
    # Public declaration order, not lexical spelling or numeric handle magnitude.
    for row in obs.get("item_inventory", []):
        intern(row["handle"])
    for key in obs.get("known_items", {}):
        intern(key)
    for key in obs.get("problems", {}):
        intern(key)
    for row in obs.get("records", []):
        intern(row["handle"])
    for key in obs.get("retrieved", {}):
        intern(key)
    # Candidate-created/invalid references also share identity within this frame.
    def discover(value, field=""):
        if isinstance(value, dict):
            for key, item in value.items():
                if field in HANDLE_MAPS:
                    intern(key)
                discover(item, key)
        elif isinstance(value, list):
            for item in value:
                discover(item, field)
        elif isinstance(value, str):
            if field in REFERENCE_FIELDS and not (field in {"source", "target"} and value == "map"):
                intern(value)
            elif field in REFERENCE_LISTS or field == "problem":
                intern(value)
            elif field == "items":  # commit_subset identity list, not numeric rows
                intern(value)
    discover(obs)
    discover(actions)
    def normalized(value, field=""):
        if isinstance(value, dict):
            # Public object property order is retained, including draft row order.
            return {(f"@handle:{handles[k]}" if field in HANDLE_MAPS else k): normalized(v, k)
                    for k, v in value.items()}
        if isinstance(value, list):
            return [normalized(v, field) for v in value]
        if isinstance(value, str):
            ref = (field in REFERENCE_FIELDS and not (field in {"source", "target"} and value == "map")) or field in REFERENCE_LISTS or field in {"problem", "items"}
            if ref and value in handles:
                return _Handle(handles[value])
            # KeyError feedback can quote a handle. Preserve the diagnostic while
            # removing irrelevant spelling; no general natural-language rewrite.
            if field == "reason":
                for spelling, index in sorted(handles.items(), key=lambda x: -len(x[0])):
                    value = value.replace(repr(spelling), repr(f"@handle:{index}"))
        return value
    return {"observation": normalized(obs), "actions": normalized(actions)}, handles


def normalized_document(observation, actions=None, limits=MemoryLimits()):
    if actions is None:
        from .campaign02_world import action_catalog
        actions = action_catalog(observation)
    document, _ = _normalize(observation, actions, limits)
    def export(value):
        if isinstance(value, _Handle):
            return {"@opaque": value.index}
        if isinstance(value, dict):
            return {k: export(v) for k, v in value.items()}
        if isinstance(value, list):
            return [export(v) for v in value]
        return value
    return export(document)


def encode_memory(observation, actions=None, limits=MemoryLimits()) -> PublicMemory:
    if actions is None:
        from .campaign02_world import action_catalog
        actions = action_catalog(observation)
    document, handles = _normalize(observation, actions, limits)
    rows, edges, anchors, references = [], [], {}, []
    action_roots, action_links = [], []
    def chunks(text):
        data = text.encode("utf-8")
        if len(data) > limits.max_string_bytes:
            raise MemoryCapacityError("String byte capacity exceeded", limit=limits.max_string_bytes, observed=len(data))
        return [data[i:i+8] for i in range(0, len(data), 8)]
    def add(kind, parent, ordinal, depth, key=None, payload=b"", handle=None):
        if len(rows) >= limits.max_nodes:
            raise MemoryCapacityError("Public memory node capacity exceeded", limit=limits.max_nodes, observed=len(rows)+1)
        if depth > limits.max_depth:
            raise MemoryCapacityError("Public memory depth capacity exceeded", limit=limits.max_depth, observed=depth)
        if ordinal >= 65536:
            raise MemoryCapacityError("Public memory child ordinal capacity exceeded")
        index = len(rows)
        row = [0.] * ROW_DIM
        row[KIND_INDEX[kind]] = 1.
        row[INDEX_START:INDEX_START+4] = [(index+1)/INDEX_SCALE,(parent+1)/INDEX_SCALE,ordinal/INDEX_SCALE,depth/INDEX_SCALE]
        row[INDEX_START+6] = len(payload)/8
        for i, byte in enumerate(payload):
            row[BYTE_START+i] = byte/256
        if handle is not None:
            row[INDEX_START+4] = (handle+1)/INDEX_SCALE
            references.append((index, handle))
            anchors.setdefault(handle, index)
        if key is None:
            row[KEY_MODE_START] = 1
        elif key.startswith("@handle:"):
            kh = int(key.split(":")[1])
            row[KEY_MODE_START+2] = 1
            row[INDEX_START+5] = (kh+1)/INDEX_SCALE
            references.append((index,kh))
            anchors.setdefault(kh,index)
        elif key in TOKEN_INDEX:
            row[KEY_MODE_START+1] = 1
            row[KEY_TOKEN_START+TOKEN_INDEX[key]] = 1
        else:
            row[KEY_MODE_START+3] = 1
        rows.append(row)
        if parent >= 0:
            edges.extend(((parent,index,0),(index,parent,1)))
        if key is not None and row[KEY_MODE_START+3]:
            for position, data in enumerate(chunks(key)):
                add("keybytes",index,position,depth+1,payload=data)
        return index
    def visit(value, parent=-1, ordinal=0, depth=0, key=None, path=()):
        if isinstance(value, _Handle):
            return add("handle",parent,ordinal,depth,key,handle=value.index)
        if value is None:
            return add("null",parent,ordinal,depth,key)
        if isinstance(value, bool):
            return add("bool",parent,ordinal,depth,key,payload=bytes([int(value)]))
        if isinstance(value, int):
            if not -(2**63) <= value < 2**63:
                raise MemoryCapacityError("Exact integer contract is signed64")
            index = add("int",parent,ordinal,depth,key,payload=struct.pack(">q",value))
        elif isinstance(value,float):
            if not math.isfinite(value):
                raise ValueError("Nonfinite public scalar unsupported")
            index = add("float",parent,ordinal,depth,key,payload=struct.pack(">d",value))
        elif isinstance(value,str):
            index = add("string",parent,ordinal,depth,key)
            if value in TOKEN_INDEX:
                rows[index][VALUE_TOKEN_START+TOKEN_INDEX[value]] = 1.
            else:
                for position, data in enumerate(chunks(value)):
                    add("bytes",index,position,depth+1,payload=data)
            return index
        elif isinstance(value,(dict,list)):
            index = add("dict" if isinstance(value,dict) else "list",parent,ordinal,depth,key)
            entries = value.items() if isinstance(value,dict) else enumerate(value)
            for position, (child_key, child) in enumerate(entries):
                child_path = path + (child_key,)
                before = len(rows)
                root = visit(child,index,position,depth+1,child_key if isinstance(value,dict) else None,child_path)
                if path == ("actions",):
                    action_roots.append(root)
                    action_links.append(list(range(before,len(rows))))
            return index
        else:
            raise TypeError("Unsupported normalized value")
        rows[index][INDEX_START+7] = math.copysign(math.log1p(abs(value))/64,value)
        rows[index][INDEX_START+8] = max(-1.,min(1.,value/65536))
        return index
    visit(document)
    for node, handle in references:
        anchor = anchors[handle]
        if node != anchor:
            edges.extend(((node,anchor,2),(anchor,node,3)))
    for links in action_links:
        support = set(links)
        for node, handle in references:
            if node in support:
                support.add(anchors[handle])
        links[:] = sorted(support)
    return PublicMemory(rows,edges,action_links,action_roots,{
        "version":VERSION,"row_dim":ROW_DIM,"nodes":len(rows),"handles":len(handles),
        "edges":len(edges),"actions":len(action_roots),"limits":asdict(limits),
        "relation_types": ["parent_to_child","child_to_parent","reference_to_anchor","anchor_to_reference"]})


def decode_memory(memory: PublicMemory):
    """Reconstruct the alpha-normalized public document from row features alone."""
    rows = memory.rows
    children = [[] for _ in rows]
    def idx(row, offset):
        return round(row[INDEX_START+offset]*INDEX_SCALE)
    def kind(row):
        return KINDS[max(range(len(KINDS)),key=lambda i:row[i])]
    def payload(row):
        return bytes(round(v*256) for v in row[BYTE_START:BYTE_START+round(row[INDEX_START+6]*8)])
    for i,row in enumerate(rows):
        if len(row) != ROW_DIM or idx(row,0) != i+1:
            raise ValueError("Malformed public memory row/index")
        parent = idx(row,1)-1
        if parent >= i or parent < -1:
            raise ValueError("Parent must precede child")
        if parent >= 0:
            children[parent].append(i)
    def text(parent, nodekind):
        selected = sorted((i for i in children[parent] if kind(rows[i])==nodekind),key=lambda i:idx(rows[i],2))
        return b"".join(payload(rows[i]) for i in selected).decode("utf-8")
    def key(i):
        row=rows[i]
        mode=max(range(4),key=lambda j:row[KEY_MODE_START+j])
        if mode == 1:
            return TOKENS[max(range(len(TOKENS)),key=lambda j:row[KEY_TOKEN_START+j])]
        if mode == 2:
            return f"@handle:{idx(row,5)-1}"
        if mode == 3:
            return text(i,"keybytes")
        raise ValueError("Dictionary child missing key")
    def decode(i):
        row=rows[i]
        k=kind(row)
        if k=="handle": return {"@opaque":idx(row,4)-1}
        if k=="null": return None
        if k=="bool": return bool(payload(row)[0])
        if k=="int": return struct.unpack(">q",payload(row))[0]
        if k=="float": return struct.unpack(">d",payload(row))[0]
        if k=="string":
            values=row[VALUE_TOKEN_START:VALUE_TOKEN_START+len(TOKENS)]
            return TOKENS[values.index(1.)] if 1. in values else text(i,"bytes")
        actual=sorted((c for c in children[i] if kind(rows[c])!="keybytes"),key=lambda c:idx(rows[c],2))
        if k=="dict": return {key(c):decode(c) for c in actual}
        if k=="list": return [decode(c) for c in actual]
        raise ValueError("Unexpected raw byte chunk")
    return decode(0)
