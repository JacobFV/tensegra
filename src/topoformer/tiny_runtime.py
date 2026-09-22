"""Auditable immutable-value runtime. No host-language evaluation or tensor dependency."""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable


class RuntimeFault(ValueError):
    """Invalid syntax, type, scope or primitive interface."""


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    payload: Any = None


@dataclass(frozen=True)
class Edge:
    source: str
    relation: str
    target: str


@dataclass(frozen=True)
class Primitive:
    """Sigma plus named validity/execution and explicit boundary hooks L/U."""
    inputs: tuple[str, ...]
    output: str
    executor: str
    constraints: str

    def lower(self, latent: Any, lowering: Callable) -> Any:
        return lowering(latent, self)

    def lift(self, value: Any, lifting: Callable) -> Any:
        return lifting(value, self)


PRIMITIVES = {
    'resolve_name': Primitive(('name','frame'),'binding','resolve_name','nearest lexical binding exists'),
    'read_field': Primitive(('record','name'),'value','read_field','field exists'),
    'read_index': Primitive(('array','integer'),'value','read_index','0 <= index < length; bool excluded'),
    'call': Primitive(('function','values'),'value','call','arity and scalar builtin types; nonrecursive'),
    'return': Primitive(('frame','value'),'returned','return_value','frame and value exist'),
    'value_of': Primitive(('binding|slot|returned',),'value','value_of','exactly one value_of or binds edge'),
}


class Runtime:
    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.trace: list[dict] = []
        self._counter = 0
        self._bindings: dict[str, dict[str,str]] = {}
        self._parents: dict[str,str|None] = {}
        self._functions: dict[str,tuple] = {}
        self._builtins: dict[str,str] = {}
        self._active: list[str] = []
        self.root_scope = self.frame()

    def _new(self, kind, payload=None):
        node = f'n{self._counter}'; self._counter += 1
        self.nodes[node] = Node(node, kind, payload)
        return node

    def _edge(self, source, relation, target):
        self.edges.append(Edge(source, relation, target))

    def _check(self, node, kinds=None):
        if node not in self.nodes or (kinds is not None and self.nodes[node].kind not in kinds):
            raise RuntimeFault(f'expected {kinds}, got node {node!r}')
        return self.nodes[node]

    def _value(self, node):
        return self._check(node, {'scalar','record','array','function'})

    @contextmanager
    def transaction(self):
        """Failed primitives leave nodes, edges, scopes and trace unchanged."""
        snapshot = (dict(self.nodes), list(self.edges), list(self.trace), self._counter,
                    {k:dict(v) for k,v in self._bindings.items()}, dict(self._parents),
                    dict(self._functions), dict(self._builtins), list(self._active))
        try:
            yield
        except Exception:
            (self.nodes,self.edges,self.trace,self._counter,self._bindings,self._parents,
             self._functions,self._builtins,self._active) = snapshot
            raise

    def scalar(self, value):
        if type(value) not in (int,float): raise RuntimeFault('scalar must be numeric')
        if isinstance(value,float):
            import math
            if not math.isfinite(value): raise RuntimeFault('scalar must be finite')
        return self._new('scalar',value)

    def record(self, fields):
        for name,value in fields.items():
            if not isinstance(name,str): raise RuntimeFault('field name must be text')
            self._value(value)
        obj=self._new('record',tuple(fields))
        for name,value in fields.items():
            slot=self._new('field_slot',name); self._edge(obj,'field',slot); self._edge(slot,'value_of',value)
        return obj

    def array(self, values):
        values=tuple(values)
        for value in values: self._value(value)
        obj=self._new('array',len(values))
        for index,value in enumerate(values):
            slot=self._new('index_slot',index); self._edge(obj,'index',slot); self._edge(slot,'value_of',value)
        return obj

    def frame(self, parent_scope=None):
        if parent_scope is not None: self._check(parent_scope,{'frame'})
        frame=self._new('frame'); self._bindings[frame]={}; self._parents[frame]=parent_scope
        if parent_scope is not None: self._edge(frame,'parent_scope',parent_scope)
        return frame

    def bind(self,name,value,scope):
        self._check(scope,{'frame'}); self._value(value)
        if not isinstance(name,str) or not name or name in self._bindings[scope]:
            raise RuntimeFault('invalid or duplicate binding')
        binding=self._new('binding',name); self._bindings[scope][name]=binding
        self._edge(scope,'binding',binding); self._edge(binding,'binds',value)
        return binding

    def resolve_name(self,name,scope):
        self._check(scope,{'frame'})
        current=scope
        while current is not None:
            if name in self._bindings[current]:
                result=self._bindings[current][name]
                self.trace.append({'primitive':'resolve_name','scope':scope,'name':name,'result':result})
                return result
            current=self._parents[current]
        raise RuntimeFault(f'unbound name {name!r}')

    def value_of(self,node):
        self._check(node,{'binding','field_slot','index_slot','argument','returned'})
        values=[e.target for e in self.edges if e.source==node and e.relation in {'binds','value_of'}]
        if len(values)!=1: raise RuntimeFault('node has no unique value')
        return values[0]

    def _read(self,obj,key,kind,relation):
        self._check(obj,{kind})
        slots=[e.target for e in self.edges if e.source==obj and e.relation==relation and self.nodes[e.target].payload==key]
        if len(slots)!=1: raise RuntimeFault(f'no {relation} {key!r}')
        value=self.value_of(slots[0])
        self.trace.append({'primitive':f'read_{relation}','source':obj,'slot':slots[0],'result':value})
        return value

    def read_field(self,obj,field):
        if not isinstance(field,str): raise RuntimeFault('field must be text')
        return self._read(obj,field,'record','field')

    def read_index(self,array,index):
        if type(index) is not int or index<0: raise RuntimeFault('index must be nonnegative integer')
        return self._read(array,index,'array','index')

    def builtin(self,name):
        if name not in {'add','sub','mul','neg'}: raise RuntimeFault('unknown builtin')
        if name not in self._builtins:
            node=self._new('function',name); self._builtins[name]=node; self._functions[node]=('builtin',name)
        return self._builtins[name]

    def define_function(self,name,parameters,body):
        """Body is a parsed AST, never Python source or executable host callback."""
        if len(set(parameters))!=len(parameters): raise RuntimeFault('duplicate parameters')
        node=self._new('function',name); self._functions[node]=('user',tuple(parameters),body)
        return node

    def return_value(self,frame,value):
        self._check(frame,{'frame'}); self._value(value)
        if any(e.source==frame and e.relation=='returns' for e in self.edges):
            raise RuntimeFault('frame already returned')
        returned=self._new('returned'); self._edge(returned,'value_of',value); self._edge(frame,'returns',returned)
        self.trace.append({'primitive':'return','frame':frame,'result':returned})
        return returned

    def call(self,function,args,scope=None):
        self._check(function,{'function'}); args=tuple(args)
        for arg in args: self._value(arg)
        if scope is not None: self._check(scope,{'frame'})
        spec=self._functions[function]
        arity=(1 if spec[1]=='neg' else 2) if spec[0]=='builtin' else len(spec[1])
        if len(args)!=arity: raise RuntimeFault('wrong argument count')
        if function in self._active: raise RuntimeFault('recursion is unsupported')
        if spec[0]=='builtin':
            for arg in args: self._check(arg,{'scalar'})
        with self.transaction():
            call=self._new('call'); self._edge(call,'callee',function)
            # No closures: a user function has a fresh, parentless lexical frame.
            frame=self.frame(); self._edge(call,'frame',frame)
            for index,arg in enumerate(args):
                slot=self._new('argument',index); self._edge(call,f'argument[{index}]',slot); self._edge(slot,'value_of',arg)
            self.trace.append({'primitive':'call','call':call,'function':function,'frame':frame,'args':args,'node_count':len(self.nodes)})
            self._active.append(function)
            if spec[0]=='builtin':
                values=[self.nodes[a].payload for a in args]; name=spec[1]
                value=self.scalar(-values[0] if name=='neg' else values[0]+values[1] if name=='add' else values[0]-values[1] if name=='sub' else values[0]*values[1])
            else:
                for name,arg in zip(spec[1],args): self.bind(name,arg,frame)
                # Function names are global definitions, never captured variable bindings.
                for fid,definition in list(self._functions.items()):
                    name=self.nodes[fid].payload
                    if name not in self._bindings[frame]: self.bind(name,fid,frame)
                from .tiny_language import evaluate_statements
                value=evaluate_statements(spec[2],self,frame,in_function=True)
            self._active.pop()
            returned=self.return_value(frame,value); self._edge(call,'returns',returned)
            return value

    def invoke(self,primitive,*args):
        if primitive not in PRIMITIVES: raise RuntimeFault('unknown primitive')
        return getattr(self,PRIMITIVES[primitive].executor)(*args)

    def to_python(self,node):
        value=self._value(node)
        if value.kind=='scalar': return value.payload
        if value.kind in {'record','array'}:
            relation='field' if value.kind=='record' else 'index'
            slots=[self.nodes[e.target] for e in self.edges if e.source==node and e.relation==relation]
            if value.kind=='record': return {slot.payload:self.to_python(self.value_of(slot.id)) for slot in slots}
            return [self.to_python(self.value_of(slot.id)) for slot in sorted(slots,key=lambda s:s.payload)]
        return {'function':value.payload,'node':node}
