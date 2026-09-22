"""Small recursive-descent language: let, fn, return, blocks and postfix access.

No assignment, loops, recursion, closures, exceptions, or host-language eval.
Semicolons are optional between statements when the next token is unambiguous.
"""
from dataclasses import dataclass
import re
from .tiny_runtime import Runtime, RuntimeFault

_TOKEN = re.compile(r'\s*(?:(\d+(?:\.\d+)?)|([A-Za-z_][A-Za-z_0-9]*)|([{}\[\]().,:;=\-]))')


class Parser:
    def __init__(self,source):
        self.tokens=[]; offset=0
        while offset<len(source):
            if not source[offset:].strip(): break
            match=_TOKEN.match(source,offset)
            if match is None: raise RuntimeFault(f'invalid token at offset {offset}')
            self.tokens.append(next(g for g in match.groups() if g is not None)); offset=match.end()
        self.tokens.append('<eof>'); self.position=0

    def peek(self): return self.tokens[self.position]

    def take(self,expected=None):
        token=self.peek()
        if token=='<eof>' or (expected is not None and token!=expected):
            raise RuntimeFault(f'expected {expected!r}, found {token!r}')
        self.position+=1; return token

    def name(self):
        name=self.take()
        if not re.fullmatch('[A-Za-z_][A-Za-z_0-9]*',name) or name in {'let','fn','return'}:
            raise RuntimeFault('expected identifier')
        return name

    def statements(self,end='<eof>'):
        out=[]
        while self.peek()!=end:
            token=self.peek()
            if token=='let':
                self.take(); name=self.name(); self.take('='); out.append(('let',name,self.expression()))
            elif token=='fn':
                self.take(); name=self.name(); self.take('('); params=[]
                if self.peek()!=')':
                    while True:
                        params.append(self.name())
                        if self.peek()!=',': break
                        self.take(',')
                self.take(')'); self.take('{'); body=self.statements('}'); self.take('}')
                out.append(('fn',name,tuple(params),body))
            elif token=='return': self.take(); out.append(('return',self.expression()))
            elif token=='{':
                self.take(); body=self.statements('}'); self.take('}'); out.append(('block',body))
            else: out.append(('expr',self.expression()))
            if self.peek()==';': self.take()
        return tuple(out)

    def expression(self):
        token=self.take()
        if token=='-':
            number=self.take()
            if not re.fullmatch(r'\d+(?:\.\d+)?',number): raise RuntimeFault('minus only prefixes literals')
            node=('scalar',-float(number) if '.' in number else -int(number))
        elif re.fullmatch(r'\d+(?:\.\d+)?',token): node=('scalar',float(token) if '.' in token else int(token))
        elif token=='{':
            fields=[]
            if self.peek()!='}':
                while True:
                    name=self.name(); self.take(':'); fields.append((name,self.expression()))
                    if self.peek()!=',': break
                    self.take(',')
            self.take('}')
            if len({name for name,_ in fields})!=len(fields): raise RuntimeFault('duplicate record field')
            node=('record',tuple(fields))
        elif token=='[':
            values=[]
            if self.peek()!=']':
                while True:
                    values.append(self.expression())
                    if self.peek()!=',': break
                    self.take(',')
            self.take(']'); node=('array',tuple(values))
        elif token=='(':
            node=self.expression(); self.take(')')
        elif re.fullmatch('[A-Za-z_][A-Za-z_0-9]*',token) and token not in {'let','fn','return'}: node=('name',token)
        else: raise RuntimeFault(f'invalid expression {token!r}')
        while self.peek() in {'.','[','('}:
            suffix=self.take()
            if suffix=='.': node=('field',node,self.name())
            elif suffix=='[':
                index=self.expression(); self.take(']'); node=('index',node,index)
            else:
                args=[]
                if self.peek()!=')':
                    while True:
                        args.append(self.expression())
                        if self.peek()!=',': break
                        self.take(',')
                self.take(')'); node=('call',node,tuple(args))
        return node


def parse(source):
    return Parser(source).statements()


def evaluate_expression(expr,runtime,scope):
    tag=expr[0]
    if tag=='scalar': return runtime.scalar(expr[1])
    if tag=='name': return runtime.value_of(runtime.resolve_name(expr[1],scope))
    if tag=='record': return runtime.record({name:evaluate_expression(value,runtime,scope) for name,value in expr[1]})
    if tag=='array': return runtime.array([evaluate_expression(v,runtime,scope) for v in expr[1]])
    if tag=='field': return runtime.read_field(evaluate_expression(expr[1],runtime,scope),expr[2])
    if tag=='index':
        array=evaluate_expression(expr[1],runtime,scope); index=evaluate_expression(expr[2],runtime,scope)
        runtime._check(index,{'scalar'})
        return runtime.read_index(array,runtime.nodes[index].payload)
    if tag=='call':
        function=evaluate_expression(expr[1],runtime,scope)
        args=[evaluate_expression(v,runtime,scope) for v in expr[2]]
        return runtime.call(function,args,scope)
    raise RuntimeFault('unknown expression')


class _Return(Exception):
    def __init__(self,value): self.value=value


def _statements(statements,runtime,scope,in_function):
    result=None
    for statement in statements:
        tag=statement[0]
        if tag=='let':
            result=evaluate_expression(statement[2],runtime,scope); runtime.bind(statement[1],result,scope)
        elif tag=='fn':
            if scope!=runtime.root_scope: raise RuntimeFault('nested function definitions unsupported')
            result=runtime.define_function(statement[1],statement[2],statement[3]); runtime.bind(statement[1],result,scope)
        elif tag=='block': result=_statements(statement[1],runtime,runtime.frame(scope),in_function)
        elif tag=='return':
            if not in_function: raise RuntimeFault('return outside function')
            raise _Return(evaluate_expression(statement[1],runtime,scope))
        else: result=evaluate_expression(statement[1],runtime,scope)
    if result is None: raise RuntimeFault('empty program or function body')
    return result


def evaluate_statements(statements,runtime,scope,in_function=False):
    try: return _statements(statements,runtime,scope,in_function)
    except _Return as returned: return returned.value


@dataclass
class ExecutionResult:
    value: object
    value_id: str
    runtime: Runtime
    trace: list[dict]


def execute(source):
    program=parse(source); runtime=Runtime()
    for name in ('add','sub','mul','neg'): runtime.bind(name,runtime.builtin(name),runtime.root_scope)
    value=evaluate_statements(program,runtime,runtime.root_scope)
    python_value=runtime.to_python(value)
    return ExecutionResult(python_value,value,runtime,list(runtime.trace))
