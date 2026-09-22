"""Controlled surface-to-runtime tasks; gold actions never enter model inputs.

Public clauses are already segmented in execution order. Continuous reference keys
are noisy lexical/entity features, not natural-language understanding. All initial
runtime nodes/edges/payloads are observable; computed call/return nodes are absent.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .tiny_runtime import Runtime

OPS = ('resolve', 'field', 'index', 'add', 'sub', 'rsub', 'mul')
FAMILIES = ('nested', 'scope', 'composition', 'ordering', 'aliases', 'mixed')
VOCAB_SIZE = 64
N_TYPES = 8
RELATIONS = ('binds', 'value_of', 'field', 'index', 'parent_scope', 'binding')
N_RELATIONS = len(RELATIONS)
RESULT_MIN, RESULT_MAX, RESULT_SCALE, OUTPUT_CLASSES = -64, 64, 64., 129
KINDS = ('scalar', 'record', 'array', 'binding', 'field_slot', 'index_slot', 'function', 'frame')


# Human-readable surface vocabulary. Reference features are a separate channel;
# these words are generated tokens, not evidence of general natural-language parsing.
TOKEN_WORDS = {0: '[pad]', 32: 'please', 33: 'the referenced item', 34: 'now'}
TOKEN_WORDS.update({1+i: word for i,word in enumerate(('resolve','read field','read index','add','subtract','subtract from','multiply'))})
TOKEN_WORDS.update({16+i: word for i,word in enumerate(('find name','get property','take element','plus','minus','reverse subtract','times'))})


def decode_surface(tokens, reference_label=None):
    phrase = ' '.join(TOKEN_WORDS.get(int(token), f'[token {int(token)}]') for token in tokens if int(token))
    return phrase if reference_label is None else f'{phrase}; reference: {reference_label}'


def render_output(predicted_class, style):
    """Fixed text decoder after a learned output class, not a learned text generator."""
    value = int(predicted_class)
    if style == 0 and 0 <= value < OUTPUT_CLASSES:
        return f'the value is {value + RESULT_MIN}'
    if style == 1 and value in (0,1):
        return ('no', 'yes')[value]
    if style == 2 and value in (0,1,2):
        return ('negative', 'zero', 'positive')[value]
    return '[invalid output class for requested style]'


def _surface_tokens(op, template):
    cue = 16+op if template == 'paraphrase' else 1+op
    return [34,33,cue,32] if template == 'heldout' else [32,cue,33,34]


@dataclass
class RuntimeTask:
    runtime: Runtime
    scope: str
    candidates: tuple[str, ...]
    gold_actions: tuple[tuple[int, int], ...]
    gold_result: int | None
    family: str
    style: int
    comparison: int
    surface: tuple[str, ...]
    gold_registers: tuple[str, ...]
    wrong_runtime: Runtime | None = None
    expected_valid: bool = True
    invalid_reason: str | None = None


def output_target(result: int, style: int, comparison: int) -> int:
    if style == 0:
        return result - RESULT_MIN
    if style == 1:
        return int(result > comparison)
    return int(result > 0) + int(result >= 0)


def _slot(rt, obj, relation, payload):
    return next(e.target for e in rt.edges if e.source == obj and e.relation == relation
                and rt.nodes[e.target].payload == payload)


def _one(seed, nodes, depth, family, style):
    rng = random.Random(seed)
    rt = Runtime(); scope = rt.root_scope
    for name in ('add', 'sub', 'mul'):
        rt.builtin(name)
    names = [f'v_{rng.getrandbits(48):012x}' for _ in range(nodes + depth + 5)]
    actions = []
    decoy_binding = None
    if family in ('nested', 'aliases', 'mixed'):
        expected_result = rng.randint(-12, 12)
        value = rt.scalar(expected_result); reverse = []; shape = []
        for step in range(depth):
            if step % 2:
                index = rng.randrange(3)
                children = [rt.scalar(rng.randint(-12, 12)) for _ in range(3)]
                children[index] = value
                value = rt.array(children)
                reverse.append((OPS.index('index'), _slot(rt,value,'index',index)))
                shape.append(('index',index))
            else:
                field = names[step+1]
                value = rt.record({field: value, f'junk_{step}': rt.scalar(rng.randint(-12,12))})
                reverse.append((OPS.index('field'), _slot(rt,value,'field',field)))
                shape.append(('field',field))
        binding = rt.bind(names[0], value, scope)
        if family == 'aliases':
            binding = rt.bind(names[-1], value, scope)
        actions = [(0,binding)] + list(reversed(reverse))
        # Matched decoy prevents the last field name from identifying a unique
        # answer globally. Both chains expose identical semantic path labels.
        decoy_result = rng.choice([n for n in range(-12,13) if n != expected_result])
        decoy = rt.scalar(decoy_result)
        for step,(kind,label) in enumerate(shape):
            if kind == 'index':
                children = [rt.scalar(rng.randint(-12,12)) for _ in range(3)]
                children[label] = decoy
                decoy = rt.array(children)
            else:
                decoy = rt.record({label:decoy, f'junk_{step}':rt.scalar(rng.randint(-12,12))})
        decoy_binding = rt.bind(f'decoy_{rng.getrandbits(48):012x}',decoy,scope)
    else:
        start = rng.randint(-4,4)
        binding = rt.bind(names[0], rt.scalar(start), scope)
        if family == 'scope':
            scope = rt.frame(scope)
            start = rng.randint(-4,4)
            binding = rt.bind(names[0], rt.scalar(start), scope)
        actions = [(0,binding)]
        # add/sub and +/-1 multiplication keep every intermediate in [-64,64]
        current = start
        for step in range(depth):
            op = rng.choice(('add','sub','rsub','mul'))
            operand = rng.choice((-1,1)) if op == 'mul' else rng.randint(-1,1)
            if family == 'ordering':
                op = 'sub' if step % 2 == 0 else 'rsub'
                operand = rng.randint(-1,1)
            predicted = {'add': lambda: current+operand, 'sub':lambda:current-operand,
                         'rsub':lambda:operand-current,'mul':lambda:current*operand}[op]()
            if abs(predicted)>60: operand=0; op='add'; predicted=current
            actions.append((OPS.index(op),rt.scalar(operand))); current=predicted
        expected_result = current
    # Each distractor is an independently named visible scalar binding.
    for name in names[depth+2:depth+2+nodes]:
        rt.bind(name,rt.scalar(rng.randint(-12,12)),rt.root_scope)
    candidates=list(rt.nodes); rng.shuffle(candidates)
    ids={node:i for i,node in enumerate(candidates)}
    action_ids=tuple((op,ids[node]) for op,node in actions)
    # Labels come from generator-side scalar arithmetic / the selected leaf.
    # Execution independently checks that the constructed world matches them.
    from .runtime_execution import run_actions
    trial = RuntimeTask(rt,scope,tuple(candidates),action_ids,0,family,style,rng.randint(-8,8),(),())
    executed=run_actions(trial,action_ids)
    assert executed.valid, executed.trace
    assert executed.result == expected_result, (executed.result, expected_result)
    if decoy_binding is not None:
        decoy_actions = ((0,ids[decoy_binding]),) + action_ids[1:]
        decoy_executed = run_actions(trial,decoy_actions)
        assert decoy_executed.valid and decoy_executed.result == decoy_result
    trial.gold_result=expected_result
    trial.gold_registers=tuple(x['register'] for x in executed.trace)
    trial.surface=tuple(f'{OPS[op]} {rt.nodes[node].payload}' for op,node in actions)
    return trial


def make_batch(batch_size=32, nodes=8, depth=4, seed=0, noise=.1,
               family='all', template='canonical', device='cpu', corruption=0.,
               key_dim=16, style=None, ambiguous=False, invalid=False):
    """Return public tensors, separate supervision, and CPU exact-runtime objects.

    ``nodes`` is distractor-object count, NOT total runtime graph size. ``depth``
    counts operations after one initial resolution. Opaque IDs are not features.
    Candidate features expose numeric slot indices, lexical names as random keys,
    root/current frame flags, node types and every initial directed relation.
    """
    if batch_size<1 or nodes<1 or depth<0 or noise<0 or not 0<=corruption<=1:
        raise ValueError('invalid task dimensions/noise/corruption')
    if family not in (*FAMILIES,'all'):
        raise ValueError(f'unknown family {family}')
    if template not in ('canonical','paraphrase','heldout'):
        raise ValueError('unknown template')
    tasks=[_one(seed*100003+i,nodes,depth,FAMILIES[i%len(FAMILIES)] if family=='all' else family,
                random.Random(seed*9176+i+41).randrange(3) if style is None else style) for i in range(batch_size)]
    B,C,D=batch_size,max(len(t.candidates) for t in tasks),depth+1
    gen=torch.Generator().manual_seed(seed+918273)
    public={
        'surface':torch.zeros(B,D,4,dtype=torch.long),
        'reference':torch.zeros(B,D,key_dim),
        'candidate_keys':torch.zeros(B,C,key_dim),
        'candidate_values':torch.zeros(B,C),
        'candidate_types':torch.zeros(B,C,dtype=torch.long),
        'candidate_payload':torch.zeros(B,C,3),
        'candidate_mask':torch.zeros(B,C,dtype=torch.bool),
        'candidate_groups':torch.full((B,C),-1,dtype=torch.long),
        'adjacency':torch.zeros(B,N_RELATIONS,C,C),
        'step_mask':torch.ones(B,D,dtype=torch.bool),
        'style':torch.tensor([t.style for t in tasks]),
        'comparison':torch.tensor([t.comparison/RESULT_SCALE for t in tasks]),
    }
    gold={'ops':torch.zeros(B,D,dtype=torch.long),'selectors':torch.zeros(B,D,dtype=torch.long),
          'result':torch.tensor([t.gold_result for t in tasks]),
          'output':torch.tensor([output_target(t.gold_result,t.style,t.comparison) for t in tasks]),
          'selector_mask':torch.zeros(B,D,C+1,dtype=torch.bool),
          'expected_valid':torch.ones(B,dtype=torch.bool)}
    for row,t in enumerate(tasks):
        ids={node:i for i,node in enumerate(t.candidates)}
        keys=F.normalize(torch.randn(len(ids),key_dim,generator=gen),dim=-1)
        # Name-equivalent shadow bindings share observable lexical features;
        # their frame edges/flags distinguish the runtime locations.
        lexical={}
        groups={}
        for i,node_id in enumerate(t.candidates):
            node=t.runtime.nodes[node_id]
            if node.kind in ('binding','field_slot','index_slot','scalar','function'):
                symbol=(node.kind,node.payload)
                if symbol in lexical: keys[i]=keys[lexical[symbol]]
                else: lexical[symbol]=i
            symbol=(node.kind,node.payload) if node.kind in ('binding','field_slot','index_slot','scalar','function') else (node.kind,node_id)
            if symbol not in groups: groups[symbol]=len(groups)
            public['candidate_groups'][row,i]=groups[symbol]
            public['candidate_types'][row,i]=KINDS.index(node.kind)
            if node.kind=='scalar': public['candidate_values'][row,i]=float(node.payload)/RESULT_SCALE
            if node.kind=='index_slot': public['candidate_payload'][row,i,0]=node.payload/RESULT_SCALE
            public['candidate_payload'][row,i,1]=float(node_id==t.scope)
            public['candidate_payload'][row,i,2]=float(node_id==t.runtime.root_scope)
        public['candidate_keys'][row,:len(ids)]=keys
        public['candidate_mask'][row,:len(ids)]=True
        for edge in t.runtime.edges:
            public['adjacency'][row,RELATIONS.index(edge.relation),ids[edge.source],ids[edge.target]]=1
        for j,(op,selector) in enumerate(t.gold_actions):
            gold['ops'][row,j]=op; gold['selectors'][row,j]=selector
            target=t.runtime.nodes[t.candidates[selector]]
            for k,node_id in enumerate(t.candidates):
                node=t.runtime.nodes[node_id]
                gold['selector_mask'][row,j,k]=(node.kind,node.payload)==(target.kind,target.payload)
            # Cue tokens correspond to operation words, not supplied operation IDs.
            # Heldout templates reorder trained lexical pieces; paraphrases use
            # synonymous token 16+op, observed during mixed-template training.
            public['surface'][row,j]=torch.tensor(_surface_tokens(op,template))
            public['reference'][row,j]=keys[selector]+noise*torch.randn(key_dim,generator=gen)
        if ambiguous or invalid:
            t.expected_valid=False
            t.invalid_reason='ambiguous_unbound' if ambiguous else 'schema'
            t.gold_result=None
            gold['expected_valid'][row]=False
            gold['result'][row]=0; gold['output'][row]=0
            actions=list(t.gold_actions)
            if ambiguous:
                public['reference'][row,0].zero_()
                gold['selectors'][row,0]=C
                gold['selector_mask'][row,0].zero_(); gold['selector_mask'][row,0,C]=True
                actions[0]=(0,C)
            else:
                # Deliberately ill-typed: field access before any object binding.
                gold['ops'][row,0]=1
                public['surface'][row,0]=torch.tensor(_surface_tokens(1,template))
                actions[0]=(1,actions[0][1])
            t.gold_actions=tuple(actions)
        t.surface=tuple(decode_surface(public['surface'][row,j].tolist(),
                        '[missing cue]' if selector>=len(t.candidates) else
                        repr(t.runtime.nodes[t.candidates[selector]].payload))
                        for j,(_,selector) in enumerate(t.gold_actions))
        if corruption:
            # Wrong-world intervention permutes value destinations while keeping
            # symbolic and tensor worlds identical; labels remain clean.
            wrong=copy.deepcopy(t.runtime)
            from .tiny_runtime import Edge
            value_ids=[node for node in t.candidates if wrong.nodes[node].kind=='scalar']
            rng=random.Random(seed*7919+row)
            changed=[]
            for edge in wrong.edges:
                if edge.relation in ('value_of','binds') and edge.target in value_ids and rng.random()<corruption:
                    edge=Edge(edge.source,edge.relation,rng.choice(value_ids))
                changed.append(edge)
            wrong.edges=changed; t.wrong_runtime=wrong
            public['adjacency'][row].zero_()
            for edge in changed:
                public['adjacency'][row,RELATIONS.index(edge.relation),ids[edge.source],ids[edge.target]]=1
    return {'public':{k:v.to(device) for k,v in public.items()},
            'gold':{k:v.to(device) for k,v in gold.items()},'tasks':tasks}
