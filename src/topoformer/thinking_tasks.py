"""Controlled progressive expressions, separate from the actual TCN benchmark.

Only PublicEpisode crosses the actor boundary. Literal memory, named result
capacity and the three evidence arrivals are supplied priors; rewrite scheduling
is learned by default. Targets never constitute an actor action menu.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import random
from .semantic_graph import SemanticGraph, SemanticNode, SemanticEdge
from .thinking_runtime import Candidate, ValueRegister

OPS = ('add', 'sub', 'mul', 'compare')
TEMPLATES = ('canonical', 'infix', 'prefix', 'reordered', 'lexical')
GENERATOR_VERSION = 'controlled-expressions-v2'

@dataclass(frozen=True)
class PublicFrame:
    tokens: tuple[str, ...]

@dataclass(frozen=True)
class PublicEpisode:
    initial_values: tuple[ValueRegister, ...]
    frames: tuple[PublicFrame, ...]
    context_tokens: tuple[str, ...]
    output_ids: tuple[str, str]
    register_ids: tuple[str, ...]
    operation_ids: tuple[str, ...]
    scheduling: str = 'learned'

@dataclass(frozen=True)
class CandidateTarget:
    candidate: Candidate
    readiness: float

@dataclass(frozen=True)
class HypothesisSet:
    """Uniform finite posterior after filtering explicit observed evidence."""
    candidates: tuple[Candidate, ...]

@dataclass(frozen=True)
class GoldEpisode:
    trace: tuple[tuple[Candidate, ...], ...]
    readiness: tuple[tuple[CandidateTarget, ...], ...]
    hypotheses: tuple[tuple[HypothesisSet, ...], ...]
    result: int
    answer: int
    graph: SemanticGraph
    semantic_digest: str
    expected_values: tuple[tuple[str, int | bool], ...]

@dataclass(frozen=True)
class Episode:
    public: PublicEpisode
    gold: GoldEpisode


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def render_frames(statements, template='canonical'):
    """Render observable (id, primitive alternatives, ordered argument options).

    No result, graph, trace, answer, selected hypothesis or target is accepted.
    Each frame represents the whole visible expression, never a scheduled action.
    """
    if template not in TEMPLATES:
        raise ValueError('unknown template')
    rendered = []
    lexical = {'add':'plus', 'sub':'minus', 'mul':'times', 'compare':'less_than'}
    for frame in statements:
        lines = []
        for identity, primitives, left, right in frame:
            op = '|'.join(lexical.get(x,x) if template == 'lexical' else x for x in primitives)
            a,b = '|'.join(left), '|'.join(right)
            if template == 'infix': line = f'{identity} = ( {a} {op} {b} )'
            elif template == 'prefix': line = f'( {op} {a} {b} ) -> {identity}'
            elif template == 'reordered': line = f'right {b} left {a} operation {op} destination {identity}'
            elif template == 'lexical': line = f'store {op} of {a} and {b} as {identity}'
            else: line = f'{identity} := {op} left {a} right {b}'
            lines.append(line)
        if template == 'reordered': lines.reverse()
        rendered.append(PublicFrame(tuple(' ; '.join(lines).split())))
    return tuple(rendered)


def _posterior_targets(sets, available):
    targets = []
    for options in sets:
        for candidate in options.candidates:
            # Explicit uniform hypothesis mass, independent from formal runtime
            # validation and from the hidden answer. Null references have zero.
            mass = sum(c.primitive == candidate.primitive and c.arguments == candidate.arguments
                       for c in options.candidates) / len(options.candidates)
            ready = mass if all(a in available for a in candidate.arguments) else 0.0
            targets.append(CandidateTarget(candidate, ready))
    return tuple(targets)


def readiness_targets(episode: Episode, frame_index: int, available_ids, context_visible=False):
    """Training/scoring only; caller passes actual currently available memory IDs."""
    available = set(available_ids)
    result = _posterior_targets(episode.gold.hypotheses[frame_index], available)
    if context_visible:
        result += _posterior_targets((HypothesisSet(episode.gold.trace[-1]),), available)
    return result


def generate_episode(seed=0, depth=2, distractors=2, template='canonical', context=0,
                     scheduling='learned', motif='parallel', operator_composition='any'):
    if not 1 <= depth <= 32 or distractors < 0 or context not in (0,1):
        raise ValueError('invalid depth/distractors/context')
    if scheduling not in ('learned','supplied'):
        raise ValueError('scheduling must explicitly be learned or supplied')
    if motif not in ('parallel','cross') or operator_composition not in ('any','train','heldout'):
        raise ValueError('unknown motif or operator composition')
    if depth < 2 and (motif == 'cross' or operator_composition == 'heldout'):
        raise ValueError('cross motif and heldout composition require depth >= 2')
    rng = random.Random(seed)
    used = set()
    def name(prefix):
        while True:
            identity = prefix + format(rng.getrandbits(40),'010x')
            if identity not in used:
                used.add(identity); return identity
    values = []
    def literal(value):
        identity = name('v')
        values.append(ValueRegister(identity,value,'integer',('supplied_literal',)))
        return identity
    current = [literal(rng.randint(-6,6)), literal(rng.randint(-6,6))]
    numerical = [values[0].value, values[1].value]
    trace, clauses, expected = [], [], []
    previous_ops = [None,None]
    for level in range(depth):
        pair = []
        previous_ids, previous_values = tuple(current), tuple(numerical)
        for lane in range(2):
            primitive = rng.choice(('add','sub','mul'))
            if operator_composition == 'train' and previous_ops[lane] == 'sub' and primitive == 'mul':
                primitive = rng.choice(('add','sub'))
            if operator_composition == 'heldout' and lane == 0 and level < 2:
                primitive = ('sub','mul')[level]
            bridge = motif == 'cross' and level == 1 and lane == 1
            if bridge and primitive == 'mul':
                primitive = rng.choice(('add','sub'))
            operand = rng.choice((-1,1))
            operand_id = literal(operand)
            alternative_id = literal(-operand)
            identity = name('c')
            if bridge:
                # A genuine DAG merge: both parent returns are read from the
                # previous microstep snapshot, never from this pair's writes.
                operand_id, alternative_id = previous_ids[0], previous_ids[1]
                operand = previous_values[0]
            arguments = (previous_ids[lane],operand_id)
            # Reverse subtraction is essential to ordered binding evaluation.
            if rng.randrange(2): arguments = arguments[::-1]
            candidate = Candidate(identity,primitive,arguments,1.0)
            a,b = (previous_values[lane],operand) if arguments[0] == previous_ids[lane] else (operand,previous_values[lane])
            value = a+b if primitive == 'add' else a-b if primitive == 'sub' else a*b
            current[lane] = 'result:'+identity; numerical[lane] = value
            expected.append((current[lane],value)); pair.append(candidate)
            alternatives = list((primitive, rng.choice([p for p in ('add','sub','mul') if p != primitive])))
            rng.shuffle(alternatives)
            uncertain_position = arguments.index(operand_id)
            option_ids = [operand_id,alternative_id]; rng.shuffle(option_ids)
            clauses.append((candidate,lane,tuple(alternatives),uncertain_position,tuple(option_ids)))
        trace.append(tuple(pair))
        previous_ops = [c.primitive for c in pair]
    for _ in range(distractors): literal(rng.randint(-8,8))
    dead = Candidate(name('c'),'add',(values[0].id,'unbound'),1.0)
    clauses.append((dead,1,('add','sub'),1,('unbound',)))
    rng.shuffle(clauses)
    statement_frames, hypotheses = [], []
    for stage in range(3):
        statements, sets = [], []
        for candidate,lane,primitive_options,position,argument_options in clauses:
            primitives = (candidate.primitive,) if stage == 2 or (stage == 1 and lane == 0) else primitive_options
            args = [(a,) for a in candidate.arguments]
            if stage == 0 and lane == 1: args[position] = argument_options
            statements.append((candidate.id,primitives,args[0],args[1]))
            possibilities = tuple(Candidate(candidate.id,op,(a,b),1.0)
                                  for op in primitives for a in args[0] for b in args[1])
            sets.append(HypothesisSet(possibilities))
        statement_frames.append(statements); hypotheses.append(tuple(sets))
    # Context is released only after both declared roots are present. It tells
    # the actor the ORDER of a subsequent comparison of returned values.
    compare_id = name('c')
    comparison_args = tuple(current if context == 0 else current[::-1])
    comparison = Candidate(compare_id,'compare',comparison_args,1.0)
    answer = int(numerical[context] < numerical[1-context])
    trace.append((comparison,)); expected.append(('result:'+compare_id,bool(answer)))
    # Both context alternatives are instructions over public return identities;
    # neither contains a numeric result or a boolean target.
    context_tokens = tuple(f'after returns compare left {comparison_args[0]} right {comparison_args[1]} destination {compare_id}'.split())
    all_candidates = [c for pair in trace for c in pair]
    operation_ids = [c.id for c in all_candidates] + [dead.id]; rng.shuffle(operation_ids)
    register_ids = [v.id for v in values] + ['result:'+c for c in operation_ids]
    rng.shuffle(register_ids); rng.shuffle(values)
    public = PublicEpisode(tuple(values),render_frames(statement_frames,template),context_tokens,
                           tuple(current),tuple(register_ids),tuple(operation_ids),scheduling)
    nodes = [SemanticNode(v.id,'integer',v.value,('supplied_literal',)) for v in values]
    edges = []
    for candidate in all_candidates:
        nodes.extend((SemanticNode(candidate.id,'operation',candidate.primitive),
                      SemanticNode('result:'+candidate.id,'return')))
        for index,argument in enumerate(candidate.arguments):
            edges.append(SemanticEdge(candidate.id,argument,'argument',index))
        edges.append(SemanticEdge(candidate.id,'result:'+candidate.id,'returns'))
    graph = SemanticGraph(tuple(nodes),tuple(edges),('result:'+compare_id,), 'controlled-dag-v1')
    ready = tuple(_posterior_targets(frame,{v.id for v in values}) for frame in hypotheses)
    gold = GoldEpisode(tuple(trace),ready,tuple(hypotheses),numerical[0],answer,graph,graph.digest(),tuple(expected))
    return Episode(public,gold)


def paired_contexts(seed=0, **kwargs):
    kwargs.pop('context',None)
    # Diagnostic pairs condition on unequal roots. Main training generation
    # does not reject by outcome, preserving the finite ambiguity prior.
    while True:
        first = generate_episode(seed,context=0,**kwargs)
        second = generate_episode(seed,context=1,**kwargs)
        if first.gold.answer != second.gold.answer:
            return first, second
        seed += 104729


def episode_hashes(episode):
    return {'generator_version':GENERATOR_VERSION,'public_sha256':_digest(asdict(episode.public)),
            'gold_sha256':_digest(asdict(episode.gold)), 'semantic_sha256':episode.gold.semantic_digest,
            'source_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()}


def audit_episode(episode):
    """Independent arithmetic replay without the protected runtime evaluator."""
    memory = {v.id:v.value for v in episode.public.initial_values}
    expected = dict(episode.gold.expected_values)
    for group in episode.gold.trace:
        updates = {}
        for candidate in group:
            if not all(a in memory for a in candidate.arguments):
                return {'valid':False,'reason':'missing_argument'}
            a,b = (memory[x] for x in candidate.arguments)
            if candidate.primitive == 'add': result = sum((a,b))
            elif candidate.primitive == 'sub': result = a + (-b)
            elif candidate.primitive == 'mul': result = a * b
            elif candidate.primitive == 'compare': result = a < b
            else: return {'valid':False,'reason':'unknown_primitive'}
            identity = 'result:'+candidate.id
            if expected.get(identity) != result: return {'valid':False,'reason':'value_mismatch'}
            updates[identity] = result
        memory.update(updates)
    valid = (memory[episode.public.output_ids[0]] == episode.gold.result and
             int(memory['result:'+episode.gold.trace[-1][0].id]) == episode.gold.answer)
    return {'valid':valid,'rewrite_count':sum(map(len,episode.gold.trace)),
            'maximum_absolute_value':max(abs(int(v)) for v in memory.values())}
