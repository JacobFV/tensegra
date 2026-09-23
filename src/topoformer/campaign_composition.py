"""P01 v1: complete public typed dispatch, preserving retention_data witnesses.

Standalone acquisition runner is separate; no C01 composition runner. Gold is confined to acquisition labels
and a private fidelity reference, neither accepted by execute_proposal.
"""
import random
import math
import torch
from .retention_data import make_batch
from .interface_proposals import ProposalModel, PRIMITIVES
from .thinking_runtime import ProtectedSession, ValueRegister, Candidate

VERSION = 'p01-v1'


def make_lowering_batch(seed, count, distractors=2):
    source = make_batch(seed, count, feature_dim=32, distractors=distractors)
    rng = random.Random(seed + 47000000)
    public_rows, labels = [], []
    for row in range(count):
        pub, gold = source['public'], source['targets']
        event = pub['event']
        keys = torch.cat((pub['argument_keys'][row], pub['provenance_keys'][row]))
        op = int(gold['operation'][row]); typ = int(gold['type'][row])
        a, b = int(gold['argument0'][row]), int(gold['argument1'][row])
        dest = 7 + int(gold['provenance'][row])
        # Public literal types determine Python execution semantics. Integral
        # floats remain floats, including float zero. Compare uses (int, float).
        values = [rng.randint(-8, 8) for _ in range(6)] + [None] * 5
        x, y = event['operand_values'][row, 0].tolist()
        values[a] = float(x) if typ == 1 else int(x)
        if op != 3:
            values[b] = float(y) if typ in (1, 2) else int(y)
        instructions = []
        for d in range(7, 11):
            if d == dest:
                instructions.append((d, op, a, b))
            else:
                other_op = rng.randrange(5)
                left, right = rng.sample(range(6), 2)
                instructions.append((d, other_op, left, 6 if other_op == 3 else right))
        rng.shuffle(instructions)
        order = list(range(11)); rng.shuffle(order)
        inverse = {old: new for new, old in enumerate(order)}
        public_rows.append(dict(
            keys=keys[order].clone(), values=tuple(values[i] for i in order),
            names=tuple(f'p01_{seed}_{row}_{rng.getrandbits(64):016x}' for _ in order),
            roles=tuple('operand' if i < 6 else 'null' if i == 6 else 'destination' for i in order),
            instruction_destinations=torch.stack([keys[d] for d, _, _, _ in instructions]),
            instruction_arguments=torch.stack([torch.stack((keys[l], keys[r])) for _, _, l, r in instructions]),
            instruction_cues=torch.nn.functional.one_hot(torch.tensor([o for _, o, _, _ in instructions]), 5).float(),
            query_destination=keys[dest].clone(), query=pub['query'][row].clone(),
            argument_keys=pub['argument_keys'][row].clone(), provenance_keys=pub['provenance_keys'][row].clone(),
            distractors=pub['distractors'][row].clone()))
        labels.append((op, tuple(inverse[i] for i in (dest, a, b))))
    return dict(public=public_rows, labels=labels, reference=source)


def public_schema_error(row):
    """Public completeness/uniqueness checks; no correct binding is consulted."""
    try:
        keys = row['keys']
        if keys.shape != (11, 32) or not torch.isfinite(keys).all():
            return 'key schema'
        if len({tuple(k.tolist()) for k in keys}) != 11:
            return 'ambiguous keys'
        if len(row['names']) != 11 or len(set(row['names'])) != 11:
            return 'ambiguous names'
        if len(row['values']) != 11 or sorted(row['roles']) != sorted(['operand'] * 6 + ['null'] + ['destination'] * 4):
            return 'table schema'
        lookup = {tuple(k.tolist()): i for i, k in enumerate(keys)}
        for i, (value, role) in enumerate(zip(row['values'], row['roles'])):
            if role == 'operand' and (type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 16):
                return 'numeric literal schema'
            if role != 'operand' and value is not None:
                return 'nonoperand literal'
            if role == 'null' and keys[i].count_nonzero():
                return 'absent key schema'
        destinations, arguments, cues = (row[k] for k in ('instruction_destinations', 'instruction_arguments', 'instruction_cues'))
        if destinations.shape != (4, 32) or arguments.shape != (4, 2, 32) or cues.shape != (4, 5):
            return 'incomplete instructions'
        destination_ids = [lookup[tuple(k.tolist())] for k in destinations]
        if len(set(destination_ids)) != 4 or any(row['roles'][i] != 'destination' for i in destination_ids):
            return 'ambiguous destinations'
        if tuple(row['query_destination'].tolist()) not in {tuple(k.tolist()) for k in destinations}:
            return 'missing requested instruction'
        for args, cue in zip(arguments, cues):
            if not (((cue == 0) | (cue == 1)).all() and cue.sum() == 1):
                return 'operation cue schema'
            ids = [lookup[tuple(k.tolist())] for k in args]
            if row['roles'][ids[0]] != 'operand' or row['roles'][ids[1]] != ('null' if int(cue.argmax()) == 3 else 'operand'):
                return 'instruction role schema'
        query = row['query']
        if query.shape != (2,) or not torch.isfinite(query).all() or abs(float(query[0])) > 8 or float(query[1]) not in (0., 1.):
            return 'query schema'
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return 'public schema'
    return None


def model_inputs(rows):
    """Only public fields; numeric table is available but this actor ignores it."""
    for row in rows:
        error = public_schema_error(row)
        if error:
            raise ValueError(error)
    names = ('keys', 'instruction_destinations', 'instruction_arguments', 'instruction_cues', 'query_destination', 'query')
    result = {name: torch.stack([r[name] for r in rows]) for name in names}
    result['valid'] = torch.ones(len(rows), 11, dtype=torch.bool)
    result['instruction_valid'] = torch.ones(len(rows), 4, dtype=torch.bool)
    # Numeric table is separately supplied for matched baseline and execution;
    # zero padding is distinguished by explicit types (0 int, 1 float, -1 absent).
    result['operand_values'] = torch.tensor([[0 if v is None else v for v in r['values']] for r in rows]).float()
    result['operand_types'] = torch.tensor([[-1 if v is None else int(type(v) is float) for v in r['values']] for r in rows])
    return result


def make_model():
    return ProposalModel(key_dim=32, hidden=1024)


def execute_proposal(public, primitive, pointers, *, accepted=True):
    """Execute actual public operands. Refusal is never repaired using gold.

    This is the P01 independent witness evaluator, not a C01 consumer. Runtime
    accepts literals up to 16 (generator witnesses can exceed result bound 8),
    then the public return-interface contract checks result range/grid.
    """
    def refuse(reason): return dict(status='refused', reason=reason, event=None)
    error = public_schema_error(public)
    if error: return refuse(error)
    if not accepted:
        return refuse('public uncertainty policy')
    if type(primitive) is not int or primitive not in range(5) or len(pointers) != 3:
        return refuse('proposal schema')
    if any(type(i) is not int or i not in range(len(public['keys'])) for i in pointers):
        return refuse('pointer bounds')
    dest, left, right = pointers
    if public['roles'][dest] != 'destination': return refuse('destination schema')
    indices = [left] if primitive == 3 else [left, right]
    if any(public['roles'][i] != 'operand' for i in indices): return refuse('operand schema')
    # Arity is supplied by the ACTUAL predicted primitive, never the target.
    # A zero-key query has tied logits in the historical bias-free pointer head;
    # canonical absence is a schema guarantee, not a learned pointer success.
    if primitive == 3:
        right = public['roles'].index('null')
    registers = [ValueRegister(public['names'][i], value, 'float' if type(value) is float else 'integer')
                 for i, value in enumerate(public['values']) if value is not None]
    session = ProtectedSession(registers, max_abs_value=16)
    result = session.execute([Candidate(public['names'][dest], PRIMITIVES[primitive], tuple(public['names'][i] for i in indices))])[0]
    if result.status != 'executed': return refuse(result.reason)
    value = float(result.value)
    if abs(value) > 8 or 2 * value != round(2 * value): return refuse('return range/grid')
    args = torch.stack((public['keys'][left], torch.zeros(32) if primitive == 3 else public['keys'][right]))
    event = dict(values=torch.tensor([[value]]), types=torch.tensor([[('integer', 'float', 'boolean').index(result.type)]]),
                 operations=torch.tensor([[primitive]]), arguments=args[None, None], provenance=public['keys'][dest][None, None],
                 operand_values=torch.tensor([[[float(public['values'][left]), 0. if primitive == 3 else float(public['values'][right])]]]),
                 argument_mask=torch.tensor([[[True, primitive != 3]]]))
    return dict(status='executed', reason='', event=event)
