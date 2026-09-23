"""Complete-evidence, role-bearing synthetic proposal acquisition (no runtime).

Semantic roles and operation cues are supplied. This measures learned keyed
selection/composition, NOT parsing, language understanding, or execution.
"""
from dataclasses import dataclass
import random
import torch
from torch import nn

PRIMITIVES = ('add', 'sub', 'mul', 'neg', 'compare')

@dataclass(frozen=True)
class ProposalExample:
    primitive: str
    keys: tuple
    names: tuple
    values: tuple
    role_keys: tuple
    targets: tuple
    composition: str
    instructions: tuple
    query_destination: tuple


def make_proposals(count, *, seed, split='iid', candidates=None, primitive=None, key_dim=16):
    rng = random.Random(seed)
    result = []
    for _ in range(count):
        n = candidates or rng.randint(*((9, 14) if split == 'ood' else (5, 8)))
        if n < 3:
            raise ValueError('at least three distinct candidates required')
        keys = []
        while len(keys) < n:
            key = tuple(rng.choice((-1., 1.)) for _ in range(key_dim))
            if key not in keys:
                keys.append(key)
        instructions = []
        numeric = [0, 1]
        # Actual acyclic arithmetic construction: later instructions can consume
        # earlier destination registers. No operation is executed in this task.
        for dest in range(2, n):
            if split == 'ood' and len(numeric) >= 4:
                args = tuple(numeric[-2:])  # heldout two-derived-input motif
            elif len(numeric) > 2 and rng.random() < .5:
                args = (numeric[-1], rng.choice(numeric[:-1]))
            else:
                args = tuple(rng.sample(range(min(dest, 2)), 2))
            op = primitive or rng.choice(('sub', 'sub', 'compare', 'compare', 'add', 'mul', 'neg'))
            if op not in PRIMITIVES:
                raise ValueError(op)
            instructions.append((keys[dest], PRIMITIVES.index(op), keys[args[0]], keys[args[1]]))
            if op != 'compare':
                numeric.append(dest)
        chosen = rng.choice(instructions)
        rng.shuffle(instructions)
        rng.shuffle(keys)
        role_keys = (chosen[0], chosen[2], chosen[3])
        targets = tuple(keys.index(key) for key in role_keys)
        prefix = 'heldout' if split == 'ood' else 'train'
        names = tuple(f'{prefix}_{rng.getrandbits(48):012x}' for _ in keys)
        values = tuple(rng.randint(21, 80) if split == 'ood' else rng.randint(-10, 10) for _ in keys)
        result.append(ProposalExample(PRIMITIVES[chosen[1]], tuple(keys), names, values, role_keys, targets,
                                      'two_derived' if split == 'ood' else 'chain_or_leaf', tuple(instructions), chosen[0]))
    return result


def collate_proposals(examples, control='complete'):
    if control not in ('complete', 'missing', 'permuted', 'reverse'):
        raise ValueError(control)
    n, d = max(len(x.keys) for x in examples), len(examples[0].keys[0])
    m = max(len(x.instructions) for x in examples)
    keys = torch.zeros(len(examples), n, d)
    valid = torch.zeros(len(examples), n, dtype=torch.bool)
    destinations = torch.zeros(len(examples), m, d)
    arguments = torch.zeros(len(examples), m, 2, d)
    cues = torch.zeros(len(examples), m, 5)
    instruction_valid = torch.zeros(len(examples), m, dtype=torch.bool)
    for i, x in enumerate(examples):
        keys[i, :len(x.keys)] = torch.tensor(x.keys)
        valid[i, :len(x.keys)] = True
        for j,(dest,op,a,b) in enumerate(x.instructions):
            destinations[i,j] = torch.tensor(dest)
            arguments[i,j] = torch.tensor((a,b))
            cues[i,j,op] = 1
            instruction_valid[i,j] = True
    if control == 'missing':
        arguments.zero_(); cues.zero_()
    elif control == 'reverse':
        arguments = arguments[:, :, [1, 0]]
    elif control == 'permuted':
        arguments = arguments.roll(1, dims=0); cues = cues.roll(1,dims=0)
    primitive = torch.tensor([PRIMITIVES.index(x.primitive) for x in examples])
    return {'keys': keys, 'valid': valid, 'instruction_destinations':destinations,
            'instruction_arguments':arguments, 'instruction_cues':cues, 'instruction_valid':instruction_valid,
            'query_destination':torch.tensor([x.query_destination for x in examples]),
            'primitive': primitive, 'targets': torch.tensor([x.targets for x in examples]),
            'binary': primitive != PRIMITIVES.index('neg')}


class ProposalModel(nn.Module):
    def __init__(self, key_dim=16, hidden=32):
        super().__init__()
        self.retrieve_query = nn.Linear(key_dim, hidden, bias=False)
        self.retrieve_key = nn.Linear(key_dim, hidden, bias=False)
        self.query = nn.Linear(key_dim, hidden, bias=False)
        self.key = nn.Linear(key_dim, hidden, bias=False)
        self.operation = nn.Sequential(nn.Linear(5, hidden), nn.ReLU(), nn.Linear(hidden, 5))
        self.scale = hidden ** -.5

    def forward(self, batch):
        retrieval = (self.retrieve_query(batch['query_destination'])[:,None] *
                     self.retrieve_key(batch['instruction_destinations'])).sum(-1) * self.scale
        weights = retrieval.masked_fill(~batch['instruction_valid'], -1e9).softmax(-1)
        selected_args = torch.einsum('bm,bmad->bad', weights, batch['instruction_arguments'])
        queries = torch.cat((batch['query_destination'][:,None],selected_args),dim=1)
        cue = torch.einsum('bm,bmc->bc',weights,batch['instruction_cues'])
        scores = self.query(queries) @ self.key(batch['keys']).transpose(1, 2) * self.scale
        return {'primitive': self.operation(cue),
                'pointers': scores.masked_fill(~batch['valid'][:, None], -1e9)}


def proposal_loss(out, batch):
    ce = torch.nn.functional.cross_entropy
    losses = {'primitive': ce(out['primitive'], batch['primitive'])}
    for role, name in enumerate(('destination', 'operand1', 'operand2')):
        mask = batch['binary'] if role == 2 else torch.ones_like(batch['binary'])
        losses[name] = ce(out['pointers'][mask, role], batch['targets'][mask, role]) if mask.any() else out['pointers'].sum()*0
    return losses


def proposal_metrics(out, batch):
    op = out['primitive'].argmax(-1) == batch['primitive']
    args = out['pointers'].argmax(-1) == batch['targets']
    binary = batch['binary']
    full = op & args[:, 0] & args[:, 1] & (args[:, 2] | ~binary)
    noncomm = (batch['primitive'] == 1) | (batch['primitive'] == 4)
    return {'count': len(op), 'primitive': op.float().mean().item(),
            'destination': args[:, 0].float().mean().item(), 'operand1': args[:, 1].float().mean().item(),
            'operand2': args[binary, 2].float().mean().item() if binary.any() else 1.,
            'operand2_count': int(binary.sum()), 'full': full.float().mean().item(),
            'noncommutative_count': int(noncomm.sum()),
            'noncommutative_order': args[noncomm, 1:].all(-1).float().mean().item() if noncomm.any() else 1.}


def gate_a(metrics):
    return len(metrics) == 2 and all(
        m.get('count', 0) > 0 and m.get('noncommutative_count', 0) > 0 and
        m.get('full', 0) > .98 and m.get('noncommutative_order', 0) > .99 and
        all(m.get(k, 0) > .99 for k in ('primitive', 'destination', 'operand1', 'operand2'))
        for m in metrics)


def gate_a_matrix(observed, expected_seeds):
    conditions = ('iid_validation', 'ood_validation')
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        return False
    for seed in expected_seeds:
        if seed not in observed or any(k not in observed[seed] for k in conditions):
            return False
        if not gate_a([observed[seed][k] for k in conditions]):
            return False
    return True


def arithmetic_delta(operation, left, right=None):
    """Auditable arithmetic reduction target; supplied operands, no rewrite engine."""
    if operation == 'neg':
        value = -left
    elif operation == 'add':
        value = left + right
    elif operation == 'sub':
        value = left - right
    elif operation == 'mul':
        value = left * right
    elif operation == 'compare':
        value = left < right
    else:
        raise ValueError(operation)
    return {'remove_operands': [left] if operation == 'neg' else [left, right], 'insert_value': value, 'value_type':'boolean' if operation == 'compare' else 'integer'}
