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
    register_types: tuple


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
                                      'two_derived' if split == 'ood' else 'chain_or_leaf', tuple(instructions), chosen[0],
                                      tuple('boolean' if any(r[0] == key and r[1] == 4 for r in instructions) else 'integer' for key in keys)))
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
            'register_types':torch.tensor([[int(t == 'boolean') for t in x.register_types]+[-1]*(n-len(x.keys)) for x in examples]),
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


def joint_evidence(*, seed, steps=6):
    """Narrow A2 scaffold with supplied finite JOINT hypotheses and likelihoods.

Public frames reveal likelihood support sequentially. Private posterior targets
are normalized accumulated likelihoods; no independently factorized arg choices
can invent the impossible (a,a) or (b,b) hypotheses.
"""
    rng = random.Random(seed)
    frames = torch.tensor([[[rng.uniform(.1,.9)] for _ in range(2)] for _ in range(steps)])
    log_support = frames.squeeze(-1).log().cumsum(0)
    return {'hypotheses':[('sub','d','a','b'),('sub','d','b','a')],
            'frames':frames, 'private_posterior':log_support.softmax(-1)}


class JointPosteriorModel(nn.Module):
    """Past/current public-frame recurrence; no future frame or target input."""
    def __init__(self, hidden=16):
        super().__init__()
        self.cell = nn.GRUCell(1,hidden)
        self.readout = nn.Linear(hidden,1)
        self.hidden = hidden

    def forward(self, frames):
        b,t,h,_ = frames.shape
        state = frames.new_zeros(b*h,self.hidden)
        beliefs = []
        for step in range(t):
            state = self.cell(frames[:,step].reshape(b*h,1),state)
            beliefs.append(self.readout(state).reshape(b,h).softmax(-1))
        return torch.stack(beliefs,1)


def progressive_instruction_episode(*, seed, operation_override=None, swap_override=None, no_executable=None, records=4):
    """Public partial instruction records with a supplied finite hypothesis set.

The unordered operand pair is supplied as a prior. Operation and ordered roles
are revealed later. The null hypothesis means no instruction matches the query
key. Private posterior is uniform over candidates consistent with visible fields.
"""
    rng = random.Random(seed)
    key_dim,feature_dim = 8,32
    keys = []
    while len(keys) < 3+records*3:
        key = tuple(rng.choice((-1.,1.)) for _ in range(key_dim))
        if key not in keys: keys.append(key)
    dest,a,b = keys[:3]
    op_sample,swap_sample = rng.randrange(5),rng.randrange(2)
    operation = op_sample if operation_override is None else operation_override
    swap = swap_sample if swap_override is None else swap_override
    null_sample = rng.random() < .2
    null = null_sample if no_executable is None else no_executable
    op_arrival,order_arrival = rng.choice((1,2)),rng.choice((3,4))
    hypotheses = torch.zeros(11,feature_dim)
    for op in range(5):
        for ordering in range(2):
            h = hypotheses[2*op+ordering]
            h[op] = 1;h[5:13] = torch.tensor(dest)
            h[13:21] = torch.tensor(b if ordering else a)
            if op != 3: h[21:29] = torch.tensor(a if ordering else b)
            h[29:31] = 1
    hypotheses[-1,-1] = 1
    raw_records = [(keys[3] if null else dest,operation,b if swap else a,a if swap else b)]
    for i in range(1,records):
        raw_records.append((keys[3+3*i],rng.randrange(5),keys[4+3*i],keys[5+3*i]))
    rng.shuffle(raw_records)
    frames = torch.zeros(5,records,feature_dim)
    targets = torch.zeros(5,11)
    for t in range(5):
        for j,(destination,op,left,right) in enumerate(raw_records):
            frames[t,j,5:13] = torch.tensor(destination)
            if t >= op_arrival: frames[t,j,op] = 1;frames[t,j,29] = 1
            if t >= order_arrival:
                frames[t,j,13:21] = torch.tensor(left)
                if op != 3: frames[t,j,21:29] = torch.tensor(right)
                frames[t,j,30] = 1
        if null:
            targets[t,-1] = 1
        elif t < op_arrival:
            targets[t,:10] = .1
        elif t < order_arrival:
            targets[t,2*operation:2*operation+2] = .5
        else:
            targets[t,2*operation+swap] = 1
    return {'frames':frames,'hypotheses':hypotheses,'private_posterior':targets,
            'operation_arrival':op_arrival,'order_arrival':order_arrival,'no_executable':null}


class ProgressiveInstructionModel(nn.Module):
    """Persistent candidate workspace using the existing four workspace blocks."""
    def __init__(self,width=24):
        super().__init__()
        from .thinking import ThinkingConfig, _WorkspaceBlock
        config = ThinkingConfig(feature_dim=32,width=width,heads=2,structural_heads=0,workspace_rows=11)
        self.encode = nn.Linear(32,width)
        self.blocks = nn.ModuleList(_WorkspaceBlock(config) for _ in range(4))
        self.score = nn.Linear(width,1)

    def forward(self,frames,hypotheses):
        state = self.encode(hypotheses)
        b,steps,n,_ = frames.shape
        mask = torch.ones(b,n,dtype=torch.bool,device=frames.device)
        bias = frames.new_zeros(b,2,11,11)
        beliefs = []
        for t in range(steps):
            memory = self.encode(frames[:,t])
            for block in self.blocks:
                state,_ = block(state,memory,mask,bias)
            beliefs.append(self.score(state).squeeze(-1).softmax(-1))
        return torch.stack(beliefs,dim=1)
