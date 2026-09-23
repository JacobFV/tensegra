"""Explicitly stratified diagnostic; unchanged generator, no primary-population substitution."""
import torch
from .return_memory import FIELDS
from .return_memory_study import move
from .return_crossdelay import tensor_hash
from .return_diagnostics_probe import capture
from .retention_data import make_batch


def balanced_indices(batch, per_cell):
    selected=[]
    for typ, labels in ((0,range(0,33,2)),(1,range(33)),(2,(16,18))):
        for label in labels:
            available=((batch['targets']['type']==typ)&(batch['targets']['value']==label)).nonzero().flatten()
            if len(available)<per_cell:
                raise ValueError(f'insufficient fixed-pool support for type={typ}, label={label}')
            selected.append(available[:per_cell])
    return torch.cat(selected)


def capture_balanced(model, seed, pool_size, per_cell, distractors, delays, batch_size, device):
    """Generate once, then capture every delay on exactly the same events."""
    pool = make_batch(seed, pool_size, distractors=distractors)
    indices = balanced_indices(pool, per_cell)
    def subset(tree):
        return {key: subset(value) if isinstance(value, dict) else value[indices] for key,value in tree.items()}
    batch = subset(pool)
    size = len(indices)
    assert torch.equal(batch['targets']['value'], (2*batch['public']['event']['values'][:,0]+16).long())
    assert torch.equal(batch['targets']['type'], batch['public']['event']['types'][:,0])
    return capture_selected(model,batch,seed,indices,distractors,delays,batch_size,device)


def capture_selected(model,batch,seed,indices,distractors,delays,batch_size,device):
    size=len(indices)
    chunks = {delay: [] for delay in delays}
    predictions = {delay: {field: [] for field in FIELDS} for delay in delays}
    with torch.no_grad():
        for start in range(0, size, batch_size):
            def section(tree):
                return {key: section(value) if isinstance(value, dict)
                        else value[start:start + batch_size] for key, value in tree.items()}
            public = move(section(batch['public']), device)
            states = []
            hook = model.norm.register_forward_pre_hook(lambda module, args: states.append(args[0].detach().clone()))
            try:
                snapshots, _ = capture(model, public, 'factorized', 'persistent', tuple(delays))
            finally:
                hook.remove()
            if len(states) != len(delays) or list(delays) != sorted(delays):
                raise ValueError('Unexpected historical capture boundary order')
            for delay, state in zip(delays, states):
                chunks[delay].append(snapshots[f'workspace_{delay}'].cpu())
                decoded = model.decode(state, public)
                for field in FIELDS:
                    predictions[delay][field].append(decoded[field].argmax(-1).cpu())
    return dict(features={delay: torch.cat(parts) for delay, parts in chunks.items()},
                labels=batch['targets']['value'], targets={field: batch['targets'][field] for field in FIELDS},
                original_predictions={delay: {field: torch.cat(parts) for field, parts in fields.items()}
                                      for delay, fields in predictions.items()},
                event_row_hashes=[tensor_hash({key: value[index] for key, value in batch['public']['event'].items()})
                                  for index in range(size)],
                data_seed=seed, event_indices=indices.tolist(),
                event_sha256=tensor_hash(batch['public']['event']),
                public_sha256=tensor_hash(batch['public']), distractors=distractors)

