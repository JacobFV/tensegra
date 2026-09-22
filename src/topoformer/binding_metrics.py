"""Supervised binding losses and trajectory diagnostics, separate from inference.

Gold paths enter only these helpers. Cycle consistency uses the model's own
forward graph transition, with a detached target; it is not an inverse-edge loss.
All diagnostic arrays retain the example axis, and conditional rates aggregate
integer counts before division. A missing denominator produces None, not zero.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _query(record, name):
    value = record[name]
    return value[:, 0] if value.ndim == 3 else value


def _next(record):
    # Cycle KL is defined only for (sub)stochastic transitions. General graph
    # attention may accept multi-edge adjacency, but cannot silently reuse that
    # unnormalized pushforward as a probability target.
    full = 'next_distribution' in record
    probability = _query(record, 'next_distribution' if full else 'pnext')
    mass = probability.sum(-1, keepdim=True)
    tolerance = 1e-5
    if (not torch.isfinite(probability).all() or (probability < -tolerance).any()
            or (mass > 1 + tolerance).any()
            or (full and (mass < 1 - tolerance).any())):
        raise ValueError('cycle target must be nonnegative finite probability mass <= 1; full targets must sum to 1')
    probability = probability.clamp_min(0)  # Roundoff only, never renormalize.
    if full:
        return probability
    return torch.cat((probability, (1 - probability.sum(-1, keepdim=True)).clamp_min(0)), -1)


def _log(probability):
    return probability.clamp_min(torch.finfo(probability.dtype).tiny).log()


def binding_losses(diags: list[dict], batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """Role-balanced supervised CE, balanced null BCE, and detached-target KL.

Ground = mean(pre query CE, post query CE, real-memory key CE). Null BCE
balances the real and distractor token groups, then averages across steps; if a
batch has no distractors only the real-token group contributes. The cycle target
is the supplied-graph pushforward of predicted grounding, never the gold path.
    """
    if not diags or len(diags) != batch['path_nodes'].shape[1] - 1:
        raise ValueError('one diagnostic record per path instruction is required')
    roles = {'query': [], 'post': [], 'key': [], 'null': [], 'cycle': []}
    token_nodes = batch['token_nodes']
    real = token_nodes >= 0
    for step, record in enumerate(diags):
        pq, post, pk = _query(record, 'pq'), _query(record, 'pq_after'), record['pk']
        roles['query'].append(F.nll_loss(_log(pq), batch['path_nodes'][:, step]))
        roles['post'].append(F.nll_loss(_log(post), batch['path_nodes'][:, step + 1]))
        roles['key'].append(F.nll_loss(_log(pk[real]), token_nodes[real]))
        null_probability = pk[..., -1].clamp(1e-7, 1 - 1e-7)
        null_terms = []
        if real.any():
            null_terms.append(-torch.log1p(-null_probability[real]).mean())
        if (~real).any():
            null_terms.append(-null_probability[~real].log().mean())
        roles['null'].append(torch.stack(null_terms).mean())
        target = _next(record).detach()
        roles['cycle'].append((target * (_log(target) - _log(post))).sum(-1).mean())
    losses = {key: torch.stack(value).mean() for key, value in roles.items()}
    losses['ground'] = (losses['query'] + losses['post'] + losses['key']) / 3
    return losses


def _margin(probability):
    top = probability.topk(2, dim=-1).values
    return top[..., 0] - top[..., 1]


@torch.no_grad()
def binding_diagnostics(diags: list[dict], batch: dict[str, Tensor]) -> dict:
    """Return {per_step: [B,D], per_example: [B], counts: scalar} tensors.

A complete path requires *both* pre- and post-update grounding at every hop,
including the final destination. first_error_hop is 0-based (initial state is 0), D+1 if censored.
Error persistence/recovery compare canonical successive states (initial pre
    plus each post), avoiding overlapping pre/post duplicate state counts. These
conditional counts describe associations and do not establish an attractor.
    """
    depth = len(diags)
    if not depth or batch['path_nodes'].shape[1] != depth + 1:
        raise ValueError('one diagnostic record per path instruction is required')
    paths, token_nodes = batch['path_nodes'], batch['token_nodes']
    real = token_nodes >= 0
    rows = torch.arange(paths.shape[0], device=paths.device)
    per_step: dict[str, list[Tensor]] = {}
    counts = {}

    def add(name, value):
        per_step.setdefault(name, []).append(value)

    def count(name, value):
        counts[name] = counts.get(name, 0) + value.sum()

    for step, record in enumerate(diags):
        pq, post, pk = _query(record, 'pq'), _query(record, 'pq_after'), record['pk']
        gold, destination = paths[:, step], paths[:, step + 1]
        add('pre_correct', pq.argmax(-1) == gold)
        add('post_correct', post.argmax(-1) == destination)
        add('pre_prediction', pq.argmax(-1))
        add('post_prediction', post.argmax(-1))
        add('entropy', -(pq * _log(pq)).sum(-1))
        add('post_entropy', -(post * _log(post)).sum(-1))
        add('margin', _margin(pq))
        add('post_margin', _margin(post))
        add('null_mass', pq[:, -1])
        add('correct_node_mass', pq[rows, gold])
        add('post_correct_node_mass', post[rows, destination])
        nxt = _next(record)
        add('structural_next_node_mass', nxt[rows, destination])
        selected = batch['adjacency'][rows, batch['relations'][:, step]]
        support = selected[rows, gold].gather(1, token_nodes.clamp_min(0)) > 0
        support = support & real
        if 'attention' in record:
            attention = record['attention'].mean(1)[:, 0]
            add('clean_next_attention_mass', (attention * (token_nodes == destination[:, None])).sum(-1))
            add('relation_attention_mass', (attention * support).sum(-1))
        predicted = pk.argmax(-1)
        key_correct = (predicted == token_nodes) & real
        null_correct = (predicted == pq.shape[-1] - 1) & ~real
        count('key_correct', key_correct)
        count('key_total', real)
        count('null_correct', null_correct)
        count('null_total', ~real)
        count('real_false_null', (predicted == pq.shape[-1] - 1) & real)
        add('key_accuracy', key_correct.sum(-1) / real.sum(-1).clamp_min(1))
        add('real_key_null_mass', (pk[..., -1] * real).sum(-1) / real.sum(-1).clamp_min(1))
        # Do not invent a per-example null rate in zero-distractor batches.
        if (~real).any():
            add('distractor_null_mass', (pk[..., -1] * ~real).sum(-1) / (~real).sum(-1).clamp_min(1))
        for field in ('projected_query_norm', 'identity_norm', 'strengths'):
            if field in record:
                value = record[field].reshape(paths.shape[0], -1).mean(-1)
                add('strength' if field == 'strengths' else field, value)
        for field, gold_index, prefix in (('state_before', gold, 'pre'), ('state_after', destination, 'post')):
            if field in record:
                identity = record[field].reshape(paths.shape[0], -1)[:, :batch['entity_keys'].shape[-1]]
                target_key = batch['entity_keys'][rows, gold_index]
                add(prefix + '_identity_norm', identity.norm(dim=-1))
                add(prefix + '_identity_drift', (identity - target_key).norm(dim=-1))
                add(prefix + '_identity_cosine_drift', 1 - F.cosine_similarity(identity, target_key, dim=-1))
        if 'mlp_identity_delta' in record:
            add('mlp_identity_proposal_norm', record['mlp_identity_delta'].reshape(paths.shape[0], -1).norm(dim=-1))
        if 'identity_proposal' in record and 'identity_write' in record:
            proposal = record['identity_proposal'].reshape(paths.shape[0], -1)
            write = record['identity_write'].reshape(paths.shape[0], -1)
            add('write_override_norm', (proposal - write).norm(dim=-1))
    steps = {name: torch.stack(values, 1) for name, values in per_step.items()}
    correct = steps['pre_correct'] & steps['post_correct']
    steps['hop_correct'] = correct
    steps['survival'] = correct.long().cumprod(1).bool()
    canonical = torch.cat((steps['pre_correct'][:, :1], steps['post_correct']), 1)
    errors = ~canonical
    positions = torch.arange(depth + 1, device=paths.device).expand_as(canonical)
    first = torch.where(errors, positions, depth + 1).min(1).values
    count('error_to_error', errors[:, :-1] & errors[:, 1:])
    count('error_to_correct', errors[:, :-1] & canonical[:, 1:])
    count('error_transition_total', errors[:, :-1])
    count('correct_to_error', canonical[:, :-1] & errors[:, 1:])
    count('correct_transition_total', canonical[:, :-1])
    # Recovering at least once and ending correct are distinct from a clean path.
    after_error = errors.long().cumsum(1) > 0
    examples = {
        'complete_path': correct.all(1),
        'canonical_complete_path': canonical.all(1),
        'pre_complete_path': steps['pre_correct'].all(1),
        'post_complete_path': steps['post_correct'].all(1),
        'final_ground_correct': steps['post_correct'][:, -1],
        'first_error_hop': first,
        'ever_error': errors.any(1),
        'ever_recovered': (after_error & canonical).any(1),
        'reconverged_final': errors.any(1) & steps['post_correct'][:, -1],
        'distinct_gold_nodes': torch.tensor([row.unique().numel() for row in paths], device=paths.device),
        'distinct_predicted_nodes': torch.tensor([row.unique().numel() for row in torch.cat((steps['pre_prediction'][:, :1], steps['post_prediction']), 1)], device=paths.device),
    }
    for name, value in steps.items():
        if name not in {'pre_prediction', 'post_prediction'}:
            examples['mean_' + name] = value.float().mean(1)
    return {'per_step': steps, 'per_example': examples, 'counts': counts,
            'canonical_correct': canonical, 'canonical_survival': canonical.long().cumprod(1).bool()}


def summarize_binding(records: list[dict]) -> dict:
    """Aggregate batches without averaging rates with different denominators.

Raw per-example/per-step arrays are JSON-compatible for joint analyses. Marginal
products are descriptive comparisons only; hops are not presumed independent.
    """
    if not records:
        raise ValueError('at least one diagnostic batch is required')
    steps = {name: torch.cat([r['per_step'][name].detach().cpu() for r in records])
             for name in records[0]['per_step']}
    examples = {name: torch.cat([r['per_example'][name].detach().cpu() for r in records])
                for name in records[0]['per_example']}
    counts = {name: sum(int(r['counts'][name]) for r in records) for name in records[0]['counts']}
    metrics = {name: float(value.float().mean()) for name, value in examples.items()}
    ratios = {'key_grounding_accuracy': ('key_correct', 'key_total'),
              'distractor_null_accuracy': ('null_correct', 'null_total'),
              'real_false_null_rate': ('real_false_null', 'key_total'),
              'error_persistence': ('error_to_error', 'error_transition_total'),
              'error_recovery': ('error_to_correct', 'error_transition_total'),
              'error_onset': ('correct_to_error', 'correct_transition_total')}
    for name, (numerator, denominator) in ratios.items():
        metrics[name] = counts[numerator] / counts[denominator] if counts[denominator] else None
    canonical = torch.cat([r['canonical_correct'].detach().cpu() for r in records])
    survival = torch.cat([r['canonical_survival'].detach().cpu() for r in records])
    metrics['product_canonical_marginal_accuracy'] = float(canonical.double().mean(0).prod())
    metrics['product_hop_marginal_accuracy'] = float(steps['hop_correct'].double().mean(0).prod())
    metrics['product_pre_marginal_accuracy'] = float(steps['pre_correct'].double().mean(0).prod())
    return {'metrics': metrics, 'counts': counts,
            'canonical_correct': [''.join('1' if bit else '0' for bit in row) for row in canonical.tolist()],
            'canonical_survival': survival.double().mean(0).tolist(),
            'per_step_means': {name: value.double().mean(0).tolist() for name, value in steps.items()
                               if name not in {'pre_prediction', 'post_prediction'}},
            'per_example': {name: value.tolist() for name, value in examples.items()},
            'per_step': {name: [''.join('1' if bit else '0' for bit in row) for row in value.tolist()]
                         for name, value in steps.items() if name in {'pre_correct', 'post_correct', 'hop_correct'}}}
