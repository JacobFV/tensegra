"""Independent calibration of SUPPLIED candidate posterior supports.

The product is a correlated-factor score, never an assumed probability.
Schema validity is exact; stable wrong candidates remain possible.
"""
import random
import torch
from torch import nn


def readiness_factors(primitive, ordered_arguments, schema, stability):
    if not 1 <= len(ordered_arguments) <= 2:
        raise ValueError('one or two ordered arguments required')
    factors = [float(primitive), float(ordered_arguments[0]),
               float(ordered_arguments[1]) if len(ordered_arguments) == 2 else 1., float(bool(schema)), float(stability)]
    if not all(0 <= x <= 1 for x in factors):
        raise ValueError('factors must lie in [0,1]')
    return factors


def make_readiness(groups, *, seed):
    rng = random.Random(seed)
    records = []
    for group in range(groups):
        rows = []
        for candidate in range(4):
            regime = group % 4
            if (regime == 0 and candidate == 0) or (regime == 1 and candidate < 2):
                latent = rng.uniform(.9985, 1.)
            elif regime == 3 and candidate in (0,1):
                latent = rng.uniform(.975, .999)
            else:
                latent = rng.uniform(.05, .995)
            # Correlated, supplied confidence features precede private sampled
            # correctness. The product is not this generator probability.
            supports = [min(1., max(0., latent+rng.uniform(-.0002,.0002))) for _ in range(3)]
            probability = .9999 if latent >= .9985 else .15*latent
            schema = candidate != 3
            correct = rng.random() < probability and schema
            stability = 1. if regime == 3 else rng.uniform(.8,1.)
            factors = readiness_factors(supports[0], supports[1:], schema, stability)
            rows.append({'group':group, 'candidate':candidate, 'factors':factors,
                         'product_score':float(torch.tensor(factors).prod()),
                         'correct':bool(correct), 'schema':schema,
                         'private_label_probability':probability if schema else 0.})
        ready = sum(r['correct'] for r in rows)
        kind = 'zero_ready' if ready == 0 else 'multiple_ready' if ready > 1 else 'mixed'
        for row in rows:
            row['kind'] = 'stable_wrong' if not row['correct'] and row['factors'][-1] == 1 else kind
            row['group_kind'] = kind
        records.extend(rows)
    return records


class Calibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 1)

    def forward(self, factors):
        features = torch.cat((factors, factors.prod(-1, keepdim=True)), -1)
        return self.linear(features).squeeze(-1).sigmoid() * factors[:, 3]


def select_threshold(scores, truth, precision=.99):
    if len(scores) != len(truth) or not scores:
        raise ValueError('nonempty matching scores and labels required')
    eligible = []
    for threshold in sorted(set(scores)):
        m = readiness_metrics(scores, truth, threshold, bins=0)
        if m['precision'] > precision:
            eligible.append((m['true_positive'], -threshold, threshold))
    return max(eligible)[2] if eligible else 1.000001


def readiness_metrics(scores, truth, threshold, bins=10):
    selected = [s >= threshold for s in scores]
    tp = sum(a and bool(y) for a, y in zip(selected, truth))
    count, positive = sum(selected), sum(truth)
    result = {'count': len(scores), 'selected': count, 'executable': positive, 'true_positive': tp,
              'precision': tp/count if count else 0., 'recall': tp/positive if positive else 0.,
              'coverage': count/len(scores) if scores else 0.,
              'brier': sum((s-y)**2 for s, y in zip(scores, truth))/max(1, len(scores))}
    reliability = []
    for i in range(bins):
        indices = [j for j, s in enumerate(scores) if i/bins <= s <= (i+1)/bins and (i == bins-1 or s < (i+1)/bins)]
        reliability.append({'lower': i/bins, 'count': len(indices),
                            'confidence': sum(scores[j] for j in indices)/len(indices) if indices else None,
                            'accuracy': sum(truth[j] for j in indices)/len(indices) if indices else None})
    result['reliability'] = reliability
    return result


def global_scores(records, scores):
    """Matched scalar accepts/rejects all candidates in a group; mean support."""
    groups = {}
    for r, s in zip(records, scores):
        groups.setdefault(r['group'], []).append(s)
    return [sum(groups[r['group']])/len(groups[r['group']]) for r in records]


def gate_b(metrics):
    return metrics['precision'] > .99 and metrics['recall'] >= .5 and metrics['local_advantage'] > 0
