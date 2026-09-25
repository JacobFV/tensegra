"""C04 private audit helpers. Never imported into a model forward path."""
from collections import Counter
import torch
from .campaign_composition_acquire import labels_from_public
from .campaign_composition_runtime import build_returns
from .campaign_returns_use import answer


def requested_labels(rows):
    oracle = labels_from_public(rows)
    bundle = build_returns(rows, oracle['primitive'], oracle['targets'])
    if len(bundle['indices']) != len(rows): raise RuntimeError('requested view refused; no filtering/resampling')
    event = bundle['public']['event']
    labels = dict(oracle, task=answer(event['values'][:,0], bundle['public']['query']), value=(2*event['values'][:,0]+16).long(), type=event['types'][:,0])
    return labels, bundle


def signatures(public, labels, with_query):
    """Private numeric tuples, excluding nonce identities and nuisance records."""
    result = []
    for i, op in enumerate(labels['primitive'].tolist()):
        fields = [str(op)]
        for slot in (1,2):
            if slot == 2 and op == 3:
                fields.append('absent'); continue
            pointer = int(labels['targets'][i,slot]); kind = int(public['operand_types'][i,pointer])
            value = float(public['operand_values'][i,pointer])
            fields.append(f'{kind}:{int(value) if kind == 0 else value.hex()}')
        if with_query:
            fields += [float(public['query'][i,0]).hex(), str(int(public['query'][i,1]))]
        result.append('|'.join(fields))
    return result


def numeric_overlap(datasets, swapped, visits, swapped_visits):
    """Actual visited view multiplicities; no filtering or sample selection."""
    audit = {}
    for with_query in (False,True):
        pools = {}; counts = Counter()
        for split,(public,labels,*_) in datasets.items():
            pools[split+'/clean'] = signatures(public,labels,with_query)
            pools[split+'/reversed'] = signatures(*swapped[split],with_query)
        for view,weights in (('clean',visits-swapped_visits),('reversed',swapped_visits)):
            for key,weight in zip(pools['train/'+view],weights.tolist()):
                if weight: counts[key] += weight
        cells = {}
        for name,keys in pools.items():
            frequencies = Counter(keys); overlap = set(frequencies) & set(counts)
            cells[name] = dict(events=len(keys),distinct_signatures=len(frequencies),training_overlap_distinct=len(overlap),training_overlap_events=sum(frequencies[k] for k in overlap),signature_counts=dict(sorted(frequencies.items())))
        audit['operation_operands_query' if with_query else 'operation_operands'] = dict(actual_training_presentations=sum(counts.values()),actual_training_distinct_signatures=len(counts),actual_training_signature_counts=dict(sorted(counts.items())),populations=cells)
    return dict(audits=audit,scope='finite IID numeric overlap; fresh identities are not new computations; inherited-lineage overlap not measured here')


def full_proposal(prediction, labels):
    return (prediction['primitive'] == labels['primitive']) & (prediction['canonical_pointers'] == labels['targets']).all(-1)


def causal_gate(clean, dropped, query, copied, original, supplied_controls):
    """Every rate uses ALL attempted examples, including persistent refusals."""
    n = len(original)
    if any(len(x) != n for x in (clean,dropped,query,copied)): raise ValueError('matched attempted denominators required')
    accuracy = lambda pred: float((pred == original).double().mean())
    c,d,q,o = map(accuracy,(clean,dropped,query,copied))
    gain = (c-q)/(o-q) if o > q else None
    changed = {}
    for kind,(prediction,supplied) in supplied_controls.items():
        mask=(supplied>=0)&(supplied!=original)
        changed[kind]=dict(correct=int(((prediction==supplied)&mask).sum()),total=int(mask.sum()))
    passed=set(changed)=={'wrong','swap'} and c-d>=.15 and gain is not None and gain>=.8 and all(x['total']>=256 and x['correct']/x['total']>=.9 for x in changed.values())
    return dict(attempted=n,clean_accuracy=c,drop_accuracy=d,query_accuracy=q,copy_accuracy=o,clean_minus_drop=c-d,normalized_gain=gain,changed_supplied=changed,passed=passed,scope='fixed inherited consumer robustness, not newly optimal Bayes ceilings')
