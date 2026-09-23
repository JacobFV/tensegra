#!/usr/bin/env python3
"""Audit-compatible summaries of frozen scalar transfer and full return counts."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

FIELDS = ('value', 'type', 'operation', 'argument0', 'argument1', 'provenance')
DELAYS = (0,1,2,4,8,16,32)
parser = argparse.ArgumentParser()
parser.add_argument('directory')
args = parser.parse_args()
root = Path(args.directory)
rows = []
for path in sorted(root.glob('*-predictions.jsonl.gz')):
    with gzip.open(path, 'rt') as stream:
        rows.extend(json.loads(line) for line in stream)
index = {(row['seed'], row['head'], row['split'], row['distractors'], row['target_delay']): row for row in rows}
if len(index) != len(rows):
    raise ValueError('Repeated prediction cells')
matrices = []
curves = []
paired = []
for seed in (10,11,12):
    for distractors in (2,8):
        matrix = []
        for source in DELAYS:
            scores = []
            for target in DELAYS:
                row = index[(seed, f'delay_{source}', 'test', distractors, target)]
                scores.append(row['counts']['value'])
                diagonal = index[(seed, f'delay_{target}', 'test', distractors, target)]
                assert row['event_sha256'] == diagonal['event_sha256'] and row['targets'] == diagonal['targets']
                y = row['targets']['value']
                outcomes = Counter(('correct' if a == t else 'wrong') + '_to_' + ('correct' if b == t else 'wrong')
                                   for a,b,t in zip(row['predictions']['value'], diagonal['predictions']['value'], y))
                paired.append(dict(seed=seed, distractors=distractors, source_delay=source,
                                   target_delay=target, comparison='source head to target-delay head on same target state', outcomes=dict(outcomes)))
            matrix.append(scores)
        matrices.append(dict(seed=seed, distractors=distractors, sources=DELAYS, targets=DELAYS, cells=matrix))
        for head in ('original', 'short_pool', 'shared_pool'):
            for delay in DELAYS:
                row = index[(seed, head, 'test', distractors, delay)]
                base = index[(seed, 'original', 'test', distractors, delay)]
                for field in FIELDS[1:]:
                    assert row['predictions'][field] == base['predictions'][field]
                err = [(p-t)/2 for p,t in zip(row['predictions']['value'], row['targets']['value'])]
                groups = {str(value): dict(total=sum(t==value for t in row['targets']['value']),
                    correct=sum(t==value and p==t for p,t in zip(row['predictions']['value'], row['targets']['value'])))
                    for value in sorted(set(row['targets']['value']))}
                curves.append(dict(seed=seed, distractors=distractors, head=head, delay=delay,
                    counts=row['counts'], metrics=row['metrics'], by_value=groups,
                    large_error_gt_1=sum(abs(e)>1 for e in err), large_error_gt_2=sum(abs(e)>2 for e in err),
                    confusion=[[a,b,n] for (a,b),n in sorted(Counter(zip(row['targets']['value'], row['predictions']['value'])).items())]))
gates = {}
for name, delays in [('scalar_covered', (0,1,2,4,8,16)), ('scalar_ood32', (32,))]:
    failed = []
    for seed in (10,11,12):
        for distractors in (2,8):
            for delay in delays:
                row = index.get((seed, 'shared_pool', 'test', distractors, delay))
                if row is None:
                    failed.append(dict(seed=seed, distractors=distractors, delay=delay, reason='missing'))
                    continue
                count = row['counts']['value']
                if count['total'] < 512 or count['correct']/count['total'] <= .98:
                    failed.append(dict(seed=seed, distractors=distractors, delay=delay, counts=count))
    gates[name] = dict(status='failed' if failed else 'passed_restricted', failures=failed,
                       scope='scalar only; does not override full return gate or composition block')
for head in ('original','short_pool','shared_pool'):
    failures = []
    for seed in (10,11,12):
        for distractors in (2,8):
            row = index[(seed,head,'test',distractors,16)]
            for field in FIELDS:
                count = row['counts']['argument1_required' if field == 'argument1' else field]
                threshold = .99 if field in ('type','operation') else .98
                if row['counts']['joint']['total'] < 512 or not count['total'] or count['correct']/count['total'] <= threshold:
                    failures.append(dict(seed=seed,distractors=distractors,field=field,counts=count,threshold=threshold))
    gates['full_return16_'+head] = dict(status='failed' if failures else 'passed_restricted', failures=failures,
                                      scope='fresh test diagnostic; historical Stage9 gate remains unchanged')
(root/'summary.json').write_text(json.dumps(dict(matrices=matrices, curves=curves, paired=paired, gates=gates), indent=2))
