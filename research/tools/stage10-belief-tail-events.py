"""Enrich archived tail examples with exact regenerated public observations only."""
import argparse
import gzip
import json
from pathlib import Path
from topoformer.belief_contracts import contract_episodes

parser = argparse.ArgumentParser()
parser.add_argument('--tails', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
with gzip.open(args.tails, 'rt') as handle:
    tails = json.load(handle)
examples = []
cache = {}
for group in tails['largest_examples_per_run'].values():
    for example in group:
        for condition in ('clean', example['condition']):
            key = (example['event_seed'], example['candidates'], condition)
            if key not in cache:
                cache[key] = contract_episodes(512, *key[:2], condition=condition)
        clean = cache[(example['event_seed'], example['candidates'], 'clean')][example['episode']]
        renamed = cache[(example['event_seed'], example['candidates'], example['condition'])][example['episode']]
        frame = example['frame']
        # Values in public records address the nonce inventory except primitive role.
        examples.append({**example, 'public_keys': clean['keys'], 'public_candidate_records': clean['records'],
                         'public_clean_events': clean['events'], 'public_renamed_events': renamed['events'],
                         'gold_supported_candidates': [i for i, value in enumerate(example['target']) if value > 0],
                         'gold_null': bool(example['target'][-1]),
                         'frame_clean_event': clean['events'][frame], 'frame_renamed_event': renamed['events'][frame]})
args.out.write_text(json.dumps({'kind': 'archived tail reconstruction, no model inference', 'examples': examples}, indent=2))
