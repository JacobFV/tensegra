"""E02 relation acquisition diagnostic: unchanged S21 actor, versioned objective.

The NODE prefix is privileged during diagnostic decoding. Neither gold edges,
edge count, nor gold termination enters free decoding. No agent runtime coupling.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
import resource
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .campaign_semantics import digest, write_gzip
from .campaign_semantics_continue import next_indices
from .campaign_semantics_data import load_cache, target
from .campaign_semantics_s19 import FIELDS, decode_evaluation, field_losses, padded_records
from .campaign_semantics_s19_actor import TypedRecordActor
from .campaign_semantics_s19_codec import EDGE, EOS, NODE, PAD, encode_row
from .campaign_semantics_s21_data import (
    CELLS, VOCAB, PublicFeatureAudit, alpha, cell, pinned_path, read_rows,
    reserve, write_rows, select_indices,
)
from .campaign_semantics_s22_controller import NodePrefix, decode
from .semantic_scaling import tokens
from .thinking_language import ActorInput, state_hash

VERSION = 'e02-relation-v1'
ARMS = ('all_records', 'relation_only')
TRAIN_COUNTS = {(3, 3): 256, (4, 3): 128, (4, 4): 128,
                (2, 4): 128, (5, 3): 192, (5, 4): 192}
KNOWN_COUNTS = {(3, 3): 128, (4, 3): 64, (4, 4): 64,
                (2, 4): 64, (5, 3): 96, (5, 4): 96}
PARENT_FILE_SHA = 'a5ad0f6d4fba9804a0ae3ed1ef8dd7b399ed2037dc89acd7b28c07f3bfc56af8'
PARENT_STATE_SHA = '2d1267174e4f7fbd93ebd77da36bec0748e74f5c412cc0144b56cf64925d683e'


def relation_loss(logits, records, arm):
    """Eight-field denominator retained; only support changes in relation arm."""
    if arm not in ARMS:
        raise ValueError('unknown objective')
    loss, sums, counts, correct = field_losses(logits, records)
    if arm == 'all_records':
        return loss, sums, counts, correct
    active = records[:, :, 0].eq(EDGE) | records[:, :, 0].eq(EOS)
    count = int(active.sum())
    counts['type'] = count
    selected = logits['type'][active].float()
    gold = records[:, :, 0][active] - 1
    sums['type'] = F.cross_entropy(selected, gold, reduction='sum') if count else logits['type'].sum() * 0
    correct['type'] = (selected.argmax(-1) == gold).sum()
    for field in ('kind', 'value', 'copy'):
        sums[field] = logits[field][..., 0].float().sum() * 0
        counts[field] = 0
        correct[field] = torch.zeros((), dtype=torch.long, device=records.device)
    loss = torch.stack([sums[k] / max(counts[k], 1) for k in FIELDS]).sum() / 8
    return loss, sums, counts, correct


def teacher_relation_rows(logits, records):
    """Compact argmax/gold pairs retain ordered local tuples and EOS decisions."""
    names = ('type', 'source', 'target', 'role', 'slot')
    predictions = torch.stack([logits[k].argmax(-1) for k in names], -1).cpu()
    rows = []
    for recs, pred in zip(records.cpu(), predictions):
        items = []
        for r, p in zip(recs.tolist(), pred.tolist()):
            if r[0] == EDGE:
                gold = [EDGE - 1, r[1], r[2], r[3], r[4] + 1]
                items.append(dict(kind='edge', gold=gold, predicted=p, correct=p == gold))
            elif r[0] == EOS:
                items.append(dict(kind='eos', gold=[EOS - 1], predicted=[p[0]], correct=p[0] == EOS - 1))
        rows.append(items)
    return rows


def node_prefix(records):
    nodes = []
    for r in records:
        if r[0] != NODE:
            break
        nodes.append(tuple(r))
    if not nodes or any(r[0] == NODE for r in records[len(nodes):]):
        raise ValueError('canonical leading NODE records required')
    return NodePrefix(tuple(nodes))


def source_bindings():
    names = ('campaign02_relation.py', 'campaign_semantics.py',
             'campaign_semantics_continue.py', 'campaign_semantics_data.py',
             'campaign_semantics_s19.py', 'campaign_semantics_s19_actor.py',
             'campaign_semantics_s19_codec.py', 'campaign_semantics_s21_data.py',
             'campaign_semantics_s22_controller.py', 'semantic_scaling.py',
             'semantic_curriculum.py', 'thinking_language.py', 'tcn_data.py')
    return {name: digest(Path(__file__).parent / name) for name in names}


def verify_sources(c):
    if c.get('source_sha256') != source_bindings():
        raise ValueError('freeze exact source bindings before execution')


def build_data(c):
    """Root-scheduled CPU-only generation; all historical exclusions are opaque."""
    from .tcn_data import verify_vendor_manifest, SOURCE_COMMIT
    verify_sources(c)
    if not verify_vendor_manifest():
        raise ValueError('vendor integrity failure')
    if not c.get('exclusion_inventory') or not c.get('exclusion_sources'):
        raise ValueError('historical alpha inventory plus later S21 split exclusions required')
    torch.set_num_threads(1)
    started = time.monotonic()
    cpu_start = time.process_time()
    out = Path(c['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    inventory_path = pinned_path(c['exclusion_inventory'])
    with gzip.open(inventory_path, 'rt') if inventory_path.suffix == '.gz' else inventory_path.open() as f:
        inventory = json.load(f)
    metadata = json.loads(pinned_path(c['exclusion_inventory_metadata']).read_text())
    if metadata['alpha_sha256'] != c['exclusion_inventory']['sha256']:
        raise ValueError('historical inventory metadata mismatch')
    seen = set(inventory if isinstance(inventory, list) else inventory['alpha_sha256'])
    if len(seen) != metadata['unique_alpha']:
        raise ValueError('historical inventory count mismatch')
    if not seen or any(not isinstance(k, str) or len(k) != 64 for k in seen):
        raise ValueError('invalid historical alpha inventory')
    exclusions = [dict(**c['exclusion_inventory'], unique_alpha=len(seen))]
    for spec in c['exclusion_sources']:
        rows = list(read_rows(pinned_path(spec)))
        keys = {alpha(r['nodes'], r['edges']) for r in rows}
        seen.update(keys)
        exclusions.append(dict(**spec, rows=len(rows), unique_alpha=len(keys)))
    initial_exclusions = len(seen)
    texts = {}
    generated = {}
    attempts = {}
    features = PublicFeatureAudit()
    for split, offset, counts in [('train', 0, TRAIN_COUNTS),
                                  ('known', 1000000, KNOWN_COUNTS),
                                  ('heldout', 2000000, {(3, 4): 512})]:
        rows = []
        for motif, count in sorted(counts.items()):
            if time.process_time() - cpu_start >= c.get('cpu_cap_seconds', 3600):
                raise TimeoutError('CPU generation cap reached between bounded reservations')
            accepted, stats = reserve(motif, count, c['first_seed'] + offset + 100000 * CELLS.index(motif), seen, texts)
            attempts[f'{split}:{motif}'] = stats
            if len(accepted) != count:
                raise ValueError('finite generator exhausted: ' + str((split, motif, stats)))
            rows.extend(accepted)
        random.Random(c['shuffle_seed'] + offset).shuffle(rows)
        for r in rows:
            features.add(r)
        path = out / (split + '.jsonl.gz')
        write_rows(path, rows)
        generated[split] = dict(path=str(path), sha256=digest(path), rows=len(rows),
            unique_alpha=len({r['alpha_sha256'] for r in rows}),
            cells=dict(collections.Counter(f"{r['arity']}x{r['facts']}" for r in rows)))
    manifest = dict(version=VERSION, config=c, generator_commit=SOURCE_COMMIT,
        generated=generated, exclusions=exclusions, initial_unique_exclusions=initial_exclusions,
        attempts=attempts, historical_coverage=metadata, public_feature_audit=features.summary(),
        process_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_start,
        scope='Fresh alpha-disjoint exploratory TRAIN1024/known512/omitted3x4-512. No confirmation.')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def prepare(c):
    data = {}
    expected = {'train': TRAIN_COUNTS, 'known': KNOWN_COUNTS, 'heldout': {(3, 4): 512}}
    seen = set()
    for split, counts in expected.items():
        rows = load_cache(pinned_path(c['inputs'][split]))
        if collections.Counter(cell(r) for r in rows) != counts:
            raise ValueError('wrong population shape: ' + split)
        keys = {r['alpha_sha256'] for r in rows}
        if len(keys) != len(rows) or seen & keys:
            raise ValueError('duplicate/overlapping constructions')
        seen.update(keys)
        result = []
        for r in rows:
            if alpha(r['nodes'], r['edges']) != r['alpha_sha256']:
                raise ValueError('alpha metadata mismatch')
            records = encode_row(r, VOCAB)
            result.append(dict(row=r, public=ActorInput(r['text'], ()), records=records,
                               prefix=node_prefix(records), gold=target(r, VOCAB)))
        data[split] = result
    panel_counts = {motif: count // 8 for motif, count in TRAIN_COUNTS.items()}
    panel_ids = select_indices([x['row'] for x in data['train']], panel_counts, 20202201)
    data['train_panel'] = [data['train'][i] for i in panel_ids]
    if c['job'] == 'profile':
        # Public length only selects the expensive profile; no evaluation tuning.
        for split in ('train_panel', 'known', 'heldout'):
            data[split] = sorted(data[split], key=lambda x: -len(tokens(x['public'])))[:c['profile_examples']]
    return data


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def evaluate(model, examples, arm, out, split, update, batch_size):
    model.eval()
    before = state_hash(model)
    rows = []
    tick = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start+batch_size]
            public = [x['public'] for x in batch]
            records = padded_records([x['records'] for x in batch], model.features.weight.device)
            logits = model.teacher_forced(public, records)
            local = teacher_relation_rows(logits, records)
            del logits, records
            generated = decode(model, public, policy='oracle_node_prefix', boundaries=[x['prefix'] for x in batch])
            for x, prediction, tf in zip(batch, generated, local):
                item = decode_evaluation(x['public'], prediction, x['gold'], len(VOCAB))
                components = item['exact_components']
                item.update(semantic_sha256=x['row']['semantic_sha256'], alpha_sha256=x['row']['alpha_sha256'],
                    cell=f"{x['row']['arity']}x{x['row']['facts']}", teacher_forced_relations=tf,
                    exact_relations=components['edges'] and components['slots'],
                    controller_stats=prediction['controller_stats'])
                rows.append(item)
    if state_hash(model) != before:
        raise ValueError('evaluation mutated weights')
    summary = dict(examples=len(rows), complete=sum(r['complete'] for r in rows),
        exact_relations=sum(r['exact_relations'] for r in rows), valid=sum(r['valid'] for r in rows))
    for kind in ('edge', 'eos'):
        support = [t for r in rows for t in r['teacher_forced_relations'] if t['kind'] == kind]
        summary['teacher_' + kind] = dict(count=len(support), correct=sum(t['correct'] for t in support))
    path = out / f'{split}-u{update}.json.gz'
    write_gzip(path, dict(arm=arm, split=split, update=update, rows=rows, summary=summary,
        scope='Privileged exact NODE-prefix free edge/EOS generation; local teacher-forced tuples separately scored.'))
    model.train()
    return dict(**summary, artifact=path.name, sha256=digest(path), process_seconds=time.monotonic()-tick)


def run(c):
    verify_sources(c)
    if c['job'] not in ('main', 'profile') or c['arms'] != list(ARMS):
        raise ValueError('paired objectives required')
    if c['width'] != 1024 or c['batch_size'] != 8:
        raise ValueError('primary width1024/batch8 contract')
    if c['job'] == 'main' and (c['updates'] != 1024 or c['checkpoints'] != [0, 256, 1024]):
        raise ValueError('prospective main exposure changed')
    if c['parent']['sha256'] != PARENT_FILE_SHA or c['parent_state_sha256'] != PARENT_STATE_SHA:
        raise ValueError('wrong inherited S21 broad endpoint')
    parent_path = pinned_path(c['parent'])
    started = time.monotonic()
    cpu_start = time.process_time()
    torch.set_num_threads(c.get('threads', 2))
    data = prepare(c)
    out = Path(c['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(c, indent=2) + '\n')
    if str(c['device']).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    results = []
    try:
        for arm in ARMS:
            torch.manual_seed(c['seed'])
            model = TypedRecordActor(value_count=len(VOCAB), width=1024, heads=8,
                node_capacity=128, max_records=160, max_slot=32, autocast_dtype='bfloat16').to(c['device'])
            checkpoint = torch.load(parent_path, map_location='cpu', weights_only=True)
            if (checkpoint['arm'], checkpoint['update'], checkpoint['seed']) != ('broad', 4096, 2101):
                raise ValueError('parent metadata mismatch')
            model.load_state_dict(checkpoint['model'])
            del checkpoint
            if state_hash(model) != PARENT_STATE_SHA:
                raise ValueError('parent state mismatch')
            optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=.01)
            scheduler = torch.Generator().manual_seed(c['schedule_seed'])
            order = torch.randperm(1024, generator=scheduler).tolist()
            position = 0
            visits = [0]*1024
            stream = hashlib.sha256()
            folder = out / arm
            folder.mkdir()
            curves, losses = [], []
            train_seconds = 0.
            for update in range(c['updates']+1):
                if time.monotonic()-started > c['cap_seconds']:
                    raise TimeoutError('root-authorized process budget reached')
                if update in c['checkpoints']:
                    metrics = {split: evaluate(model, data[split], arm, folder, split, update, c['eval_batch_size']) for split in ('train_panel', 'known', 'heldout')}
                    path = folder / f'model-u{update}.pt'
                    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), update=update,
                        arm=arm, seed=c['seed'], order=order, position=position, schedule=scheduler.get_state(), visits=visits), path)
                    curves.append(dict(update=update, metrics=metrics, state_sha256=state_hash(model),
                        checkpoint=path.name, checkpoint_sha256=digest(path), presentations=sum(visits)))
                    write_gzip(folder / 'progress.json.gz', dict(curves=curves, losses=losses))
                if update == c['updates']:
                    break
                indices, order, position = next_indices(order, position, scheduler, 8)
                batch = [data['train'][i] for i in indices]
                for i in indices:
                    visits[i] += 1
                stream.update(json.dumps(indices).encode())
                records = padded_records([x['records'] for x in batch], c['device'])
                synchronize(c['device'])
                tick = time.monotonic()
                optimizer.zero_grad()
                logits = model.teacher_forced([x['public'] for x in batch], records)
                loss, sums, counts, _ = relation_loss(logits, records, arm)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                synchronize(c['device'])
                train_seconds += time.monotonic()-tick
                if (update+1) % 64 == 0 or update+1 == c['updates']:
                    losses.append(dict(update=update+1, loss=float(loss.detach()),
                        sums={k:float(v.detach()) for k,v in sums.items()}, counts=counts))
                    print(json.dumps(dict(event='progress', arm=arm, **losses[-1])), flush=True)
                del logits, loss, sums, records
            results.append(dict(arm=arm, curves=curves, losses=losses, visits=visits,
                training_seconds=train_seconds, parameters=sum(p.numel() for p in model.parameters()),
                construction_stream_sha256=stream.hexdigest(), parent_optimizer_reset=True))
            write_gzip(folder / 'manifest.json.gz', results[-1])
            del optimizer, model
        if results[0]['construction_stream_sha256'] != results[1]['construction_stream_sha256']:
            raise ValueError('paired training streams diverged')
        manifest = dict(config=c, status='completed', results=results,
            process_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_start,
            process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_cuda_allocated=torch.cuda.max_memory_allocated() if str(c['device']).startswith('cuda') else 0)
        write_gzip(out / 'manifest.json.gz', manifest)
        return manifest
    except BaseException as exc:
        write_gzip(out / 'failure.json.gz', dict(status='failed', error=repr(exc),
            process_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_start, completed_arms=results))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('build-data', 'run', 'source-bindings'))
    parser.add_argument('config', nargs='?')
    args = parser.parse_args()
    if args.command == 'source-bindings':
        print(json.dumps(source_bindings(), indent=2))
    else:
        config = json.loads(Path(args.config).read_text())
        (build_data if args.command == 'build-data' else run)(config)
