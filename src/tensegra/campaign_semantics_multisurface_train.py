"""S13 versioned paired continuation with one English calibration policy."""
from __future__ import annotations
import argparse, gzip, hashlib, json, resource, time
from functools import lru_cache
from pathlib import Path
import torch
from torch import nn
from . import semantic_scaling as base
from .campaign_semantics import digest, write_gzip, calibrated_evaluation, decode
from .campaign_semantics_data import load_cache
from .campaign_semantics_multisurface_data import public_view, target
from .campaign_semantics_continue import next_indices, first_batch_probe
from .campaign_semantics_lr import override_learning_rate
from .semantic_curriculum import SemanticCurriculumActor, sampled_pairs, curriculum_weights, pack_graph
from .semantic_text_acquisition import corrected_losses
from .thinking_language import state_hash


class SurfaceDataset:
    def __init__(self, rows, vocab, language):
        self.rows, self.vocab, self.language = rows, vocab, language

    def __len__(self):
        return len(self.rows)

    @lru_cache(maxsize=512)
    def __getitem__(self, index):
        row = self.rows[index]
        return public_view(row, self.language), target(row, self.vocab, self.language), row


def renderer_for(arm, index, added_visits, phases):
    if arm not in ('english', 'mixed'):
        raise ValueError('unregistered renderer arm')
    return 'spanish' if arm == 'mixed' and (added_visits[index] + phases[index]) % 2 else 'english'


def fixed_threshold_evaluation(model, dataset, thresholds, out, update):
    """No fitting: reuse the English TRAIN policy on Spanish public input."""
    model.eval(); rows = []; tick = time.monotonic()
    with torch.no_grad():
        for index in range(len(dataset)):
            public, gold, row = dataset[index]
            raw, calibrated = decode(model(public), public, thresholds)
            rows.append(dict(seed=row['seed'], semantic_sha256=row['semantic_sha256'],
                             graph_sha256=row['graph_sha256'], raw=pack_graph(raw),
                             calibrated_edges=pack_graph(calibrated)['edges'], target=pack_graph(gold),
                             raw_metrics=base.metrics(raw, gold), calibrated_metrics=base.metrics(calibrated, gold)))
    path = out / f'spanish-evaluation-u{update}.json.gz'
    write_gzip(path, dict(update=update, thresholds=thresholds.tolist(), threshold_source='same-checkpoint English TRAIN128', rows=rows))
    model.train()
    return dict(artifact=path.name, sha256=digest(path), seconds=time.monotonic()-tick,
                raw_exact=sum(r['raw_metrics']['semantic_equivalence'] for r in rows),
                calibrated_exact=sum(r['calibrated_metrics']['semantic_equivalence'] for r in rows), examples=len(rows))


S11_CHECKPOINT_SHA256 = '3799ade595500a083b9a558b6a5bc97c5cbc1b8bdce6e7e4b86d97b937ea373e'


def validate_config(config):
    if config.get('budget_status') != 'frozen':
        raise ValueError('profile/main requires separate coordinator freeze')
    if config['arm'] not in ('english', 'mixed') or config['learning_rate'] != 1e-5:
        raise ValueError('registered intervention is surface exposure only')
    if (config['batch_size'], config['negative_pairs'], config['renderer_seed']) != (8, 128, 913001):
        raise ValueError('registered sampling recipe changed')
    expected = {'profile': (20, [0, 20]), 'main': (4096, [0, 1024, 2048, 4096])}
    if config.get('job') not in expected or (config['added_updates'], config['checkpoints']) != expected[config['job']]:
        raise ValueError('unregistered exposure/checkpoint schedule')
    if config['parent_checkpoint_sha256'] != S11_CHECKPOINT_SHA256:
        raise ValueError('registered S11 parent changed')


def run(config):
    validate_config(config)
    start = time.monotonic(); torch.set_num_threads(2)
    out = Path(config['output_dir']); out.mkdir(parents=True, exist_ok=False)
    data = Path(config['data_dir']); audit = json.loads((data/'audit.json').read_text())
    if digest(data/'audit.json') != config['audit_sha256']:
        raise ValueError('cache audit changed')
    for split in ('train', 'development'):
        if digest(data/(split+'.jsonl.gz')) != config['cache_sha256'][split]:
            raise ValueError('cache bytes changed')
    vocab = audit['value_vocabulary']
    if vocab != ['<unknown>', '"parent"', '"unify"', 'null']:
        raise ValueError('inherited value vocabulary changed')
    rows = load_cache(data/'train.jsonl.gz'); dev_rows = load_cache(data/'development.jsonl.gz')
    if len(rows) != 8192 or len(dev_rows) != 512:
        raise ValueError('semantic support changed')
    train = {language: SurfaceDataset(rows, vocab, language) for language in ('english', 'spanish')}
    dev = {language: SurfaceDataset(dev_rows, vocab, language) for language in train}
    if digest(config['parent_checkpoint']) != config['parent_checkpoint_sha256']:
        raise ValueError('fixed S11 development parent changed')
    checkpoint = torch.load(config['parent_checkpoint'], map_location='cpu', weights_only=True)
    if checkpoint['update'] != 24576:
        raise ValueError('wrong S11 exposure')
    torch.manual_seed(201)
    model = SemanticCurriculumActor(value_count=len(vocab), width=1024, capacity=128, workspace_rows=8,
                                   microsteps=2, autocast_dtype='bfloat16').to(config['device'])
    model.load_state_dict(checkpoint['model'])
    if any(isinstance(m, nn.Dropout) and m.p for m in model.modules()):
        raise ValueError('inherited checkpoint lacks dropout RNG')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer']); lr_record = override_learning_rate(optimizer, config['learning_rate'])
    generator = torch.Generator(); generator.set_state(checkpoint['generator'])
    schedule = torch.Generator(); schedule.set_state(checkpoint['schedule'])
    order, position = checkpoint['order'], checkpoint['position']
    inherited_visits = checkpoint['visits']
    if len(inherited_visits) != 8192 or sum(inherited_visits) != 196608:
        raise ValueError('inherited construction exposure changed')
    expected_first_batch = first_batch_probe(checkpoint, train['english'], config)
    inherited_steps = sorted({float(state['step']) for state in optimizer.state.values()})
    phases = torch.randint(2, (8192,), generator=torch.Generator().manual_seed(config['renderer_seed'])).tolist()
    added_visits = [0]*8192; renderer_visits = [[0, 0] for _ in rows]
    curves = []; losses = []; train_seconds = 0.; tokens = 0; sequence = hashlib.sha256(); pair_sequence = hashlib.sha256()
    source_names = ('campaign_semantics_multisurface_train.py', 'campaign_semantics_multisurface_data.py',
                    'campaign_semantics_surface_contract.py', 'campaign_semantics.py', 'campaign_semantics_data.py',
                    'campaign_semantics_continue.py', 'campaign_semantics_lr.py', 'semantic_curriculum.py',
                    'semantic_text_acquisition.py', 'semantic_scaling.py', 'semantic_contracts.py',
                    'thinking.py', 'thinking_language.py', 'semantic_graph.py', 'tcn_data.py')
    manifest = dict(config=config, parameters=sum(p.numel() for p in model.parameters()), width=1024, workspace_rows=8,
                    initial_state_sha256=state_hash(model), learning_rate_override=lr_record, curves=curves, losses=losses,
                    source_sha256={name:digest(Path(__file__).with_name(name)) for name in source_names},
                    inherited_presentations=sum(inherited_visits), inherited_optimizer_steps=inherited_steps, renderer_phases=phases)
    del checkpoint; torch.cuda.reset_peak_memory_stats()
    for step in range(config['added_updates']+1):
        update = 24576+step
        if step in config['checkpoints']:
            english = calibrated_evaluation(model, train['english'], dev['english'], out, update, 128, 512)
            saved = json.load(gzip.open(out/english['artifact'], 'rt'))
            spanish = fixed_threshold_evaluation(model, dev['spanish'], torch.tensor(saved['thresholds']), out, update)
            curve = dict(added_update=step, update=update, english=english, spanish=spanish,
                         added_presentations=sum(added_visits), unique_visited=sum(v>0 for v in added_visits), optimizer_tokens=tokens)
            curves.append(curve); print(json.dumps(dict(event='evaluation', **curve)), flush=True)
            path = out/f'model-u{update}.pt'
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), generator=generator.get_state(),
                            schedule=schedule.get_state(), order=order, position=position,
                            visits=[a+b for a,b in zip(inherited_visits,added_visits)], update=update,
                            added_visits=added_visits, renderer_visits=renderer_visits, renderer_phases=phases), path)
            curve['checkpoint_sha256'] = digest(path)
        if step == config['added_updates']:
            break
        indices, order, position = next_indices(order, position, schedule, config['batch_size'])
        batch = []
        for index in indices:
            language = renderer_for(config['arm'], index, added_visits, phases)
            batch.append(train[language][index]); added_visits[index] += 1
            renderer_visits[index][int(language == 'spanish')] += 1
            tokens += rows[index]['surfaces'][language]['tokens']
        queries = [sampled_pairs(gold, generator, config['negative_pairs']) for _,gold,_ in batch]
        if step == 0:
            actual = dict(indices=indices, seeds=[rows[i]['seed'] for i in indices],
                          pair_sha256=[hashlib.sha256(p.numpy().tobytes()).hexdigest() for p in queries])
            if actual != expected_first_batch:
                raise ValueError('construction or negative-pair RNG differs from inherited English schedule')
            manifest['first_batch_replay'] = actual
        sequence.update(json.dumps(indices).encode())
        for pairs in queries:
            pair_sequence.update(pairs.numpy().tobytes())
        torch.cuda.synchronize(); tick=time.monotonic(); optimizer.zero_grad()
        outputs=model.forward_batch([p for p,_,_ in batch], pairs=queries)
        weights=curriculum_weights(update*config['batch_size'],1000,2000)
        parts=[corrected_losses(o,gold,pairs) for o,(_,gold,_),pairs in zip(outputs,batch,queries)]
        loss=torch.stack([sum(weights[k]*v for k,v in row.items()) for row in parts]).mean()
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        torch.cuda.synchronize(); train_seconds+=time.monotonic()-tick
        if (step+1)%128 == 0:
            values=torch.stack([torch.stack([r[k].detach() for k in parts[0]]) for r in parts]).mean(0).cpu().tolist()
            point=dict(added_update=step+1, parts=dict(zip(parts[0],values)), training_seconds=train_seconds)
            losses.append(point); print(json.dumps(dict(event='progress',**point)),flush=True)
    if config['job'] == 'main':
        expected = [4, 0] if config['arm'] == 'english' else [2, 2]
        if any(v != 4 for v in added_visits) or any(v != expected for v in renderer_visits):
            raise ValueError('fixed four-epoch renderer exposure did not complete')
    manifest.update(added_visits=added_visits, renderer_visits=renderer_visits, added_presentations=sum(added_visits),
                    optimizer_tokens=tokens, construction_sequence_sha256=sequence.hexdigest(), pair_sequence_sha256=pair_sequence.hexdigest(),
                    training_seconds=train_seconds, process_seconds=time.monotonic()-start,
                    peak_cuda_allocated=torch.cuda.max_memory_allocated(), process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    final_state_sha256=state_hash(model))
    write_gzip(out/'manifest.json.gz', manifest)
    (out/'completion.json').write_text(json.dumps(dict(success=True, process_seconds=time.monotonic()-start))+'\n')
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('config'); args=parser.parse_args()
    run(json.loads(Path(args.config).read_text()))
