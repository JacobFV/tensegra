"""E02 RL02: content-addressed (pointer) node references after a supplied NODE prefix.

Versioned extension of RL01 (``campaign02_relation``), which is imported and never
modified. The historical actor predicts EDGE source/target as one of 128 ordinal
node-ID classes from a linear head. RL02 replaces those two heads, in one arm
only, with a fresh bilinear pointer: the decoder state that emits an EDGE record
queries keys computed from the decoder states at the input positions holding
already-emitted NODE records. Classes are node ordinals exactly as before, so the
codec, decoder controller, loss, evaluation and scoring are untouched.

Supplied structural constraint (disclosed): pointer classes are masked to NODE
records strictly present among the causal PREVIOUS inputs. The mask is computed
from the model's own previous records (the supplied prefix during evaluation,
teacher-forced gold records during training exactly as in RL01), never from gold
edges or future records. Out-of-range node references are therefore impossible
by construction whenever at least one NODE precedes an EDGE.

Unchanged: emitted indices are re-embedded through the historical ordinal
source/target embeddings; keys include the historical sinusoidal positions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path

import torch
from torch import nn

from . import campaign02_relation as rl01
from .campaign02_relation import (
    PARENT_FILE_SHA, PARENT_STATE_SHA, evaluate, prepare, relation_loss, synchronize,
)
from .campaign_semantics import digest, write_gzip
from .campaign_semantics_continue import next_indices
from .campaign_semantics_s19 import padded_records
from .campaign_semantics_s19_actor import DecodeCache, TypedRecordActor
from .campaign_semantics_s19_codec import BOS, NODE, PAD
from .campaign_semantics_s21_data import VOCAB, pinned_path

VERSION = 'e02-relation-pointer-v1'
ARMS = ('all_records', 'pointer')
POINTER_FIELDS = ('source', 'target')
POINTER_MODES = ('replace', 'residual')
MASK = -1e9  # finite: masked classes get exactly zero softmax mass, and 0*MASK is not NaN
RL01_STREAM_SHA = 'eb3c0564628f421dd614930c0d51ba9874ea461d665427f845020c034fc4a5fd'
POINTER_PREFIX = 'pointer_'


@dataclass(frozen=True)
class PointerDecodeCache(DecodeCache):
    pointer_keys: torch.Tensor   # [batch, previous inputs, pointer_dim]
    pointer_nodes: torch.Tensor  # [batch, previous inputs] bool: input record is NODE


class PointerRecordActor(TypedRecordActor):
    """TypedRecordActor plus optional pointer heads for EDGE source/target.

    ``pointer_fields=()`` adds no parameters and delegates every call to the
    historical implementation, so it is bitwise the RL01 actor.
    """

    def __init__(self, *, pointer_fields=POINTER_FIELDS, pointer_dim=256, pointer_mode='replace',
                 pointer_init_seed=0, **kwargs):
        super().__init__(**kwargs)
        pointer_fields = tuple(pointer_fields)
        if any(f not in POINTER_FIELDS for f in pointer_fields) or len(set(pointer_fields)) != len(pointer_fields):
            raise ValueError('pointer fields must be a subset of source/target')
        if pointer_mode not in POINTER_MODES or pointer_dim <= 0:
            raise ValueError('invalid pointer configuration')
        self.pointer_fields = pointer_fields
        self.pointer_mode = pointer_mode
        self.pointer_dim = pointer_dim
        if pointer_fields:
            # Fresh parameters from a dedicated seed; global RNG is left untouched.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(pointer_init_seed)
                self.pointer_key = nn.Linear(self.width, pointer_dim)
                self.pointer_query = nn.ModuleDict({f: nn.Linear(self.width, pointer_dim) for f in pointer_fields})

    def pointer_parameters(self):
        return [p for n, p in self.named_parameters() if n.startswith(POINTER_PREFIX)]

    def historical_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith(POINTER_PREFIX)]

    def _pointer(self, x, keys, nodes, query_positions, logits):
        """Overwrite source/target logits [B,T,C] with masked node-ordinal pointer scores.

        x: decoder output at query positions [B,T,W]; keys/nodes: all previous input
        positions [B,S,*]. Key s is admissible for query t iff input s is a NODE and
        s <= query position t (causal; key s only summarizes inputs <= s).
        """
        batch, steps, capacity = x.shape[0], x.shape[1], self.node_capacity
        span = keys.shape[1]
        allowed = nodes[:, None, :] & (torch.arange(span, device=x.device)[None, None, :]
                                       <= query_positions.to(x.device)[None, :, None])
        ordinal = nodes.long().cumsum(1) - 1
        index = ordinal.masked_fill(~nodes, capacity).clamp(max=capacity)[:, None, :].expand(batch, steps, span)
        # Duplicate scatter targets can only occur in the discarded dump column `capacity`.
        valid = allowed.new_zeros(batch, steps, capacity + 1).scatter(2, index, allowed)[..., :capacity]
        h = self.decoder_norm(x)
        for name in self.pointer_fields:
            scores = (self.pointer_query[name](h) @ keys.transpose(-1, -2)).float() / math.sqrt(self.pointer_dim)
            scores = scores.masked_fill(~allowed, MASK)
            pointer = scores.new_full((batch, steps, capacity + 1), MASK).scatter(2, index, scores)[..., :capacity]
            value = pointer if self.pointer_mode == 'replace' else logits[name] + pointer
            logits[name] = torch.where(valid, value, torch.full_like(value, MASK))
        return logits

    def begin(self, publics):
        base = super().begin(publics)
        if not self.pointer_fields:
            return base
        batch = base.memory.shape[0]
        return PointerDecodeCache(**{f.name: getattr(base, f.name) for f in dataclass_fields(DecodeCache)},
            pointer_keys=base.memory.new_zeros(batch, 0, self.pointer_dim),
            pointer_nodes=torch.zeros(batch, 0, dtype=torch.bool, device=base.memory.device))

    def forward(self, publics, previous_records):
        if not self.pointer_fields:
            return super().forward(publics, previous_records)
        # Same computation as TypedRecordActor.forward, then pointer heads.
        with self.autocast():
            cache = self.begin(publics); x = self._embed(previous_records, cache)
            if not previous_records[:, 0, 0].eq(BOS).all():
                raise ValueError('preceding sequence must start with BOS')
            mask = previous_records[:, :, 0].ne(PAD)
            for layer, kv in zip(self.decoder, cache.cross_kv):
                x = layer.full(x, mask, kv, cache.public_mask)
            logits = self._heads(x, cache)
            keys = self.pointer_key(self.decoder_norm(x))
            nodes = previous_records[:, :, 0].eq(NODE)
            return self._pointer(x, keys, nodes, torch.arange(x.shape[1]), logits)

    def _step(self, previous_record, cache, validate):
        if not self.pointer_fields:
            return super()._step(previous_record, cache, validate)
        # Same computation as TypedRecordActor._step, plus cached pointer keys.
        if previous_record.ndim != 2:
            raise ValueError('one preceding record per batch item required')
        if validate and cache.position == 0 and not previous_record[:, 0].eq(BOS).all():
            raise ValueError('incremental decoding must begin with BOS')
        with self.autocast():
            x = self._embed(previous_record[:, None], cache, cache.position, validate)
            mask = torch.cat((cache.record_mask, previous_record[:, 0, None].ne(PAD)), dim=1); updated = []
            for layer, past, cross in zip(self.decoder, cache.self_kv, cache.cross_kv):
                x, kv = layer.step(x, past, mask, cross, cache.public_mask); updated.append(kv)
            logits = self._heads(x, cache)
            keys = torch.cat((cache.pointer_keys, self.pointer_key(self.decoder_norm(x)).to(cache.pointer_keys.dtype)), 1)
            nodes = torch.cat((cache.pointer_nodes, previous_record[:, 0, None].eq(NODE)), 1)
            logits = self._pointer(x, keys, nodes, torch.tensor([cache.position]), logits)
            logits = {k: v[:, 0] for k, v in logits.items()}
            return logits, PointerDecodeCache(cache.memory, cache.public_mask, cache.cross_kv, cache.copy_keys,
                cache.copy_inputs, tuple(updated), mask, cache.position + 1, keys, nodes)


def historical_state_hash(model):
    """thinking_language.state_hash restricted to non-pointer tensors."""
    digest_ = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith(POINTER_PREFIX):
            continue
        value = tensor.detach().cpu().contiguous()
        digest_.update(name.encode()); digest_.update(str(value.dtype).encode())
        digest_.update(str(tuple(value.shape)).encode()); digest_.update(value.numpy().tobytes())
    return digest_.hexdigest()


def load_historical(model, state):
    """Load a historical TypedRecordActor state; only pointer tensors may be absent."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected = {n for n in model.state_dict() if n.startswith(POINTER_PREFIX)}
    if unexpected or set(missing) != expected:
        raise ValueError('historical state must cover exactly the non-pointer actor')


def build_actor(c, arm, **overrides):
    """Comparator is the unchanged TypedRecordActor; pointer arm adds fresh heads."""
    dims = dict(value_count=len(VOCAB), width=c['width'], heads=8, node_capacity=128,
                max_records=160, max_slot=32, autocast_dtype='bfloat16')
    dims.update(overrides)
    if arm == 'all_records':
        return TypedRecordActor(**dims)
    if arm == 'pointer':
        p = c['pointer']
        return PointerRecordActor(pointer_fields=tuple(p['fields']), pointer_dim=p['dim'],
            pointer_mode=p['mode'], pointer_init_seed=p['init_seed'], **dims)
    raise ValueError('unknown RL02 arm')


def build_optimizer(model, c):
    """AdamW reset as in RL01; fresh pointer parameters get a registered group lr."""
    if isinstance(model, PointerRecordActor) and model.pointer_fields:
        groups = [dict(params=model.historical_parameters()),
                  dict(params=model.pointer_parameters(), lr=c['pointer']['learning_rate'])]
        return torch.optim.AdamW(groups, lr=c['learning_rate'], weight_decay=.01)
    return torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=.01)


def source_bindings():
    result = rl01.source_bindings()
    result['campaign02_relation_pointer.py'] = digest(Path(__file__))
    return result


def verify_config(c):
    if c.get('source_sha256') != source_bindings():
        raise ValueError('freeze exact source bindings before execution')
    if c['job'] not in ('main', 'profile') or c['arms'] != list(ARMS):
        raise ValueError('paired comparator/pointer arms required')
    if c['width'] != 1024 or c['batch_size'] != 8 or c['learning_rate'] != 3e-5:
        raise ValueError('primary width1024/batch8/lr3e-5 contract')
    if c['job'] == 'main' and (c['updates'] != 1024 or c['checkpoints'] != [0, 256, 1024]):
        raise ValueError('prospective main exposure changed')
    if c['parent']['sha256'] != PARENT_FILE_SHA or c['parent_state_sha256'] != PARENT_STATE_SHA:
        raise ValueError('wrong inherited S21 broad endpoint')
    p = c['pointer']
    if (p['fields'] != list(POINTER_FIELDS) or p['mode'] != 'replace' or p['dim'] != 256
            or p['learning_rate'] != 3e-4 or not isinstance(p['init_seed'], int)):
        raise ValueError('registered pointer configuration changed')


def run(c):
    verify_config(c)
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
            model = build_actor(c, arm).to(c['device'])
            checkpoint = torch.load(parent_path, map_location='cpu', weights_only=True)
            if (checkpoint['arm'], checkpoint['update'], checkpoint['seed']) != ('broad', 4096, 2101):
                raise ValueError('parent metadata mismatch')
            load_historical(model, checkpoint['model'])
            del checkpoint
            if historical_state_hash(model) != PARENT_STATE_SHA:
                raise ValueError('parent state mismatch')
            initial_pointer_sha = None
            if arm == 'pointer':
                initial_pointer_sha = hashlib.sha256(b''.join(
                    p.detach().cpu().float().numpy().tobytes() for p in model.pointer_parameters())).hexdigest()
            optimizer = build_optimizer(model, c)
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
                    metrics = {split: evaluate(model, data[split], arm, folder, split, update, c['eval_batch_size'])
                               for split in ('train_panel', 'known', 'heldout')}
                    path = folder / f'model-u{update}.pt'
                    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), update=update,
                        arm=arm, seed=c['seed'], order=order, position=position, schedule=scheduler.get_state(), visits=visits), path)
                    curves.append(dict(update=update, metrics=metrics, historical_state_sha256=historical_state_hash(model),
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
                # Both arms: unchanged historical eight-field objective (RL01 all_records).
                loss, sums, counts, _ = relation_loss(logits, records, 'all_records')
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                synchronize(c['device'])
                train_seconds += time.monotonic()-tick
                if (update+1) % 64 == 0 or update+1 == c['updates']:
                    losses.append(dict(update=update+1, loss=float(loss.detach()),
                        sums={k: float(v.detach()) for k, v in sums.items()}, counts=counts))
                    print(json.dumps(dict(event='progress', arm=arm, **losses[-1])), flush=True)
                del logits, loss, sums, records
            pointer_count = sum(p.numel() for p in model.pointer_parameters()) if arm == 'pointer' else 0
            results.append(dict(arm=arm, curves=curves, losses=losses, visits=visits,
                training_seconds=train_seconds, parameters=sum(p.numel() for p in model.parameters()),
                pointer_parameters=pointer_count, initial_pointer_sha256=initial_pointer_sha,
                construction_stream_sha256=stream.hexdigest(), parent_optimizer_reset=True))
            write_gzip(folder / 'manifest.json.gz', results[-1])
            del optimizer, model
        if results[0]['construction_stream_sha256'] != results[1]['construction_stream_sha256']:
            raise ValueError('paired training streams diverged')
        if c['job'] == 'main' and results[0]['construction_stream_sha256'] != RL01_STREAM_SHA:
            raise ValueError('construction stream differs from RL01')
        manifest = dict(version=VERSION, config=c, status='completed', results=results,
            process_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_start,
            process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_cuda_allocated=torch.cuda.max_memory_allocated() if str(c['device']).startswith('cuda') else 0,
            scope='Privileged exact NODE-prefix diagnostic. Pointer classes masked to previous NODE inputs '
                  '(supplied structural constraint). Development data reused from RL01; no confirmation.')
        write_gzip(out / 'manifest.json.gz', manifest)
        return manifest
    except BaseException as exc:
        write_gzip(out / 'failure.json.gz', dict(status='failed', error=repr(exc),
            process_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_start, completed_arms=results))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('run', 'source-bindings'))
    parser.add_argument('config', nargs='?')
    parser.add_argument('--output', help='overrides output_dir; recorded in the written config.json')
    args = parser.parse_args()
    if args.command == 'source-bindings':
        print(json.dumps(source_bindings(), indent=2))
    else:
        config = json.loads(Path(args.config).read_text())
        if args.output:
            config['output_dir'] = args.output
        run(config)
