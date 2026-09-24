"""CPU-only S19 archived free-running records/support audit. Never loads an actor."""
import argparse, collections, gzip, hashlib, json, math, time
from pathlib import Path
import torch
from topoformer.campaign_semantics_data import load_cache, target
from topoformer.campaign_semantics_s19_codec import (
    NODE, EDGE, EOS, KINDS, CodecError, decode_records,
    canonicalize_public_copies, records_to_targets, encode_row,
)
from topoformer.semantic_curriculum import unpack_graph
from topoformer.semantic_scaling import metrics, tokens
from topoformer.thinking_language import ActorInput


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def inferred_status(records, token_count, vocab_size):
    """Derive terminal actor status solely from emitted records and public bounds."""
    nodes, seen, edge_phase = 0, set(), False
    status = 'pending'
    for i, record in enumerate(records):
        tag, a, b, c, d = record
        assert status == 'pending', 'records continued after terminal status'
        if tag == EOS:
            status = 'eos'
        elif tag == NODE:
            if edge_phase:
                status = 'node_after_edge'
            elif nodes == 128:
                status = 'node_capacity_overflow'
            else:
                nodes += 1
        elif tag == EDGE:
            edge_phase = True
            if a >= nodes or b >= nodes:
                status = 'invalid_generated_node_reference'
            elif tuple(record) in seen:
                status = 'duplicate_edge'
            else:
                seen.add(tuple(record))
        else:
            raise AssertionError('impossible actor output tag')
    assert 0 < len(records) <= 160
    if status == 'pending':
        assert len(records) == 160
        return 'record_capacity_overflow'
    if status == 'eos':
        try:
            decode_records(records, token_count=token_count, vocab_size=vocab_size)
        except CodecError as error:
            return 'malformed: ' + str(error)
    return status


def score(raw, row, vocab):
    public = ActorInput(row['text'], ())
    public_tokens = tokens(public)
    gold = target(row, vocab)
    assert raw['seed'] == row['seed']
    assert raw['semantic_sha256'] == row['semantic_sha256']
    assert raw['graph_sha256'] == row['graph_sha256']
    archived_gold = unpack_graph(raw['target'])
    assert all(torch.equal(gold[k], archived_gold[k]) for k in gold)
    status = inferred_status(raw['records'], len(public_tokens), len(vocab))
    canonical = None
    try:
        if status != 'eos':
            raise CodecError(status)
        canonical = canonicalize_public_copies(raw['records'], public_tokens)
        pred = records_to_targets(canonical, token_count=len(public_tokens), vocab_size=len(vocab))
    except CodecError as error:
        assert not raw['valid'] and not raw['complete']
        assert raw['reason'] == str(error)
        assert raw['prediction'] is None and raw['metrics'] is None
        assert not any(raw['exact_components'].values())
        assert raw['canonical_records'] == (None if canonical is None else [list(r) for r in canonical])
        return False, False, str(error)
    assert raw['valid'] and raw['reason'] is None
    assert raw['canonical_records'] == [list(r) for r in canonical]
    saved = unpack_graph(raw['prediction'])
    assert all(torch.equal(pred[k], saved[k]) for k in pred)
    components = {
        'presence': torch.equal(pred['presence'], gold['presence']),
        'kind': bool(pred['kind'][gold['presence']].eq(gold['kind'][gold['presence']]).all()),
        'value': bool(pred['value'][gold['value'] >= 0].eq(gold['value'][gold['value'] >= 0]).all()),
        'copy': bool(pred['copy'][gold['copy'] >= 0].eq(gold['copy'][gold['copy'] >= 0]).all()),
        'edges': torch.equal(pred['edges'], gold['edges']),
        'slots': bool(pred['slots'][gold['edges'].any(-1)].eq(gold['slots'][gold['edges'].any(-1)]).all()),
    }
    exact = all(components.values())
    assert raw['exact_components'] == components and raw['complete'] == exact
    assert metrics(pred, gold) == raw['metrics']
    edges = [tuple(r[1:]) for r in raw['records'] if r[0] == EDGE]
    assert raw['canonical_edge_order'] == (edges == sorted(edges))
    return True, exact, None


def audit(artifact_root, data_root):
    started = time.monotonic()
    torch.set_num_threads(2)
    artifact_root, data_root = Path(artifact_root), Path(data_root)
    manifest = read(artifact_root / 'manifest.json.gz')
    config = manifest['config']
    assert config['job'] in ('profile', 'main')
    paths = {k: data_root / v['path'] for k, v in config['inputs'].items()}
    for k, path in paths.items():
        assert digest(path) == config['inputs'][k]['sha256']
    vocab = json.loads(paths['vocabulary_audit'].read_text())['value_vocabulary']
    train, dev = load_cache(paths['train']), load_cache(paths['development'])
    selected = json.loads(paths['selection'].read_text())['mixed']
    panel = [train[r['index']] for r in selected]
    development = [r for a in (3, 4) for f in (3, 4)
                   for r in [r for r in dev if (r['arity'], r['facts']) == (a, f)][:config['dev_per_cell']]]
    assert [c['update'] for c in manifest['curves']] == config['checkpoints']
    reports, bindings = [], {'manifest.json.gz': digest(artifact_root / 'manifest.json.gz')}
    for curve in manifest['curves']:
        for name, population in [('train', panel), ('development', development)]:
            summary = curve[name]
            path = artifact_root / summary['artifact']
            bindings[path.name] = digest(path)
            assert bindings[path.name] == summary['sha256']
            data = read(path)
            assert data['update'] == curve['update'] and data['population'] == name
            assert len(data['rows']) == len(population) == summary['examples']
            supports, invalid, cells = collections.Counter(), collections.Counter(), {}
            flags = []
            for raw, row in zip(data['rows'], population):
                valid, exact, reason = score(raw, row, vocab)
                cell = f"{row['arity']}x{row['facts']}"
                assert raw['cell'] == cell
                counts = cells.setdefault(cell, {'examples': 0, 'complete': 0})
                counts['examples'] += 1
                counts['complete'] += int(exact)
                if not valid:
                    invalid[reason] += 1
                records = encode_row(row, vocab)
                nodes = [r for r in records if r[0] == NODE]
                lexical = sum(r[1] in (KINDS.index('ident'), KINDS.index('entity')) for r in nodes)
                edges = sum(r[0] == EDGE for r in records)
                supports.update(type=len(records), kind=len(nodes), value=len(nodes)-lexical,
                                copy=lexical, source=edges, target=edges, role=edges, slot=edges)
                flags.append({'semantic_sha256': row['semantic_sha256'], 'cell': cell, 'valid': valid, 'complete': exact})
            assert summary['cells'] == cells and summary['invalid_reasons'] == dict(invalid)
            assert summary['complete'] == sum(r['complete'] for r in flags)
            assert summary['valid'] == sum(r['valid'] for r in flags)
            assert summary['teacher_forced'] == data['teacher_forced']
            for key, record in data['teacher_forced'].items():
                assert record['count'] == supports[key]
                assert 0 <= record['correct'] <= record['count'] and math.isfinite(record['loss_sum'])
                assert math.isclose(record['mean_loss'], record['loss_sum']/max(1,record['count']))
                assert math.isclose(record['accuracy'], record['correct']/max(1,record['count']))
            assert math.isclose(data['teacher_forced_loss'], sum(x['mean_loss'] for x in data['teacher_forced'].values())/8)
            assert summary['teacher_forced_loss'] == data['teacher_forced_loss']
            reports.append({'update': curve['update'], 'population': name, 'cells': cells,
                            'invalid_reasons': dict(invalid), 'supports': dict(supports), 'flags': flags})
    return {'status': 'pass', 'seconds': time.monotonic()-started, 'reports': reports,
            'artifact_sha256': bindings, 'scope': 'Archived record/scoring/support audit only; no actor or new forward. TF logit sums remain source-bound aggregates, not re-inferred.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('artifact_root')
    parser.add_argument('data_root')
    parser.add_argument('output')
    args = parser.parse_args()
    result = audit(args.artifact_root, args.data_root)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status':result['status'], 'seconds':result['seconds'], 'evaluations':len(result['reports'])}))
