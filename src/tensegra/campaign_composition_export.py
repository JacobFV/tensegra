"""CPU-only compact P01 raw export; lossless logits remain in immutable .pt files."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import torch
from tensegra.campaign_composition import make_lowering_batch, model_inputs
from tensegra.campaign_composition_acquire import canonical_prediction, tensor_digest


def export(run, output):
    run = Path(run); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((run / 'summary.json').read_text())
    assert hashlib.sha256((run / 'selected.pt').read_bytes()).hexdigest() == summary['selected_checkpoint_sha256']
    spec = summary['config']['data']['calibration']
    calibration = model_inputs(make_lowering_batch(spec['seed'], spec['count'])['public'])
    assert tensor_digest(calibration) == summary['data']['calibration']['public_tensor_sha256']
    manifest = dict(selected_checkpoint_sha256=summary['selected_checkpoint_sha256'],
                    source_sha256=summary['source_sha256'], config=summary['config'],
                    config_canonical_sha256=summary['config_sha256'], data=summary['data'], files={})
    names = [f'calibration-{step}' for step in summary['config']['checkpoints']] + ['validation'] + [f'control-{kind}' for kind in summary['config']['controls']]
    for name in names:
        path = run / (name + '.pt'); data = torch.load(path, map_location='cpu', weights_only=True)
        public = calibration if name.startswith('calibration-') else data['public']
        op, raw, canonical = canonical_prediction(data['logits'], public)
        truth = data['labels']; comparable = data.get('original_labels_comparable', True)
        original = data.get('original_labels', truth)
        primitive_margin = data['logits']['primitive'].topk(2).values.diff(dim=-1).neg().squeeze(-1)
        pointer_margin = data['logits']['pointers'].topk(2).values.diff(dim=-1).neg().squeeze(-1)
        rows = []
        for i in range(len(op)):
            rows.append(dict(index=i, predicted_primitive=int(op[i]), raw_pointers=raw[i].tolist(), canonical_pointers=canonical[i].tolist(),
                             supplied_primitive=int(truth['primitive'][i]), supplied_pointers=truth['targets'][i].tolist(),
                             original_primitive=int(original['primitive'][i]), original_pointers=original['targets'][i].tolist(),
                             original_pointer_coordinates_comparable=comparable,
                             primitive_logit_margin=float(primitive_margin[i]), raw_pointer_logit_margins=pointer_margin[i].tolist()))
        payload = dict(kind=name, original_pt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       public_sha256=tensor_digest(public), rows=rows,
                       note='Raw margins are scores, not probabilities. Canonical null follows predicted arity and public zero key.')
        destination = output / (name + '.json.gz')
        with destination.open('wb') as stream:
            with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
                compressed.write(json.dumps(payload, separators=(',', ':')).encode())
        manifest['files'][destination.name] = dict(sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), rows=len(rows), original_pt_sha256=payload['original_pt_sha256'])
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); export(args.run, args.output)
