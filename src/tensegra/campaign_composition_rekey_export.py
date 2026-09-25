"""CPU C02 compact prediction exports; full public/logit PT evidence stays remote."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import torch
from .campaign_composition_c01_export import clean
from .campaign_composition_acquire import canonical_prediction
from .campaign_composition_study import get_data


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(run, output):
    run = Path(run); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((run/'summary.json').read_text())
    manifest = dict(arm=summary['arm'], config=summary['config'], source_sha256=summary['source_sha256'],
                    checkpoint_sha256=summary['checkpoint_sha256'], export_source_sha256=sha(__file__), files={})
    for name, expected in summary['checkpoint_sha256'].items():
        if sha(run/f'{name}.pt') != expected: raise RuntimeError('checkpoint byte hash mismatch')
    calibration = None
    for path in sorted(run.glob('*.pt')):
        if path.stem in ('endpoint','selected','resume','visitation'): continue
        data = torch.load(path, map_location='cpu', weights_only=True)
        if 'public' in data: public = data['public']
        else:
            if calibration is None: _,calibration,_ = get_data(summary['config']['data']['calibration'])
            public = calibration
        logits = data['logits']; op, raw, canonical = canonical_prediction(logits, public)
        payload = dict(kind=path.stem, original_pt_sha256=sha(path), predictions=logits['answer'].argmax(-1),
                       answer_logits=logits['answer'], value_predictions=logits['value'].argmax(-1),
                       primitive=op, raw_pointers=raw, canonical_pointers=canonical,
                       labels=data['labels'], original_labels=data.get('original_labels'))
        target = output/(path.stem+'.json.gz')
        with target.open('wb') as stream:
            with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:
                compressed.write(json.dumps(clean(payload),separators=(',',':'),allow_nan=False).encode())
        manifest['files'][target.name]=dict(sha256=sha(target),original_pt_sha256=payload['original_pt_sha256'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();export(args.run,args.output)
