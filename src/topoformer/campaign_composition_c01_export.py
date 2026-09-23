"""CPU compact C01 predictions/labels; full logits and public tensors stay remote."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import torch
from topoformer.campaign_composition import make_lowering_batch, model_inputs
from topoformer.campaign_composition_acquire import canonical_prediction


def clean(value):
    if isinstance(value, torch.Tensor): return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    return value


def export(run, output):
    run = Path(run); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((run/'summary.json').read_text()); config = summary['config']
    manifest = dict(phase=summary['phase'], config=config, config_sha256=summary['config_sha256'], source_sha256=summary['source_sha256'],
                    selected_checkpoint_sha256=summary.get('selected_checkpoint_sha256'), frozen_state=summary.get('frozen_state'),
                    export_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), files={})
    if summary.get('selected_checkpoint_sha256'):
        assert hashlib.sha256((run/'selected.pt').read_bytes()).hexdigest() == summary['selected_checkpoint_sha256']
    calibration = None
    for path in sorted(run.glob('*.pt')):
        if path.name == 'selected.pt' or path.name.startswith('visitation-'): continue
        data = torch.load(path, map_location='cpu', weights_only=True)
        payload = dict(kind=path.stem, original_pt_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if 'result' in data:
            result = data['result']; payload.update(original=data['original'], supplied=result['supplied'], supplied_values=result['values'], supplied_types=result['types'],
                    reasons=data.get('reasons'), requested=data.get('requested'), full_proposal=data.get('full_proposal'),
                    exact_copy=data.get('exact_copy'), query_only=data.get('query_only'), cells={d:dict(predictions=v['predictions'],logits=v['logits']) for d,v in result['cells'].items()})
            if data.get('proposal'):
                proposal=data['proposal']; payload['proposal']={k:proposal[k] for k in ('primitive','raw_pointers','canonical_pointers')}
            if data.get('supplied_public'):
                event=data['supplied_public']['event']; payload['supplied_event']={k:event[k] for k in ('values','types','operations','operand_values','argument_mask','provenance')}
        elif 'predictions' in data:
            payload.update(data)
        else:
            if 'public' in data: public=data['public']
            else:
                if calibration is None:
                    spec=config['data']['calibration'];calibration=model_inputs(make_lowering_batch(spec['seed'],spec['count'])['public'])
                public=calibration
            logits=data['logits'];op,raw,canonical=canonical_prediction(logits,public)
            payload.update(predictions=logits['answer'].argmax(-1),answer_logits=logits['answer'],value_predictions=logits['value'].argmax(-1),
                           primitive=op,raw_pointers=raw,canonical_pointers=canonical,labels=data['labels'],original_labels=data.get('original_labels'))
        destination=output/(path.stem+'.json.gz')
        with destination.open('wb') as stream:
            with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
                compressed.write(json.dumps(clean(payload),separators=(',',':'),allow_nan=False).encode())
        manifest['files'][destination.name]=dict(sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),original_pt_sha256=payload['original_pt_sha256'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();export(args.run,args.output)
