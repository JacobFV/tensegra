"""C01 actual-proposal adapter and frozen consumer. Refusals remain absent."""
import hashlib
from pathlib import Path
import torch
from torch import nn
from .campaign_composition import make_model, model_inputs, execute_proposal
from .campaign_composition_acquire import canonical_prediction
from .campaign_returns_use import workspace, alter_public, answer
from .return_memory import ReturnMemoryModel
from .return_crossdelay import tensor_hash


def load_checked(path, expected, device):
    path = Path(path).expanduser()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected: raise ValueError(f'checkpoint hash mismatch: {path}')
    return torch.load(path, map_location=device, weights_only=True)


def frozen_interfaces(spec, device):
    lowerer = make_model().to(device)
    lowerer.load_state_dict(load_checked(spec['lowerer'], spec['lowerer_sha256'], device))
    backbone = ReturnMemoryModel(width=1024, encoding='factorized').to(device)
    backbone.load_state_dict(load_checked(spec['backbone'], spec['backbone_sha256'], device))
    saved = load_checked(spec['accessor'], spec['accessor_sha256'], device)
    accessor = nn.Linear(1024, 33).to(device); accessor.load_state_dict(saved['state'])
    consumer = nn.Sequential(nn.Linear(35, 1024), nn.GELU(), nn.Linear(1024, 2)).to(device)
    consumer.load_state_dict(load_checked(spec['consumer'], spec['consumer_sha256'], device))
    oracle = nn.Sequential(nn.Linear(35, 1024), nn.GELU(), nn.Linear(1024, 2)).to(device)
    oracle.load_state_dict(load_checked(spec['oracle_consumer'], spec['oracle_consumer_sha256'], device))
    query_only = nn.Sequential(nn.Linear(35, 1024), nn.GELU(), nn.Linear(1024, 2)).to(device)
    query_only.load_state_dict(load_checked(spec['query_consumer'], spec['query_consumer_sha256'], device))
    for model in (lowerer, backbone, accessor, consumer, oracle, query_only): model.eval().requires_grad_(False)
    return dict(lowerer=lowerer, backbone=backbone, accessor=accessor, mean=saved['mean'], scale=saved['scale'], consumer=consumer, oracle=oracle, query_only=query_only)


def propose(lowerer, rows, device, chunk=256):
    public = model_inputs(rows); parts = []
    with torch.no_grad():
        for start in range(0, len(rows), chunk):
            out = lowerer({k: v[start:start+chunk].to(device) for k, v in public.items()})
            parts.append({k: v.cpu() for k, v in out.items()})
    out = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    op, raw, canonical = canonical_prediction(out, public)
    return dict(primitive=op, raw_pointers=raw, canonical_pointers=canonical, logits=out)


def build_returns(rows, primitives, pointers):
    """Public rows and proposed actions only. Never accepts original events/labels."""
    accepted, events, reasons = [], [], []
    for i, row in enumerate(rows):
        result = execute_proposal(row, int(primitives[i]), tuple(int(x) for x in pointers[i]))
        reasons.append(result['reason'])
        if result['status'] == 'executed': accepted.append(i); events.append(result['event'])
    public = None
    if accepted:
        public = {key: torch.stack([rows[i][key] for i in accepted]) for key in ('argument_keys', 'provenance_keys', 'query', 'distractors')}
        public['event'] = {key: torch.cat([e[key] for e in events]) for key in events[0]}
    return dict(indices=torch.tensor(accepted, dtype=torch.long), public=public, reasons=reasons, total=len(rows))


def manipulate(bundle, kind, swap=None):
    """Interventions operate on actually returned facts, never gold reset."""
    if bundle['public'] is None: return dict(bundle, kind=kind)
    public = bundle['public']
    if kind == 'swap':
        if swap is None: raise ValueError('swap public event required')
        ids = bundle['indices']
        swap = {key: {k: v[ids] for k, v in item.items()} if isinstance(item, dict) else item[ids] for key, item in swap.items()}
    return dict(bundle, public=alter_public(public, kind, swap), kind=kind)


def consume(interfaces, bundle, delays, device, chunk=64):
    total = bundle['total']; indices = bundle['indices']; public = bundle['public']; drop = bundle.get('kind') == 'drop'
    result = {d: dict(logits=torch.full((total, 2), float('nan')), scores=torch.full((total, 33), float('nan')), predictions=torch.full((total,), -1, dtype=torch.long)) for d in delays}
    supplied = torch.full((total,), -1, dtype=torch.long)
    values = torch.full((total,), float('nan')); types = torch.full((total,), -1, dtype=torch.long)
    if public is None: return dict(cells=result, supplied=supplied, values=values, types=types)
    if not drop:
        supplied[indices] = answer(public['event']['values'][:, 0], public['query'])
        values[indices] = public['event']['values'][:, 0]; types[indices] = public['event']['types'][:, 0]
    with torch.no_grad():
        for start in range(0, len(indices), chunk):
            sl = slice(start, start+chunk); output_ids = indices[sl]
            small = {k: {n: v[sl].to(device) for n, v in item.items()} if isinstance(item, dict) else item[sl].to(device) for k, item in public.items()}
            states = workspace(interfaces['backbone'], small, delays, drop=drop)
            query = small['query'] / small['query'].new_tensor([8., 1.])
            for delay, state in states.items():
                scores = interfaces['accessor']((state-interfaces['mean'])/interfaces['scale'])
                logits = interfaces['consumer'](torch.cat((scores.softmax(-1), query), -1))
                result[delay]['scores'][output_ids] = scores.cpu(); result[delay]['logits'][output_ids] = logits.cpu(); result[delay]['predictions'][output_ids] = logits.argmax(-1).cpu()
    return dict(cells=result, supplied=supplied, values=values, types=types)


def exact_copy(interfaces, bundle, device):
    """Engineering scalar copy into separately acquired R05 oracle comparator."""
    pred = torch.full((bundle['total'],), -1, dtype=torch.long)
    if bundle['public'] is None: return pred
    public = bundle['public']; values = public['event']['values'][:, 0]
    encoded = torch.nn.functional.one_hot((2*values+16).long(), 33).float().to(device)
    if bundle.get('kind') == 'drop': encoded.zero_()
    query = public['query'].to(device) / encoded.new_tensor([8., 1.])
    with torch.no_grad(): pred[bundle['indices']] = interfaces['oracle'](torch.cat((encoded, query), -1)).argmax(-1).cpu()
    return pred


def interfaces_state_hashes(interfaces):
    """In-memory frozen parameters/buffers plus scalar normalization state."""
    return {name: tensor_hash(value.state_dict() if isinstance(value, nn.Module) else value)
            for name, value in interfaces.items() if isinstance(value, (nn.Module, torch.Tensor))}
