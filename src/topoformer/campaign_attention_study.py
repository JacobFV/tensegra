"""Bounded A01 runner, compact independent-audit records."""
import argparse
import gzip
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .campaign_attention import RoutingModel, generate, targets, corrupt, metrics


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, config, output, label, *, device):
    model.eval()
    rows = []
    arrays = {}
    conditions = config.get('conditions', [{'nodes':16,'depth':4}, {'nodes':32,'depth':4},
         {'nodes':16,'depth':8}, {'nodes':32,'depth':8}, {'nodes':16,'depth':4,'composition':True}])
    started = time.monotonic()
    for ci, condition in enumerate(conditions):
        values = {}
        preds, golds, routes, starts, relations, successors = [], [], [], [], [], []
        for offset in range(0, config['eval_examples'], config['eval_batch']):
            count = min(config['eval_batch'], config['eval_examples'] - offset)
            batch = generate(count, condition['nodes'], condition['depth'],
                seed=config['eval_seed'] + condition.get('data_group',ci) * 100000 + offset, device=device,
                heldout_composition=condition.get('composition', False))
            gold = targets(batch)
            given = corrupt(batch, condition.get('corruption','clean'), config['eval_seed']+ci*100000+offset+50000)
            result = model(given, config['mode'], zero_strength=condition.get('zero_strength', False),
                           strength_override=config.get('strength_override'), size_adjust=config.get('size_adjust',False))
            # Path agreement and values remain relative to the original clean graph.
            scores = metrics(result, gold, batch)
            for key, value in scores.items():
                values.setdefault(key, []).append(value.cpu().numpy())
            preds.append(result['logits'].argmax(-1).cpu().numpy().astype('uint8'))
            golds.append(gold.cpu().numpy().astype('uint8'))
            routes.append(result['routes'].cpu().numpy().astype('uint8'))
            starts.append(batch.starts.cpu().numpy().astype('uint8'))
            relations.append(batch.relations.cpu().numpy().astype('uint8'))
            successors.append(batch.adjacency.argmax(-1).cpu().numpy().astype('uint8'))
        values = {k: np.concatenate(v) for k,v in values.items()}
        rows.append({'condition':condition, 'examples':config['eval_examples'],
                     **{k:float(v.mean()) for k,v in values.items()}})
        for key,value in values.items(): arrays[f'c{ci}_{key}'] = value
        for key,value in [('pred',preds),('gold',golds),('route',routes),('start',starts),('relation',relations),('successor',successors)]:
            arrays[f'c{ci}_{key}'] = np.concatenate(value)
    sync(device)
    np.savez_compressed(output / f'{label}.npz', **arrays)
    write(output / f'{label}.json', {'rows':rows, 'wall_seconds_including_export':time.monotonic()-started})
    model.train()
    return rows


def run(config, output):
    output.mkdir(parents=True, exist_ok=False)
    write(output/'config.json', config)
    write(output/'source.json', {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
         for p in (Path(__file__), Path(__file__).with_name('campaign_attention.py'))})
    device = config.get('device','cuda')
    torch.set_num_threads(2)
    torch.manual_seed(config['seed'])
    model = RoutingModel(width=config.get('width',1024), strength=config.get('strength',4.)).to(device)
    if config.get('checkpoint'):
        path = Path(config['checkpoint'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != config['checkpoint_sha256']:
            raise ValueError('checkpoint checksum differs')
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
    initial_hash = hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr',3e-4), weight_decay=1e-4)
    start = time.monotonic()
    curves, losses = [], []
    microsteps = 0
    steps = config['steps']
    checkpoints = sorted(set([0,steps]+config.get('checkpoints',[])))
    if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    for step in range(steps+1):
        if step in checkpoints:
            rows = evaluate(model,config,output,f'eval-{step:05d}',device=device)
            curves.append({'step':step,'rows':rows,'elapsed_seconds':time.monotonic()-start})
            print(json.dumps({'step':step,'mode':config['mode'],'task':[r['task'] for r in rows],
                              'elapsed':time.monotonic()-start}),flush=True)
        if step == steps: break
        depth = step % 4 + 1
        batch = generate(config['batch'],config.get('nodes',16),depth,
                         seed=config['train_seed']+step,device=device,train=True)
        gold = targets(batch)
        optimizer.zero_grad(set_to_none=True)
        result = model(batch,config['mode'], size_adjust=config.get('size_adjust',False))
        loss = F.cross_entropy(result['logits'].flatten(0,2),gold.flatten())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        microsteps += depth*config['batch']
        if step % 10 == 0:
            losses.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(norm),
                           'depth':depth})
    sync(device)
    checkpoint=output/'checkpoint.pt'
    torch.save(model.state_dict(),checkpoint)
    manifest = {'mode':config['mode'],'seed':config['seed'],'width':model.width,
                'parameters_allocated':sum(p.numel() for p in model.parameters()),
                'parameters_with_gradient':sum(p.numel() for p in model.parameters() if p.grad is not None),
                'presentations':steps*config['batch'],'generated_graph_examples':steps*config['batch'],
                'unique_canonical_graphs':None, 'deduplication':'not performed; independent procedural draws',
                'initial_tensor_sha256':initial_hash,'config_sha256':config_hash,
                'node_microsteps':microsteps*config.get('nodes',16), 'graph_microsteps':microsteps,
                'wall_seconds_including_eval_export':time.monotonic()-start,
                'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
                'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'learned_strength':model.strength.detach().cpu().tolist(),
                'context_scale':float(model.context_log_scale.exp().detach()),'curves':curves}
    write(output/'manifest.json',manifest)
    with gzip.open(output/'losses.json.gz','wt') as f: json.dump(losses,f)
    print(json.dumps({'completed':str(output),'seconds':manifest['wall_seconds_including_eval_export']}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(json.loads(args.config.read_text()),args.output)
