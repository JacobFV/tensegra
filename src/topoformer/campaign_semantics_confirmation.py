"""S12 frozen job dispatch and mechanical parent-checkpoint binding."""
import argparse,gzip,json
from pathlib import Path
from .campaign_semantics import digest,run as parent_run

CONFIRMATION_SHA='fc85da89fe4e1a522d1e0256fc1e58f8afb279494cb1af7ee4ac7235403a43f8'

def validate(config):
    if config['seed'] not in (701,702,703):raise ValueError('unregistered initialization seed')
    common=dict(batch_size=8,negative_pairs=128,calibration_count=128,train_limit=8192)
    if any(config[k]!=v for k,v in common.items()):raise ValueError('confirmation recipe changed')
    if config.get('budget_status')!='frozen':raise ValueError('unfrozen job')
    if config['job']=='parent':
        if config['updates']!=16384 or config['learning_rate']!=1e-4 or config['checkpoints']!=[0,8192,16384]:raise ValueError('parent recipe changed')
    elif config['job']=='fork':
        if config['updates']!=24576 or config['learning_rate'] not in (1e-4,1e-5) or config['checkpoints']!=[16384,24576]:raise ValueError('fork recipe changed')
        if config['confirmation_sha256']!=CONFIRMATION_SHA:raise ValueError('confirmation cache contract changed')
    else:raise ValueError('unknown job')

def bind_parent(template,parent_dir,output):
    """Attach byte hashes only; never inspect/select prediction correctness."""
    c=json.loads(Path(template).read_text());validate(c);root=Path(parent_dir)
    if c['job']!='fork':raise ValueError('fork template required')
    manifest=root/'manifest.json.gz';m=json.load(gzip.open(manifest,'rt'))
    if m['config']['seed']!=c['seed'] or m['config']['updates']!=16384:raise ValueError('wrong independent parent')
    checkpoint=root/'model-u16384.pt';evaluation=root/'evaluation-u16384.json.gz'
    final=next(x for x in m['curves'] if x['update']==16384)
    if digest(checkpoint)!=final['checkpoint_sha256'] or digest(evaluation)!=final['sha256']:raise ValueError('parent artifact hash mismatch')
    c.update(parent_checkpoint=str(checkpoint),parent_checkpoint_sha256=digest(checkpoint),parent_evaluation=str(evaluation),parent_evaluation_sha256=digest(evaluation),parent_manifest=str(manifest),parent_manifest_sha256=digest(manifest))
    path=Path(output)
    with path.open('x') as f:f.write(json.dumps(c,indent=2)+'\n')
    return c

def run(c):
    validate(c)
    if c['job']=='parent':return parent_run(c)
    from .campaign_semantics_confirmation_train import run as fork_run
    return fork_run(c)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--bind-parent');p.add_argument('--output');a=p.parse_args()
    if a.bind_parent:bind_parent(a.config,a.bind_parent,a.output)
    else:run(json.loads(Path(a.config).read_text()))
