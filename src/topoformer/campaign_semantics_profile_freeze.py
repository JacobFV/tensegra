"""CPU-only immutable profile preparation; freezing never grants GPU permission."""
import argparse,gzip,hashlib,json
from pathlib import Path

COMMON=('campaign_semantics.py','campaign_semantics_data.py','campaign_semantics_continue.py','campaign_semantics_lr.py',
        'semantic_curriculum.py','semantic_text_acquisition.py','semantic_scaling.py','semantic_contracts.py',
        'thinking.py','thinking_language.py','semantic_graph.py','tcn_data.py')
SOURCES={
    's13':COMMON+('campaign_semantics_multisurface_data.py','campaign_semantics_surface_contract.py',
                 'campaign_semantics_multisurface_train.py','campaign_semantics_multisurface_launch.py'),
    's14':COMMON+('campaign_semantics_motif_inference.py','campaign_semantics_motif_launch.py'),
    'rename':COMMON+('campaign_semantics_confirmation_rename.py','campaign_semantics_confirmation_rename_profile.py',
                    'campaign_semantics_confirmation_rename_profile_launch.py'),
}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def freeze(prepared,output,source_root,lane):
    config=json.loads(Path(prepared).read_text())
    if config.get('budget_status')!='prepared_not_released' or config.get('job')!='profile':raise ValueError('prepared profile only')
    if lane not in SOURCES:raise ValueError('unregistered lane')
    if lane=='s13' and (config['added_updates'],config['checkpoints'],config['batch_size'],config['learning_rate'])!=(20,[0,20],8,1e-5):raise ValueError('S13 mechanical profile changed')
    if lane in ('s14','rename') and sorted((e['seed'],e['arm']) for e in config['runs'])!=[(s,a) for s in (701,702,703) for a in ('constant','decay')]:raise ValueError('all six required')
    names=SOURCES[lane]+('campaign_semantics_profile_freeze.py',)
    config.update(budget_status='frozen',profile_lane=lane,prepared_sha256=digest(prepared),
                  profile_source_sha256={name:digest(Path(source_root)/name) for name in names},
                  launch_authority='separate explicit root release required; no allocation granted by this file')
    with Path(output).open('x') as stream:stream.write(json.dumps(config,indent=2)+'\n')
    return config


def verify_profile_source(config,source_root):
    lane=config['profile_lane']
    if config.get('budget_status')!='frozen' or config.get('job')!='profile':raise ValueError('frozen profile only')
    expected=set(SOURCES[lane]+('campaign_semantics_profile_freeze.py',))
    if set(config['profile_source_sha256'])!=expected:raise ValueError('incomplete profile source binding')
    for name,sha in config['profile_source_sha256'].items():
        if digest(Path(source_root)/name)!=sha:raise ValueError('profile source changed: '+name)


def require_complete_matrix(root='results'):
    """Bytes only: all six fixed outcomes must exist before any diagnostic profile."""
    receipts=[]
    for seed in (701,702,703):
        for arm in ('constant','decay'):
            run=Path(root)/f's12-{arm}-{seed}';manifest=json.load(gzip.open(run/'manifest.json.gz','rt'))
            completion=json.loads((run/'completion.json').read_text())
            if completion.get('success') is not True:raise ValueError('incomplete primary endpoint')
            c=manifest['config']
            if c['seed']!=seed or c['updates']!=24576 or c['learning_rate']!={'constant':1e-4,'decay':1e-5}[arm]:raise ValueError('wrong primary endpoint')
            endpoint=next(e for e in manifest['curves'] if e['update']==24576)
            if digest(run/'model-u24576.pt')!=endpoint['checkpoint_sha256'] or digest(run/endpoint['artifact'])!=endpoint['sha256']:raise ValueError('primary bytes changed')
            receipts.append(dict(seed=seed,arm=arm,manifest_sha256=digest(run/'manifest.json.gz'),checkpoint_sha256=endpoint['checkpoint_sha256']))
    return receipts

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('lane',choices=SOURCES);p.add_argument('prepared');p.add_argument('output');p.add_argument('--source-root',default='src/topoformer');a=p.parse_args()
    freeze(a.prepared,a.output,a.source_root,a.lane)
