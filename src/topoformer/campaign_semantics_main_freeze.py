"""Immutable post-profile main preparations; every launch needs root release."""
import argparse,json
from pathlib import Path
from campaign_semantics_profile_freeze import COMMON,digest,require_complete_matrix

SOURCES={
 's13':COMMON+('campaign_semantics_multisurface_data.py','campaign_semantics_surface_contract.py','campaign_semantics_multisurface_train.py','campaign_semantics_multisurface_main_launch.py'),
 's14':COMMON+('campaign_semantics_motif_inference.py','campaign_semantics_motif_main_launch.py'),
 'rename':COMMON+('campaign_semantics_confirmation_rename.py','campaign_semantics_confirmation_rename_main_launch.py'),
}


def validate_recipe(config,lane):
    if config.get('job')!='main':raise ValueError('registered main only')
    if lane=='s13':
        if (config['added_updates'],config['checkpoints'],config['batch_size'],config['negative_pairs'],config['renderer_seed'],config['learning_rate'])!=(4096,[0,1024,2048,4096],8,128,913001,1e-5):raise ValueError('S13 main recipe changed')
        if config['arm'] not in ('english','mixed') or config['parent_checkpoint_sha256']!='3799ade595500a083b9a558b6a5bc97c5cbc1b8bdce6e7e4b86d97b937ea373e':raise ValueError('fixed S11 arm/parent changed')
    elif lane in ('s14','rename'):
        if sorted((e['seed'],e['arm']) for e in config['runs'])!=[(s,a) for s in (701,702,703) for a in ('constant','decay')]:raise ValueError('all six endpoints required')
        expected={'s14':'8b75c555e33ea69e3cc15a1ed7a922acd5b631294a0b1c57ccd9429711eea4d5','rename':'a3f3bc5ed3d6fb188f5fb787d079af532cc67ec170a6b89937ea1d0f7777a4fb'}[lane]
        if config['cache_sha256']!=expected:raise ValueError('fixed diagnostic population changed')
    else:raise ValueError('unregistered lane')


def freeze(prepared,output,source_root,lane,matrix,cap):
    config=json.loads(Path(prepared).read_text());validate_recipe(config,lane)
    if config.get('budget_status')!='prepared_not_released':raise ValueError('prepared main only')
    if not 0<cap<=3600:raise ValueError('invalid proposed cap')
    if sorted((e['seed'],e['arm']) for e in matrix)!=[(s,a) for s in (701,702,703) for a in ('constant','decay')]:raise ValueError('complete immutable primary matrix required')
    names=SOURCES[lane]+('campaign_semantics_main_freeze.py','campaign_semantics_profile_freeze.py')
    config.update(budget_status='frozen',main_lane=lane,prepared_sha256=digest(prepared),primary_matrix=matrix,proposed_cap_seconds=cap,
        main_source_sha256={name:digest(Path(source_root)/name) for name in names},
        launch_authority='proposal for review only; separate root allocation and explicit release required')
    with Path(output).open('x') as stream:stream.write(json.dumps(config,indent=2)+'\n')
    return config


def verify_main_source(config,source_root,cap):
    lane=config['main_lane'];validate_recipe(config,lane)
    if config.get('budget_status')!='frozen' or cap!=config['proposed_cap_seconds']:raise ValueError('frozen proposed main/cap mismatch')
    names=set(SOURCES[lane]+('campaign_semantics_main_freeze.py','campaign_semantics_profile_freeze.py'))
    if set(config['main_source_sha256'])!=names:raise ValueError('incomplete main source binding')
    for name,sha in config['main_source_sha256'].items():
        if digest(Path(source_root)/name)!=sha:raise ValueError('main source changed: '+name)
    matrix=require_complete_matrix()
    if matrix!=config['primary_matrix']:raise ValueError('primary matrix differs from frozen main binding')
    return matrix

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('lane',choices=SOURCES);p.add_argument('prepared');p.add_argument('output');p.add_argument('--source-root',default='src/topoformer');p.add_argument('--matrix-json',required=True);p.add_argument('--cap',type=float,required=True);a=p.parse_args()
    freeze(a.prepared,a.output,a.source_root,a.lane,json.loads(Path(a.matrix_json).read_text())['all_six_matrix'],a.cap)
