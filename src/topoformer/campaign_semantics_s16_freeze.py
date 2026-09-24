"""S16 stdlib-only recipe/source guard; no release authority or model import."""
import argparse
import hashlib
import json
from pathlib import Path

ORIGINAL_SHA='fc85da89fe4e1a522d1e0256fc1e58f8afb279494cb1af7ee4ac7235403a43f8'
RENAMED_SHA='a3f3bc5ed3d6fb188f5fb787d079af532cc67ec170a6b89937ea1d0f7777a4fb'
SOURCE_NAMES=(
    'campaign_semantics_s16_freeze.py','campaign_semantics_s16.py','campaign_semantics_s16_launch.py',
    'campaign_semantics_identity_contract.py','campaign_semantics_motif_inference.py',
    'campaign_semantics_profile_freeze.py','campaign_semantics.py','campaign_semantics_data.py',
    'campaign_semantics_continue.py','campaign_semantics_lr.py','semantic_curriculum.py',
    'semantic_text_acquisition.py','semantic_scaling.py','semantic_contracts.py','thinking.py',
    'thinking_language.py','semantic_graph.py','tcn_data.py')
ENDPOINTS=[(seed,arm) for seed in (701,702,703) for arm in ('constant','decay')]


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def validate(config):
    if config['job'] not in ('profile','main'):raise ValueError('unregistered S16 job')
    if [(e['seed'],e['arm']) for e in config['runs']]!=ENDPOINTS:raise ValueError('all six fixed ordered endpoints required')
    if [(e['seed'],e['arm']) for e in config['primary_matrix']]!=ENDPOINTS:raise ValueError('all six inherited bindings required')
    if config['cache_sha256']!=ORIGINAL_SHA or config['renamed_cache_sha256']!=RENAMED_SHA:raise ValueError('fixed public populations changed')
    if config['normalization']!='S16 namespace-aware public first occurrence v1':raise ValueError('normalization recipe changed')
    if config['examples']!=(8 if config['job']=='profile' else 1024):raise ValueError('fixed prefix/full population required')
    if config['threshold_policy']!='original TRAIN unchanged' or config['original_reference']!='frozen primary outputs':raise ValueError('refit or reference change prohibited')
    if config['device']!='cuda':raise ValueError('profile/main require coordinator CUDA scheduling')
    if config['job']=='profile' and config.get('proposed_cap_seconds')!=60:raise ValueError('registered first profile cap60 required')


def freeze(prepared,output,source_root,cap):
    config=json.loads(Path(prepared).read_text());validate(config)
    if config['budget_status']!='prepared_not_released':raise ValueError('fresh prepared config required')
    if cap<=0 or cap>3600 or (config['job']=='profile' and cap!=60):raise ValueError('invalid cap')
    config.update(budget_status='frozen',proposed_cap_seconds=cap,prepared_sha256=digest(prepared),
        source_sha256={name:digest(Path(source_root)/name) for name in SOURCE_NAMES},
        launch_authority='separate root GPU release required')
    with Path(output).open('x') as stream:json.dump(config,stream,indent=2)
    return config


def verify(config,source_root,cap=None):
    validate(config)
    if config['budget_status']!='frozen':raise ValueError('source/budget freeze required')
    registered_cap=config.get('proposed_cap_seconds')
    if not isinstance(registered_cap,(int,float)) or not 0<registered_cap<=3600:
        raise ValueError('positive frozen process cap required')
    if cap is not None and cap!=config['proposed_cap_seconds']:raise ValueError('registered cap mismatch')
    if set(config['source_sha256'])!=set(SOURCE_NAMES):raise ValueError('incomplete source binding')
    for name,sha in config['source_sha256'].items():
        if digest(Path(source_root)/name)!=sha:raise ValueError('source changed: '+name)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prepared');p.add_argument('output')
    p.add_argument('--source-root',default='src/topoformer');p.add_argument('--cap',type=float,required=True)
    a=p.parse_args();freeze(a.prepared,a.output,a.source_root,a.cap)
