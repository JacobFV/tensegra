"""S12 CPU contract fixtures: no model predictions or confirmation exposure."""
import copy,gzip,json
from pathlib import Path
import pytest
from topoformer.campaign_semantics import digest
from topoformer.campaign_semantics_confirmation import validate,bind_parent

CONFIG=Path(__file__).resolve().parents[1]/'configs'

def test_prescribed_recipe_and_independent_seeds():
    for seed in (701,702,703):
        for label in ('parent','constant','decay'):
            suffix='' if label=='parent' else '-template'
            c=json.loads((CONFIG/f'campaign-s12-{label}-{seed}{suffix}.json').read_text());validate(c)
            bad=copy.deepcopy(c);bad['seed']=201
            with pytest.raises(ValueError):validate(bad)
            bad=copy.deepcopy(c);bad['updates']+=1
            with pytest.raises(ValueError):validate(bad)

def test_binding_is_same_parent_and_byte_hash_only(tmp_path):
    parent=tmp_path/'parent';parent.mkdir()
    checkpoint=parent/'model-u16384.pt';checkpoint.write_bytes(b'not-loaded-as-a-model')
    evaluation=parent/'evaluation-u16384.json.gz';evaluation.write_bytes(b'not-loaded-as-predictions')
    m=dict(config=dict(seed=701,updates=16384),curves=[dict(update=16384,checkpoint_sha256=digest(checkpoint),sha256=digest(evaluation))])
    with gzip.open(parent/'manifest.json.gz','wt') as f:json.dump(m,f)
    bound=[]
    for arm in ('constant','decay'):
        bound.append(bind_parent(CONFIG/f'campaign-s12-{arm}-701-template.json',parent,tmp_path/f'{arm}.json'))
    for key in ('parent_checkpoint_sha256','parent_evaluation_sha256','parent_manifest_sha256'):
        assert bound[0][key]==bound[1][key]
    assert bound[0]['learning_rate']==1e-4 and bound[1]['learning_rate']==1e-5
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError):bind_parent(CONFIG/'campaign-s12-decay-701-template.json',parent,tmp_path/'corrupt.json')
