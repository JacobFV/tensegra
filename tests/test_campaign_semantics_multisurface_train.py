"""CPU safeguards for the prospective fixed-parent surface comparison."""
import copy
import json
from pathlib import Path
import pytest
from topoformer.campaign_semantics_multisurface_train import renderer_for, validate_config


def test_four_visits_balance_each_mixed_construction():
    # Both independently drawn phase outcomes must yield exactly two visits per language.
    for phase in (0, 1):
        mixed = [renderer_for('mixed', 0, [visit], [phase]) for visit in range(4)]
        assert mixed.count('english') == mixed.count('spanish') == 2
        assert [renderer_for('english', 0, [visit], [phase]) for visit in range(4)] == ['english'] * 4


def test_prepared_configs_cannot_launch_and_frozen_recipe_is_narrow():
    paths = [Path(f'configs/campaign-s13-{arm}-{job}-prepared.json')
             for arm in ('english', 'mixed') for job in ('profile', 'main')]
    for path in paths:
        config = json.loads(path.read_text())
        with pytest.raises(ValueError, match='freeze'):
            validate_config(config)
        config['budget_status'] = 'frozen'
        validate_config(config)
        for key, value in [('parent_checkpoint_sha256', 'wrong'), ('batch_size', 16),
                           ('negative_pairs', 64), ('renderer_seed', 2), ('learning_rate', 1e-4),
                           ('added_updates', 8192), ('checkpoints', [0, 8192])]:
            altered = copy.deepcopy(config); altered[key] = value
            with pytest.raises(ValueError):
                validate_config(altered)
