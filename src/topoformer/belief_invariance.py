"""Stage10: separate opaque ledger addresses from learned semantic compatibility."""
import random
import torch
from torch import nn
from .belief_state import BeliefModel
from .belief_contracts import contract_episodes, empty_ledger_frames

ARMS = ('ledger_only', 'raw_randomized', 'raw_correlated')
CONDITIONS = ('clean', 'reorder', 'duplicate', 'long_duplicate', 'contradiction',
              'retract', 'partial', 'empty', 'full_retract', 'distinct_equal',
              'candidate_permutation', 'id_rename', 'id_permute_seen',
              'distinct_equal_seen_id', 'distinct_equal_unseen_id')

class ContentEncoder(nn.Module):
    """Same allocated linear map in all arms; ledger-only masks ID inputs."""
    def __init__(self, linear, use_ids):
        super().__init__()
        self.linear = linear
        self.use_ids = use_ids

    def forward(self, features):
        if not self.use_ids:
            features = torch.cat((features[..., :-16], torch.zeros_like(features[..., -16:])), -1)
        return self.linear(features)

class InvariantBeliefModel(BeliefModel):
    def __init__(self, arm='ledger_only', width=1024, inner=2048):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__('protected', width=width, inner=inner, observation_id_features=True)
        self.arm = arm
        self.encode = ContentEncoder(self.encode, use_ids=arm != 'ledger_only')

    def forward(self, public):
        output = super().forward(public)
        prior = torch.zeros_like(output['logits'])
        prior[..., :-1] = prior[..., :-1].masked_fill(~public['valid'][:, None, :], -1e9)
        prior[..., -1] = -1e9
        output['logits'] = torch.where(empty_ledger_frames(public)[..., None], prior, output['logits'])
        return output

def rename_observations(episodes, allocator_seed):
    """Independent bijection preserves duplicate/retraction identity semantics."""
    result = []
    for index, episode in enumerate(episodes):
        handles = sorted({event[0] for event in episode['events'] if event[0] >= 0})
        assigned = random.Random(allocator_seed * 1000003 + index).sample(range(65536), len(handles))
        mapping = dict(zip(handles, assigned))
        result.append({**episode, 'events': [(mapping.get(i, i), action, role, value)
                                           for i, action, role, value in episode['events']]})
    return result

def make_training_episodes(count, seed, arm, condition='clean', candidates=8):
    if arm not in ARMS:
        raise ValueError(arm)
    episodes = contract_episodes(count, seed, candidates, condition)
    return rename_observations(episodes, seed + 17000000) if arm == 'raw_randomized' else episodes
