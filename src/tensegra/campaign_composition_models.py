"""C01 equal-public-evidence neural controls; no exact primitive in forward."""
import torch
from torch import nn
from .interface_proposals import ProposalModel


def numeric_features(public):
    types = public['operand_types']
    return torch.cat((public['operand_values'][..., None] / 16,
                      torch.stack((types == 0, types == 1, types == -1), -1).float(),
                      (types >= 0)[..., None].float()), -1)


def query_features(public):
    return public['query'] / public['query'].new_tensor([8., 1.])


class NeuralOperandBaseline(nn.Module):
    """N1: learned soft address retrieval plus unconstrained neural answer head."""
    def __init__(self, hidden=1024):
        super().__init__()
        self.lowerer = ProposalModel(key_dim=32, hidden=hidden)
        self.mlp = nn.Sequential(nn.Linear(17, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.answer = nn.Linear(hidden, 2); self.value = nn.Linear(hidden, 33)

    def forward(self, public):
        lowering = self.lowerer(public)
        operands = lowering['pointers'][:, 1:].softmax(-1) @ numeric_features(public)
        features = torch.cat((operands.flatten(1), lowering['primitive'].softmax(-1), query_features(public)), -1)
        hidden = self.mlp(features)
        return dict(**lowering, answer=self.answer(hidden), value=self.value(hidden))


class ContextualBaseline(nn.Module):
    """N2: all16 public tokens, dense learned attention, no oracle addressing."""
    def __init__(self, hidden=1024, heads=8):
        super().__init__()
        self.instruction = nn.Linear(101, hidden)
        self.table = nn.Linear(39, hidden)
        self.query_input = nn.Linear(34, hidden)
        self.kind = nn.Parameter(torch.zeros(3, hidden))
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=hidden, dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.answer = nn.Linear(hidden, 2); self.value = nn.Linear(hidden, 33)
        self.operation = nn.Linear(hidden, 5)
        self.pointer_query = nn.Linear(hidden, 3 * hidden)
        self.pointer_key = nn.Linear(hidden, hidden, bias=False)
        self.hidden = hidden

    def forward(self, public):
        instruction = torch.cat((public['instruction_destinations'], public['instruction_arguments'].flatten(2), public['instruction_cues']), -1)
        types = public['operand_types']; null = public['keys'].abs().sum(-1) == 0
        roles = torch.stack((types >= 0, null, (types == -1) & ~null), -1).float()
        # numeric value +3 type indicators +3 public role indicators =7.
        table = torch.cat((public['keys'], numeric_features(public)[..., :4], roles), -1)
        query = torch.cat((public['query_destination'], query_features(public)), -1)
        tokens = torch.cat((self.instruction(instruction) + self.kind[0], self.table(table) + self.kind[1],
                            (self.query_input(query) + self.kind[2])[:, None]), 1)
        context = self.norm(self.blocks(tokens)); state = context[:, -1]
        pointers = self.pointer_query(state).reshape(-1, 3, self.hidden) @ self.pointer_key(context[:, 4:15]).transpose(1, 2)
        return dict(answer=self.answer(state), value=self.value(state), primitive=self.operation(state),
                    pointers=pointers * self.hidden ** -.5)
