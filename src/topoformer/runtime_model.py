"""Learned semantic boundary around an external, protected exact runtime.

Only observable tensors enter this module. No interpreter state can be mutated by
these networks, and discrete execution has no implicit task-gradient estimator.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def binding_confidence(op_logits, binding_logits):
    """Local confidence, never a global mode; null probability vetoes confidence.

    This deliberately conservative combination is an uncalibrated score. Execution
    must independently check primitive types/schema and validate thresholds.
    """
    op = op_logits.softmax(-1)
    binding = binding_logits.softmax(-1)
    def certainty(p):
        entropy = -(p * p.clamp_min(1e-12).log()).sum(-1)
        top = p.topk(min(2, p.shape[-1]), dim=-1).values
        margin = top[..., 0] - (top[..., 1] if top.shape[-1] > 1 else 0)
        return (1 - entropy / math.log(max(2, p.shape[-1]))).clamp(0, 1) * margin
    return (certainty(op) * certainty(binding)).sqrt() * (1 - binding[..., -1])


class RuntimeBindingModel(nn.Module):
    """Small clause transformer + cold cosine lowering + learned lifting.

    Candidate order has no positional encoding. Surface word positions do have a
    learned encoding. Public clause segmentation and sequential scheduling are
    supplied priors, not learned parsing. All controls unroll the same schedule.
    """
    MODES = {'runtime', 'none', 'graph_data', 'soft', 'protected_learned'}

    def __init__(self, vocab_size, n_ops, *, key_dim=16, width=48, heads=2,
                 n_types=8, n_relations=4, output_classes=129, result_classes=129,
                 result_min=-64, result_scale=64., n_styles=3, temperature=.15,
                 strength=4., max_words=32):
        super().__init__()
        if width % heads or temperature <= 0 or result_scale <= 0:
            raise ValueError('invalid dimensions, temperature or result scale')
        self.key_dim, self.width = key_dim, width
        self.n_relations, self.max_words = n_relations, max_words
        self.result_scale, self.result_min = float(result_scale), result_min
        self.temperature, self.strength = float(temperature), float(strength)
        self.words = nn.Embedding(vocab_size, width, padding_idx=0)
        self.word_positions = nn.Embedding(max_words, width)
        layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.,
                                          batch_first=True, activation='gelu')
        self.clause_encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.reference = nn.Linear(key_dim, width)
        self.lower_query = nn.Linear(width, key_dim, bias=False)
        self.lower_key = nn.Linear(key_dim, key_dim, bias=False)
        self.null = nn.Linear(width, 1)
        self.operation = nn.Linear(width, n_ops)
        self.memory_key = nn.Linear(key_dim, width)
        self.memory_value = nn.Linear(1, width)
        self.types = nn.Embedding(n_types, width)
        self.relation_messages = nn.ModuleList(nn.Linear(width, width, bias=False)
                                               for _ in range(n_relations))
        self.neural_query = nn.Linear(width, width, bias=False)
        self.neural_key = nn.Linear(width, width, bias=False)
        self.recurrent = nn.GRUCell(2 * width, width)
        self.register_update = nn.GRUCell(2 * width, key_dim)
        self.register_query = nn.Linear(key_dim, width, bias=False)
        self.relation_selector = nn.Linear(width, n_relations)
        self.output = nn.Linear(width, output_classes)
        self.result = nn.Linear(width, result_classes)
        self.style = nn.Embedding(n_styles, width)
        self.comparison = nn.Linear(1, width)
        self.lifter = nn.Sequential(nn.Linear(width + 2, width), nn.GELU(),
                                    nn.Linear(width, output_classes))

    def lift(self, result, style, comparison=None):
        """Raw scalar + normalized comparison -> learned output, no class rule."""
        if comparison is None:
            comparison = torch.zeros_like(result)
        numeric = torch.stack((result.float() / self.result_scale, comparison.float()), -1)
        return self.lifter(torch.cat((numeric, self.style(style.long())), -1))

    def forward(self, public, *, mode='runtime'):
        if mode not in self.MODES:
            raise ValueError(f'unknown mode {mode}')
        surface = public['surface'].long()
        batch, steps, words = surface.shape
        if words > self.max_words:
            raise ValueError('surface exceeds max_words')
        flat = surface.reshape(batch * steps, words)
        padding = flat.eq(0)
        # Empty padded clauses need one finite attention key; their output is masked.
        safe_padding = padding.clone()
        safe_padding[:, 0] = False
        positions = torch.arange(words, device=surface.device)
        tokens = self.words(flat) + self.word_positions(positions)[None]
        encoded = self.clause_encoder(tokens, src_key_padding_mask=safe_padding)
        weights = (~padding).to(encoded.dtype)
        clause = (encoded * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
        clause = clause.reshape(batch, steps, self.width) + self.reference(public['reference'])
        op_logits = self.operation(clause)
        q = F.normalize(self.lower_query(clause), dim=-1)
        k = F.normalize(self.lower_key(public['candidate_keys']), dim=-1)
        scores = torch.einsum('bdk,bck->bdc', q, k) / self.temperature
        mask = public['candidate_mask'].bool()
        scores = scores.masked_fill(~mask[:, None], float('-inf'))
        binding_logits = torch.cat((scores, self.null(clause)), -1)
        answer = {'op_logits': op_logits, 'binding_logits': binding_logits,
                  'confidences': binding_confidence(op_logits, binding_logits),
                  'clause_states': clause}
        if mode == 'runtime':
            return answer
        memory = (self.memory_key(public['candidate_keys'])
                  + self.memory_value(public['candidate_values'][..., None])
                  + self.types(public['candidate_types'].long()))
        adjacency = public['adjacency'].to(memory.dtype)
        if adjacency.shape[1] != self.n_relations:
            raise ValueError('relation count differs from model')
        if mode == 'graph_data':
            messages = torch.zeros_like(memory)
            for r, transform in enumerate(self.relation_messages):
                degree = adjacency[:, r].sum(-1, keepdim=True).clamp_min(1)
                messages = messages + torch.bmm(adjacency[:, r], transform(memory)) / degree
            memory = memory + messages
        state = torch.zeros(batch, self.width, device=memory.device, dtype=memory.dtype)
        register = torch.zeros(batch, self.key_dim, device=memory.device, dtype=memory.dtype)
        attentions = []
        for t in range(steps):
            query = self.neural_query(state + clause[:, t])
            if mode == 'protected_learned':
                query = query + self.register_query(register)
            logits = torch.einsum('bw,bcw->bc', query, self.neural_key(memory)) / math.sqrt(self.width)
            if mode == 'soft':
                relation = self.relation_selector(clause[:, t]).softmax(-1)
                graph = torch.einsum('br,brij->bij', relation, adjacency)
                ground = binding_logits[:, t].softmax(-1)[:, :-1]
                logits = logits + self.strength * torch.bmm(ground[:, None], graph).squeeze(1)
            logits = logits.masked_fill(~mask, float('-inf'))
            # Null-only examples remain finite and retrieve zero observable memory.
            valid = mask.any(-1)
            logits = torch.where(valid[:, None], logits, torch.zeros_like(logits))
            attention = logits.softmax(-1) * mask
            retrieved = torch.bmm(attention[:, None], memory).squeeze(1)
            inputs = torch.cat((clause[:, t], retrieved), -1)
            update = self.recurrent(inputs, state)
            active = public['step_mask'][:, t].bool()[:, None]
            state = torch.where(active, update, state)
            if mode == 'protected_learned':
                # Separate learned register, no exact transition or generic-state write.
                register = torch.where(active, self.register_update(inputs, register), register)
            attentions.append(attention)
        style = self.style(public['style'].long())
        if 'comparison' in public:
            style = style + self.comparison(public['comparison'][..., None])
        answer.update(output_logits=self.output(state + style), result_logits=self.result(state),
                      attentions=torch.stack(attentions, 1), register=register)
        return answer
