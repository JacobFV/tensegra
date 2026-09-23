"""Capacity-matched return encoding, availability and workspace-only readout.

Six facet encoders contain exactly ``width`` coordinates in total. Compressed
encoding sums their disjoint padded channels into one token; factorized encoding
keeps six tokens. Thus trainable capacity and information coordinates match,
while attention memory allocation/token cost deliberately differs.
"""
import math
import torch
from torch import nn
from .thinking import ThinkingConfig, _WorkspaceBlock
from .retention import ReturnRegister

FIELDS = ('value', 'type', 'operation', 'argument0', 'argument1', 'provenance')


class ReturnMemoryModel(nn.Module):
    def __init__(self, feature_dim=32, width=1024, value_limit=8, heads=8, encoding="factorized"):
        super().__init__()
        if width < 6 or width % heads:
            raise ValueError('width must support six facets and attention heads')
        self.width, self.feature_dim, self.value_limit = width, feature_dim, value_limit
        self.encoding = encoding
        self.sizes = [width]*6 if encoding == "mixed" else [width // 6 + (i < width % 6) for i in range(6)]
        self.encoders = nn.ModuleList([
            nn.Linear(1, self.sizes[0]), nn.Embedding(3, self.sizes[1]),
            nn.Embedding(5, self.sizes[2]), nn.Linear(feature_dim, self.sizes[3]),
            nn.Linear(feature_dim, self.sizes[4]), nn.Linear(feature_dim, self.sizes[5])])
        self.mixer = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.initial = nn.Parameter(torch.randn(6, width) * .02)
        self.initial_read = nn.MultiheadAttention(width, heads, batch_first=True)
        self.features = nn.Linear(feature_dim, width)
        config = ThinkingConfig(feature_dim=feature_dim, width=width, heads=heads, structural_heads=0)
        self.blocks = nn.ModuleList(_WorkspaceBlock(config) for _ in range(4))
        self.head_count = heads
        self.norm = nn.LayerNorm(width)
        self.scalar_heads = nn.ModuleList([nn.Linear(width, 4*value_limit+1), nn.Linear(width,3), nn.Linear(width,5)])
        self.identity_heads = nn.ModuleList(nn.Linear(width, feature_dim) for _ in range(3))
        self.null_heads = nn.ModuleList(nn.Linear(width,1) for _ in range(2))

    def encode(self, event, encoding):
        if encoding not in ('compressed', 'factorized', 'mixed'):
            raise ValueError('unknown encoding')
        if (encoding == 'mixed') != (self.encoding == 'mixed'):
            raise ValueError('mixed encoder must be selected at construction')
        values = [event['values']/self.value_limit, event['types'], event['operations'],
                  event['arguments'][:,:,0], event['arguments'][:,:,1], event['provenance']]
        facets, offset = [], 0
        for i, (encoder, value) in enumerate(zip(self.encoders, values)):
            if i == 0: value = value.unsqueeze(-1)
            encoded = encoder(value)
            facets.append(encoded if encoding == 'mixed' else torch.nn.functional.pad(encoded, (offset, self.width-offset-self.sizes[i])))
            offset += self.sizes[i]
        tokens = torch.cat(facets,1)
        if encoding in ('compressed','mixed'): tokens = tokens.sum(1,keepdim=True)
        return self.mixer(tokens)

    def decode(self, state, public):
        state = self.norm(state)
        result = {field: head(state[:,i]) for i,(field,head) in enumerate(zip(FIELDS,self.scalar_heads))}
        for j, field in enumerate(FIELDS[3:]):
            keys = public['provenance_keys'] if field == 'provenance' else public['argument_keys']
            scores = torch.einsum('bf,bkf->bk',self.identity_heads[j](state[:,j+3]),keys)/math.sqrt(self.feature_dim)
            if j < 2: scores = torch.cat((scores[:,:-1],self.null_heads[j](state[:,j+3])),1)
            result[field] = scores
        return result

    def forward(self, public, steps=1, encoding='factorized', availability='persistent',
                intervention='none', replacement_event=None):
        if availability not in ('once','persistent') or not 0 <= steps <= 32:
            raise ValueError('invalid availability or step count')
        if intervention not in ('none','event_drop','wrong_value','wrong_type','wrong_provenance','release','overwrite','unrelated_activity'):
            raise ValueError('unknown intervention')
        event = {k:v.clone() for k,v in public['event'].items()}
        if intervention == 'wrong_value':
            event['values'] = torch.where(event['values'] != 0,-event['values'],torch.ones_like(event['values']))
        if intervention == 'wrong_type': event['types'] = (event['types']+1)%3
        if intervention == 'wrong_provenance':
            keys = public['provenance_keys']; current = (keys-event['provenance']).square().sum(-1).argmin(-1)
            event['provenance'] = keys[torch.arange(keys.shape[0],device=keys.device),(current+1)%keys.shape[1]][:,None]
        register = ReturnRegister(event)
        state = self.initial[None].expand(event['values'].shape[0],-1,-1)
        if intervention == 'event_drop': register.release()
        else:
            memory = self.encode(event,encoding)
            read,_ = self.initial_read(state,memory,memory,need_weights=False)
            state = state+read
        initial_state = state
        lifecycle_before = lifecycle_after = None
        for step in range(steps):
            if step == steps//2 and intervention in ('release','overwrite'):
                lifecycle_before = state.detach().clone()
                if intervention == 'release': register.release()
                else:
                    if replacement_event is None: raise ValueError('overwrite requires replacement event')
                    register.overwrite(replacement_event)
                lifecycle_after = state.detach().clone()
            distractors = public['distractors'][:,step]
            if intervention == 'unrelated_activity': distractors = -distractors
            memory = self.features(distractors)
            record = register.read()
            if availability == 'persistent' and record is not None:
                memory = torch.cat((memory,self.encode(record,encoding)),1)
            mask = torch.ones(memory.shape[:2],dtype=torch.bool,device=state.device)
            bias = state.new_zeros(state.shape[0],self.head_count,6,6)
            for block in self.blocks: state,_ = block(state,memory,mask,bias)
        return dict(logits=self.decode(state,public),state=state,initial_state=initial_state,
                    register=register.read(),lifecycle_before=lifecycle_before,lifecycle_after=lifecycle_after,
                    memory_tokens=6 if encoding=='factorized' else 1,facet_coordinates=sum(self.sizes))
