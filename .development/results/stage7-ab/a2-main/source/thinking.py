"""Recurrent current-state workspace; no latent-history cache or privileged inputs.

All tensor features describe current public observations. Caller-owned persistent
runtime values are distinct from workspace rows, which carry no thought labels.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ThinkingConfig:
    feature_dim: int
    width: int = 32
    heads: int = 4
    structural_heads: int = 2
    workspace_rows: int = 8
    candidates: int = 2
    n_ops: int = 5
    max_arguments: int = 2
    n_types: int = 3
    n_relations: int = 2
    output_classes: int = 129
    min_microsteps: int = 2
    max_microsteps: int = 8
    soft_min_microsteps: float = 2.
    soft_max_microsteps: float = 10.
    emit_alpha: float = .5
    emit_beta: float = .05
    temperature: float = .3
    structural_strength: float = 2.
    ponder_cost: float = .01

    def __post_init__(self):
        if self.width % self.heads or not 0 <= self.structural_heads <= self.heads:
            raise ValueError('invalid attention dimensions')
        if min(self.feature_dim, self.workspace_rows, self.candidates, self.max_arguments, self.n_relations) < 1:
            raise ValueError('dimensions must be positive')
        if self.temperature <= 0 or not 0 <= self.min_microsteps <= self.max_microsteps:
            raise ValueError('invalid temperature or microstep limits')
        if not (0 < self.soft_min_microsteps <= self.soft_max_microsteps and self.emit_alpha > 0 and self.emit_beta > 0):
            raise ValueError('soft emission thresholds and coefficients must be positive and ordered')


def _masked_softmax(logits, mask):
    masked = logits.masked_fill(~mask, -torch.inf)
    safe = torch.where(mask.any(-1, keepdim=True), masked, torch.zeros_like(masked))
    return safe.softmax(-1) * mask.to(logits.dtype)


class _WorkspaceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        w = config.width
        self.heads, self.head_width = config.heads, w // config.heads
        self.norm = nn.LayerNorm(w)
        self.qkv = nn.Linear(w, 3*w)
        self.output = nn.Linear(w,w)
        self.cross_norm = nn.LayerNorm(w)
        self.cross = nn.MultiheadAttention(w, config.heads, dropout=0, batch_first=True)
        self.ff_norm = nn.LayerNorm(w)
        self.ff = nn.Sequential(nn.Linear(w,2*w),nn.GELU(),nn.Linear(2*w,w))

    def forward(self, state, memory, mask, bias):
        b,s,w = state.shape
        q,k,v = self.qkv(self.norm(state)).reshape(b,s,3,self.heads,self.head_width).permute(2,0,3,1,4)
        attention = (q @ k.transpose(-1,-2) / math.sqrt(self.head_width) + bias).softmax(-1)
        state = state + self.output((attention @ v).transpose(1,2).reshape(b,s,w))
        # An explicit zero sentinel makes empty or entirely masked public memory safe.
        memory = torch.cat((memory,memory.new_zeros(b,1,w)),1)
        mask = torch.cat((mask,torch.ones(b,1,dtype=torch.bool,device=mask.device)),1)
        read,_ = self.cross(self.cross_norm(state),memory,memory,key_padding_mask=~mask,need_weights=False)
        state = state + read
        return state + self.ff(self.ff_norm(state)), attention


class ThinkingModel(nn.Module):
    def __init__(self, config: ThinkingConfig):
        super().__init__()
        self.config = config
        c,w = config,config.width
        self.features = nn.Linear(c.feature_dim,w)
        self.initial_rows = nn.Parameter(torch.randn(c.workspace_rows,w)*.02)
        self.blocks = nn.ModuleList(_WorkspaceBlock(c) for _ in range(4))
        self.ground_query = nn.Linear(w,max(1,c.structural_heads)*w,bias=False)
        self.ground_key = nn.Linear(w,max(1,c.structural_heads)*w,bias=False)
        self.ground_nodes_q = nn.Linear(w,w,bias=False)
        self.ground_nodes_k = nn.Linear(w,w,bias=False)
        self.memory_refresh = nn.MultiheadAttention(w,c.heads,dropout=0,batch_first=True)
        self.graph_source = nn.Linear(w,c.n_relations*w,bias=False)
        self.graph_destination = nn.Linear(w,c.n_relations*w,bias=False)
        self.ground_null_q = nn.Linear(w,max(1,c.structural_heads))
        self.ground_null_k = nn.Linear(w,max(1,c.structural_heads))
        self.relation_strength = nn.Parameter(torch.ones(c.structural_heads,c.n_relations))
        self.candidate_queries = nn.Parameter(torch.randn(c.candidates,w)*.1)
        self.operation = nn.Linear(w,c.n_ops)
        self.argument_queries = nn.Linear(w,c.max_arguments*w)
        self.argument_keys = nn.Linear(w,w,bias=False)
        self.identity_query = nn.Linear(w,w,bias=False)
        self.identity_key = nn.Linear(w,w,bias=False)
        self.argument_null = nn.Linear(w,c.max_arguments)
        self.readiness = nn.Linear(w,1)
        self.output = nn.Linear(w,c.output_classes)
        self.emit_probe = nn.Linear(w,1)
        self.token_clock = nn.Linear(2,w,bias=False)
        self.event_value = nn.Linear(1,w,bias=False)
        self.event_type = nn.Embedding(c.n_types,w)
        self.event_operation = nn.Embedding(c.n_ops,w)
        self.event_arguments = nn.Linear(c.max_arguments*c.feature_dim,w,bias=False)
        self.event_provenance = nn.Linear(c.feature_dim,w,bias=False)
        self.event_encoder = nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w),nn.GELU())
        self.roundtrip_value = nn.Linear(w,1)
        self.roundtrip_type = nn.Linear(w,c.n_types)
        self.roundtrip_operation = nn.Linear(w,c.n_ops)

    def initialize(self, public):
        context = self.features(public['context'])
        mask = public.get('context_mask',torch.ones(context.shape[:2],device=context.device,dtype=torch.bool))
        pooled = (context*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        return self.initial_rows[None]+pooled[:,None]

    def step(self, workspace, context, current_memory, *, adjacency=None,
             context_mask=None, memory_mask=None, microstep=1, token_time=0,
             structural_strength=None, identity_memory=None, identity_mask=None,
             graph_permutation=None, graph_keep=None):
        """Advance one shared four-block microstep from current observations only.

        adjacency must be observable; omitted adjacency is predicted from current
        memory. Hidden graph targets belong in auxiliary losses outside this API.
        Token time identifies emitted-token position, never latent microstep time.
        """
        c = self.config
        ctx,mem = self.features(context),self.features(current_memory)
        b,m,w = mem.shape
        if context_mask is None:
            context_mask = torch.ones(ctx.shape[:2],device=ctx.device,dtype=torch.bool)
        if memory_mask is None:
            memory_mask = torch.ones(mem.shape[:2],device=mem.device,dtype=torch.bool)
        context_mask,memory_mask = context_mask.bool(),memory_mask.bool()
        # Memory identities acquire public textual/current-workspace context before
        # predicting edges; no historical state or hidden target enters this read.
        source = torch.cat((ctx,workspace),1)
        source_mask = torch.cat((context_mask,torch.ones(workspace.shape[:2],device=mem.device,dtype=torch.bool)),1)
        refresh,_ = self.memory_refresh(mem,source,source,key_padding_mask=~source_mask,need_weights=False)
        mem = mem + refresh
        gs = self.graph_source(mem).reshape(b,m,c.n_relations,w).transpose(1,2)
        gd = self.graph_destination(mem).reshape(b,m,c.n_relations,w).transpose(1,2)
        predicted = (gs @ gd.transpose(-1,-2)/math.sqrt(w)).sigmoid()
        graph = predicted if adjacency is None else adjacency.to(mem.dtype)
        if graph.ndim == 3:
            # Untyped public graph applies equally to each relation channel;
            # mean aggregation below preserves its scale at unit strengths.
            graph = graph[:,None].expand(-1,c.n_relations,-1,-1)
        if graph.shape != (b,c.n_relations,m,m):
            raise ValueError('adjacency must match relation count and current memory')
        graph = graph * memory_mask[:,None,:,None] * memory_mask[:,None,None,:]
        # Interventions affect only geometry actually used for attention. The
        # unmodified prediction remains available for losses and diagnostics.
        if graph_permutation is not None:
            permutation = graph_permutation
            if permutation.shape != (b,m) or permutation.dtype not in (torch.int32,torch.int64):
                raise ValueError('graph_permutation must be integer [batch,nodes]')
            permutation = permutation.to(device=mem.device,dtype=torch.long)
            expected = torch.arange(m,device=mem.device).expand(b,-1)
            if not torch.equal(permutation.sort(-1).values,expected):
                raise ValueError('graph_permutation must contain each node exactly once')
            if not torch.equal(memory_mask.gather(1,permutation),memory_mask):
                raise ValueError('graph_permutation must preserve padding support')
            graph = graph.gather(2,permutation[:,None,:,None].expand(b,c.n_relations,m,m))
            graph = graph.gather(3,permutation[:,None,None,:].expand(b,c.n_relations,m,m))
        if graph_keep is not None:
            keep = graph_keep.to(device=mem.device,dtype=mem.dtype)
            if keep.shape == (b,m,m):
                keep = keep[:,None].expand(-1,c.n_relations,-1,-1)
            if keep.shape != (b,c.n_relations,m,m):
                raise ValueError('graph_keep must match batch, relations and nodes')
            if not torch.isfinite(keep).all() or (keep < 0).any() or (keep > 1).any():
                raise ValueError('graph_keep must contain finite values in [0,1]')
            graph = graph*keep
        strength = c.structural_strength if structural_strength is None else structural_strength
        clock = workspace.new_tensor([math.sin(float(token_time)),math.cos(float(token_time))])
        state = workspace + self.token_clock(clock)[None,None]
        attentions = []
        pq = pk = state.new_zeros(b,c.structural_heads,state.shape[1],m+1)
        for block in self.blocks:
            bias = state.new_zeros(b,c.heads,state.shape[1],state.shape[1])
            if c.structural_heads:
                q = F.normalize(self.ground_query(state).reshape(b,-1,c.structural_heads,w).transpose(1,2),dim=-1)
                k = F.normalize(self.ground_key(state).reshape(b,-1,c.structural_heads,w).transpose(1,2),dim=-1)
                nq,nk = F.normalize(self.ground_nodes_q(mem),dim=-1),F.normalize(self.ground_nodes_k(mem),dim=-1)
                q_logits = torch.cat((q @ nq[:,None].transpose(-1,-2)/c.temperature,
                                      self.ground_null_q(state).transpose(1,2)[...,None]),-1)
                k_logits = torch.cat((k @ nk[:,None].transpose(-1,-2)/c.temperature,
                                      self.ground_null_k(state).transpose(1,2)[...,None]),-1)
                node_mask = torch.cat((memory_mask,torch.ones(b,1,device=mem.device,dtype=torch.bool)),-1)
                pq = _masked_softmax(q_logits,node_mask[:,None,None,:])
                pk = _masked_softmax(k_logits,node_mask[:,None,None,:])
                # Null is a grounding destination, never a structural graph node.
                per_relation = (pq[...,:-1][:,:,None] @ graph[:,None]
                                @ pk[...,:-1][:,:,None].transpose(-1,-2))
                induced = (per_relation*self.relation_strength[None,:,:,None,None]).mean(2)
                bias = torch.cat((bias[:,:c.heads-c.structural_heads], strength*induced),1)
            state,attention = block(state,torch.cat((ctx,mem),1),torch.cat((context_mask,memory_mask),1),bias)
            attentions.append(attention)
        routes = (self.candidate_queries[None] @ state.transpose(-1,-2)/math.sqrt(w)).softmax(-1)
        candidates = routes @ state
        queries = F.normalize(self.argument_queries(candidates).reshape(b,c.candidates,c.max_arguments,w),dim=-1)
        keys = F.normalize(self.argument_keys(mem),dim=-1)
        bindings = torch.einsum('bkaw,bmw->bkam',queries,keys)/c.temperature
        bindings = bindings.masked_fill(~memory_mask[:,None,None,:],-torch.inf)
        bindings = torch.cat((bindings,self.argument_null(candidates)[...,None]),-1)
        identities = mem if identity_memory is None else self.features(identity_memory)
        if identity_mask is None:
            identity_mask = memory_mask if identity_memory is None else torch.ones(identities.shape[:2],device=mem.device,dtype=torch.bool)
        identity_logits = (F.normalize(self.identity_query(candidates),dim=-1) @ F.normalize(self.identity_key(identities),dim=-1).transpose(-1,-2))/c.temperature
        identity_logits = identity_logits.masked_fill(~identity_mask[:,None,:].bool(),-torch.inf)
        pooled = state.mean(1)
        emit_logits = self.emit_probe(pooled).squeeze(-1)
        # Soft stopping priors are independent of the hard safety budget, so
        # extending evaluation depth cannot change emission at the same state/t.
        t = emit_logits.new_tensor(float(microstep))
        bias = (-c.emit_alpha*F.softplus(c.soft_min_microsteps-t)
                +c.emit_beta*F.softplus(t-c.soft_max_microsteps))
        probability = (emit_logits + bias).sigmoid()
        if microstep < c.min_microsteps:
            probability = torch.zeros_like(probability)
        if microstep >= c.max_microsteps:
            probability = torch.ones_like(probability)
        return dict(workspace=state,op_logits=self.operation(candidates),binding_logits=bindings,
                    readiness_logits=self.readiness(candidates).squeeze(-1),routes=routes,
                    candidate_id_logits=identity_logits,
                    overlap=routes @ routes.transpose(-1,-2),output_logits=self.output(pooled),
                    emit_logits=emit_logits,emit_probability=probability,emit=probability.ge(.5),
                    ponder=c.ponder_cost*(1-probability),grounding_q=pq,grounding_k=pk,
                    predicted_adjacency=predicted,used_adjacency=graph,attentions=tuple(attentions))

    def encode_events(self, events):
        """Encode typed events; ordered arguments are public feature rows.

        Types use integer=0,float=1,boolean=2. Operation IDs are caller's fixed
        primitive vocabulary. Provenance is an optional public feature vector.
        Values are scalar numbers (boolean coerces to 0/1), with no class oracle.
        """
        values = events['values'].to(self.event_value.weight.dtype)
        encoded = (self.event_value(values[...,None]) + self.event_type(events['types'].long())
                   + self.event_operation(events['operations'].long())
                   + self.event_arguments(events['arguments'].flatten(-2)))
        if 'provenance' in events:
            encoded = encoded + self.event_provenance(events['provenance'])
        return self.event_encoder(encoded)

    def event_roundtrip(self, encoded):
        return dict(value=self.roundtrip_value(encoded).squeeze(-1),
                    semantic_logits=self.roundtrip_type(encoded),op_logits=self.roundtrip_operation(encoded))

    def inject_events(self, workspace, encoded, routes, event_mask=None):
        """Add event messages without erasing residual or unrelated components.

        Driver must perform >=1 subsequent step before it can emit an answer.
        """
        if event_mask is not None:
            encoded = encoded * event_mask[...,None].to(encoded.dtype)
        return workspace + routes.transpose(-1,-2) @ encoded
