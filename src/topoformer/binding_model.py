"""Stable identity registers plus free learned content; no gold state inputs.

Pointer writes explicitly apply P A and retain graph use at zero logit strength.
"""
import math
import torch
from torch.nn import functional as F
from .attention import structural_attention
from .grounding import induce_bias
from .runtime_graph import RuntimeGraph
from .traversal_model import TraversalTransformer


class BindingTransformer(TraversalTransformer):
    def __init__(self, *, matcher="cosine", identity_update="attention",
                 identity_only=True, null_init=0.65, heads=4, **kwargs):
        if matcher not in {"dot", "cosine"}:
            raise ValueError("matcher must be dot or cosine")
        if identity_update not in {"mixed", "attention", "pointer"}:
            raise ValueError("identity_update must be mixed, attention or pointer")
        if not math.isfinite(null_init):
            raise ValueError("null_init must be finite")
        super().__init__(heads=heads, **kwargs)
        self.matcher, self.identity_update = matcher, identity_update
        self.identity_only = bool(identity_only)
        with torch.no_grad():
            for g in self.grounders:
                g.query_null_logit.fill_(null_init)
                g.key_null_logit.fill_(null_init)

    def grounding_input(self, state):
        return (F.pad(state[..., :self.key_dim], (0, self.width-self.key_dim))
                if self.identity_only else state)

    def ground(self, state, graph, slot=0, role="query"):
        """Separate role projections; final probability column is unbound."""
        if role not in {"query", "key"}:
            raise ValueError("grounding role must be query or key")
        if state.ndim != 3 or state.shape[0] != graph.node_ids.shape[0] or state.shape[-1] != self.width:
            raise ValueError("latent shape must be [B,T,width] matching graph batch")
        if graph.embeddings.shape[-1] != self.key_dim:
            raise ValueError("entity embedding dimension must match key_dim")
        if not torch.isfinite(state).all():
            raise ValueError("grounding latents must be finite")
        g = self.grounders[slot]
        state = self.grounding_input(state)
        if self.matcher == "dot":
            return getattr(g, role)(state, graph)
        q = getattr(g, role + "_projection")(state)
        e = getattr(g, role + "_entity_projection")(graph.embeddings)
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        scores = F.normalize(q.to(dtype), dim=-1) @ F.normalize(e.to(dtype), dim=-1).transpose(-1, -2)
        null = getattr(g, role + "_null_logit").to(dtype).expand(*scores.shape[:-1], 1)
        scores = torch.cat((scores, null), -1) / g.temperature.to(dtype)
        valid = torch.cat((graph.node_mask, torch.ones(graph.node_ids.shape[0], 1,
                           dtype=torch.bool, device=scores.device)), -1)
        return scores.masked_fill(~valid[:, None], -torch.inf).softmax(-1)

    def forward(self, batch, *, mode="soft", return_diagnostics=False):
        if mode not in self.MODES:
            raise ValueError(f"unknown traversal mode: {mode}")
        entities, adjacency = batch["entity_keys"], batch["adjacency"]
        tokens, values = batch["token_keys"], batch["token_values"]
        instructions = batch["relations"]
        if instructions.ndim != 2 or instructions.shape[1] < 1:
            raise ValueError("relations must have positive path length [B,depth]")
        if adjacency.shape[1] != self.relations:
            raise ValueError("relation count differs from model")
        b, n, _ = entities.shape
        graph = RuntimeGraph(batch["node_ids"], entities, adjacency)
        if self.identity_update == "pointer":
            transitions = adjacency if self.typed else adjacency.amax(dim=1)
            if (transitions < 0).any() or (transitions.sum(-1) > 1 + 1e-6).any():
                raise ValueError("pointer writes require nonnegative substochastic transitions (row sums <= 1)")
        memory = F.pad(torch.cat((tokens, F.one_hot(values, self.classes).to(tokens)), -1),
                       (0, self.width - self.key_dim - self.classes))
        state = F.pad(batch["start_keys"], (0, self.width - self.key_dim)).unsqueeze(1)
        memory_keys = self._heads(self.key(self.norm(memory)))
        frozen = None
        diagnostics = []
        for step in range(instructions.shape[1]):
            slot = step % self.projection_period
            relation = instructions[:, step]
            selected = adjacency[torch.arange(b, device=adjacency.device), relation]
            if not self.typed:
                selected = adjacency.amax(dim=1)
            fast_none = mode == "none" and not return_diagnostics and self.identity_update != "pointer"
            if fast_none:
                pq = pk = None
            elif mode in {"known", "graph_input"}:
                pq, pk = self._known(state, entities), self._known(memory, entities)
            elif mode == "frozen" and frozen is not None:
                pq, pk = frozen
            else:
                pq, pk = self.ground(state, graph, slot), self.ground(memory, graph, slot, "key")
                if mode == "frozen":
                    frozen = (pq, pk)
            if mode == "permuted":
                pq = torch.cat((pq[..., :-1].roll(1, -1), pq[..., -1:]), -1)
            route = None if fast_none else induce_bias(pq, selected[:, None], pk)[:, 0]
            strengths = self.strengths[slot].expand(self.heads, -1)
            strengths = strengths[:, relation if self.typed else torch.zeros_like(relation)].T
            bias = None if fast_none else route[:, None] * strengths[:, :, None, None]
            allowed = None
            current_memory = memory
            if mode == "hard":
                oq = F.one_hot(pq.argmax(-1), n + 1).to(pq)
                ok = F.one_hot(pk.argmax(-1), n + 1).to(pk)
                legal = induce_bias(oq, selected[:, None], ok)[:, 0] > 0
                # No mapped legal destination => ordinary attention, never NaN rows.
                legal = legal | ~legal.any(-1, keepdim=True)
                allowed = legal[:, None]
                bias = None
            elif mode == "graph_input":
                token_edges = induce_bias(pk, selected[:, None], pk)[:, 0]
                current_memory = token_edges @ memory / token_edges.sum(-1, keepdim=True).clamp_min(1e-8)
                bias = None
            elif mode == "none":
                bias = None
            content_bias = None
            if self.content_identity_bias:
                similarity = F.normalize(state[..., :self.key_dim], dim=-1) @ F.normalize(tokens, dim=-1).transpose(-1, -2)
                content_bias = self.content_identity_bias * similarity[:, None]
                bias = content_bias if bias is None else bias + content_bias
            state_before = state
            attended, attention = structural_attention(
                self._heads(self.query(self.norm(state + self.instruction(relation)[:, None]))), memory_keys,
                self._heads(current_memory + self.value(current_memory)),
                bias=bias, strength=1.0, allowed=allowed)
            retrieved = attended.transpose(1, 2).reshape(b, 1, self.width)
            proposal = state + self.update_gate.sigmoid() * (retrieved - state)
            mlp_delta = self.mlp(proposal)
            proposal = proposal + mlp_delta
            pnext = None if pq is None else pq[..., :n] @ selected
            identity_write = proposal[..., :self.key_dim]
            if self.identity_update == "attention":
                # Graph-input memory already contains successor messages.
                identity_write = attention.mean(1) @ current_memory[..., :self.key_dim]
            elif self.identity_update == "pointer":
                # No renormalization: null mass writes a zero identity.
                identity_write = pnext @ entities
            state = (proposal if self.identity_update == "mixed" else
                     torch.cat((identity_write, proposal[..., self.key_dim:]), -1))
            if return_diagnostics:
                pq_after = (self._known(state, entities) if mode in {"known", "graph_input"}
                            else self.ground(state, graph, slot))
                if mode == "permuted":
                    pq_after = torch.cat((pq_after[..., :-1].roll(1, -1), pq_after[..., -1:]), -1)
                diagnostics.append({"pq": pq, "pq_after": pq_after, "pk": pk, "attention": attention,
                                    "bias": route, "content_bias": content_bias,
                                    "strengths": strengths, "state": state,
                                    "state_before": state_before, "state_after": state,
                                    "identity_write": identity_write,
                                    "identity_proposal": proposal[..., :self.key_dim],
                                    "mlp_identity_delta": mlp_delta[..., :self.key_dim],
                                    "immutable_identity_retrieval": attention.mean(1) @ current_memory[..., :self.key_dim],
                                    "pnext": pnext,
                                    "projected_query_norm": self.grounders[slot].query_projection(
                                        self.grounding_input(state_before)).norm(dim=-1)})
        logits = self.readout(state[:, 0])
        return (logits, diagnostics) if return_diagnostics else logits

