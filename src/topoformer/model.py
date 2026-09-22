import torch

from .attention import graph_structure, structural_attention


class AttentionBlock(torch.nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = torch.nn.LayerNorm(width)
        self.qkv = torch.nn.Linear(width, 3 * width, bias=False)
        self.out = torch.nn.Linear(width, width, bias=False)
        self.norm2 = torch.nn.LayerNorm(width)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, 2 * width), torch.nn.GELU(),
            torch.nn.Linear(2 * width, width),
        )

    def forward(self, x, bias, allowed, strength):
        batch, nodes, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        def heads(t):
            return t.reshape(batch, nodes, self.heads, width // self.heads).transpose(1, 2)
        attended, _ = structural_attention(
            heads(q), heads(k), heads(v), bias=bias, allowed=allowed, strength=strength
        )
        x = x + self.out(attended.transpose(1, 2).reshape(batch, nodes, width))
        return x + self.mlp(self.norm2(x))


class GraphPredictor(torch.nn.Module):
    """Node-count agnostic history-to-next-state predictor."""

    def __init__(self, history: int, width: int, heads: int, layers: int):
        super().__init__()
        if min(history, width, heads, layers) <= 0 or width % heads:
            raise ValueError("positive sizes and width divisible by heads are required")
        self.history = history
        self.input = torch.nn.Linear(history, width)
        self.blocks = torch.nn.ModuleList(AttentionBlock(width, heads) for _ in range(layers))
        self.final_norm = torch.nn.LayerNorm(width)
        self.readout = torch.nn.Linear(width, 1)

    def forward(self, x, graph, *, mode="none", strength=0.0):
        if x.ndim != 3 or x.shape[-1] != self.history:
            raise ValueError("x must have shape [B, N, history]")
        if graph.ndim == 2:
            graph = graph.unsqueeze(0).expand(x.shape[0], -1, -1)
        if graph.shape != (x.shape[0], x.shape[1], x.shape[1]):
            raise ValueError("graph must have shape [N,N] or [B,N,N]")
        graph = graph.to(device=x.device)
        bias, allowed = graph_structure(graph, mode)
        hidden = self.input(x)
        for block in self.blocks:
            hidden = block(hidden, bias, allowed, strength)
        return self.readout(self.final_norm(hidden)).squeeze(-1)


class TokenMLP(torch.nn.Module):
    def __init__(self, history: int, width: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(history, width), torch.nn.GELU(), torch.nn.Linear(width, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)
