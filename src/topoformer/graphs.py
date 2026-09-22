import warnings

import torch


def corrupt_graph(graph: torch.Tensor, *, mode: str, seed: int) -> torch.Tensor:
    """Return a seeded boolean graph reversal, relabeling, or edge deletion."""
    if not isinstance(graph, torch.Tensor) or graph.dtype != torch.bool:
        raise TypeError("graph must be a boolean tensor")
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("graph must have shape [N, N]")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    if mode == "reversed":
        if torch.equal(graph, graph.T):
            warnings.warn(
                "reversing this symmetric graph is a redundant control",
                UserWarning,
                stacklevel=2,
            )
        return graph.T.clone()
    if mode == "permuted":
        permutation = torch.randperm(graph.shape[0], generator=generator)
        return graph[permutation][:, permutation].clone()
    if mode == "dropped":
        keep = torch.rand(graph.shape, generator=generator) >= 0.5
        keep.fill_diagonal_(True)
        return graph & keep
    raise ValueError("mode must be one of: reversed, permuted, dropped")
