import math

import torch


def _validate_attention_inputs(q, k, v):
    tensors = {"q": q, "k": k, "v": v}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape [B, H, sequence, features]")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")

    if not (q.device == k.device == v.device):
        raise ValueError("q, k, and v must use the same device")
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        raise ValueError("q, k, and v must have matching batch and head dimensions")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have matching feature dimensions")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have matching sequence lengths")
    if q.shape[-1] == 0 or k.shape[-2] == 0:
        raise ValueError("attention dimensions must be nonempty")


def _broadcast_relation(value, target_shape, *, name, device, dtype=None):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.device != device:
        raise ValueError(f"{name} must use the same device as q, k, and v")
    try:
        broadcast_shape = torch.broadcast_shapes(value.shape, target_shape)
    except RuntimeError as error:
        raise ValueError(f"{name} must broadcast to attention scores") from error
    if broadcast_shape != target_shape:
        raise ValueError(f"{name} must broadcast to attention scores")
    if dtype is not None:
        value = value.to(dtype=dtype)
    return torch.broadcast_to(value, target_shape)


def structural_attention(q, k, v, *, bias=None, strength=0.0, allowed=None):
    """Compute scaled dot-product attention with optional structural constraints."""
    _validate_attention_inputs(q, k, v)
    score_shape = (*q.shape[:-2], q.shape[-2], k.shape[-2])
    score_dtype = (
        torch.float64
        if torch.float64 in {q.dtype, k.dtype, v.dtype}
        else torch.float32
    )

    if isinstance(strength, torch.Tensor):
        if strength.numel() != 1:
            raise ValueError("strength must be a finite scalar")
        if strength.device != q.device:
            raise ValueError("strength must use the same device as q, k, and v")
        strength_value = strength.to(dtype=score_dtype)
        strength_is_finite = torch.isfinite(strength_value).item()
    else:
        try:
            strength_value = torch.as_tensor(strength, dtype=score_dtype, device=q.device)
        except (TypeError, ValueError) as error:
            raise TypeError("strength must be a finite scalar") from error
        strength_is_finite = strength_value.numel() == 1 and torch.isfinite(strength_value).item()
    if not strength_is_finite:
        raise ValueError("strength must be a finite scalar")
    strength_value = strength_value.reshape(())

    scores = q.to(score_dtype) @ k.to(score_dtype).transpose(-2, -1)
    scores = scores / math.sqrt(q.shape[-1])

    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise TypeError("bias must be a tensor")
        if not bias.is_floating_point():
            raise TypeError("bias must be floating point")
        if not torch.isfinite(bias).all():
            raise ValueError("bias must contain only finite values")
        broadcast_bias = _broadcast_relation(
            bias, score_shape, name="bias", device=q.device, dtype=score_dtype
        )
        scores = scores + strength_value * broadcast_bias

    if allowed is not None:
        if not isinstance(allowed, torch.Tensor) or allowed.dtype != torch.bool:
            raise TypeError("allowed must be a boolean tensor")
        broadcast_allowed = _broadcast_relation(
            allowed, score_shape, name="allowed", device=q.device
        )
        if not broadcast_allowed.any(dim=-1).all():
            raise ValueError("allowed contains a row with no legal key")
        scores = scores.masked_fill(~broadcast_allowed, -torch.inf)

    weights = torch.softmax(scores, dim=-1)
    output = (weights @ v.to(dtype=score_dtype)).to(dtype=v.dtype)
    return output, weights


def graph_structure(read_graph, mode):
    """Convert a directed adjacency matrix into an attention relation."""
    if not isinstance(read_graph, torch.Tensor) or read_graph.dtype != torch.bool:
        raise TypeError("read_graph must be a boolean tensor")
    if read_graph.ndim != 3 or read_graph.shape[-2] != read_graph.shape[-1]:
        raise ValueError("read_graph must have shape [B, N, N]")
    if mode not in {"none", "soft", "hard"}:
        raise ValueError("mode must be one of: none, soft, hard")

    graph = read_graph[:, None]
    zero_bias = torch.zeros(graph.shape, dtype=torch.float32, device=read_graph.device)
    if mode == "none":
        return zero_bias, None
    if mode == "soft":
        return graph.to(torch.float32), None

    identity = torch.eye(read_graph.shape[-1], dtype=torch.bool, device=read_graph.device)
    return zero_bias, graph | identity[None, None]
