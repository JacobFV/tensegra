import torch

from .data import dynamics_step


@torch.no_grad()
def one_step(model, x, y, *, graph=None, mode="none", strength=0.0):
    prediction = model(x, graph, mode=mode, strength=strength) if graph is not None else model(x)
    return torch.mean((prediction - y) ** 2).item()


@torch.no_grad()
def rollout(model, series, history, horizon, mean, std, *, graph=None, mode="none", strength=0.0):
    errors = []
    observed = series[:, :history].clone()
    stop = min(series.shape[1], history + horizon)
    for target in range(history, stop):
        x = ((observed[:, -history:] - mean) / std).transpose(1, 2)
        pred = model(x, graph, mode=mode, strength=strength) if graph is not None else model(x)
        value = pred * std + mean
        observed = torch.cat((observed, value[:, None]), dim=1)
        errors.append(torch.mean((value - series[:, target]) ** 2))
    return torch.stack(errors).mean().item()


@torch.no_grad()
def diagnostics(system, x_original, y_original):
    last = x_original[..., -1]
    oracle = dynamics_step(system, last)
    return {
        "zero": torch.mean(y_original.square()).item(),
        "privileged_oracle": torch.mean((oracle - y_original) ** 2).item(),
    }
