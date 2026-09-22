import time

import torch


def train_model(model, x, y, validation, *, graph=None, mode="none", strength=0.0,
                steps=300, batch_size=32, learning_rate=1e-3,
                validation_interval=50, batch_indices=None, progress=None,
                deadline=None):
    """Train with a caller-supplied paired batch schedule."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if batch_indices is None:
        batch_indices = torch.randint(len(x), (steps, batch_size), device=x.device)
    started, curve = time.monotonic(), []
    for step, indices in enumerate(batch_indices, 1):
        if step > 1 and deadline is not None and time.monotonic() >= deadline:
            break
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x[indices], graph, mode=mode, strength=strength) if graph is not None else model(x[indices])
        loss = torch.nn.functional.mse_loss(prediction, y[indices])
        loss.backward()
        optimizer.step()
        if step % validation_interval == 0 or step == steps or (deadline is not None and time.monotonic() >= deadline):
            with torch.no_grad():
                vx, vy = validation
                vp = model(vx, graph, mode=mode, strength=strength) if graph is not None else model(vx)
                validation_loss = torch.nn.functional.mse_loss(vp, vy).item()
            point = {"step": step, "train_loss": loss.item(), "validation_loss": validation_loss,
                     "elapsed_seconds": time.monotonic() - started}
            curve.append(point)
            if progress: progress(point)
    return curve
