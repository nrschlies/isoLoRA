# =============================================================================
# Quick top‑eigenvalue estimation via power iteration
# =============================================================================
from __future__ import annotations

import torch
from torch import nn
from torch.autograd import grad

__all__ = ["hessian_top_eig"]


def _hvp(loss: torch.Tensor, params, vector):
    grad1 = grad(loss, params, create_graph=True, retain_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grad1])
    hvp = grad(flat_grad, params, grad_outputs=vector, retain_graph=True)
    return torch.cat([g.reshape(-1) for g in hvp])


def hessian_top_eig(
    model: nn.Module,
    loss_fn,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    max_iter: int = 20,
    tol: float = 1e-3,
    device: str | torch.device = "cpu",
):
    """Return (λ_max, v) for the Hessian of `loss_fn(model(x), y)`."""
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in params)

    v = torch.randn(total, device=device)
    v = v / v.norm()
    prev = None

    for _ in range(max_iter):
        model.zero_grad(set_to_none=True)
        out = model(inputs.to(device))
        loss = loss_fn(out, targets.to(device))

        hvp = _hvp(loss, params, v)
        lam = torch.dot(v, hvp).item()
        v = hvp / hvp.norm()

        if prev is not None and abs(lam - prev) < tol:
            break
        prev = lam

    return lam, v.detach()