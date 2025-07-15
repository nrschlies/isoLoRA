# =============================================================================
# Function‑preserving block‑diagonal expansion helpers  (updated impl)
# =============================================================================
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

__all__ = ["block_diagonal_expand", "expand_linear"]


def block_diagonal_expand(w: torch.Tensor, factor: int) -> torch.Tensor:
    """
    True block‑diagonal replication of *w*:

        W_big = diag([w, w, … , w])  (factor times)

    Parameters
    ----------
    w       : (out_small, in_small) – weight matrix
    factor  : int ≥ 1                – replication count

    Returns
    -------
    (out_small × factor, in_small × factor) tensor, zeros elsewhere.
    """
    if w.dim() != 2:
        raise ValueError("w must be 2‑D")

    out_s, in_s = w.shape
    out_b, in_b = out_s * factor, in_s * factor
    w_big = w.new_zeros(out_b, in_b)

    for k in range(factor):
        r0, r1 = k * out_s, (k + 1) * out_s
        c0, c1 = k * in_s, (k + 1) * in_s
        w_big[r0:r1, c0:c1] = w
    return w_big


def expand_linear(
    layer: nn.Linear,
    factor: int,
) -> Tuple[nn.Linear, torch.Tensor, torch.Tensor]:
    """
    Build a *larger* nn.Linear with block‑diagonal‐expanded weights so that
    zero‑padded inputs yield **identical outputs** for the first block.

    Returns
    -------
    big_layer : nn.Linear            – expanded in/out features
    mask_in   : Bool[ in_big ]       – True for the *small* input dims
    mask_out  : Bool[ out_big ]      – True for the *small* output dims
    """
    if factor < 1:
        raise ValueError("factor must be ≥ 1")

    w_small = layer.weight.data.clone()
    b_small = layer.bias.data.clone() if layer.bias is not None else None

    w_big = block_diagonal_expand(w_small, factor)
    in_big, out_big = w_big.shape[1], w_big.shape[0]

    big = nn.Linear(in_big, out_big, bias=b_small is not None)
    big.weight.data.copy_(w_big)
    if b_small is not None:
        big.bias.data = b_small.repeat(factor)

    # masks for convenience – first block
    mask_in = torch.zeros(in_big, dtype=torch.bool)
    mask_in[: w_small.shape[1]] = True

    mask_out = torch.zeros(out_big, dtype=torch.bool)
    mask_out[: w_small.shape[0]] = True

    return big, mask_in, mask_out