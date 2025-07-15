# =============================================================================
# Low‑Rank Adaptation (LoRA) building blocks
# =============================================================================
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["LoRALinear", "freeze_except_lora"]


class LoRALinear(nn.Module):
    """
    **Additive** LoRA wrapper for `nn.Linear`.

    Forward:  y = x W_frozenᵀ + α / r · x Aᵀ Bᵀ + b
              (A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r})

    Parameters
    ----------
    base : nn.Linear
        The frozen *pre‑trained* weight tensor **W** and optional bias **b**.
    rank : int
        Low‑rank dimension **r** (LoRA paper uses 4‑8 for BERT‑base).
    alpha : int | None
        Scaling factor **α**.  If `None`, defaults to `rank` → scale = 1.
    """
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: int | None = None):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.scale = (alpha or rank) / rank

        # --- freeze base weight & bias ---------------------------------
        self.weight = base.weight  # share storage
        self.weight.requires_grad = False
        if base.bias is not None:
            self.bias = base.bias
            self.bias.requires_grad = False
        else:
            self.bias = None

        # --- trainable low‑rank factors --------------------------------
        # A: [r, d_in]   (initialised to zeros so initial function ≡ base)
        self.A = nn.Parameter(torch.zeros(rank, self.in_features))
        # B: [d_out, r]
        self.B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Kaiming uniform for B (same as nn.Linear default), zeros for A
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """
        Forward pass with additive LoRA term.

        Shapes
        -------
        x         : [B, d_in]
        weight    : [d_out, d_in]   (frozen)
        bias      : [d_out] or None
        A         : [r,     d_in]   (trainable)
        B         : [d_out, r]      (trainable)
        """
        # Base frozen projection
        base_out = F.linear(x, self.weight, self.bias)          # [B, d_out]

        # LoRA path — first reduce dim (xAᵀ) → [B, r], then expand (•Bᵀ)
        x_a = F.linear(x, self.A)                               # note: NO .t()
        lora_out = F.linear(x_a, self.B)                        # NO .t()

        return base_out + self.scale * lora_out

    # ---------------------------------------------------------------------
    @classmethod
    def from_linear(cls, layer: nn.Linear, **kw) -> "LoRALinear":
        """Wrap an existing `nn.Linear` in place and return the adapter."""
        adapter = cls(layer, **kw)
        return adapter


# ------------------------------------------------------------------------- #
# Helper to freeze everything except LoRA parameters
# ------------------------------------------------------------------------- #
def freeze_except_lora(model: nn.Module) -> None:
    """
    Set `requires_grad = False` for **all** parameters except those that
    belong to `LoRALinear` modules (their `.A` and `.B`).

    Call this once after replacing Linear layers with LoRA adapters.
    """
    for name, p in model.named_parameters():
        # LoRA trainable params end with '.A' or '.B'
        if any(name.endswith(suffix) for suffix in (".A", ".B")):
            p.requires_grad = True
        else:
            p.requires_grad = False