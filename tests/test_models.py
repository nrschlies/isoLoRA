# =============================================================================
# Smoke tests – factory & expansion (revised)
# =============================================================================
import torch
from torch import nn

from iso_lora.models.base import build_small, build_large
from iso_lora.models.expansion import block_diagonal_expand, expand_linear


# ---------- factory sanity checks -------------------------------------
def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_factories_sizes():
    vis_small = build_small("vision", num_classes=10)
    vis_large = build_large("vision", num_classes=10)
    assert _param_count(vis_small) < _param_count(vis_large)

    nlp_small = build_small("nlp", num_classes=2)
    nlp_large = build_large("nlp", num_classes=2)
    assert _param_count(nlp_small) < _param_count(nlp_large)


# ---------- block‑diagonal logic --------------------------------------
def test_block_diagonal_expansion_preserves_function():
    torch.manual_seed(0)
    small = nn.Linear(4, 3, bias=True)
    factor = 2
    big, mask_in, mask_out = expand_linear(small, factor)

    # small inputs
    x_small = torch.randn(5, 4)

    # zero‑pad into big input space
    x_big = torch.zeros(5, small.in_features * factor)
    x_big[:, mask_in] = x_small

    y_small = small(x_small)
    y_big = big(x_big)

    # select outputs from the first block
    assert torch.allclose(y_small, y_big[:, mask_out], atol=1e-6)

    # round‑trip check for weight builder
    w_big_manual = block_diagonal_expand(small.weight, factor)
    assert torch.allclose(big.weight, w_big_manual)
