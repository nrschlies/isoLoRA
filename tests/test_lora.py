# =============================================================================
# Unit tests – LoRA adapter
# =============================================================================
import torch
from torch import nn

from iso_lora.models.lora import LoRALinear, freeze_except_lora


def test_lora_identity_then_shift():
    torch.manual_seed(0)
    lin = nn.Linear(8, 4, bias=True)
    lora = LoRALinear.from_linear(lin, rank=2, alpha=2)

    x = torch.randn(3, 8)

    # ---------------- first: A zeros => identical to base ----------------
    y_base = lin(x)
    y_lora = lora(x)
    assert torch.allclose(y_base, y_lora, atol=1e-6)

    # ---------------- modify A => outputs diverge ------------------------
    nn.init.normal_(lora.A, mean=0.0, std=0.02)
    y_shift = lora(x)
    assert not torch.allclose(y_base, y_shift)

    # scale should be alpha / r = 1 for alpha=2, r=2
    assert lora.scale == 1.0


def test_freeze_except_lora():
    lin = nn.Linear(5, 5, bias=False)
    lora = LoRALinear.from_linear(lin, rank=1)
    model = nn.Sequential(lora)

    freeze_except_lora(model)
    # Only A & B trainable
    for name, p in model.named_parameters():
        if name.endswith((".A", ".B")):
            assert p.requires_grad
        else:
            assert not p.requires_grad
