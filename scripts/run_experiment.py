#!/usr/bin/env python
"""
IsoLoRA two‑stage CIFAR‑10 demo with profiling.
"""
from __future__ import annotations

import sys
from contextlib import ExitStack
from typing import List

import torch
from torch import nn, optim
from torchvision import models as tvm

from iso_lora.runtime.edge_device import edge_monitor
from iso_lora.runtime.profiling import gpu_profiler
from iso_lora.train.engine import Trainer
from iso_lora.data.vision import get_cifar10_loaders
from iso_lora.models.base import build_large
from iso_lora.models.expansion import expand_linear
from iso_lora.models.lora import LoRALinear, freeze_except_lora

# ---------------- Hydra‑style CLI overrides ---------------------------
sys.argv += [
    "name=cifar10_iso",
    "epochs=10",
    "batch_size=128",
    "subset_size=1024",
]

# ---------------- Profiling contexts ----------------------------------
edge_ctx = edge_monitor(tag="cifar10_iso")
gpu_ctx = gpu_profiler(tag="cifar10_iso")


# ---------------- Helper utilities ------------------------------------
def loraize(model: nn.Module, rank: int) -> None:
    """Replace every nn.Linear with a LoRALinear adapter wrapper."""
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parent, child_name = _parent_child(model, name)
            setattr(parent, child_name, LoRALinear.from_linear(module, rank=rank))


def _parent_child(root: nn.Module, dotted: str):
    parts = dotted.split(".")
    for p in parts[:-1]:
        root = getattr(root, p)
    return root, parts[-1]


# ---------------- IsoLoRA pipeline ------------------------------------
def run(device: str = "cpu", rank: int = 8) -> None:
    # 0. Seed: ImageNet MobileNet‑V2 -> swap head to 10 classes
    small = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
    small.classifier[1] = nn.Linear(small.last_channel, 10)

    # 1. Expand into MobileNet‑V3‑Large
    large = build_large("vision", num_classes=10)
    for (n_s, m_s), (n_l, m_l) in zip(small.named_modules(), large.named_modules()):
        if isinstance(m_s, nn.Linear) and isinstance(m_l, nn.Linear):
            big, _, _ = expand_linear(m_s, factor=2)
            m_l.weight.data.copy_(big.weight)
            if m_l.bias is not None:
                m_l.bias.data.copy_(big.bias)

    # 2. Attach LoRA adapters
    loraize(large, rank=rank)
    freeze_except_lora(large)

    # Data loaders
    train_dl, val_dl, _, _ = get_cifar10_loaders(
        batch_size=128,
        subset_size=1024,
        num_workers=0,
    )

    # Optimiser & Trainer
    opt = optim.Adam(filter(lambda p: p.requires_grad, large.parameters()), lr=5e-4)
    trainer = Trainer(
        model=large,
        train_loader=train_dl,
        val_loader=val_dl,
        optimizer=opt,
        scheduler=None,
        max_epochs=10,
        grad_clip=1.0,
        device=device,
        log_dir="outputs/cifar10_iso",
    )
    trainer.fit()


# ---------------- Main guard (multiprocessing safe) -------------------
if __name__ == "__main__":
    with ExitStack() as stack:
        stack.enter_context(edge_ctx)
        stack.enter_context(gpu_ctx)

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        run(device=dev)
