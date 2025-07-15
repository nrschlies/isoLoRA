from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

__all__ = ["VisionBatch", "get_cifar10_loaders"]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class VisionBatch:
    images: torch.Tensor  # [B, 3, 32, 32]
    labels: torch.Tensor  # [B]


def _tfms(train: bool) -> transforms.Compose:
    aug = (
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if train
        else []
    )
    aug += [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return transforms.Compose(aug)


def get_cifar10_loaders(
    data_dir: str | Path = "data/cifar10",
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool | None = None,      # auto‑detect if None
    subset_size: int | None = None,      # NEW: limit dataset for fast debug
) -> Tuple[DataLoader, DataLoader, Tuple[int, ...], int]:
    """
    Return (train_loader, val_loader, input_shape, num_classes).

    * If `subset_size` is given, each split is truncated to that many samples,
      yielding very fast CPU runs for CI / debugging.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    data_dir = Path(data_dir)
    train_set = datasets.CIFAR10(
        root=data_dir, train=True, transform=_tfms(True), download=True
    )
    val_set = datasets.CIFAR10(
        root=data_dir, train=False, transform=_tfms(False), download=True
    )

    # ---------------------- optional down‑sampling ----------------------
    if subset_size is not None:
        train_set = Subset(train_set, list(range(subset_size)))
        val_set = Subset(val_set, list(range(subset_size)))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    input_shape = (3, 32, 32)
    num_classes = 10
    return train_loader, val_loader, input_shape, num_classes
