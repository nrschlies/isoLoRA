from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch
from torch import optim

from iso_lora.data.vision import get_cifar10_loaders
from iso_lora.models.base import build_small
from iso_lora.train.engine import Trainer

# --------------------------- Config dataclass --------------------------
@dataclass
class TrainConfig:
    # basic
    name: str = "run"
    epochs: int = 1
    device: str = "cpu"
    # optimiser
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    # dataset / model
    domain: str = "vision"
    dataset: str = "cifar10"
    batch_size: int = 32
    num_classes: int = 10
    subset_size: int | None = None      # NEW: pass through to data loader
    lora_rank: int = 4
    log_dir: str = "outputs/debug"


ConfigStore.instance().store(name="train_schema", node=TrainConfig)

# --------------------------- Hydra main --------------------------------
@hydra.main(version_base=None, config_path=None, config_name="train_schema")
def main(cfg: TrainConfig):  # type: ignore
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Data (vision only for now)
    train_dl, val_dl, _, _ = get_cifar10_loaders(
        batch_size=cfg.batch_size,
        subset_size=cfg.subset_size,
    )

    # Model (small MobileNet‑V2 for debug)
    model = build_small("vision", num_classes=cfg.num_classes)

    # Optimiser / scheduler – train *all* params for this quick run
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    trainer = Trainer(
        model,
        train_dl,
        val_dl,
        optimizer=opt,
        scheduler=sch,
        max_epochs=cfg.epochs,
        grad_clip=cfg.grad_clip,
        device=cfg.device,
        log_dir=Path(cfg.log_dir),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
