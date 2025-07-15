from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch
from torch import optim

from iso_lora.data.vision import get_cifar10_loaders
from iso_lora.models.base import build_small
from iso_lora.train.engine import Trainer

# ---------------------------------------------------------------------
#  Configuration schema
# ---------------------------------------------------------------------
@dataclass
class TrainConfig:
    name: str = "run"
    epochs: int = 1
    device: str = "cpu"
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    domain: str = "vision"
    dataset: str = "cifar10"
    batch_size: int = 32
    num_classes: int = 10
    subset_size: int | None = None      # truncate dataset for debug
    lora_rank: int = 4
    log_dir: str = "outputs/debug"


ConfigStore.instance().store(name="train_schema", node=TrainConfig)

# ---------------------------------------------------------------------
#  Hydra entry point
# ---------------------------------------------------------------------
@hydra.main(version_base=None, config_path=None, config_name="train_schema")
def main(cfg: TrainConfig, runtime_ctxs: List = None):  # type: ignore
    # -------- CUDA fallback -------------------------------------------
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA unavailable — using CPU instead")
        cfg.device = "cpu"

    print(OmegaConf.to_yaml(cfg, resolve=True))

    # -------- Data ----------------------------------------------------
    # Use single‑process loader when debugging small subset on CPU
    n_workers = 0 if (cfg.subset_size is not None and cfg.device == "cpu") else 2
    train_dl, val_dl, _, _ = get_cifar10_loaders(
        batch_size=cfg.batch_size,
        subset_size=cfg.subset_size,
        num_workers=n_workers,
    )

    # -------- Model & optimiser --------------------------------------
    model = build_small("vision", num_classes=cfg.num_classes)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    # -------- Trainer -------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_dl,
        val_loader=val_dl,
        optimizer=opt,
        scheduler=sch,
        max_epochs=cfg.epochs,
        grad_clip=cfg.grad_clip,
        device=cfg.device,
        log_dir=Path(cfg.log_dir),
        runtime_ctxs=runtime_ctxs or [],
    )
    trainer.fit()


if __name__ == "__main__":
    main()  # plain CLI usage
