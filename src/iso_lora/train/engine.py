# =============================================================================
# Train / eval engine with tqdm progress bars + optional runtime profilers
# =============================================================================
from __future__ import annotations

import time
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, List

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from iso_lora.utils.logging import get_logger
from iso_lora.utils.metrics import MetricTracker, accuracy


class Trainer:
    """
    Lightweight trainer for debug runs and small‑to‑medium experiments.

    Parameters
    ----------
    model : nn.Module
    train_loader / val_loader : DataLoader
    optimizer, scheduler : torch.optim objects
    grad_clip : float | None
        Max-norm gradient clipping (None = disabled).
    runtime_ctxs : list[context manager]
        Any context managers you want wrapped around the entire `fit()` call,
        e.g. `edge_monitor(logger)` or `gpu_profiler(logger)`.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        *,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler | None = None,
        max_epochs: int = 1,
        grad_clip: float | None = None,
        device: str | torch.device = "cpu",
        log_dir: Path | str = "outputs/debug",
        runtime_ctxs: List = None,                   # NEW
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.device = device
        self.runtime_ctxs = runtime_ctxs or []      # NEW

        self.logger, self.tb = get_logger("iso_lora.train", log_dir)

    # ------------------------------------------------------------------ #
    def _run_loader(
        self,
        loader: DataLoader,
        train: bool,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        epoch: int,
    ) -> float:
        mode = "train" if train else "val"
        self.model.train(mode == "train")

        loss_meter, acc_meter = MetricTracker(), MetricTracker()
        bar = tqdm(loader, leave=False, desc=f"{mode.capitalize()} e{epoch:03d}")

        for x, y in bar:
            x, y = x.to(self.device), y.to(self.device)

            with torch.set_grad_enabled(train):
                logits = self.model(x)
                loss = loss_fn(logits, y)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            # metrics
            loss_val = loss.item()
            acc_val = accuracy(logits.argmax(dim=1), y)
            loss_meter.update(loss_val, n=len(x))
            acc_meter.update(acc_val, n=len(x))
            bar.set_postfix(loss=f"{loss_val:.3f}", acc=f"{acc_val:.3f}")

        # epoch summary
        self.logger.info(
            f"[{mode}] epoch {epoch:03d} "
            f"loss {loss_meter.avg:.4f} acc {acc_meter.avg:.3f}"
        )
        if self.tb:
            self.tb.add_scalar(f"{mode}/loss", loss_meter.avg, epoch)
            self.tb.add_scalar(f"{mode}/acc", acc_meter.avg, epoch)

        return acc_meter.avg

    # ------------------------------------------------------------------ #
    def fit(self):
        loss_fn = F.cross_entropy
        best_acc, t0 = 0.0, time.time()

        # Optional profilers (edge monitor, GPU profiler, …)
        with ExitStack() as stack:
            for ctx in self.runtime_ctxs:
                stack.enter_context(ctx)

            for epoch in range(1, self.max_epochs + 1):
                self._run_loader(self.train_loader, True, loss_fn, epoch)
                if self.val_loader is not None:
                    acc = self._run_loader(self.val_loader, False, loss_fn, epoch)
                    best_acc = max(best_acc, acc)
                if self.scheduler:
                    self.scheduler.step()

        self.logger.info(
            f"Finished training in {(time.time() - t0)/60:.2f} min | "
            f"best val‑acc {best_acc:.3f}"
        )
        if self.tb:
            self.tb.close()
