# =============================================================================
# PyTest mini‑suite for the utility layer
# =============================================================================
from pathlib import Path

import torch
from torch import nn

from iso_lora.utils.logging import get_logger
from iso_lora.utils.metrics import MetricTracker, accuracy, f1
from iso_lora.utils.hessian import hessian_top_eig


# ---------- logging ----------------------------------------------------
def test_get_logger(tmp_path: Path):
    logger, writer = get_logger("iso_lora.test", log_dir=tmp_path)
    logger.info("hello utils")
    assert any(h for h in logger.handlers if h.level == logger.level)
    if writer:
        writer.add_scalar("dummy/metric", 1.0, 0)
        writer.flush()


# ---------- metrics ----------------------------------------------------
def test_metrics_numpy():
    pred = [0, 1, 2, 2]
    true = [0, 1, 1, 2]
    assert abs(accuracy(pred, true) - 0.75) < 1e-6
    assert abs(f1(pred, true, average="micro") - 0.75) < 1e-6


def test_metric_tracker():
    m = MetricTracker()
    m.update(2.0, n=2)
    m.update(4.0, n=1)
    assert m.avg == 8.0 / 3
    m.reset()
    assert m.count == 0 and m.total == 0


# ---------- hessian ----------------------------------------------------
def test_hessian_top_eig():
    torch.manual_seed(0)
    X = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    model = nn.Linear(4, 3, bias=False)
    loss_fn = nn.CrossEntropyLoss()

    lam, vec = hessian_top_eig(model, loss_fn, X, y, max_iter=10)
    assert lam > 0
    assert abs(vec.norm().item() - 1.0) < 1e-4
