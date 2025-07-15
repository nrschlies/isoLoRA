# =============================================================================
# Metric helpers (accuracy, F1, running mean)
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import f1_score as _sk_f1

__all__ = ["accuracy", "f1", "MetricTracker"]


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


def accuracy(pred, target) -> float:  # noqa: D401
    """Top‑1 accuracy for class‑ID tensors/arrays."""
    p, t = map(_to_numpy, (pred, target))
    return float((p == t).mean())


def f1(pred, target, average: str = "micro") -> float:  # noqa: D401
    """Sklearn F1 wrapper that accepts tensors."""
    p, t = map(_to_numpy, (pred, target))
    return float(_sk_f1(t, p, average=average, zero_division=0))


@dataclass
class MetricTracker:
    """Running mean for any scalar metric."""
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:  # noqa: D401
        return 0.0 if self.count == 0 else self.total / self.count

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0