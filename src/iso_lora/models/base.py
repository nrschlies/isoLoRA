# =============================================================================
# Model factory  – vision & NLP
# =============================================================================
from __future__ import annotations

from typing import Literal, Tuple

import torch.nn as nn
from torchvision import models as tvm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    DistilBertConfig,
    DistilBertForSequenceClassification,
)

__all__ = ["build_small", "build_large", "vis_output_dim", "nlp_output_dim"]

# ---------- Vision -----------------------------------------------------------
def _mobilenet_v2(num_classes: int) -> nn.Module:
    net = tvm.mobilenet_v2(weights=None)
    net.classifier[1] = nn.Linear(net.last_channel, num_classes)
    return net


def _mobilenet_v3_large(num_classes: int) -> nn.Module:
    net = tvm.mobilenet_v3_large(weights=None)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
    return net


def vis_output_dim(model: nn.Module) -> int:
    """Return feature‑dim before the classifier head (needed by IsoLoRA)."""
    if isinstance(model, tvm.MobileNetV2):
        return model.last_channel
    if isinstance(model, tvm.MobileNetV3):
        return model.classifier[3].in_features  # 1280
    raise ValueError("Unsupported vision backbone")


# ---------- NLP --------------------------------------------------------------
def _distilbert_cls(num_classes: int) -> nn.Module:
    cfg = DistilBertConfig.from_pretrained("distilbert-base-uncased", num_labels=num_classes)
    return DistilBertForSequenceClassification(cfg)


def _bert_base_cls(num_classes: int) -> nn.Module:
    cfg = AutoConfig.from_pretrained("bert-base-uncased", num_labels=num_classes)
    return AutoModelForSequenceClassification.from_config(cfg)


def nlp_output_dim(model: nn.Module) -> int:
    """Return hidden size for the [CLS] token."""
    return model.config.hidden_size


# ---------- Public factory ---------------------------------------------------
def build_small(
    domain: Literal["vision", "nlp"],
    num_classes: int,
    **kwargs,
) -> nn.Module:
    """Return the *seed* (small) backbone."""
    if domain == "vision":
        return _mobilenet_v2(num_classes)
    if domain == "nlp":
        return _distilbert_cls(num_classes)
    raise ValueError(f"Unknown domain {domain}")


def build_large(
    domain: Literal["vision", "nlp"],
    num_classes: int,
    **kwargs,
) -> nn.Module:
    """Return the target *large* backbone."""
    if domain == "vision":
        return _mobilenet_v3_large(num_classes)
    if domain == "nlp":
        return _bert_base_cls(num_classes)
    raise ValueError(f"Unknown domain {domain}")