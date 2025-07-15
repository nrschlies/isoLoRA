# =============================================================================
# HuggingFace IMDb helper  (tokenised for BERT‑style models)
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

__all__ = ["NLPBatch", "get_imdb_loaders"]


# --------------------------------------------------------------------- dataclass
@dataclass
class NLPBatch:
    input_ids: torch.Tensor        # [B, L]
    attention_mask: torch.Tensor   # [B, L]
    labels: torch.Tensor           # [B]  (0/1 sentiment)


# ---------------------------------------------------------------- tokenisation
def _tokenise_dataset(ds, tokenizer, max_len: int):
    """Tokenise an HF dataset *in‑place* and keep only model‑ready tensors."""

    def _proc(batch):
        tok = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
        tok["labels"] = batch["label"]          # rename to PyTorch convention
        return tok

    processed = ds.map(
        _proc,
        batched=True,
        remove_columns=ds.column_names,         # drop everything we replaced
    )
    return processed


# ------------------------------------------------------- custom collate (list→Tensor)
def _to_tensor_dict(batch):
    """
    Convert `list[dict[str, list[int]]]` → `dict[str, LongTensor]`

    Works irrespective of sequence length because we already fixed
    padding to `max_len` during tokenisation.
    """
    keys = batch[0].keys()
    collated = {
        k: torch.tensor([sample[k] for sample in batch], dtype=torch.long)
        for k in keys
    }
    return collated


# -------------------------------------------------------------- public loader
def get_imdb_loaders(
    tokenizer_name: str = "distilbert-base-uncased",
    max_len: int = 256,
    batch_size: int = 32,
    data_dir: str | Path = "data/imdb",
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Return (train_loader, val_loader, vocab_size).

    IMDb (~80 MB) is downloaded once and cached under `data_dir`.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    raw = load_dataset("imdb", cache_dir=data_dir)
    train_ds = _tokenise_dataset(raw["train"], tokenizer, max_len)
    test_ds = _tokenise_dataset(raw["test"], tokenizer, max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_to_tensor_dict,
    )
    val_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_to_tensor_dict,
    )
    return train_loader, val_loader, len(tokenizer)
