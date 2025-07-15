from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import torch         # NEW
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

__all__ = ["NLPBatch", "get_imdb_loaders"]

@dataclass
class NLPBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

def _tokenise(ds, tok, max_len):
    def f(x):
        out = tok(x["text"], padding="max_length",
                  truncation=True, max_length=max_len)
        out["labels"] = x["label"]; return out
    return ds.map(f, batched=True, remove_columns=["text"])

def get_imdb_loaders(
    tokenizer_name: str = "distilbert-base-uncased",
    max_len: int = 256,
    batch_size: int = 32,
    data_dir: str | Path = "data/imdb",
    num_workers: int = 2,
    pin_memory: bool | None = None,
) -> Tuple[DataLoader, DataLoader, int]:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    tok   = AutoTokenizer.from_pretrained(tokenizer_name)
    raw   = load_dataset("imdb", cache_dir=data_dir)
    tr_ds = _tokenise(raw["train"], tok, max_len)
    ts_ds = _tokenise(raw["test"],  tok, max_len)

    def _collate(b): return {k: torch.tensor(v) for k, v in b.items()}

    tr_dl = DataLoader(tr_ds, batch_size, True,
                       num_workers=num_workers, pin_memory=pin_memory,
                       collate_fn=_collate)
    ts_dl = DataLoader(ts_ds, batch_size, False,
                       num_workers=num_workers, pin_memory=pin_memory,
                       collate_fn=_collate)
    return tr_dl, ts_dl, len(tok)
