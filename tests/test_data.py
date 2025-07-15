# =============================================================================
# Smoke‑tests for the data loaders (2‑sample batches for speed)
# =============================================================================
import torch

from iso_lora.data.vision import get_cifar10_loaders, VisionBatch
from iso_lora.data.nlp import get_imdb_loaders, NLPBatch


def test_cifar10_one_batch(tmp_path):
    train_loader, _, shp, n_cls = get_cifar10_loaders(
        data_dir=tmp_path / "cifar10",
        batch_size=2,          # smaller batch
        num_workers=0,
        pin_memory=False,
    )
    vb = VisionBatch(*next(iter(train_loader)))
    assert vb.images.shape == (2, *shp)
    assert vb.labels.shape == (2,)
    assert vb.labels.max() < n_cls


def test_imdb_one_batch(tmp_path):
    train_loader, _, vocab = get_imdb_loaders(
        data_dir=tmp_path / "imdb",
        batch_size=2,          # smaller batch
        num_workers=0,
        pin_memory=False,
    )
    nb = NLPBatch(**next(iter(train_loader)))
    assert nb.input_ids.shape[0] == 2
    assert nb.input_ids.max() < vocab
    assert isinstance(nb.input_ids, torch.Tensor)