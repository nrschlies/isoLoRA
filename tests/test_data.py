# =============================================================================
# Smoke‑tests for the data loaders (fast!)
# =============================================================================
import torch

from iso_lora.data.vision import get_cifar10_loaders, VisionBatch
from iso_lora.data.nlp import get_imdb_loaders, NLPBatch


def test_cifar10_one_batch(tmp_path):
    train_loader, _, shp, n_cls = get_cifar10_loaders(
        data_dir=tmp_path / "cifar10", batch_size=4, num_workers=0, pin_memory=False
    )
    batch = next(iter(train_loader))
    vb = VisionBatch(*batch)
    assert vb.images.shape == (4, *shp)
    assert vb.labels.shape == (4,)
    assert vb.labels.max() < n_cls


def test_imdb_one_batch(tmp_path):
    train_loader, _, vocab = get_imdb_loaders(
        data_dir=tmp_path / "imdb", batch_size=2, num_workers=0, pin_memory=False
    )
    batch = next(iter(train_loader))
    nb = NLPBatch(**batch)
    assert nb.input_ids.shape[0] == nb.attention_mask.shape[0] == nb.labels.shape[0] == 2
    assert nb.input_ids.max() < vocab
    # ensure tensors
    assert isinstance(nb.input_ids, torch.Tensor)