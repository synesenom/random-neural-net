"""Dataset loading: MNIST with reproducible fixed-size subsets for the
sample-efficiency experiments. Every subset is drawn with an explicit seed so
runs are exactly reproducible."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


class FlattenedMNIST(Dataset):
    """MNIST wrapped to return (784,)-flattened float pixels in [0, 1] and int label."""

    def __init__(self, train: bool, root: Path = DATA_ROOT):
        tfm = transforms.Compose([transforms.ToTensor()])
        self.base = datasets.MNIST(root=str(root), train=train, download=True, transform=tfm)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base[idx]
        return img.view(-1), label


def make_subset(dataset: Dataset, n_samples: int | None, seed: int) -> Dataset:
    if n_samples is None or n_samples >= len(dataset):
        return dataset
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=gen)[:n_samples]
    return Subset(dataset, perm.tolist())


def get_mnist_loaders(
    n_train: int | None,
    n_test: int | None,
    batch_size: int,
    seed: int,
    root: Path = DATA_ROOT,
) -> tuple[DataLoader, DataLoader]:
    train_ds = FlattenedMNIST(train=True, root=root)
    test_ds = FlattenedMNIST(train=False, root=root)
    train_sub = make_subset(train_ds, n_train, seed)
    test_sub = make_subset(test_ds, n_test, seed + 1)
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_sub, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
