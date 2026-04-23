from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

from clwl_experiments_module3_synthetic_clean_data import SyntheticDataset


Array = np.ndarray


@dataclass
class MNISTRealDataConfig:
    root: str = "data"
    val_frac: float = 0.1
    teacher_hidden_dim: int = 256
    teacher_num_epochs: int = 12
    teacher_batch_size: int = 256
    teacher_learning_rate: float = 1e-3
    teacher_weight_decay: float = 1e-4
    device: str = "cpu"
    seed: int = 0
    flatten: bool = True
    download: bool = True
    max_train_samples: Optional[int] = None
    max_test_samples: Optional[int] = None


class MNISTTeacherMLP(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _set_torch_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_raw_mnist(root: str, *, train: bool, download: bool) -> tuple[Array, Array]:
    ds = MNIST(root=root, train=train, download=download)
    X = ds.data.numpy().astype(np.float32) / 255.0
    y = ds.targets.numpy().astype(np.int64)
    return X, y


def _split_train_val(
    X: Array,
    y: Array,
    *,
    val_frac: float,
    seed: int,
) -> tuple[Array, Array, Array, Array]:
    if not (0.0 < val_frac < 0.5):
        raise ValueError(f"val_frac must be in (0, 0.5), got {val_frac}.")

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_val = int(np.floor(val_frac * n))
    if n_val <= 0 or n_val >= n:
        raise ValueError(f"Invalid val size from n={n}, val_frac={val_frac}.")

    idx_val = perm[:n_val]
    idx_train = perm[n_val:]

    return X[idx_train], y[idx_train], X[idx_val], y[idx_val]


def _subsample(
    X: Array,
    y: Array,
    *,
    max_samples: Optional[int],
    seed: int,
) -> tuple[Array, Array]:
    if max_samples is None or max_samples >= X.shape[0]:
        return X, y

    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=int(max_samples), replace=False)
    return X[idx], y[idx]


def _normalize_from_train(
    X_train: Array,
    X_val: Array,
    X_test: Array,
) -> tuple[Array, Array, Array, dict[str, float]]:
    mean = float(X_train.mean())
    std = float(X_train.std())
    if std < 1e-8:
        std = 1.0

    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std
    X_test_n = (X_test - mean) / std

    return X_train_n, X_val_n, X_test_n, {"mean": mean, "std": std}


def _flatten_images(X: Array) -> Array:
    return X.reshape(X.shape[0], -1).astype(np.float32, copy=False)


def _make_loader(
    X: Array,
    y: Array,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _make_feature_only_loader(
    X: Array,
    *,
    batch_size: int,
) -> DataLoader:
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def train_mnist_teacher(
    X_train: Array,
    y_train: Array,
    cfg: MNISTRealDataConfig,
) -> nn.Module:
    if not cfg.flatten:
        raise ValueError("Current formal trainers expect 2D X; use flatten=True for MNIST.")

    _set_torch_seed(cfg.seed)
    device = _resolve_device(cfg.device)

    model = MNISTTeacherMLP(
        input_dim=X_train.shape[1],
        hidden_dim=cfg.teacher_hidden_dim,
        num_classes=10,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.teacher_learning_rate,
        weight_decay=cfg.teacher_weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    loader = _make_loader(X_train, y_train, batch_size=cfg.teacher_batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.teacher_num_epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return model.eval()


def logits_from_model(
    model: nn.Module,
    X: Array,
    cfg: MNISTRealDataConfig,
) -> Array:
    device = _resolve_device(cfg.device)
    loader = _make_feature_only_loader(X, batch_size=cfg.teacher_batch_size)

    outs: list[Array] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb).cpu().numpy().astype(np.float64)
            outs.append(logits)

    if not outs:
        raise RuntimeError("Empty dataset passed to logits_from_model.")
    return np.concatenate(outs, axis=0)


def softmax_numpy(logits: Array) -> Array:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


def build_mnist_real_splits(cfg: MNISTRealDataConfig) -> dict[str, SyntheticDataset]:
    X_train_all, y_train_all = _load_raw_mnist(cfg.root, train=True, download=cfg.download)
    X_test, y_test = _load_raw_mnist(cfg.root, train=False, download=cfg.download)

    X_train, y_train, X_val, y_val = _split_train_val(
        X_train_all,
        y_train_all,
        val_frac=cfg.val_frac,
        seed=cfg.seed,
    )

    X_train, y_train = _subsample(X_train, y_train, max_samples=cfg.max_train_samples, seed=cfg.seed + 1)
    X_val, y_val = _subsample(X_val, y_val, max_samples=cfg.max_train_samples, seed=cfg.seed + 2)
    X_test, y_test = _subsample(X_test, y_test, max_samples=cfg.max_test_samples, seed=cfg.seed + 3)

    X_train, X_val, X_test, norm_stats = _normalize_from_train(X_train, X_val, X_test)

    if cfg.flatten:
        X_train = _flatten_images(X_train)
        X_val = _flatten_images(X_val)
        X_test = _flatten_images(X_test)

    teacher = train_mnist_teacher(X_train, y_train, cfg)

    logits_train = logits_from_model(teacher, X_train, cfg)
    logits_val = logits_from_model(teacher, X_val, cfg)
    logits_test = logits_from_model(teacher, X_test, cfg)

    eta_train = softmax_numpy(logits_train)
    eta_val = softmax_numpy(logits_val)
    eta_test = softmax_numpy(logits_test)

    teacher_name = "mnist_teacher_mlp_ce"

    base_meta = {
        "dataset_name": "mnist",
        "eta_source": "teacher_estimated",
        "oracle_metrics": False,
        "input_kind": "flattened_grayscale",
        "normalization_mean": norm_stats["mean"],
        "normalization_std": norm_stats["std"],
        "teacher_hidden_dim": cfg.teacher_hidden_dim,
        "teacher_num_epochs": cfg.teacher_num_epochs,
        "teacher_batch_size": cfg.teacher_batch_size,
        "teacher_learning_rate": cfg.teacher_learning_rate,
        "teacher_weight_decay": cfg.teacher_weight_decay,
        "seed": cfg.seed,
    }

    return {
        "train": SyntheticDataset(
            X=X_train.astype(np.float64),
            y=y_train.astype(np.intp),
            eta=eta_train.astype(np.float64),
            logits=logits_train.astype(np.float64),
            teacher_name=teacher_name,
            metadata={**base_meta, "subset": "train", "subset_size": int(X_train.shape[0])},
        ),
        "val": SyntheticDataset(
            X=X_val.astype(np.float64),
            y=y_val.astype(np.intp),
            eta=eta_val.astype(np.float64),
            logits=logits_val.astype(np.float64),
            teacher_name=teacher_name,
            metadata={**base_meta, "subset": "val", "subset_size": int(X_val.shape[0])},
        ),
        "test": SyntheticDataset(
            X=X_test.astype(np.float64),
            y=y_test.astype(np.intp),
            eta=eta_test.astype(np.float64),
            logits=logits_test.astype(np.float64),
            teacher_name=teacher_name,
            metadata={**base_meta, "subset": "test", "subset_size": int(X_test.shape[0])},
        ),
    }