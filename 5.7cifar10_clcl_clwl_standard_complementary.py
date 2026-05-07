"""
CIFAR-10 standard complementary-label experiment for CLCL vs CLWL.

This script is for the native/home setting of complementary-label learning:
for each clean class y, the observed weak label z is sampled uniformly from
all incorrect classes z != y. There is no biased-complementary sweep here.

Compared methods:
  clcl_or   : native order-preserving complementary-label loss
  clcl_orw  : practical weighted CLCL-style variant
  clwl      : proposed CLWL loss using T(M) with the same uniform complementary M

The goal is to produce a table analogous to the MNIST complementary-label row:
  Setting | Method | Clean Acc. | MaxPres | PairOrd

Since CIFAR-10 has hard labels rather than known posteriors, MaxPres is computed
against the hard clean label, and PairOrd is the hard-label ranking proxy:
the fraction of class pairs (y,b), b != y, for which score_y > score_b.

Expected outputs in --out_dir:
  raw.csv
  summary.csv
  diagnostics.csv
  epoch_history.csv
  cifar10_standard_complementary_table.tex
  cifar10_standard_complementary_table.csv
  accuracy_bar.pdf/png
  loss_trend_over_epochs.pdf/png
  progress.txt

Recommended fast paper-quality run on kumo/GPU:
  python cifar10_clcl_clwl_standard_complementary_smallcnn.py \
      --data_root ./data \
      --out_dir artifacts_cifar10_clcl_clwl_standard_comp_smallcnn \
      --model small_cnn \
      --cnn_width 32 \
      --epochs 60 \
      --seeds 0 1 2 \
      --methods clcl_or,clcl_orw,clwl \
      --batch_size 256 \
      --optimizer adamw \
      --scheduler none \
      --lr 1e-3 \
      --weight_decay 1e-4 \
      --augment \
      --amp \
      --channels_last \
      --num_workers 4 \
      --eval_every 10 \
      --device cuda

Quick smoke test without CIFAR-10, not for reporting:
  python cifar10_clcl_clwl_standard_complementary_smallcnn.py --fake_data --max_train 128 --max_test 64 --epochs 1 --seeds 0 --methods clcl_or,clcl_orw,clwl --device cpu
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import tarfile
import time
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# -----------------------------------------------------------------------------
# Complementary-label transition model
# -----------------------------------------------------------------------------


def make_cifar_pi() -> np.ndarray:
    """Semantic-confusion permutation for CIFAR-10.

    Standard CIFAR-10 label order:
      0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer,
      5 dog, 6 frog, 7 horse, 8 ship, 9 truck.

    We use two no-fixed-point cycles:
      transport: airplane -> ship -> automobile -> truck -> airplane
      animals:   bird -> deer -> horse -> dog -> cat -> frog -> bird

    The biased complementary label for class y is pi(y), and pi(y) != y.
    """
    pi = np.empty(10, dtype=int)
    for cyc in ([0, 8, 1, 9], [2, 4, 7, 5, 3, 6]):
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            pi[a] = b
    if np.any(pi == np.arange(10)):
        raise RuntimeError("pi must have no fixed points for complementary labels")
    return pi


def build_semantic_bias_matrix(c: int = 10, seed: int = 20260506, main_mass: float = 0.70):
    """Build a fixed non-uniform complementary-label bias matrix B.

    Each column y is a probability distribution over wrong labels only.  Most
    mass is assigned to a semantic complement pi(y), while the remaining mass is
    spread over the other wrong labels with a fixed Dirichlet draw.  This keeps
    the labels purely complementary and avoids the accidental rank deficiency
    that can occur when mixing the uniform channel with a single permutation.
    """
    pi = make_cifar_pi()
    rng = np.random.default_rng(seed)
    B = np.zeros((c, c), dtype=np.float64)
    for y in range(c):
        B[int(pi[y]), y] = main_mass
        others = [z for z in range(c) if z != y and z != int(pi[y])]
        tail = rng.dirichlet(0.8 * np.ones(len(others)))
        for z, val in zip(others, tail):
            B[z, y] += (1.0 - main_mass) * float(val)
    return B, pi


def build_complementary_model(rho: float = 0.0, c: int = 10):
    """Build the standard uniform complementary-label channel.

    M is c x c, column-stochastic, where M[z, y] = P(Z=z | Y=y).
    The diagonal is zero and all off-diagonal entries are 1/(c-1).
    The argument rho is ignored and kept only for compatibility with the
    older biased-complementary script.
    """
    assert c == 10, "This CIFAR-10 protocol assumes c=10."
    M = (np.ones((c, c), dtype=np.float64) - np.eye(c, dtype=np.float64)) / (c - 1)
    pi = make_cifar_pi()
    if not np.allclose(M.sum(axis=0), 1.0):
        raise RuntimeError("M is not column-stochastic")
    if not np.allclose(np.diag(M), 0.0):
        raise RuntimeError("M has nonzero diagonal, so labels are not purely complementary")
    return M, pi


def construct_T(M: np.ndarray, alpha_scale: float = 0.99):
    """Construct CLWL transformation T from a full-column-rank M.

    M is d x c, here d=c=10.  N=(M^T M)^{-1}M^T is a left inverse.  The
    construction T=alpha(N-1 q^T) gives T in [0,1] and TM=alpha I + 1 v^T.
    """
    rank = np.linalg.matrix_rank(M)
    c = M.shape[1]
    if rank < c:
        raise ValueError(f"M must have full column rank {c}; got rank {rank}")

    N = np.linalg.solve(M.T @ M, M.T)
    q = N.min(axis=0)
    ranges = N.max(axis=0) - N.min(axis=0)
    Delta = ranges.max()
    alpha = alpha_scale / Delta if Delta > 0 else 1.0
    T = alpha * (N - np.ones((N.shape[0], 1)) @ q.reshape(1, -1))
    A = T @ M
    return T, A, alpha, q


def native_clcl_T(c: int = 10):
    """Implicit T for the native complementary-label OP/OvA loss.

    For a complementary one-hot label z, native CLCL-OR with OvA logistic is
      softplus(s_z) + sum_{j != z} softplus(-s_j),
    which is CLWL with Tz = 1-z, i.e. T = 11^T - I.
    """
    return np.ones((c, c), dtype=np.float64) - np.eye(c, dtype=np.float64)


def order_preserving_residual(A: np.ndarray):
    """Relative residual to A = lambda I + 1 v^T."""
    c = A.shape[0]
    rows, b = [], []
    for i in range(c):
        for j in range(c):
            row = np.zeros(c + 1)
            row[0] = 1.0 if i == j else 0.0
            row[1 + j] = 1.0
            rows.append(row)
            b.append(A[i, j])
    X = np.vstack(rows)
    y = np.asarray(b)
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    lam = sol[0]
    v = sol[1:]
    A_hat = lam * np.eye(c) + np.ones((c, 1)) @ v.reshape(1, -1)
    rel = np.linalg.norm(A - A_hat, ord="fro") / max(np.linalg.norm(A, ord="fro"), 1e-12)
    return float(lam), float(rel)


def matrix_order_rates(A: np.ndarray, n: int = 20000, seed: int = 0):
    """Monte-Carlo order diagnostics over random posteriors eta.

    Returns max-preservation and pairwise order-preservation rates for A eta
    relative to eta.  This is a matrix-level diagnostic, independent of images.
    """
    rng = np.random.default_rng(seed)
    c = A.shape[0]
    eta = rng.dirichlet(alpha=np.ones(c), size=n)
    score = eta @ A.T
    max_rate = np.mean(np.argmax(score, axis=1) == np.argmax(eta, axis=1))

    ok = 0
    total = 0
    for i in range(n):
        e = eta[i]
        s = score[i]
        for a in range(c):
            for b in range(c):
                if a == b:
                    continue
                if e[a] > e[b]:
                    total += 1
                    ok += int(s[a] > s[b])
    pair_rate = ok / max(total, 1)
    return float(max_rate), float(pair_rate)


def complementary_diagnostics(M: np.ndarray, rho: float):
    T_clwl, A_clwl, alpha, q = construct_T(M)
    lam_clwl, rel_clwl = order_preserving_residual(A_clwl)
    T_native = native_clcl_T(M.shape[1])
    A_native = T_native @ M
    lam_native, rel_native = order_preserving_residual(A_native)
    native_max, native_pair = matrix_order_rates(A_native, seed=123)
    clwl_max, clwl_pair = matrix_order_rates(A_clwl, seed=123)
    return {
        "rho": rho,
        "rankM": int(np.linalg.matrix_rank(M)),
        "cond_MtM": float(np.linalg.cond(M.T @ M)),
        "diag_max_abs": float(np.max(np.abs(np.diag(M)))),
        "alpha": float(alpha),
        "clwl_lambda_hat": lam_clwl,
        "clwl_relres_TM": rel_clwl,
        "clwl_T_min": float(T_clwl.min()),
        "clwl_T_max": float(T_clwl.max()),
        "native_lambda_hat": lam_native,
        "native_relres_TM": rel_native,
        "native_matrix_maxpres": native_max,
        "native_matrix_pairord": native_pair,
        "clwl_matrix_maxpres": clwl_max,
        "clwl_matrix_pairord": clwl_pair,
    }, T_clwl, A_clwl, T_native, A_native


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def _load_cifar10_pickled(data_root: Path):
    base = data_root / "cifar-10-batches-py"
    if not base.exists():
        raise FileNotFoundError(f"Could not find {base}")

    xs, ys = [], []
    for i in range(1, 6):
        with open(base / f"data_batch_{i}", "rb") as f:
            obj = pickle.load(f, encoding="latin1")
        xs.append(obj["data"])
        ys.extend(obj["labels"])
    X_train = np.concatenate(xs, axis=0).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_train = np.asarray(ys, dtype=np.int64)

    with open(base / "test_batch", "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    X_test = obj["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_test = np.asarray(obj["labels"], dtype=np.int64)
    return X_train, y_train, X_test, y_test


def maybe_download_cifar10(data_root: Path):
    base = data_root / "cifar-10-batches-py"
    if base.exists():
        return
    data_root.mkdir(parents=True, exist_ok=True)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = data_root / "cifar-10-python.tar.gz"
    print(f"Downloading CIFAR-10 from {url} to {tar_path} ...", flush=True)
    urlretrieve(url, tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_root)


def stratified_subset_indices(y: np.ndarray, n_total: Optional[int], seed: int):
    if n_total is None or n_total <= 0 or n_total >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, n_total // len(classes))
    idxs = []
    for k in classes:
        ids = np.where(y == k)[0]
        take = min(per_class, len(ids))
        idxs.append(rng.choice(ids, size=take, replace=False))
    idx = np.concatenate(idxs)
    rng.shuffle(idx)
    return idx


def make_fake_cifar(n_train=512, n_test=256, seed=0):
    rng = np.random.default_rng(seed)
    X_train = rng.random((n_train, 3, 32, 32), dtype=np.float32)
    y_train = np.arange(n_train, dtype=np.int64) % 10
    rng.shuffle(y_train)
    X_test = rng.random((n_test, 3, 32, 32), dtype=np.float32)
    y_test = np.arange(n_test, dtype=np.int64) % 10
    rng.shuffle(y_test)
    return X_train, y_train, X_test, y_test


def load_cifar10(args):
    if args.fake_data:
        return make_fake_cifar(args.max_train or 512, args.max_test or 256, seed=123)

    data_root = Path(args.data_root)
    if args.download:
        maybe_download_cifar10(data_root)

    try:
        return _load_cifar10_pickled(data_root)
    except FileNotFoundError:
        try:
            from torchvision.datasets import CIFAR10
            ds_tr = CIFAR10(root=str(data_root), train=True, download=args.download)
            ds_te = CIFAR10(root=str(data_root), train=False, download=args.download)
            X_train = np.asarray(ds_tr.data).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            y_train = np.asarray(ds_tr.targets, dtype=np.int64)
            X_test = np.asarray(ds_te.data).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            y_test = np.asarray(ds_te.targets, dtype=np.int64)
            return X_train, y_train, X_test, y_test
        except Exception as e:
            raise FileNotFoundError(
                "CIFAR-10 was not found. Put cifar-10-batches-py under --data_root, "
                "or run with --download on a machine with internet. "
                "For a non-scientific smoke test, use --fake_data."
            ) from e


class CIFARWeakDataset(Dataset):
    def __init__(self, X, y, z=None, train: bool = False, augment: bool = False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.z = None if z is None else torch.tensor(z, dtype=torch.long)
        self.train = train
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def _augment(self, x: torch.Tensor):
        if not (self.train and self.augment):
            return x
        x = F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
        top = torch.randint(0, 9, (1,)).item()
        left = torch.randint(0, 9, (1,)).item()
        x = x[:, top:top + 32, left:left + 32]
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=(2,))
        return x

    def __getitem__(self, idx):
        x = self._augment(self.X[idx])
        x = (x - CIFAR10_MEAN) / CIFAR10_STD
        if self.z is None:
            return x, self.y[idx]
        return x, self.y[idx], self.z[idx]


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------



class TinyCNN(nn.Module):
    """Very small model for smoke tests; not intended for reported CIFAR-10 results."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10, width=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1, bias=False),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1, bias=False),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * width, 4 * width, 3, padding=1, bias=False),
            nn.BatchNorm2d(4 * width),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(4 * width, num_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


class ResNetSmall(nn.Module):
    def __init__(self, num_classes=10, width=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(BasicBlock(width, width), BasicBlock(width, width))
        self.layer2 = nn.Sequential(BasicBlock(width, 2 * width, stride=2), BasicBlock(2 * width, 2 * width))
        self.layer3 = nn.Sequential(BasicBlock(2 * width, 4 * width, stride=2), BasicBlock(4 * width, 4 * width))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4 * width, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.fc(self.pool(x).flatten(1))


def make_model(name: str, cnn_width: int = 32):
    if name == "tiny_cnn":
        return TinyCNN()
    if name == "small_cnn":
        return SmallCNN(width=cnn_width)
    if name == "resnet_small":
        return ResNetSmall(width=32)
    raise ValueError(f"Unknown model: {name}")


# -----------------------------------------------------------------------------
# Losses and training
# -----------------------------------------------------------------------------


def sample_weak_indices(y: np.ndarray, M: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    z = np.empty(len(y), dtype=np.int64)
    for i, yy in enumerate(y.astype(int)):
        z[i] = rng.choice(M.shape[0], p=M[:, yy])
    return z


def weak_loss(logits, z, method: str, T_rows=None, orw_q: float = 0.5, eps: float = 1e-6):
    c = logits.shape[1]
    z_onehot = F.one_hot(z, num_classes=c).to(dtype=logits.dtype)

    if method in {"clcl_or", "clcl"}:
        # CLCL/OP with OvA logistic: ell(-g,z) = softplus(g_z)+sum_{j!=z}softplus(-g_j).
        return (z_onehot * F.softplus(logits) + (1.0 - z_onehot) * F.softplus(-logits)).sum(dim=1).mean()

    if method == "clcl_orw":
        base = (z_onehot * F.softplus(logits) + (1.0 - z_onehot) * F.softplus(-logits)).sum(dim=1)
        # Practical OP-W style weighting: put more weight on samples where the
        # model currently assigns high probability to the complementary label.
        # The batch-normalization keeps the average scale comparable to CLCL-OR.
        pz = torch.softmax(logits, dim=1).gather(1, z.view(-1, 1)).squeeze(1)
        weight = (pz + eps).pow(orw_q)
        weight = weight / weight.detach().mean().clamp_min(eps)
        return (weight * base).mean()

    if method == "clwl":
        if T_rows is None:
            raise ValueError("T_rows is required for CLWL")
        tz = T_rows[z]
        return (tz * F.softplus(-logits) + (1.0 - tz) * F.softplus(logits)).sum(dim=1).mean()

    raise ValueError(method)


@torch.no_grad()
def evaluate(model, loader, device, channels_last: bool = False):
    metrics = evaluate_metrics(model, loader, device, channels_last=channels_last)
    return metrics["clean_acc"]


@torch.no_grad()
def evaluate_metrics(model, loader, device, channels_last: bool = False):
    """Evaluate clean accuracy and hard-label ranking proxies.

    MaxPres equals top-1 agreement with the hard clean label. PairOrd is the
    fraction of pairs (y,b), b != y, for which score_y > score_b. This is a
    hard-label analogue of posterior pairwise order preservation.
    """
    model.eval()
    correct = 0
    pair_ok = 0
    pair_total = 0
    total = 0
    for batch in loader:
        x = batch[0].to(device)
        if channels_last and device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        y = batch[1].to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        sy = logits.gather(1, y.view(-1, 1))
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, y.view(-1, 1), False)
        pair_ok += (sy > logits[mask].view(logits.size(0), -1)).sum().item()
        pair_total += logits.size(0) * (logits.size(1) - 1)
    clean_acc = correct / max(total, 1)
    return {
        "clean_acc": clean_acc,
        "maxpres": clean_acc,
        "pairord": pair_ok / max(pair_total, 1),
    }


def make_optimizer(args, model):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9,
            weight_decay=args.weight_decay, nesterov=True,
        )
    raise ValueError(args.optimizer)


def make_scheduler(args, opt):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    raise ValueError(args.scheduler)


def train_one(args, Xtr, ytr, Xte, yte, ztr, T, method: str, rho: float, seed: int):
    set_seed(seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    train_ds = CIFARWeakDataset(Xtr, ytr, z=ztr, train=True, augment=args.augment)
    test_ds = CIFARWeakDataset(Xte, yte, z=None, train=False, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    train_eval_loader = DataLoader(CIFARWeakDataset(Xtr, ytr, z=None, train=False, augment=False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = make_model(args.model, cnn_width=args.cnn_width).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    opt = make_optimizer(args, model)
    scheduler = make_scheduler(args, opt)
    T_rows = torch.tensor(T.T, dtype=torch.float32, device=device) if T is not None else None
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for x, _y, z in train_loader:
            x = x.to(device, non_blocking=True)
            if args.channels_last and device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            z = z.to(device, non_blocking=True)
            use_amp = bool(args.amp and device.type == "cuda")
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = weak_loss(logits, z, method, T_rows=T_rows, orw_q=args.orw_q)
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                loss = weak_loss(logits, z, method, T_rows=T_rows, orw_q=args.orw_q)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
            loss_sum += float(loss.detach().cpu()) * x.size(0)
            n_seen += x.size(0)
        if scheduler is not None:
            scheduler.step()

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            train_acc = evaluate(model, train_eval_loader, device, channels_last=args.channels_last)
            test_acc = evaluate(model, test_loader, device, channels_last=args.channels_last)
        else:
            train_acc = math.nan
            test_acc = math.nan

        history.append({
            "rho": rho,
            "seed": seed,
            "method": method,
            "epoch": epoch,
            "train_loss": loss_sum / max(n_seen, 1),
            "train_acc": train_acc,
            "test_acc": test_acc,
        })
        print(f"rho={rho:g} seed={seed} method={method} epoch={epoch:03d} "
              f"loss={history[-1]['train_loss']:.4f} test_acc={test_acc:.4f}", flush=True)

    final_train_metrics = evaluate_metrics(model, train_eval_loader, device, channels_last=args.channels_last)
    final_test_metrics = evaluate_metrics(model, test_loader, device, channels_last=args.channels_last)
    return final_test_metrics, final_train_metrics, history


# -----------------------------------------------------------------------------
# Plotting and live saving
# -----------------------------------------------------------------------------


def plot_accuracy_vs_bias(summary: pd.DataFrame, out_base: Path):
    """Bar plot for the standard complementary-label comparison."""
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    method_order = [m for m in ["clcl_or", "clcl_orw", "clwl"] if m in set(summary["method"])]
    label_map = {"clcl_or": "CLCL-OR", "clcl_orw": "CLCL-ORW", "clwl": "CLWL"}
    xs = np.arange(len(method_order))
    vals = []
    errs = []
    labels = []
    for m in method_order:
        sub = summary[summary["method"] == m]
        vals.append(float(sub["test_acc_mean"].iloc[0]))
        errs.append(float(sub["test_acc_std"].fillna(0.0).iloc[0]))
        labels.append(label_map.get(m, m.upper()))
    ax.bar(xs, vals, yerr=errs, capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Test accuracy")
    ax.set_title("CIFAR-10: standard complementary labels")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loss_trend_over_epochs(epoch_df: pd.DataFrame, out_base: Path, rhos_to_plot=None):
    plot_df = epoch_df.copy()
    if rhos_to_plot:
        keep = np.zeros(len(plot_df), dtype=bool)
        for rho in rhos_to_plot:
            keep |= np.isclose(plot_df["rho"].to_numpy(float), rho)
        plot_df = plot_df.loc[keep]

    mean_df = plot_df.groupby(["rho", "method", "epoch"], as_index=False).agg(
        train_loss_mean=("train_loss", "mean"),
        train_loss_std=("train_loss", "std"),
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    label_map = {"clcl_or": "CLCL-OR", "clcl_orw": "CLCL-ORW", "clwl": "CLWL"}
    for (rho, method), sub in mean_df.groupby(["rho", "method"]):
        sub = sub.sort_values("epoch")
        ax.plot(sub["epoch"], sub["train_loss_mean"], linewidth=1.6,
                label=f"{label_map.get(method, method.upper())}, rho={rho:g}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("CIFAR-10: loss trend over epochs")
    ax.grid(True, alpha=0.6)
    ax.legend(fontsize=8, ncol=2, framealpha=1.0)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def build_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    return raw_df.groupby(["dataset", "model", "cnn_width", "rho", "method"], as_index=False).agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        test_maxpres_mean=("test_maxpres", "mean"),
        test_maxpres_std=("test_maxpres", "std"),
        test_pairord_mean=("test_pairord", "mean"),
        test_pairord_std=("test_pairord", "std"),
        train_acc_mean=("train_acc", "mean"),
        train_acc_std=("train_acc", "std"),
        n_completed=("test_acc", "count"),
        rankM=("rankM", "first"),
        cond_MtM=("cond_MtM", "first"),
        diag_max_abs=("diag_max_abs", "first"),
        alpha=("alpha", "first"),
        clwl_lambda_hat=("clwl_lambda_hat", "first"),
        clwl_relres_TM=("clwl_relres_TM", "first"),
        clwl_T_min=("clwl_T_min", "first"),
        clwl_T_max=("clwl_T_max", "first"),
        native_lambda_hat=("native_lambda_hat", "first"),
        native_relres_TM=("native_relres_TM", "first"),
        native_matrix_maxpres=("native_matrix_maxpres", "first"),
        native_matrix_pairord=("native_matrix_pairord", "first"),
        clwl_matrix_maxpres=("clwl_matrix_maxpres", "first"),
        clwl_matrix_pairord=("clwl_matrix_pairord", "first"),
    )


def format_mean_std(mean, std, digits=3):
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def save_paper_table(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    label_map = {"clcl_or": "CLCL-OR", "clcl_orw": "CLCL-ORW", "clwl": "CLWL"}
    method_order = [m for m in ["clcl_or", "clcl_orw", "clwl"] if m in set(summary["method"])]
    rows = []
    for m in method_order:
        sub = summary[summary["method"] == m].iloc[0]
        rows.append({
            "Setting": "CIFAR-10 complementary",
            "Method": label_map.get(m, m.upper()),
            "Clean Acc.": format_mean_std(sub["test_acc_mean"], sub["test_acc_std"]),
            "MaxPres": format_mean_std(sub["test_maxpres_mean"], sub["test_maxpres_std"]),
            "PairOrd": format_mean_std(sub["test_pairord_mean"], sub["test_pairord_std"]),
        })
    table_df = pd.DataFrame(rows)
    table_df.to_csv(out_dir / "cifar10_standard_complementary_table.csv", index=False)
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{CIFAR-10 standard complementary-label comparison. MaxPres and PairOrd are computed using the hard clean labels as ranking references.}",
        r"\label{tab:cifar10-standard-complementary}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        "Setting & Method & Clean Acc. $\\uparrow$ & MaxPres $\\uparrow$ & PairOrd $\\uparrow$ \\\\",
        r"\midrule",
    ]
    for r in rows:
        tex_lines.append(f"{r['Setting']} & {r['Method']} & {r['Clean Acc.']} & {r['MaxPres']} & {r['PairOrd']} \\\\")
    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    (out_dir / "cifar10_standard_complementary_table.tex").write_text("\n".join(tex_lines))


def save_live_outputs(rows, epoch_rows, out_dir: Path, args, completed_runs: int, final: bool = False) -> pd.DataFrame:
    raw_df = pd.DataFrame(rows)
    epoch_df = pd.DataFrame(epoch_rows)

    if not raw_df.empty:
        _atomic_to_csv(raw_df, out_dir / "raw.csv")
        summary = build_summary(raw_df)
        _atomic_to_csv(summary, out_dir / "summary.csv")
        save_paper_table(summary, out_dir)
        plot_accuracy_vs_bias(summary, out_dir / "accuracy_bar")
    else:
        summary = pd.DataFrame()

    if not epoch_df.empty:
        _atomic_to_csv(epoch_df, out_dir / "epoch_history.csv")
        plot_loss_trend_over_epochs(epoch_df, out_dir / "loss_trend_over_epochs", args.loss_plot_rhos)

    progress_lines = [
        f"completed_runs={completed_runs}",
        f"final={final}",
        f"updated_at_unix={time.time():.3f}",
    ]
    if not raw_df.empty:
        last = raw_df.iloc[-1]
        progress_lines.append(
            "last="
            f"rho={last['rho']}, seed={last['seed']}, method={last['method']}, "
            f"test_acc={last['test_acc']:.6f}, train_acc={last['train_acc']:.6f}"
        )
    (out_dir / "progress.txt").write_text("\n".join(progress_lines) + "\n")

    print(
        f"[live-save] completed_runs={completed_runs} | updated raw.csv, summary.csv, "
        f"epoch_history.csv, accuracy_bar.pdf/png, loss_trend_over_epochs.pdf/png",
        flush=True,
    )
    return summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="artifacts_cifar10_clcl_clwl_standard_comp_smallcnn")
    p.add_argument("--download", action="store_true", help="Download CIFAR-10 if not present")
    p.add_argument("--fake_data", action="store_true", help="Use random CIFAR-shaped data for a code smoke test only")

    p.add_argument("--rhos", type=float, nargs="+", default=[0.0], help="Kept for compatibility. For the standard complementary experiment use only 0.0.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--methods", type=str, default="clcl_or,clcl_orw,clwl", help="Comma-separated subset of clcl_or,clcl_orw,clwl")

    p.add_argument("--model", choices=["tiny_cnn", "small_cnn", "resnet_small"], default="small_cnn")
    p.add_argument("--cnn_width", type=int, default=32, help="Width of SmallCNN; use 32 for fastest paper-quality runs, 48/64 for stronger appendix runs")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="adamw")
    p.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--orw_q", type=float, default=0.5, help="Exponent for practical CLCL-ORW softmax weight")
    p.add_argument("--amp", action="store_true", help="Use automatic mixed precision on CUDA for speed")
    p.add_argument("--channels_last", action="store_true", help="Use channels-last memory format on CUDA for speed")

    p.add_argument("--max_train", type=int, default=None, help="Optional stratified subset size for quick tests")
    p.add_argument("--max_test", type=int, default=None, help="Optional stratified subset size for quick tests")
    p.add_argument("--loss_plot_rhos", type=float, nargs="*", default=[0.0])
    p.add_argument("--diagnose_only", action="store_true", help="Only compute M/T diagnostics; no data loading or training")
    return p.parse_args()


def main():
    torch.set_num_threads(2)
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(abs(r) > 1e-12 for r in args.rhos):
        raise ValueError("This script is for the standard uniform complementary setting. Use --rhos 0 only.")
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    aliases = {"clcl": "clcl_or"}
    methods = [aliases.get(m, m) for m in methods]
    for m in methods:
        if m not in {"clcl_or", "clcl_orw", "clwl"}:
            raise ValueError(f"Unknown method {m}")

    t0 = time.time()

    # Weak-channel diagnostics independent of images.
    diag_rows = []
    weak_cache = {}
    for rho in args.rhos:
        M, pi = build_complementary_model(rho)
        diag, T_clwl, A_clwl, T_native, A_native = complementary_diagnostics(M, rho)
        diag_rows.append(diag)
        weak_cache[rho] = (M, T_clwl, A_clwl, T_native, A_native)
    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(out_dir / "diagnostics.csv", index=False)
    print("\nCOMPLEMENTARY-CHANNEL DIAGNOSTICS")
    print(diag_df.to_string(index=False))

    if args.diagnose_only:
        print(f"\nSaved diagnostics to {out_dir / 'diagnostics.csv'}")
        return

    Xtr, ytr, Xte, yte = load_cifar10(args)
    idx_tr = stratified_subset_indices(ytr, args.max_train, seed=123)
    idx_te = stratified_subset_indices(yte, args.max_test, seed=456)
    Xtr, ytr = Xtr[idx_tr], ytr[idx_tr]
    Xte, yte = Xte[idx_te], yte[idx_te]
    print(f"Loaded CIFAR-10: train={len(ytr)}, test={len(yte)}, fake_data={args.fake_data}")

    rows = []
    epoch_rows = []
    for rho in args.rhos:
        M, T_clwl, A_clwl, T_native, A_native = weak_cache[rho]
        diag = diag_df[diag_df["rho"] == rho].iloc[0].to_dict()
        for seed in args.seeds:
            ztr = sample_weak_indices(ytr, M, seed + 23456 + int(round(1000 * rho)))
            for method in methods:
                T_method = T_clwl if method == "clwl" else None
                test_metrics, train_metrics, history = train_one(
                    args, Xtr, ytr, Xte, yte, ztr, T_method, method, rho, seed
                )
                test_acc = test_metrics["clean_acc"]
                train_acc = train_metrics["clean_acc"]
                for h in history:
                    h.update({"dataset": "cifar10", "model": args.model, "cnn_width": args.cnn_width, **diag})
                epoch_rows.extend(history)
                row = {
                    "dataset": "cifar10",
                    "model": args.model,
                    "cnn_width": args.cnn_width,
                    "rho": rho,
                    "seed": seed,
                    "method": method,
                    "test_acc": test_acc,
                    "test_maxpres": test_metrics["maxpres"],
                    "test_pairord": test_metrics["pairord"],
                    "train_acc": train_acc,
                    "train_maxpres": train_metrics["maxpres"],
                    "train_pairord": train_metrics["pairord"],
                    **diag,
                }
                rows.append(row)
                print("FINAL", row, flush=True)
                save_live_outputs(rows, epoch_rows, out_dir, args, completed_runs=len(rows), final=False)

    summary = save_live_outputs(rows, epoch_rows, out_dir, args, completed_runs=len(rows), final=True)

    print("\nSUMMARY")
    if summary.empty:
        print("No training runs were completed.")
    else:
        print(summary.to_string(index=False))
    print(f"\nSaved to {out_dir}. Runtime: {time.time() - t0:.1f}s")
    print(f"Saved plot: {out_dir / 'accuracy_bar.pdf'}")
    print(f"Saved plot: {out_dir / 'loss_trend_over_epochs.pdf'}")


if __name__ == "__main__":
    main()
