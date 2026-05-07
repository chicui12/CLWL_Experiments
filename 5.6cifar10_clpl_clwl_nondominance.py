#!/usr/bin/env python3
"""
CIFAR-10 incomplete asymmetric partial-label experiment for CLPL vs CLWL.

This script ports the MNIST non-dominance experiment to CIFAR-10.
The weak-label model is the same size-two partial-label protocol:

    with probability 1-rho: B = {y, u}, u uniform over non-y labels
    with probability rho:   B = {pi(y), pi^2(y)}

where pi is a fixed CIFAR-10 semantic-confusion permutation.  The second
branch deliberately omits the true class and injects structured decoys.

The comparison is fair in the following sense:
  * both methods see the same CIFAR-10 examples;
  * both methods see the same sampled weak labels;
  * both methods use the same architecture, optimizer, and training budget;
  * CLPL uses the native candidate-set loss;
  * CLWL uses the known transition matrix M_rho to construct T_rho.

Expected outputs in --out_dir:
  raw.csv
  summary.csv
  diagnostics.csv
  epoch_history.csv
  accuracy_vs_probability.pdf/png
  loss_trend_over_epochs.pdf/png
  progress.txt

During a long run, raw.csv, summary.csv, epoch_history.csv, and both plots are
updated after every completed (rho, seed, method) run, so you can monitor the
comparison without waiting for the full grid to finish.

Examples:
  # Recommended full run on kumo/GPU
  python cifar10_clpl_clwl_nondominance.py \
      --data_root ./data \
      --out_dir artifacts_cifar10_clpl_clwl_nondom \
      --model resnet_small \
      --epochs 120 \
      --rhos 0 0.1 0.2 0.3 0.4 0.5 0.6 \
      --seeds 0 1 2 \
      --batch_size 128 \
      --optimizer sgd \
      --lr 0.05 \
      --weight_decay 5e-4 \
      --augment \
      --device cuda

  # Quick smoke test without CIFAR-10, not for reporting
  python cifar10_clpl_clwl_nondominance.py --fake_data --max_train 512 --max_test 256 --epochs 1
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import pickle
import random
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
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
# Weak-label model
# -----------------------------------------------------------------------------


def make_cifar_pi() -> np.ndarray:
    """Semantic-confusion permutation for CIFAR-10.

    Labels follow the standard CIFAR-10 order:
      0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer,
      5 dog, 6 frog, 7 horse, 8 ship, 9 truck.

    We use two cycles:
      transport: airplane -> ship -> automobile -> truck -> airplane
      animals:   bird -> deer -> horse -> dog -> cat -> frog -> bird

    The decoy pair for class y is {pi(y), pi^2(y)}, so the true class is absent
    in the incomplete asymmetric branch.
    """
    pi = np.empty(10, dtype=int)
    for cyc in ([0, 8, 1, 9], [2, 4, 7, 5, 3, 6]):
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            pi[a] = b
    return pi


def build_weak_model(rho: float, c: int = 10):
    """Build size-two candidate-set channel M_rho.

    Returns:
      pairs: list of weak labels, each a tuple of two clean labels
      Z:     c x d binary candidate-set matrix; Z[j,k]=1 iff class j in pair k
      M:     d x c column-stochastic transition matrix P(Z=k | Y=j)
      pi:    semantic-confusion permutation
    """
    assert c == 10, "This CIFAR-10 protocol assumes c=10."
    pi = make_cifar_pi()
    pairs = [tuple(p) for p in itertools.combinations(range(c), 2)]
    pair_to_idx = {p: i for i, p in enumerate(pairs)}

    Z = np.zeros((c, len(pairs)), dtype=np.float64)
    for k, pair in enumerate(pairs):
        Z[list(pair), k] = 1.0

    M = np.zeros((len(pairs), c), dtype=np.float64)
    for y in range(c):
        # Standard partial-label branch: true label plus one uniformly random wrong label.
        for u in range(c):
            if u == y:
                continue
            M[pair_to_idx[tuple(sorted((y, u)))], y] += (1.0 - rho) / (c - 1)

        # Incomplete asymmetric branch: true label is absent; structured decoys are shown.
        decoy_pair = tuple(sorted((int(pi[y]), int(pi[pi[y]]))))
        M[pair_to_idx[decoy_pair], y] += rho

    # Numerical sanity check.
    if not np.allclose(M.sum(axis=0), 1.0):
        raise RuntimeError("M is not column-stochastic")
    return pairs, Z, M, pi


def construct_T(M: np.ndarray, alpha_scale: float = 0.99):
    """Construct CLWL transformation T from a full-column-rank M.

    M is d x c.  N=(M^T M)^{-1}M^T is a left inverse.  The construction
    T=alpha(N-1 q^T) yields T in [0,1] and TM=alpha I + 1 v^T.
    """
    rank = np.linalg.matrix_rank(M)
    c = M.shape[1]
    if rank < c:
        raise ValueError(f"M must have full column rank {c}; got rank {rank}")

    N = np.linalg.solve(M.T @ M, M.T)  # c x d
    q = N.min(axis=0)                 # d
    ranges = N.max(axis=0) - N.min(axis=0)
    Delta = ranges.max()
    alpha = alpha_scale / Delta if Delta > 0 else 1.0
    T = alpha * (N - np.ones((N.shape[0], 1)) @ q.reshape(1, -1))
    A = T @ M
    return T, A, alpha, q


def order_preserving_residual(A: np.ndarray):
    """Relative residual to the form lambda I + 1 v^T.

    Returns (lambda_hat, rel_resid).  This is used only as a diagnostic.
    """
    c = A.shape[0]
    rows = []
    b = []
    # Unknowns are lambda and v_0,...,v_{c-1}.
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


def dominance_diagnostics(M: np.ndarray, Z: np.ndarray, pairs):
    """Cour-style dominance diagnostics for size-two candidate sets.

    For this rho-family, dominance should hold at rho=0 and be violated for
    sufficiently large rho.  The argmax-set diagnostic checks whether the true
    class is the unique maximizer of the marginal candidate-inclusion probability.
    """
    c = Z.shape[0]
    argmax_set_ok = 0
    dom_violations = 0
    dom_checks = 0
    labels = set(range(c))

    for y in range(c):
        p = M[:, y]
        inclusion = Z @ p
        argq = set(np.where(np.abs(inclusion - inclusion.max()) < 1e-10)[0].tolist())
        if argq == {y}:
            argmax_set_ok += 1

        prob = {tuple(pair): p[k] for k, pair in enumerate(pairs) if p[k] > 1e-15}
        for b in range(c):
            if b == y:
                continue
            others = list(labels - {y, b})
            # For size-two sets, only |C|=1 contributes nonzero probability;
            # we nevertheless keep the generic check for compatibility with the MNIST script.
            for r in range(len(others) + 1):
                for C in itertools.combinations(others, r):
                    S_y = tuple(sorted(C + (y,)))
                    S_b = tuple(sorted(C + (b,)))
                    py = prob.get(S_y, 0.0)
                    pb = prob.get(S_b, 0.0)
                    dom_checks += 1
                    if py + 1e-12 < pb:
                        dom_violations += 1

    return {
        "argmax_set_equal_rate": argmax_set_ok / c,
        "dominance_violation_rate": dom_violations / dom_checks,
        "dominance_violations": dom_violations,
        "dominance_checks": dom_checks,
    }


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
    """Download official CIFAR-10 tarball if needed.

    On clusters without internet, manually place/extract cifar-10-batches-py under
    --data_root.  The script first tries direct pickle loading, so torchvision is
    not required when the files already exist.
    """
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
    """Small random CIFAR-shaped data for code smoke tests only."""
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
        # Fallback to torchvision if available.  This is convenient on machines where
        # torchvision is correctly installed.  Direct pickle loading remains preferred.
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
                "For a non-scientific code smoke test, use --fake_data."
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
        # Padding 4 then random crop 32x32.
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
        x = self.features(x).flatten(1)
        return self.fc(x)


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
        x = self.pool(x).flatten(1)
        return self.fc(x)


def make_model(name: str):
    if name == "small_cnn":
        return SmallCNN(width=64)
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


def weak_loss(logits, z, method: str, Z_rows, T_rows=None):
    if method == "clpl":
        b = Z_rows[z]  # batch x c
        mean_s = (b * logits).sum(dim=1) / b.sum(dim=1).clamp_min(1.0)
        return (F.softplus(-mean_s) + ((1.0 - b) * F.softplus(logits)).sum(dim=1)).mean()
    if method == "clwl":
        if T_rows is None:
            raise ValueError("T_rows is required for CLWL")
        tz = T_rows[z]  # batch x c
        return (tz * F.softplus(-logits) + (1.0 - tz) * F.softplus(logits)).sum(dim=1).mean()
    raise ValueError(method)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


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


def train_one(args, Xtr, ytr, Xte, yte, ztr, Z, T, method: str, rho: float, seed: int):
    set_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    train_ds = CIFARWeakDataset(Xtr, ytr, z=ztr, train=True, augment=args.augment)
    test_ds = CIFARWeakDataset(Xte, yte, z=None, train=False, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    train_eval_loader = DataLoader(CIFARWeakDataset(Xtr, ytr, z=None, train=False, augment=False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = make_model(args.model).to(device)
    opt = make_optimizer(args, model)
    scheduler = make_scheduler(args, opt)

    Z_rows = torch.tensor(Z.T, dtype=torch.float32, device=device)  # d x c
    T_rows = torch.tensor(T.T, dtype=torch.float32, device=device) if T is not None else None

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for x, _y, z in train_loader:
            x = x.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)
            logits = model(x)
            loss = weak_loss(logits, z, method, Z_rows, T_rows)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            loss_sum += float(loss.detach().cpu()) * x.size(0)
            n_seen += x.size(0)
        if scheduler is not None:
            scheduler.step()

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            train_acc = evaluate(model, train_eval_loader, device)
            test_acc = evaluate(model, test_loader, device)
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

    # Ensure final metrics are not NaN.
    final_train_acc = evaluate(model, train_eval_loader, device)
    final_test_acc = evaluate(model, test_loader, device)
    return final_test_acc, final_train_acc, history


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_accuracy_vs_probability(summary: pd.DataFrame, out_base: Path):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    method_order = [m for m in ["clpl", "clwl"] if m in set(summary["method"])]
    for method in method_order:
        sub = summary[summary["method"] == method].sort_values("rho")
        yerr = sub["test_acc_std"].fillna(0.0).to_numpy()
        ax.errorbar(sub["rho"], sub["test_acc_mean"], yerr=yerr,
                    marker="o", capsize=3, label=method.upper())
    ax.set_xlabel(r"Incomplete asymmetric probability $\rho$")
    ax.set_ylabel("Test accuracy")
    ax.set_title("CIFAR-10: accuracy vs. incomplete asymmetric probability")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.6)
    ax.legend(framealpha=1.0)
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
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for (rho, method), sub in mean_df.groupby(["rho", "method"]):
        sub = sub.sort_values("epoch")
        ax.plot(sub["epoch"], sub["train_loss_mean"], linewidth=1.6,
                label=f"{method.upper()}, rho={rho:g}")
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
    """Write a CSV through a temporary file, then atomically replace the target.

    This avoids leaving a half-written CSV if the process is interrupted while
    the file is being updated during a long CIFAR-10 run.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def build_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Create the same method-comparison summary from complete or partial raw rows."""
    if raw_df.empty:
        return pd.DataFrame()
    return raw_df.groupby(["dataset", "model", "rho", "method"], as_index=False).agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        train_acc_mean=("train_acc", "mean"),
        train_acc_std=("train_acc", "std"),
        n_completed=("test_acc", "count"),
        rankM=("rankM", "first"),
        cond_MtM=("cond_MtM", "first"),
        alpha=("alpha", "first"),
        lambda_hat=("lambda_hat", "first"),
        relres_TM_to_standard_form=("relres_TM_to_standard_form", "first"),
        T_min=("T_min", "first"),
        T_max=("T_max", "first"),
        argmax_set_equal_rate=("argmax_set_equal_rate", "first"),
        dominance_violation_rate=("dominance_violation_rate", "first"),
    )


def save_live_outputs(rows, epoch_rows, out_dir: Path, args, completed_runs: int, final: bool = False) -> pd.DataFrame:
    """Save raw logs, partial summary, and plots after each completed run.

    The same filenames are overwritten every time, so VS Code or a PDF/PNG
    viewer can be refreshed while the experiment is still running.
    """
    raw_df = pd.DataFrame(rows)
    epoch_df = pd.DataFrame(epoch_rows)

    if not raw_df.empty:
        _atomic_to_csv(raw_df, out_dir / "raw.csv")
        summary = build_summary(raw_df)
        _atomic_to_csv(summary, out_dir / "summary.csv")
        plot_accuracy_vs_probability(summary, out_dir / "accuracy_vs_probability")
    else:
        summary = pd.DataFrame()

    if not epoch_df.empty:
        _atomic_to_csv(epoch_df, out_dir / "epoch_history.csv")
        # This can be a little slower than the accuracy plot, but it is useful
        # for diagnosing whether CLPL is still optimizing while accuracy drops.
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
        f"[live-save] completed_runs={completed_runs} | "
        f"updated raw.csv, summary.csv, epoch_history.csv, "
        f"accuracy_vs_probability.pdf/png, loss_trend_over_epochs.pdf/png",
        flush=True,
    )
    return summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="artifacts_cifar10_clpl_clwl_nondom")
    p.add_argument("--download", action="store_true", help="Download CIFAR-10 if not present")
    p.add_argument("--fake_data", action="store_true", help="Use random CIFAR-shaped data for a code smoke test only")

    p.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--methods", type=str, default="clpl,clwl", help="Comma-separated subset of clpl,clwl")

    p.add_argument("--model", choices=["small_cnn", "resnet_small"], default="small_cnn")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="adamw")
    p.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eval_every", type=int, default=5)

    p.add_argument("--max_train", type=int, default=None, help="Optional stratified subset size for quick tests")
    p.add_argument("--max_test", type=int, default=None, help="Optional stratified subset size for quick tests")
    p.add_argument("--loss_plot_rhos", type=float, nargs="*", default=[0.0, 0.3, 0.6])
    p.add_argument("--diagnose_only", action="store_true", help="Only compute M/T diagnostics; no data loading or training")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in {"clpl", "clwl"}:
            raise ValueError(f"Unknown method {m}")

    t0 = time.time()

    # Weak-channel diagnostics are independent of the image data.
    diag_rows = []
    weak_cache = {}
    for rho in args.rhos:
        pairs, Z, M, pi = build_weak_model(rho)
        T, A, alpha, q = construct_T(M)
        lam_hat, relres = order_preserving_residual(A)
        dom = dominance_diagnostics(M, Z, pairs)
        diag = {
            "rho": rho,
            "rankM": int(np.linalg.matrix_rank(M)),
            "cond_MtM": float(np.linalg.cond(M.T @ M)),
            "alpha": float(alpha),
            "lambda_hat": lam_hat,
            "relres_TM_to_standard_form": relres,
            "T_min": float(T.min()),
            "T_max": float(T.max()),
            **dom,
        }
        diag_rows.append(diag)
        weak_cache[rho] = (pairs, Z, M, T, A, alpha)
    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(out_dir / "diagnostics.csv", index=False)
    print("\nWEAK-CHANNEL DIAGNOSTICS")
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
        pairs, Z, M, T, A, alpha = weak_cache[rho]
        diag = diag_df[diag_df["rho"] == rho].iloc[0].to_dict()
        for seed in args.seeds:
            ztr = sample_weak_indices(ytr, M, seed + 12345 + int(round(1000 * rho)))
            for method in methods:
                T_method = T if method == "clwl" else None
                test_acc, train_acc, history = train_one(
                    args, Xtr, ytr, Xte, yte, ztr, Z, T_method, method, rho, seed
                )
                for h in history:
                    h.update({"dataset": "cifar10", "model": args.model, **diag})
                epoch_rows.extend(history)
                row = {
                    "dataset": "cifar10",
                    "model": args.model,
                    "rho": rho,
                    "seed": seed,
                    "method": method,
                    "test_acc": test_acc,
                    "train_acc": train_acc,
                    **diag,
                }
                rows.append(row)
                print("FINAL", row, flush=True)
                completed_runs = len(rows)
                summary = save_live_outputs(
                    rows, epoch_rows, out_dir, args, completed_runs=completed_runs, final=False
                )

    summary = save_live_outputs(
        rows, epoch_rows, out_dir, args, completed_runs=len(rows), final=True
    )

    print("\nSUMMARY")
    if summary.empty:
        print("No training runs were completed.")
    else:
        print(summary.to_string(index=False))
    print(f"\nSaved to {out_dir}. Runtime: {time.time() - t0:.1f}s")
    print(f"Saved plot: {out_dir / 'accuracy_vs_probability.pdf'}")
    print(f"Saved plot: {out_dir / 'loss_trend_over_epochs.pdf'}")


if __name__ == "__main__":
    main()
