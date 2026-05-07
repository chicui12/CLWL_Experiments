#!/usr/bin/env python3
"""
CIFAR-10 CLCL-native vs CLWL under non-complementary weak-label misspecification.

Experiment:
  True clean class y in {0,...,9}.
  Observed weak label z is sampled from
      M[z,y] = q                    if z == y,
             = (1-q)/(c-1)          otherwise.

Native CLCL treats z as a complementary label and minimizes CE(-f(x), z),
which is misspecified when q>0 because z equals the true label with nonzero
probability. CLWL uses the same transition matrix M and constructs T so that
T M has the standard ranking-consistent form.

This script deliberately avoids torchvision. It loads the official CIFAR-10
Python batches directly from data_root/cifar-10-batches-py/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

Array = np.ndarray


# -----------------------------
# Transition and diagnostics
# -----------------------------


def make_noncomplementary_M(c: int = 10, q: float = 0.4) -> Array:
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0,1]")
    M = np.full((c, c), (1.0 - q) / (c - 1), dtype=np.float64)
    np.fill_diagonal(M, q)
    return M


def construct_clwl_T(M: Array, safety_factor: float = 0.95) -> Array:
    """Construct nonnegative CLWL T by shifting a left inverse.

    T = alpha (N - 1 q^T), where N is a left inverse and q is the columnwise
    minimum of N. Then T >= 0 and T M = alpha I - alpha 1 q^T M, which has
    the standard order-preserving form when N M = I.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError("M must be a matrix")
    if M.min() < -1e-12:
        raise ValueError(f"M has negative entries: min={M.min()}")
    if not np.allclose(M.sum(axis=0), 1.0, atol=1e-8):
        raise ValueError("M columns must sum to one")
    N = np.linalg.pinv(M)  # c x d, left inverse when M has full column rank
    q = N.min(axis=0)
    shifted = N - np.ones((N.shape[0], 1)) @ q.reshape(1, -1)
    max_entry = float(shifted.max())
    alpha = safety_factor / max(max_entry, 1e-12)
    T = alpha * shifted
    T[np.abs(T) < 1e-12] = 0.0
    return T


def fit_standard_form(A: Array) -> Dict[str, float]:
    """Fit A ≈ lambda I + 1 v^T and return lambda/residual/margin."""
    A = np.asarray(A, dtype=np.float64)
    c = A.shape[0]
    X = np.zeros((c * c, 1 + c), dtype=np.float64)
    y = np.zeros(c * c, dtype=np.float64)
    n = 0
    for i in range(c):
        for j in range(c):
            X[n, 0] = 1.0 if i == j else 0.0
            X[n, 1 + j] = 1.0
            y[n] = A[i, j]
            n += 1
    sol = np.linalg.lstsq(X, y, rcond=None)[0]
    lam = float(sol[0])
    v = sol[1:]
    Ahat = lam * np.eye(c) + np.ones((c, 1)) @ v.reshape(1, -1)
    residual = float(np.linalg.norm(A - Ahat, ord="fro"))
    rel = residual / max(float(np.linalg.norm(A, ord="fro")), 1e-12)
    margin = float(min(A[j, j] - np.max(np.delete(A[:, j], j)) for j in range(c)))
    return {
        "lambda": lam,
        "relative_residual": rel,
        "ranking_margin": margin,
    }


def evaluate_A_on_vertices(A: Array) -> Dict[str, float]:
    """Order diagnostics on one-hot clean posteriors.

    Max preservation: whether argmax(A e_y) is y.
    Pairwise top-vs-rest: fraction of pairs (y,k) where (A e_y)_y > (A e_y)_k.
    """
    A = np.asarray(A, dtype=np.float64)
    c = A.shape[0]
    max_ok = 0
    pair_total = 0
    pair_ok = 0
    margins = []
    for y in range(c):
        col = A[:, y]
        max_ok += int(np.argmax(col) == y)
        for k in range(c):
            if k == y:
                continue
            pair_total += 1
            margin = float(col[y] - col[k])
            margins.append(margin)
            pair_ok += int(margin > 0)
    return {
        "vertex_max_preservation": max_ok / c,
        "vertex_pairwise_order": pair_ok / pair_total,
        "vertex_mean_margin": float(np.mean(margins)),
        "vertex_min_margin": float(np.min(margins)),
    }


def matrix_diagnostics(c: int, q_values: Sequence[float], safety_factor: float) -> pd.DataFrame:
    rows = []
    for q in q_values:
        M = make_noncomplementary_M(c=c, q=float(q))
        T_clwl = construct_clwl_T(M, safety_factor=safety_factor)
        T_clcl = np.ones((c, c), dtype=np.float64) - np.eye(c, dtype=np.float64)
        for name, T in [("CLCL_native", T_clcl), ("CLWL", T_clwl)]:
            A = T @ M
            row = {
                "q": float(q),
                "transform": name,
                **fit_standard_form(A),
                **evaluate_A_on_vertices(A),
                "T_min": float(T.min()),
                "T_max": float(T.max()),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def plot_diagnostics(diag: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.7), constrained_layout=True)
    for transform in ["CLCL_native", "CLWL"]:
        d = diag[diag["transform"] == transform].sort_values("q")
        x = d["q"].to_numpy(float)
        axes[0].plot(x, d["vertex_max_preservation"], marker="o", label=transform)
        axes[1].plot(x, d["vertex_pairwise_order"], marker="o", label=transform)
        axes[2].plot(x, d["ranking_margin"], marker="o", label=transform)
        axes[3].plot(x, d["relative_residual"], marker="o", label=transform)
    axes[0].set_title("Max preservation")
    axes[1].set_title("Pairwise order")
    axes[2].set_title("Ranking margin")
    axes[3].set_title("Standard-form residual")
    for ax in axes:
        ax.set_xlabel("q = P(z=y|y)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(out_dir / f"{prefix}_matrix_diagnostics.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{prefix}_matrix_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# CIFAR-10 loading
# -----------------------------

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def maybe_download_cifar10(data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    target_dir = data_root / "cifar-10-batches-py"
    if target_dir.exists():
        return
    tar_path = data_root / "cifar-10-python.tar.gz"
    if not tar_path.exists():
        print(f"Downloading {CIFAR_URL} to {tar_path}")
        urllib.request.urlretrieve(CIFAR_URL, tar_path)
    print(f"Extracting {tar_path} to {data_root}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(data_root)


def _unpickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def load_cifar10_batches(data_root: Path, download: bool = False) -> Tuple[Array, Array, Array, Array]:
    if download:
        maybe_download_cifar10(data_root)
    base = data_root / "cifar-10-batches-py"
    if not base.exists():
        raise FileNotFoundError(
            f"Could not find {base}. Download/extract CIFAR-10 Python version so that "
            "data_batch_1,...,data_batch_5,test_batch are under this directory, "
            "or rerun with --download."
        )
    xs, ys = [], []
    for i in range(1, 6):
        batch = _unpickle(base / f"data_batch_{i}")
        xs.append(batch["data"])
        ys.extend(batch["labels"])
    test = _unpickle(base / "test_batch")
    Xtr = np.concatenate(xs, axis=0).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    ytr = np.asarray(ys, dtype=np.int64)
    Xte = test["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    yte = np.asarray(test["labels"], dtype=np.int64)
    return Xtr, ytr, Xte, yte


def subsample_per_class(X: Array, y: Array, max_per_class: Optional[int], seed: int) -> Tuple[Array, Array]:
    if max_per_class is None:
        return X, y
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], y[keep]


def sample_weak_labels(y: Array, M: Array, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    z = np.empty(len(y), dtype=np.int64)
    for i, yi in enumerate(y):
        z[i] = rng.choice(M.shape[0], p=M[:, int(yi)])
    return z


# -----------------------------
# Torch model/training
# -----------------------------


def lazy_import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    return torch, nn, F, Dataset, DataLoader


class CifarArrayDataset:  # actual base class assigned after lazy import not needed
    def __init__(self, X: Array, y: Array, train: bool, augment: bool, seed: int = 0):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.train = train
        self.augment = augment
        self.seed = seed
        self.mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.train and self.augment:
            # Random crop with padding=4 and random horizontal flip.
            # Uses numpy's global RNG because DataLoader worker seeding is enough for this use.
            padded = np.pad(x, ((0, 0), (4, 4), (4, 4)), mode="reflect")
            top = np.random.randint(0, 9)
            left = np.random.randint(0, 9)
            x = padded[:, top:top + 32, left:left + 32]
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy()
        x = (x - self.mean) / self.std
        return x, int(self.y[idx])


def make_model(model_name: str, num_classes: int = 10):
    torch, nn, F, Dataset, DataLoader = lazy_import_torch()

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(256, num_classes)
        def forward(self, x):
            return self.classifier(self.features(x).flatten(1))

    class BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, in_planes, planes, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )
        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)
            return F.relu(out)

    class SmallResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_planes = 32
            self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.layer1 = self._make_layer(32, 2, stride=1)
            self.layer2 = self._make_layer(64, 2, stride=2)
            self.layer3 = self._make_layer(128, 2, stride=2)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(128, num_classes)
        def _make_layer(self, planes, blocks, stride):
            strides = [stride] + [1] * (blocks - 1)
            layers = []
            for st in strides:
                layers.append(BasicBlock(self.in_planes, planes, st))
                self.in_planes = planes
            return nn.Sequential(*layers)
        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.pool(out).flatten(1)
            return self.fc(out)

    if model_name == "small_cnn":
        return SmallCNN()
    if model_name == "resnet_small":
        return SmallResNet()
    raise ValueError(f"Unknown model {model_name}")


def evaluate_model(model, loader, device: str) -> Dict[str, float]:
    torch, nn, F, Dataset, DataLoader = lazy_import_torch()
    model.eval()
    correct = 0
    total = 0
    pair_ok = 0
    pair_total = 0
    margins = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
            # one-hot posterior top-vs-rest order
            for k in range(logits.shape[1]):
                mask = (yb != k)
                if mask.any():
                    margin = logits[mask, yb[mask]] - logits[mask, k]
                    pair_ok += int((margin > 0).sum().item())
                    pair_total += int(margin.numel())
                    margins.append(margin.detach().cpu())
    if margins:
        mean_margin = float(torch.cat(margins).mean().item())
    else:
        mean_margin = float("nan")
    return {
        "clean_accuracy": correct / max(total, 1),
        "top_vs_rest_pairwise_order": pair_ok / max(pair_total, 1),
        "top_vs_rest_margin": mean_margin,
    }


def train_one_method(
    method: str,
    Xtr: Array,
    ztr: Array,
    Xval: Array,
    yval: Array,
    Xte: Array,
    yte: Array,
    M_or_T: Array,
    args,
    seed: int,
) -> Dict[str, float]:
    torch, nn, F, Dataset, DataLoader = lazy_import_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = args.device

    train_ds = CifarArrayDataset(Xtr, ztr, train=True, augment=args.augment, seed=seed)
    val_ds = CifarArrayDataset(Xval, yval, train=False, augment=False, seed=seed)
    test_ds = CifarArrayDataset(Xte, yte, train=False, augment=False, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.startswith("cuda")))
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    model = make_model(args.model, num_classes=args.num_classes).to(device)
    if args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    else:
        raise ValueError(args.optimizer)

    M_or_T_t = torch.tensor(M_or_T.astype(np.float32), dtype=torch.float32, device=device)
    best_state = None
    best_val = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, zb in train_loader:
            xb = xb.to(device)
            zb = zb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            if method == "clcl_native":
                # Native complementary-label loss: observed z is assumed to be a class to suppress.
                loss = F.cross_entropy(-logits, zb)
            elif method == "clwl":
                T = M_or_T_t
                target = T[:, zb].T
                loss = (target * F.softplus(-logits) + (1.0 - target) * F.softplus(logits)).sum(dim=1).mean()
            elif method == "forward":
                M = M_or_T_t
                pc = torch.softmax(logits, dim=1)
                pz = torch.clamp(pc @ M.T, min=1e-8)
                pz = pz / pz.sum(dim=1, keepdim=True)
                loss = F.nll_loss(torch.log(pz), zb)
            else:
                raise ValueError(method)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        sched.step()
        val_metrics = evaluate_model(model, val_loader, device=device)
        if val_metrics["clean_accuracy"] > best_val:
            best_val = val_metrics["clean_accuracy"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if args.verbose and (epoch == 1 or epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs):
            print(f"[{method}] epoch {epoch}/{args.epochs} val_acc={val_metrics['clean_accuracy']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_model(model, test_loader, device=device)
    test_metrics.update({"best_val_accuracy": best_val, "best_epoch": best_epoch})
    return test_metrics


def split_train_val(X: Array, y: Array, val_frac: float, seed: int) -> Tuple[Array, Array, Array, Array]:
    rng = np.random.default_rng(seed)
    tr_idx, va_idx = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_val = int(round(val_frac * len(idx)))
        va_idx.append(idx[:n_val])
        tr_idx.append(idx[n_val:])
    tr_idx = np.concatenate(tr_idx)
    va_idx = np.concatenate(va_idx)
    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)
    return X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]


def run_training(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr_all, ytr_all, Xte, yte = load_cifar10_batches(Path(args.data_root), download=args.download)
    Xtr_all, ytr_all = subsample_per_class(Xtr_all, ytr_all, args.max_train_per_class, seed=123)
    Xte, yte = subsample_per_class(Xte, yte, args.max_test_per_class, seed=124)

    c = args.num_classes
    M = make_noncomplementary_M(c=c, q=args.q)
    T_clwl = construct_clwl_T(M, safety_factor=args.safety_factor)
    T_clcl = np.ones((c, c), dtype=np.float64) - np.eye(c, dtype=np.float64)
    diag = matrix_diagnostics(c=c, q_values=[args.q], safety_factor=args.safety_factor)
    diag.to_csv(out_dir / "matrix_diagnostics.csv", index=False)
    plot_diagnostics(matrix_diagnostics(c=c, q_values=args.q_grid, safety_factor=args.safety_factor), out_dir, "cifar10_clcl_noncomp")
    np.save(out_dir / "M_noncomplementary.npy", M)
    np.save(out_dir / "T_clwl.npy", T_clwl)

    rows = []
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for seed in args.train_seeds:
        Xtv, ytv = Xtr_all, ytr_all
        Xtr, ytr, Xval, yval = split_train_val(Xtv, ytv, args.val_frac, seed)
        ztr = sample_weak_labels(ytr, M, seed=10_000 + seed)
        for method in methods:
            if method == "clcl_native":
                obj = T_clcl  # unused by loss, but kept for consistency
            elif method == "clwl":
                obj = T_clwl
            elif method == "forward":
                obj = M
            else:
                raise ValueError(f"Unknown method {method}")
            metrics = train_one_method(method, Xtr, ztr, Xval, yval, Xte, yte, obj, args, seed=20_000 + seed + hash(method) % 1000)
            rows.append({"seed": seed, "method": method, "q": args.q, **metrics})
            print({"seed": seed, "method": method, **metrics})

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "raw_results.csv", index=False)
    metric_cols = ["clean_accuracy", "top_vs_rest_pairwise_order", "top_vs_rest_margin", "best_val_accuracy", "best_epoch"]
    summ = raw.groupby(["method", "q"], as_index=False)[metric_cols].agg(["mean", "std"]).reset_index()
    summ.columns = ["_".join([str(x) for x in col if x]) for col in summ.columns]
    summ.to_csv(out_dir / "summary_results.csv", index=False)
    print("=== Summary ===")
    print(summ.to_string(index=False))
    plot_results(summ, out_dir)

    meta = {
        "dataset": "CIFAR-10",
        "classes": CIFAR_CLASSES,
        "q": args.q,
        "num_train_total_after_subsample": int(len(ytr_all)),
        "num_test_after_subsample": int(len(yte)),
        "args": vars(args),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def plot_results(summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    # One q value in normal use: bar plot.
    methods = list(summary["method"].values)
    acc = summary["clean_accuracy_mean"].to_numpy(float)
    acc_std = np.nan_to_num(summary["clean_accuracy_std"].to_numpy(float), nan=0.0)
    pair = summary["top_vs_rest_pairwise_order_mean"].to_numpy(float)
    pair_std = np.nan_to_num(summary["top_vs_rest_pairwise_order_std"].to_numpy(float), nan=0.0)

    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].bar(x, acc, yerr=acc_std, capsize=4)
    axes[0].set_xticks(x, methods, rotation=20, ha="right")
    axes[0].set_ylabel("clean test accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, pair, yerr=pair_std, capsize=4)
    axes[1].set_xticks(x, methods, rotation=20, ha="right")
    axes[1].set_ylabel("top-vs-rest pairwise order")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.savefig(out_dir / "results.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "results.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--out_dir", default="artifacts_cifar10_clcl_noncomp")
    p.add_argument("--diagnostic_only", action="store_true")
    p.add_argument("--q", type=float, default=0.4)
    p.add_argument("--q_grid", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument("--safety_factor", type=float, default=0.95)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--model", choices=["small_cnn", "resnet_small"], default="small_cnn")
    p.add_argument("--methods", default="clcl_native,clwl,forward")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--train_seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_train_per_class", type=int, default=None)
    p.add_argument("--max_test_per_class", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = matrix_diagnostics(c=args.num_classes, q_values=args.q_grid, safety_factor=args.safety_factor)
    diag.to_csv(out_dir / "matrix_diagnostics_grid.csv", index=False)
    plot_diagnostics(diag, out_dir, "cifar10_clcl_noncomp")
    print("=== Matrix diagnostics ===")
    print(diag.to_string(index=False))
    if args.diagnostic_only:
        return
    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("[warning] CUDA unavailable; falling back to CPU")
                args.device = "cpu"
        except Exception:
            args.device = "cpu"
    run_training(args)


if __name__ == "__main__":
    main()
