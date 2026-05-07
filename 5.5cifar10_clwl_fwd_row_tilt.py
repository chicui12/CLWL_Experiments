#!/usr/bin/env python3
"""
CIFAR-10 FWD-vs-CLWL row-tilt experiment (fixed, CIFAR-aware version).

Why this version exists
-----------------------
The previous CIFAR utility was only a plotting helper / smoke test. This script is
self-contained: it (i) reads the official CIFAR-10 python batches directly, (ii)
constructs a CIFAR-aware M_true and row-tilt H using semantic class pairs, (iii)
checks the matrix diagnostics, and (iv) trains Forward(M_true), Forward(Mhat), and
CLWL(T(Mhat)) on the same weak labels.

CIFAR-10 class order in the official python batches:
  0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer,
  5 dog, 6 frog, 7 horse, 8 ship, 9 truck.

Weak response modes for each class k (d = 5c = 50):
  k_H: high-confidence response for class k
  k_L: low-confidence response for class k
  k_A: ambiguous response for class k
  k_F: fallback/style response for class k
  k_S: secondary/style response for class k

Semantically plausible CIFAR-10 confusion pairs:
  airplane <-> ship, automobile <-> truck, bird <-> frog,
  cat <-> dog, deer <-> horse.

Protocol:
  weak labels are sampled from M_true;
  learners receive Mhat_s(z|y) ∝ M_true(z|y) exp(s H[z,y]);
  compare Forward(M_true), Forward(Mhat_s), and CLWL(T(Mhat_s)).

Examples
--------
# matrix diagnostics only (no CIFAR data needed)
python cifar10_clwl_fwd_row_tilt_fixed.py --diagnostic_only

# quick GPU check on a subset once CIFAR-10 is extracted under ./data/cifar-10-batches-py
python cifar10_clwl_fwd_row_tilt_fixed.py \
  --data_root ./data --model small_cnn --epochs 20 --train_seeds 0 \
  --s_grid 0 0.5 0.75 1.0 --max_train_per_class 1000 --max_test_per_class 300 \
  --device cuda --out_dir artifacts_cifar10_quick

# final-ish run
python cifar10_clwl_fwd_row_tilt_fixed.py \
  --data_root ./data --model resnet_small --epochs 100 --train_seeds 0 1 2 \
  --s_grid 0 0.25 0.5 0.75 1.0 --batch_size 128 --device cuda \
  --out_dir artifacts_cifar10_final
"""
from __future__ import annotations

import argparse
import json
import pickle
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

Array = np.ndarray
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
PAIR = {0: 8, 8: 0, 1: 9, 9: 1, 2: 6, 6: 2, 3: 5, 5: 3, 4: 7, 7: 4}


# ---------------------------------------------------------------------
# Transition construction
# ---------------------------------------------------------------------

def validate_transition(M: Array, name: str = "M", atol: float = 1e-8) -> None:
    if M.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {M.shape}")
    if np.min(M) < -atol:
        raise ValueError(f"{name} has negative entries: min={np.min(M)}")
    col_sums = M.sum(axis=0)
    if not np.allclose(col_sums, np.ones(M.shape[1]), atol=atol, rtol=0.0):
        raise ValueError(f"{name} is not column-stochastic: {col_sums}")


def make_cifar10_mtrue_and_h() -> tuple[Array, Array, dict[int, int]]:
    """Search-selected CIFAR-aware M_true and H.

    These parameters were selected by matrix-level search, not by test accuracy.
    At s=1.0 the forward pure-class diagnostic flips to the semantic paired
    confuser, while T(Mhat_s) M_true still has low standard-form residual and
    positive ranking margin.
    """
    c = 10
    modes = 5
    d = c * modes

    # Search-selected masses. Correct-class responses are concentrated on H;
    # paired-confuser responses are concentrated on ambiguous/low/fallback modes.
    p = np.array([
        0.6423750500, 0.0235579623, 0.0051567212, 0.0000339750, 0.0255183118
    ], dtype=np.float64)
    q = np.array([
        0.0272655200, 0.0396841300, 0.0836966300, 0.0510831500, 0.0316285500
    ], dtype=np.float64)
    restA = 0.7492084193104684
    restF = 0.2007915806895315

    true_penalty = np.array([
        5.21632422, 1.20728320, 0.86530423, 0.99844232, 1.38715957
    ], dtype=np.float64)
    pair_boost = np.array([
        1.18531610, 3.82570716, 6.70142683, 3.31159486, 2.63283157
    ], dtype=np.float64)
    otherA = 0.022667476345551208
    otherF = 0.053658532464826485

    M = np.zeros((d, c), dtype=np.float64)
    H = np.zeros((d, c), dtype=np.float64)

    for y in range(c):
        r = PAIR[y]
        for m, val in enumerate(p):
            M[modes * y + m, y] += val
        for m, val in enumerate(q):
            M[modes * r + m, y] += val
        rem = 1.0 - M[:, y].sum()
        if rem < -1e-12:
            raise ValueError("Mass parameters sum to more than one.")
        others = [j for j in range(c) if j not in (y, r)]
        for j in others:
            M[modes * j + 2, y] += rem * restA / len(others)
            M[modes * j + 3, y] += rem * restF / len(others)
            M[modes * j + 1, y] += rem * (1.0 - restA - restF) / len(others)

        for m, val in enumerate(true_penalty):
            H[modes * y + m, y] -= val
        for m, val in enumerate(pair_boost):
            H[modes * r + m, y] += val
        for j in range(c):
            if j not in (y, r):
                H[modes * j + 2, y] += otherA
                H[modes * j + 3, y] += otherF

    H = H - H.mean(axis=0, keepdims=True)
    validate_transition(M, "M_true")
    return M, H, PAIR.copy()


def row_tilt(M_true: Array, H: Array, s: float) -> Array:
    L = np.log(np.clip(M_true, 1e-12, 1.0)) + float(s) * H
    L = L - L.max(axis=0, keepdims=True)
    E = np.exp(L)
    M_hat = E / E.sum(axis=0, keepdims=True)
    validate_transition(M_hat, "M_hat")
    return M_hat


def construct_clwl_T(M: Array, safety_factor: float = 0.99, ridge: float = 1e-9) -> Array:
    M = np.asarray(M, dtype=np.float64)
    d, c = M.shape
    N = np.linalg.solve(M.T @ M + ridge * np.eye(c), M.T)  # c x d
    q = N.min(axis=0)
    shifted = N - np.ones((c, 1), dtype=np.float64) @ q.reshape(1, -1)
    ranges = N.max(axis=0) - N.min(axis=0)
    alpha = safety_factor / max(float(ranges.max()), 1e-12)
    T = alpha * shifted
    # This should be in [0,1] by construction; clipping only protects roundoff.
    return np.clip(T, 0.0, 1.0)


def standard_form_fit(A: Array) -> dict[str, float]:
    A = np.asarray(A, dtype=np.float64)
    c = A.shape[0]
    v = (A.sum(axis=0) - np.diag(A)) / max(c - 1, 1)
    lam = float(np.mean(np.diag(A) - v))
    A_hat = lam * np.eye(c) + np.ones((c, 1), dtype=np.float64) @ v.reshape(1, -1)
    residual = float(np.linalg.norm(A - A_hat, ord="fro"))
    relative = residual / max(float(np.linalg.norm(A, ord="fro")), 1e-12)
    margin = float(min(A[j, j] - np.delete(A[:, j], j).max() for j in range(c)))
    return {"lambda_hat": lam, "relative_residual": relative, "ranking_margin": margin}


def forward_pure_class_diagnostic(M_true: Array, M_hat: Array, pair: dict[int, int]) -> dict[str, object]:
    c = M_true.shape[1]
    CE = -(M_true.T @ np.log(np.clip(M_hat, 1e-12, 1.0)))  # true y x predicted k
    pred = CE.argmin(axis=1)
    scores = -CE
    scores = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=1, keepdims=True)
    true_prob = float(np.mean(probs[np.arange(c), np.arange(c)]))
    pair_prob = float(np.mean([probs[y, pair[y]] for y in range(c)]))
    pair_ce_margin = float(np.mean([CE[y, pair[y]] - CE[y, y] for y in range(c)]))
    return {
        "forward_pure_acc": float(np.mean(pred == np.arange(c))),
        "forward_true_prob": true_prob,
        "forward_pair_prob": pair_prob,
        "forward_pair_ce_margin": pair_ce_margin,
        "forward_pure_pred": pred.tolist(),
    }


def compute_diagnostics(M_true: Array, H: Array, s_grid: list[float], pair: dict[int, int]) -> pd.DataFrame:
    rows = []
    for s in s_grid:
        M_hat = row_tilt(M_true, H, s)
        T_hat = construct_clwl_T(M_hat)
        A = T_hat @ M_true
        rows.append({"s": float(s), **standard_form_fit(A), **forward_pure_class_diagnostic(M_true, M_hat, pair)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# CIFAR-10 loader
# ---------------------------------------------------------------------

def _unpickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def maybe_download_cifar10(data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    target = data_root / "cifar-10-python.tar.gz"
    extracted = data_root / "cifar-10-batches-py"
    if extracted.exists():
        return
    if not target.exists():
        print(f"Downloading CIFAR-10 from {CIFAR10_URL} ...")
        urllib.request.urlretrieve(CIFAR10_URL, target)
    print(f"Extracting {target} ...")
    with tarfile.open(target, "r:gz") as tar:
        tar.extractall(data_root)


def find_cifar_dir(data_root: Path) -> Path:
    candidates = [
        data_root / "cifar-10-batches-py",
        data_root,
        Path("./data/cifar-10-batches-py"),
        Path("./cifar-10-batches-py"),
    ]
    for c in candidates:
        if (c / "data_batch_1").exists() and (c / "test_batch").exists():
            return c
    raise FileNotFoundError(
        "Could not find CIFAR-10 python batches. Expected files data_batch_1,...,test_batch "
        "under --data_root/cifar-10-batches-py. Use --download on a machine with internet."
    )


def load_cifar10(data_root: str, download: bool = False) -> tuple[Array, Array, Array, Array]:
    root = Path(data_root)
    if download:
        maybe_download_cifar10(root)
    cifar_dir = find_cifar_dir(root)
    xs, ys = [], []
    for k in range(1, 6):
        d = _unpickle(cifar_dir / f"data_batch_{k}")
        xs.append(d["data"])
        ys.extend(d["labels"])
    Xtr = np.concatenate(xs, axis=0).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    ytr = np.asarray(ys, dtype=np.int64)
    d = _unpickle(cifar_dir / "test_batch")
    Xte = d["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    yte = np.asarray(d["labels"], dtype=np.int64)
    return Xtr, ytr, Xte, yte


def balanced_subsample_indices(y: Array, max_per_class: Optional[int], seed: int) -> Array:
    if max_per_class is None:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    keep = []
    for cls in sorted(np.unique(y)):
        idx = np.flatnonzero(y == cls)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return keep


def split_train_val_indices(y: Array, val_frac: float, seed: int) -> tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in sorted(np.unique(y)):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        nv = int(round(val_frac * len(idx)))
        val_idx.append(idx[:nv])
        train_idx.append(idx[nv:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def sample_weak_labels(y: Array, M_true: Array, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    z = np.empty(len(y), dtype=np.int64)
    for i, yi in enumerate(y):
        z[i] = int(rng.choice(M_true.shape[0], p=M_true[:, int(yi)]))
    return z


class CifarWeakDataset(Dataset):
    def __init__(self, X: Array, labels: Array, augment: bool = False):
        self.X = X
        self.labels = labels.astype(np.int64)
        self.augment = bool(augment)
        # CIFAR-10 common normalization; exact values are not critical for this experiment.
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        self.std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = torch.tensor(self.X[i], dtype=torch.float32)
        if self.augment:
            x = F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
            top = torch.randint(0, 9, ()).item()
            left = torch.randint(0, 9, ()).item()
            x = x[:, top:top+32, left:left+32]
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[2])
        x = (x - self.mean) / self.std
        return x, int(self.labels[i])


# ---------------------------------------------------------------------
# Models and losses
# ---------------------------------------------------------------------

class SmallCNN(nn.Module):
    def __init__(self, c: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, c)
    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
        else:
            self.skip = nn.Identity()
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x), inplace=True)


class ResNetSmall(nn.Module):
    def __init__(self, c: int = 10, width: int = 32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(BasicBlock(width, width), BasicBlock(width, width))
        self.layer2 = nn.Sequential(BasicBlock(width, 2*width, stride=2), BasicBlock(2*width, 2*width))
        self.layer3 = nn.Sequential(BasicBlock(2*width, 4*width, stride=2), BasicBlock(4*width, 4*width))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4*width, c)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def make_model(model_name: str) -> nn.Module:
    if model_name == "small_cnn":
        return SmallCNN(10)
    if model_name == "resnet_small":
        return ResNetSmall(10, width=32)
    raise ValueError(f"Unknown model {model_name}")


def clwl_loss(logits: torch.Tensor, z: torch.Tensor, T_t: torch.Tensor) -> torch.Tensor:
    targets = T_t[:, z].T.clamp(0.0, 1.0)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


def forward_loss(logits: torch.Tensor, z: torch.Tensor, M_t: torch.Tensor) -> torch.Tensor:
    p_clean = torch.softmax(logits, dim=1)
    p_weak = torch.clamp(p_clean @ M_t.T, min=1e-8)
    p_weak = p_weak / p_weak.sum(dim=1, keepdim=True)
    return F.nll_loss(torch.log(p_weak), z)


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    dev = torch.device(device)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(dev)
            pred = model(xb).argmax(dim=1).cpu()
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
    return correct / max(total, 1)


def train_one(
    Xtr: Array, ytr_or_ztr: Array, Xval: Array, yval: Array, Xte: Array, yte: Array,
    *, method: str, M_hat: Array, T_hat: Array, seed: int, model_name: str, epochs: int,
    batch_size: int, lr: float, weight_decay: float, device: str, augment: bool,
) -> dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    model = make_model(model_name).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    train_ds = CifarWeakDataset(Xtr, ytr_or_ztr, augment=augment)
    val_ds = CifarWeakDataset(Xval, yval, augment=False)
    test_ds = CifarWeakDataset(Xte, yte, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    M_t = torch.tensor(M_hat, dtype=torch.float32, device=dev)
    T_t = torch.tensor(T_hat, dtype=torch.float32, device=dev)

    best_state = None
    best_val = -1.0
    best_epoch = 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, zb in train_loader:
            xb = xb.to(dev)
            zb = zb.to(dev)
            logits = model(xb)
            if method == "forward":
                loss = forward_loss(logits, zb, M_t)
            elif method == "clwl":
                loss = clwl_loss(logits, zb, T_t)
            else:
                raise ValueError(method)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    return {"test_accuracy": float(test_acc), "best_val_accuracy": float(best_val), "best_epoch": int(best_epoch)}


# ---------------------------------------------------------------------
# Experiment runner and plots
# ---------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s_grid = [float(x) for x in args.s_grid]

    M_true, H, pair = make_cifar10_mtrue_and_h()
    np.save(out / "M_true.npy", M_true)
    np.save(out / "H.npy", H)
    diag = compute_diagnostics(M_true, H, s_grid, pair)
    diag.to_csv(out / "diagnostics.csv", index=False)
    print("=== matrix diagnostics ===")
    print(diag.to_string(index=False))

    if args.diagnostic_only:
        plot_results(None, diag, out, args.title)
        meta = {"class_names": CIFAR10_CLASSES, "pair": {CIFAR10_CLASSES[k]: CIFAR10_CLASSES[v] for k, v in pair.items()}, "s_grid": s_grid}
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved diagnostics to {out.resolve()}")
        return

    Xtr_all, ytr_all, Xte_all, yte_all = load_cifar10(args.data_root, download=args.download)
    test_idx = balanced_subsample_indices(yte_all, args.max_test_per_class, seed=999)
    Xte, yte = Xte_all[test_idx], yte_all[test_idx]

    rows = []
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)

    for split_seed in args.train_seeds:
        trainval_idx = balanced_subsample_indices(ytr_all, args.max_train_per_class, seed=1000 + split_seed)
        Xtv, ytv = Xtr_all[trainval_idx], ytr_all[trainval_idx]
        tr_idx, val_idx = split_train_val_indices(ytv, args.val_frac, seed=2000 + split_seed)
        Xtr, ytr = Xtv[tr_idx], ytv[tr_idx]
        Xval, yval = Xtv[val_idx], ytv[val_idx]
        ztr = sample_weak_labels(ytr, M_true, seed=3000 + split_seed)

        # Oracle forward with M_true.
        oracle = train_one(
            Xtr, ztr, Xval, yval, Xte, yte,
            method="forward", M_hat=M_true, T_hat=construct_clwl_T(M_true),
            seed=4000 + split_seed, model_name=args.model, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
            device=device, augment=args.augment,
        )

        for s in s_grid:
            M_hat = row_tilt(M_true, H, s)
            T_hat = construct_clwl_T(M_hat)
            clwl = train_one(
                Xtr, ztr, Xval, yval, Xte, yte,
                method="clwl", M_hat=M_hat, T_hat=T_hat,
                seed=5000 + split_seed, model_name=args.model, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                device=device, augment=args.augment,
            )
            fwd = train_one(
                Xtr, ztr, Xval, yval, Xte, yte,
                method="forward", M_hat=M_hat, T_hat=T_hat,
                seed=6000 + split_seed, model_name=args.model, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                device=device, augment=args.augment,
            )
            rows.append({"method": "CLWL_T_Mhat", "seed": split_seed, "s": s, **clwl})
            rows.append({"method": "Forward_Mhat", "seed": split_seed, "s": s, **fwd})
            rows.append({"method": "Forward_oracle_Mtrue", "seed": split_seed, "s": s, **oracle})

    raw = pd.DataFrame(rows)
    raw.to_csv(out / "raw_results.csv", index=False)
    summary = raw.groupby(["method", "s"], as_index=False).agg(
        test_accuracy_mean=("test_accuracy", "mean"),
        test_accuracy_std=("test_accuracy", "std"),
        best_val_accuracy_mean=("best_val_accuracy", "mean"),
        best_epoch_mean=("best_epoch", "mean"),
    )
    summary.to_csv(out / "summary_results.csv", index=False)
    print("=== summary ===")
    print(summary.to_string(index=False))
    plot_results(summary, diag, out, args.title)

    meta = {
        "class_names": CIFAR10_CLASSES,
        "pair": {CIFAR10_CLASSES[k]: CIFAR10_CLASSES[v] for k, v in pair.items()},
        "num_train_all": int(len(ytr_all)),
        "num_test_used": int(len(yte)),
        "config": vars(args),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved to {out.resolve()}")


def plot_results(summary: Optional[pd.DataFrame], diag: pd.DataFrame, out: Path, title: str) -> None:
    if plt is None:
        return
    if summary is not None and len(summary):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for method in ["CLWL_T_Mhat", "Forward_Mhat", "Forward_oracle_Mtrue"]:
            df = summary[summary.method == method].sort_values("s")
            if df.empty:
                continue
            x = df.s.to_numpy(float)
            y = df.test_accuracy_mean.to_numpy(float)
            e = np.nan_to_num(df.test_accuracy_std.to_numpy(float), nan=0.0)
            ax[0].plot(x, y, marker="o", label=method)
            ax[0].fill_between(x, y - e, y + e, alpha=0.15)
        ax[0].set_title(title)
        ax[0].set_xlabel("estimate-bias strength $s$")
        ax[0].set_ylabel("clean test accuracy")
        ax[0].set_ylim(0, 1)
        ax[0].grid(True, alpha=0.3)
        ax[0].legend()
        cl = summary[summary.method == "CLWL_T_Mhat"].sort_values("s")
        fw = summary[summary.method == "Forward_Mhat"].sort_values("s")
        if not cl.empty and not fw.empty:
            ax[1].plot(cl.s, cl.test_accuracy_mean.to_numpy(float) - fw.test_accuracy_mean.to_numpy(float), marker="o")
            ax[1].axhline(0, linestyle="--", alpha=0.7)
        ax[1].set_xlabel("estimate-bias strength $s$")
        ax[1].set_ylabel("CLWL $-$ Forward")
        ax[1].grid(True, alpha=0.3)
        fig.savefig(out / "results.png", dpi=200, bbox_inches="tight")
        fig.savefig(out / "results.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
    x = diag.s.to_numpy(float)
    ax[0].plot(x, diag.lambda_hat, marker="o")
    ax[0].axhline(0, linestyle="--", alpha=0.7)
    ax[0].set_title("$\\lambda$ of $T(\\hat M_s)M_\\star$")
    ax[1].plot(x, diag.relative_residual, marker="o")
    ax[1].set_title("standard-form rel. residual")
    ax[2].plot(x, diag.ranking_margin, marker="o")
    ax[2].axhline(0, linestyle="--", alpha=0.7)
    ax[2].set_title("CLWL ranking margin")
    ax[3].plot(x, diag.forward_true_prob, marker="o", label="true class")
    ax[3].plot(x, diag.forward_pair_prob, marker="o", label="paired confuser")
    ax[3].axhline(0.5, linestyle="--", alpha=0.7)
    ax[3].set_title("Forward population preference")
    ax[3].legend()
    for a in ax:
        a.set_xlabel("estimate-bias strength $s$")
        a.grid(True, alpha=0.3)
    fig.savefig(out / "diagnostics.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--diagnostic_only", action="store_true")
    p.add_argument("--out_dir", type=str, default="artifacts_cifar10_clwl_fwd_row_tilt_fixed")
    p.add_argument("--title", type=str, default="CIFAR-10 row-tilt misspecification")
    p.add_argument("--model", choices=["small_cnn", "resnet_small"], default="small_cnn")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--train_seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--s_grid", nargs="+", type=float, default=[0, 0.25, 0.5, 0.75, 1.0, 1.1])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--max_train_per_class", type=int, default=None)
    p.add_argument("--max_test_per_class", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--augment", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
