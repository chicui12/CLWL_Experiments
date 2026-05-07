from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms

from clwl_experiments_module1_t_construction import construct_clwl_T
from clwl_experiments_module4_weak_label_dataset import WeakLabelDataset
from clwl_experiments_module6_metrics import evaluate_scores_on_dataset
from clwl_experiments_module7_clwl_training import (
    CLWLTrainConfig,
    train_clwl_model,
    evaluate_model_on_dataset as evaluate_clwl_model_on_dataset,
)

Array = np.ndarray

# ============================================================
# CLWL MNIST row-tilt response-bias experiment
# ============================================================
# Fixed true weak-label transition M_true; vary only the estimate M_hat_s.
# Weak labels: for each clean digit k, two weak responses:
#   z=2k:   high-confidence response for digit k
#   z=2k+1: low-confidence / ambiguous response for digit k
# Therefore d=2c. Default is MNIST-4 for speed; pass --classes 0 1 ... 9
# to run the full MNIST-10 version.


# ============================================================
# Dataset containers
# ============================================================

@dataclass
class RealCleanDataset:
    X: Array
    y: Array
    eta: Array
    logits: Array
    metadata: dict[str, Any]


def one_hot(y: Array, c: int) -> Array:
    y = np.asarray(y, dtype=np.intp)
    out = np.zeros((len(y), c), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


def safe_logits_from_onehot(eta: Array, eps: float = 1e-12) -> Array:
    return np.log(np.clip(eta, eps, 1.0))


def load_mnist_subset_splits(
    *,
    root: str,
    classes: Sequence[int],
    val_frac: float,
    seed: int,
    max_trainval_samples: Optional[int],
    max_test_samples: Optional[int],
    standardize: bool,
    download: bool = True,
) -> dict[str, RealCleanDataset]:
    classes = tuple(int(k) for k in classes)
    class_to_new = {old: new for new, old in enumerate(classes)}
    c = len(classes)

    transform = transforms.ToTensor()
    train_raw = MNIST(root=root, train=True, download=download, transform=transform)
    test_raw = MNIST(root=root, train=False, download=download, transform=transform)

    def extract(ds: MNIST, max_samples: Optional[int], seed_offset: int) -> tuple[Array, Array]:
        Xs, ys = [], []
        for img, label in ds:
            label = int(label)
            if label in class_to_new:
                Xs.append(img.numpy().reshape(-1).astype(np.float64))
                ys.append(class_to_new[label])
        X = np.stack(Xs, axis=0)
        y = np.asarray(ys, dtype=np.intp)
        if max_samples is not None and max_samples < len(y):
            rng = np.random.default_rng(seed + seed_offset)
            idx = rng.choice(len(y), size=max_samples, replace=False)
            X, y = X[idx], y[idx]
        return X, y

    X_trainval, y_trainval = extract(train_raw, max_trainval_samples, 0)
    X_test, y_test = extract(test_raw, max_test_samples, 1)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y_trainval))
    n_val = int(round(val_frac * len(idx)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
    X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]

    if standardize:
        mean = X_train.mean(axis=0, keepdims=True)
        std = np.maximum(X_train.std(axis=0, keepdims=True), 1e-8)
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std

    def make_ds(X: Array, y: Array, split: str) -> RealCleanDataset:
        eta = one_hot(y, c)
        return RealCleanDataset(
            X=np.asarray(X, dtype=np.float64),
            y=np.asarray(y, dtype=np.intp),
            eta=eta,
            logits=safe_logits_from_onehot(eta),
            metadata={
                "dataset_name": "MNIST",
                "classes_original": list(classes),
                "split": split,
                "eta_source": "one_hot_clean_label",
                "num_classes": c,
            },
        )

    print("=== MNIST split summary ===")
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(name, "n=", len(y), "counts=", np.bincount(y, minlength=c).tolist())

    return {
        "train": make_ds(X_train, y_train, "train"),
        "val": make_ds(X_val, y_val, "val"),
        "test": make_ds(X_test, y_test, "test"),
    }


# ============================================================
# Transition matrices and row-tilt estimate model
# ============================================================

def default_confusion_pairs(c: int) -> dict[int, int]:
    if c == 4:
        return {0: 2, 2: 0, 1: 3, 3: 1}
    if c == 10:
        return {0: 2, 2: 0, 1: 7, 7: 1, 3: 8, 8: 3, 4: 9, 9: 4, 5: 6, 6: 5}
    # Fallback: pair adjacent classes cyclically.
    return {k: (k + 1) % c for k in range(c)}


def validate_transition(M: Array, *, atol: float = 1e-8) -> None:
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"M must be 2D, got {M.shape}.")
    if np.min(M) < -atol:
        raise ValueError(f"M has negative entries: min={np.min(M):.3e}.")
    col_sums = M.sum(axis=0)
    if not np.allclose(col_sums, np.ones(M.shape[1]), atol=atol, rtol=0.0):
        raise ValueError(f"M must be column-stochastic. col_sums={col_sums}")


def make_mnist_weak_M_true(
    c: int,
    pairs: dict[int, int],
    *,
    true_high: float = 0.46,
    true_low: float = 0.20,
    conf_high: float = 0.14,
    conf_low: float = 0.08,
    other_low_total: float = 0.12,
) -> Array:
    """
    d=2c weak labels:
      2k   = high-confidence digit-k response
      2k+1 = low-confidence / ambiguous digit-k response

    For class y, most mass goes to true high/low responses, some to a visually
    paired confusion digit, and remaining mass to other low-confidence responses.
    """
    d = 2 * c
    M = np.zeros((d, c), dtype=np.float64)
    for y in range(c):
        p = pairs[y]
        M[2 * y, y] += true_high
        M[2 * y + 1, y] += true_low
        M[2 * p, y] += conf_high
        M[2 * p + 1, y] += conf_low
        others = [k for k in range(c) if k not in {y, p}]
        if others:
            for k in others:
                M[2 * k + 1, y] += other_low_total / len(others)
        else:
            M[2 * y + 1, y] += other_low_total
    validate_transition(M)
    return M


def row_tilt(M_true: Array, H: Array, s: float) -> Array:
    M = np.asarray(M_true, dtype=np.float64) * np.exp(float(s) * np.asarray(H, dtype=np.float64))
    M = M / M.sum(axis=0, keepdims=True)
    validate_transition(M)
    return M


def base_structured_H(c: int, pairs: dict[int, int]) -> Array:
    """
    A realistic pilot-bias template:
      - overestimate high-confidence responses for visually confused digits;
      - underestimate some low-confidence true-class responses;
      - overestimate shared low-confidence modes.
    This is used as a prior for random search.
    """
    H = np.zeros((2 * c, c), dtype=np.float64)
    for y in range(c):
        p = pairs[y]
        # Pilot overcounts confident wrong-pair responses for class y.
        H[2 * p, y] += 1.8
        H[2 * p + 1, y] += 0.8
        # Pilot undercounts ambiguous true-class responses.
        H[2 * y + 1, y] -= 1.2
        H[2 * y, y] -= 0.2
        # Mild bias on other low-confidence responses.
        for k in range(c):
            if k not in {y, p}:
                H[2 * k + 1, y] += 0.15
    return H


# ============================================================
# Diagnostics
# ============================================================

def standard_form_fit(A: Array) -> dict[str, float]:
    A = np.asarray(A, dtype=np.float64)
    c = A.shape[0]
    if A.shape != (c, c):
        raise ValueError(f"A must be square, got {A.shape}.")
    v = np.zeros(c, dtype=np.float64)
    for j in range(c):
        off = [A[i, j] for i in range(c) if i != j]
        v[j] = float(np.mean(off)) if off else 0.0
    lam = float(np.mean(np.diag(A) - v))
    A_hat = lam * np.eye(c) + np.ones((c, 1)) @ v.reshape(1, -1)
    residual = float(np.linalg.norm(A - A_hat, ord="fro"))
    relative = residual / max(float(np.linalg.norm(A, ord="fro")), 1e-12)
    return {"lambda_hat": lam, "residual": residual, "relative_residual": relative}


def project_simplex(v: Array) -> Array:
    """Euclidean projection onto probability simplex."""
    v = np.asarray(v, dtype=np.float64)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        return np.ones_like(v) / len(v)
    rho = rho[-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0.0)


def forward_proxy_projection(M_hat: Array, M_true: Array) -> Array:
    """
    Fast proxy for Forward Bayes projection:
      q_proxy_y = projection_simplex(pinv(M_hat) M_true[:,y]).
    Used only to optimize H. Exact training remains empirical.
    """
    B = np.linalg.pinv(M_hat) @ M_true
    Q = np.stack([project_simplex(B[:, y]) for y in range(B.shape[1])], axis=1)
    return Q


def diagnostic_path(M_true: Array, H: Array, s_grid: Sequence[float]) -> pd.DataFrame:
    rows = []
    c = M_true.shape[1]
    for s in s_grid:
        M_hat = row_tilt(M_true, H, s)
        T_hat = construct_clwl_T(M_hat).T
        A = T_hat @ M_true
        fit = standard_form_fit(A)
        Q = forward_proxy_projection(M_hat, M_true)
        proxy_acc = float(np.mean(np.argmax(Q, axis=0) == np.arange(c)))
        proxy_true_prob = float(np.mean(np.diag(Q)))
        rows.append({
            "s": float(s),
            "lambda": fit["lambda_hat"],
            "residual": fit["residual"],
            "relative_residual": fit["relative_residual"],
            "forward_proxy_class_recovery": proxy_acc,
            "forward_proxy_true_prob": proxy_true_prob,
        })
    return pd.DataFrame(rows)


def score_H(M_true: Array, H: Array, s_grid: Sequence[float], *, min_lambda: float, max_resid: float) -> float:
    df = diagnostic_path(M_true, H, s_grid)
    lam = df["lambda"].to_numpy()
    resid = df["relative_residual"].to_numpy()
    true_prob = df["forward_proxy_true_prob"].to_numpy()
    proxy_acc = df["forward_proxy_class_recovery"].to_numpy()

    # Hard feasibility over the whole path.
    if np.min(lam) < min_lambda:
        return float("-inf")

    if np.max(resid) > max_resid:
        return float("-inf")

    # Hard monotonicity of the Forward proxy.
    if np.any(np.diff(true_prob) > 0.0):
        return float("-inf")
    
    

    # We want Forward proxy to decline while CLWL lambda remains positive and residual controlled.
    damage = (true_prob[0] - true_prob[-1]) + 0.7 * (proxy_acc[0] - proxy_acc[-1])
    monotone_penalty = np.sum(np.maximum(0.0, np.diff(true_prob)))
    lambda_penalty = np.sum(np.maximum(0.0, min_lambda - lam))
    resid_penalty = np.sum(np.maximum(0.0, resid - max_resid))
    return float(damage - 2.0 * monotone_penalty - 3.0 * lambda_penalty - 1.5 * resid_penalty)


def optimize_H(
    M_true: Array,
    pairs: dict[int, int],
    *,
    s_grid: Sequence[float],
    seed: int,
    candidates: int,
    min_lambda: float,
    max_resid: float,
) -> tuple[Array, pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    c = M_true.shape[1]
    H0 = base_structured_H(c, pairs)

    best_H = H0.copy()
    best_score = score_H(M_true, best_H, s_grid, min_lambda=min_lambda, max_resid=max_resid)

    for _ in range(candidates):
        scale = rng.uniform(0.6, 2.2)
        noise = rng.normal(scale=rng.uniform(0.05, 0.45), size=H0.shape)
        # Keep the structured direction but allow per-entry deviations.
        H = scale * H0 + noise
        # Column-center H because adding a column constant has no effect after normalization.
        H = H - H.mean(axis=0, keepdims=True)
        sc = score_H(M_true, H, s_grid, min_lambda=min_lambda, max_resid=max_resid)
        if sc > best_score:
            best_H, best_score = H, sc

    diag = diagnostic_path(M_true, best_H, s_grid)
    return best_H, diag, best_score


# ============================================================
# Weak-label dataset construction
# ============================================================

def sample_weak_labels(y: Array, M_true: Array, *, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.intp)
    z = np.empty_like(y)
    for i, yi in enumerate(y):
        z[i] = int(rng.choice(M_true.shape[0], p=M_true[:, yi]))
    return z


def build_weak_dataset(clean: RealCleanDataset, M_true: Array, *, seed: int) -> WeakLabelDataset:
    c = M_true.shape[1]
    z = sample_weak_labels(clean.y, M_true, seed=seed)
    names = []
    for k in range(c):
        names.append(f"high_digit_{k}")
        names.append(f"low_digit_{k}")
    return WeakLabelDataset(
        X=np.asarray(clean.X, dtype=np.float64).copy(),
        y=np.asarray(clean.y, dtype=np.intp).copy(),
        eta=np.asarray(clean.eta, dtype=np.float64).copy(),
        logits=np.asarray(clean.logits, dtype=np.float64).copy(),
        z=z,
        M=np.asarray(M_true, dtype=np.float64).copy(),
        family_name="mnist_row_tilt_fixed_Mtrue",
        weak_label_matrix=None,
        weak_label_vectors=None,
        weak_label_names=names,
        metadata={**clean.metadata, "M_true": np.asarray(M_true).tolist()},
    )


def build_weak_splits(clean_splits: dict[str, RealCleanDataset], M_true: Array, *, seed: int) -> dict[str, WeakLabelDataset]:
    return {
        "train": build_weak_dataset(clean_splits["train"], M_true, seed=seed + 0),
        "val": build_weak_dataset(clean_splits["val"], M_true, seed=seed + 1000),
        "test": build_weak_dataset(clean_splits["test"], M_true, seed=seed + 2000),
    }


# ============================================================
# Rectangular Forward trainer
# ============================================================

class TorchWeakDataset(Dataset):
    def __init__(self, ds: WeakLabelDataset) -> None:
        self.X = torch.tensor(np.asarray(ds.X, dtype=np.float32), dtype=torch.float32)
        self.z = torch.tensor(np.asarray(ds.z, dtype=np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.z[idx]


class MLPScoreModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(ds: WeakLabelDataset, batch_size: int, *, shuffle: bool) -> DataLoader:
    return DataLoader(TorchWeakDataset(ds), batch_size=batch_size, shuffle=shuffle)


def forward_rect_loss(logits: torch.Tensor, z: torch.Tensor, M_torch: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    probs_clean = torch.softmax(logits, dim=1)
    probs_weak = probs_clean @ M_torch.transpose(0, 1)
    probs_weak = torch.clamp(probs_weak, min=eps)
    probs_weak = probs_weak / probs_weak.sum(dim=1, keepdim=True)
    return F.nll_loss(torch.log(probs_weak), z)


def scores_from_model(model: nn.Module, ds: WeakLabelDataset, *, batch_size: int, device: str) -> Array:
    loader = make_loader(ds, batch_size, shuffle=False)
    device_obj = torch.device(device)
    outs = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device_obj)
            outs.append(model(xb).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outs, axis=0)


def evaluate_forward_model(model: nn.Module, ds: WeakLabelDataset, *, batch_size: int, device: str) -> dict[str, float]:
    scores = scores_from_model(model, ds, batch_size=batch_size, device=device)
    metrics = evaluate_scores_on_dataset(scores, ds)
    return {
        "clean_accuracy": float(metrics.clean_accuracy),
        "pairwise_order_rate": float(metrics.pairwise_order_rate),
        "max_preservation_rate": float(metrics.max_preservation_rate),
    }


def validation_forward_risk(model: nn.Module, ds: WeakLabelDataset, M_train: Array, *, batch_size: int, device: str) -> float:
    loader = make_loader(ds, batch_size, shuffle=False)
    device_obj = torch.device(device)
    M_torch = torch.tensor(np.asarray(M_train, dtype=np.float32), device=device_obj)
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for xb, zb in loader:
            xb = xb.to(device_obj)
            zb = zb.to(device_obj)
            loss = forward_rect_loss(model(xb), zb, M_torch)
            total += float(loss.item()) * int(xb.shape[0])
            count += int(xb.shape[0])
    return total / max(count, 1)


def train_forward_rect_model(train_ds: WeakLabelDataset, M_train: Array, *, val_ds: Optional[WeakLabelDataset], cfg: CLWLTrainConfig):
    set_torch_seed(cfg.seed)
    input_dim = train_ds.X.shape[1]
    c = train_ds.eta.shape[1]
    model = MLPScoreModel(input_dim, cfg.hidden_dim, c).to(torch.device(cfg.device))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    M_torch = torch.tensor(np.asarray(M_train, dtype=np.float32), device=torch.device(cfg.device))
    loader = make_loader(train_ds, cfg.batch_size, shuffle=True)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        for xb, zb in loader:
            xb = xb.to(torch.device(cfg.device))
            zb = zb.to(torch.device(cfg.device))
            opt.zero_grad()
            loss = forward_rect_loss(model(xb), zb, M_torch)
            loss.backward()
            if cfg.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            opt.step()

        if val_ds is not None and (epoch % cfg.log_every == 0 or epoch == cfg.num_epochs):
            val_loss = validation_forward_risk(model, val_ds, M_train, batch_size=max(cfg.batch_size, 512), device=cfg.device)
            if val_loss < best_val - cfg.early_stop_min_delta:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


# ============================================================
# Experiment runner
# ============================================================

def run_experiment(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = [int(x) for x in args.classes]
    c = len(classes)
    pairs = default_confusion_pairs(c)
    s_grid = [float(x) for x in args.s_grid]

    clean_splits = load_mnist_subset_splits(
        root=args.data_root,
        classes=classes,
        val_frac=args.val_frac,
        seed=args.seed,
        max_trainval_samples=args.max_trainval_samples,
        max_test_samples=args.max_test_samples,
        standardize=not args.no_standardize,
        download=True,
    )

    M_true = make_mnist_weak_M_true(c, pairs)
    np.save(out_dir / "M_true.npy", M_true)

    if args.optimize_h:
        H, diag_df, best_score = optimize_H(
            M_true,
            pairs,
            s_grid=s_grid,
            seed=args.h_seed,
            candidates=args.h_candidates,
            min_lambda=args.min_lambda,
            max_resid=args.max_resid,
        )
        print("Best H score:", best_score)
    else:
        H = base_structured_H(c, pairs)
        diag_df = diagnostic_path(M_true, H, s_grid)

    np.save(out_dir / "H.npy", H)
    diag_df = diagnostic_path(M_true, H, s_grid)
    diag_df.to_csv(out_dir / "diagnostics.csv", index=False)
    print("=== Diagnostics ===")
    print(diag_df)

    with open(out_dir / "experiment_metadata.json", "w") as f:
        json.dump({
            "classes": classes,
            "pairs": pairs,
            "s_grid": s_grid,
            "M_true_shape": list(M_true.shape),
            "H_shape": list(H.shape),
            "weak_label_dimension_d": int(M_true.shape[0]),
            "clean_dimension_c": int(M_true.shape[1]),
        }, f, indent=2)

    weak_splits_by_seed = {}
    cfg_base = CLWLTrainConfig(
        model_type="mlp",
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=0,
        log_every=1,
        gradient_clip_norm=5.0,
        scheduler_factor=0.5,
        scheduler_patience=5,
        min_learning_rate=1e-5,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=1e-8,
    )

    rows = []
    for seed in args.train_seeds:
        weak_splits = build_weak_splits(clean_splits, M_true, seed=4000 + int(seed))
        weak_splits_by_seed[int(seed)] = weak_splits
        cfg_seed = replace(cfg_base, seed=int(seed))

        # Oracle Forward is fixed across s for the same seed.
        oracle_model, oracle_epoch = train_forward_rect_model(weak_splits["train"], M_true, val_ds=weak_splits["val"], cfg=cfg_seed)
        oracle_test = evaluate_forward_model(oracle_model, weak_splits["test"], batch_size=max(args.batch_size, 512), device=args.device)

        for s in s_grid:
            M_hat = row_tilt(M_true, H, s)
            T_hat = construct_clwl_T(M_hat).T
            np.save(out_dir / f"M_hat_s_{s:.3f}.npy", M_hat)
            np.save(out_dir / f"T_hat_s_{s:.3f}.npy", T_hat)
            diag = diag_df.loc[np.isclose(diag_df["s"], s)].iloc[0].to_dict()

            clwl_result = train_clwl_model(weak_splits["train"], T=T_hat, val_dataset=weak_splits["val"], config=cfg_seed)
            clwl_test = evaluate_clwl_model_on_dataset(
                clwl_result.model, weak_splits["test"], T_hat, batch_size=max(args.batch_size, 512), device=args.device
            )
            rows.append({
                "method": "CLWL_T_Mhat",
                "seed": int(seed),
                "s": float(s),
                "test_clean_accuracy": float(clwl_test["clean_accuracy"]),
                "test_pairwise_order_rate": float(clwl_test["pairwise_order_rate"]),
                "test_max_preservation_rate": float(clwl_test["max_preservation_rate"]),
                "best_epoch": int(clwl_result.best_epoch),
                **diag,
            })

            fwd_model, fwd_epoch = train_forward_rect_model(weak_splits["train"], M_hat, val_ds=weak_splits["val"], cfg=cfg_seed)
            fwd_test = evaluate_forward_model(fwd_model, weak_splits["test"], batch_size=max(args.batch_size, 512), device=args.device)
            rows.append({
                "method": "Forward_Mhat",
                "seed": int(seed),
                "s": float(s),
                "test_clean_accuracy": float(fwd_test["clean_accuracy"]),
                "test_pairwise_order_rate": float(fwd_test["pairwise_order_rate"]),
                "test_max_preservation_rate": float(fwd_test["max_preservation_rate"]),
                "best_epoch": int(fwd_epoch),
                **diag,
            })

            rows.append({
                "method": "Forward_oracle_Mtrue",
                "seed": int(seed),
                "s": float(s),
                "test_clean_accuracy": float(oracle_test["clean_accuracy"]),
                "test_pairwise_order_rate": float(oracle_test["pairwise_order_rate"]),
                "test_max_preservation_rate": float(oracle_test["max_preservation_rate"]),
                "best_epoch": int(oracle_epoch),
                **diag,
            })

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(out_dir / "raw_results.csv", index=False)

    summary_df = (
        raw_df.groupby(["method", "s"], as_index=False)
        .agg({
            "test_clean_accuracy": ["mean", "std"],
            "test_pairwise_order_rate": ["mean", "std"],
            "test_max_preservation_rate": ["mean", "std"],
            "best_epoch": ["mean", "std"],
            "lambda": ["mean"],
            "relative_residual": ["mean"],
            "forward_proxy_class_recovery": ["mean"],
            "forward_proxy_true_prob": ["mean"],
        })
    )
    summary_df.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary_df.columns]
    summary_df.to_csv(out_dir / "summary_results.csv", index=False)
    print("=== Summary ===")
    print(summary_df)

    plot_results(summary_df, out_dir / "mnist_row_tilt_results.png")
    plot_diagnostics(diag_df, out_dir / "mnist_row_tilt_diagnostics.png")
    print("Saved outputs to", out_dir.resolve())


def _std(df: pd.DataFrame, col: str) -> Array:
    if col not in df.columns:
        return np.zeros(len(df), dtype=float)
    return np.nan_to_num(df[col].to_numpy(dtype=float), nan=0.0)


def plot_results(summary_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    methods = ["CLWL_T_Mhat", "Forward_Mhat", "Forward_oracle_Mtrue"]
    for method in methods:
        dfm = summary_df[summary_df["method"] == method].sort_values("s")
        x = dfm["s"].to_numpy(dtype=float)
        for ax, metric, ylabel in [
            (axes[0], "test_clean_accuracy", "clean accuracy"),
            (axes[1], "test_pairwise_order_rate", "pairwise order rate"),
        ]:
            y = dfm[f"{metric}_mean"].to_numpy(dtype=float)
            ystd = _std(dfm, f"{metric}_std")
            ax.plot(x, y, marker="o", label=method)
            ax.fill_between(x, y - ystd, y + ystd, alpha=0.15)
            ax.set_xlabel("estimate response-bias strength s")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)

    dfc = summary_df[summary_df["method"] == "CLWL_T_Mhat"].sort_values("s")
    dff = summary_df[summary_df["method"] == "Forward_Mhat"].sort_values("s")
    x = dfc["s"].to_numpy(dtype=float)
    gap = dfc["test_clean_accuracy_mean"].to_numpy(dtype=float) - dff["test_clean_accuracy_mean"].to_numpy(dtype=float)
    axes[2].plot(x, gap, marker="o", label="CLWL - Forward_Mhat")
    axes[2].axhline(0.0, linestyle="--", alpha=0.7)
    axes[2].set_xlabel("estimate response-bias strength s")
    axes[2].set_ylabel("accuracy gap")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[0].legend()
    axes[1].legend()
    axes[0].set_title("MNIST row-tilt: accuracy")
    axes[1].set_title("MNIST row-tilt: ranking")
    axes[2].set_title("Gap to misspecified Forward")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(diag_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    x = diag_df["s"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    axes[0].plot(x, diag_df["lambda"].to_numpy(dtype=float), marker="o")
    axes[0].axhline(0.0, linestyle="--", alpha=0.7)
    axes[0].set_title("lambda of T(Mhat)Mtrue")
    axes[0].set_xlabel("s")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, diag_df["relative_residual"].to_numpy(dtype=float), marker="o")
    axes[1].set_title("standard-form residual")
    axes[1].set_xlabel("s")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, diag_df["forward_proxy_class_recovery"].to_numpy(dtype=float), marker="o")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Forward proxy class recovery")
    axes[2].set_xlabel("s")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(x, diag_df["forward_proxy_true_prob"].to_numpy(dtype=float), marker="o")
    axes[3].set_ylim(0.0, 1.0)
    axes[3].set_title("Forward proxy true-class prob")
    axes[3].set_xlabel("s")
    axes[3].grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="artifacts_mnist_row_tilt")
    p.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2, 3], help="Use 0 1 ... 9 for full MNIST-10.")
    p.add_argument("--s_grid", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    p.add_argument("--train_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--h_seed", type=int, default=123)
    p.add_argument("--h_candidates", type=int, default=200)
    p.add_argument("--optimize_h", action="store_true", help="Run matrix-level random search for H before training.")
    p.add_argument("--min_lambda", type=float, default=0.05)
    p.add_argument("--max_resid", type=float, default=0.25)
    p.add_argument("--max_trainval_samples", type=int, default=12000)
    p.add_argument("--max_test_samples", type=int, default=3000)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--no_standardize", action="store_true")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
