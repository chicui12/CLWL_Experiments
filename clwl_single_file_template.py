from __future__ import annotations

# ==========================================================
# CLWL single-file experiment script
# Recommended for the current stage:
# - CLWL-only
# - theory verification first
# - E3 first, then E1 / E2 later
# - easy to paste into VS Code and run directly
# ==========================================================

import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from itertools import combinations
from typing import Dict, List, Tuple, Any



# ==========================================================
# 0. Basic utilities
# Put here:
# - seed
# - one_hot
# - mkdir
# ==========================================================


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(indices), num_classes), dtype=np.float32)
    out[np.arange(len(indices)), indices] = 1.0
    return out



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ==========================================================
# 1. Theory diagnostics
# Put here:
# - order_preservation_rate
# - strict_order_violation_count
# - fit_lambda_1v
# ==========================================================


def order_preservation_rate(
    A: np.ndarray,
    num_samples: int = 20000,
    tol_eta: float = 1e-10,
    tol_out: float = 1e-10,
    seed: int = 0,
) -> float:
    A = np.asarray(A, dtype=np.float64)
    c = A.shape[0]
    rng = np.random.default_rng(seed)

    good = 0
    total = 0
    for _ in range(num_samples):
        eta = rng.dirichlet(np.ones(c))
        out = A @ eta
        for i in range(c):
            for j in range(c):
                if i == j:
                    continue
                if eta[i] > eta[j] + tol_eta:
                    total += 1
                    if out[i] > out[j] + tol_out:
                        good += 1

    return 1.0 if total == 0 else good / total



def strict_order_violation_count(
    A: np.ndarray,
    num_samples: int = 20000,
    tol_eta: float = 1e-10,
    tol_out: float = 1e-10,
    seed: int = 0,
) -> Dict[str, float]:
    A = np.asarray(A, dtype=np.float64)
    c = A.shape[0]
    rng = np.random.default_rng(seed)

    violations = 0
    total = 0
    max_margin_flip = 0.0

    for _ in range(num_samples):
        eta = rng.dirichlet(np.ones(c))
        out = A @ eta
        for i in range(c):
            for j in range(c):
                if i == j:
                    continue
                if eta[i] > eta[j] + tol_eta:
                    total += 1
                    if not (out[i] > out[j] + tol_out):
                        violations += 1
                        flip = (eta[i] - eta[j]) - (out[i] - out[j])
                        max_margin_flip = max(max_margin_flip, flip)

    return {
        "violation_count": float(violations),
        "pair_count": float(total),
        "violation_rate": 0.0 if total == 0 else violations / total,
        "max_margin_flip": float(max_margin_flip),
    }



def fit_lambda_1v(A: np.ndarray) -> Dict[str, Any]:
    A = np.asarray(A, dtype=np.float64)
    c = A.shape[0]
    assert A.shape == (c, c), "A must be square"

    v_hat = np.zeros(c, dtype=np.float64)
    for k in range(c):
        mask = np.ones(c, dtype=bool)
        mask[k] = False
        v_hat[k] = A[mask, k].mean()

    lambda_hat = float(np.mean(np.diag(A) - v_hat))
    A_hat = lambda_hat * np.eye(c) + np.ones((c, 1)) @ v_hat[None, :]
    residual = A - A_hat

    fro_error = float(np.linalg.norm(residual, ord="fro"))
    relative_error = float(fro_error / (np.linalg.norm(A, ord="fro") + 1e-12))
    max_abs_error = float(np.abs(residual).max())

    offdiag_spreads = []
    for k in range(c):
        mask = np.ones(c, dtype=bool)
        mask[k] = False
        offdiag = A[mask, k]
        offdiag_spreads.append(offdiag.max() - offdiag.min())

    return {
        "lambda_hat": lambda_hat,
        "v_hat": v_hat,
        "A_hat": A_hat,
        "residual": residual,
        "fro_error": fro_error,
        "relative_error": relative_error,
        "max_abs_error": max_abs_error,
        "offdiag_spread_mean": float(np.mean(offdiag_spreads)),
        "offdiag_spread_max": float(np.max(offdiag_spreads)),
    }


# ==========================================================
# 2. Build T from M
# Put here:
# - build_T_from_M
# ==========================================================


def build_T_from_M(M: np.ndarray) -> Dict[str, Any]:
    M = np.asarray(M, dtype=np.float64)
    d, c = M.shape
    assert d >= c, "Need d >= c"
    assert np.linalg.matrix_rank(M) == c, "M must be full column rank"

    N = np.linalg.inv(M.T @ M) @ M.T
    q = N.min(axis=0)
    col_ranges = N.max(axis=0) - N.min(axis=0)
    alpha = float(0.99 / col_ranges.max())

    T = alpha * (N - np.ones((c, 1)) @ q[None, :])
    NM = N @ M
    A = T @ M

    return {
        "M": M,
        "N": N,
        "q": q,
        "alpha": alpha,
        "T": T,
        "NM": NM,
        "A": A,
        "T_min": float(T.min()),
        "T_max": float(T.max()),
        "NM_fro_to_I": float(np.linalg.norm(NM - np.eye(c), ord="fro")),
    }


# ==========================================================
# 3. Losses
# Put here:
# - CLWLLoss
# ==========================================================


class CLWLLoss(nn.Module):
    def __init__(self, T: np.ndarray):
        super().__init__()
        self.register_buffer("T", torch.tensor(T, dtype=torch.float32))

    @staticmethod
    def beta(f: torch.Tensor) -> torch.Tensor:
        return F.softplus(-f)

    def forward(self, logits: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        support = Z @ self.T.T
        loss_pos = support * self.beta(logits)
        loss_neg = (1.0 - support) * self.beta(-logits)
        return (loss_pos + loss_neg).sum(dim=1).mean()


# ==========================================================
# 4. Models
# Put here:
# - LinearClassifier
# - MLP1
# - build_model
# ==========================================================


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, c: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLP1(nn.Module):
    def __init__(self, input_dim: int, c: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



def build_model(model_type: str, input_dim: int, c: int, hidden_dim: int = 64) -> nn.Module:
    if model_type == "linear":
        return LinearClassifier(input_dim=input_dim, c=c)
    if model_type == "mlp":
        return MLP1(input_dim=input_dim, c=c, hidden_dim=hidden_dim)
    raise ValueError(f"Unknown model_type: {model_type}")


# ==========================================================
# 5. Synthetic data
# Put here:
# - make_gaussian_classification_data
# - sample_weak_labels_from_M
# ==========================================================


def make_gaussian_classification_data(
    c: int,
    input_dim: int,
    n_train_per_class: int,
    n_test_per_class: int,
    seed: int,
    class_sep: float = 3.0,
    noise_std: float = 1.0,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    means = np.zeros((c, input_dim), dtype=np.float64)
    for j in range(c):
        if input_dim == 2:
            angle = 2.0 * np.pi * j / c
            means[j, 0] = class_sep * np.cos(angle)
            means[j, 1] = class_sep * np.sin(angle)
        else:
            means[j, j % input_dim] = class_sep
            means[j, (j + 1) % input_dim] = 0.5 * class_sep

    cov = (noise_std ** 2) * np.eye(input_dim)
    Xtr_list, ytr_list, Xte_list, yte_list = [], [], [], []

    for cls in range(c):
        Xtr = rng.multivariate_normal(mean=means[cls], cov=cov, size=n_train_per_class)
        Xte = rng.multivariate_normal(mean=means[cls], cov=cov, size=n_test_per_class)
        ytr = np.full(n_train_per_class, cls, dtype=np.int64)
        yte = np.full(n_test_per_class, cls, dtype=np.int64)
        Xtr_list.append(Xtr)
        Xte_list.append(Xte)
        ytr_list.append(ytr)
        yte_list.append(yte)

    X_train = np.concatenate(Xtr_list, axis=0).astype(np.float32)
    y_train = np.concatenate(ytr_list, axis=0)
    X_test = np.concatenate(Xte_list, axis=0).astype(np.float32)
    y_test = np.concatenate(yte_list, axis=0)

    perm_tr = rng.permutation(len(y_train))
    perm_te = rng.permutation(len(y_test))

    return {
        "X_train": X_train[perm_tr],
        "y_train": y_train[perm_tr],
        "X_test": X_test[perm_te],
        "y_test": y_test[perm_te],
        "means": means,
    }



def sample_weak_labels_from_M(y: np.ndarray, M: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    M = np.asarray(M, dtype=np.float64)
    d, _ = M.shape
    rng = np.random.default_rng(seed)

    z_idx = np.empty(len(y), dtype=np.int64)
    for n, cls in enumerate(y):
        z_idx[n] = rng.choice(d, p=M[:, cls])

    Z = one_hot(z_idx, d)
    return z_idx, Z


# ==========================================================
# 6. Train / eval
# Put here:
# - evaluate_model
# - train_clwl
# ==========================================================


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total += y.numel()

    return {"acc": total_correct / max(total, 1)}



def train_clwl(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
) -> List[Dict[str, float]]:
    model.to(device)
    loss_fn.to(device)
    history: List[Dict[str, float]] = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for x, Z, y in train_loader:
            x = x.to(device)
            Z = Z.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, Z)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_count += x.size(0)

        train_loss = total_loss / max(total_count, 1)
        train_acc = total_correct / max(total_count, 1)
        test_acc = evaluate_model(model, test_loader, device)["acc"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "test_acc": float(test_acc),
            }
        )

    return history


# ==========================================================
# 7. Experiment config + runner
# Put here:
# - ExperimentConfig
# - run_single_experiment
# ==========================================================


@dataclass
class ExperimentConfig:
    case_name: str
    family: str
    matrix_name: str
    model_type: str
    M: np.ndarray
    seed: int
    c: int = 4
    input_dim: int = 2
    hidden_dim: int = 64
    n_train_per_class: int = 500
    n_test_per_class: int = 500
    batch_size: int = 128
    lr: float = 1e-2
    weight_decay: float = 1e-4
    num_epochs: int = 100
    class_sep: float = 3.0
    noise_std: float = 1.0
    device: str = "cpu"



def run_single_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)

    M = np.asarray(cfg.M, dtype=np.float64)
    d, c_from_M = M.shape
    assert c_from_M == cfg.c, "cfg.c must match M.shape[1]"

    T_info = build_T_from_M(M)
    T = T_info["T"]
    A = T_info["A"]
    order_rate = order_preservation_rate(A, seed=cfg.seed)
    violation_info = strict_order_violation_count(A, seed=cfg.seed)
    fit_info = fit_lambda_1v(A)

    data = make_gaussian_classification_data(
        c=cfg.c,
        input_dim=cfg.input_dim,
        n_train_per_class=cfg.n_train_per_class,
        n_test_per_class=cfg.n_test_per_class,
        seed=cfg.seed,
        class_sep=cfg.class_sep,
        noise_std=cfg.noise_std,
    )

    z_idx_train, Z_train = sample_weak_labels_from_M(data["y_train"], M, seed=cfg.seed + 1000)

    Xtr = torch.tensor(data["X_train"], dtype=torch.float32)
    ytr = torch.tensor(data["y_train"], dtype=torch.long)
    Ztr = torch.tensor(Z_train, dtype=torch.float32)

    Xte = torch.tensor(data["X_test"], dtype=torch.float32)
    yte = torch.tensor(data["y_test"], dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(Xtr, Ztr, ytr),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(Xte, yte),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    device = torch.device(cfg.device)
    model = build_model(cfg.model_type, input_dim=cfg.input_dim, c=cfg.c, hidden_dim=cfg.hidden_dim)
    loss_fn = CLWLLoss(T)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = train_clwl(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=cfg.num_epochs,
        device=device,
    )

    result_row = {
        "case_name": cfg.case_name,
        "family": cfg.family,
        "matrix_name": cfg.matrix_name,
        "seed": cfg.seed,
        "model_type": cfg.model_type,
        "c": cfg.c,
        "d": d,
        "input_dim": cfg.input_dim,
        "test_acc": history[-1]["test_acc"],
        "train_acc": history[-1]["train_acc"],
        "final_train_loss": history[-1]["train_loss"],
        "order_rate": order_rate,
        "violation_rate": violation_info["violation_rate"],
        "violation_count": violation_info["violation_count"],
        "max_margin_flip": violation_info["max_margin_flip"],
        "lambda_hat": fit_info["lambda_hat"],
        "fit_fro_error": fit_info["fro_error"],
        "fit_relative_error": fit_info["relative_error"],
        "fit_max_abs_error": fit_info["max_abs_error"],
        "offdiag_spread_mean": fit_info["offdiag_spread_mean"],
        "offdiag_spread_max": fit_info["offdiag_spread_max"],
        "T_min": T_info["T_min"],
        "T_max": T_info["T_max"],
        "alpha": T_info["alpha"],
        "NM_fro_to_I": T_info["NM_fro_to_I"],
    }

    artifacts = {
        "config": asdict(cfg),
        "M": M,
        "T": T,
        "A": A,
        "N": T_info["N"],
        "q": T_info["q"],
        "A_hat": fit_info["A_hat"],
        "fit_residual": fit_info["residual"],
        "history": history,
        "means": data["means"],
        "z_idx_train": z_idx_train,
    }

    return {"result": result_row, "artifacts": artifacts}


# ==========================================================
# 8. Case builders / matrix bank
# Put here now:
# - E3 matrix generators
# Put here later:
# - E1 builders
# - E2 builders
# - dominance checker
# ==========================================================


def normalize_columns(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    return M / M.sum(axis=0, keepdims=True)



def random_full_rank_column_stochastic_matrix(
    c: int,
    diag_strength: float,
    asymmetry: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = np.zeros((c, c), dtype=np.float64)

    for col in range(c):
        off_idx = [i for i in range(c) if i != col]
        off = rng.dirichlet(np.ones(c - 1) * max(asymmetry, 1e-3))
        off = (1.0 - diag_strength) * off
        M[col, col] = diag_strength
        M[off_idx, col] = off

    M += 1e-4 * rng.normal(size=M.shape)
    M = np.clip(M, 1e-8, None)
    M = normalize_columns(M)

    if np.linalg.matrix_rank(M) < c:
        M += 1e-3 * np.eye(c)
        M = normalize_columns(M)

    return M



def make_e3_manual_bank(c: int = 4) -> Dict[str, List[np.ndarray]]:
    assert c == 4, "Manual bank currently written for c=d=4"

    E3A_1 = np.array([
        [0.70, 0.10, 0.08, 0.12],
        [0.10, 0.70, 0.12, 0.08],
        [0.10, 0.10, 0.70, 0.10],
        [0.10, 0.10, 0.10, 0.70],
    ])
    E3A_2 = np.array([
        [0.76, 0.08, 0.10, 0.06],
        [0.08, 0.72, 0.08, 0.14],
        [0.10, 0.10, 0.68, 0.10],
        [0.06, 0.10, 0.14, 0.70],
    ])

    E3B_1 = np.array([
        [0.50, 0.18, 0.15, 0.17],
        [0.20, 0.48, 0.17, 0.15],
        [0.15, 0.16, 0.50, 0.18],
        [0.15, 0.18, 0.18, 0.50],
    ])
    E3B_2 = np.array([
        [0.52, 0.12, 0.18, 0.20],
        [0.18, 0.50, 0.12, 0.18],
        [0.14, 0.20, 0.48, 0.14],
        [0.16, 0.18, 0.22, 0.48],
    ])

    E3C_1 = np.array([
        [0.46, 0.12, 0.08, 0.30],
        [0.34, 0.56, 0.10, 0.18],
        [0.12, 0.22, 0.62, 0.12],
        [0.08, 0.10, 0.20, 0.40],
    ])
    E3C_2 = np.array([
        [0.58, 0.08, 0.14, 0.28],
        [0.20, 0.54, 0.08, 0.12],
        [0.14, 0.26, 0.60, 0.18],
        [0.08, 0.12, 0.18, 0.42],
    ])

    return {
        "E3-A": [normalize_columns(E3A_1), normalize_columns(E3A_2)],
        "E3-B": [normalize_columns(E3B_1), normalize_columns(E3B_2)],
        "E3-C": [normalize_columns(E3C_1), normalize_columns(E3C_2)],
    }



def make_e3_random_bank(c: int = 4, n_each: int = 3, base_seed: int = 1234) -> Dict[str, List[np.ndarray]]:
    bank = {"E3-A": [], "E3-B": [], "E3-C": []}

    for k in range(n_each):
        bank["E3-A"].append(
            random_full_rank_column_stochastic_matrix(c=c, diag_strength=0.70, asymmetry=1.20, seed=base_seed + k)
        )
    for k in range(n_each):
        bank["E3-B"].append(
            random_full_rank_column_stochastic_matrix(c=c, diag_strength=0.50, asymmetry=0.80, seed=base_seed + 100 + k)
        )
    for k in range(n_each):
        bank["E3-C"].append(
            random_full_rank_column_stochastic_matrix(c=c, diag_strength=0.42, asymmetry=0.35, seed=base_seed + 200 + k)
        )

    return bank



#Clwl E1E2 Builders Patch
# ==========================================================
# Patch for the single-file CLWL script:
# E1 / E2 builders + dominance checker for pair-size-2 partial labels
#
# How to use:
# 1. Paste the functions in this file into section 8 of your single-file script
#    (case builders / matrix bank).
# 2. Keep E3 code as it is.
# 3. Later add a run_e1e2_batch(...) function that calls these builders.
#
# This patch assumes:
# - c classes
# - partial-label weak labels are candidate sets of size 2
# - weak labels are represented by one-hot z over all candidate sets
# - Z is the matrix whose rows are binary candidate-set vectors
# ==========================================================

# ==========================================================
# 1. Pair-candidate utilities
# ==========================================================


def make_pair_candidate_sets(c: int) -> Tuple[List[Tuple[int, int]], np.ndarray, Dict[Tuple[int, int], int]]:
    """
    Build all size-2 candidate sets.

    Returns:
        pairs: list like [(0,1), (0,2), ...]
        Z:     shape (d, c), each row is a binary candidate-set vector
        pair_to_idx: maps pair (i,j) with i<j to row index in Z
    """
    pairs = list(combinations(range(c), 2))
    d = len(pairs)
    Z = np.zeros((d, c), dtype=np.float64)
    pair_to_idx: Dict[Tuple[int, int], int] = {}

    for idx, (i, j) in enumerate(pairs):
        Z[idx, i] = 1.0
        Z[idx, j] = 1.0
        pair_to_idx[(i, j)] = idx

    return pairs, Z, pair_to_idx



def pair_index(i: int, j: int, pair_to_idx: Dict[Tuple[int, int], int]) -> int:
    if i < j:
        return pair_to_idx[(i, j)]
    return pair_to_idx[(j, i)]


# ==========================================================
# 2. Build partial-label M for size-2 candidate sets
# ==========================================================


def build_pair_partial_label_M(c: int, weights: np.ndarray) -> Dict[str, Any]:
    """
    Build M for pair-size-2 partial labels.

    Parameters
    ----------
    c : number of true classes
    weights : shape (c, c)
        weights[y, k] is the unnormalized weight for emitting candidate set {y, k}
        when the true class is y. The diagonal weights[y, y] is ignored.

    Output
    ------
    M : shape (d, c), where d = c choose 2
        Column y is P(z | y), supported only on candidate sets containing y.

    Also returns Z and pair metadata.
    """
    weights = np.asarray(weights, dtype=np.float64)
    assert weights.shape == (c, c), "weights must have shape (c, c)"

    pairs, Z, pair_to_idx = make_pair_candidate_sets(c)
    d = len(pairs)
    M = np.zeros((d, c), dtype=np.float64)

    for y in range(c):
        col_weights = np.zeros(d, dtype=np.float64)
        for k in range(c):
            if k == y:
                continue
            idx = pair_index(y, k, pair_to_idx)
            col_weights[idx] = max(weights[y, k], 0.0)

        s = col_weights.sum()
        assert s > 0.0, f"Column {y} has zero total mass"
        M[:, y] = col_weights / s

    return {
        "M": M,
        "Z": Z,
        "pairs": pairs,
        "pair_to_idx": pair_to_idx,
    }


# ==========================================================
# 3. E1 / E2 manual builders
# ==========================================================


def make_e1_uniform_pair_case(c: int = 4) -> Dict[str, Any]:
    """
    E1: partial-label friendly, symmetric, pair-size-2.
    For each true class y, choose uniformly among all pairs containing y.

    This is the cleanest first E1 case.
    """
    weights = np.ones((c, c), dtype=np.float64)
    np.fill_diagonal(weights, 0.0)
    out = build_pair_partial_label_M(c=c, weights=weights)
    out["case_name"] = "E1_uniform_pairs"
    out["weights"] = weights
    return out



def make_e1_mild_symmetric_case(c: int = 4) -> Dict[str, Any]:
    """
    E1: still partial-label friendly, but not perfectly uniform.
    Symmetric and mild, intended to remain dominance-friendly.
    """
    weights = np.array(
        [
            [0.0, 1.0, 0.9, 1.1],
            [1.0, 0.0, 1.1, 0.9],
            [0.9, 1.1, 0.0, 1.0],
            [1.1, 0.9, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    out = build_pair_partial_label_M(c=c, weights=weights)
    out["case_name"] = "E1_mild_symmetric_pairs"
    out["weights"] = weights
    return out



def make_e2_biased_pair_case_1(c: int = 4) -> Dict[str, Any]:
    """
    E2: pair-style partial labels with clear directional bias.
    Intended to break dominance while preserving the 'true class is in candidate set' property.
    """
    assert c == 4, "Current manual E2 builders are written for c=4"
    weights = np.array(
        [
            [0.0, 5.0, 1.0, 1.0],
            [1.0, 0.0, 5.0, 1.0],
            [1.0, 1.0, 0.0, 5.0],
            [4.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    out = build_pair_partial_label_M(c=c, weights=weights)
    out["case_name"] = "E2_biased_pairs_1"
    out["weights"] = weights
    return out



def make_e2_biased_pair_case_2(c: int = 4) -> Dict[str, Any]:
    """
    Another E2 case with stronger non-symmetric pair preference.
    """
    assert c == 4, "Current manual E2 builders are written for c=4"
    weights = np.array(
        [
            [0.0, 6.0, 2.0, 1.0],
            [1.0, 0.0, 4.0, 2.0],
            [2.0, 1.0, 0.0, 6.0],
            [5.0, 2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    out = build_pair_partial_label_M(c=c, weights=weights)
    out["case_name"] = "E2_biased_pairs_2"
    out["weights"] = weights
    return out



def make_e1_bank(c: int = 4) -> List[Dict[str, Any]]:
    return [
        make_e1_uniform_pair_case(c=c),
        make_e1_mild_symmetric_case(c=c),
    ]



def make_e2_bank(c: int = 4) -> List[Dict[str, Any]]:
    return [
        make_e2_biased_pair_case_1(c=c),
        make_e2_biased_pair_case_2(c=c),
    ]


# ==========================================================
# 4. Partial-label diagnostics
# ==========================================================


def class_support_matrix(Z: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Return Z^T M.

    For partial labels, (Z^T M eta)_i is the probability that class i
    appears in the candidate set under class-posterior eta.
    """
    Z = np.asarray(Z, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    return Z.T @ M



def max_preserving_rate(
    A: np.ndarray,
    num_samples: int = 20000,
    tol: float = 1e-10,
    seed: int = 0,
) -> float:
    """
    Empirical rate for unique-max preservation:
        argmax eta == argmax A eta
    only on samples with unique maxima on both sides.
    """
    A = np.asarray(A, dtype=np.float64)
    c = A.shape[0]
    rng = np.random.default_rng(seed)

    good = 0
    total = 0
    for _ in range(num_samples):
        eta = rng.dirichlet(np.ones(c))
        out = A @ eta

        max_eta = np.max(eta)
        max_out = np.max(out)
        idx_eta = np.flatnonzero(eta >= max_eta - tol)
        idx_out = np.flatnonzero(out >= max_out - tol)

        if len(idx_eta) == 1 and len(idx_out) == 1:
            total += 1
            if idx_eta[0] == idx_out[0]:
                good += 1

    return 1.0 if total == 0 else good / total



def dominance_rate_pair_partial_labels(
    Z: np.ndarray,
    M: np.ndarray,
    num_samples: int = 20000,
    tol: float = 1e-10,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Empirical dominance checker for pair-size-2 candidate sets.

    The theorem in the paper states the dominance condition in terms of candidate-set
    probabilities. For size-2 candidate sets, the shared context c has exactly one active
    class k different from a and b, and we compare the probabilities of candidate sets
    {a, k} and {b, k}.

    We check:
      - compute q = Z^T M eta
      - choose a in argmax(q)
      - choose b not in argmax(q)
      - for every context class k not in {a, b}, compare
            P({a,k}|eta) >= P({b,k}|eta)

    This is empirical, not symbolic, which is appropriate for experiments.
    """
    Z = np.asarray(Z, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    d, c = Z.shape
    pairs, _, pair_to_idx = make_pair_candidate_sets(c)
    assert d == len(pairs), "This checker assumes all size-2 candidate sets are present"

    rng = np.random.default_rng(seed)
    total = 0
    good = 0

    ZTM = Z.T @ M

    for _ in range(num_samples):
        eta = rng.dirichlet(np.ones(c))
        q = ZTM @ eta
        weak_probs = M @ eta

        q_max = np.max(q)
        argmax_set = set(np.flatnonzero(q >= q_max - tol).tolist())
        nonmax_set = [i for i in range(c) if i not in argmax_set]

        if len(nonmax_set) == 0:
            continue

        for a in argmax_set:
            for b in nonmax_set:
                for k in range(c):
                    if k == a or k == b:
                        continue
                    idx_ak = pair_index(a, k, pair_to_idx)
                    idx_bk = pair_index(b, k, pair_to_idx)
                    total += 1
                    if weak_probs[idx_ak] >= weak_probs[idx_bk] - tol:
                        good += 1

    return {
        "dominance_rate": 1.0 if total == 0 else good / total,
        "dominance_violation_rate": 0.0 if total == 0 else 1.0 - good / total,
        "dominance_total_checks": float(total),
    }


# ==========================================================
# 5. Convenience wrapper for E1 / E2 diagnostics
# ==========================================================


def diagnose_partial_label_case(
    M: np.ndarray,
    Z: np.ndarray,
    num_samples: int = 20000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Return the most important CLPL-style diagnostics for E1 / E2:
    - max-preserving rate of Z^T M
    - dominance rate
    - order-preservation rate of Z^T M
    """
    ZTM = Z.T @ M
    max_rate = max_preserving_rate(ZTM, num_samples=num_samples, seed=seed)

    # reuse the strict order checker from the single-file template if available
    try:
        order_rate = order_preservation_rate(ZTM, num_samples=num_samples, seed=seed)
    except NameError:
        order_rate = np.nan

    dom = dominance_rate_pair_partial_labels(
        Z=Z,
        M=M,
        num_samples=num_samples,
        seed=seed,
    )

    return {
        "ztm_max_rate": float(max_rate),
        "ztm_order_rate": float(order_rate),
        "dominance_rate": float(dom["dominance_rate"]),
        "dominance_violation_rate": float(dom["dominance_violation_rate"]),
        "dominance_total_checks": float(dom["dominance_total_checks"]),
    }


# ==========================================================
# 6. Example usage inside your single-file script
#
# e1_cases = make_e1_bank(c=4)
# e2_cases = make_e2_bank(c=4)
#
# for case in e1_cases + e2_cases:
#     M = case["M"]
#     Z = case["Z"]
#     info = diagnose_partial_label_case(M, Z, num_samples=20000, seed=0)
#     print(case["case_name"], info)
# ==========================================================



# ==========================================================
# 9. Batch runner
# Put here:
# - run_e3_batch
# Later you can add:
# - run_e1_batch
# - run_e2_batch
# ==========================================================


def run_e3_batch(
    output_dir: str,
    input_dims: List[int] | None = None,
    seeds: List[int] | None = None,
    model_types: List[str] | None = None,
    num_epochs: int = 100,
    device: str = "cpu",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if input_dims is None:
        input_dims = [2, 10]
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    if model_types is None:
        model_types = ["linear", "mlp"]

    ensure_dir(output_dir)

    manual_bank = make_e3_manual_bank(c=4)
    random_bank = make_e3_random_bank(c=4, n_each=3, base_seed=1234)

    all_rows: List[Dict[str, Any]] = []
    all_artifacts: Dict[str, Any] = {}

    for family in ["E3-A", "E3-B", "E3-C"]:
        family_matrices: List[Tuple[str, np.ndarray]] = []

        for idx, M in enumerate(manual_bank[family], start=1):
            family_matrices.append((f"{family}_manual_{idx}", M))
        for idx, M in enumerate(random_bank[family], start=1):
            family_matrices.append((f"{family}_random_{idx}", M))

        for matrix_name, M in family_matrices:
            for input_dim in input_dims:
                for seed in seeds:
                    for model_type in model_types:
                        case_name = f"{matrix_name}_dim{input_dim}_{model_type}_seed{seed}"
                        cfg = ExperimentConfig(
                            case_name=case_name,
                            family=family,
                            matrix_name=matrix_name,
                            model_type=model_type,
                            M=M,
                            seed=seed,
                            input_dim=input_dim,
                            num_epochs=num_epochs,
                            device=device,
                        )
                        out = run_single_experiment(cfg)
                        all_rows.append(out["result"])
                        all_artifacts[case_name] = out["artifacts"]

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

    summary_df = (
        results_df.groupby(["family", "matrix_name", "input_dim", "model_type"])[
            ["test_acc", "order_rate", "fit_relative_error", "final_train_loss", "violation_rate"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    torch.save(all_artifacts, os.path.join(output_dir, "artifacts.pt"))
    return results_df, all_artifacts


# ==========================================================
# 10. Main entry
# This is the part you will edit most often in VS Code.
# Recommended workflow:
#   1. run smoke test first
#   2. confirm results.csv appears
#   3. then switch to full run
# ==========================================================

RUN_MODE = "full"   # change to "full" when ready
DEVICE = "cpu"       # or "cuda" if your environment is ready


if __name__ == "__main__":
    
    e1_cases = make_e1_bank(c=4)
    e2_cases = make_e2_bank(c=4)

    for case in e1_cases + e2_cases:
        info = diagnose_partial_label_case(case["M"], case["Z"], num_samples=20000, seed=0)
        print(case["case_name"], info)

    for case in e1_cases + e2_cases:
        M = case["M"]
        Z = case["Z"]

        partial_info = diagnose_partial_label_case(M, Z, num_samples=20000, seed=0)

        T_info = build_T_from_M(M)
        A = T_info["A"]
        clwl_order_rate = order_preservation_rate(A, num_samples=20000, seed=0)
        fit_info = fit_lambda_1v(A)

        print("\ncase:", case["case_name"])
        print("partial-label diagnostics:", partial_info)
        print("CLWL A=TM order_rate:", clwl_order_rate)
        print("CLWL fit_relative_error:", fit_info["relative_error"])
        print("lambda_hat:", fit_info["lambda_hat"])

    # if RUN_MODE == "smoke":
    #     results_df, artifacts = run_e3_batch(
    #         output_dir="./outputs/e3_smoke",
    #         input_dims=[2],
    #         seeds=[0],
    #         model_types=["linear"],
    #         num_epochs=20,
    #         device=DEVICE,
    #     )
    # elif RUN_MODE == "full":
    #     results_df, artifacts = run_e3_batch(
    #         output_dir="./outputs/e3_full",
    #         input_dims=[2, 10],
    #         seeds=[0, 1, 2, 3, 4],
    #         model_types=["linear", "mlp"],
    #         num_epochs=100,
    #         device=DEVICE,
    #     )
    # else:
    #     raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")

    #print(results_df.head())
    #print("Done.")


# ==========================================================
# Recommended VS Code folder layout
#
# clwl_project/
# ├─ clwl_experiments.py      <- paste this whole file here
# └─ outputs/
#
# Then run in terminal:
#   python clwl_experiments.py
# ==========================================================
