from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from clwl_experiments_module4_weak_label_dataset import WeakLabelDataset
from clwl_experiments_module6_metrics import evaluate_scores_on_dataset
from clwl_experiments_module7_clwl_training import (
    CLWLTrainConfig,
    train_clwl_model,
    evaluate_model_on_dataset as evaluate_clwl_model_on_dataset,
)


Array = np.ndarray


# ============================================================
# Module 19: Binary synthetic CLWL-friendly / Forward-unfriendly
# ============================================================
# Goal:
#   Build a binary c=2 experiment where T is designed from the semantic meaning
#   of weak labels and the same fixed T satisfies
#       T M_true(gamma) = lambda I + 1 v^T, lambda > 0
#   for every gamma in the misspecification path.
#
# Weak labels, d=4:
#   z=0: observed class-0-like label
#   z=1: observed class-1-like label
#   z=2: ambiguity bucket A
#   z=3: ambiguity bucket B
#
# CLWL uses semantic T:
#       T = [[1, 0, 1/2, 1/2],
#            [0, 1, 1/2, 1/2]].
#   Hence buckets A and B are semantically equivalent for CLWL.
#
# Forward uses M_est and treats A/B as different observed weak labels. We set
# M_est so that A/B are class-informative in one direction, while M_true(gamma)
# gradually swaps the A/B class association. This makes Forward_Mest strongly
# misspecified, while CLWL remains exact-good because T only sees A+B.


# ============================================================
# 1) Synthetic binary clean data with known eta(x)
# ============================================================

@dataclass
class CleanSyntheticDataset:
    X: Array
    y: Array
    eta: Array
    logits: Array
    metadata: dict[str, Any]


def sigmoid(t: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-t))


def make_binary_synthetic_dataset(
    *,
    n: int,
    input_dim: int = 8,
    seed: int = 0,
    label_seed: int = 1,
    margin_scale: float = 1.6,
) -> CleanSyntheticDataset:
    """Generate binary synthetic data with oracle posterior eta(x)."""
    rng = np.random.default_rng(seed)
    label_rng = np.random.default_rng(label_seed)

    X = rng.normal(size=(n, input_dim)).astype(np.float64)
    w = rng.normal(size=input_dim).astype(np.float64)
    w /= max(np.linalg.norm(w), 1e-12)

    raw = X @ w
    if input_dim >= 3:
        raw += 0.30 * np.sin(X[:, 0]) - 0.25 * np.cos(X[:, 1]) + 0.15 * X[:, 2] ** 2
    raw *= margin_scale

    p1 = sigmoid(raw)
    eta = np.stack([1.0 - p1, p1], axis=1)
    y = np.asarray([label_rng.choice(2, p=eta_i) for eta_i in eta], dtype=np.intp)
    logits = np.stack([np.zeros_like(raw), raw], axis=1)

    return CleanSyntheticDataset(
        X=X,
        y=y,
        eta=eta,
        logits=logits,
        metadata={
            "dataset_name": "binary_synthetic_logistic_teacher",
            "n": int(n),
            "input_dim": int(input_dim),
            "seed": int(seed),
            "label_seed": int(label_seed),
            "eta_source": "oracle_binary_teacher",
            "oracle_metrics": True,
        },
    )


def train_val_test_split_clean(
    ds: CleanSyntheticDataset,
    *,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> dict[str, CleanSyntheticDataset]:
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0,1).")
    if not (0.0 <= val_frac < 1.0):
        raise ValueError("val_frac must be in [0,1).")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.")

    n = ds.X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    split_indices = {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }

    out: dict[str, CleanSyntheticDataset] = {}
    for split, ids in split_indices.items():
        out[split] = CleanSyntheticDataset(
            X=ds.X[ids].copy(),
            y=ds.y[ids].copy(),
            eta=ds.eta[ids].copy(),
            logits=ds.logits[ids].copy(),
            metadata={**ds.metadata, "split": split},
        )
    return out


# ============================================================
# 2) CLWL-friendly but Forward-unfriendly transition family
# ============================================================

def validate_column_stochastic(M: Array, *, atol: float = 1e-8) -> None:
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"M must be 2D, got shape {M.shape}.")
    if np.min(M) < -atol:
        raise ValueError(f"M has negative entries: min={np.min(M):.3e}.")
    col_sums = M.sum(axis=0)
    if not np.allclose(col_sums, np.ones(M.shape[1]), atol=atol, rtol=0.0):
        raise ValueError(f"M must be column-stochastic. col_sums={col_sums}.")


def make_semantic_T() -> Array:
    """
    Semantic CLWL transformation.

    Ambiguity buckets A and B are identical under T.
    """
    return np.array(
        [
            [1.0, 0.0, 0.5, 0.5],
            [0.0, 1.0, 0.5, 0.5],
        ],
        dtype=np.float64,
    )


def make_estimated_binary_ambiguous_M(
    *,
    ambiguity_mass: float = 0.65,
    clean_flip: float = 0.10,
    est_split: float = 0.90,
) -> Array:
    """
    Estimated weak-label matrix M_est in R^{4 x 2}.

    Rows:
      0 = observed class-0-like
      1 = observed class-1-like
      2 = ambiguity bucket A
      3 = ambiguity bucket B

    In M_est, ambiguity bucket A is strongly associated with class 0 and
    ambiguity bucket B is strongly associated with class 1.
    """
    if not (0.0 <= ambiguity_mass < 1.0):
        raise ValueError(f"ambiguity_mass must be in [0,1), got {ambiguity_mass}.")
    if not (0.0 <= clean_flip < 0.5):
        raise ValueError(f"clean_flip must be in [0,0.5), got {clean_flip}.")
    if not (0.5 <= est_split <= 1.0):
        raise ValueError(f"est_split must be in [0.5,1], got {est_split}.")

    clean_mass = 1.0 - ambiguity_mass
    p_correct = clean_mass * (1.0 - clean_flip)
    p_flip = clean_mass * clean_flip
    p_major = ambiguity_mass * est_split
    p_minor = ambiguity_mass * (1.0 - est_split)

    M_est = np.array(
        [
            [p_correct, p_flip],   # observed class 0
            [p_flip, p_correct],   # observed class 1
            [p_major, p_minor],    # ambiguity A: estimated as class-0-associated
            [p_minor, p_major],    # ambiguity B: estimated as class-1-associated
        ],
        dtype=np.float64,
    )
    validate_column_stochastic(M_est)
    if np.linalg.matrix_rank(M_est) != 2:
        raise ValueError("M_est must be full column rank.")
    return M_est


def make_wrong_binary_ambiguous_M(M_est: Array) -> Array:
    """
    Swap ambiguity buckets A and B while keeping clean-like rows unchanged.
    """
    M_est = np.asarray(M_est, dtype=np.float64)
    if M_est.shape != (4, 2):
        raise ValueError(f"Expected M_est shape (4,2), got {M_est.shape}.")
    M_wrong = M_est.copy()
    M_wrong[[2, 3], :] = M_wrong[[3, 2], :]
    validate_column_stochastic(M_wrong)
    return M_wrong


def make_true_binary_ambiguous_M(M_est: Array, M_wrong: Array, *, gamma: float) -> Array:
    """
    True transition matrix:
        M_true(gamma) = (1-gamma) M_est + gamma M_wrong.

    For all gamma, rows 2 and 3 keep the same total ambiguity mass per class.
    Since T[:,2] = T[:,3], T M_true(gamma) is exactly invariant to this swap.
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0,1], got {gamma}.")
    M_est = np.asarray(M_est, dtype=np.float64)
    M_wrong = np.asarray(M_wrong, dtype=np.float64)
    M_true = (1.0 - gamma) * M_est + gamma * M_wrong
    validate_column_stochastic(M_true)
    return M_true


def sample_weak_labels(y: Array, M_true: Array, *, seed: int) -> Array:
    y = np.asarray(y, dtype=np.intp)
    M_true = np.asarray(M_true, dtype=np.float64)
    validate_column_stochastic(M_true)
    d, c = M_true.shape
    if np.min(y) < 0 or np.max(y) >= c:
        raise ValueError(f"y must lie in [0,{c-1}].")
    rng = np.random.default_rng(seed)
    z = np.empty_like(y)
    for i, yi in enumerate(y):
        z[i] = int(rng.choice(d, p=M_true[:, yi]))
    return z


def build_weak_dataset_from_clean(
    clean: CleanSyntheticDataset,
    M_true: Array,
    *,
    seed: int,
    family_name: str,
) -> WeakLabelDataset:
    z = sample_weak_labels(clean.y, M_true, seed=seed)
    return WeakLabelDataset(
        X=np.asarray(clean.X, dtype=np.float64).copy(),
        y=np.asarray(clean.y, dtype=np.intp).copy(),
        eta=np.asarray(clean.eta, dtype=np.float64).copy(),
        logits=np.asarray(clean.logits, dtype=np.float64).copy(),
        z=z,
        M=np.asarray(M_true, dtype=np.float64).copy(),
        family_name=family_name,
        weak_label_matrix=None,
        weak_label_vectors=None,
        weak_label_names=["obs_0", "obs_1", "ambig_A", "ambig_B"],
        metadata={
            **clean.metadata,
            "family_name": family_name,
            "M_true": np.asarray(M_true, dtype=np.float64).tolist(),
            "num_weak_labels": int(M_true.shape[0]),
            "num_classes": int(M_true.shape[1]),
        },
    )


def build_weak_splits(
    clean_splits: dict[str, CleanSyntheticDataset],
    M_true: Array,
    *,
    seed: int,
    family_name: str,
) -> dict[str, WeakLabelDataset]:
    order = ["train", "val", "test"]
    out: dict[str, WeakLabelDataset] = {}
    for k, split in enumerate(order):
        out[split] = build_weak_dataset_from_clean(
            clean_splits[split],
            M_true,
            seed=seed + k * 1000,
            family_name=family_name,
        )
    return out


# ============================================================
# 3) Matrix diagnostics
# ============================================================

def standard_form_fit(A: Array) -> dict[str, float]:
    """Fit A ≈ lambda I + 1 v^T and return lambda and residual."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got {A.shape}.")
    c = A.shape[0]

    v = np.zeros(c, dtype=np.float64)
    for j in range(c):
        off = [A[i, j] for i in range(c) if i != j]
        v[j] = float(np.mean(off)) if off else 0.0
    lambda_hat = float(np.mean(np.diag(A) - v))
    A_hat = lambda_hat * np.eye(c, dtype=np.float64) + np.ones((c, 1), dtype=np.float64) @ v.reshape(1, -1)
    residual = float(np.linalg.norm(A - A_hat, ord="fro"))
    relative_residual = residual / max(float(np.linalg.norm(A, ord="fro")), 1e-12)
    return {
        "lambda_hat": lambda_hat,
        "standard_form_residual": residual,
        "standard_form_relative_residual": relative_residual,
    }


def print_matrix_diagnostics(T: Array, M_est: Array, M_wrong: Array, gamma_grid: list[float]) -> None:
    print("=== Semantic T ===")
    print(T)
    print("=== M_est ===")
    print(M_est)
    print("=== M_wrong ===")
    print(M_wrong)
    print("=== A_est = T @ M_est ===")
    print(T @ M_est)
    print("A_est fit:", standard_form_fit(T @ M_est))

    for gamma in gamma_grid:
        M_true = make_true_binary_ambiguous_M(M_est, M_wrong, gamma=gamma)
        A_true = T @ M_true
        print(f"\n=== gamma={gamma:.2f} ===")
        print("M_true=")
        print(M_true)
        print("A_true=T@M_true=")
        print(A_true)
        print("A_true fit:", standard_form_fit(A_true))


# ============================================================
# 4) Rectangular Forward trainer
# ============================================================

@dataclass
class ForwardRectEpochLog:
    epoch: int
    train_loss: float
    val_loss: Optional[float]
    train_clean_accuracy: float
    val_clean_accuracy: Optional[float]
    current_learning_rate: float


@dataclass
class ForwardRectTrainResult:
    model: nn.Module
    config: CLWLTrainConfig
    logs: list[ForwardRectEpochLog]
    final_train_metrics: dict[str, Any]
    final_val_metrics: Optional[dict[str, Any]]
    best_epoch: int
    best_monitor_value: float


class TorchWeakDataset(Dataset):
    def __init__(self, ds: WeakLabelDataset) -> None:
        self.X = torch.tensor(np.asarray(ds.X, dtype=np.float32), dtype=torch.float32)
        self.z = torch.tensor(np.asarray(ds.z, dtype=np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.z[idx]


class LinearScoreModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPScoreModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(input_dim: int, num_classes: int, config: CLWLTrainConfig) -> nn.Module:
    if config.model_type == "linear":
        return LinearScoreModel(input_dim, num_classes)
    if config.model_type == "mlp":
        return MLPScoreModel(input_dim, config.hidden_dim, num_classes)
    raise ValueError(f"Unsupported model_type={config.model_type}")


def make_loader(ds: WeakLabelDataset, batch_size: int, *, shuffle: bool) -> DataLoader:
    return DataLoader(TorchWeakDataset(ds), batch_size=batch_size, shuffle=shuffle)


def forward_rect_loss(
    logits: torch.Tensor,
    z: torch.Tensor,
    M_torch: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Rectangular forward correction.

    M_torch has shape (d,c), M[z,y]=P(z|y).
    logits has shape (n,c).
    probs_weak = softmax(logits) @ M.T has shape (n,d).
    """
    probs_clean = torch.softmax(logits, dim=1)
    probs_weak = probs_clean @ M_torch.transpose(0, 1)
    probs_weak = torch.clamp(probs_weak, min=eps)
    probs_weak = probs_weak / probs_weak.sum(dim=1, keepdim=True)
    return F.nll_loss(torch.log(probs_weak), z)


def forward_rect_empirical_risk(
    model: nn.Module,
    ds: WeakLabelDataset,
    M: Array,
    *,
    batch_size: int,
    device: str,
) -> float:
    device_obj = torch.device(device)
    M_torch = torch.tensor(np.asarray(M, dtype=np.float32), dtype=torch.float32, device=device_obj)
    loader = make_loader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for xb, zb in loader:
            xb = xb.to(device_obj)
            zb = zb.to(device_obj)
            loss = forward_rect_loss(model(xb), zb, M_torch)
            n = int(xb.shape[0])
            total += float(loss.item()) * n
            count += n
    return total / max(count, 1)


def scores_from_model(model: nn.Module, ds: WeakLabelDataset, *, batch_size: int, device: str) -> Array:
    device_obj = torch.device(device)
    loader = make_loader(ds, batch_size=batch_size, shuffle=False)
    outs: list[Array] = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device_obj)
            outs.append(model(xb).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outs, axis=0)


def evaluate_forward_rect_model(
    model: nn.Module,
    ds: WeakLabelDataset,
    M: Array,
    *,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    scores = scores_from_model(model, ds, batch_size=batch_size, device=device)
    metrics = evaluate_scores_on_dataset(scores, ds)
    risk = forward_rect_empirical_risk(model, ds, M, batch_size=batch_size, device=device)
    return {
        "clean_accuracy": float(metrics.clean_accuracy),
        "max_preservation_rate": float(metrics.max_preservation_rate),
        "pairwise_order_rate": float(metrics.pairwise_order_rate),
        "empirical_risk": float(risk),
    }


def clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_forward_rect_model(
    train_ds: WeakLabelDataset,
    M_train: Array,
    *,
    val_ds: Optional[WeakLabelDataset],
    config: CLWLTrainConfig,
) -> ForwardRectTrainResult:
    X = np.asarray(train_ds.X, dtype=np.float64)
    eta = np.asarray(train_ds.eta, dtype=np.float64)
    input_dim = X.shape[1]
    num_classes = eta.shape[1]
    d, c = np.asarray(M_train).shape
    if c != num_classes:
        raise ValueError(f"M_train has {c} clean classes but dataset has {num_classes}.")

    set_torch_seed(config.seed)
    device_obj = torch.device(config.device)
    model = build_model(input_dim, num_classes, config).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_learning_rate,
    )

    M_torch = torch.tensor(np.asarray(M_train, dtype=np.float32), dtype=torch.float32, device=device_obj)
    train_loader = make_loader(train_ds, config.batch_size, shuffle=True)

    logs: list[ForwardRectEpochLog] = []
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_monitor = float("inf")
    no_improve = 0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for xb, zb in train_loader:
            xb = xb.to(device_obj)
            zb = zb.to(device_obj)
            optimizer.zero_grad()
            loss = forward_rect_loss(model(xb), zb, M_torch)
            loss.backward()
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            n = int(xb.shape[0])
            total_loss += float(loss.item()) * n
            total_count += n

        train_loss = total_loss / max(total_count, 1)

        if epoch % config.log_every == 0 or epoch == config.num_epochs:
            train_metrics = evaluate_forward_rect_model(
                model,
                train_ds,
                M_train,
                batch_size=max(config.batch_size, 512),
                device=config.device,
            )
            val_metrics = None
            if val_ds is not None:
                val_metrics = evaluate_forward_rect_model(
                    model,
                    val_ds,
                    M_train,
                    batch_size=max(config.batch_size, 512),
                    device=config.device,
                )
            monitor = train_loss if val_metrics is None else float(val_metrics["empirical_risk"])
            scheduler.step(monitor)
            lr = float(optimizer.param_groups[0]["lr"])

            logs.append(
                ForwardRectEpochLog(
                    epoch=epoch,
                    train_loss=float(train_loss),
                    val_loss=None if val_metrics is None else float(val_metrics["empirical_risk"]),
                    train_clean_accuracy=float(train_metrics["clean_accuracy"]),
                    val_clean_accuracy=None if val_metrics is None else float(val_metrics["clean_accuracy"]),
                    current_learning_rate=lr,
                )
            )

            if monitor < best_monitor - config.early_stop_min_delta:
                best_monitor = float(monitor)
                best_epoch = epoch
                best_state = clone_state_dict_to_cpu(model)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= config.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train = evaluate_forward_rect_model(
        model,
        train_ds,
        M_train,
        batch_size=max(config.batch_size, 512),
        device=config.device,
    )
    final_val = None
    if val_ds is not None:
        final_val = evaluate_forward_rect_model(
            model,
            val_ds,
            M_train,
            batch_size=max(config.batch_size, 512),
            device=config.device,
        )

    return ForwardRectTrainResult(
        model=model,
        config=config,
        logs=logs,
        final_train_metrics=final_train,
        final_val_metrics=final_val,
        best_epoch=int(best_epoch),
        best_monitor_value=float(best_monitor),
    )


# ============================================================
# 5) Experiment runner
# ============================================================

def run_binary_semantic_ambiguity_experiment(
    *,
    gamma_grid: list[float],
    seeds: list[int],
    n: int = 6000,
    input_dim: int = 8,
    ambiguity_mass: float = 0.65,
    clean_flip: float = 0.10,
    est_split: float = 0.90,
    include_forward_oracle: bool = True,
    config: Optional[CLWLTrainConfig] = None,
    out_dir: str = "artifacts_binary_semantic_ambiguity_clwl_vs_forward",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config is None:
        config = CLWLTrainConfig(
            model_type="mlp",
            hidden_dim=128,
            batch_size=256,
            num_epochs=80,
            learning_rate=3e-4,
            weight_decay=1e-4,
            device="cpu",
            seed=0,
            log_every=1,
            gradient_clip_norm=5.0,
            scheduler_factor=0.5,
            scheduler_patience=5,
            min_learning_rate=1e-5,
            early_stop_patience=12,
            early_stop_min_delta=1e-8,
        )

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    T = make_semantic_T()
    M_est = make_estimated_binary_ambiguous_M(
        ambiguity_mass=ambiguity_mass,
        clean_flip=clean_flip,
        est_split=est_split,
    )
    M_wrong = make_wrong_binary_ambiguous_M(M_est)

    print_matrix_diagnostics(T, M_est, M_wrong, gamma_grid)

    np.save(Path(out_dir) / "T_semantic.npy", T)
    np.save(Path(out_dir) / "M_est.npy", M_est)
    np.save(Path(out_dir) / "M_wrong.npy", M_wrong)

    rows: list[dict[str, Any]] = []

    for gamma in gamma_grid:
        M_true = make_true_binary_ambiguous_M(M_est, M_wrong, gamma=gamma)
        A_true = T @ M_true
        fit_true = standard_form_fit(A_true)
        np.save(Path(out_dir) / f"M_true_gamma_{gamma:.3f}.npy", M_true)

        for seed in seeds:
            clean = make_binary_synthetic_dataset(
                n=n,
                input_dim=input_dim,
                seed=1000 + seed,
                label_seed=2000 + seed,
                margin_scale=1.6,
            )
            clean_splits = train_val_test_split_clean(clean, train_frac=0.6, val_frac=0.2, seed=3000 + seed)
            weak_splits = build_weak_splits(
                clean_splits,
                M_true,
                seed=4000 + seed,
                family_name="binary_semantic_ambiguity_swap_true_M",
            )

            cfg = replace(config, seed=seed)

            # ----- CLWL with semantic T -----
            clwl_result = train_clwl_model(
                train_dataset=weak_splits["train"],
                val_dataset=weak_splits["val"],
                T=T,
                config=cfg,
            )
            clwl_test = evaluate_clwl_model_on_dataset(
                clwl_result.model,
                weak_splits["test"],
                T,
                batch_size=max(cfg.batch_size, 512),
                device=cfg.device,
            )
            rows.append({
                "method": "CLWL_semantic_T",
                "seed": seed,
                "gamma_mismatch": gamma,
                "ambiguity_mass": ambiguity_mass,
                "clean_flip": clean_flip,
                "est_split": est_split,
                "test_clean_accuracy": float(clwl_test["clean_accuracy"]),
                "test_pairwise_order_rate": float(clwl_test["pairwise_order_rate"]),
                "test_max_preservation_rate": float(clwl_test["max_preservation_rate"]),
                "test_empirical_risk": float(clwl_test["empirical_risk"]),
                "best_epoch": int(clwl_result.best_epoch),
                "A_true_lambda": fit_true["lambda_hat"],
                "A_true_residual": fit_true["standard_form_residual"],
                "A_true_relative_residual": fit_true["standard_form_relative_residual"],
            })

            # ----- Forward using misspecified M_est -----
            fwd_result = train_forward_rect_model(
                train_ds=weak_splits["train"],
                val_ds=weak_splits["val"],
                M_train=M_est,
                config=cfg,
            )
            fwd_test = evaluate_forward_rect_model(
                fwd_result.model,
                weak_splits["test"],
                M_est,
                batch_size=max(cfg.batch_size, 512),
                device=cfg.device,
            )
            rows.append({
                "method": "Forward_Mest",
                "seed": seed,
                "gamma_mismatch": gamma,
                "ambiguity_mass": ambiguity_mass,
                "clean_flip": clean_flip,
                "est_split": est_split,
                "test_clean_accuracy": float(fwd_test["clean_accuracy"]),
                "test_pairwise_order_rate": float(fwd_test["pairwise_order_rate"]),
                "test_max_preservation_rate": float(fwd_test["max_preservation_rate"]),
                "test_empirical_risk": float(fwd_test["empirical_risk"]),
                "best_epoch": int(fwd_result.best_epoch),
                "A_true_lambda": np.nan,
                "A_true_residual": np.nan,
                "A_true_relative_residual": np.nan,
            })

            # ----- Optional oracle Forward using M_true -----
            if include_forward_oracle:
                fwd_oracle_result = train_forward_rect_model(
                    train_ds=weak_splits["train"],
                    val_ds=weak_splits["val"],
                    M_train=M_true,
                    config=cfg,
                )
                fwd_oracle_test = evaluate_forward_rect_model(
                    fwd_oracle_result.model,
                    weak_splits["test"],
                    M_true,
                    batch_size=max(cfg.batch_size, 512),
                    device=cfg.device,
                )
                rows.append({
                    "method": "Forward_oracle_Mtrue",
                    "seed": seed,
                    "gamma_mismatch": gamma,
                    "ambiguity_mass": ambiguity_mass,
                    "clean_flip": clean_flip,
                    "est_split": est_split,
                    "test_clean_accuracy": float(fwd_oracle_test["clean_accuracy"]),
                    "test_pairwise_order_rate": float(fwd_oracle_test["pairwise_order_rate"]),
                    "test_max_preservation_rate": float(fwd_oracle_test["max_preservation_rate"]),
                    "test_empirical_risk": float(fwd_oracle_test["empirical_risk"]),
                    "best_epoch": int(fwd_oracle_result.best_epoch),
                    "A_true_lambda": np.nan,
                    "A_true_residual": np.nan,
                    "A_true_relative_residual": np.nan,
                })

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(Path(out_dir) / "raw_results.csv", index=False)

    summary_df = (
        raw_df
        .groupby(["method", "gamma_mismatch"], as_index=False)
        .agg({
            "test_clean_accuracy": ["mean", "std"],
            "test_pairwise_order_rate": ["mean", "std"],
            "test_max_preservation_rate": ["mean", "std"],
            "test_empirical_risk": ["mean", "std"],
            "best_epoch": ["mean", "std"],
            "A_true_lambda": ["mean", "std"],
            "A_true_residual": ["mean", "std"],
            "A_true_relative_residual": ["mean", "std"],
        })
    )
    summary_df.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary_df.columns]
    summary_df.to_csv(Path(out_dir) / "summary_results.csv", index=False)

    return raw_df, summary_df


# ============================================================
# 6) Plotting
# ============================================================

def _std_array(df: pd.DataFrame, col: str) -> Array:
    if col not in df.columns:
        return np.zeros(len(df), dtype=float)
    return np.nan_to_num(df[col].to_numpy(dtype=float), nan=0.0)


def plot_binary_semantic_results(
    summary_df: pd.DataFrame,
    *,
    out_path: str = "binary_semantic_ambiguity_clwl_vs_forward.png",
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    method_order = [m for m in ["CLWL_semantic_T", "Forward_Mest", "Forward_oracle_Mtrue"] if m in set(summary_df["method"])]
    for method in method_order:
        dfm = summary_df[summary_df["method"] == method].sort_values("gamma_mismatch")
        x = dfm["gamma_mismatch"].to_numpy(dtype=float)

        acc = dfm["test_clean_accuracy_mean"].to_numpy(dtype=float)
        acc_std = _std_array(dfm, "test_clean_accuracy_std")
        axes[0].plot(x, acc, marker="o", label=method)
        axes[0].fill_between(x, acc - acc_std, acc + acc_std, alpha=0.18)

        pair = dfm["test_pairwise_order_rate_mean"].to_numpy(dtype=float)
        pair_std = _std_array(dfm, "test_pairwise_order_rate_std")
        axes[1].plot(x, pair, marker="o", label=method)
        axes[1].fill_between(x, pair - pair_std, pair + pair_std, alpha=0.18)

    axes[0].set_xlabel("A/B swap strength gamma")
    axes[0].set_ylabel("test clean accuracy")
    axes[0].set_title("Binary synthetic: clean accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("A/B swap strength gamma")
    axes[1].set_ylabel("test pairwise order rate")
    axes[1].set_title("Binary synthetic: ranking")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    dfc = summary_df[summary_df["method"] == "CLWL_semantic_T"].sort_values("gamma_mismatch")
    dff = summary_df[summary_df["method"] == "Forward_Mest"].sort_values("gamma_mismatch")
    x = dfc["gamma_mismatch"].to_numpy(dtype=float)
    gap = dfc["test_clean_accuracy_mean"].to_numpy(dtype=float) - dff["test_clean_accuracy_mean"].to_numpy(dtype=float)
    axes[2].plot(x, gap, marker="o", label="CLWL - Forward_Mest")
    axes[2].axhline(0.0, linestyle="--", alpha=0.7)
    axes[2].set_xlabel("A/B swap strength gamma")
    axes[2].set_ylabel("accuracy gap")
    axes[2].set_title("Gap to misspecified Forward")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_A_true_diagnostics(
    summary_df: pd.DataFrame,
    *,
    out_path: str = "A_true_diagnostics.png",
) -> None:
    import matplotlib.pyplot as plt

    dfc = summary_df[summary_df["method"] == "CLWL_semantic_T"].sort_values("gamma_mismatch")
    x = dfc["gamma_mismatch"].to_numpy(dtype=float)
    lam = dfc["A_true_lambda_mean"].to_numpy(dtype=float)
    resid = dfc["A_true_relative_residual_mean"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(x, lam, marker="o")
    axes[0].axhline(0.0, linestyle="--", alpha=0.7)
    axes[0].set_xlabel("A/B swap strength gamma")
    axes[0].set_ylabel("lambda_hat for T M_true")
    axes[0].set_title("CLWL true induced gain")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, resid, marker="o")
    axes[1].set_xlabel("A/B swap strength gamma")
    axes[1].set_ylabel("relative residual")
    axes[1].set_title("Residual of T M_true standard form")
    axes[1].grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7) Main
# ============================================================

if __name__ == "__main__":
    out_dir = "artifacts_binary_semantic_ambiguity_clwl_vs_forward"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    cfg = CLWLTrainConfig(
        model_type="mlp",
        hidden_dim=128,
        batch_size=256,
        num_epochs=80,
        learning_rate=3e-4,
        weight_decay=1e-4,
        device=device,
        seed=0,
        log_every=1,
        gradient_clip_norm=5.0,
        scheduler_factor=0.5,
        scheduler_patience=5,
        min_learning_rate=1e-5,
        early_stop_patience=12,
        early_stop_min_delta=1e-8,
    )

    raw_df, summary_df = run_binary_semantic_ambiguity_experiment(
        gamma_grid=[0.0, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00],
        seeds=[0, 1, 2, 3, 4],
        n=6000,
        input_dim=8,
        ambiguity_mass=0.65,
        clean_flip=0.10,
        est_split=0.90,
        include_forward_oracle=True,
        config=cfg,
        out_dir=out_dir,
    )

    print("\n=== Raw results head ===")
    print(raw_df.head())
    print("\n=== Summary results ===")
    print(summary_df)

    plot_binary_semantic_results(
        summary_df,
        out_path=str(Path(out_dir) / "binary_semantic_ambiguity_clwl_vs_forward.png"),
    )
    plot_A_true_diagnostics(
        summary_df,
        out_path=str(Path(out_dir) / "A_true_diagnostics.png"),
    )

    print("\nSaved outputs to:")
    print(Path(out_dir).resolve())
