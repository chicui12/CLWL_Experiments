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
# Module 21: Setting-A robustness experiment
# ============================================================
# Practical logic:
#   1. Choose an estimated weak-label transition matrix M_hat.
#   2. Construct T_hat = T(M_hat) using the same module1 CLWL construction.
#   3. Construct true matrices M_true = M_hat + drift * Delta, where
#          T_hat Delta = 0,
#          1^T Delta = 0,
#          M_true >= 0.
#   4. Compare:
#          CLWL using T_hat,
#          Forward using M_hat,
#          Oracle Forward using M_true.
#
# Practical meaning of M_hat:
#   Binary clean task with four weak outputs:
#       z0: class-0-like weak label
#       z1: class-1-like weak label
#       z2: auxiliary weak bucket A
#       z3: auxiliary weak bucket B
#   M_hat is a pilot estimate of how often each clean class generates each weak
#   output. M_true differs by a deployment-time redistribution among weak
#   outputs that lies in the nullspace of T_hat. This is not a semantic T setup:
#   T_hat is computed from M_hat.
#
# Important expected conclusion:
#   This experiment may NOT produce a Forward collapse. In fact, for the generic
#   construction T(M_hat), exact CLWL-friendly perturbations often also induce an
#   order-preserving clean map for Forward. The module therefore records
#       B = pinv(M_hat) M_true
#   to check whether Forward is actually harmed in ranking.


# ============================================================
# 1) Synthetic binary clean data
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
    margin_scale: float = 1.8,
) -> CleanSyntheticDataset:
    rng = np.random.default_rng(seed)
    label_rng = np.random.default_rng(label_seed)

    X = rng.normal(size=(n, input_dim)).astype(np.float64)
    w = rng.normal(size=input_dim).astype(np.float64)
    w /= max(np.linalg.norm(w), 1e-12)

    raw = X @ w
    if input_dim >= 3:
        raw += 0.35 * np.sin(X[:, 0]) - 0.25 * np.cos(X[:, 1]) + 0.15 * X[:, 2] ** 2
    raw *= margin_scale

    p1 = sigmoid(raw)
    eta = np.stack([1.0 - p1, p1], axis=1)
    y = np.asarray([label_rng.choice(2, p=p) for p in eta], dtype=np.intp)
    logits = np.stack([np.zeros_like(raw), raw], axis=1)

    return CleanSyntheticDataset(
        X=X,
        y=y,
        eta=eta,
        logits=logits,
        metadata={
            "dataset_name": "binary_synthetic_teacher",
            "n": int(n),
            "input_dim": int(input_dim),
            "eta_source": "oracle_binary_teacher",
            "oracle_metrics": True,
        },
    )


def split_clean_dataset(
    ds: CleanSyntheticDataset,
    *,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> dict[str, CleanSyntheticDataset]:
    n = ds.X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    ids = {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }
    out: dict[str, CleanSyntheticDataset] = {}
    for split, id_split in ids.items():
        out[split] = CleanSyntheticDataset(
            X=ds.X[id_split].copy(),
            y=ds.y[id_split].copy(),
            eta=ds.eta[id_split].copy(),
            logits=ds.logits[id_split].copy(),
            metadata={**ds.metadata, "split": split},
        )
    return out


# ============================================================
# 2) Estimated matrix and nullspace true-matrix construction
# ============================================================

def validate_transition(M: Array, *, atol: float = 1e-8) -> None:
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"M must be 2D, got {M.shape}.")
    if np.min(M) < -atol:
        raise ValueError(f"M has negative entries: min={np.min(M):.3e}.")
    if not np.allclose(M.sum(axis=0), np.ones(M.shape[1]), atol=atol, rtol=0.0):
        raise ValueError(f"M must be column stochastic. col_sums={M.sum(axis=0)}")


def make_practical_estimated_Mhat() -> Array:
    """
    Pilot-estimated transition matrix M_hat in R^{4 x 2}.

    Rows:
      0: class-0-like weak label
      1: class-1-like weak label
      2: auxiliary weak bucket A
      3: auxiliary weak bucket B

    Columns are clean classes y=0 and y=1.
    """
    M_hat = np.array(
        [
            [0.42, 0.08],
            [0.08, 0.42],
            [0.35, 0.15],
            [0.15, 0.35],
        ],
        dtype=np.float64,
    )
    validate_transition(M_hat)
    if np.linalg.matrix_rank(M_hat) != 2:
        raise ValueError("M_hat must be full column rank.")
    return M_hat


def nullspace_basis(C: Array, *, tol: float = 1e-10) -> Array:
    C = np.asarray(C, dtype=np.float64)
    _, s, vt = np.linalg.svd(C, full_matrices=True)
    rank = int(np.sum(s > tol))
    return vt[rank:].T


def max_nonnegative_scale(M: Array, Delta: Array) -> float:
    M = np.asarray(M, dtype=np.float64)
    Delta = np.asarray(Delta, dtype=np.float64)
    candidates = []
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if Delta[i, j] < -1e-12:
                candidates.append(M[i, j] / (-Delta[i, j]))
    if not candidates:
        return float("inf")
    return float(min(candidates))


def standard_form_fit(A: Array) -> dict[str, float]:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got {A.shape}.")
    c = A.shape[0]
    v = np.zeros(c, dtype=np.float64)
    for j in range(c):
        off = [A[i, j] for i in range(c) if i != j]
        v[j] = float(np.mean(off)) if off else 0.0
    lam = float(np.mean(np.diag(A) - v))
    A_hat = lam * np.eye(c) + np.ones((c, 1)) @ v.reshape(1, -1)
    residual = float(np.linalg.norm(A - A_hat, ord="fro"))
    relative = residual / max(float(np.linalg.norm(A, ord="fro")), 1e-12)
    return {"lambda_hat": lam, "residual": residual, "relative_residual": relative}


def construct_nullspace_delta(
    M_hat: Array,
    T_hat: Array,
    *,
    search_seed: int = 0,
    num_random_candidates: int = 2000,
    target_drift_fraction: float = 0.95,
) -> tuple[Array, dict[str, Any]]:
    """
    Construct Delta satisfying:
        T_hat Delta = 0,
        1^T Delta = 0.

    We search over the nullspace of [T_hat; 1^T] and pick the direction that
    gives the smallest ranking factor for B=pinv(M_hat)(M_hat+Delta_scaled)
    at the maximum allowed drift. If the theory makes Forward also friendly,
    this diagnostic will reveal it.
    """
    M_hat = np.asarray(M_hat, dtype=np.float64)
    T_hat = np.asarray(T_hat, dtype=np.float64)
    d, c = M_hat.shape
    if c != 2:
        raise ValueError("This module is written for binary c=2.")

    C = np.vstack([T_hat, np.ones((1, d), dtype=np.float64)])
    basis = nullspace_basis(C)
    if basis.shape[1] == 0:
        raise ValueError("No nonzero u satisfies T_hat u=0 and 1^T u=0.")

    rng = np.random.default_rng(search_seed)
    candidate_us = []
    for k in range(basis.shape[1]):
        candidate_us.append(basis[:, k])
        candidate_us.append(-basis[:, k])
    for _ in range(num_random_candidates):
        coeff = rng.normal(size=basis.shape[1])
        u = basis @ coeff
        candidate_us.append(u)
        candidate_us.append(-u)

    best: Optional[dict[str, Any]] = None
    for u in candidate_us:
        norm = float(np.max(np.abs(u)))
        if norm < 1e-12:
            continue
        u = u / norm
        Delta_unit = np.outer(u, np.array([1.0, -1.0], dtype=np.float64))
        gamma_max = max_nonnegative_scale(M_hat, Delta_unit)
        if not np.isfinite(gamma_max) or gamma_max <= 1e-12:
            continue
        Delta_scaled = gamma_max * Delta_unit
        M_test = M_hat + target_drift_fraction * Delta_scaled
        if np.min(M_test) < -1e-9:
            continue
        B_test = np.linalg.pinv(M_hat) @ M_test
        B_fit = standard_form_fit(B_test)
        score = B_fit["lambda_hat"]
        record = {
            "u": u,
            "Delta_scaled": Delta_scaled,
            "gamma_max": gamma_max,
            "B_lambda_at_target": B_fit["lambda_hat"],
            "B_residual_at_target": B_fit["residual"],
            "B_relative_residual_at_target": B_fit["relative_residual"],
            "target_drift_fraction": target_drift_fraction,
        }
        if best is None or score < best["B_lambda_at_target"]:
            best = record

    if best is None:
        raise RuntimeError("Failed to find a feasible nullspace perturbation.")

    Delta = best["Delta_scaled"]
    checks = {
        **best,
        "TDelta_norm": float(np.linalg.norm(T_hat @ Delta, ord="fro")),
        "column_sum_Delta_norm": float(np.linalg.norm(Delta.sum(axis=0))),
        "basis_dimension": int(basis.shape[1]),
    }
    return Delta, checks


def make_true_M(M_hat: Array, Delta: Array, drift_fraction: float) -> Array:
    if not (0.0 <= drift_fraction <= 1.0):
        raise ValueError(f"drift_fraction must be in [0,1], got {drift_fraction}.")
    M_true = np.asarray(M_hat, dtype=np.float64) + drift_fraction * np.asarray(Delta, dtype=np.float64)
    M_true = np.where(np.abs(M_true) < 1e-12, 0.0, M_true)
    validate_transition(M_true)
    return M_true


def describe_delta(Delta: Array) -> str:
    names = ["z0_class0_like", "z1_class1_like", "z2_aux_A", "z3_aux_B"]
    lines = []
    for y in [0, 1]:
        col = Delta[:, y]
        inc = [(names[i], float(col[i])) for i in range(len(names)) if col[i] > 1e-10]
        dec = [(names[i], float(col[i])) for i in range(len(names)) if col[i] < -1e-10]
        lines.append(f"class {y}: increases={inc}, decreases={dec}")
    return "\n".join(lines)


# ============================================================
# 3) Weak-label dataset construction
# ============================================================

def sample_weak_labels(y: Array, M: Array, *, seed: int) -> Array:
    y = np.asarray(y, dtype=np.intp)
    M = np.asarray(M, dtype=np.float64)
    rng = np.random.default_rng(seed)
    z = np.empty_like(y)
    for i, yi in enumerate(y):
        z[i] = int(rng.choice(M.shape[0], p=M[:, yi]))
    return z


def build_weak_dataset(clean: CleanSyntheticDataset, M_true: Array, *, seed: int) -> WeakLabelDataset:
    z = sample_weak_labels(clean.y, M_true, seed=seed)
    return WeakLabelDataset(
        X=np.asarray(clean.X, dtype=np.float64).copy(),
        y=np.asarray(clean.y, dtype=np.intp).copy(),
        eta=np.asarray(clean.eta, dtype=np.float64).copy(),
        logits=np.asarray(clean.logits, dtype=np.float64).copy(),
        z=z,
        M=np.asarray(M_true, dtype=np.float64).copy(),
        family_name="settingA_nullspace_true_M",
        weak_label_matrix=None,
        weak_label_vectors=None,
        weak_label_names=["z0_class0_like", "z1_class1_like", "z2_aux_A", "z3_aux_B"],
        metadata={
            **clean.metadata,
            "family_name": "settingA_nullspace_true_M",
            "M_true": np.asarray(M_true, dtype=np.float64).tolist(),
            "num_weak_labels": int(M_true.shape[0]),
            "num_classes": int(M_true.shape[1]),
        },
    )


def build_weak_splits(clean_splits: dict[str, CleanSyntheticDataset], M_true: Array, *, seed: int) -> dict[str, WeakLabelDataset]:
    return {
        "train": build_weak_dataset(clean_splits["train"], M_true, seed=seed + 0),
        "val": build_weak_dataset(clean_splits["val"], M_true, seed=seed + 1000),
        "test": build_weak_dataset(clean_splits["test"], M_true, seed=seed + 2000),
    }


# ============================================================
# 4) Rectangular Forward trainer
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
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearScoreModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def build_model(input_dim: int, num_classes: int, cfg: CLWLTrainConfig) -> nn.Module:
    if cfg.model_type == "linear":
        return LinearScoreModel(input_dim, num_classes)
    if cfg.model_type == "mlp":
        return MLPScoreModel(input_dim, cfg.hidden_dim, num_classes)
    raise ValueError(f"Unsupported model_type={cfg.model_type}")


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
    device_obj = torch.device(device)
    loader = make_loader(ds, batch_size=batch_size, shuffle=False)
    outs: list[Array] = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device_obj)
            outs.append(model(xb).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outs, axis=0)


def forward_rect_empirical_risk(model: nn.Module, ds: WeakLabelDataset, M: Array, *, batch_size: int, device: str) -> float:
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


def evaluate_forward_rect_model(model: nn.Module, ds: WeakLabelDataset, M: Array, *, batch_size: int, device: str) -> dict[str, Any]:
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


def train_forward_rect_model(train_ds: WeakLabelDataset, M_train: Array, *, val_ds: Optional[WeakLabelDataset], cfg: CLWLTrainConfig):
    X = np.asarray(train_ds.X, dtype=np.float64)
    c = np.asarray(train_ds.eta).shape[1]
    set_torch_seed(cfg.seed)
    device_obj = torch.device(cfg.device)
    model = build_model(X.shape[1], c, cfg).to(device_obj)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        min_lr=cfg.min_learning_rate,
    )
    M_torch = torch.tensor(np.asarray(M_train, dtype=np.float32), dtype=torch.float32, device=device_obj)
    loader = make_loader(train_ds, cfg.batch_size, shuffle=True)

    best_state = None
    best_epoch = 0
    best_monitor = float("inf")
    no_improve = 0

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, zb in loader:
            xb = xb.to(device_obj)
            zb = zb.to(device_obj)
            opt.zero_grad()
            loss = forward_rect_loss(model(xb), zb, M_torch)
            loss.backward()
            if cfg.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            opt.step()
            n = int(xb.shape[0])
            total += float(loss.item()) * n
            count += n

        train_loss = total / max(count, 1)
        if epoch % cfg.log_every == 0 or epoch == cfg.num_epochs:
            if val_ds is not None:
                val_metrics = evaluate_forward_rect_model(model, val_ds, M_train, batch_size=max(cfg.batch_size, 512), device=cfg.device)
                monitor = float(val_metrics["empirical_risk"])
            else:
                monitor = train_loss
            sched.step(monitor)
            if monitor < best_monitor - cfg.early_stop_min_delta:
                best_monitor = float(monitor)
                best_epoch = epoch
                best_state = clone_state_dict_to_cpu(model)
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_monitor


# ============================================================
# 5) Experiment runner
# ============================================================

def run_settingA_nullspace_experiment(
    *,
    drift_grid: list[float],
    seeds: list[int],
    n: int = 6000,
    input_dim: int = 8,
    cfg: Optional[CLWLTrainConfig] = None,
    out_dir: str = "artifacts_settingA_nullspace_Mhat_robustness",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if cfg is None:
        cfg = CLWLTrainConfig(
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

    M_hat = make_practical_estimated_Mhat()
    construction = construct_clwl_T(M_hat)
    T_hat = construction.T
    Delta, delta_info = construct_nullspace_delta(M_hat, T_hat)

    print("=== Practical estimated M_hat ===")
    print(M_hat)
    print("=== T_hat = T(M_hat) from module1 ===")
    print(T_hat)
    print("=== A_hat = T_hat @ M_hat ===")
    print(T_hat @ M_hat)
    print("A_hat fit:", standard_form_fit(T_hat @ M_hat))
    print("=== Nullspace Delta ===")
    print(Delta)
    print("Delta practical meaning:")
    print(describe_delta(Delta))
    print("Delta checks:", {k: v for k, v in delta_info.items() if k != "u" and k != "Delta_scaled"})

    np.save(Path(out_dir) / "M_hat.npy", M_hat)
    np.save(Path(out_dir) / "T_hat.npy", T_hat)
    np.save(Path(out_dir) / "Delta.npy", Delta)

    rows: list[dict[str, Any]] = []

    for drift in drift_grid:
        M_true = make_true_M(M_hat, Delta, drift)
        A_true = T_hat @ M_true
        A_fit = standard_form_fit(A_true)
        B = np.linalg.pinv(M_hat) @ M_true
        B_fit = standard_form_fit(B)
        np.save(Path(out_dir) / f"M_true_drift_{drift:.3f}.npy", M_true)

        print(f"\n=== drift_fraction={drift:.2f} ===")
        print("M_true=")
        print(M_true)
        print("A_true=T_hat@M_true fit:", A_fit)
        print("B=pinv(M_hat)@M_true fit:", B_fit)

        for seed in seeds:
            clean = make_binary_synthetic_dataset(
                n=n,
                input_dim=input_dim,
                seed=1000 + seed,
                label_seed=2000 + seed,
                margin_scale=1.8,
            )
            clean_splits = split_clean_dataset(clean, seed=3000 + seed)
            weak_splits = build_weak_splits(clean_splits, M_true, seed=4000 + seed)
            cfg_seed = replace(cfg, seed=seed)

            clwl_result = train_clwl_model(
                train_dataset=weak_splits["train"],
                val_dataset=weak_splits["val"],
                T=T_hat,
                config=cfg_seed,
            )
            clwl_test = evaluate_clwl_model_on_dataset(
                clwl_result.model,
                weak_splits["test"],
                T_hat,
                batch_size=max(cfg_seed.batch_size, 512),
                device=cfg_seed.device,
            )
            rows.append({
                "method": "CLWL_T_Mhat",
                "seed": seed,
                "drift_fraction": drift,
                "test_clean_accuracy": float(clwl_test["clean_accuracy"]),
                "test_pairwise_order_rate": float(clwl_test["pairwise_order_rate"]),
                "test_max_preservation_rate": float(clwl_test["max_preservation_rate"]),
                "test_empirical_risk": float(clwl_test["empirical_risk"]),
                "best_epoch": int(clwl_result.best_epoch),
                "A_true_lambda": A_fit["lambda_hat"],
                "A_true_residual": A_fit["residual"],
                "A_true_relative_residual": A_fit["relative_residual"],
                "B_lambda": B_fit["lambda_hat"],
                "B_residual": B_fit["residual"],
                "B_relative_residual": B_fit["relative_residual"],
            })

            fwd_model, fwd_epoch, fwd_monitor = train_forward_rect_model(
                train_ds=weak_splits["train"],
                val_ds=weak_splits["val"],
                M_train=M_hat,
                cfg=cfg_seed,
            )
            fwd_test = evaluate_forward_rect_model(
                fwd_model,
                weak_splits["test"],
                M_hat,
                batch_size=max(cfg_seed.batch_size, 512),
                device=cfg_seed.device,
            )
            rows.append({
                "method": "Forward_Mhat",
                "seed": seed,
                "drift_fraction": drift,
                "test_clean_accuracy": float(fwd_test["clean_accuracy"]),
                "test_pairwise_order_rate": float(fwd_test["pairwise_order_rate"]),
                "test_max_preservation_rate": float(fwd_test["max_preservation_rate"]),
                "test_empirical_risk": float(fwd_test["empirical_risk"]),
                "best_epoch": int(fwd_epoch),
                "A_true_lambda": np.nan,
                "A_true_residual": np.nan,
                "A_true_relative_residual": np.nan,
                "B_lambda": B_fit["lambda_hat"],
                "B_residual": B_fit["residual"],
                "B_relative_residual": B_fit["relative_residual"],
            })

            oracle_model, oracle_epoch, oracle_monitor = train_forward_rect_model(
                train_ds=weak_splits["train"],
                val_ds=weak_splits["val"],
                M_train=M_true,
                cfg=cfg_seed,
            )
            oracle_test = evaluate_forward_rect_model(
                oracle_model,
                weak_splits["test"],
                M_true,
                batch_size=max(cfg_seed.batch_size, 512),
                device=cfg_seed.device,
            )
            rows.append({
                "method": "Forward_oracle_Mtrue",
                "seed": seed,
                "drift_fraction": drift,
                "test_clean_accuracy": float(oracle_test["clean_accuracy"]),
                "test_pairwise_order_rate": float(oracle_test["pairwise_order_rate"]),
                "test_max_preservation_rate": float(oracle_test["max_preservation_rate"]),
                "test_empirical_risk": float(oracle_test["empirical_risk"]),
                "best_epoch": int(oracle_epoch),
                "A_true_lambda": np.nan,
                "A_true_residual": np.nan,
                "A_true_relative_residual": np.nan,
                "B_lambda": B_fit["lambda_hat"],
                "B_residual": B_fit["residual"],
                "B_relative_residual": B_fit["relative_residual"],
            })

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(Path(out_dir) / "raw_results.csv", index=False)

    summary_df = (
        raw_df
        .groupby(["method", "drift_fraction"], as_index=False)
        .agg({
            "test_clean_accuracy": ["mean", "std"],
            "test_pairwise_order_rate": ["mean", "std"],
            "test_max_preservation_rate": ["mean", "std"],
            "test_empirical_risk": ["mean", "std"],
            "best_epoch": ["mean", "std"],
            "A_true_lambda": ["mean", "std"],
            "A_true_residual": ["mean", "std"],
            "A_true_relative_residual": ["mean", "std"],
            "B_lambda": ["mean", "std"],
            "B_residual": ["mean", "std"],
            "B_relative_residual": ["mean", "std"],
        })
    )
    summary_df.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary_df.columns]
    summary_df.to_csv(Path(out_dir) / "summary_results.csv", index=False)
    return raw_df, summary_df


# ============================================================
# 6) Plots
# ============================================================

def _std(df: pd.DataFrame, col: str) -> Array:
    if col not in df.columns:
        return np.zeros(len(df), dtype=float)
    return np.nan_to_num(df[col].to_numpy(dtype=float), nan=0.0)


def plot_results(summary_df: pd.DataFrame, *, out_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    methods = ["CLWL_T_Mhat", "Forward_Mhat", "Forward_oracle_Mtrue"]
    for method in methods:
        if method not in set(summary_df["method"]):
            continue
        dfm = summary_df[summary_df["method"] == method].sort_values("drift_fraction")
        x = dfm["drift_fraction"].to_numpy(dtype=float)
        acc = dfm["test_clean_accuracy_mean"].to_numpy(dtype=float)
        acc_std = _std(dfm, "test_clean_accuracy_std")
        pair = dfm["test_pairwise_order_rate_mean"].to_numpy(dtype=float)
        pair_std = _std(dfm, "test_pairwise_order_rate_std")

        axes[0].plot(x, acc, marker="o", label=method)
        axes[0].fill_between(x, acc - acc_std, acc + acc_std, alpha=0.18)
        axes[1].plot(x, pair, marker="o", label=method)
        axes[1].fill_between(x, pair - pair_std, pair + pair_std, alpha=0.18)

    axes[0].set_xlabel("drift fraction")
    axes[0].set_ylabel("test clean accuracy")
    axes[0].set_title("Setting-A nullspace drift: accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("drift fraction")
    axes[1].set_ylabel("pairwise order rate")
    axes[1].set_title("Setting-A nullspace drift: ranking")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    dfc = summary_df[summary_df["method"] == "CLWL_T_Mhat"].sort_values("drift_fraction")
    dff = summary_df[summary_df["method"] == "Forward_Mhat"].sort_values("drift_fraction")
    x = dfc["drift_fraction"].to_numpy(dtype=float)
    gap = dfc["test_clean_accuracy_mean"].to_numpy(dtype=float) - dff["test_clean_accuracy_mean"].to_numpy(dtype=float)
    axes[2].plot(x, gap, marker="o", label="CLWL - Forward_Mhat")
    axes[2].axhline(0.0, linestyle="--", alpha=0.7)
    axes[2].set_xlabel("drift fraction")
    axes[2].set_ylabel("accuracy gap")
    axes[2].set_title("Gap to misspecified Forward")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(summary_df: pd.DataFrame, *, out_path: str) -> None:
    import matplotlib.pyplot as plt

    dfc = summary_df[summary_df["method"] == "CLWL_T_Mhat"].sort_values("drift_fraction")
    x = dfc["drift_fraction"].to_numpy(dtype=float)
    A_lam = dfc["A_true_lambda_mean"].to_numpy(dtype=float)
    A_resid = dfc["A_true_relative_residual_mean"].to_numpy(dtype=float)
    B_lam = dfc["B_lambda_mean"].to_numpy(dtype=float)
    B_resid = dfc["B_relative_residual_mean"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    axes[0].plot(x, A_lam, marker="o")
    axes[0].axhline(0.0, linestyle="--", alpha=0.7)
    axes[0].set_xlabel("drift fraction")
    axes[0].set_ylabel("lambda of T_hat M_true")
    axes[0].set_title("CLWL induced gain")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, A_resid, marker="o")
    axes[1].set_xlabel("drift fraction")
    axes[1].set_ylabel("relative residual")
    axes[1].set_title("T_hat M_true residual")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, B_lam, marker="o")
    axes[2].axhline(0.0, linestyle="--", alpha=0.7)
    axes[2].set_xlabel("drift fraction")
    axes[2].set_ylabel("lambda of pinv(M_hat)M_true")
    axes[2].set_title("Forward induced clean-map gain")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(x, B_resid, marker="o")
    axes[3].set_xlabel("drift fraction")
    axes[3].set_ylabel("relative residual")
    axes[3].set_title("Forward clean-map residual")
    axes[3].grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7) Main
# ============================================================

if __name__ == "__main__":
    out_dir = "artifacts_settingA_nullspace_Mhat_robustness"
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

    raw_df, summary_df = run_settingA_nullspace_experiment(
        drift_grid=[0.0, 0.10, 0.20, 0.40, 0.60, 0.80, 0.95],
        seeds=[0, 1, 2, 3, 4],
        n=6000,
        input_dim=8,
        cfg=cfg,
        out_dir=out_dir,
    )

    print("\n=== Raw results head ===")
    print(raw_df.head())
    print("\n=== Summary results ===")
    print(summary_df)

    plot_results(summary_df, out_path=str(Path(out_dir) / "settingA_nullspace_results.png"))
    plot_diagnostics(summary_df, out_path=str(Path(out_dir) / "settingA_nullspace_diagnostics.png"))

    print("\nSaved outputs to:")
    print(Path(out_dir).resolve())
