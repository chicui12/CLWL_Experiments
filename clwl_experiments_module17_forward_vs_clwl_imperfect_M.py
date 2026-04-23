from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

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
    evaluate_model_on_dataset as eval_clwl_model,
)


Array = np.ndarray


# ============================================================
# 1) Noise-matrix utilities
# ============================================================

def _as_float_2d(name: str, x: Array) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _as_index_1d(name: str, x: Array) -> Array:
    arr = np.asarray(x, dtype=np.intp)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}.")
    return arr


def validate_noisy_transition_matrix(M: Array, *, atol: float = 1e-8) -> tuple[int, int]:
    M = _as_float_2d("M", M)
    d, c = M.shape
    if d != c:
        raise ValueError(f"For noisy-label forward experiments we require square M, got {M.shape}.")
    if np.min(M) < -atol:
        raise ValueError(f"M has negative entries: min={np.min(M):.3e}.")
    col_sums = M.sum(axis=0)
    if not np.allclose(col_sums, np.ones(c), atol=atol, rtol=0.0):
        raise ValueError(f"M must be column-stochastic. Column sums are {col_sums}.")
    return d, c


def matrix_rank(M: Array, *, tol: Optional[float] = None) -> int:
    return int(np.linalg.matrix_rank(np.asarray(M, dtype=np.float64), tol=tol))


def project_to_column_stochastic_nonnegative(M: Array, *, eps: float = 1e-8) -> Array:
    M = np.asarray(M, dtype=np.float64)
    M = np.clip(M, eps, None)
    M /= M.sum(axis=0, keepdims=True)
    return M


def make_symmetric_noisy_M(c: int, noise_rate: float) -> Array:
    if c < 2:
        raise ValueError(f"Need c >= 2, got {c}.")
    if not (0.0 <= noise_rate < 1.0):
        raise ValueError(f"noise_rate must be in [0,1), got {noise_rate}.")
    M = (1.0 - noise_rate) * np.eye(c, dtype=np.float64)
    if c > 1:
        M += (noise_rate / (c - 1)) * (np.ones((c, c), dtype=np.float64) - np.eye(c, dtype=np.float64))
    validate_noisy_transition_matrix(M)
    return M


def make_confusion_map_noisy_M(
    c: int,
    confusion_map: dict[int, int],
    *,
    total_noise: float,
    special_noise: float,
) -> Array:
    """
    Build a square noisy-label transition matrix M[z, y] = P(z | y).

    For classes y in confusion_map:
      - stay on y with prob 1 - total_noise
      - flip to confusion_map[y] with prob special_noise
      - spread the remaining noise uniformly over other wrong labels

    For classes y not in confusion_map:
      - use symmetric noise with total_noise
    """
    if c < 2:
        raise ValueError(f"Need c >= 2, got {c}.")
    if not (0.0 <= total_noise < 1.0):
        raise ValueError(f"total_noise must be in [0,1), got {total_noise}.")
    if not (0.0 <= special_noise <= total_noise):
        raise ValueError(
            f"special_noise must be in [0,total_noise], got special_noise={special_noise}, total_noise={total_noise}."
        )

    M = np.zeros((c, c), dtype=np.float64)

    for y in range(c):
        M[y, y] = 1.0 - total_noise

        if y in confusion_map:
            z_star = int(confusion_map[y])
            if z_star == y:
                raise ValueError(f"confusion_map[{y}] points to itself.")
            if z_star < 0 or z_star >= c:
                raise ValueError(f"confusion_map[{y}]={z_star} out of range.")

            M[z_star, y] = special_noise
            remaining = total_noise - special_noise
            other_wrong = [z for z in range(c) if z != y and z != z_star]
            if other_wrong:
                share = remaining / len(other_wrong)
                for z in other_wrong:
                    M[z, y] = share
        else:
            wrong = [z for z in range(c) if z != y]
            share = total_noise / len(wrong)
            for z in wrong:
                M[z, y] = share

    M = project_to_column_stochastic_nonnegative(M)
    validate_noisy_transition_matrix(M)
    return M


def make_wrong_confusion_map(c: int, true_confusion_map: dict[int, int], *, shift: int = 2) -> dict[int, int]:
    """
    Create a deliberately wrong confusion map by shifting target classes.
    Example: if true map says 0->1, wrong map may say 0->3.
    """
    wrong_map: dict[int, int] = {}
    for y, z in true_confusion_map.items():
        z_wrong = (z + shift) % c
        if z_wrong == y:
            z_wrong = (z_wrong + 1) % c
        wrong_map[int(y)] = int(z_wrong)
    return wrong_map


def make_structured_misspecified_M(
    M_true: Array,
    M_wrong: Array,
    mismatch_strength: float,
) -> Array:
    """
    Structured mismatch:
        M_est = (1-gamma) M_true + gamma M_wrong
    where M_wrong has the wrong confusion directions.
    """
    M_true = np.asarray(M_true, dtype=np.float64)
    M_wrong = np.asarray(M_wrong, dtype=np.float64)
    validate_noisy_transition_matrix(M_true)
    validate_noisy_transition_matrix(M_wrong)

    if M_true.shape != M_wrong.shape:
        raise ValueError(f"M_true and M_wrong must have same shape, got {M_true.shape} vs {M_wrong.shape}.")
    if not (0.0 <= mismatch_strength <= 1.0):
        raise ValueError(f"mismatch_strength must be in [0,1], got {mismatch_strength}.")

    M_est = (1.0 - mismatch_strength) * M_true + mismatch_strength * M_wrong
    M_est = project_to_column_stochastic_nonnegative(M_est)
    validate_noisy_transition_matrix(M_est)

    c = M_est.shape[1]
    if matrix_rank(M_est) != c:
        M_est = M_est + 1e-4 * np.eye(c, dtype=np.float64)
        M_est = project_to_column_stochastic_nonnegative(M_est)
        validate_noisy_transition_matrix(M_est)

    if matrix_rank(M_est) != c:
        raise ValueError(
            "M_est is not full rank, so generic CLWL T-construction is not applicable. "
            f"rank(M_est)={matrix_rank(M_est)}, c={c}."
        )
    return M_est


# ============================================================
# 2) Build noisy-label weak datasets from M_true
# ============================================================

def sample_noisy_labels_from_M(y: Array, M: Array, *, seed: int = 0) -> Array:
    y = _as_index_1d("y", y)
    M = _as_float_2d("M", M)
    d, c = validate_noisy_transition_matrix(M)
    if np.min(y) < 0 or np.max(y) >= c:
        raise ValueError(f"y must lie in [0, {c - 1}].")
    rng = np.random.default_rng(seed)
    z = np.empty_like(y)
    for i, yi in enumerate(y):
        z[i] = int(rng.choice(d, p=M[:, yi]))
    return z


def build_noisy_weak_label_dataset_from_true_M(
    clean_dataset: Any,
    M_true: Array,
    *,
    seed: int = 0,
    family_name: str = "noisy_label_true_M",
) -> WeakLabelDataset:
    X = np.asarray(clean_dataset.X, dtype=np.float64).copy()
    y = np.asarray(clean_dataset.y, dtype=np.intp).copy()
    eta = np.asarray(clean_dataset.eta, dtype=np.float64).copy()

    if hasattr(clean_dataset, "logits"):
        logits = np.asarray(clean_dataset.logits, dtype=np.float64).copy()
    else:
        logits = np.log(np.clip(eta, 1e-12, 1.0))

    M_true = np.asarray(M_true, dtype=np.float64).copy()
    _, c = validate_noisy_transition_matrix(M_true)
    if eta.shape[1] != c:
        raise ValueError(f"eta has {eta.shape[1]} classes but M_true has size {c}.")

    z = sample_noisy_labels_from_M(y, M_true, seed=seed)

    metadata = {
        "family_name": family_name,
        "seed": seed,
        "num_samples": int(X.shape[0]),
        "num_classes": int(c),
        "num_weak_labels": int(c),
        "eta_source": getattr(clean_dataset, "metadata", {}).get("eta_source", "oracle"),
        "oracle_metrics": bool(getattr(clean_dataset, "metadata", {}).get("oracle_metrics", True)),
        "dataset_name": getattr(clean_dataset, "metadata", {}).get("dataset_name", "unknown"),
        "input_kind": getattr(clean_dataset, "metadata", {}).get("input_kind", "vector"),
        "dataset_metadata": dict(getattr(clean_dataset, "metadata", {})),
        "family_metadata": {
            "setting": "noisy_label",
            "M_true": M_true.tolist(),
        },
    }

    return WeakLabelDataset(
        X=X,
        y=y,
        eta=eta,
        logits=logits,
        z=z,
        M=M_true,
        family_name=family_name,
        weak_label_matrix=None,
        weak_label_vectors=None,
        weak_label_names=[f"noisy_{i}" for i in range(c)],
        metadata=metadata,
    )


def build_noisy_weak_label_splits_from_true_M(
    clean_splits: dict[str, Any],
    M_true: Array,
    *,
    seed: int = 0,
    seed_stride: int = 1000,
    family_name: str = "noisy_label_true_M",
) -> dict[str, WeakLabelDataset]:
    out: dict[str, WeakLabelDataset] = {}
    for i, split_name in enumerate(sorted(clean_splits.keys())):
        out[split_name] = build_noisy_weak_label_dataset_from_true_M(
            clean_splits[split_name],
            M_true,
            seed=seed + i * seed_stride,
            family_name=family_name,
        )
    return out


# ============================================================
# 3) Forward-correction trainer
# ============================================================

@dataclass
class ForwardEpochLog:
    epoch: int
    train_loss: float
    val_loss: Optional[float]
    train_clean_accuracy: float
    train_max_preservation_rate: float
    train_pairwise_order_rate: float
    val_clean_accuracy: Optional[float]
    val_max_preservation_rate: Optional[float]
    val_pairwise_order_rate: Optional[float]
    current_learning_rate: float


@dataclass
class ForwardTrainResult:
    model: nn.Module
    config: CLWLTrainConfig
    logs: list[ForwardEpochLog]
    final_train_metrics: dict[str, Any]
    final_val_metrics: Optional[dict[str, Any]]
    best_epoch: int
    best_monitor_value: float


class ForwardTrainingError(ValueError):
    pass


class TorchNoisyLabelDataset(Dataset):
    def __init__(self, dataset: WeakLabelDataset) -> None:
        self.X = torch.tensor(np.asarray(dataset.X, dtype=np.float32), dtype=torch.float32)
        self.z = torch.tensor(np.asarray(dataset.z, dtype=np.int64), dtype=torch.long)

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


def build_score_model(
    input_dim: int,
    num_classes: int,
    config: CLWLTrainConfig,
) -> nn.Module:
    if config.model_type == "linear":
        return LinearScoreModel(input_dim=input_dim, num_classes=num_classes)
    if config.model_type == "mlp":
        return MLPScoreModel(input_dim=input_dim, hidden_dim=config.hidden_dim, num_classes=num_classes)
    raise ForwardTrainingError(f"Unsupported model_type={config.model_type}.")


def make_dataloader(dataset: WeakLabelDataset, batch_size: int, *, shuffle: bool) -> DataLoader:
    torch_dataset = TorchNoisyLabelDataset(dataset)
    return DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle)


def validate_noisy_dataset(dataset: WeakLabelDataset) -> tuple[int, int]:
    X = np.asarray(dataset.X, dtype=np.float64)
    y = np.asarray(dataset.y, dtype=np.intp)
    eta = np.asarray(dataset.eta, dtype=np.float64)
    z = np.asarray(dataset.z, dtype=np.intp)
    M = np.asarray(dataset.M, dtype=np.float64)

    if X.ndim != 2:
        raise ForwardTrainingError(f"dataset.X must be 2D, got shape {X.shape}.")
    if y.ndim != 1:
        raise ForwardTrainingError(f"dataset.y must be 1D, got shape {y.shape}.")
    if eta.ndim != 2:
        raise ForwardTrainingError(f"dataset.eta must be 2D, got shape {eta.shape}.")
    if z.ndim != 1:
        raise ForwardTrainingError(f"dataset.z must be 1D, got shape {z.shape}.")

    n, input_dim = X.shape
    if y.shape[0] != n or eta.shape[0] != n or z.shape[0] != n:
        raise ForwardTrainingError("X, y, eta, z must agree on sample size.")

    _, c = validate_noisy_transition_matrix(M)
    if eta.shape[1] != c:
        raise ForwardTrainingError(f"eta has {eta.shape[1]} classes but M has size {c}.")
    if np.min(z) < 0 or np.max(z) >= c:
        raise ForwardTrainingError(f"z must lie in [0, {c - 1}].")

    return input_dim, c


def forward_corrected_loss(
    logits: torch.Tensor,
    z: torch.Tensor,
    M_est_torch: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Repo convention: M[z, y] = P(z | y).
    For clean posterior probs_clean[n, y], the noisy posterior is:
        probs_noisy[n, z] = sum_y probs_clean[n, y] * M[z, y]
    hence probs_noisy = probs_clean @ M_est.T
    """
    probs_clean = torch.softmax(logits, dim=1)
    probs_noisy = probs_clean @ M_est_torch.transpose(0, 1)
    probs_noisy = torch.clamp(probs_noisy, min=eps)
    probs_noisy = probs_noisy / probs_noisy.sum(dim=1, keepdim=True)
    return F.nll_loss(torch.log(probs_noisy), z)


def scores_from_model(model: nn.Module, dataset: WeakLabelDataset, *, batch_size: int = 512, device: str = "cpu") -> Array:
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    outputs: list[Array] = []
    device_obj = torch.device(device)
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device_obj)
            outputs.append(model(x_batch).detach().cpu().numpy().astype(np.float64))
    if not outputs:
        raise ForwardTrainingError("Empty dataset.")
    return np.concatenate(outputs, axis=0)


def empirical_forward_risk_from_model(
    model: nn.Module,
    dataset: WeakLabelDataset,
    M_est: Array,
    *,
    batch_size: int = 512,
    device: str = "cpu",
) -> float:
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False)
    device_obj = torch.device(device)
    model.eval()
    M_est_torch = torch.tensor(np.asarray(M_est, dtype=np.float32), dtype=torch.float32, device=device_obj)

    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x_batch, z_batch in loader:
            x_batch = x_batch.to(device_obj)
            z_batch = z_batch.to(device_obj)
            loss = forward_corrected_loss(model(x_batch), z_batch, M_est_torch)
            batch_n = int(x_batch.shape[0])
            total_loss += float(loss.item()) * batch_n
            total_count += batch_n

    if total_count == 0:
        raise ForwardTrainingError("Empty dataset.")
    return total_loss / total_count


def evaluate_forward_model_on_dataset(
    model: nn.Module,
    dataset: WeakLabelDataset,
    M_est: Array,
    *,
    batch_size: int = 512,
    device: str = "cpu",
) -> dict[str, Any]:
    scores = scores_from_model(model, dataset, batch_size=batch_size, device=device)
    metrics = evaluate_scores_on_dataset(scores, dataset)
    empirical_risk = empirical_forward_risk_from_model(model, dataset, M_est, batch_size=batch_size, device=device)
    return {
        "clean_accuracy": metrics.clean_accuracy,
        "max_preservation_rate": metrics.max_preservation_rate,
        "pairwise_order_rate": metrics.pairwise_order_rate,
        "pairwise_total": metrics.pairwise_total,
        "pairwise_correct": metrics.pairwise_correct,
        "mean_margin_on_ordered_pairs": metrics.mean_margin_on_ordered_pairs,
        "empirical_risk": empirical_risk,
    }


def _clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_forward_model(
    train_dataset: WeakLabelDataset,
    M_est: Array,
    *,
    val_dataset: Optional[WeakLabelDataset] = None,
    config: Optional[CLWLTrainConfig] = None,
) -> ForwardTrainResult:
    if config is None:
        config = CLWLTrainConfig()

    input_dim, num_classes = validate_noisy_dataset(train_dataset)
    validate_noisy_transition_matrix(M_est)
    if np.asarray(M_est).shape != (num_classes, num_classes):
        raise ForwardTrainingError(f"M_est must have shape {(num_classes, num_classes)}.")

    if val_dataset is not None:
        validate_noisy_dataset(val_dataset)

    set_torch_seed(config.seed)
    device_obj = torch.device(config.device)
    model = build_score_model(input_dim=input_dim, num_classes=num_classes, config=config).to(device_obj)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_learning_rate,
    )

    M_est_torch = torch.tensor(np.asarray(M_est, dtype=np.float32), dtype=torch.float32, device=device_obj)
    train_loader = make_dataloader(train_dataset, batch_size=config.batch_size, shuffle=True)

    logs: list[ForwardEpochLog] = []
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_monitor_value = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        running_loss = 0.0
        total_count = 0

        for x_batch, z_batch in train_loader:
            x_batch = x_batch.to(device_obj)
            z_batch = z_batch.to(device_obj)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = forward_corrected_loss(logits, z_batch, M_est_torch)
            loss.backward()

            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)

            optimizer.step()

            batch_n = int(x_batch.shape[0])
            running_loss += float(loss.item()) * batch_n
            total_count += batch_n

        train_loss = running_loss / max(total_count, 1)

        if epoch % config.log_every == 0 or epoch == config.num_epochs:
            train_metrics = evaluate_forward_model_on_dataset(
                model, train_dataset, M_est,
                batch_size=max(config.batch_size, 512),
                device=config.device,
            )
            val_metrics = None
            if val_dataset is not None:
                val_metrics = evaluate_forward_model_on_dataset(
                    model, val_dataset, M_est,
                    batch_size=max(config.batch_size, 512),
                    device=config.device,
                )

            monitor_value = train_loss if val_metrics is None else float(val_metrics["empirical_risk"])
            scheduler.step(monitor_value)
            current_lr = float(optimizer.param_groups[0]["lr"])

            logs.append(
                ForwardEpochLog(
                    epoch=epoch,
                    train_loss=float(train_loss),
                    val_loss=None if val_metrics is None else float(val_metrics["empirical_risk"]),
                    train_clean_accuracy=float(train_metrics["clean_accuracy"]),
                    train_max_preservation_rate=float(train_metrics["max_preservation_rate"]),
                    train_pairwise_order_rate=float(train_metrics["pairwise_order_rate"]),
                    val_clean_accuracy=None if val_metrics is None else float(val_metrics["clean_accuracy"]),
                    val_max_preservation_rate=None if val_metrics is None else float(val_metrics["max_preservation_rate"]),
                    val_pairwise_order_rate=None if val_metrics is None else float(val_metrics["pairwise_order_rate"]),
                    current_learning_rate=current_lr,
                )
            )

            if monitor_value < best_monitor_value - config.early_stop_min_delta:
                best_monitor_value = float(monitor_value)
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = _clone_state_dict_to_cpu(model)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train_metrics = evaluate_forward_model_on_dataset(
        model, train_dataset, M_est,
        batch_size=max(config.batch_size, 512),
        device=config.device,
    )
    final_val_metrics = None
    if val_dataset is not None:
        final_val_metrics = evaluate_forward_model_on_dataset(
            model, val_dataset, M_est,
            batch_size=max(config.batch_size, 512),
            device=config.device,
        )

    return ForwardTrainResult(
        model=model,
        config=config,
        logs=logs,
        final_train_metrics=final_train_metrics,
        final_val_metrics=final_val_metrics,
        best_epoch=best_epoch,
        best_monitor_value=float(best_monitor_value),
    )


# ============================================================
# 4) Experiment runner: CLWL vs Forward under imperfect M
# ============================================================

def run_clwl_vs_forward_imperfect_M(
    clean_splits: dict[str, Any],
    M_true: Array,
    M_wrong: Array,
    mismatch_grid: list[float],
    *,
    seeds: list[int],
    clwl_config: CLWLTrainConfig,
    forward_config: CLWLTrainConfig,
    out_dir: str = "artifacts_forward_vs_clwl_imperfect_M",
    family_name: str = "noisy_label_true_M",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    clean_splits: dict with keys like train/val/test, each item must expose
                  X, y, eta, and preferably logits.
    M_true: true noisy-label transition matrix used to sample z.
    M_wrong: wrong transition matrix used to induce structured mismatch.
    mismatch_grid: list of mismatch strengths in [0,1].
    seeds: repeat seeds.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    M_true = np.asarray(M_true, dtype=np.float64)
    _, c = validate_noisy_transition_matrix(M_true)

    if matrix_rank(M_true) != c:
        raise ValueError(
            "M_true must be full rank for generic CLWL T-construction in this experiment. "
            f"rank(M_true)={matrix_rank(M_true)}, c={c}."
        )

    rows: list[dict[str, Any]] = []
    np.save(Path(out_dir) / "M_true.npy", M_true)
    np.save(Path(out_dir) / "M_wrong.npy", np.asarray(M_wrong, dtype=np.float64))

    for mismatch_strength in mismatch_grid:
        M_est = make_structured_misspecified_M(M_true, M_wrong, mismatch_strength)
        fro_error = float(np.linalg.norm(M_est - M_true, ord="fro"))
        np.save(Path(out_dir) / f"M_est_mismatch_{mismatch_strength:.3f}.npy", M_est)

        for seed in seeds:
            weak_splits = build_noisy_weak_label_splits_from_true_M(
                clean_splits,
                M_true,
                seed=seed,
                family_name=family_name,
            )

            construction = construct_clwl_T(M_est)
            T_est = construction.T
            clwl_result = train_clwl_model(
                train_dataset=weak_splits["train"],
                val_dataset=weak_splits.get("val"),
                T=T_est,
                config=clwl_config,
            )
            clwl_test = eval_clwl_model(
                clwl_result.model,
                weak_splits["test"],
                T_est,
                batch_size=max(clwl_config.batch_size, 512),
                device=clwl_config.device,
            )
            rows.append({
                "method": "CLWL",
                "seed": seed,
                "mismatch_strength": mismatch_strength,
                "fro_error": fro_error,
                "best_epoch": int(clwl_result.best_epoch),
                "best_monitor_value": float(clwl_result.best_monitor_value),
                "test_clean_accuracy": float(clwl_test["clean_accuracy"]),
                "test_max_preservation_rate": float(clwl_test["max_preservation_rate"]),
                "test_pairwise_order_rate": float(clwl_test["pairwise_order_rate"]),
                "test_empirical_risk": float(clwl_test["empirical_risk"]),
                "lambda_value": float(construction.lambda_value),
                "reconstruction_error": float(construction.reconstruction_error),
            })

            fwd_result = train_forward_model(
                train_dataset=weak_splits["train"],
                val_dataset=weak_splits.get("val"),
                M_est=M_est,
                config=forward_config,
            )
            fwd_test = evaluate_forward_model_on_dataset(
                fwd_result.model,
                weak_splits["test"],
                M_est,
                batch_size=max(forward_config.batch_size, 512),
                device=forward_config.device,
            )
            rows.append({
                "method": "Forward",
                "seed": seed,
                "mismatch_strength": mismatch_strength,
                "fro_error": fro_error,
                "best_epoch": int(fwd_result.best_epoch),
                "best_monitor_value": float(fwd_result.best_monitor_value),
                "test_clean_accuracy": float(fwd_test["clean_accuracy"]),
                "test_max_preservation_rate": float(fwd_test["max_preservation_rate"]),
                "test_pairwise_order_rate": float(fwd_test["pairwise_order_rate"]),
                "test_empirical_risk": float(fwd_test["empirical_risk"]),
                "lambda_value": np.nan,
                "reconstruction_error": np.nan,
            })

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(Path(out_dir) / "raw_results.csv", index=False)

    summary_df = (
        raw_df
        .groupby(["method", "mismatch_strength"], as_index=False)
        .agg({
            "fro_error": ["mean", "std"],
            "best_epoch": ["mean", "std"],
            "best_monitor_value": ["mean", "std"],
            "test_clean_accuracy": ["mean", "std"],
            "test_max_preservation_rate": ["mean", "std"],
            "test_pairwise_order_rate": ["mean", "std"],
            "test_empirical_risk": ["mean", "std"],
            "lambda_value": ["mean", "std"],
            "reconstruction_error": ["mean", "std"],
        })
    )
    summary_df.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in summary_df.columns
    ]
    summary_df.to_csv(Path(out_dir) / "summary_results.csv", index=False)

    return raw_df, summary_df


def plot_clwl_vs_forward_imperfect_M(
    summary_df: pd.DataFrame,
    *,
    out_path: str = "clwl_vs_forward_imperfect_M.png",
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    for method in sorted(summary_df["method"].unique()):
        dfm = summary_df[summary_df["method"] == method].sort_values("mismatch_strength")
        x = dfm["mismatch_strength"].to_numpy(dtype=float)
        y = dfm["test_clean_accuracy_mean"].to_numpy(dtype=float)
        y_std = dfm["test_clean_accuracy_std"].to_numpy(dtype=float)

        axes[0].plot(x, y, marker="o", label=method)
        axes[0].fill_between(x, y - y_std, y + y_std, alpha=0.2)

    axes[0].set_xlabel("mismatch strength")
    axes[0].set_ylabel("test clean accuracy")
    axes[0].set_title("CLWL vs Forward under structured imperfect M")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for method in sorted(summary_df["method"].unique()):
        dfm = summary_df[summary_df["method"] == method].sort_values("mismatch_strength").copy()
        x = dfm["mismatch_strength"].to_numpy(dtype=float)
        acc0 = float(dfm.loc[dfm["mismatch_strength"] == 0.0, "test_clean_accuracy_mean"].iloc[0])
        y = dfm["test_clean_accuracy_mean"].to_numpy(dtype=float) - acc0
        y_std = dfm["test_clean_accuracy_std"].to_numpy(dtype=float)

        axes[1].plot(x, y, marker="o", label=method)
        axes[1].fill_between(x, y - y_std, y + y_std, alpha=0.2)

    axes[1].axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].set_xlabel("mismatch strength")
    axes[1].set_ylabel("accuracy drop from mismatch=0")
    axes[1].set_title("Robustness to transition misspecification")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_gap_to_forward(summary_df: pd.DataFrame, *, out_path: str = "gap_to_forward.png") -> None:
    import matplotlib.pyplot as plt

    dfc = summary_df[summary_df["method"] == "CLWL"].sort_values("mismatch_strength")
    dff = summary_df[summary_df["method"] == "Forward"].sort_values("mismatch_strength")

    x = dfc["mismatch_strength"].to_numpy(dtype=float)
    gap = (
        dfc["test_clean_accuracy_mean"].to_numpy(dtype=float)
        - dff["test_clean_accuracy_mean"].to_numpy(dtype=float)
    )

    plt.figure(figsize=(6, 4))
    plt.plot(x, gap, marker="o")
    plt.axhline(0.0, linestyle="--", alpha=0.7)
    plt.xlabel("mismatch strength")
    plt.ylabel("CLWL acc - Forward acc")
    plt.title("Relative gap to Forward")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# 5) Example main
# ============================================================

if __name__ == "__main__":
    from clwl_experiments_module3_synthetic_clean_data import (
        generate_mlp_softmax_dataset,
        train_val_test_split,
    )

    clean_ds = generate_mlp_softmax_dataset(
        n=4000,
        input_dim=8,
        num_classes=4,
        hidden_dim=32,
        feature_seed=0,
        teacher_seed=1,
        label_seed=2,
    )
    clean_splits = train_val_test_split(clean_ds, train_frac=0.6, val_frac=0.2, seed=42)

    true_confusion_map = {
        0: 1,
        1: 0,
        2: 3,
        3: 2,
    }
    wrong_confusion_map = {
        0: 2,
        1: 3,
        2: 0,
        3: 1,
    }

    M_true = make_confusion_map_noisy_M(
        c=4,
        confusion_map=true_confusion_map,
        total_noise=0.50,
        special_noise=0.42,
    )
    M_wrong = make_confusion_map_noisy_M(
        c=4,
        confusion_map=wrong_confusion_map,
        total_noise=0.50,
        special_noise=0.42,
    )

    print("=== M_true ===")
    print(M_true)
    print("\n=== M_wrong ===")
    print(M_wrong)

    for gamma in [0.0, 0.2, 0.5, 0.8, 1.0]:
        M_est = make_structured_misspecified_M(M_true, M_wrong, gamma)
        print(f"\n=== gamma={gamma:.2f} ===")
        print(M_est)
        print("fro_error =", np.linalg.norm(M_est - M_true, ord="fro"))

    common_cfg = CLWLTrainConfig(
        model_type="mlp",
        hidden_dim=256,
        batch_size=256,
        num_epochs=80,
        learning_rate=3e-4,
        weight_decay=1e-4,
        device="cpu",  # change to "cuda" if available
        seed=0,
        log_every=1,
        gradient_clip_norm=5.0,
        scheduler_factor=0.5,
        scheduler_patience=5,
        min_learning_rate=1e-5,
        early_stop_patience=12,
        early_stop_min_delta=1e-8,
    )

    raw_df, summary_df = run_clwl_vs_forward_imperfect_M(
        clean_splits=clean_splits,
        M_true=M_true,
        M_wrong=M_wrong,
        mismatch_grid=[0.0, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00],
        seeds=[0, 1, 2, 3, 4],
        clwl_config=common_cfg,
        forward_config=common_cfg,
        out_dir="artifacts_forward_vs_clwl_structured_imperfect_M",
        family_name="confusion_aware_noisy_true_M",
    )

    print("\n=== Raw results ===")
    print(raw_df.head())

    print("\n=== Summary results ===")
    print(summary_df)

    plot_clwl_vs_forward_imperfect_M(
        summary_df,
        out_path="artifacts_forward_vs_clwl_structured_imperfect_M/clwl_vs_forward_structured_imperfect_M.png",
    )
    plot_gap_to_forward(
        summary_df,
        out_path="artifacts_forward_vs_clwl_structured_imperfect_M/gap_to_forward.png",
    )
