from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from clwl_experiments_module4_weak_label_dataset import WeakLabelDataset
from clwl_experiments_module6_metrics import evaluate_scores_on_dataset


Array = np.ndarray


@dataclass
class CLCLTrainConfig:
    model_type: Literal["linear", "mlp"] = "linear"
    variant: Literal["or", "or_w"] = "or_w"
    hidden_dim: int = 128
    batch_size: int = 128
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    seed: int = 0
    log_every: int = 1
    weight_power: float = 1.0
    min_weight: float = 1e-3
    weight_eps: float = 1e-6
    detach_weight: bool = True


@dataclass
class CLCLEpochLog:
    epoch: int
    train_loss: float
    val_loss: Optional[float]
    train_clean_accuracy: float
    train_max_preservation_rate: float
    train_pairwise_order_rate: float
    val_clean_accuracy: Optional[float]
    val_max_preservation_rate: Optional[float]
    val_pairwise_order_rate: Optional[float]


@dataclass
class CLCLTrainResult:
    model: nn.Module
    config: CLCLTrainConfig
    logs: list[CLCLEpochLog]
    final_train_metrics: dict[str, Any]
    final_val_metrics: Optional[dict[str, Any]]


class CLCLTrainingError(ValueError):
    pass


class TorchComplementaryDataset(Dataset):
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



def _validate_complementary_dataset(dataset: WeakLabelDataset, *, atol: float = 1e-8) -> tuple[int, int]:
    X = np.asarray(dataset.X, dtype=np.float64)
    eta = np.asarray(dataset.eta, dtype=np.float64)
    y = np.asarray(dataset.y, dtype=np.intp)
    z = np.asarray(dataset.z, dtype=np.intp)
    M = np.asarray(dataset.M, dtype=np.float64)

    if X.ndim != 2:
        raise CLCLTrainingError(f"dataset.X must be 2D, got shape {X.shape}.")
    if eta.ndim != 2:
        raise CLCLTrainingError(f"dataset.eta must be 2D, got shape {eta.shape}.")
    if y.ndim != 1:
        raise CLCLTrainingError(f"dataset.y must be 1D, got shape {y.shape}.")
    if z.ndim != 1:
        raise CLCLTrainingError(f"dataset.z must be 1D, got shape {z.shape}.")
    if M.ndim != 2:
        raise CLCLTrainingError(f"dataset.M must be 2D, got shape {M.shape}.")

    n, input_dim = X.shape
    n_eta, c = eta.shape
    d, c_M = M.shape
    if n_eta != n or y.shape[0] != n or z.shape[0] != n:
        raise CLCLTrainingError(
            f"Dataset arrays are inconsistent: X={X.shape}, y={y.shape}, eta={eta.shape}, z={z.shape}."
        )
    if c_M != c:
        raise CLCLTrainingError(f"dataset.M shape {M.shape} is incompatible with eta shape {eta.shape}.")
    if d != c:
        raise CLCLTrainingError(
            f"This module expects single complementary labels with d=c, got M.shape={M.shape}."
        )
    if np.min(z) < 0 or np.max(z) >= d:
        raise CLCLTrainingError(f"dataset.z entries must lie in [0, {d - 1}].")

    # Stronger protocol check for uniform complementary labels:
    # diagonal should be zero, off-diagonals should be 1/(c-1).
    diag = np.diag(M)
    if not np.allclose(diag, np.zeros_like(diag), atol=atol, rtol=0.0):
        raise CLCLTrainingError(
            "CLCL module expects complementary labels with zero diagonal in M. "
            f"Found diagonal range [{diag.min():.6f}, {diag.max():.6f}]."
        )

    if c > 1:
        off_diag_target = 1.0 / (c - 1)
        off_diag_mask = ~np.eye(c, dtype=bool)
        if not np.allclose(M[off_diag_mask], off_diag_target, atol=atol, rtol=0.0):
            raise CLCLTrainingError(
                "CLCL module currently assumes the uniform complementary transition model. "
                f"Expected off-diagonal entries {off_diag_target:.6f}."
            )

    return input_dim, c



def build_clcl_model(input_dim: int, num_classes: int, config: CLCLTrainConfig) -> nn.Module:
    if config.model_type == "linear":
        return LinearScoreModel(input_dim=input_dim, num_classes=num_classes)
    if config.model_type == "mlp":
        if config.hidden_dim <= 0:
            raise CLCLTrainingError(f"hidden_dim must be positive, got {config.hidden_dim}.")
        return MLPScoreModel(input_dim=input_dim, hidden_dim=config.hidden_dim, num_classes=num_classes)
    raise CLCLTrainingError(f"Unsupported model_type={config.model_type}.")



def _validate_scores_and_labels(scores: torch.Tensor, z: torch.Tensor) -> tuple[int, int]:
    if scores.ndim != 2:
        raise CLCLTrainingError(f"scores must be 2D, got shape {tuple(scores.shape)}.")
    if z.ndim != 1:
        raise CLCLTrainingError(f"z must be 1D, got shape {tuple(z.shape)}.")
    n, c = scores.shape
    if z.shape[0] != n:
        raise CLCLTrainingError(f"z must have length {n}, got {z.shape[0]}.")
    if torch.min(z).item() < 0 or torch.max(z).item() >= c:
        raise CLCLTrainingError(f"z entries must lie in [0, {c - 1}].")
    return n, c



def complementary_or_loss(scores: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    _validate_scores_and_labels(scores, z)
    # Liu-style OR with cross-entropy base loss: ell(-g(x), z_bar).
    return F.cross_entropy(-scores, z, reduction="mean")



def complementary_weight(
    scores: torch.Tensor,
    z: torch.Tensor,
    *,
    power: float = 1.0,
    min_weight: float = 1e-3,
    eps: float = 1e-6,
    detach_weight: bool = True,
) -> torch.Tensor:
    if power <= 0:
        raise CLCLTrainingError(f"power must be positive, got {power}.")
    if min_weight <= 0:
        raise CLCLTrainingError(f"min_weight must be positive, got {min_weight}.")
    if eps <= 0:
        raise CLCLTrainingError(f"eps must be positive, got {eps}.")

    n, c = _validate_scores_and_labels(scores, z)

    probs_pos = torch.softmax(scores, dim=1)
    probs_neg = torch.softmax(-scores, dim=1)

    # Paper-inspired vector construction:
    # u = 1 / s(-g), then weight uses s(u + 1) * s(g) at the complementary class.
    u = 1.0 / torch.clamp(probs_neg, min=eps)
    probs_u = torch.softmax(u + 1.0, dim=1)

    idx = torch.arange(n, device=scores.device)
    gathered_pos = probs_pos[idx, z]
    gathered_u = probs_u[idx, z]

    weights = gathered_pos * gathered_u
    if power != 1.0:
        weights = weights.pow(power)
    weights = torch.clamp(weights + eps, min=min_weight)

    if detach_weight:
        weights = weights.detach()
    return weights



def clcl_torch_loss(scores: torch.Tensor, z: torch.Tensor, config: CLCLTrainConfig) -> torch.Tensor:
    _validate_scores_and_labels(scores, z)

    base_per_sample = F.cross_entropy(-scores, z, reduction="none")
    if config.variant == "or":
        return base_per_sample.mean()
    if config.variant == "or_w":
        weights = complementary_weight(
            scores,
            z,
            power=config.weight_power,
            min_weight=config.min_weight,
            eps=config.weight_eps,
            detach_weight=config.detach_weight,
        )
        return (weights * base_per_sample).mean()
    raise CLCLTrainingError(f"Unsupported variant={config.variant}.")



def make_dataloader(dataset: WeakLabelDataset, batch_size: int, *, shuffle: bool) -> DataLoader:
    if batch_size <= 0:
        raise CLCLTrainingError(f"batch_size must be positive, got {batch_size}.")
    torch_dataset = TorchComplementaryDataset(dataset)
    return DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle)



def scores_from_model(
    model: nn.Module,
    dataset: WeakLabelDataset,
    *,
    batch_size: int = 512,
    device: str = "cpu",
) -> Array:
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    outputs: list[Array] = []
    device_obj = torch.device(device)

    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device_obj)
            scores = model(x_batch).detach().cpu().numpy().astype(np.float64)
            outputs.append(scores)

    if not outputs:
        raise CLCLTrainingError("scores_from_model received an empty dataset.")
    return np.concatenate(outputs, axis=0)



def empirical_risk_from_model(
    model: nn.Module,
    dataset: WeakLabelDataset,
    config: CLCLTrainConfig,
    *,
    batch_size: int = 512,
    device: str = "cpu",
) -> float:
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    total_count = 0
    device_obj = torch.device(device)

    with torch.no_grad():
        for x_batch, z_batch in loader:
            x_batch = x_batch.to(device_obj)
            z_batch = z_batch.to(device_obj)
            loss = clcl_torch_loss(model(x_batch), z_batch, config)
            batch_n = int(x_batch.shape[0])
            total_loss += float(loss.item()) * batch_n
            total_count += batch_n

    if total_count == 0:
        raise CLCLTrainingError("empirical_risk_from_model received an empty dataset.")
    return total_loss / total_count



def evaluate_model_on_dataset(
    model: nn.Module,
    dataset: WeakLabelDataset,
    config: CLCLTrainConfig,
    *,
    batch_size: int = 512,
    device: str = "cpu",
) -> dict[str, Any]:
    scores = scores_from_model(model, dataset, batch_size=batch_size, device=device)
    metrics = evaluate_scores_on_dataset(scores, dataset)
    empirical_risk = empirical_risk_from_model(model, dataset, config, batch_size=batch_size, device=device)
    return {
        "clean_accuracy": metrics.clean_accuracy,
        "max_preservation_rate": metrics.max_preservation_rate,
        "pairwise_order_rate": metrics.pairwise_order_rate,
        "pairwise_total": metrics.pairwise_total,
        "pairwise_correct": metrics.pairwise_correct,
        "mean_margin_on_ordered_pairs": metrics.mean_margin_on_ordered_pairs,
        "empirical_risk": empirical_risk,
    }



def train_clcl_model(
    train_dataset: WeakLabelDataset,
    *,
    val_dataset: Optional[WeakLabelDataset] = None,
    config: Optional[CLCLTrainConfig] = None,
) -> CLCLTrainResult:
    if config is None:
        config = CLCLTrainConfig()

    input_dim, num_classes = _validate_complementary_dataset(train_dataset)
    if val_dataset is not None:
        _validate_complementary_dataset(val_dataset)

    set_torch_seed(config.seed)
    device_obj = torch.device(config.device)
    model = build_clcl_model(input_dim=input_dim, num_classes=num_classes, config=config).to(device_obj)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = make_dataloader(train_dataset, batch_size=config.batch_size, shuffle=True)

    logs: list[CLCLEpochLog] = []

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        running_loss = 0.0
        total_count = 0

        for x_batch, z_batch in train_loader:
            x_batch = x_batch.to(device_obj)
            z_batch = z_batch.to(device_obj)

            optimizer.zero_grad()
            scores = model(x_batch)
            loss = clcl_torch_loss(scores, z_batch, config)
            loss.backward()
            optimizer.step()

            batch_n = int(x_batch.shape[0])
            running_loss += float(loss.item()) * batch_n
            total_count += batch_n

        if total_count == 0:
            raise CLCLTrainingError("Training loader is empty.")
        train_loss = running_loss / total_count

        if epoch % config.log_every == 0 or epoch == config.num_epochs:
            train_metrics = evaluate_model_on_dataset(
                model,
                train_dataset,
                config,
                batch_size=max(config.batch_size, 512),
                device=config.device,
            )
            val_metrics = None
            if val_dataset is not None:
                val_metrics = evaluate_model_on_dataset(
                    model,
                    val_dataset,
                    config,
                    batch_size=max(config.batch_size, 512),
                    device=config.device,
                )

            logs.append(
                CLCLEpochLog(
                    epoch=epoch,
                    train_loss=float(train_loss),
                    val_loss=None if val_metrics is None else float(val_metrics["empirical_risk"]),
                    train_clean_accuracy=float(train_metrics["clean_accuracy"]),
                    train_max_preservation_rate=float(train_metrics["max_preservation_rate"]),
                    train_pairwise_order_rate=float(train_metrics["pairwise_order_rate"]),
                    val_clean_accuracy=None if val_metrics is None else float(val_metrics["clean_accuracy"]),
                    val_max_preservation_rate=None if val_metrics is None else float(val_metrics["max_preservation_rate"]),
                    val_pairwise_order_rate=None if val_metrics is None else float(val_metrics["pairwise_order_rate"]),
                )
            )

    final_train_metrics = evaluate_model_on_dataset(
        model,
        train_dataset,
        config,
        batch_size=max(config.batch_size, 512),
        device=config.device,
    )
    final_val_metrics = None
    if val_dataset is not None:
        final_val_metrics = evaluate_model_on_dataset(
            model,
            val_dataset,
            config,
            batch_size=max(config.batch_size, 512),
            device=config.device,
        )

    return CLCLTrainResult(
        model=model,
        config=config,
        logs=logs,
        final_train_metrics=final_train_metrics,
        final_val_metrics=final_val_metrics,
    )


if __name__ == "__main__":
    from clwl_experiments_module2_weak_label_generators import make_uniform_complementary_family
    from clwl_experiments_module3_synthetic_clean_data import generate_linear_softmax_dataset, train_val_test_split
    from clwl_experiments_module4_weak_label_dataset import build_weak_label_splits

    ds = generate_linear_softmax_dataset(
        n=1000,
        input_dim=8,
        num_classes=4,
        feature_seed=0,
        teacher_seed=1,
        label_seed=2,
    )
    splits = train_val_test_split(ds, train_frac=0.6, val_frac=0.2, seed=42)
    family = make_uniform_complementary_family(c=4)
    weak_splits = build_weak_label_splits(splits, family, seed=10)

    for variant in ["or", "or_w"]:
        config = CLCLTrainConfig(
            model_type="linear",
            variant=variant,
            batch_size=128,
            num_epochs=10,
            learning_rate=1e-2,
            device="cpu",
            seed=0,
            log_every=1,
            weight_power=1.0,
            min_weight=1e-3,
        )
        result = train_clcl_model(
            train_dataset=weak_splits["train"],
            val_dataset=weak_splits["val"],
            config=config,
        )
        print(f"=== Variant: {variant} ===")
        print("Final train metrics:", result.final_train_metrics)
        print("Final val metrics:", result.final_val_metrics)
        print("Last epoch log:", asdict(result.logs[-1]))