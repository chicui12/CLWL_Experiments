from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from clwl_experiments_module4_weak_label_dataset import WeakLabelDataset
from clwl_experiments_module6_metrics import evaluate_scores_on_dataset


Array = np.ndarray


@dataclass
class CLWLTrainConfig:
    model_type: Literal["linear", "mlp"] = "mlp"
    hidden_dim: int = 256
    batch_size: int = 128
    num_epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    device: str = "cpu"
    seed: int = 0
    log_every: int = 1

    # Stability / convergence controls
    gradient_clip_norm: Optional[float] = 5.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    min_learning_rate: float = 1e-5
    early_stop_patience: int = 12
    early_stop_min_delta: float = 1e-8


@dataclass
class CLWLEpochLog:
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
class CLWLTrainResult:
    model: nn.Module
    config: CLWLTrainConfig
    logs: list[CLWLEpochLog]
    final_train_metrics: dict[str, Any]
    final_val_metrics: Optional[dict[str, Any]]
    best_epoch: int
    best_monitor_value: float


class CLWLTrainingError(ValueError):
    pass


class TorchWeakLabelDataset(Dataset):
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



def _validate_dataset_and_T(dataset: WeakLabelDataset, T: Array) -> tuple[int, int, int]:
    X = np.asarray(dataset.X, dtype=np.float64)
    z = np.asarray(dataset.z, dtype=np.intp)
    eta = np.asarray(dataset.eta, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    if X.ndim != 2:
        raise CLWLTrainingError(f"dataset.X must be 2D, got shape {X.shape}.")
    if z.ndim != 1:
        raise CLWLTrainingError(f"dataset.z must be 1D, got shape {z.shape}.")
    if eta.ndim != 2:
        raise CLWLTrainingError(f"dataset.eta must be 2D, got shape {eta.shape}.")
    if T.ndim != 2:
        raise CLWLTrainingError(f"T must be 2D, got shape {T.shape}.")

    n, input_dim = X.shape
    n_z = z.shape[0]
    n_eta, num_classes = eta.shape
    c_T, d_T = T.shape

    if n_z != n or n_eta != n:
        raise CLWLTrainingError(
            f"dataset arrays disagree on sample count: X={n}, z={n_z}, eta={n_eta}."
        )
    if c_T != num_classes:
        raise CLWLTrainingError(
            f"T has {c_T} rows but dataset.eta has {num_classes} classes."
        )
    if np.min(z) < 0 or np.max(z) >= d_T:
        raise CLWLTrainingError(f"dataset.z entries must lie in [0, {d_T - 1}].")

    return input_dim, num_classes, d_T



def build_clwl_model(
    input_dim: int,
    num_classes: int,
    config: CLWLTrainConfig,
) -> nn.Module:
    if config.model_type == "linear":
        return LinearScoreModel(input_dim=input_dim, num_classes=num_classes)
    if config.model_type == "mlp":
        if config.hidden_dim <= 0:
            raise CLWLTrainingError(f"hidden_dim must be positive, got {config.hidden_dim}.")
        return MLPScoreModel(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_classes=num_classes,
        )
    raise CLWLTrainingError(f"Unsupported model_type={config.model_type}.")



def clwl_torch_loss(
    scores: torch.Tensor,
    z: torch.Tensor,
    T_torch: torch.Tensor,
) -> torch.Tensor:
    if scores.ndim != 2:
        raise CLWLTrainingError(f"scores must be 2D, got shape {tuple(scores.shape)}.")
    if z.ndim != 1:
        raise CLWLTrainingError(f"z must be 1D, got shape {tuple(z.shape)}.")
    if T_torch.ndim != 2:
        raise CLWLTrainingError(f"T_torch must be 2D, got shape {tuple(T_torch.shape)}.")

    n, c = scores.shape
    c_T, d = T_torch.shape
    if c_T != c:
        raise CLWLTrainingError(f"scores has {c} classes but T has {c_T} rows.")
    if z.shape[0] != n:
        raise CLWLTrainingError(f"z must have length {n}, got {z.shape[0]}.")
    if torch.min(z).item() < 0 or torch.max(z).item() >= d:
        raise CLWLTrainingError(f"z entries must lie in [0, {d - 1}].")

    T_cols = T_torch[:, z].transpose(0, 1)
    pos = T_cols * torch.nn.functional.softplus(-scores)
    neg = (1.0 - T_cols) * torch.nn.functional.softplus(scores)
    return (pos + neg).sum(dim=1).mean()



def make_dataloader(dataset: WeakLabelDataset, batch_size: int, *, shuffle: bool) -> DataLoader:
    if batch_size <= 0:
        raise CLWLTrainingError(f"batch_size must be positive, got {batch_size}.")
    torch_dataset = TorchWeakLabelDataset(dataset)
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
        raise CLWLTrainingError("scores_from_model received an empty dataset.")
    return np.concatenate(outputs, axis=0)



def empirical_risk_from_model(
    model: nn.Module,
    dataset: WeakLabelDataset,
    T: Array,
    *,
    batch_size: int = 512,
    device: str = "cpu",
) -> float:
    T_torch = torch.tensor(np.asarray(T, dtype=np.float32), dtype=torch.float32, device=torch.device(device))
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    total_count = 0
    device_obj = torch.device(device)

    with torch.no_grad():
        for x_batch, z_batch in loader:
            x_batch = x_batch.to(device_obj)
            z_batch = z_batch.to(device_obj)
            loss = clwl_torch_loss(model(x_batch), z_batch, T_torch)
            batch_n = int(x_batch.shape[0])
            total_loss += float(loss.item()) * batch_n
            total_count += batch_n

    if total_count == 0:
        raise CLWLTrainingError("empirical_risk_from_model received an empty dataset.")
    return total_loss / total_count



def evaluate_model_on_dataset(
    model: nn.Module,
    dataset: WeakLabelDataset,
    T: Array,
    *,
    batch_size: int = 512,
    device: str = "cpu",
) -> dict[str, Any]:
    scores = scores_from_model(model, dataset, batch_size=batch_size, device=device)
    metrics = evaluate_scores_on_dataset(scores, dataset)
    empirical_risk = empirical_risk_from_model(model, dataset, T, batch_size=batch_size, device=device)
    out = {
        "clean_accuracy": metrics.clean_accuracy,
        "max_preservation_rate": metrics.max_preservation_rate,
        "pairwise_order_rate": metrics.pairwise_order_rate,
        "pairwise_total": metrics.pairwise_total,
        "pairwise_correct": metrics.pairwise_correct,
        "mean_margin_on_ordered_pairs": metrics.mean_margin_on_ordered_pairs,
        "empirical_risk": empirical_risk,
    }
    return out



def _clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}



def train_clwl_model(
    train_dataset: WeakLabelDataset,
    T: Array,
    *,
    val_dataset: Optional[WeakLabelDataset] = None,
    config: Optional[CLWLTrainConfig] = None,
) -> CLWLTrainResult:
    if config is None:
        config = CLWLTrainConfig()

    input_dim, num_classes, _ = _validate_dataset_and_T(train_dataset, T)
    if val_dataset is not None:
        _validate_dataset_and_T(val_dataset, T)

    set_torch_seed(config.seed)
    device_obj = torch.device(config.device)
    model = build_clwl_model(input_dim=input_dim, num_classes=num_classes, config=config).to(device_obj)
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

    T_torch = torch.tensor(np.asarray(T, dtype=np.float32), dtype=torch.float32, device=device_obj)
    train_loader = make_dataloader(train_dataset, batch_size=config.batch_size, shuffle=True)

    logs: list[CLWLEpochLog] = []
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
            scores = model(x_batch)
            loss = clwl_torch_loss(scores, z_batch, T_torch)
            loss.backward()

            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)

            optimizer.step()

            batch_n = int(x_batch.shape[0])
            running_loss += float(loss.item()) * batch_n
            total_count += batch_n

        if total_count == 0:
            raise CLWLTrainingError("Training loader is empty.")
        train_loss = running_loss / total_count

        train_metrics = None
        val_metrics = None

        if epoch % config.log_every == 0 or epoch == config.num_epochs:
            train_metrics = evaluate_model_on_dataset(
                model,
                train_dataset,
                T,
                batch_size=max(config.batch_size, 512),
                device=config.device,
            )
            if val_dataset is not None:
                val_metrics = evaluate_model_on_dataset(
                    model,
                    val_dataset,
                    T,
                    batch_size=max(config.batch_size, 512),
                    device=config.device,
                )

            monitor_value = float(train_loss) if val_metrics is None else float(val_metrics["empirical_risk"])
            scheduler.step(monitor_value)
            current_learning_rate = float(optimizer.param_groups[0]["lr"])

            logs.append(
                CLWLEpochLog(
                    epoch=epoch,
                    train_loss=float(train_loss),
                    val_loss=None if val_metrics is None else float(val_metrics["empirical_risk"]),
                    train_clean_accuracy=float(train_metrics["clean_accuracy"]),
                    train_max_preservation_rate=float(train_metrics["max_preservation_rate"]),
                    train_pairwise_order_rate=float(train_metrics["pairwise_order_rate"]),
                    val_clean_accuracy=None if val_metrics is None else float(val_metrics["clean_accuracy"]),
                    val_max_preservation_rate=None if val_metrics is None else float(val_metrics["max_preservation_rate"]),
                    val_pairwise_order_rate=None if val_metrics is None else float(val_metrics["pairwise_order_rate"]),
                    current_learning_rate=current_learning_rate,
                )
            )

            if monitor_value < best_monitor_value - config.early_stop_min_delta:
                best_monitor_value = monitor_value
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = _clone_state_dict_to_cpu(model)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train_metrics = evaluate_model_on_dataset(
        model,
        train_dataset,
        T,
        batch_size=max(config.batch_size, 512),
        device=config.device,
    )
    final_val_metrics = None
    if val_dataset is not None:
        final_val_metrics = evaluate_model_on_dataset(
            model,
            val_dataset,
            T,
            batch_size=max(config.batch_size, 512),
            device=config.device,
        )

    return CLWLTrainResult(
        model=model,
        config=config,
        logs=logs,
        final_train_metrics=final_train_metrics,
        final_val_metrics=final_val_metrics,
        best_epoch=best_epoch,
        best_monitor_value=float(best_monitor_value),
    )



def plot_clwl_convergence(
    result: CLWLTrainResult,
    *,
    out_path: str = "clwl_convergence.png",
) -> None:
    import matplotlib.pyplot as plt

    if not result.logs:
        raise CLWLTrainingError("No training logs available for plotting.")

    epochs = [log.epoch for log in result.logs]
    train_loss = [log.train_loss for log in result.logs]
    val_loss = [np.nan if log.val_loss is None else log.val_loss for log in result.logs]
    train_pair = [log.train_pairwise_order_rate for log in result.logs]
    val_pair = [np.nan if log.val_pairwise_order_rate is None else log.val_pairwise_order_rate for log in result.logs]
    train_acc = [log.train_clean_accuracy for log in result.logs]
    val_acc = [np.nan if log.val_clean_accuracy is None else log.val_clean_accuracy for log in result.logs]
    lrs = [log.current_learning_rate for log in result.logs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    axes[0].plot(epochs, train_loss, label="train_loss")
    if not np.all(np.isnan(val_loss)):
        axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].axvline(result.best_epoch, linestyle="--", alpha=0.7, label="best_epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("CLWL convergence: loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_pair, label="train_pairwise_order_rate")
    if not np.all(np.isnan(val_pair)):
        axes[1].plot(epochs, val_pair, label="val_pairwise_order_rate")
    axes[1].plot(epochs, train_acc, label="train_clean_accuracy")
    if not np.all(np.isnan(val_acc)):
        axes[1].plot(epochs, val_acc, label="val_clean_accuracy")
    axes[1].axvline(result.best_epoch, linestyle="--", alpha=0.7)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("metric")
    axes[1].set_title("CLWL convergence: ranking / accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, lrs, label="learning_rate")
    axes[2].axvline(result.best_epoch, linestyle="--", alpha=0.7)
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("lr")
    axes[2].set_title("CLWL convergence: learning rate")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    from clwl_experiments_module1_t_construction import construct_clwl_T
    from clwl_experiments_module2_weak_label_generators import make_uniform_partial_label_family
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
    family = make_uniform_partial_label_family(c=4, candidate_size=2)
    weak_splits = build_weak_label_splits(splits, family, seed=10)

    construction = construct_clwl_T(weak_splits["train"].M)
    T = construction.T

    config = CLWLTrainConfig(
        model_type="mlp",
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
    )
    result = train_clwl_model(
        train_dataset=weak_splits["train"],
        val_dataset=weak_splits["val"],
        T=T,
        config=config,
    )
    plot_clwl_convergence(result, out_path="clwl_convergence_demo.png")

    print("=== Final train metrics ===")
    print(result.final_train_metrics)
    print("\n=== Final val metrics ===")
    print(result.final_val_metrics)
    print(f"\n=== Best epoch === {result.best_epoch}")
    print(f"=== Best monitor value === {result.best_monitor_value:.6f}")
    print("\n=== Last epoch log ===")
    print(asdict(result.logs[-1]))
