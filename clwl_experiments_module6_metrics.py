from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from clwl_experiments_module4_weak_label_dataset import WeakLabelDataset


Array = np.ndarray


@dataclass
class MetricsSummary:
    clean_accuracy: float
    max_preservation_rate: float
    pairwise_order_rate: float
    pairwise_total: int
    pairwise_correct: int
    mean_margin_on_ordered_pairs: float
    metadata: dict[str, Any]


class MetricsError(ValueError):
    pass


def _as_float_2d(name: str, x: Array) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise MetricsError(f"{name} must be 2D, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise MetricsError(f"{name} must contain only finite values.")
    return arr


def _as_index_1d(name: str, x: Array) -> Array:
    arr = np.asarray(x, dtype=np.intp)
    if arr.ndim != 1:
        raise MetricsError(f"{name} must be 1D, got shape {arr.shape}.")
    return arr


def _maybe_float_2d(name: str, x: Optional[Array]) -> Optional[Array]:
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise MetricsError(f"{name} must be 2D, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise MetricsError(f"{name} must contain only finite values.")
    return arr


def validate_scores_and_dataset(
    scores: Array,
    dataset: WeakLabelDataset,
    *,
    require_eta: bool = False,
) -> tuple[int, int, Optional[Array]]:
    scores = _as_float_2d("scores", scores)
    y = _as_index_1d("dataset.y", dataset.y)
    eta = _maybe_float_2d("dataset.eta", getattr(dataset, "eta", None))

    n, c = scores.shape
    if y.shape[0] != n:
        raise MetricsError(f"dataset.y must have length {n}, got {y.shape[0]}.")
    if np.min(y) < 0 or np.max(y) >= c:
        raise MetricsError(f"dataset.y entries must lie in [0, {c - 1}].")

    if require_eta and eta is None:
        raise MetricsError("dataset.eta is required but missing.")
    if eta is not None and eta.shape != (n, c):
        raise MetricsError(f"dataset.eta must have shape {(n, c)}, got {eta.shape}.")

    return n, c, eta


def predict_top1(scores: Array) -> Array:
    scores = _as_float_2d("scores", scores)
    return np.argmax(scores, axis=1).astype(np.intp)


def eta_argmax(scores_or_eta: Array, *, atol: float = 1e-12) -> Array:
    arr = _as_float_2d("scores_or_eta", scores_or_eta)
    row_max = np.max(arr, axis=1, keepdims=True)
    return np.isclose(arr, row_max, atol=atol, rtol=0.0)


def clean_top1_accuracy(scores: Array, y: Array) -> float:
    scores = _as_float_2d("scores", scores)
    y = _as_index_1d("y", y)
    if scores.shape[0] != y.shape[0]:
        raise MetricsError(
            f"scores and y must agree on sample size, got {scores.shape[0]} and {y.shape[0]}."
        )
    pred = predict_top1(scores)
    return float(np.mean(pred == y))


def max_preservation_rate(
    scores: Array,
    eta: Array,
    *,
    atol: float = 1e-12,
    require_unique_eta_max: bool = False,
) -> float:
    scores = _as_float_2d("scores", scores)
    eta = _as_float_2d("eta", eta)
    if scores.shape != eta.shape:
        raise MetricsError(f"scores shape {scores.shape} must match eta shape {eta.shape}.")

    score_max_mask = eta_argmax(scores, atol=atol)
    eta_max_mask = eta_argmax(eta, atol=atol)

    if require_unique_eta_max:
        unique_mask = np.sum(eta_max_mask, axis=1) == 1
        if not np.any(unique_mask):
            raise MetricsError("No samples have a unique eta argmax under the requested setting.")
        matches = np.all(score_max_mask[unique_mask] == eta_max_mask[unique_mask], axis=1)
        return float(np.mean(matches))

    matches = np.all(score_max_mask == eta_max_mask, axis=1)
    return float(np.mean(matches))


def pairwise_order_statistics(
    scores: Array,
    eta: Array,
    *,
    atol: float = 1e-12,
    margin_atol: float = 0.0,
) -> dict[str, float | int]:
    scores = _as_float_2d("scores", scores)
    eta = _as_float_2d("eta", eta)
    if scores.shape != eta.shape:
        raise MetricsError(f"scores shape {scores.shape} must match eta shape {eta.shape}.")

    n, c = scores.shape
    total = 0
    correct = 0
    margin_sum = 0.0

    for i in range(n):
        eta_i = eta[i]
        scores_i = scores[i]
        for a in range(c):
            for b in range(c):
                if a == b:
                    continue
                if eta_i[a] > eta_i[b] + atol:
                    total += 1
                    margin = float(scores_i[a] - scores_i[b])
                    margin_sum += margin
                    if margin > margin_atol:
                        correct += 1

    rate = float(correct / total) if total > 0 else float("nan")
    mean_margin = float(margin_sum / total) if total > 0 else float("nan")
    return {
        "pairwise_total": int(total),
        "pairwise_correct": int(correct),
        "pairwise_order_rate": rate,
        "mean_margin_on_ordered_pairs": mean_margin,
    }


def evaluate_scores_on_dataset(
    scores: Array,
    dataset: WeakLabelDataset,
    *,
    atol: float = 1e-12,
    margin_atol: float = 0.0,
    require_unique_eta_max: bool = False,
) -> MetricsSummary:
    n, c, eta = validate_scores_and_dataset(scores, dataset, require_eta=False)
    scores = np.asarray(scores, dtype=np.float64)

    acc = clean_top1_accuracy(scores, dataset.y)
    dataset_meta = dict(getattr(dataset, "metadata", {}) or {})
    eta_source = dataset_meta.get("eta_source", "unknown")
    oracle_metrics = bool(dataset_meta.get("oracle_metrics", eta_source == "oracle"))

    metadata = {
        "family_name": dataset.family_name,
        "num_samples": int(n),
        "num_classes": int(c),
        "atol": float(atol),
        "margin_atol": float(margin_atol),
        "require_unique_eta_max": bool(require_unique_eta_max),
        "eta_source": eta_source,
        "oracle_metrics": oracle_metrics,
    }

    if eta is None:
        metadata["eta_available"] = False
        metadata["order_metrics_available"] = False
        return MetricsSummary(
            clean_accuracy=acc,
            max_preservation_rate=float("nan"),
            pairwise_order_rate=float("nan"),
            pairwise_total=0,
            pairwise_correct=0,
            mean_margin_on_ordered_pairs=float("nan"),
            metadata=metadata,
        )

    max_rate = max_preservation_rate(
        scores,
        eta,
        atol=atol,
        require_unique_eta_max=require_unique_eta_max,
    )
    pairwise_stats = pairwise_order_statistics(
        scores,
        eta,
        atol=atol,
        margin_atol=margin_atol,
    )

    metadata["eta_available"] = True
    metadata["order_metrics_available"] = True

    return MetricsSummary(
        clean_accuracy=acc,
        max_preservation_rate=max_rate,
        pairwise_order_rate=float(pairwise_stats["pairwise_order_rate"]),
        pairwise_total=int(pairwise_stats["pairwise_total"]),
        pairwise_correct=int(pairwise_stats["pairwise_correct"]),
        mean_margin_on_ordered_pairs=float(pairwise_stats["mean_margin_on_ordered_pairs"]),
        metadata=metadata,
    )


def scores_from_logits(logits: Array) -> Array:
    logits = _as_float_2d("logits", logits)
    return logits.copy()


def zero_scores_like_dataset(dataset: WeakLabelDataset) -> Array:
    template = getattr(dataset, "eta", None)
    if template is None:
        template = getattr(dataset, "logits", None)

    if template is not None:
        template = _as_float_2d("template", template)
        return np.zeros_like(template, dtype=np.float64)

    M = _as_float_2d("dataset.M", dataset.M)
    y = _as_index_1d("dataset.y", dataset.y)
    return np.zeros((y.shape[0], M.shape[1]), dtype=np.float64)