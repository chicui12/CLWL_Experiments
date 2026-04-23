from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from clwl_experiments_module4_weak_label_dataset import WeakLabelDataset


Array = np.ndarray


@dataclass
class CLWLLossSummary:
    sample_losses: Array
    empirical_risk: float
    q: Array | None
    conditional_risks: Array | None
    metadata: dict[str, Any]


class CLWLLossError(ValueError):
    pass



def _as_float_1d(name: str, x: Array) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise CLWLLossError(f"{name} must be 1D, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise CLWLLossError(f"{name} must contain only finite values.")
    return arr



def _as_float_2d(name: str, x: Array) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise CLWLLossError(f"{name} must be 2D, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise CLWLLossError(f"{name} must contain only finite values.")
    return arr



def _as_index_1d(name: str, x: Array) -> Array:
    arr = np.asarray(x, dtype=np.intp)
    if arr.ndim != 1:
        raise CLWLLossError(f"{name} must be 1D, got shape {arr.shape}.")
    return arr



def validate_clwl_shapes(T: Array, scores: Array, z: Array) -> tuple[int, int, int]:
    T = _as_float_2d("T", T)
    scores = _as_float_2d("scores", scores)
    z = _as_index_1d("z", z)

    c, d = T.shape
    n, c_scores = scores.shape
    if c_scores != c:
        raise CLWLLossError(
            f"scores has {c_scores} class dimensions but T has {c}."
        )
    if z.shape[0] != n:
        raise CLWLLossError(f"z must have length {n}, got {z.shape[0]}.")
    if np.min(z) < 0 or np.max(z) >= d:
        raise CLWLLossError(f"z entries must lie in [0, {d - 1}].")

    return n, c, d



def logistic_beta(f: Array) -> Array:
    f = np.asarray(f, dtype=np.float64)
    if not np.all(np.isfinite(f)):
        raise CLWLLossError("logistic_beta input must contain only finite values.")
    return np.logaddexp(0.0, -f)



def logistic_beta_derivative(f: Array) -> Array:
    f = np.asarray(f, dtype=np.float64)
    if not np.all(np.isfinite(f)):
        raise CLWLLossError("logistic_beta_derivative input must contain only finite values.")
    return -1.0 / (1.0 + np.exp(f))



def gather_t_columns(T: Array, z: Array) -> Array:
    T = _as_float_2d("T", T)
    z = _as_index_1d("z", z)
    _, d = T.shape
    if np.min(z) < 0 or np.max(z) >= d:
        raise CLWLLossError(f"z entries must lie in [0, {d - 1}].")
    return T[:, z].T.copy()



def clwl_sample_losses(scores: Array, z: Array, T: Array) -> Array:
    n, c, _ = validate_clwl_shapes(T, scores, z)
    scores = np.asarray(scores, dtype=np.float64)
    z = np.asarray(z, dtype=np.intp)
    T_columns = gather_t_columns(T, z)
    if T_columns.shape != (n, c):
        raise CLWLLossError(
            f"Gathered T-columns must have shape {(n, c)}, got {T_columns.shape}."
        )

    pos_term = T_columns * logistic_beta(scores)
    neg_term = (1.0 - T_columns) * logistic_beta(-scores)
    losses = np.sum(pos_term + neg_term, axis=1)
    return losses



def clwl_empirical_risk(scores: Array, z: Array, T: Array) -> float:
    losses = clwl_sample_losses(scores, z, T)
    return float(np.mean(losses))



def clwl_score_gradients(scores: Array, z: Array, T: Array) -> Array:
    n, c, _ = validate_clwl_shapes(T, scores, z)
    scores = np.asarray(scores, dtype=np.float64)
    z = np.asarray(z, dtype=np.intp)
    T_columns = gather_t_columns(T, z)
    grad_pos = T_columns * logistic_beta_derivative(scores)
    grad_neg = -(1.0 - T_columns) * logistic_beta_derivative(-scores)
    grads = grad_pos + grad_neg
    if grads.shape != (n, c):
        raise CLWLLossError(f"Gradient matrix must have shape {(n, c)}, got {grads.shape}.")
    return grads



def compute_A_from_M_and_T(M: Array, T: Array) -> Array:
    M = _as_float_2d("M", M)
    T = _as_float_2d("T", T)
    c, d = T.shape
    d_M, c_M = M.shape
    if d_M != d or c_M != c:
        raise CLWLLossError(
            f"Shape mismatch: T has shape {T.shape}, M has shape {M.shape}."
        )
    return T @ M



def compute_q_from_eta(eta: Array, M: Array, T: Array) -> Array:
    eta = _as_float_2d("eta", eta)
    A = compute_A_from_M_and_T(M, T)
    c = A.shape[0]
    if eta.shape[1] != c:
        raise CLWLLossError(
            f"eta has {eta.shape[1]} classes but A has size {c}."
        )
    q = eta @ A.T
    if q.shape != eta.shape:
        raise CLWLLossError(f"q must have shape {eta.shape}, got {q.shape}.")
    return q



def clwl_conditional_risks(scores: Array, eta: Array, M: Array, T: Array) -> tuple[Array, Array]:
    scores = _as_float_2d("scores", scores)
    eta = _as_float_2d("eta", eta)
    if scores.shape != eta.shape:
        raise CLWLLossError(
            f"scores shape {scores.shape} must match eta shape {eta.shape}."
        )
    q = compute_q_from_eta(eta, M, T)
    risks = np.sum(q * logistic_beta(scores) + (1.0 - q) * logistic_beta(-scores), axis=1)
    return q, risks



def clwl_summary_from_dataset(scores: Array, dataset: WeakLabelDataset, T: Array) -> CLWLLossSummary:
    scores = _as_float_2d("scores", scores)
    if scores.shape != dataset.eta.shape:
        raise CLWLLossError(
            f"scores shape {scores.shape} must match dataset.eta shape {dataset.eta.shape}."
        )

    sample_losses = clwl_sample_losses(scores, dataset.z, T)
    empirical_risk = float(np.mean(sample_losses))
    q, conditional_risks = clwl_conditional_risks(scores, dataset.eta, dataset.M, T)

    metadata = {
        "family_name": dataset.family_name,
        "num_samples": int(scores.shape[0]),
        "num_classes": int(scores.shape[1]),
        "empirical_risk": empirical_risk,
        "mean_conditional_risk": float(np.mean(conditional_risks)),
    }

    return CLWLLossSummary(
        sample_losses=sample_losses,
        empirical_risk=empirical_risk,
        q=q,
        conditional_risks=conditional_risks,
        metadata=metadata,
    )



def zero_scores(num_samples: int, num_classes: int) -> Array:
    if num_samples <= 0 or num_classes <= 0:
        raise CLWLLossError(
            f"num_samples and num_classes must be positive, got {num_samples}, {num_classes}."
        )
    return np.zeros((int(num_samples), int(num_classes)), dtype=np.float64)


if __name__ == "__main__":
    from clwl_experiments_module1_t_construction import construct_clwl_T
    from clwl_experiments_module2_weak_label_generators import (
        make_uniform_complementary_family,
        make_uniform_partial_label_family,
    )
    from clwl_experiments_module3_synthetic_clean_data import (
        generate_linear_softmax_dataset,
        train_val_test_split,
    )
    from clwl_experiments_module4_weak_label_dataset import build_weak_label_splits, weak_dataset_summary

    ds = generate_linear_softmax_dataset(
        n=600,
        input_dim=6,
        num_classes=4,
        feature_seed=0,
        teacher_seed=1,
        label_seed=2,
    )
    splits = train_val_test_split(ds, train_frac=0.6, val_frac=0.2, seed=42)

    print("=== CLWL loss on partial-label family ===")
    partial_family = make_uniform_partial_label_family(c=4, candidate_size=2)
    partial_splits = build_weak_label_splits(splits, partial_family, seed=10)
    partial_train = partial_splits["train"]
    T_partial = construct_clwl_T(partial_train.M).T
    scores_partial = zero_scores(partial_train.X.shape[0], partial_train.eta.shape[1])
    summary_partial = clwl_summary_from_dataset(scores_partial, partial_train, T_partial)
    print(weak_dataset_summary(partial_train))
    print(summary_partial.metadata)

    print("\n=== CLWL loss on complementary-label family ===")
    comp_family = make_uniform_complementary_family(c=4)
    comp_splits = build_weak_label_splits(splits, comp_family, seed=20)
    comp_train = comp_splits["train"]
    T_comp = construct_clwl_T(comp_train.M).T
    scores_comp = zero_scores(comp_train.X.shape[0], comp_train.eta.shape[1])
    summary_comp = clwl_summary_from_dataset(scores_comp, comp_train, T_comp)
    print(weak_dataset_summary(comp_train))
    print(summary_comp.metadata)
