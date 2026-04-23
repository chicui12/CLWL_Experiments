from __future__ import annotations

"""
Module 2: weak-label data generators.

This module solves one core problem:
- generate weak-label transition matrices M for the experiment families we want to compare:
  1) partial-label protocols,
  2) uniform complementary-label protocols,
  3) non-uniform complementary-label protocols,
  4) general weak-label protocols.

Interface with existing modules:
- this module outputs transition matrices M with shape (d, c), where d is the number of weak labels
  and c is the number of clean classes;
- Module 1 can consume any returned M via `construct_clwl_T(M)`;
- later training modules can also use the returned weak-label library (binary-set matrix Z or label names)
  to sample weak labels for each clean class.

Conventions:
- M[z, y] = P(weak label z | clean class y)
- every returned M is column-stochastic;
- for partial-label families, we also return a binary weak-label library Z with shape (c, d),
  whose k-th column is the candidate-set indicator vector for weak label k.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Optional

import numpy as np


Array = np.ndarray


@dataclass
class WeakLabelFamily:
    """Container for a weak-label family and its transition matrix."""

    name: str
    M: Array  # shape (d, c)
    weak_label_matrix: Optional[Array] = None  # shape (c, d), columns are weak-label vectors when available
    weak_label_names: Optional[list[str]] = None
    metadata: Optional[dict] = None


class WeakLabelGeneratorError(ValueError):
    """Raised when a requested weak-label family is invalid."""


def _validate_c(c: int) -> None:
    if c < 2:
        raise WeakLabelGeneratorError(f"Need c >= 2 classes, got c={c}.")


def _ensure_column_stochastic(M: Array, *, atol: float = 1e-10) -> None:
    if M.ndim != 2:
        raise WeakLabelGeneratorError(f"M must be 2D, got shape {M.shape}.")
    if np.min(M) < -atol:
        raise WeakLabelGeneratorError(f"M has negative entries: min={np.min(M):.3e}.")
    col_sums = M.sum(axis=0)
    if not np.allclose(col_sums, np.ones(M.shape[1]), atol=atol, rtol=0.0):
        raise WeakLabelGeneratorError(
            f"M is not column-stochastic. Column sums are {col_sums}."
        )
    print(f"[_ensure_column_stochastic] shape={M.shape}, min={np.min(M):.6f}, max={np.max(M):.6f}")


def _normalize_columns(M: Array) -> Array:
    M = np.asarray(M, dtype=np.float64)
    col_sums = M.sum(axis=0, keepdims=True)
    if np.any(col_sums <= 0):
        raise WeakLabelGeneratorError("Every column must have positive mass before normalization.")
    out = M / col_sums
    print(f"[_normalize_columns] shape={out.shape}, col_sums_after={out.sum(axis=0)}")
    return out


def _subset_name(indices: Iterable[int], *, prefix: str = "{") -> str:
    items = ",".join(str(i) for i in indices)
    return f"{prefix}{items}}}"


def enumerate_candidate_sets(
    c: int,
    *,
    min_size: int = 2,
    max_size: Optional[int] = None,
    include_singletons: bool = False,
) -> tuple[Array, list[str]]:
    """
    Enumerate candidate-label sets for partial-label experiments.

    Returns
    -------
    Z:
        Binary matrix with shape (c, d). Each column is one candidate set.
    names:
        Human-readable names for the candidate sets.
    """
    _validate_c(c)

    if max_size is None:
        max_size = c
    if include_singletons:
        min_size = min(min_size, 1)

    if not (1 <= min_size <= max_size <= c):
        raise WeakLabelGeneratorError(
            f"Invalid set sizes: min_size={min_size}, max_size={max_size}, c={c}."
        )

    cols: list[Array] = []
    names: list[str] = []
    for size in range(min_size, max_size + 1):
        for idxs in combinations(range(c), size):
            col = np.zeros(c, dtype=np.float64)
            col[list(idxs)] = 1.0
            cols.append(col)
            names.append(_subset_name(idxs))

    if not cols:
        raise WeakLabelGeneratorError("No candidate sets were generated.")

    Z = np.stack(cols, axis=1)
    print(
        f"[enumerate_candidate_sets] c={c}, min_size={min_size}, max_size={max_size}, "
        f"include_singletons={include_singletons}, num_sets={Z.shape[1]}"
    )
    return Z, names


def _top_three_confusions(confusion: Array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    confusion[y, j] should measure how likely true class y is confused with class j.
    Diagonal is ignored.
    Returns:
        primary[y]   = most confusing distractor for y
        secondary[y] = second most confusing distractor for y
        tertiary[y]  = third most confusing distractor for y
    """
    confusion = np.asarray(confusion, dtype=np.float64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise WeakLabelGeneratorError(
            f"confusion must be square, got shape {confusion.shape}."
        )
    if np.min(confusion) < 0:
        raise WeakLabelGeneratorError("confusion must be nonnegative.")

    C = confusion.copy()
    np.fill_diagonal(C, -np.inf)
    order = np.argsort(-C, axis=1)
    primary = order[:, 0]
    secondary = order[:, 1]
    tertiary = order[:, 2]
    return primary.astype(np.int64), secondary.astype(np.int64), tertiary.astype(np.int64)


def make_uniform_partial_label_family(
    c: int,
    *,
    candidate_size: int = 2,
) -> WeakLabelFamily:
    """
    Construct a uniform partial-label family where every candidate set has the same size.

    For each clean class y, all candidate sets of the specified size that contain y are equally likely.
    """
    _validate_c(c)
    if not (1 <= candidate_size <= c):
        raise WeakLabelGeneratorError(
            f"candidate_size must be in [1, c], got {candidate_size}."
        )

    Z, names = enumerate_candidate_sets(c, min_size=candidate_size, max_size=candidate_size)
    d = Z.shape[1]
    M = np.zeros((d, c), dtype=np.float64)

    for y in range(c):
        mask = Z[y, :] == 1.0
        count = int(np.sum(mask))
        if count == 0:
            raise WeakLabelGeneratorError(f"No candidate set contains class y={y}.")
        M[mask, y] = 1.0 / count

    _ensure_column_stochastic(M)
    family = WeakLabelFamily(
        name=f"partial_uniform_size_{candidate_size}",
        M=M,
        weak_label_matrix=Z,
        weak_label_names=names,
        metadata={"candidate_size": candidate_size, "type": "partial_uniform"},
    )
    print(
        f"[make_uniform_partial_label_family] name={family.name}, shape={family.M.shape}, "
        f"rank={np.linalg.matrix_rank(family.M)}"
    )
    return family


def make_size_mixture_partial_label_family(
    c: int,
    *,
    size_weights: dict[int, float],
) -> WeakLabelFamily:
    """
    Construct a partial-label family with a mixture over candidate-set sizes.

    Example
    -------
    size_weights={2: 0.7, 3: 0.3}
    """
    _validate_c(c)
    if not size_weights:
        raise WeakLabelGeneratorError("size_weights cannot be empty.")

    sizes = sorted(size_weights.keys())
    if any((s < 1 or s > c) for s in sizes):
        raise WeakLabelGeneratorError(f"Invalid candidate sizes in {sizes} for c={c}.")

    weight_sum = float(sum(size_weights.values()))
    if weight_sum <= 0:
        raise WeakLabelGeneratorError("size_weights must sum to a positive value.")
    normalized_size_weights = {s: float(w) / weight_sum for s, w in size_weights.items()}

    Z, names = enumerate_candidate_sets(
        c,
        min_size=min(sizes),
        max_size=max(sizes),
        include_singletons=(1 in sizes),
    )
    d = Z.shape[1]
    set_sizes = Z.sum(axis=0).astype(int)
    M = np.zeros((d, c), dtype=np.float64)

    for y in range(c):
        for size, size_weight in normalized_size_weights.items():
            mask = (Z[y, :] == 1.0) & (set_sizes == size)
            count = int(np.sum(mask))
            if count == 0:
                raise WeakLabelGeneratorError(
                    f"No candidate set of size {size} contains class y={y}."
                )
            M[mask, y] += size_weight / count

    _ensure_column_stochastic(M)
    family = WeakLabelFamily(
        name="partial_size_mixture",
        M=M,
        weak_label_matrix=Z,
        weak_label_names=names,
        metadata={"size_weights": normalized_size_weights, "type": "partial_size_mixture"},
    )
    print(
        f"[make_size_mixture_partial_label_family] name={family.name}, shape={family.M.shape}, "
        f"rank={np.linalg.matrix_rank(family.M)}, size_weights={normalized_size_weights}"
    )
    return family


def make_biased_partial_label_family(
    c: int,
    *,
    candidate_size: int,
    distractor_affinity: Array,
) -> WeakLabelFamily:
    """
    Construct a biased partial-label family.

    Parameters
    ----------
    c:
        Number of clean classes.
    candidate_size:
        Size of each candidate set.
    distractor_affinity:
        Array of shape (c, c). Entry [y, j] controls how likely class j is to appear
        as a distractor when the clean class is y. The diagonal is ignored.

    Notes
    -----
    This is useful for creating dominance-friendly or dominance-violating partial-label regimes.
    For a fixed clean class y, the probability of a candidate set S that contains y is proportional to
        prod_{j in S, j != y} affinity[y, j].
    """
    _validate_c(c)
    if not (1 <= candidate_size <= c):
        raise WeakLabelGeneratorError(
            f"candidate_size must be in [1, c], got {candidate_size}."
        )
    distractor_affinity = np.asarray(distractor_affinity, dtype=np.float64)
    if distractor_affinity.shape != (c, c):
        raise WeakLabelGeneratorError(
            f"distractor_affinity must have shape {(c, c)}, got {distractor_affinity.shape}."
        )
    if np.min(distractor_affinity) < 0:
        raise WeakLabelGeneratorError("distractor_affinity must be nonnegative.")

    Z, names = enumerate_candidate_sets(c, min_size=candidate_size, max_size=candidate_size)
    d = Z.shape[1]
    M = np.zeros((d, c), dtype=np.float64)

    for y in range(c):
        weights = np.zeros(d, dtype=np.float64)
        for k in range(d):
            if Z[y, k] == 0:
                continue
            members = np.where(Z[:, k] == 1)[0]
            distractors = [j for j in members if j != y]
            weight = 1.0
            for j in distractors:
                weight *= max(distractor_affinity[y, j], 0.0)
            weights[k] = weight

        if np.sum(weights) <= 0:
            raise WeakLabelGeneratorError(
                f"All candidate-set weights vanished for clean class y={y}."
            )
        M[:, y] = weights / np.sum(weights)

    _ensure_column_stochastic(M)
    family = WeakLabelFamily(
        name=f"partial_biased_size_{candidate_size}",
        M=M,
        weak_label_matrix=Z,
        weak_label_names=names,
        metadata={"candidate_size": candidate_size, "type": "partial_biased"},
    )
    print(
        f"[make_biased_partial_label_family] name={family.name}, shape={family.M.shape}, "
        f"rank={np.linalg.matrix_rank(family.M)}"
    )
    return family


def make_confusion_aware_quad_partial_label_family(
    c: int,
    *,
    confusion: Array,
    lambda3: float = 32.0,
    lambda2: float = 8.0,
    lambda1: float = 2.0,
    base_weight: float = 1.0,
) -> WeakLabelFamily:
    """
    Construct a size-4 confusion-aware partial-label family designed to be much harder for CLPL.

    Weak labels are all size-4 candidate sets S.
    For each clean class y:
      - sets containing y and all three main distractors {r1(y), r2(y), r3(y)} get weight lambda3
      - sets containing y and exactly two of them get weight lambda2
      - sets containing y and exactly one of them get weight lambda1
      - all other size-4 sets containing y get weight base_weight

    where r1(y), r2(y), r3(y) are the three strongest off-diagonal confusions for class y.
    """
    _validate_c(c)

    if c < 4:
        raise WeakLabelGeneratorError(f"Need c >= 4 for size-4 families, got c={c}.")
    if not (lambda3 > lambda2 > lambda1 >= base_weight > 0):
        raise WeakLabelGeneratorError(
            "Need lambda3 > lambda2 > lambda1 >= base_weight > 0. "
            f"Got lambda3={lambda3}, lambda2={lambda2}, lambda1={lambda1}, base_weight={base_weight}."
        )

    confusion = np.asarray(confusion, dtype=np.float64)
    if confusion.shape != (c, c):
        raise WeakLabelGeneratorError(
            f"confusion must have shape {(c, c)}, got {confusion.shape}."
        )
    if np.min(confusion) < 0:
        raise WeakLabelGeneratorError("confusion must be nonnegative.")

    primary, secondary, tertiary = _top_three_confusions(confusion)

    Z, names = enumerate_candidate_sets(c, min_size=4, max_size=4)
    d = Z.shape[1]
    M = np.zeros((d, c), dtype=np.float64)

    for y in range(c):
        core = {int(primary[y]), int(secondary[y]), int(tertiary[y])}

        weights = np.zeros(d, dtype=np.float64)
        for k in range(d):
            if Z[y, k] == 0.0:
                continue

            members = np.where(Z[:, k] == 1.0)[0]
            distractors = [j for j in members if j != y]
            hit_count = sum(int(j in core) for j in distractors)

            if hit_count == 3:
                weights[k] = lambda3
            elif hit_count == 2:
                weights[k] = lambda2
            elif hit_count == 1:
                weights[k] = lambda1
            else:
                weights[k] = base_weight

        if np.sum(weights) <= 0:
            raise WeakLabelGeneratorError(
                f"All size-4 weights vanished for clean class y={y}."
            )
        M[:, y] = weights / np.sum(weights)

    _ensure_column_stochastic(M)
    family = WeakLabelFamily(
        name=f"partial_confusion_aware_quad_l3_{lambda3:g}_l2_{lambda2:g}_l1_{lambda1:g}",
        M=M,
        weak_label_matrix=Z,
        weak_label_names=names,
        metadata={
            "type": "partial_confusion_aware_quad",
            "candidate_size": 4,
            "lambda3": float(lambda3),
            "lambda2": float(lambda2),
            "lambda1": float(lambda1),
            "base_weight": float(base_weight),
            "primary_confusions": primary.tolist(),
            "secondary_confusions": secondary.tolist(),
            "tertiary_confusions": tertiary.tolist(),
        },
    )
    print(
        f"[make_confusion_aware_quad_partial_label_family] name={family.name}, "
        f"shape={family.M.shape}, rank={np.linalg.matrix_rank(family.M)}"
    )
    return family


def make_manual_mnist_quad_partial_label_family(
    *,
    lambda3: float = 32.0,
    lambda2: float = 8.0,
    lambda1: float = 2.0,
) -> WeakLabelFamily:
    """
    Hand-crafted MNIST size-4 partial-label family for a stronger CLPL-hard regime.

    Main distractor triples per true class:
        0 -> (6, 8, 5)
        1 -> (7, 2, 4)
        2 -> (7, 3, 8)
        3 -> (5, 8, 2)
        4 -> (9, 7, 1)
        5 -> (3, 8, 6)
        6 -> (0, 8, 5)
        7 -> (1, 2, 9)
        8 -> (3, 5, 2)
        9 -> (4, 7, 8)
    """
    c = 10
    primary =   [6, 7, 7, 5, 9, 3, 0, 1, 3, 4]
    secondary = [8, 2, 3, 8, 7, 8, 8, 2, 5, 7]
    tertiary =  [5, 4, 8, 2, 1, 6, 5, 9, 2, 8]

    confusion = np.zeros((c, c), dtype=np.float64)
    for y in range(c):
        confusion[y, primary[y]] = 3.0
        confusion[y, secondary[y]] = 2.0
        confusion[y, tertiary[y]] = 1.0

    family = make_confusion_aware_quad_partial_label_family(
        c=c,
        confusion=confusion,
        lambda3=lambda3,
        lambda2=lambda2,
        lambda1=lambda1,
        base_weight=1.0,
    )
    family.name = (
        f"mnist_manual_quad_partial_l3_{lambda3:g}_l2_{lambda2:g}_l1_{lambda1:g}"
    )
    family.metadata = {
        **(family.metadata or {}),
        "type": "mnist_manual_quad_partial",
        "primary": primary,
        "secondary": secondary,
        "tertiary": tertiary,
    }
    print(
        f"[make_manual_mnist_quad_partial_label_family] name={family.name}, "
        f"shape={family.M.shape}, rank={np.linalg.matrix_rank(family.M)}"
    )
    return family


def make_uniform_complementary_family(c: int) -> WeakLabelFamily:
    """
    Construct the standard single complementary-label family.

    Weak labels are singleton complementary labels z in {0, ..., c-1}.
    For each clean class y, all incorrect complementary labels are equally likely.
    """
    _validate_c(c)
    M = np.ones((c, c), dtype=np.float64) - np.eye(c, dtype=np.float64)
    M /= (c - 1)

    weak_label_names = [f"not_{i}" for i in range(c)]
    _ensure_column_stochastic(M)
    family = WeakLabelFamily(
        name="complementary_uniform",
        M=M,
        weak_label_matrix=None,
        weak_label_names=weak_label_names,
        metadata={"type": "complementary_uniform"},
    )
    print(
        f"[make_uniform_complementary_family] name={family.name}, shape={family.M.shape}, "
        f"rank={np.linalg.matrix_rank(family.M)}"
    )
    return family


def make_nonuniform_complementary_family(
    c: int,
    *,
    wrong_label_weights: Array,
) -> WeakLabelFamily:
    """
    Construct a single complementary-label family with non-uniform wrong-label sampling.

    Parameters
    ----------
    wrong_label_weights:
        Array of shape (c, c). For each clean class y, the column wrong_label_weights[:, y]
        gives unnormalized probabilities over complementary labels. The diagonal entries are
        ignored and forced to zero.
    """
    _validate_c(c)
    W = np.asarray(wrong_label_weights, dtype=np.float64)
    if W.shape != (c, c):
        raise WeakLabelGeneratorError(
            f"wrong_label_weights must have shape {(c, c)}, got {W.shape}."
        )
    if np.min(W) < 0:
        raise WeakLabelGeneratorError("wrong_label_weights must be nonnegative.")

    W = W.copy()
    np.fill_diagonal(W, 0.0)

    for y in range(c):
        if np.sum(W[:, y]) <= 0:
            raise WeakLabelGeneratorError(
                f"Column y={y} has no positive mass on wrong complementary labels."
            )

    M = _normalize_columns(W)
    weak_label_names = [f"not_{i}" for i in range(c)]
    _ensure_column_stochastic(M)
    family = WeakLabelFamily(
        name="complementary_nonuniform",
        M=M,
        weak_label_matrix=None,
        weak_label_names=weak_label_names,
        metadata={"type": "complementary_nonuniform"},
    )
    print(
        f"[make_nonuniform_complementary_family] name={family.name}, shape={family.M.shape}, "
        f"rank={np.linalg.matrix_rank(family.M)}"
    )
    return family


def make_noisy_complementary_family(c: int, *, q: float) -> WeakLabelFamily:
    """
    Construct the noisy complementary-label family M_q used in the CLWL draft.

    Interpretation:
    - with probability q, the observed complementary label equals the true class;
    - with probability 1-q, another label is sampled uniformly at random.

    This is no longer a valid complementary label in the strict semantic sense when q > 0,
    but it is useful as a controlled weak-label family in the CLWL complementary analysis.
    """
    _validate_c(c)
    if not (0.0 <= q <= 1.0 / (c - 1)):
        raise WeakLabelGeneratorError(f"q must lie in [0, 1], got q={q}.")

    off_diag = (1.0 - q) / (c - 1)
    M = off_diag * np.ones((c, c), dtype=np.float64)
    np.fill_diagonal(M, q)

    weak_label_names = [f"obs_{i}" for i in range(c)]
    _ensure_column_stochastic(M)
    family = WeakLabelFamily(
        name=f"complementary_noisy_q_{q:.3f}",
        M=M,
        weak_label_matrix=None,
        weak_label_names=weak_label_names,
        metadata={"type": "complementary_noisy", "q": q},
    )
    print(
        f"[make_noisy_complementary_family] name={family.name}, shape={family.M.shape}, "
        f"rank={np.linalg.matrix_rank(family.M)}, q={q}"
    )
    return family


def make_random_general_weak_label_family(
    d: int,
    c: int,
    *,
    seed: int = 0,
    min_entry: float = 1e-3,
) -> WeakLabelFamily:
    """
    Construct a general random weak-label family.

    This generator is useful for CLWL-only experiments outside the native regimes of CLPL and CLCL.
    The output is a nonnegative column-stochastic matrix M with high probability of full column rank.
    """
    _validate_c(c)
    if d < c:
        raise WeakLabelGeneratorError(f"Need d >= c, got d={d}, c={c}.")
    if min_entry < 0:
        raise WeakLabelGeneratorError(f"min_entry must be nonnegative, got {min_entry}.")

    rng = np.random.default_rng(seed)
    raw = rng.random((d, c)) + min_entry
    M = _normalize_columns(raw)

    if np.linalg.matrix_rank(M) != c:
        M = M + 1e-4 * rng.standard_normal((d, c))
        M = np.clip(M, 1e-8, None)
        M = _normalize_columns(M)

    if np.linalg.matrix_rank(M) != c:
        raise WeakLabelGeneratorError("Failed to generate a full-column-rank general weak-label M.")

    weak_label_names = [f"w_{k}" for k in range(d)]
    _ensure_column_stochastic(M)
    family = WeakLabelFamily(
        name=f"general_random_d_{d}_c_{c}",
        M=M,
        weak_label_matrix=None,
        weak_label_names=weak_label_names,
        metadata={"type": "general_random", "seed": seed},
    )
    print(
        f"[make_random_general_weak_label_family] name={family.name}, shape={family.M.shape}, "
        f"rank={np.linalg.matrix_rank(family.M)}, seed={seed}"
    )
    return family


def sample_weak_labels_from_M(
    clean_labels: Array,
    M: Array,
    *,
    seed: int = 0,
) -> Array:
    """
    Sample weak-label indices z from a transition matrix M.

    Parameters
    ----------
    clean_labels:
        Integer array of shape (n,), each entry in {0, ..., c-1}.
    M:
        Transition matrix of shape (d, c).

    Returns
    -------
    weak_labels:
        Integer array of shape (n,), each entry in {0, ..., d-1}.
    """
    clean_labels = np.asarray(clean_labels, dtype=np.int64)
    M = np.asarray(M, dtype=np.float64)
    _ensure_column_stochastic(M)

    d, c = M.shape
    if clean_labels.ndim != 1:
        raise WeakLabelGeneratorError(
            f"clean_labels must be a 1D integer array, got shape {clean_labels.shape}."
        )
    if np.min(clean_labels) < 0 or np.max(clean_labels) >= c:
        raise WeakLabelGeneratorError(
            f"clean_labels entries must lie in [0, {c-1}]."
        )

    rng = np.random.default_rng(seed)
    out = np.empty_like(clean_labels)
    weak_indices = np.arange(d)
    for i, y in enumerate(clean_labels):
        out[i] = rng.choice(weak_indices, p=M[:, int(y)])
    print(
        f"[sample_weak_labels_from_M] n={clean_labels.shape[0]}, d={d}, c={c}, "
        f"seed={seed}, unique_clean={np.unique(clean_labels)}, unique_weak={np.unique(out)}"
    )
    return out


def family_summary(family: WeakLabelFamily) -> dict[str, object]:
    """Return compact diagnostics for logging and debugging."""
    M = family.M
    d, c = M.shape
    col_support_sizes = (M > 0).sum(axis=0).tolist()
    summary = {
        "name": family.name,
        "shape": tuple(M.shape),
        "rank": int(np.linalg.matrix_rank(M)),
        "min_entry": float(np.min(M)),
        "max_entry": float(np.max(M)),
        "column_support_sizes": col_support_sizes,
        "has_Z": family.weak_label_matrix is not None,
        "metadata": family.metadata or {},
    }
    print(f"[family_summary] {summary}")
    return summary


if __name__ == "__main__":
    print("=== Partial uniform family ===")
    fam1 = make_uniform_partial_label_family(c=4, candidate_size=2)
    print(family_summary(fam1))

    print("\n=== Partial size-mixture family ===")
    fam2 = make_size_mixture_partial_label_family(c=4, size_weights={2: 0.7, 3: 0.3})
    print(family_summary(fam2))

    print("\n=== Biased partial family ===")
    affinity = np.array(
        [
            [0.0, 4.0, 1.0, 1.0],
            [1.0, 0.0, 4.0, 1.0],
            [1.0, 1.0, 0.0, 4.0],
            [4.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    fam3 = make_biased_partial_label_family(c=4, candidate_size=2, distractor_affinity=affinity)
    print(family_summary(fam3))

    print("\n=== Confusion-aware size-4 partial family (MNIST style) ===")
    fam3d = make_manual_mnist_quad_partial_label_family(lambda3=32.0, lambda2=8.0, lambda1=2.0)
    print(family_summary(fam3d))

    print("\n=== Uniform complementary family ===")
    fam4 = make_uniform_complementary_family(c=4)
    print(family_summary(fam4))

    print("\n=== Non-uniform complementary family ===")
    W = np.array(
        [
            [0.0, 2.0, 1.0, 3.0],
            [3.0, 0.0, 1.0, 1.0],
            [1.0, 4.0, 0.0, 1.0],
            [2.0, 1.0, 5.0, 0.0],
        ],
        dtype=np.float64,
    )
    fam5 = make_nonuniform_complementary_family(c=4, wrong_label_weights=W)
    print(family_summary(fam5))

    print("\n=== General random weak-label family ===")
    fam6 = make_random_general_weak_label_family(d=6, c=4, seed=123)
    print(family_summary(fam6))