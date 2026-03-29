# bootstrap_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class BootstrapSample:
    """
    Container for one bootstrap resample.

    Attributes
    ----------
    sample_idx:
        Resampled indices of length n_samples.
    unique_fraction:
        Fraction of unique original observations appearing in the resample.
    method:
        Bootstrap method name.
    block_length:
        Effective / requested block length.
    seed:
        RNG seed used for this sample.
    """
    sample_idx: np.ndarray
    unique_fraction: float
    method: str
    block_length: int
    seed: Optional[int]


def _validate_n_samples(n_samples: int) -> None:
    if not isinstance(n_samples, int) or n_samples <= 1:
        raise ValueError(f"n_samples must be an integer > 1, got {n_samples}.")


def _validate_block_length(block_length: int, n_samples: int) -> None:
    if not isinstance(block_length, int) or block_length <= 0:
        raise ValueError(f"block_length must be a positive integer, got {block_length}.")
    if block_length > n_samples:
        raise ValueError(
            f"block_length cannot exceed n_samples. Got block_length={block_length}, "
            f"n_samples={n_samples}."
        )


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _trim_to_length(idx: List[int], n_samples: int) -> np.ndarray:
    arr = np.asarray(idx[:n_samples], dtype=int)
    if len(arr) != n_samples:
        raise RuntimeError(
            f"Bootstrap sampler produced incorrect length: expected {n_samples}, got {len(arr)}."
        )
    return arr


def contiguous_blocks_from_index(idx: np.ndarray) -> List[np.ndarray]:
    """
    Recover contiguous runs from a resampled index vector.

    Example
    -------
    [10, 11, 12, 50, 51, 9, 10] -> [[10,11,12], [50,51], [9,10]]
    """
    idx = np.asarray(idx, dtype=int)
    if idx.ndim != 1:
        raise ValueError("idx must be a 1D array.")

    if len(idx) == 0:
        return []

    blocks: List[np.ndarray] = []
    start = 0

    for i in range(1, len(idx)):
        if idx[i] != idx[i - 1] + 1:
            blocks.append(idx[start:i].copy())
            start = i

    blocks.append(idx[start:].copy())
    return blocks


def moving_block_bootstrap_indices(
    n_samples: int,
    block_length: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Moving Block Bootstrap (MBB).

    Samples fixed-length contiguous blocks with replacement until n_samples
    observations are obtained, then trims to exact length.

    Parameters
    ----------
    n_samples:
        Number of observations in the original series.
    block_length:
        Fixed block length.
    seed:
        Optional random seed.

    Returns
    -------
    np.ndarray
        Bootstrap resample indices of shape (n_samples,).
    """
    _validate_n_samples(n_samples)
    _validate_block_length(block_length, n_samples)

    rng = _rng(seed)
    max_start = n_samples - block_length
    starts = np.arange(max_start + 1, dtype=int)

    out: List[int] = []
    while len(out) < n_samples:
        start = int(rng.choice(starts))
        out.extend(range(start, start + block_length))

    return _trim_to_length(out, n_samples)


def circular_block_bootstrap_indices(
    n_samples: int,
    block_length: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Circular Block Bootstrap (CBB).

    Similar to moving block bootstrap, but blocks can wrap around the end
    of the series. This avoids under-sampling tail positions.

    Parameters
    ----------
    n_samples:
        Number of observations in the original series.
    block_length:
        Fixed block length.
    seed:
        Optional random seed.

    Returns
    -------
    np.ndarray
        Bootstrap resample indices of shape (n_samples,).
    """
    _validate_n_samples(n_samples)
    _validate_block_length(block_length, n_samples)

    rng = _rng(seed)

    out: List[int] = []
    while len(out) < n_samples:
        start = int(rng.integers(0, n_samples))
        block = [(start + j) % n_samples for j in range(block_length)]
        out.extend(block)

    return _trim_to_length(out, n_samples)


def stationary_bootstrap_indices(
    n_samples: int,
    expected_block_length: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Stationary Bootstrap (Politis & Romano style).

    At each step:
    - with probability p = 1 / expected_block_length, start a new block
    - otherwise continue from the next time index (with circular wrap-around)

    This gives random block lengths with geometric distribution and preserves
    short-range dependence better than iid bootstrap.

    Parameters
    ----------
    n_samples:
        Number of observations in the original series.
    expected_block_length:
        Mean block length.
    seed:
        Optional random seed.

    Returns
    -------
    np.ndarray
        Bootstrap resample indices of shape (n_samples,).
    """
    _validate_n_samples(n_samples)
    _validate_block_length(expected_block_length, n_samples)

    rng = _rng(seed)
    p = 1.0 / float(expected_block_length)

    out = np.empty(n_samples, dtype=int)

    current = int(rng.integers(0, n_samples))
    out[0] = current

    for t in range(1, n_samples):
        if rng.random() < p:
            current = int(rng.integers(0, n_samples))
        else:
            current = (current + 1) % n_samples
        out[t] = current

    return out


def bootstrap_indices(
    n_samples: int,
    method: str = "stationary",
    block_length: int = 20,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Unified entry point for bootstrap resampling.

    Parameters
    ----------
    n_samples:
        Number of original observations.
    method:
        One of {"stationary", "moving", "circular"}.
    block_length:
        Fixed or expected block length depending on method.
    seed:
        Optional random seed.

    Returns
    -------
    np.ndarray
        Resampled indices of length n_samples.
    """
    method = method.lower().strip()

    if method == "stationary":
        return stationary_bootstrap_indices(
            n_samples=n_samples,
            expected_block_length=block_length,
            seed=seed,
        )

    if method == "moving":
        return moving_block_bootstrap_indices(
            n_samples=n_samples,
            block_length=block_length,
            seed=seed,
        )

    if method == "circular":
        return circular_block_bootstrap_indices(
            n_samples=n_samples,
            block_length=block_length,
            seed=seed,
        )

    raise ValueError(
        f"Unknown bootstrap method: {method}. "
        f"Supported methods are: 'stationary', 'moving', 'circular'."
    )


def iter_bootstrap_samples(
    n_samples: int,
    n_bootstrap: int,
    method: str = "stationary",
    block_length: int = 20,
    base_seed: int = 42,
) -> Generator[BootstrapSample, None, None]:
    """
    Yield multiple bootstrap resamples with deterministic seed progression.

    Parameters
    ----------
    n_samples:
        Number of original observations.
    n_bootstrap:
        Number of bootstrap replications.
    method:
        Bootstrap method.
    block_length:
        Fixed / expected block length.
    base_seed:
        Base RNG seed. Each replication uses base_seed + b.

    Yields
    ------
    BootstrapSample
    """
    if not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be a positive integer, got {n_bootstrap}.")

    for b in range(n_bootstrap):
        seed = base_seed + b
        idx = bootstrap_indices(
            n_samples=n_samples,
            method=method,
            block_length=block_length,
            seed=seed,
        )
        unique_fraction = float(len(np.unique(idx)) / n_samples)
        yield BootstrapSample(
            sample_idx=idx,
            unique_fraction=unique_fraction,
            method=method,
            block_length=block_length,
            seed=seed,
        )


def bootstrap_metric_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    method: str = "stationary",
    block_length: int = 20,
    base_seed: int = 42,
) -> np.ndarray:
    """
    Compute a bootstrap distribution of a metric from fixed predictions.

    This is useful for fixed-model bootstrap CI. It does NOT retrain the model.

    Parameters
    ----------
    y_true:
        Ground-truth targets, shape (n_samples,).
    y_pred:
        Predictions, shape (n_samples,).
    metric_fn:
        Callable metric_fn(y_true_boot, y_pred_boot) -> float
    n_bootstrap:
        Number of bootstrap replications.
    method:
        Bootstrap method.
    block_length:
        Fixed / expected block length.
    base_seed:
        Base seed for reproducibility.

    Returns
    -------
    np.ndarray
        Metric values across bootstrap replications.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must both be 1D arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    values = []
    for sample in iter_bootstrap_samples(
        n_samples=len(y_true),
        n_bootstrap=n_bootstrap,
        method=method,
        block_length=block_length,
        base_seed=base_seed,
    ):
        idx = sample.sample_idx
        val = float(metric_fn(y_true[idx], y_pred[idx]))
        values.append(val)

    return np.asarray(values, dtype=float)


def percentile_confidence_interval(
    values: Sequence[float],
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Percentile confidence interval from a bootstrap distribution.

    Parameters
    ----------
    values:
        Bootstrap metric values.
    alpha:
        Significance level. alpha=0.05 -> 95% CI.

    Returns
    -------
    (lower, upper)
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("values must be a non-empty 1D array-like.")

    lower = float(np.quantile(arr, alpha / 2.0))
    upper = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return lower, upper


def summarize_bootstrap_distribution(
    values: Sequence[float],
    alpha: float = 0.05,
) -> dict:
    """
    Summary stats for a bootstrap distribution.

    Returns a dict containing mean, std, median, and percentile CI.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("values must be a non-empty 1D array-like.")

    ci_low, ci_high = percentile_confidence_interval(arr, alpha=alpha)

    return {
        "bootstrap_mean": float(np.mean(arr)),
        "bootstrap_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "bootstrap_median": float(np.median(arr)),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_n": int(len(arr)),
        "bootstrap_alpha": float(alpha),
    }


def suggest_block_length(
    n_samples: int,
    rule: str = "sqrt",
    min_block_length: int = 5,
    max_block_length: Optional[int] = None,
) -> int:
    """
    Simple heuristic block-length selector.

    This is intentionally conservative and lightweight. For a full paper,
    you can later replace it with a more formal selector.

    Supported rules
    ---------------
    - "sqrt": round(sqrt(n))
    - "cube_root": round(n ** (1/3))
    - "log2": round(log2(n))

    Returns
    -------
    int
        Suggested block length clipped into [min_block_length, max_block_length].
    """
    _validate_n_samples(n_samples)

    if rule == "sqrt":
        block = int(round(np.sqrt(n_samples)))
    elif rule == "cube_root":
        block = int(round(n_samples ** (1.0 / 3.0)))
    elif rule == "log2":
        block = int(round(np.log2(n_samples)))
    else:
        raise ValueError(
            f"Unknown rule: {rule}. Supported rules are 'sqrt', 'cube_root', 'log2'."
        )

    block = max(block, min_block_length)

    if max_block_length is None:
        max_block_length = max(min_block_length, n_samples // 4)

    block = min(block, max_block_length)
    block = max(block, 1)
    return int(block)


if __name__ == "__main__":
    # Minimal smoke test
    n = 100
    idx_mbb = moving_block_bootstrap_indices(n_samples=n, block_length=10, seed=42)
    idx_sbb = stationary_bootstrap_indices(n_samples=n, expected_block_length=10, seed=42)

    print("MBB length:", len(idx_mbb), "unique frac:", len(np.unique(idx_mbb)) / n)
    print("SBB length:", len(idx_sbb), "unique frac:", len(np.unique(idx_sbb)) / n)
    print("Suggested block length:", suggest_block_length(n))