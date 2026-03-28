# splitter.py
from dataclasses import dataclass
from typing import Generator, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FoldIndices:
    train_idx: np.ndarray
    valid_idx: np.ndarray


def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def train_dev_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reserve the most recent block as untouched test set.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    df = sort_by_date(df)
    split_idx = int(len(df) * (1 - test_size))

    dev_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    return dev_df, test_df


def apply_purge_and_embargo(
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    purge: int = 0,
    embargo: int = 0,
) -> np.ndarray:
    """
    Remove training observations too close to validation block.
    Useful when labels use future windows.
    """
    if len(valid_idx) == 0:
        return train_idx

    valid_start = valid_idx[0]
    valid_end = valid_idx[-1]

    forbidden_left = valid_start - purge
    forbidden_right = valid_end + embargo

    filtered = train_idx[
        (train_idx < forbidden_left) | (train_idx > forbidden_right)
    ]
    return filtered


def rolling_window_splits(
    n_samples: int,
    train_size: int,
    valid_size: int,
    step_size: int,
    purge: int = 0,
    embargo: int = 0,
) -> Generator[FoldIndices, None, None]:
    """
    Fixed-length train window + next validation window.
    """
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_size
        valid_start = train_end
        valid_end = valid_start + valid_size

        if valid_end > n_samples:
            break

        train_idx = np.arange(train_start, train_end)
        valid_idx = np.arange(valid_start, valid_end)
        train_idx = apply_purge_and_embargo(train_idx, valid_idx, purge, embargo)

        if len(train_idx) > 0:
            yield FoldIndices(train_idx=train_idx, valid_idx=valid_idx)

        start += step_size


def expanding_window_splits(
    n_samples: int,
    min_train_size: int,
    valid_size: int,
    step_size: int,
    purge: int = 0,
    embargo: int = 0,
) -> Generator[FoldIndices, None, None]:
    """
    Expanding train window + next validation window.
    """
    train_end = min_train_size

    while True:
        valid_start = train_end
        valid_end = valid_start + valid_size

        if valid_end > n_samples:
            break

        train_idx = np.arange(0, train_end)
        valid_idx = np.arange(valid_start, valid_end)
        train_idx = apply_purge_and_embargo(train_idx, valid_idx, purge, embargo)

        if len(train_idx) > 0:
            yield FoldIndices(train_idx=train_idx, valid_idx=valid_idx)

        train_end += step_size


def describe_splits(folds: List[FoldIndices]) -> None:
    for i, fold in enumerate(folds, start=1):
        print(
            f"Fold {i}: "
            f"train [{fold.train_idx[0]}..{fold.train_idx[-1]}] "
            f"({len(fold.train_idx)} obs), "
            f"valid [{fold.valid_idx[0]}..{fold.valid_idx[-1]}] "
            f"({len(fold.valid_idx)} obs)"
        )