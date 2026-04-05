"""Split strategies: standard, domain-shift, and OOD calibration sweep.

All split logic lives here — single source of truth for data partitioning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def standard_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.80,
    cal_frac: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """80/10/10 split stratified by chemistry family.

    Returns (train, cal, test).
    """
    # Stratify by chemistry_family
    strat_col = df["chemistry_family"]

    # First split: train vs. (cal + test)
    remaining_frac = cal_frac + (1.0 - train_frac - cal_frac)
    train, remaining = train_test_split(
        df,
        test_size=remaining_frac,
        random_state=seed,
        stratify=strat_col,
    )

    # Second split: cal vs. test (50/50 of the remaining 20%)
    cal_of_remaining = cal_frac / remaining_frac
    remaining_strat = remaining["chemistry_family"]
    cal, test = train_test_split(
        remaining,
        test_size=1.0 - cal_of_remaining,
        random_state=seed,
        stratify=remaining_strat,
    )

    return (
        train.reset_index(drop=True),
        cal.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def domain_shift_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.80,
    cal_frac: float = 0.10,
) -> dict[str, pd.DataFrame]:
    """Domain-shift split: train/cal/test on oxides, OOD families as test sets.

    Returns dict with keys: train, cal, test_id,
    test_ood_sulfide, test_ood_nitride, test_ood_halide.
    """
    oxides = df[df["chemistry_family"] == "oxide"].copy()
    remaining_frac = 1.0 - train_frac

    train, remaining = train_test_split(
        oxides,
        test_size=remaining_frac,
        random_state=seed,
    )

    cal_of_remaining = cal_frac / remaining_frac
    cal, test_id = train_test_split(
        remaining,
        test_size=1.0 - cal_of_remaining,
        random_state=seed,
    )

    return {
        "train": train.reset_index(drop=True),
        "cal": cal.reset_index(drop=True),
        "test_id": test_id.reset_index(drop=True),
        "test_ood_sulfide": df[df["chemistry_family"] == "sulfide"]
        .reset_index(drop=True)
        .copy(),
        "test_ood_nitride": df[df["chemistry_family"] == "nitride"]
        .reset_index(drop=True)
        .copy(),
        "test_ood_halide": df[df["chemistry_family"] == "halide"]
        .reset_index(drop=True)
        .copy(),
    }


def ood_calibration_sweep(
    df_ood_family: pd.DataFrame,
    cal_sizes: list[int] | None = None,
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """For each cal_size, return (cal_subset, test_remainder).

    Used for the calibration efficiency curve: how many OOD samples
    are needed for reliable conformal intervals?
    """
    if cal_sizes is None:
        cal_sizes = [5, 10, 25, 50, 100]

    n = len(df_ood_family)
    pairs: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    rng = np.random.RandomState(seed)

    for cal_size in cal_sizes:
        if cal_size >= n:
            raise ValueError(
                f"cal_size={cal_size} exceeds data size={n}"
            )

        indices = rng.permutation(n)
        cal_idx = indices[:cal_size]
        test_idx = indices[cal_size:]

        cal = df_ood_family.iloc[cal_idx].reset_index(drop=True)
        test = df_ood_family.iloc[test_idx].reset_index(drop=True)
        pairs.append((cal, test))

    return pairs
