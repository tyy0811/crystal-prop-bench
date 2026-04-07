"""Split strategies: standard, domain-shift, mixed-train, and OOD calibration sweep.

All split logic lives here — single source of truth for data partitioning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def standard_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
    cal_frac: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/10/10/10 split stratified by chemistry family.

    Returns (train, val, cal, test).
    val is used for early stopping; cal is held out strictly for conformal.
    """
    strat_col = df["chemistry_family"]
    remaining_frac = 1.0 - train_frac

    # First split: train vs. rest
    train, remaining = train_test_split(
        df,
        test_size=remaining_frac,
        random_state=seed,
        stratify=strat_col,
    )

    # Split rest into val / cal / test (each 1/3 of remaining 30%)
    remaining_strat = remaining["chemistry_family"]
    val_of_remaining = val_frac / remaining_frac
    val, rest2 = train_test_split(
        remaining,
        test_size=1.0 - val_of_remaining,
        random_state=seed,
        stratify=remaining_strat,
    )

    rest2_strat = rest2["chemistry_family"]
    cal_of_rest2 = cal_frac / (remaining_frac - val_frac)
    cal, test = train_test_split(
        rest2,
        test_size=1.0 - cal_of_rest2,
        random_state=seed,
        stratify=rest2_strat,
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        cal.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def domain_shift_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
    cal_frac: float = 0.10,
    stratify_col: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Domain-shift split: train/val/cal/test on oxides, OOD families as test sets.

    Returns dict with keys: train, val, cal, test_id,
    test_ood_sulfide, test_ood_nitride, test_ood_halide.

    If *stratify_col* is given (e.g. ``"formation_energy_per_atom"``),
    oxide partitions are stratified by target-value quartile so that
    each seed sees a similar distribution of easy/hard examples.
    """
    oxides = df[df["chemistry_family"] == "oxide"].copy()
    remaining_frac = 1.0 - train_frac

    _BIN = "_strat_bin"
    if stratify_col and stratify_col in oxides.columns:
        oxides[_BIN] = pd.qcut(
            oxides[stratify_col], q=4, labels=False, duplicates="drop",
        )

    def _strat(subset: pd.DataFrame) -> pd.Series | None:
        if _BIN not in subset.columns:
            return None
        if subset[_BIN].value_counts().min() < 2:
            return None
        return subset[_BIN]

    train, remaining = train_test_split(
        oxides,
        test_size=remaining_frac,
        random_state=seed,
        stratify=_strat(oxides),
    )

    val_of_remaining = val_frac / remaining_frac
    val, rest2 = train_test_split(
        remaining,
        test_size=1.0 - val_of_remaining,
        random_state=seed,
        stratify=_strat(remaining),
    )

    cal_of_rest2 = cal_frac / (remaining_frac - val_frac)
    cal, test_id = train_test_split(
        rest2,
        test_size=1.0 - cal_of_rest2,
        random_state=seed,
        stratify=_strat(rest2),
    )

    drops = [s for s in [train, val, cal, test_id] if _BIN in s.columns]
    for s in drops:
        s.drop(columns=[_BIN], inplace=True)

    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
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


def mixed_train_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
    cal_frac: float = 0.10,
) -> dict[str, pd.DataFrame]:
    """Mixed-train split: train on ALL families, test per family.

    Used to compare against domain_shift_split — does mixed training
    recover OOD performance (domain randomization effect)?

    Returns dict with keys: train, val, cal, test,
    test_oxide, test_sulfide, test_nitride, test_halide.
    """
    train, val, cal, test = standard_split(
        df, seed=seed, train_frac=train_frac,
        val_frac=val_frac, cal_frac=cal_frac,
    )

    result: dict[str, pd.DataFrame] = {
        "train": train,
        "val": val,
        "cal": cal,
        "test": test,
    }
    for family in ["oxide", "sulfide", "nitride", "halide"]:
        family_test = test[test["chemistry_family"] == family].reset_index(drop=True)
        if len(family_test) > 0:
            result[f"test_{family}"] = family_test

    return result


def ood_calibration_sweep(
    df_ood_family: pd.DataFrame,
    cal_sizes: list[int] | None = None,
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """For each cal_size, return (cal_subset, test_remainder).

    Uses a nested design: the 5-sample set is a subset of the 10-sample set,
    which is a subset of the 25-sample set, etc. This ensures the calibration
    efficiency curve measures the effect of adding more data, not resampling.
    """
    if cal_sizes is None:
        cal_sizes = [5, 10, 25, 50, 100]

    n = len(df_ood_family)
    for cal_size in cal_sizes:
        if cal_size >= n:
            raise ValueError(
                f"cal_size={cal_size} exceeds data size={n}"
            )

    # Single permutation — nested subsets
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    pairs: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for cal_size in cal_sizes:
        cal_idx = indices[:cal_size]
        test_idx = indices[cal_size:]

        cal = df_ood_family.iloc[cal_idx].reset_index(drop=True)
        test = df_ood_family.iloc[test_idx].reset_index(drop=True)
        pairs.append((cal, test))

    return pairs
