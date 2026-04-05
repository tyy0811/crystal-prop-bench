"""Feature engineering wrappers around matminer featurizers.

Tier 1: Magpie composition-only features (~150 descriptors).
Tier 2: Voronoi structural features (~200-300 descriptors).
Both cache results to parquet for subsequent runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

logger = logging.getLogger(__name__)


def compute_magpie_features(
    df: pd.DataFrame,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Compute Magpie composition-only features.

    Parameters
    ----------
    df : DataFrame
        Must contain 'material_id' and 'formula_pretty' columns.
    cache_path : Path, optional
        If provided and file exists, load from cache. Otherwise compute and save.

    Returns
    -------
    DataFrame with 'material_id' + ~150 Magpie feature columns.
    """
    if cache_path and cache_path.exists():
        logger.info("Loading cached Magpie features from %s", cache_path)
        return pd.read_parquet(cache_path)

    featurizer = ElementProperty.from_preset("magpie")
    compositions = df["formula_pretty"].apply(Composition)

    logger.info("Computing Magpie features for %d compositions...", len(df))
    feature_labels = featurizer.feature_labels()
    feature_rows = []

    for i, comp in enumerate(compositions):
        features = featurizer.featurize(comp)
        feature_rows.append(features)
        if (i + 1) % 10000 == 0:
            logger.info("  Featurized %d/%d", i + 1, len(df))

    feature_df = pd.DataFrame(feature_rows, columns=feature_labels)
    feature_df.insert(0, "material_id", df["material_id"].values)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_parquet(cache_path, index=False)
        logger.info("Cached Magpie features to %s", cache_path)

    return feature_df


def compute_voronoi_features(
    df: pd.DataFrame,
    structures: dict,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Compute Voronoi structural features (Tier 2).

    Includes Magpie composition features + density, symmetry, and
    coordination number statistics. Structures that fail Voronoi
    tessellation are dropped and logged.

    Parameters
    ----------
    df : DataFrame
        Must contain 'material_id' and 'formula_pretty' columns.
    structures : dict
        Maps material_id -> pymatgen Structure.
    cache_path : Path, optional
        Cache location.

    Returns
    -------
    DataFrame with 'material_id' + all feature columns.
    Rows with failed featurization are dropped.
    """
    if cache_path and cache_path.exists():
        logger.info("Loading cached Voronoi features from %s", cache_path)
        return pd.read_parquet(cache_path)

    from matminer.featurizers.site import CoordinationNumber
    from matminer.featurizers.structure import (
        DensityFeatures,
        GlobalSymmetryFeatures,
        SiteStatsFingerprint,
    )

    # First compute Magpie features (always succeed)
    magpie_df = compute_magpie_features(df)

    # Structural featurizers
    density_feat = DensityFeatures()
    sym_feat = GlobalSymmetryFeatures()
    site_stats = SiteStatsFingerprint(
        CoordinationNumber.from_preset("VoronoiNN")
    )

    struct_featurizers = [
        ("density", density_feat),
        ("symmetry", sym_feat),
        ("site_stats", site_stats),
    ]

    logger.info("Computing Voronoi structural features for %d structures...", len(df))

    successful_ids = []
    struct_feature_rows = []
    struct_labels: list[str] | None = None
    failed_count = 0
    failed_by_family: dict[str, int] = {}

    for i, (_, row) in enumerate(df.iterrows()):
        mid = row["material_id"]
        if mid not in structures:
            failed_count += 1
            family = row.get("chemistry_family", "unknown")
            failed_by_family[family] = failed_by_family.get(family, 0) + 1
            continue

        structure = structures[mid]
        try:
            all_features = []
            all_labels = []
            for name, feat in struct_featurizers:
                features = feat.featurize(structure)
                all_features.extend(features)
                if struct_labels is None:
                    all_labels.extend(feat.feature_labels())

            if struct_labels is None:
                struct_labels = all_labels

            struct_feature_rows.append(all_features)
            successful_ids.append(mid)

        except Exception as e:
            failed_count += 1
            family = row.get("chemistry_family", "unknown")
            failed_by_family[family] = failed_by_family.get(family, 0) + 1
            logger.debug("Featurization failed for %s: %s", mid, e)

        if (i + 1) % 10000 == 0:
            logger.info("  Processed %d/%d (failed: %d)", i + 1, len(df), failed_count)

    logger.info(
        "Voronoi featurization: %d succeeded, %d failed (%.1f%%)",
        len(successful_ids),
        failed_count,
        100.0 * failed_count / len(df) if len(df) > 0 else 0.0,
    )
    for family, count in sorted(failed_by_family.items()):
        logger.info("  Failed in %s: %d", family, count)

    # Combine Magpie + structural features for successful rows
    struct_df = pd.DataFrame(struct_feature_rows, columns=struct_labels or [])
    struct_df.insert(0, "material_id", successful_ids)

    # Merge with Magpie features
    result = magpie_df[magpie_df["material_id"].isin(successful_ids)].merge(
        struct_df, on="material_id", how="inner"
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path, index=False)
        logger.info("Cached Voronoi features to %s", cache_path)

    return result
