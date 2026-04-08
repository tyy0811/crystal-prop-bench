"""Feature engineering wrappers around matminer featurizers.

Tier 1: Magpie composition-only features (~150 descriptors).
Tier 2: Voronoi structural features (~200-300 descriptors).
Both cache results to parquet for subsequent runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

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
    feature_df.insert(0, "material_id", df["material_id"].values)  # type: ignore[arg-type]

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

    # Get feature labels from a single successful structure
    struct_labels: list[str] | None = None
    for _, row in df.iterrows():
        mid = row["material_id"]
        if mid in structures:
            try:
                labels = []
                for _, feat in struct_featurizers:
                    labels.extend(feat.feature_labels())
                struct_labels = labels
                break
            except Exception:
                continue

    def _featurize_one(mid: str, structure: object) -> list[float] | None:
        """Featurize a single structure. Returns features or None on failure."""
        try:
            all_features: list[float] = []
            for _, feat in struct_featurizers:
                all_features.extend(feat.featurize(structure))
            return all_features
        except Exception:
            return None

    # Build work items
    work_items = []
    missing_ids: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        mid = row["material_id"]
        if mid in structures:
            work_items.append((mid, structures[mid], row.get("chemistry_family", "unknown")))
        else:
            missing_ids.append((mid, row.get("chemistry_family", "unknown")))

    # Parallel featurization
    from joblib import Parallel, delayed

    n_jobs = min(8, len(work_items))
    logger.info("Running Voronoi featurization with %d parallel jobs...", n_jobs)

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_featurize_one)(mid, struct)
        for mid, struct, _ in work_items
    )

    # Collect results
    successful_ids = []
    struct_feature_rows = []
    failed_count = len(missing_ids)
    failed_by_family: dict[str, int] = {}

    for (mid, _, family), feat_result in zip(work_items, results, strict=True):
        if feat_result is not None:
            successful_ids.append(mid)
            struct_feature_rows.append(feat_result)
        else:
            failed_count += 1
            failed_by_family[family] = failed_by_family.get(family, 0) + 1

    for _mid, family in missing_ids:
        failed_by_family[family] = failed_by_family.get(family, 0) + 1

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
