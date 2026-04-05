# Phase A Design — crystal-prop-bench

Validated 2026-04-05. This document captures the design decisions made
during brainstorming. It governs Phase A implementation.

---

## 1. Data Layer

### DatasetAdapter ABC (`src/crystal_prop_bench/data/adapter.py`)

Template-method pattern. Two abstract methods, one concrete `load()`.

```python
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

class DatasetAdapter(ABC):
    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        """Fetch data from source. Return raw tabular DataFrame."""
        ...

    @abstractmethod
    def cache_path(self) -> Path:
        """Path for reading/writing cached data."""
        ...

    def load(self) -> pd.DataFrame:
        """Concrete: load_raw() -> classify chemistry -> validate schema -> filter -> cache."""
        # Calls load_raw(), applies classify_chemistry_family() from chemistry.py,
        # validates with Pandera schema, filters rows where family is None
        # (logs dropped count), caches to parquet.
        ...
```

Rationale: Portfolio consistency (matches finetune-bench, demandops-lite).
ABC is ~20 lines. No speculative methods — add when Bonus A (JARVIS) forces it.

### Chemistry classifier (`src/crystal_prop_bench/data/chemistry.py`)

Standalone function, not a method on the ABC. Importable by evaluation,
plotting, and analysis code without adapter dependency.

```python
def classify_chemistry_family(
    composition: Composition,
    purity_threshold: float = 0.80,
) -> str | None:
    """Classify crystal by dominant anion.

    Returns 'oxide', 'sulfide', 'nitride', 'halide', or None
    if no anion family exceeds the purity threshold.
    """
```

- `None` for below-threshold compounds makes filtering explicit at call sites
- 80% threshold documented in DECISIONS.md with dropped compound counts

### Splits (`src/crystal_prop_bench/data/splits.py`)

All split logic in one module. Three functions:

1. **Standard split** — 80/10/10 stratified by family + discretized target, 3 seeds
2. **Domain-shift split** — train/cal/test on oxides, full OOD families as test sets, mixed-train variant
3. **OOD calibration sweep**:

```python
def ood_calibration_sweep(
    df_ood_family: pd.DataFrame,
    cal_sizes: list[int] = [5, 10, 25, 50, 100],
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """For each cal_size, return (cal_subset, test_remainder)."""
```

Keeps calibration sweep under same seeded sampling as all other splits.

### Featurizers (`src/crystal_prop_bench/data/featurizers.py`)

- Tier 1 (Magpie): ~150 composition descriptors. Never fails.
- Tier 2 (Voronoi structural): ~200-300 features. Can fail — log, drop, report fraction per family.
- Both cache computed features to parquet.

**Featurization bias check:** Tier 1 metrics computed on both full dataset
and Tier-2-compatible subset. Reported as standard rows in results CSV
(`tier1_full` vs `tier1_voronoi_subset`).

---

## 2. Evaluation Pipeline

### Metrics (`src/crystal_prop_bench/evaluation/metrics.py`)

- Per-run: MAE, RMSE, R² overall and per chemistry family
- Seed aggregation:

```python
def aggregate_seeds(seed_results: list[dict]) -> dict:
    """Aggregate per-seed metric dicts into mean +/- std."""
```

- Featurization bias check computed as standard part of evaluation

### Conformal (`src/crystal_prop_bench/evaluation/conformal.py`)

Three evaluation modes:

1. **Standard** — calibrate on oxide cal set, test on oxide test (ID baseline)
2. **Cross-domain** — calibrate on oxide cal set, test on each OOD family (Finding 4: coverage breaks)
3. **Calibration sweep** — calibrate on [5, 10, 25, 50, 100] samples from each OOD family, plot coverage vs. calibration budget (the standout deployable finding)

Reports empirical coverage and mean interval width at alpha = 0.10, 0.20, 0.30.

### Domain shift (`src/crystal_prop_bench/evaluation/domain_shift.py`)

- Degradation ratios: MAE_OOD / MAE_ID per family
- Mixed-train comparison: does training on all families recover OOD performance?

### Explainability (`src/crystal_prop_bench/evaluation/explainability.py`)

- SHAP TreeExplainer on Tier 1 and Tier 2 LightGBM
- Global feature importance (beeswarm)
- Per-family comparison
- Failure-case extraction: 50 worst per tier, characterized by family/space group/n_atoms, cross-tier overlap

---

## 3. Models and Scripts

### Model (`src/crystal_prop_bench/models/lgbm_baseline.py`)

Single `train_lgbm()` function. Takes feature matrix, targets, config.
Returns trained model + calibration residuals. Logs to MLflow.

### Prediction interchange format

Training scripts save predictions to `results/predictions/`:

```
results/predictions/tier1_standard_seed0_ef.parquet
```

Columns: `[material_id, y_true, y_pred, chemistry_family, split]`

This is the seam between training and evaluation. Evaluation scripts
read these files — no model reloading needed.

### Scripts

```
scripts/
├── download_data.py     # MPAdapter.load(), caches to data/
├── run_tier1.py         # Magpie featurization + LightGBM + save predictions
├── run_tier2.py         # Voronoi featurization + LightGBM + save predictions
├── run_evaluation.py    # reads predictions, writes results CSVs
├── run_shap.py          # reads models + predictions, writes CSVs + SHAP data
└── run_plots.py         # reads CSVs, writes PNGs (independent re-rendering)
```

### Makefile

```makefile
download-data:
    python scripts/download_data.py
run-tier1:
    python scripts/run_tier1.py
run-tier2:
    python scripts/run_tier2.py
run-evaluation:
    python scripts/run_evaluation.py
run-shap:
    python scripts/run_shap.py
run-plots:
    python scripts/run_plots.py
run-all: download-data run-tier1 run-tier2 run-evaluation run-shap run-plots
```

### MLflow

Lightweight: log params + metrics to local `mlruns/` during development.
Final results also written to `results/tables/*.csv` and
`results/figures/*.png` as flat files. Flat files are the public interface;
MLflow is the development tool.

---

## 4. Output Structure

```
results/
├── tables/
│   ├── benchmark.csv           # Tier x Split x Target x MAE +/- std
│   ├── domain_shift.csv        # per-family degradation ratios
│   ├── conformal_coverage.csv  # coverage x width at each alpha
│   ├── calibration_sweep.csv   # coverage vs. cal budget per OOD family
│   └── bias_check.csv          # tier1_full vs. tier1_voronoi_subset
├── predictions/
│   └── tier{1,2}_{split}_{seed}_{target}.parquet
├── figures/
│   ├── domain_shift_bars.png
│   ├── conformal_coverage.png
│   ├── calibration_sweep.png   # the standout figure
│   └── shap_summary.png
└── models/                     # saved LightGBM for SHAP, gitignored
```

---

## 5. Testing and CI

### Test fixture

100 crystals (25 per family) in `tests/fixtures/`:
- `fixture_crystals.parquet` — tabular fields
- `fixture_structures.pkl` — pymatgen Structure objects
- `fixture_magpie_features.parquet` — pre-computed Tier 1 features
- `fixture_voronoi_features.parquet` — pre-computed Tier 2 features

Pin exact Voronoi survival count. Assert in `test_featurizers.py`:
`assert len(voronoi_features) == EXPECTED_SURVIVING_COUNT`

### Test modules

```
tests/
├── conftest.py              # loads fixtures, shared pytest fixtures
├── test_adapters.py         # mock MP API, test response -> DataFrame transform
├── test_chemistry.py        # classify edge cases (pure, mixed, halide combo, empty)
├── test_schemas.py          # Pandera catches invalid/missing/out-of-range
├── test_splits.py           # determinism, stratification, ood_calibration_sweep sizes
├── test_featurizers.py      # Magpie shape, Voronoi graceful failure, survival count
├── test_conformal.py        # synthetic residuals -> known coverage
├── test_metrics.py          # aggregate_seeds, per-family breakdown
└── test_integration.py      # full pipeline on 100-crystal fixture
```

### API testing

MP API calls mocked in CI. `@pytest.mark.network` marker for optional
live validation runs. Adapter test confirms API response -> DataFrame
transformation logic against canned response.

### CI pipeline

```yaml
jobs:
  lint:           # parallel
    - ruff check .
    - mypy src/
  test:           # parallel with lint
    - pytest tests/ -x --ignore=tests/test_integration.py
  integration:    # after test
    - pytest tests/test_integration.py -x
  regression-gate:  # after integration
    - python scripts/check_regression.py
```

Regression gate: train Tier 1 on fixture, check MAE <= generous threshold
(~2x initial MAE). Catches broken pipelines, not hyperparameter drift.

---

## 6. Key Design Decisions (to document in DECISIONS.md)

1. Why Materials Project only (not JARVIS in v1)
2. Why formation energy + band gap
3. Chemistry classification: 80% anion-purity, standalone function
4. Why Magpie descriptors
5. Why Voronoi over fixed-radius neighbor lists
6. Why split conformal regression (not APS, not CQR in v1)
7. Split strategy: stratified random with frozen fixture
8. Featurization failure handling: drop, report, bias-check via Tier 1 comparison
9. Why 3 seeds
10. Why LightGBM over XGBoost or random forest
11. DatasetAdapter ABC with 2 abstract methods (portfolio pattern)
12. MLflow as development tool, flat files as public interface
13. Prediction parquet as interchange format between training and evaluation
14. Calibration sweep for deployable UQ finding

---

## 7. Findings to Probe

1. Composition baseline strength (MAE < 0.15 eV/atom on formation energy)
2. Structure helps band gap more than formation energy
3. Domain-shift degradation pattern (composition vs. structure-aware)
4. Conditional coverage breaks under shift
5. Mixed training as domain randomization
6. **Calibration efficiency curve** — how many OOD samples needed for reliable UQ
