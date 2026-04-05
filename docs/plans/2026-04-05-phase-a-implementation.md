# Phase A Implementation Plan — crystal-prop-bench

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship a materials property prediction benchmark with calibrated uncertainty and chemical domain-shift evaluation — producing one benchmark table, one domain-shift figure, one calibration-sweep figure, and a polished README.

**Architecture:** Template-method DatasetAdapter ABC with 2 abstract methods. Standalone chemistry classifier. Prediction parquets as interchange between training and evaluation scripts. Split conformal regression with 3 evaluation modes. Flat CSV/PNG as public interface, MLflow as development tool.

**Tech Stack:** pymatgen, matminer, mp-api, LightGBM, scikit-learn, SHAP, Pandera, MLflow, pandas, matplotlib, pytest, ruff, mypy

**Design document:** `docs/plans/2026-04-05-phase-a-design.md`

---

## Block 1: Repo Setup & Data Foundation

### Task 1: Initialize Repository

**Files:**
- Create: `pyproject.toml`
- Create: `Makefile`
- Create: `src/crystal_prop_bench/__init__.py`
- Create: `src/crystal_prop_bench/data/__init__.py`
- Create: `src/crystal_prop_bench/models/__init__.py`
- Create: `src/crystal_prop_bench/evaluation/__init__.py`
- Create: `src/crystal_prop_bench/visualization/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`
- Create: `configs/base.yaml`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "crystal-prop-bench"
version = "0.1.0"
description = "Materials property prediction with calibrated uncertainty and chemical domain-shift evaluation"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
dependencies = [
    "pymatgen>=2024.2.0",
    "matminer>=0.9.0",
    "mp-api>=0.39.0",
    "lightgbm>=4.3.0",
    "scikit-learn>=1.4.0",
    "shap>=0.45.0",
    "pandas>=2.2.0",
    "pandera>=0.18.0",
    "mlflow>=2.11.0",
    "matplotlib>=3.8.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
    "pandas-stubs>=2.2.0",
]

[tool.ruff]
target-version = "py311"
line-length = 99

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[[tool.mypy.overrides]]
module = [
    "matminer.*",
    "mp_api.*",
    "shap.*",
    "lightgbm.*",
    "mlflow.*",
    "pymatgen.*",
]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "network: tests that require network access (deselect with '-m not network')",
]
```

**Step 2: Create Makefile**

```makefile
.PHONY: download-data run-tier1 run-tier2 run-evaluation run-shap run-plots run-all lint test

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

lint:
	ruff check .
	mypy src/

test:
	pytest tests/ -x --ignore=tests/test_integration.py

test-all:
	pytest tests/ -x

test-integration:
	pytest tests/test_integration.py -x
```

**Step 3: Create .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.venv/
mlruns/
data/*.parquet
data/*.pkl
results/models/
results/predictions/
.env
*.DS_Store
```

**Step 4: Create configs/base.yaml**

```yaml
data:
  cache_dir: data
  purity_threshold: 0.80

model:
  n_estimators: 1000
  learning_rate: 0.05
  num_leaves: 127
  min_child_samples: 20
  subsample: 0.8
  colsample_bytree: 0.8
  early_stopping_rounds: 50

evaluation:
  seeds: [42, 123, 456]
  targets: [formation_energy_per_atom, band_gap]
  alphas: [0.10, 0.20, 0.30]
  cal_sizes: [5, 10, 25, 50, 100]

split:
  train_frac: 0.8
  cal_frac: 0.1
  test_frac: 0.1
```

**Step 5: Create directory structure and init files**

```bash
mkdir -p src/crystal_prop_bench/data
mkdir -p src/crystal_prop_bench/models
mkdir -p src/crystal_prop_bench/evaluation
mkdir -p src/crystal_prop_bench/visualization
mkdir -p tests/fixtures
mkdir -p scripts
mkdir -p configs
mkdir -p data
mkdir -p results/tables results/figures results/predictions results/models
```

All `__init__.py` files are empty. `tests/conftest.py` starts as:

```python
from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR
```

**Step 6: Init git and commit**

```bash
git init
git add -A
git commit -m "chore: init repo structure with pyproject.toml, Makefile, configs"
```

---

### Task 2: Chemistry Classifier (TDD)

**Files:**
- Create: `src/crystal_prop_bench/data/chemistry.py`
- Create: `tests/test_chemistry.py`

**Step 1: Write failing tests**

```python
# tests/test_chemistry.py
from pymatgen.core import Composition

from crystal_prop_bench.data.chemistry import classify_chemistry_family


class TestClassifyChemistryFamily:
    """Test dominant-anion chemistry classification."""

    def test_pure_oxide(self) -> None:
        result = classify_chemistry_family(Composition("Fe2O3"))
        assert result == "oxide"

    def test_pure_sulfide(self) -> None:
        result = classify_chemistry_family(Composition("ZnS"))
        assert result == "sulfide"

    def test_pure_nitride(self) -> None:
        result = classify_chemistry_family(Composition("GaN"))
        assert result == "nitride"

    def test_pure_fluoride(self) -> None:
        result = classify_chemistry_family(Composition("CaF2"))
        assert result == "halide"

    def test_pure_chloride(self) -> None:
        result = classify_chemistry_family(Composition("NaCl"))
        assert result == "halide"

    def test_mixed_halide_combined_above_threshold(self) -> None:
        """NaF0.5Cl0.5 — F and Cl are both halide, 100% of anions."""
        result = classify_chemistry_family(Composition("NaF0.5Cl0.5"))
        assert result == "halide"

    def test_oxide_above_threshold(self) -> None:
        """Composition with O at 85% of anion sites."""
        # Ba2O8.5S1.5 -> O is 8.5/10 = 85% of anions
        result = classify_chemistry_family(Composition("Ba2O8.5S1.5"))
        assert result == "oxide"

    def test_mixed_below_threshold_returns_none(self) -> None:
        """O at 50% of anion sites — below 80% threshold."""
        result = classify_chemistry_family(Composition("LaON"))
        assert result is None

    def test_pure_metal_returns_none(self) -> None:
        """No anions at all."""
        result = classify_chemistry_family(Composition("Fe"))
        assert result is None

    def test_no_recognized_anion_returns_none(self) -> None:
        """Elements not in any anion family (e.g., Si, C)."""
        result = classify_chemistry_family(Composition("SiC"))
        assert result is None

    def test_custom_threshold(self) -> None:
        """50/50 oxide/sulfide passes at 0.5 threshold."""
        result = classify_chemistry_family(
            Composition("LaOS"), purity_threshold=0.50
        )
        assert result is not None

    def test_exactly_at_threshold(self) -> None:
        """Exactly 80% should pass (>=, not >)."""
        # O4S1 -> O is 4/5 = 0.80 of anions
        result = classify_chemistry_family(Composition("LaO4S1"))
        assert result == "oxide"

    def test_just_below_threshold(self) -> None:
        """79% should fail."""
        # O79S21 -> O is 79/100 = 0.79 of anions
        result = classify_chemistry_family(Composition("LaO79S21"))
        assert result is None
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_chemistry.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'crystal_prop_bench.data.chemistry'`

**Step 3: Implement chemistry classifier**

```python
# src/crystal_prop_bench/data/chemistry.py
"""Chemistry-family classification by dominant anion.

Standalone function — importable by evaluation, plotting, and analysis
code without adapter dependency.
"""

from pymatgen.core import Composition, Element

ANION_FAMILIES: dict[str, frozenset[Element]] = {
    "oxide": frozenset({Element("O")}),
    "sulfide": frozenset({Element("S")}),
    "nitride": frozenset({Element("N")}),
    "halide": frozenset({Element("F"), Element("Cl"), Element("Br"), Element("I")}),
}

# Reverse lookup: element -> family name
_ELEMENT_TO_FAMILY: dict[Element, str] = {}
for family_name, elements in ANION_FAMILIES.items():
    for el in elements:
        _ELEMENT_TO_FAMILY[el] = family_name


def classify_chemistry_family(
    composition: Composition,
    purity_threshold: float = 0.80,
) -> str | None:
    """Classify crystal by dominant anion.

    Returns 'oxide', 'sulfide', 'nitride', 'halide', or None
    if no anion family exceeds the purity threshold.

    Parameters
    ----------
    composition : Composition
        Pymatgen Composition object.
    purity_threshold : float
        Minimum fraction of anion sites that must belong to one family.
        Default 0.80 per DECISIONS.md.

    Returns
    -------
    str or None
        Family name, or None if below threshold or no recognized anions.
    """
    el_amounts = composition.get_el_amt_dict()

    # Accumulate anion amounts per family
    family_amounts: dict[str, float] = {}
    total_anion_amount = 0.0

    for el, amt in el_amounts.items():
        el_obj = Element(el) if isinstance(el, str) else el
        if el_obj in _ELEMENT_TO_FAMILY:
            family = _ELEMENT_TO_FAMILY[el_obj]
            family_amounts[family] = family_amounts.get(family, 0.0) + amt
            total_anion_amount += amt

    if total_anion_amount == 0.0:
        return None

    # Find dominant family
    for family, amt in sorted(
        family_amounts.items(), key=lambda x: x[1], reverse=True
    ):
        fraction = amt / total_anion_amount
        if fraction >= purity_threshold:
            return family

    return None
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_chemistry.py -v
```

Expected: All 13 tests PASS.

**Step 5: Commit**

```bash
git add src/crystal_prop_bench/data/chemistry.py tests/test_chemistry.py
git commit -m "feat: add chemistry-family classifier with 80% anion-purity threshold"
```

---

### Task 3: DatasetAdapter ABC

**Files:**
- Create: `src/crystal_prop_bench/data/adapter.py`

**Step 1: Implement the ABC**

```python
# src/crystal_prop_bench/data/adapter.py
"""DatasetAdapter ABC — template-method pattern.

Two abstract methods (load_raw, cache_path). Concrete load() handles
chemistry classification, schema validation, filtering, and caching.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from crystal_prop_bench.data.chemistry import classify_chemistry_family

logger = logging.getLogger(__name__)


class DatasetAdapter(ABC):
    """Base adapter for materials property datasets.

    Subclasses implement load_raw() and cache_path().
    The concrete load() method handles the shared pipeline:
    load_raw -> classify chemistry -> validate -> filter -> cache.
    """

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        """Fetch data from source. Return raw tabular DataFrame.

        Expected columns: material_id, formula_pretty,
        formation_energy_per_atom, band_gap, nsites, spacegroup_number.
        Structure objects stored separately.
        """
        ...

    @abstractmethod
    def cache_path(self) -> Path:
        """Directory for reading/writing cached data."""
        ...

    def load(self, force_refresh: bool = False) -> pd.DataFrame:
        """Load dataset with chemistry classification and caching.

        Returns DataFrame with 'chemistry_family' column added.
        Rows with chemistry_family=None are filtered out.
        """
        cache_dir = self.cache_path()
        parquet_path = cache_dir / "crystals.parquet"

        if parquet_path.exists() and not force_refresh:
            logger.info("Loading cached data from %s", parquet_path)
            return pd.read_parquet(parquet_path)

        logger.info("Fetching raw data...")
        df = self.load_raw()

        # Classify chemistry family
        from pymatgen.core import Composition

        df["chemistry_family"] = df["formula_pretty"].apply(
            lambda f: classify_chemistry_family(Composition(f))
        )

        # Log drop statistics
        total = len(df)
        dropped = df["chemistry_family"].isna().sum()
        logger.info(
            "Chemistry classification: %d/%d dropped (%.1f%%) — below purity threshold",
            dropped,
            total,
            100.0 * dropped / total if total > 0 else 0.0,
        )
        for family in ["oxide", "sulfide", "nitride", "halide"]:
            count = (df["chemistry_family"] == family).sum()
            logger.info("  %s: %d crystals", family, count)

        # Filter out unclassified compounds
        df = df[df["chemistry_family"].notna()].reset_index(drop=True)

        # Validate schema (imported here to avoid circular deps)
        from crystal_prop_bench.data.schemas import validate_crystal_df

        df = validate_crystal_df(df)

        # Cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        logger.info("Cached %d crystals to %s", len(df), parquet_path)

        return df
```

**Step 2: Commit**

```bash
git add src/crystal_prop_bench/data/adapter.py
git commit -m "feat: add DatasetAdapter ABC with template-method load()"
```

---

### Task 4: Pandera Schemas (TDD)

**Files:**
- Create: `src/crystal_prop_bench/data/schemas.py`
- Create: `tests/test_schemas.py`

**Step 1: Write failing tests**

```python
# tests/test_schemas.py
import pandas as pd
import pandera as pa
import pytest

from crystal_prop_bench.data.schemas import CrystalSchema, validate_crystal_df


class TestCrystalSchema:
    """Test Pandera schema validation."""

    def test_valid_dataframe_passes(self) -> None:
        df = pd.DataFrame({
            "material_id": ["mp-1", "mp-2"],
            "formula_pretty": ["Fe2O3", "ZnS"],
            "formation_energy_per_atom": [-1.5, -0.8],
            "band_gap": [2.0, 3.5],
            "nsites": [10, 4],
            "spacegroup_number": [167, 216],
            "chemistry_family": ["oxide", "sulfide"],
        })
        result = validate_crystal_df(df)
        assert len(result) == 2

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({
            "material_id": ["mp-1"],
            "formula_pretty": ["Fe2O3"],
            # missing formation_energy_per_atom
        })
        with pytest.raises(pa.errors.SchemaError):
            validate_crystal_df(df)

    def test_negative_band_gap_raises(self) -> None:
        df = pd.DataFrame({
            "material_id": ["mp-1"],
            "formula_pretty": ["Fe2O3"],
            "formation_energy_per_atom": [-1.5],
            "band_gap": [-0.5],  # invalid
            "nsites": [10],
            "spacegroup_number": [167],
            "chemistry_family": ["oxide"],
        })
        with pytest.raises(pa.errors.SchemaError):
            validate_crystal_df(df)

    def test_spacegroup_out_of_range_raises(self) -> None:
        df = pd.DataFrame({
            "material_id": ["mp-1"],
            "formula_pretty": ["Fe2O3"],
            "formation_energy_per_atom": [-1.5],
            "band_gap": [2.0],
            "nsites": [10],
            "spacegroup_number": [300],  # invalid: max is 230
            "chemistry_family": ["oxide"],
        })
        with pytest.raises(pa.errors.SchemaError):
            validate_crystal_df(df)

    def test_zero_nsites_raises(self) -> None:
        df = pd.DataFrame({
            "material_id": ["mp-1"],
            "formula_pretty": ["Fe2O3"],
            "formation_energy_per_atom": [-1.5],
            "band_gap": [2.0],
            "nsites": [0],  # invalid
            "spacegroup_number": [167],
            "chemistry_family": ["oxide"],
        })
        with pytest.raises(pa.errors.SchemaError):
            validate_crystal_df(df)

    def test_invalid_chemistry_family_raises(self) -> None:
        df = pd.DataFrame({
            "material_id": ["mp-1"],
            "formula_pretty": ["Fe2O3"],
            "formation_energy_per_atom": [-1.5],
            "band_gap": [2.0],
            "nsites": [10],
            "spacegroup_number": [167],
            "chemistry_family": ["carbonate"],  # not a valid family
        })
        with pytest.raises(pa.errors.SchemaError):
            validate_crystal_df(df)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_schemas.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement schemas**

```python
# src/crystal_prop_bench/data/schemas.py
"""Pandera schemas for crystal property data validation."""

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema

VALID_FAMILIES = {"oxide", "sulfide", "nitride", "halide"}

CrystalSchema = DataFrameSchema(
    {
        "material_id": Column(str, nullable=False),
        "formula_pretty": Column(str, nullable=False),
        "formation_energy_per_atom": Column(float, nullable=False),
        "band_gap": Column(float, pa.Check.ge(0), nullable=False),
        "nsites": Column(int, pa.Check.gt(0), nullable=False),
        "spacegroup_number": Column(
            int,
            [pa.Check.ge(1), pa.Check.le(230)],
            nullable=False,
        ),
        "chemistry_family": Column(
            str,
            pa.Check.isin(VALID_FAMILIES),
            nullable=False,
        ),
    },
    coerce=True,
)


def validate_crystal_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate DataFrame against CrystalSchema. Returns validated copy."""
    return CrystalSchema.validate(df)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_schemas.py -v
```

Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add src/crystal_prop_bench/data/schemas.py tests/test_schemas.py
git commit -m "feat: add Pandera crystal schema with validation"
```

---

### Task 5: MPAdapter (TDD with mocks)

**Files:**
- Create: `src/crystal_prop_bench/data/mp_adapter.py`
- Create: `tests/test_adapters.py`

**Step 1: Write failing tests**

```python
# tests/test_adapters.py
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from crystal_prop_bench.data.mp_adapter import MPAdapter


class TestMPAdapter:
    """Test Materials Project adapter with mocked API."""

    def _make_mock_doc(
        self,
        material_id: str,
        formula: str,
        ef: float,
        bg: float,
        nsites: int,
        sg: int,
    ) -> MagicMock:
        doc = MagicMock()
        doc.material_id = material_id
        doc.formula_pretty = formula
        doc.formation_energy_per_atom = ef
        doc.band_gap = bg
        doc.nsites = nsites
        doc.symmetry = MagicMock()
        doc.symmetry.number = sg
        doc.structure = MagicMock()
        return doc

    @patch("crystal_prop_bench.data.mp_adapter.MPRester")
    def test_load_raw_returns_dataframe(self, mock_rester_cls: MagicMock) -> None:
        mock_ctx = MagicMock()
        mock_rester_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_rester_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_ctx.materials.summary.search.return_value = [
            self._make_mock_doc("mp-1", "Fe2O3", -1.5, 2.0, 10, 167),
            self._make_mock_doc("mp-2", "ZnS", -0.8, 3.5, 4, 216),
        ]

        adapter = MPAdapter(api_key="fake_key", cache_dir=Path("/tmp/test_mp"))
        df = adapter.load_raw()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "material_id" in df.columns
        assert "formula_pretty" in df.columns
        assert "formation_energy_per_atom" in df.columns
        assert "band_gap" in df.columns
        assert "nsites" in df.columns
        assert "spacegroup_number" in df.columns

    @patch("crystal_prop_bench.data.mp_adapter.MPRester")
    def test_load_raw_correct_values(self, mock_rester_cls: MagicMock) -> None:
        mock_ctx = MagicMock()
        mock_rester_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_rester_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_ctx.materials.summary.search.return_value = [
            self._make_mock_doc("mp-1", "Fe2O3", -1.5, 2.0, 10, 167),
        ]

        adapter = MPAdapter(api_key="fake_key", cache_dir=Path("/tmp/test_mp"))
        df = adapter.load_raw()

        assert df.iloc[0]["material_id"] == "mp-1"
        assert df.iloc[0]["formula_pretty"] == "Fe2O3"
        assert df.iloc[0]["formation_energy_per_atom"] == pytest.approx(-1.5)
        assert df.iloc[0]["band_gap"] == pytest.approx(2.0)

    def test_cache_path_returns_configured_dir(self) -> None:
        adapter = MPAdapter(api_key="fake", cache_dir=Path("/tmp/test_cache"))
        assert adapter.cache_path() == Path("/tmp/test_cache")

    @pytest.mark.network
    def test_live_api_fetch(self) -> None:
        """Requires MP_API_KEY env var. Run with: pytest -m network"""
        adapter = MPAdapter(cache_dir=Path("/tmp/test_mp_live"))
        df = adapter.load_raw()
        assert len(df) > 100_000
        assert "material_id" in df.columns
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_adapters.py -v -m "not network"
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement MPAdapter**

```python
# src/crystal_prop_bench/data/mp_adapter.py
"""Materials Project adapter using mp-api client."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import pandas as pd
from mp_api.client import MPRester

from crystal_prop_bench.data.adapter import DatasetAdapter

logger = logging.getLogger(__name__)


class MPAdapter(DatasetAdapter):
    """Adapter for Materials Project data via mp-api."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path = Path("data/mp"),
    ) -> None:
        self._api_key = api_key or os.environ.get("MP_API_KEY", "")
        self._cache_dir = cache_dir

    def cache_path(self) -> Path:
        return self._cache_dir

    def load_raw(self) -> pd.DataFrame:
        """Fetch all crystals from Materials Project with target properties."""
        logger.info("Fetching data from Materials Project API...")

        with MPRester(self._api_key) as mpr:
            docs = mpr.materials.summary.search(
                fields=[
                    "material_id",
                    "formula_pretty",
                    "formation_energy_per_atom",
                    "band_gap",
                    "nsites",
                    "symmetry",
                    "structure",
                ],
            )

        logger.info("Fetched %d documents from Materials Project", len(docs))

        # Store structures separately (too complex for parquet)
        structures = {}
        rows = []
        for doc in docs:
            mid = str(doc.material_id)
            structures[mid] = doc.structure
            rows.append({
                "material_id": mid,
                "formula_pretty": doc.formula_pretty,
                "formation_energy_per_atom": doc.formation_energy_per_atom,
                "band_gap": doc.band_gap,
                "nsites": doc.nsites,
                "spacegroup_number": doc.symmetry.number,
            })

        df = pd.DataFrame(rows)

        # Cache structures as pickle
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        structures_path = self._cache_dir / "structures.pkl"
        with open(structures_path, "wb") as f:
            pickle.dump(structures, f)
        logger.info("Cached %d structures to %s", len(structures), structures_path)

        return df
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_adapters.py -v -m "not network"
```

Expected: 3 tests PASS (network test skipped).

**Step 5: Commit**

```bash
git add src/crystal_prop_bench/data/mp_adapter.py tests/test_adapters.py
git commit -m "feat: add MPAdapter with mocked API tests"
```

---

### Task 6: Standard and Domain-Shift Splits (TDD)

**Files:**
- Create: `src/crystal_prop_bench/data/splits.py`
- Create: `tests/test_splits.py`

**Step 1: Write failing tests**

```python
# tests/test_splits.py
import pandas as pd
import pytest

from crystal_prop_bench.data.splits import (
    domain_shift_split,
    ood_calibration_sweep,
    standard_split,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """200-row dataset: 80 oxide, 50 sulfide, 40 nitride, 30 halide."""
    import numpy as np

    rng = np.random.RandomState(42)
    families = (
        ["oxide"] * 80 + ["sulfide"] * 50 + ["nitride"] * 40 + ["halide"] * 30
    )
    return pd.DataFrame({
        "material_id": [f"mp-{i}" for i in range(200)],
        "formula_pretty": [f"X{i}" for i in range(200)],
        "formation_energy_per_atom": rng.randn(200),
        "band_gap": rng.rand(200) * 5,
        "nsites": rng.randint(1, 50, 200),
        "spacegroup_number": rng.randint(1, 231, 200),
        "chemistry_family": families,
    })


class TestStandardSplit:
    def test_returns_three_sets(self, sample_df: pd.DataFrame) -> None:
        train, cal, test = standard_split(sample_df, seed=42)
        assert len(train) + len(cal) + len(test) == len(sample_df)

    def test_approximate_proportions(self, sample_df: pd.DataFrame) -> None:
        train, cal, test = standard_split(sample_df, seed=42)
        n = len(sample_df)
        assert abs(len(train) / n - 0.80) < 0.05
        assert abs(len(cal) / n - 0.10) < 0.05
        assert abs(len(test) / n - 0.10) < 0.05

    def test_deterministic(self, sample_df: pd.DataFrame) -> None:
        t1, c1, te1 = standard_split(sample_df, seed=42)
        t2, c2, te2 = standard_split(sample_df, seed=42)
        pd.testing.assert_frame_equal(t1, t2)
        pd.testing.assert_frame_equal(c1, c2)
        pd.testing.assert_frame_equal(te1, te2)

    def test_different_seeds_differ(self, sample_df: pd.DataFrame) -> None:
        t1, _, _ = standard_split(sample_df, seed=42)
        t2, _, _ = standard_split(sample_df, seed=123)
        assert not t1["material_id"].equals(t2["material_id"])

    def test_no_overlap(self, sample_df: pd.DataFrame) -> None:
        train, cal, test = standard_split(sample_df, seed=42)
        train_ids = set(train["material_id"])
        cal_ids = set(cal["material_id"])
        test_ids = set(test["material_id"])
        assert train_ids.isdisjoint(cal_ids)
        assert train_ids.isdisjoint(test_ids)
        assert cal_ids.isdisjoint(test_ids)

    def test_stratified_by_family(self, sample_df: pd.DataFrame) -> None:
        """Each split should contain all families."""
        train, cal, test = standard_split(sample_df, seed=42)
        for split in [train, cal, test]:
            families = set(split["chemistry_family"])
            assert families == {"oxide", "sulfide", "nitride", "halide"}


class TestDomainShiftSplit:
    def test_returns_expected_keys(self, sample_df: pd.DataFrame) -> None:
        splits = domain_shift_split(sample_df, seed=42)
        expected_keys = {
            "train", "cal", "test_id",
            "test_ood_sulfide", "test_ood_nitride", "test_ood_halide",
        }
        assert set(splits.keys()) == expected_keys

    def test_train_cal_test_are_oxides_only(self, sample_df: pd.DataFrame) -> None:
        splits = domain_shift_split(sample_df, seed=42)
        for key in ["train", "cal", "test_id"]:
            families = splits[key]["chemistry_family"].unique()
            assert list(families) == ["oxide"]

    def test_ood_families_correct(self, sample_df: pd.DataFrame) -> None:
        splits = domain_shift_split(sample_df, seed=42)
        assert set(splits["test_ood_sulfide"]["chemistry_family"]) == {"sulfide"}
        assert set(splits["test_ood_nitride"]["chemistry_family"]) == {"nitride"}
        assert set(splits["test_ood_halide"]["chemistry_family"]) == {"halide"}

    def test_oxide_splits_sum(self, sample_df: pd.DataFrame) -> None:
        splits = domain_shift_split(sample_df, seed=42)
        total_oxide = (sample_df["chemistry_family"] == "oxide").sum()
        oxide_in_splits = (
            len(splits["train"]) + len(splits["cal"]) + len(splits["test_id"])
        )
        assert oxide_in_splits == total_oxide

    def test_ood_families_use_all_data(self, sample_df: pd.DataFrame) -> None:
        splits = domain_shift_split(sample_df, seed=42)
        assert len(splits["test_ood_sulfide"]) == 50
        assert len(splits["test_ood_nitride"]) == 40
        assert len(splits["test_ood_halide"]) == 30


class TestOODCalibrationSweep:
    def test_returns_correct_number_of_pairs(self) -> None:
        import numpy as np

        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "material_id": [f"mp-{i}" for i in range(200)],
            "y": rng.randn(200),
        })
        cal_sizes = [5, 10, 25, 50]
        pairs = ood_calibration_sweep(df, cal_sizes=cal_sizes, seed=42)
        assert len(pairs) == 4

    def test_cal_sizes_correct(self) -> None:
        import numpy as np

        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "material_id": [f"mp-{i}" for i in range(200)],
            "y": rng.randn(200),
        })
        cal_sizes = [5, 10, 25]
        pairs = ood_calibration_sweep(df, cal_sizes=cal_sizes, seed=42)
        for (cal, test), expected_size in zip(pairs, cal_sizes):
            assert len(cal) == expected_size
            assert len(test) == 200 - expected_size

    def test_no_overlap_in_pairs(self) -> None:
        import numpy as np

        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "material_id": [f"mp-{i}" for i in range(200)],
            "y": rng.randn(200),
        })
        pairs = ood_calibration_sweep(df, cal_sizes=[10, 50], seed=42)
        for cal, test in pairs:
            cal_ids = set(cal["material_id"])
            test_ids = set(test["material_id"])
            assert cal_ids.isdisjoint(test_ids)

    def test_deterministic(self) -> None:
        import numpy as np

        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "material_id": [f"mp-{i}" for i in range(200)],
            "y": rng.randn(200),
        })
        p1 = ood_calibration_sweep(df, cal_sizes=[10], seed=42)
        p2 = ood_calibration_sweep(df, cal_sizes=[10], seed=42)
        pd.testing.assert_frame_equal(p1[0][0], p2[0][0])

    def test_cal_size_exceeding_data_raises(self) -> None:
        df = pd.DataFrame({
            "material_id": ["mp-1", "mp-2"],
            "y": [1.0, 2.0],
        })
        with pytest.raises(ValueError, match="exceeds"):
            ood_calibration_sweep(df, cal_sizes=[5], seed=42)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_splits.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement splits**

```python
# src/crystal_prop_bench/data/splits.py
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
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_splits.py -v
```

Expected: All 14 tests PASS.

**Step 5: Commit**

```bash
git add src/crystal_prop_bench/data/splits.py tests/test_splits.py
git commit -m "feat: add standard, domain-shift, and OOD calibration sweep splits"
```

---

### Task 7: Test Fixture Creation Script

**Files:**
- Create: `scripts/create_fixture.py`

This is a one-time script that fetches 100 crystals from MP (25 per family),
featurizes them, and saves fixtures to `tests/fixtures/`. The fixtures are
committed to the repo so CI doesn't need MP API access.

**Step 1: Write the fixture creation script**

```python
# scripts/create_fixture.py
"""One-time script to create test fixtures from Materials Project.

Fetches 25 crystals per chemistry family, computes features, and saves
to tests/fixtures/. Requires MP_API_KEY environment variable.

Run once, commit the results. CI never needs API access.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition

from crystal_prop_bench.data.chemistry import classify_chemistry_family
from crystal_prop_bench.data.mp_adapter import MPAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIXTURES_DIR = Path("tests/fixtures")
SAMPLES_PER_FAMILY = 25
FAMILIES = ["oxide", "sulfide", "nitride", "halide"]


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch full dataset
    adapter = MPAdapter(cache_dir=Path("data/mp"))
    df = adapter.load()

    # Load structures
    with open(adapter.cache_path() / "structures.pkl", "rb") as f:
        structures = pickle.load(f)

    # Sample 25 per family, deterministic
    sampled_rows = []
    for family in FAMILIES:
        family_df = df[df["chemistry_family"] == family]
        sample = family_df.sample(n=SAMPLES_PER_FAMILY, random_state=42)
        sampled_rows.append(sample)
        logger.info("Sampled %d %s crystals", len(sample), family)

    fixture_df = pd.concat(sampled_rows, ignore_index=True)
    logger.info("Total fixture size: %d", len(fixture_df))

    # Save tabular fixture
    fixture_df.to_parquet(FIXTURES_DIR / "fixture_crystals.parquet", index=False)

    # Save corresponding structures
    fixture_structures = {
        row["material_id"]: structures[row["material_id"]]
        for _, row in fixture_df.iterrows()
        if row["material_id"] in structures
    }
    with open(FIXTURES_DIR / "fixture_structures.pkl", "wb") as f:
        pickle.dump(fixture_structures, f)

    logger.info(
        "Saved %d fixture crystals and %d structures",
        len(fixture_df),
        len(fixture_structures),
    )

    # Pre-compute Magpie features
    from crystal_prop_bench.data.featurizers import compute_magpie_features

    magpie_df = compute_magpie_features(fixture_df)
    magpie_df.to_parquet(FIXTURES_DIR / "fixture_magpie_features.parquet", index=False)
    logger.info("Saved Magpie features: %d rows, %d cols", *magpie_df.shape)

    # Pre-compute Voronoi features (some will fail)
    from crystal_prop_bench.data.featurizers import compute_voronoi_features

    voronoi_df = compute_voronoi_features(fixture_df, fixture_structures)
    voronoi_df.to_parquet(
        FIXTURES_DIR / "fixture_voronoi_features.parquet", index=False
    )
    logger.info("Saved Voronoi features: %d rows, %d cols", *voronoi_df.shape)
    logger.info(
        "Voronoi survival rate: %d/%d (%.1f%%)",
        len(voronoi_df),
        len(fixture_df),
        100.0 * len(voronoi_df) / len(fixture_df),
    )

    # Save the expected survival count for test pinning
    survival_meta = {"expected_voronoi_count": len(voronoi_df)}
    pd.Series(survival_meta).to_json(FIXTURES_DIR / "fixture_meta.json")


if __name__ == "__main__":
    main()
```

**Step 2: Commit** (fixtures are committed after running the script)

```bash
git add scripts/create_fixture.py
git commit -m "feat: add fixture creation script for 100-crystal test set"
```

**Note:** This script depends on Task 9 (featurizers). Run it after featurizers
are implemented. The actual fixture files are committed after running this script.

---

### Task 8: Update conftest.py with fixture loaders

**Files:**
- Modify: `tests/conftest.py`

**Step 1: Update conftest.py**

```python
# tests/conftest.py
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def fixture_crystals() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES_DIR / "fixture_crystals.parquet")


@pytest.fixture
def fixture_structures() -> dict:
    with open(FIXTURES_DIR / "fixture_structures.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture
def fixture_magpie_features() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES_DIR / "fixture_magpie_features.parquet")


@pytest.fixture
def fixture_voronoi_features() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES_DIR / "fixture_voronoi_features.parquet")


@pytest.fixture
def fixture_meta() -> dict:
    with open(FIXTURES_DIR / "fixture_meta.json") as f:
        return json.load(f)
```

**Step 2: Commit**

```bash
git add tests/conftest.py
git commit -m "feat: add fixture loaders to conftest.py"
```

---

## Block 2: Featurizers & Models

### Task 9: Magpie Featurizer Wrapper (TDD)

**Files:**
- Create: `src/crystal_prop_bench/data/featurizers.py`
- Create: `tests/test_featurizers.py`

**Step 1: Write failing tests**

```python
# tests/test_featurizers.py
import pandas as pd
import pytest

from crystal_prop_bench.data.featurizers import compute_magpie_features


class TestMagpieFeaturizer:
    def test_returns_dataframe(self) -> None:
        df = pd.DataFrame({
            "material_id": ["mp-1", "mp-2"],
            "formula_pretty": ["Fe2O3", "ZnS"],
        })
        result = compute_magpie_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_material_id(self) -> None:
        df = pd.DataFrame({
            "material_id": ["mp-1", "mp-2"],
            "formula_pretty": ["Fe2O3", "ZnS"],
        })
        result = compute_magpie_features(df)
        assert "material_id" in result.columns
        assert list(result["material_id"]) == ["mp-1", "mp-2"]

    def test_feature_columns_present(self) -> None:
        df = pd.DataFrame({
            "material_id": ["mp-1"],
            "formula_pretty": ["Fe2O3"],
        })
        result = compute_magpie_features(df)
        # Magpie produces ~150 features
        feature_cols = [c for c in result.columns if c != "material_id"]
        assert len(feature_cols) > 100

    def test_no_nan_in_features(self) -> None:
        df = pd.DataFrame({
            "material_id": ["mp-1", "mp-2", "mp-3"],
            "formula_pretty": ["Fe2O3", "ZnS", "GaN"],
        })
        result = compute_magpie_features(df)
        feature_cols = [c for c in result.columns if c != "material_id"]
        assert not result[feature_cols].isna().any().any()
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_featurizers.py::TestMagpieFeaturizer -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement Magpie featurizer**

```python
# src/crystal_prop_bench/data/featurizers.py
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
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_featurizers.py::TestMagpieFeaturizer -v
```

Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add src/crystal_prop_bench/data/featurizers.py tests/test_featurizers.py
git commit -m "feat: add Magpie composition featurizer wrapper"
```

---

### Task 10: Voronoi Structural Featurizer Wrapper (TDD)

**Files:**
- Modify: `src/crystal_prop_bench/data/featurizers.py`
- Modify: `tests/test_featurizers.py`

**Step 1: Add failing tests to test_featurizers.py**

```python
# Append to tests/test_featurizers.py

from crystal_prop_bench.data.featurizers import compute_voronoi_features


class TestVoronoiFeaturizer:
    def test_returns_dataframe(self, fixture_crystals, fixture_structures):
        # Use first 5 crystals for speed
        small_df = fixture_crystals.head(5)
        small_structs = {
            mid: fixture_structures[mid]
            for mid in small_df["material_id"]
            if mid in fixture_structures
        }
        result = compute_voronoi_features(small_df, small_structs)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_material_id(self, fixture_crystals, fixture_structures):
        small_df = fixture_crystals.head(5)
        small_structs = {
            mid: fixture_structures[mid]
            for mid in small_df["material_id"]
            if mid in fixture_structures
        }
        result = compute_voronoi_features(small_df, small_structs)
        assert "material_id" in result.columns
        # All returned rows should have valid material_ids
        assert result["material_id"].isin(small_df["material_id"]).all()

    def test_drops_failed_structures(self):
        """Structures that fail featurization are dropped, not errored."""
        # This test uses the fixture which has a known survival count
        # Actual assertion uses fixture_meta pinned count
        pass  # Covered by test_pinned_survival_count below

    def test_pinned_survival_count(
        self, fixture_voronoi_features, fixture_meta
    ):
        """Voronoi survival count must match pinned value."""
        expected = fixture_meta["expected_voronoi_count"]
        assert len(fixture_voronoi_features) == expected

    def test_more_features_than_magpie(
        self, fixture_magpie_features, fixture_voronoi_features
    ):
        """Tier 2 should have more feature columns than Tier 1."""
        magpie_cols = len([
            c for c in fixture_magpie_features.columns if c != "material_id"
        ])
        voronoi_cols = len([
            c for c in fixture_voronoi_features.columns if c != "material_id"
        ])
        assert voronoi_cols > magpie_cols
```

**Step 2: Implement Voronoi featurizer**

Add to `src/crystal_prop_bench/data/featurizers.py`:

```python
from matminer.featurizers.structure import (
    DensityFeatures,
    GlobalSymmetryFeatures,
    SiteStatsFingerprint,
)
from matminer.featurizers.site import CoordinationNumber


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
```

**Step 3: Run tests**

```bash
pytest tests/test_featurizers.py -v
```

Expected: Tests that use fixture files will skip/fail until fixtures are
created (Task 7 script). Magpie tests should still pass. Voronoi tests
with `fixture_*` parameters pass once fixtures exist.

**Step 4: Commit**

```bash
git add src/crystal_prop_bench/data/featurizers.py tests/test_featurizers.py
git commit -m "feat: add Voronoi structural featurizer with failure handling"
```

---

### Task 11: LightGBM Training Function (TDD)

**Files:**
- Create: `src/crystal_prop_bench/models/lgbm_baseline.py`
- Create: `tests/test_models.py`

**Step 1: Write failing tests**

```python
# tests/test_models.py
import numpy as np
import pytest

from crystal_prop_bench.models.lgbm_baseline import train_lgbm


class TestTrainLGBM:
    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.RandomState(42)
        n_train, n_cal, n_features = 200, 50, 10
        X_train = rng.randn(n_train, n_features)
        y_train = X_train[:, 0] * 2 + rng.randn(n_train) * 0.1
        X_cal = rng.randn(n_cal, n_features)
        y_cal = X_cal[:, 0] * 2 + rng.randn(n_cal) * 0.1
        return X_train, y_train, X_cal, y_cal

    def test_returns_model_and_residuals(self, synthetic_data):
        X_train, y_train, X_cal, y_cal = synthetic_data
        model, cal_residuals = train_lgbm(
            X_train, y_train, X_cal, y_cal, seed=42
        )
        assert model is not None
        assert len(cal_residuals) == len(y_cal)

    def test_residuals_are_absolute(self, synthetic_data):
        X_train, y_train, X_cal, y_cal = synthetic_data
        _, cal_residuals = train_lgbm(
            X_train, y_train, X_cal, y_cal, seed=42
        )
        assert (cal_residuals >= 0).all()

    def test_model_predicts_correct_shape(self, synthetic_data):
        X_train, y_train, X_cal, y_cal = synthetic_data
        model, _ = train_lgbm(X_train, y_train, X_cal, y_cal, seed=42)
        preds = model.predict(X_cal)
        assert preds.shape == (len(X_cal),)

    def test_model_learns_linear_pattern(self, synthetic_data):
        X_train, y_train, X_cal, y_cal = synthetic_data
        model, _ = train_lgbm(X_train, y_train, X_cal, y_cal, seed=42)
        preds = model.predict(X_cal)
        mae = np.mean(np.abs(preds - y_cal))
        # Should achieve < 0.5 MAE on this simple linear problem
        assert mae < 0.5

    def test_deterministic_with_same_seed(self, synthetic_data):
        X_train, y_train, X_cal, y_cal = synthetic_data
        _, res1 = train_lgbm(X_train, y_train, X_cal, y_cal, seed=42)
        _, res2 = train_lgbm(X_train, y_train, X_cal, y_cal, seed=42)
        np.testing.assert_array_almost_equal(res1, res2)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_models.py -v
```

**Step 3: Implement LightGBM training**

```python
# src/crystal_prop_bench/models/lgbm_baseline.py
"""LightGBM baseline model for Tier 1 and Tier 2."""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np

logger = logging.getLogger(__name__)


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    seed: int = 42,
    n_estimators: int = 1000,
    learning_rate: float = 0.05,
    num_leaves: int = 127,
    min_child_samples: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 50,
) -> tuple[lgb.LGBMRegressor, np.ndarray]:
    """Train LightGBM regressor and compute calibration residuals.

    Parameters
    ----------
    X_train, y_train : Training data.
    X_cal, y_cal : Calibration data (used for early stopping and residuals).
    seed : Random seed.
    Other params : LightGBM hyperparameters.

    Returns
    -------
    (model, cal_residuals) where cal_residuals = |y_cal - y_pred_cal|.
    """
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="mae",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=seed,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_cal, y_cal)],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )

    cal_preds = model.predict(X_cal)
    cal_residuals = np.abs(y_cal - cal_preds)

    logger.info(
        "Trained LightGBM (seed=%d): best_iteration=%d, cal_MAE=%.4f",
        seed,
        model.best_iteration_,
        np.mean(cal_residuals),
    )

    return model, cal_residuals
```

**Step 4: Run tests**

```bash
pytest tests/test_models.py -v
```

Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add src/crystal_prop_bench/models/lgbm_baseline.py tests/test_models.py
git commit -m "feat: add LightGBM training with calibration residual output"
```

---

### Task 12: Metrics Module (TDD)

**Files:**
- Create: `src/crystal_prop_bench/evaluation/metrics.py`
- Create: `tests/test_metrics.py`

**Step 1: Write failing tests**

```python
# tests/test_metrics.py
import numpy as np
import pytest

from crystal_prop_bench.evaluation.metrics import (
    aggregate_seeds,
    compute_metrics,
    compute_per_family_metrics,
)


class TestComputeMetrics:
    def test_returns_expected_keys(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        result = compute_metrics(y_true, y_pred)
        assert "mae" in result
        assert "rmse" in result
        assert "r2" in result

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        result = compute_metrics(y, y)
        assert result["mae"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["r2"] == pytest.approx(1.0)

    def test_known_mae(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        result = compute_metrics(y_true, y_pred)
        assert result["mae"] == pytest.approx(0.5)


class TestPerFamilyMetrics:
    def test_returns_all_families(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1])
        families = np.array(["oxide", "oxide", "sulfide", "sulfide"])
        result = compute_per_family_metrics(y_true, y_pred, families)
        assert "oxide" in result
        assert "sulfide" in result

    def test_per_family_correct_values(self):
        y_true = np.array([1.0, 2.0, 10.0, 20.0])
        y_pred = np.array([1.0, 2.0, 11.0, 21.0])  # oxide perfect, sulfide MAE=1
        families = np.array(["oxide", "oxide", "sulfide", "sulfide"])
        result = compute_per_family_metrics(y_true, y_pred, families)
        assert result["oxide"]["mae"] == pytest.approx(0.0)
        assert result["sulfide"]["mae"] == pytest.approx(1.0)


class TestAggregateSeeds:
    def test_mean_and_std(self):
        seed_results = [
            {"mae": 1.0, "rmse": 2.0},
            {"mae": 2.0, "rmse": 3.0},
            {"mae": 3.0, "rmse": 4.0},
        ]
        result = aggregate_seeds(seed_results)
        assert result["mae_mean"] == pytest.approx(2.0)
        assert result["mae_std"] == pytest.approx(np.std([1.0, 2.0, 3.0], ddof=1))
        assert result["rmse_mean"] == pytest.approx(3.0)

    def test_single_seed(self):
        seed_results = [{"mae": 1.5}]
        result = aggregate_seeds(seed_results)
        assert result["mae_mean"] == pytest.approx(1.5)
        assert result["mae_std"] == pytest.approx(0.0)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_metrics.py -v
```

**Step 3: Implement metrics**

```python
# src/crystal_prop_bench/evaluation/metrics.py
"""Evaluation metrics: MAE, RMSE, R², per-family breakdown, seed aggregation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute MAE, RMSE, R² for a single run."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_per_family_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    families: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down by chemistry family."""
    result = {}
    for family in np.unique(families):
        mask = families == family
        result[family] = compute_metrics(y_true[mask], y_pred[mask])
    return result


def aggregate_seeds(
    seed_results: list[dict[str, float]],
) -> dict[str, float]:
    """Aggregate per-seed metric dicts into mean +/- std.

    Input: [{"mae": 1.0, "rmse": 2.0}, {"mae": 2.0, "rmse": 3.0}, ...]
    Output: {"mae_mean": 1.5, "mae_std": 0.5, "rmse_mean": 2.5, "rmse_std": 0.5, ...}
    """
    if len(seed_results) == 0:
        return {}

    keys = seed_results[0].keys()
    result = {}
    for key in keys:
        values = [r[key] for r in seed_results]
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return result
```

**Step 4: Run tests**

```bash
pytest tests/test_metrics.py -v
```

Expected: All 7 tests PASS.

**Step 5: Commit**

```bash
git add src/crystal_prop_bench/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: add metrics module with per-family breakdown and seed aggregation"
```

---

### Task 13: Training Scripts

**Files:**
- Create: `scripts/run_tier1.py`
- Create: `scripts/run_tier2.py`
- Create: `scripts/download_data.py`

**Step 1: Create download_data.py**

```python
# scripts/download_data.py
"""Download and cache Materials Project data."""

import logging
from pathlib import Path

from crystal_prop_bench.data.mp_adapter import MPAdapter

logging.basicConfig(level=logging.INFO)


def main() -> None:
    adapter = MPAdapter(cache_dir=Path("data/mp"))
    df = adapter.load()
    print(f"Loaded {len(df)} crystals")
    print(f"Families: {df['chemistry_family'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
```

**Step 2: Create run_tier1.py**

```python
# scripts/run_tier1.py
"""Tier 1 training: Magpie composition features + LightGBM.

Trains 3 seeds x 2 targets x 2 splits (standard + domain-shift).
Saves predictions to results/predictions/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from crystal_prop_bench.data.featurizers import compute_magpie_features
from crystal_prop_bench.data.mp_adapter import MPAdapter
from crystal_prop_bench.data.splits import domain_shift_split, standard_split
from crystal_prop_bench.evaluation.metrics import compute_metrics
from crystal_prop_bench.models.lgbm_baseline import train_lgbm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODELS_DIR = RESULTS_DIR / "models"


def save_predictions(
    material_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    families: np.ndarray,
    split_label: str,
    filename: str,
) -> None:
    """Save prediction parquet with standard columns."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "material_id": material_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "chemistry_family": families,
        "split": split_label,
    }).to_parquet(PREDICTIONS_DIR / filename, index=False)


def run_standard_split(
    df: pd.DataFrame,
    features: pd.DataFrame,
    target: str,
    seeds: list[int],
) -> None:
    """Train and evaluate on standard split."""
    for seed in seeds:
        train_df, cal_df, test_df = standard_split(df, seed=seed)

        # Align features
        train_feat = features[features["material_id"].isin(train_df["material_id"])]
        cal_feat = features[features["material_id"].isin(cal_df["material_id"])]
        test_feat = features[features["material_id"].isin(test_df["material_id"])]

        # Merge to align rows
        train_merged = train_df.merge(train_feat, on="material_id")
        cal_merged = cal_df.merge(cal_feat, on="material_id")
        test_merged = test_df.merge(test_feat, on="material_id")

        feature_cols = [c for c in features.columns if c != "material_id"]

        X_train = train_merged[feature_cols].values
        y_train = train_merged[target].values
        X_cal = cal_merged[feature_cols].values
        y_cal = cal_merged[target].values
        X_test = test_merged[feature_cols].values
        y_test = test_merged[target].values

        with mlflow.start_run(run_name=f"tier1_standard_{target}_seed{seed}"):
            mlflow.log_params({"tier": 1, "split": "standard", "target": target, "seed": seed})

            model, cal_residuals = train_lgbm(X_train, y_train, X_cal, y_cal, seed=seed)
            test_preds = model.predict(X_test)

            metrics = compute_metrics(y_test, test_preds)
            mlflow.log_metrics(metrics)
            logger.info("Tier 1 standard %s seed=%d: %s", target, seed, metrics)

            # Save predictions
            target_short = "ef" if target == "formation_energy_per_atom" else "bg"
            save_predictions(
                test_merged["material_id"].values,
                y_test, test_preds,
                test_merged["chemistry_family"].values,
                "standard_test",
                f"tier1_standard_seed{seed}_{target_short}.parquet",
            )
            # Also save cal predictions for conformal
            cal_preds = model.predict(X_cal)
            save_predictions(
                cal_merged["material_id"].values,
                y_cal, cal_preds,
                cal_merged["chemistry_family"].values,
                "standard_cal",
                f"tier1_standard_seed{seed}_{target_short}_cal.parquet",
            )

            # Save model for SHAP
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            import joblib
            joblib.dump(model, MODELS_DIR / f"tier1_standard_seed{seed}_{target_short}.joblib")


def run_domain_shift(
    df: pd.DataFrame,
    features: pd.DataFrame,
    target: str,
    seeds: list[int],
) -> None:
    """Train and evaluate on domain-shift split."""
    for seed in seeds:
        splits = domain_shift_split(df, seed=seed)

        train_feat = features[features["material_id"].isin(splits["train"]["material_id"])]
        cal_feat = features[features["material_id"].isin(splits["cal"]["material_id"])]

        train_merged = splits["train"].merge(train_feat, on="material_id")
        cal_merged = splits["cal"].merge(cal_feat, on="material_id")

        feature_cols = [c for c in features.columns if c != "material_id"]

        X_train = train_merged[feature_cols].values
        y_train = train_merged[target].values
        X_cal = cal_merged[feature_cols].values
        y_cal = cal_merged[target].values

        with mlflow.start_run(run_name=f"tier1_domshift_{target}_seed{seed}"):
            mlflow.log_params({"tier": 1, "split": "domain_shift", "target": target, "seed": seed})

            model, cal_residuals = train_lgbm(X_train, y_train, X_cal, y_cal, seed=seed)

            target_short = "ef" if target == "formation_energy_per_atom" else "bg"

            # Save cal predictions
            cal_preds = model.predict(X_cal)
            save_predictions(
                cal_merged["material_id"].values,
                y_cal, cal_preds,
                cal_merged["chemistry_family"].values,
                "domshift_cal",
                f"tier1_domshift_seed{seed}_{target_short}_cal.parquet",
            )

            # Predict on each test set
            for split_key in ["test_id", "test_ood_sulfide", "test_ood_nitride", "test_ood_halide"]:
                test_df = splits[split_key]
                test_feat = features[features["material_id"].isin(test_df["material_id"])]
                test_merged = test_df.merge(test_feat, on="material_id")

                if len(test_merged) == 0:
                    continue

                X_test = test_merged[feature_cols].values
                y_test = test_merged[target].values
                test_preds = model.predict(X_test)

                metrics = compute_metrics(y_test, test_preds)
                mlflow.log_metrics({f"{split_key}_{k}": v for k, v in metrics.items()})
                logger.info("Tier 1 domshift %s %s seed=%d: %s", target, split_key, seed, metrics)

                save_predictions(
                    test_merged["material_id"].values,
                    y_test, test_preds,
                    test_merged["chemistry_family"].values,
                    f"domshift_{split_key}",
                    f"tier1_domshift_seed{seed}_{target_short}_{split_key}.parquet",
                )

            # Save model
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            import joblib
            joblib.dump(model, MODELS_DIR / f"tier1_domshift_seed{seed}_{target_short}.joblib")


def main() -> None:
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)

    seeds = config["evaluation"]["seeds"]
    targets = config["evaluation"]["targets"]

    # Load data
    adapter = MPAdapter(cache_dir=Path("data/mp"))
    df = adapter.load()

    # Compute features
    features = compute_magpie_features(
        df, cache_path=Path("data/mp/magpie_features.parquet")
    )

    mlflow.set_experiment("crystal-prop-bench-tier1")

    for target in targets:
        run_standard_split(df, features, target, seeds)
        run_domain_shift(df, features, target, seeds)


if __name__ == "__main__":
    main()
```

**Step 3: Create run_tier2.py**

Same structure as `run_tier1.py` but uses `compute_voronoi_features` and
filters to the Voronoi-survivable subset. Also runs Tier 1 on the same
subset for the bias check.

```python
# scripts/run_tier2.py
"""Tier 2 training: Voronoi structural features + LightGBM.

Also runs Tier 1 on the Voronoi-survivable subset for bias check.
Saves predictions to results/predictions/.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from crystal_prop_bench.data.featurizers import (
    compute_magpie_features,
    compute_voronoi_features,
)
from crystal_prop_bench.data.mp_adapter import MPAdapter
from crystal_prop_bench.data.splits import domain_shift_split, standard_split
from crystal_prop_bench.evaluation.metrics import compute_metrics
from crystal_prop_bench.models.lgbm_baseline import train_lgbm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODELS_DIR = RESULTS_DIR / "models"


def save_predictions(
    material_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    families: np.ndarray,
    split_label: str,
    filename: str,
) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "material_id": material_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "chemistry_family": families,
        "split": split_label,
    }).to_parquet(PREDICTIONS_DIR / filename, index=False)


def train_and_predict(
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: pd.DataFrame,
    target: str,
    seed: int,
    tier_label: str,
    split_label: str,
    test_split_key: str,
) -> None:
    """Generic train-predict-save loop."""
    feature_cols = [c for c in features.columns if c != "material_id"]

    train_m = train_df.merge(features, on="material_id")
    cal_m = cal_df.merge(features, on="material_id")
    test_m = test_df.merge(features, on="material_id")

    if len(train_m) == 0 or len(test_m) == 0:
        return

    model, _ = train_lgbm(
        train_m[feature_cols].values, train_m[target].values,
        cal_m[feature_cols].values, cal_m[target].values,
        seed=seed,
    )

    test_preds = model.predict(test_m[feature_cols].values)
    target_short = "ef" if target == "formation_energy_per_atom" else "bg"

    save_predictions(
        test_m["material_id"].values,
        test_m[target].values, test_preds,
        test_m["chemistry_family"].values,
        f"{split_label}_{test_split_key}",
        f"{tier_label}_{split_label}_seed{seed}_{target_short}_{test_split_key}.parquet",
    )

    # Save cal predictions
    cal_preds = model.predict(cal_m[feature_cols].values)
    save_predictions(
        cal_m["material_id"].values,
        cal_m[target].values, cal_preds,
        cal_m["chemistry_family"].values,
        f"{split_label}_cal",
        f"{tier_label}_{split_label}_seed{seed}_{target_short}_cal.parquet",
    )

    metrics = compute_metrics(test_m[target].values, test_preds)
    mlflow.log_metrics({f"{test_split_key}_{k}": v for k, v in metrics.items()})
    logger.info("%s %s %s %s seed=%d: %s", tier_label, split_label, target, test_split_key, seed, metrics)

    if test_split_key in ("test", "test_id"):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(model, MODELS_DIR / f"{tier_label}_{split_label}_seed{seed}_{target_short}.joblib")


def main() -> None:
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)

    seeds = config["evaluation"]["seeds"]
    targets = config["evaluation"]["targets"]

    # Load data and structures
    adapter = MPAdapter(cache_dir=Path("data/mp"))
    df = adapter.load()

    with open(adapter.cache_path() / "structures.pkl", "rb") as f:
        structures = pickle.load(f)

    # Compute Voronoi features (includes Magpie)
    voronoi_features = compute_voronoi_features(
        df, structures, cache_path=Path("data/mp/voronoi_features.parquet")
    )

    # Also get pure Magpie features for bias check
    magpie_features = compute_magpie_features(
        df, cache_path=Path("data/mp/magpie_features.parquet")
    )

    # Voronoi-survivable subset
    voronoi_ids = set(voronoi_features["material_id"])
    df_voronoi = df[df["material_id"].isin(voronoi_ids)].copy()
    magpie_voronoi = magpie_features[magpie_features["material_id"].isin(voronoi_ids)]

    logger.info(
        "Voronoi subset: %d/%d crystals (%.1f%%)",
        len(df_voronoi), len(df),
        100.0 * len(df_voronoi) / len(df),
    )

    mlflow.set_experiment("crystal-prop-bench-tier2")

    for target in targets:
        for seed in seeds:
            # --- Standard split ---
            train, cal, test = standard_split(df_voronoi, seed=seed)

            with mlflow.start_run(run_name=f"tier2_standard_{target}_seed{seed}"):
                mlflow.log_params({"tier": 2, "split": "standard", "target": target, "seed": seed})
                train_and_predict(train, cal, test, voronoi_features, target, seed, "tier2", "standard", "test")

            # Bias check: Tier 1 on Voronoi subset
            with mlflow.start_run(run_name=f"tier1_voronoi_subset_standard_{target}_seed{seed}"):
                mlflow.log_params({"tier": "1_voronoi_subset", "split": "standard", "target": target, "seed": seed})
                train_and_predict(train, cal, test, magpie_voronoi, target, seed, "tier1sub", "standard", "test")

            # --- Domain-shift split ---
            splits = domain_shift_split(df_voronoi, seed=seed)

            with mlflow.start_run(run_name=f"tier2_domshift_{target}_seed{seed}"):
                mlflow.log_params({"tier": 2, "split": "domain_shift", "target": target, "seed": seed})
                for split_key in ["test_id", "test_ood_sulfide", "test_ood_nitride", "test_ood_halide"]:
                    train_and_predict(
                        splits["train"], splits["cal"], splits[split_key],
                        voronoi_features, target, seed, "tier2", "domshift", split_key,
                    )


if __name__ == "__main__":
    main()
```

**Step 4: Commit**

```bash
git add scripts/download_data.py scripts/run_tier1.py scripts/run_tier2.py
git commit -m "feat: add training scripts for Tier 1 and Tier 2 with prediction output"
```

---

## Block 3: UQ & Domain Shift

### Task 14: Split Conformal Regression (TDD)

**Files:**
- Create: `src/crystal_prop_bench/evaluation/conformal.py`
- Create: `tests/test_conformal.py`

**Step 1: Write failing tests**

```python
# tests/test_conformal.py
import numpy as np
import pytest

from crystal_prop_bench.evaluation.conformal import (
    conformal_regression_interval,
    evaluate_conformal_coverage,
)


class TestConformalRegressionInterval:
    def test_returns_lower_upper(self):
        cal_residuals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        test_preds = np.array([1.0, 2.0])
        lower, upper = conformal_regression_interval(cal_residuals, test_preds, alpha=0.20)
        assert lower.shape == (2,)
        assert upper.shape == (2,)
        assert (upper > lower).all()

    def test_intervals_centered_on_predictions(self):
        cal_residuals = np.array([0.1, 0.2, 0.3])
        test_preds = np.array([5.0])
        lower, upper = conformal_regression_interval(cal_residuals, test_preds, alpha=0.10)
        midpoint = (lower[0] + upper[0]) / 2
        assert midpoint == pytest.approx(5.0)

    def test_coverage_on_uniform_residuals(self):
        """With enough uniform residuals, coverage should be close to 1-alpha."""
        rng = np.random.RandomState(42)
        n_cal = 10000
        cal_residuals = rng.uniform(0, 1, n_cal)
        test_preds = rng.randn(5000)
        test_true = test_preds + rng.uniform(-1, 1, 5000)

        lower, upper = conformal_regression_interval(cal_residuals, test_preds, alpha=0.10)
        covered = ((test_true >= lower) & (test_true <= upper)).mean()
        assert covered >= 0.88  # some slack for finite sample

    def test_zero_residuals_give_zero_width(self):
        cal_residuals = np.zeros(100)
        test_preds = np.array([1.0, 2.0])
        lower, upper = conformal_regression_interval(cal_residuals, test_preds, alpha=0.10)
        np.testing.assert_array_almost_equal(lower, test_preds)
        np.testing.assert_array_almost_equal(upper, test_preds)

    def test_alpha_affects_width(self):
        cal_residuals = np.linspace(0, 1, 100)
        test_preds = np.array([0.0])
        _, upper_10 = conformal_regression_interval(cal_residuals, test_preds, alpha=0.10)
        _, upper_30 = conformal_regression_interval(cal_residuals, test_preds, alpha=0.30)
        # Tighter alpha -> wider interval
        assert upper_10[0] > upper_30[0]


class TestEvaluateConformalCoverage:
    def test_returns_expected_keys(self):
        cal_residuals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.05, 2.1, 3.15])
        result = evaluate_conformal_coverage(
            cal_residuals, y_true, y_pred, alphas=[0.10, 0.20]
        )
        assert len(result) == 2
        assert "alpha" in result[0]
        assert "coverage" in result[0]
        assert "mean_width" in result[0]

    def test_perfect_predictions_have_full_coverage(self):
        cal_residuals = np.array([0.1, 0.2, 0.3])
        y = np.array([1.0, 2.0, 3.0])
        result = evaluate_conformal_coverage(
            cal_residuals, y, y, alphas=[0.10]
        )
        assert result[0]["coverage"] == pytest.approx(1.0)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_conformal.py -v
```

**Step 3: Implement conformal module**

```python
# src/crystal_prop_bench/evaluation/conformal.py
"""Split conformal regression intervals.

Implements Vovk et al. / Lei et al. split conformal prediction
for regression. Guarantees marginal coverage >= 1 - alpha on
exchangeable data.
"""

from __future__ import annotations

import numpy as np


def conformal_regression_interval(
    cal_residuals: np.ndarray,
    test_predictions: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute split conformal regression intervals.

    Parameters
    ----------
    cal_residuals : array of shape (n_cal,)
        Absolute residuals |y_cal - y_pred_cal| on calibration set.
    test_predictions : array of shape (n_test,)
        Point predictions on test set.
    alpha : float
        Miscoverage level. Intervals target 1 - alpha coverage.

    Returns
    -------
    (lower, upper) arrays of shape (n_test,).
    """
    n = len(cal_residuals)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = float(np.quantile(np.abs(cal_residuals), min(q, 1.0)))
    return test_predictions - q_hat, test_predictions + q_hat


def evaluate_conformal_coverage(
    cal_residuals: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alphas: list[float] | None = None,
) -> list[dict[str, float]]:
    """Evaluate conformal coverage at multiple alpha levels.

    Returns list of dicts with keys: alpha, coverage, mean_width.
    """
    if alphas is None:
        alphas = [0.10, 0.20, 0.30]

    results = []
    for alpha in alphas:
        lower, upper = conformal_regression_interval(cal_residuals, y_pred, alpha)
        covered = ((y_true >= lower) & (y_true <= upper)).mean()
        width = (upper - lower).mean()
        results.append({
            "alpha": alpha,
            "coverage": float(covered),
            "mean_width": float(width),
        })
    return results
```

**Step 4: Run tests**

```bash
pytest tests/test_conformal.py -v
```

Expected: All 7 tests PASS.

**Step 5: Commit**

```bash
git add src/crystal_prop_bench/evaluation/conformal.py tests/test_conformal.py
git commit -m "feat: add split conformal regression with coverage evaluation"
```

---

### Task 15: Domain Shift Analysis (TDD)

**Files:**
- Create: `src/crystal_prop_bench/evaluation/domain_shift.py`
- Create: `tests/test_domain_shift.py`

**Step 1: Write failing tests**

```python
# tests/test_domain_shift.py
import numpy as np
import pytest

from crystal_prop_bench.evaluation.domain_shift import compute_degradation_ratios


class TestDegradationRatios:
    def test_returns_expected_keys(self):
        id_metrics = {"mae": 0.1, "rmse": 0.2}
        ood_metrics = {
            "sulfide": {"mae": 0.3, "rmse": 0.5},
            "nitride": {"mae": 0.4, "rmse": 0.6},
        }
        result = compute_degradation_ratios(id_metrics, ood_metrics)
        assert "sulfide" in result
        assert "nitride" in result
        assert "mae_ratio" in result["sulfide"]

    def test_correct_ratio(self):
        id_metrics = {"mae": 0.1}
        ood_metrics = {"sulfide": {"mae": 0.3}}
        result = compute_degradation_ratios(id_metrics, ood_metrics)
        assert result["sulfide"]["mae_ratio"] == pytest.approx(3.0)

    def test_delta_mae(self):
        id_metrics = {"mae": 0.1}
        ood_metrics = {"sulfide": {"mae": 0.3}}
        result = compute_degradation_ratios(id_metrics, ood_metrics)
        assert result["sulfide"]["mae_delta"] == pytest.approx(0.2)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_domain_shift.py -v
```

**Step 3: Implement**

```python
# src/crystal_prop_bench/evaluation/domain_shift.py
"""Domain-shift degradation analysis."""

from __future__ import annotations


def compute_degradation_ratios(
    id_metrics: dict[str, float],
    ood_metrics: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compute degradation ratios and deltas for OOD families.

    Parameters
    ----------
    id_metrics : dict
        In-distribution metrics (e.g., {"mae": 0.1, "rmse": 0.2}).
    ood_metrics : dict of dict
        Per-family OOD metrics (e.g., {"sulfide": {"mae": 0.3}}).

    Returns
    -------
    Dict mapping family -> {"mae_ratio", "mae_delta", "rmse_ratio", "rmse_delta", ...}.
    """
    result = {}
    for family, metrics in ood_metrics.items():
        family_result = {}
        for key in metrics:
            if key in id_metrics and id_metrics[key] > 0:
                family_result[f"{key}_ratio"] = metrics[key] / id_metrics[key]
                family_result[f"{key}_delta"] = metrics[key] - id_metrics[key]
        result[family] = family_result
    return result
```

**Step 4: Run tests**

```bash
pytest tests/test_domain_shift.py -v
```

Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add src/crystal_prop_bench/evaluation/domain_shift.py tests/test_domain_shift.py
git commit -m "feat: add domain-shift degradation ratio analysis"
```

---

### Task 16: Evaluation Script

**Files:**
- Create: `scripts/run_evaluation.py`

**Step 1: Implement evaluation script**

```python
# scripts/run_evaluation.py
"""Cross-cutting evaluation: conformal, domain shift, calibration sweep, bias check.

Reads prediction parquets from results/predictions/.
Writes result CSVs to results/tables/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from crystal_prop_bench.data.splits import ood_calibration_sweep
from crystal_prop_bench.evaluation.conformal import (
    conformal_regression_interval,
    evaluate_conformal_coverage,
)
from crystal_prop_bench.evaluation.domain_shift import compute_degradation_ratios
from crystal_prop_bench.evaluation.metrics import (
    aggregate_seeds,
    compute_metrics,
    compute_per_family_metrics,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREDICTIONS_DIR = Path("results/predictions")
TABLES_DIR = Path("results/tables")


def load_predictions(pattern: str) -> list[pd.DataFrame]:
    """Load all prediction files matching a glob pattern."""
    files = sorted(PREDICTIONS_DIR.glob(pattern))
    return [pd.read_parquet(f) for f in files]


def build_benchmark_table(config: dict) -> pd.DataFrame:
    """Build the main benchmark table: tier x split x target x MAE +/- std."""
    rows = []
    targets = config["evaluation"]["targets"]

    for tier in ["tier1", "tier2", "tier1sub"]:
        for split in ["standard", "domshift"]:
            for target in targets:
                target_short = "ef" if target == "formation_energy_per_atom" else "bg"
                test_key = "test" if split == "standard" else "test_id"

                seed_metrics = []
                for seed in config["evaluation"]["seeds"]:
                    pattern = f"{tier}_{split}_seed{seed}_{target_short}_{test_key}.parquet"
                    files = list(PREDICTIONS_DIR.glob(pattern))
                    if not files:
                        continue
                    pred_df = pd.read_parquet(files[0])
                    metrics = compute_metrics(
                        pred_df["y_true"].values, pred_df["y_pred"].values
                    )
                    seed_metrics.append(metrics)

                if seed_metrics:
                    agg = aggregate_seeds(seed_metrics)
                    rows.append({
                        "tier": tier,
                        "split": split,
                        "target": target_short,
                        **agg,
                    })

    return pd.DataFrame(rows)


def build_domain_shift_table(config: dict) -> pd.DataFrame:
    """Build domain-shift degradation table."""
    rows = []
    targets = config["evaluation"]["targets"]

    for tier in ["tier1", "tier2"]:
        for target in targets:
            target_short = "ef" if target == "formation_energy_per_atom" else "bg"

            for seed in config["evaluation"]["seeds"]:
                # ID metrics
                id_files = list(PREDICTIONS_DIR.glob(
                    f"{tier}_domshift_seed{seed}_{target_short}_test_id.parquet"
                ))
                if not id_files:
                    continue
                id_df = pd.read_parquet(id_files[0])
                id_metrics = compute_metrics(id_df["y_true"].values, id_df["y_pred"].values)

                # OOD metrics per family
                ood_metrics = {}
                for family in ["sulfide", "nitride", "halide"]:
                    ood_files = list(PREDICTIONS_DIR.glob(
                        f"{tier}_domshift_seed{seed}_{target_short}_test_ood_{family}.parquet"
                    ))
                    if ood_files:
                        ood_df = pd.read_parquet(ood_files[0])
                        ood_metrics[family] = compute_metrics(
                            ood_df["y_true"].values, ood_df["y_pred"].values
                        )

                ratios = compute_degradation_ratios(id_metrics, ood_metrics)
                for family, ratio_dict in ratios.items():
                    rows.append({
                        "tier": tier,
                        "target": target_short,
                        "seed": seed,
                        "ood_family": family,
                        "id_mae": id_metrics["mae"],
                        "ood_mae": ood_metrics[family]["mae"],
                        **ratio_dict,
                    })

    return pd.DataFrame(rows)


def build_conformal_table(config: dict) -> pd.DataFrame:
    """Build conformal coverage table for all tiers and splits."""
    rows = []
    alphas = config["evaluation"]["alphas"]
    targets = config["evaluation"]["targets"]

    for tier in ["tier1", "tier2"]:
        for split in ["standard", "domshift"]:
            for target in targets:
                target_short = "ef" if target == "formation_energy_per_atom" else "bg"

                for seed in config["evaluation"]["seeds"]:
                    # Load cal residuals
                    cal_files = list(PREDICTIONS_DIR.glob(
                        f"{tier}_{split}_seed{seed}_{target_short}_cal.parquet"
                    ))
                    if not cal_files:
                        continue
                    cal_df = pd.read_parquet(cal_files[0])
                    cal_residuals = np.abs(
                        cal_df["y_true"].values - cal_df["y_pred"].values
                    )

                    # Evaluate on test sets
                    if split == "standard":
                        test_keys = ["test"]
                    else:
                        test_keys = ["test_id", "test_ood_sulfide", "test_ood_nitride", "test_ood_halide"]

                    for test_key in test_keys:
                        test_files = list(PREDICTIONS_DIR.glob(
                            f"{tier}_{split}_seed{seed}_{target_short}_{test_key}.parquet"
                        ))
                        if not test_files:
                            continue
                        test_df = pd.read_parquet(test_files[0])

                        coverage_results = evaluate_conformal_coverage(
                            cal_residuals,
                            test_df["y_true"].values,
                            test_df["y_pred"].values,
                            alphas=alphas,
                        )

                        for cr in coverage_results:
                            rows.append({
                                "tier": tier,
                                "split": split,
                                "target": target_short,
                                "seed": seed,
                                "test_set": test_key,
                                **cr,
                            })

    return pd.DataFrame(rows)


def build_calibration_sweep_table(config: dict) -> pd.DataFrame:
    """Build calibration sweep table: coverage vs. cal budget per OOD family."""
    rows = []
    cal_sizes = config["evaluation"]["cal_sizes"]
    alphas = config["evaluation"]["alphas"]
    targets = config["evaluation"]["targets"]

    for tier in ["tier1", "tier2"]:
        for target in targets:
            target_short = "ef" if target == "formation_energy_per_atom" else "bg"

            for seed in config["evaluation"]["seeds"]:
                # Load the model's OOD predictions for each family
                for family in ["sulfide", "nitride", "halide"]:
                    ood_files = list(PREDICTIONS_DIR.glob(
                        f"{tier}_domshift_seed{seed}_{target_short}_test_ood_{family}.parquet"
                    ))
                    if not ood_files:
                        continue
                    ood_df = pd.read_parquet(ood_files[0])

                    # Sweep calibration sizes
                    try:
                        pairs = ood_calibration_sweep(ood_df, cal_sizes=cal_sizes, seed=seed)
                    except ValueError:
                        continue

                    for (cal_subset, test_remainder), cal_size in zip(pairs, cal_sizes):
                        cal_resid = np.abs(
                            cal_subset["y_true"].values - cal_subset["y_pred"].values
                        )

                        for alpha in alphas:
                            lower, upper = conformal_regression_interval(
                                cal_resid,
                                test_remainder["y_pred"].values,
                                alpha,
                            )
                            covered = (
                                (test_remainder["y_true"].values >= lower)
                                & (test_remainder["y_true"].values <= upper)
                            ).mean()
                            width = (upper - lower).mean()

                            rows.append({
                                "tier": tier,
                                "target": target_short,
                                "seed": seed,
                                "ood_family": family,
                                "cal_size": cal_size,
                                "alpha": alpha,
                                "coverage": float(covered),
                                "mean_width": float(width),
                            })

    return pd.DataFrame(rows)


def build_bias_check_table(config: dict) -> pd.DataFrame:
    """Compare Tier 1 on full set vs. Voronoi subset."""
    rows = []
    targets = config["evaluation"]["targets"]

    for target in targets:
        target_short = "ef" if target == "formation_energy_per_atom" else "bg"

        for seed in config["evaluation"]["seeds"]:
            for tier_label, label in [("tier1", "tier1_full"), ("tier1sub", "tier1_voronoi_subset")]:
                files = list(PREDICTIONS_DIR.glob(
                    f"{tier_label}_standard_seed{seed}_{target_short}_test.parquet"
                ))
                if not files:
                    continue
                df = pd.read_parquet(files[0])
                metrics = compute_metrics(df["y_true"].values, df["y_pred"].values)
                rows.append({
                    "variant": label,
                    "target": target_short,
                    "seed": seed,
                    **metrics,
                })

    return pd.DataFrame(rows)


def main() -> None:
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    benchmark = build_benchmark_table(config)
    benchmark.to_csv(TABLES_DIR / "benchmark.csv", index=False)
    logger.info("Wrote benchmark.csv (%d rows)", len(benchmark))

    domain_shift = build_domain_shift_table(config)
    domain_shift.to_csv(TABLES_DIR / "domain_shift.csv", index=False)
    logger.info("Wrote domain_shift.csv (%d rows)", len(domain_shift))

    conformal = build_conformal_table(config)
    conformal.to_csv(TABLES_DIR / "conformal_coverage.csv", index=False)
    logger.info("Wrote conformal_coverage.csv (%d rows)", len(conformal))

    cal_sweep = build_calibration_sweep_table(config)
    cal_sweep.to_csv(TABLES_DIR / "calibration_sweep.csv", index=False)
    logger.info("Wrote calibration_sweep.csv (%d rows)", len(cal_sweep))

    bias_check = build_bias_check_table(config)
    bias_check.to_csv(TABLES_DIR / "bias_check.csv", index=False)
    logger.info("Wrote bias_check.csv (%d rows)", len(bias_check))


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/run_evaluation.py
git commit -m "feat: add evaluation script producing benchmark, domain-shift, conformal, and bias-check tables"
```

---

## Block 4: Explainability

### Task 17: SHAP + Failure Cases (TDD)

**Files:**
- Create: `src/crystal_prop_bench/evaluation/explainability.py`
- Create: `tests/test_explainability.py`

**Step 1: Write failing tests**

```python
# tests/test_explainability.py
import numpy as np
import pytest


class TestSHAPExplainer:
    def test_shap_values_shape(self):
        """SHAP values should match input feature shape."""
        from crystal_prop_bench.evaluation.explainability import compute_shap_values

        # Train a tiny model
        from crystal_prop_bench.models.lgbm_baseline import train_lgbm

        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = X[:, 0] * 2 + rng.randn(100) * 0.1
        model, _ = train_lgbm(X[:80], y[:80], X[80:], y[80:], seed=42)

        shap_values = compute_shap_values(model, X[:10])
        assert shap_values.shape == (10, 5)

    def test_feature_importance_ranking(self):
        """Most important feature should be feature 0 (it determines y)."""
        from crystal_prop_bench.evaluation.explainability import (
            compute_shap_values,
            global_feature_importance,
        )
        from crystal_prop_bench.models.lgbm_baseline import train_lgbm

        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        y = X[:, 0] * 5 + rng.randn(200) * 0.01
        model, _ = train_lgbm(X[:160], y[:160], X[160:], y[160:], seed=42)

        shap_values = compute_shap_values(model, X[:50])
        importance = global_feature_importance(shap_values, ["f0", "f1", "f2", "f3", "f4"])
        assert importance[0][0] == "f0"  # first element is most important


class TestFailureCases:
    def test_extracts_worst_predictions(self):
        from crystal_prop_bench.evaluation.explainability import extract_failure_cases

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 10.0])  # last one is worst
        ids = np.array(["a", "b", "c", "d", "e"])

        failures = extract_failure_cases(y_true, y_pred, ids, n=2)
        assert len(failures) == 2
        assert failures.iloc[0]["material_id"] == "e"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_explainability.py -v
```

**Step 3: Implement explainability module**

```python
# src/crystal_prop_bench/evaluation/explainability.py
"""SHAP explainability and failure-case analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def compute_shap_values(
    model: object,
    X: np.ndarray,
) -> np.ndarray:
    """Compute SHAP values using TreeExplainer.

    Returns array of shape (n_samples, n_features).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return np.array(shap_values)


def global_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> list[tuple[str, float]]:
    """Rank features by mean |SHAP value|.

    Returns list of (feature_name, mean_abs_shap) sorted descending.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    ranked = sorted(
        zip(feature_names, mean_abs),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked


def extract_failure_cases(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    material_ids: np.ndarray,
    n: int = 50,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extract the N worst-predicted crystals.

    Returns DataFrame sorted by absolute error descending.
    """
    errors = np.abs(y_true - y_pred)
    indices = np.argsort(errors)[::-1][:n]

    result = pd.DataFrame({
        "material_id": material_ids[indices],
        "y_true": y_true[indices],
        "y_pred": y_pred[indices],
        "abs_error": errors[indices],
    })

    if metadata is not None:
        result = result.merge(metadata, on="material_id", how="left")

    return result.reset_index(drop=True)
```

**Step 4: Run tests**

```bash
pytest tests/test_explainability.py -v
```

Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add src/crystal_prop_bench/evaluation/explainability.py tests/test_explainability.py
git commit -m "feat: add SHAP explainability and failure-case extraction"
```

---

### Task 18: SHAP Script

**Files:**
- Create: `scripts/run_shap.py`

**Step 1: Implement SHAP script**

```python
# scripts/run_shap.py
"""SHAP analysis on Tier 1 and Tier 2 LightGBM models.

Reads models from results/models/ and predictions from results/predictions/.
Writes SHAP data to results/tables/ for plotting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from crystal_prop_bench.data.featurizers import compute_magpie_features
from crystal_prop_bench.data.mp_adapter import MPAdapter
from crystal_prop_bench.evaluation.explainability import (
    compute_shap_values,
    extract_failure_cases,
    global_feature_importance,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("results/models")
PREDICTIONS_DIR = Path("results/predictions")
TABLES_DIR = Path("results/tables")


def main() -> None:
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data for feature names and metadata
    adapter = MPAdapter(cache_dir=Path("data/mp"))
    df = adapter.load()
    features = compute_magpie_features(
        df, cache_path=Path("data/mp/magpie_features.parquet")
    )
    feature_cols = [c for c in features.columns if c != "material_id"]

    # Use seed 42, standard split, formation energy as representative
    seed = config["evaluation"]["seeds"][0]

    for tier in ["tier1", "tier2"]:
        model_path = MODELS_DIR / f"{tier}_standard_seed{seed}_ef.joblib"
        if not model_path.exists():
            logger.warning("Model not found: %s", model_path)
            continue

        model = joblib.load(model_path)

        # Load test predictions
        pred_files = list(PREDICTIONS_DIR.glob(f"{tier}_standard_seed{seed}_ef_test.parquet"))
        if not pred_files:
            continue
        pred_df = pd.read_parquet(pred_files[0])

        # Get features for test set
        test_features = features[features["material_id"].isin(pred_df["material_id"])]
        test_features = test_features.merge(
            pred_df[["material_id"]], on="material_id"
        )

        X_test = test_features[feature_cols].values

        # Compute SHAP (subsample for speed if large)
        n_shap = min(len(X_test), 5000)
        shap_values = compute_shap_values(model, X_test[:n_shap])

        # Global importance
        importance = global_feature_importance(shap_values, feature_cols)
        importance_df = pd.DataFrame(importance, columns=["feature", "mean_abs_shap"])
        importance_df["tier"] = tier
        importance_df.to_csv(
            TABLES_DIR / f"shap_importance_{tier}.csv", index=False
        )
        logger.info("Top 10 %s features: %s", tier, importance[:10])

        # Per-family SHAP comparison
        test_with_family = test_features.merge(
            pred_df[["material_id", "chemistry_family"]], on="material_id"
        )
        for family in test_with_family["chemistry_family"].unique():
            mask = test_with_family["chemistry_family"].values[:n_shap] == family
            if mask.sum() > 0:
                family_importance = global_feature_importance(
                    shap_values[mask], feature_cols
                )
                family_df = pd.DataFrame(
                    family_importance, columns=["feature", "mean_abs_shap"]
                )
                family_df["tier"] = tier
                family_df["family"] = family
                family_df.to_csv(
                    TABLES_DIR / f"shap_importance_{tier}_{family}.csv", index=False
                )

        # Failure cases
        failures = extract_failure_cases(
            pred_df["y_true"].values,
            pred_df["y_pred"].values,
            pred_df["material_id"].values,
            n=50,
            metadata=df[["material_id", "chemistry_family", "spacegroup_number", "nsites"]],
        )
        failures["tier"] = tier
        failures.to_csv(TABLES_DIR / f"failure_cases_{tier}.csv", index=False)
        logger.info(
            "%s failure cases — family distribution: %s",
            tier,
            failures["chemistry_family"].value_counts().to_dict(),
        )

    # Cross-tier failure overlap
    tier1_failures = TABLES_DIR / "failure_cases_tier1.csv"
    tier2_failures = TABLES_DIR / "failure_cases_tier2.csv"
    if tier1_failures.exists() and tier2_failures.exists():
        t1 = pd.read_csv(tier1_failures)
        t2 = pd.read_csv(tier2_failures)
        overlap = set(t1["material_id"]) & set(t2["material_id"])
        logger.info(
            "Cross-tier failure overlap: %d/%d (%.1f%%)",
            len(overlap),
            min(len(t1), len(t2)),
            100.0 * len(overlap) / min(len(t1), len(t2)) if min(len(t1), len(t2)) > 0 else 0,
        )


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/run_shap.py
git commit -m "feat: add SHAP analysis script with per-family comparison and failure cases"
```

---

## Block 5: Visualization, Docs & CI

### Task 19: Visualization Module + Script

**Files:**
- Create: `src/crystal_prop_bench/visualization/plots.py`
- Create: `scripts/run_plots.py`

**Step 1: Implement plots.py**

```python
# src/crystal_prop_bench/visualization/plots.py
"""Plotting functions. Read CSVs, write PNGs. No model or evaluation logic."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_domain_shift_bars(
    domain_shift_csv: Path,
    output_path: Path,
) -> None:
    """Bar chart: MAE per family, ID vs each OOD family."""
    df = pd.read_csv(domain_shift_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, target in zip(axes, ["ef", "bg"]):
        target_df = df[df["target"] == target]
        # Average over seeds
        agg = target_df.groupby("ood_family").agg(
            id_mae=("id_mae", "mean"),
            ood_mae=("ood_mae", "mean"),
        ).reset_index()

        x = np.arange(len(agg))
        width = 0.35
        ax.bar(x - width / 2, agg["id_mae"], width, label="ID (oxide)", color="#4C72B0")
        ax.bar(x + width / 2, agg["ood_mae"], width, label="OOD", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(agg["ood_family"])
        ax.set_ylabel("MAE")
        ax.set_title(f"{'Formation Energy' if target == 'ef' else 'Band Gap'}")
        ax.legend()

    fig.suptitle("Domain-Shift Degradation")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_conformal_coverage(
    conformal_csv: Path,
    output_path: Path,
) -> None:
    """Coverage vs interval width at multiple alpha levels."""
    df = pd.read_csv(conformal_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, target in zip(axes, ["ef", "bg"]):
        target_df = df[df["target"] == target]
        for tier in target_df["tier"].unique():
            tier_df = target_df[target_df["tier"] == tier]
            agg = tier_df.groupby("alpha").agg(
                coverage=("coverage", "mean"),
                width=("mean_width", "mean"),
            ).reset_index()
            ax.plot(agg["width"], agg["coverage"], "o-", label=tier)

        ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="90% target")
        ax.set_xlabel("Mean Interval Width")
        ax.set_ylabel("Empirical Coverage")
        ax.set_title(f"{'Formation Energy' if target == 'ef' else 'Band Gap'}")
        ax.legend()

    fig.suptitle("Conformal Coverage vs. Interval Width")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_sweep(
    sweep_csv: Path,
    output_path: Path,
) -> None:
    """Coverage vs calibration budget per OOD family — the standout figure."""
    df = pd.read_csv(sweep_csv)

    families = df["ood_family"].unique()
    targets = df["target"].unique()

    fig, axes = plt.subplots(
        len(targets), len(families),
        figsize=(5 * len(families), 4 * len(targets)),
        squeeze=False,
    )

    for i, target in enumerate(targets):
        for j, family in enumerate(families):
            ax = axes[i][j]
            subset = df[(df["target"] == target) & (df["ood_family"] == family)]

            for alpha in sorted(subset["alpha"].unique()):
                alpha_df = subset[subset["alpha"] == alpha]
                agg = alpha_df.groupby("cal_size")["coverage"].agg(["mean", "std"]).reset_index()
                ax.errorbar(
                    agg["cal_size"], agg["mean"], yerr=agg["std"],
                    fmt="o-", label=f"alpha={alpha:.2f}", capsize=3,
                )
                ax.axhline(y=1 - alpha, color="gray", linestyle="--", alpha=0.3)

            ax.set_xlabel("Calibration Set Size")
            ax.set_ylabel("Empirical Coverage")
            ax.set_title(f"{family.title()} — {'Ef' if target == 'ef' else 'Bg'}")
            ax.legend(fontsize=8)

    fig.suptitle("Coverage vs. Calibration Budget (OOD Families)", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shap_summary(
    shap_csv_pattern: str,
    tables_dir: Path,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Horizontal bar chart of top-N global SHAP importance per tier."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, tier in zip(axes, ["tier1", "tier2"]):
        csv_path = tables_dir / f"shap_importance_{tier}.csv"
        if not csv_path.exists():
            ax.set_title(f"{tier} — not available")
            continue

        df = pd.read_csv(csv_path)
        top = df.nlargest(top_n, "mean_abs_shap")

        ax.barh(range(len(top)), top["mean_abs_shap"].values, color="#4C72B0")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{'Composition (Tier 1)' if tier == 'tier1' else 'Structure (Tier 2)'}")

    fig.suptitle("Global Feature Importance (SHAP)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

**Step 2: Implement run_plots.py**

```python
# scripts/run_plots.py
"""Generate all figures from result CSVs.

Reads from results/tables/, writes to results/figures/.
Can be re-run independently to tweak aesthetics.
"""

from __future__ import annotations

import logging
from pathlib import Path

from crystal_prop_bench.visualization.plots import (
    plot_calibration_sweep,
    plot_conformal_coverage,
    plot_domain_shift_bars,
    plot_shap_summary,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if (TABLES_DIR / "domain_shift.csv").exists():
        plot_domain_shift_bars(
            TABLES_DIR / "domain_shift.csv",
            FIGURES_DIR / "domain_shift_bars.png",
        )
        logger.info("Wrote domain_shift_bars.png")

    if (TABLES_DIR / "conformal_coverage.csv").exists():
        plot_conformal_coverage(
            TABLES_DIR / "conformal_coverage.csv",
            FIGURES_DIR / "conformal_coverage.png",
        )
        logger.info("Wrote conformal_coverage.png")

    if (TABLES_DIR / "calibration_sweep.csv").exists():
        plot_calibration_sweep(
            TABLES_DIR / "calibration_sweep.csv",
            FIGURES_DIR / "calibration_sweep.png",
        )
        logger.info("Wrote calibration_sweep.png")

    plot_shap_summary(
        "shap_importance_*.csv",
        TABLES_DIR,
        FIGURES_DIR / "shap_summary.png",
    )
    logger.info("Wrote shap_summary.png")


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add src/crystal_prop_bench/visualization/plots.py scripts/run_plots.py
git commit -m "feat: add visualization module and plotting script"
```

---

### Task 20: Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

This test uses the pre-computed fixtures to run the full pipeline
on 100 crystals without network access.

```python
# tests/test_integration.py
"""End-to-end integration test on 100-crystal fixture.

Runs: load fixture -> featurize (from cache) -> split -> train -> predict
-> conformal -> metrics. All on CPU, no network, uses pre-computed features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crystal_prop_bench.data.splits import standard_split
from crystal_prop_bench.evaluation.conformal import evaluate_conformal_coverage
from crystal_prop_bench.evaluation.metrics import aggregate_seeds, compute_metrics
from crystal_prop_bench.models.lgbm_baseline import train_lgbm


class TestIntegration:
    def test_full_pipeline_tier1(
        self,
        fixture_crystals: pd.DataFrame,
        fixture_magpie_features: pd.DataFrame,
    ) -> None:
        """Full Tier 1 pipeline on 100-crystal fixture."""
        df = fixture_crystals
        features = fixture_magpie_features
        target = "formation_energy_per_atom"

        # Split
        train_df, cal_df, test_df = standard_split(df, seed=42)

        # Align features
        feature_cols = [c for c in features.columns if c != "material_id"]

        train_m = train_df.merge(features, on="material_id")
        cal_m = cal_df.merge(features, on="material_id")
        test_m = test_df.merge(features, on="material_id")

        assert len(train_m) > 0
        assert len(cal_m) > 0
        assert len(test_m) > 0

        # Train
        model, cal_residuals = train_lgbm(
            train_m[feature_cols].values,
            train_m[target].values,
            cal_m[feature_cols].values,
            cal_m[target].values,
            seed=42,
        )

        # Predict
        test_preds = model.predict(test_m[feature_cols].values)
        assert test_preds.shape == (len(test_m),)

        # Metrics
        metrics = compute_metrics(test_m[target].values, test_preds)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0

        # Conformal
        coverage_results = evaluate_conformal_coverage(
            cal_residuals,
            test_m[target].values,
            test_preds,
            alphas=[0.10, 0.20],
        )
        assert len(coverage_results) == 2
        for cr in coverage_results:
            assert 0.0 <= cr["coverage"] <= 1.0
            assert cr["mean_width"] >= 0.0

    def test_seed_aggregation(
        self,
        fixture_crystals: pd.DataFrame,
        fixture_magpie_features: pd.DataFrame,
    ) -> None:
        """Verify seed aggregation works across multiple seeds."""
        df = fixture_crystals
        features = fixture_magpie_features
        target = "formation_energy_per_atom"
        feature_cols = [c for c in features.columns if c != "material_id"]

        seed_results = []
        for seed in [42, 123]:
            train_df, cal_df, test_df = standard_split(df, seed=seed)
            train_m = train_df.merge(features, on="material_id")
            cal_m = cal_df.merge(features, on="material_id")
            test_m = test_df.merge(features, on="material_id")

            model, _ = train_lgbm(
                train_m[feature_cols].values, train_m[target].values,
                cal_m[feature_cols].values, cal_m[target].values,
                seed=seed,
            )
            preds = model.predict(test_m[feature_cols].values)
            seed_results.append(compute_metrics(test_m[target].values, preds))

        agg = aggregate_seeds(seed_results)
        assert "mae_mean" in agg
        assert "mae_std" in agg
        assert agg["mae_mean"] > 0
```

**Step 2: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add integration test for full pipeline on fixture"
```

---

### Task 21: Regression Gate

**Files:**
- Create: `scripts/check_regression.py`
- Create: `tests/fixtures/regression_thresholds.json`

**Step 1: Implement regression gate**

```python
# scripts/check_regression.py
"""CI regression gate: train on fixture, check MAE <= threshold.

Catches broken pipelines, not hyperparameter drift.
Threshold is set generously (~2x initial MAE).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

from crystal_prop_bench.data.splits import standard_split
from crystal_prop_bench.evaluation.metrics import compute_metrics
from crystal_prop_bench.models.lgbm_baseline import train_lgbm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIXTURES_DIR = Path("tests/fixtures")
THRESHOLDS_PATH = FIXTURES_DIR / "regression_thresholds.json"


def main() -> int:
    df = pd.read_parquet(FIXTURES_DIR / "fixture_crystals.parquet")
    features = pd.read_parquet(FIXTURES_DIR / "fixture_magpie_features.parquet")

    with open(THRESHOLDS_PATH) as f:
        thresholds = json.load(f)

    feature_cols = [c for c in features.columns if c != "material_id"]
    failures = []

    for target, threshold in thresholds.items():
        train_df, cal_df, test_df = standard_split(df, seed=42)

        train_m = train_df.merge(features, on="material_id")
        cal_m = cal_df.merge(features, on="material_id")
        test_m = test_df.merge(features, on="material_id")

        model, _ = train_lgbm(
            train_m[feature_cols].values, train_m[target].values,
            cal_m[feature_cols].values, cal_m[target].values,
            seed=42,
        )
        preds = model.predict(test_m[feature_cols].values)
        metrics = compute_metrics(test_m[target].values, preds)

        if metrics["mae"] > threshold:
            failures.append(
                f"{target}: MAE={metrics['mae']:.4f} > threshold={threshold}"
            )
            logger.error("REGRESSION: %s", failures[-1])
        else:
            logger.info(
                "PASS: %s MAE=%.4f <= %.4f", target, metrics["mae"], threshold
            )

    if failures:
        logger.error("Regression gate FAILED:\n%s", "\n".join(failures))
        return 1

    logger.info("Regression gate PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Note:** `regression_thresholds.json` is created after the first successful run
of the fixture pipeline. Initial content (placeholder to be updated):

```json
{
    "formation_energy_per_atom": 2.0,
    "band_gap": 3.0
}
```

Set generous initial thresholds; tighten after observing actual fixture MAE.

**Step 2: Commit**

```bash
git add scripts/check_regression.py
git commit -m "feat: add regression gate for CI"
```

---

### Task 22: CI Pipeline

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Create CI config**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: mypy src/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -x --ignore=tests/test_integration.py -m "not network"

  integration:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/test_integration.py -x

  regression-gate:
    runs-on: ubuntu-latest
    needs: integration
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: python scripts/check_regression.py
```

**Step 2: Commit**

```bash
mkdir -p .github/workflows
git add .github/workflows/ci.yml
git commit -m "ci: add lint, test, integration, and regression gate pipeline"
```

---

### Task 23: Dockerfile

**Files:**
- Create: `Dockerfile`

**Step 1: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY Makefile .

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python"]
CMD ["scripts/run_tier1.py"]
```

**Step 2: Commit**

```bash
git add Dockerfile
git commit -m "chore: add minimal Dockerfile for reproducibility"
```

---

### Task 24: DECISIONS.md

**Files:**
- Create: `DECISIONS.md`

**Step 1: Write DECISIONS.md**

```markdown
# Design Decisions

This document records the rationale for every non-obvious technical
decision in crystal-prop-bench. Each entry explains what was chosen,
what alternatives were considered, and why.

---

## 1. Why Materials Project (not OQMD, AFLOW, Alexandria, or JARVIS)

Materials Project offers the largest freely accessible collection of
DFT-computed crystal properties (~150K) with a well-maintained Python
API (`mp-api`). OQMD and AFLOW have comparable coverage but less
ergonomic programmatic access. JARVIS is deferred to Bonus A because
cross-database generalization requires polymorph matching (same
composition can map to different crystal structures), which is a
non-trivial data-engineering task that would delay the MVP.

## 2. Why formation energy + band gap

Formation energy per atom is the most widely predicted property in
materials ML benchmarks, enabling direct comparison with published
results. Band gap adds a property with different physical character —
it depends on electronic structure (geometry-sensitive) while formation
energy is largely composition-determined. This contrast enables Finding 2.

## 3. Chemistry-family classification: 80% anion-purity threshold

Crystals are classified by dominant anion: oxide, sulfide, nitride, or
halide. The 80% purity threshold (fraction of anion sites belonging to
one family) filters out mixed-anion compounds (oxysulfides, oxynitrides)
that would confound domain-shift analysis. The threshold was chosen to
balance coverage (keeping most crystals) against purity (avoiding
ambiguous classifications).

**Dropped compounds:** [TO BE FILLED after data download — report count
and percentage per family]

## 4. Why Magpie descriptors

Magpie (Materials-Agnostic Platform for Informatics and Exploration)
provides ~150 composition-based descriptors computed from elemental
property statistics. These are interpretable, fast to compute, and
provide a strong baseline. More expressive learned representations
(e.g., Roost, CrabNet) would obscure the composition-vs-structure
comparison that is central to this benchmark.

## 5. Why Voronoi over fixed-radius neighbor lists

Voronoi tessellation partitions space around atoms without requiring a
distance cutoff hyperparameter. Fixed-radius methods (e.g., 8 Angstrom
cutoff) introduce an arbitrary choice that affects coordination number
statistics. Voronoi is parameter-free and produces topologically
consistent neighbor assignments.

Trade-off: Voronoi tessellation can fail on pathological structures
(overlapping atoms, extreme cell shapes). We accept this and document
the failure rate per chemistry family (see Decision 8).

## 6. Why split conformal regression (not APS, not CQR)

Split conformal regression provides distribution-free coverage
guarantees with minimal assumptions. APS (Adaptive Prediction Sets)
applies to classification, not regression. CQR (Conformalized Quantile
Regression) produces heteroscedastic intervals but requires training a
quantile model — added complexity that is deferred to Bonus C.

Split conformal intervals are fixed-width (same q_hat for all test
points), which is a known limitation. The calibration sweep experiment
(Decision 14) partially addresses this by showing how interval quality
varies across OOD families.

## 7. Split strategy: stratified random with frozen fixture

Standard 80/10/10 split stratified by chemistry family ensures each
split contains all families in proportion. Frozen test fixture (100
crystals) enables CI regression testing without network access.

Domain-shift split trains on oxides only and tests on other families.
This is the realistic deployment scenario: calibrate on available
chemistry, encounter new chemistry.

## 8. Featurization failure handling: drop, report, bias-check

Structures that fail Voronoi tessellation are dropped rather than
imputed. Imputation (e.g., filling with medians) would introduce noise
in exactly the hard cases and mask the failure. Dropping is transparent.

To detect whether dropping introduces selection bias, Tier 1 (Magpie,
which never fails) is evaluated on both the full dataset and the
Voronoi-survivable subset. If the MAE difference is negligible, the
drop-and-report strategy is validated.

## 9. Why 3 seeds

Three seeds provide mean +/- std estimates of model performance. This
is the convention across the portfolio (diffusion-physics, sim-to-data,
finetune-bench, demandops-lite). At the scale of Materials Project
(~150K crystals), variance across seeds is small, so 3 seeds suffice
without being wasteful.

## 10. Why LightGBM over XGBoost or random forest

LightGBM is faster than XGBoost on datasets of this size (leaf-wise
growth vs. level-wise), supports native categorical features, and
integrates with SHAP's TreeExplainer for exact Shapley values. Random
forest is slower and produces less interpretable feature importances.
The LightGBM + SHAP combination is well-established in materials
informatics literature.

## 11. DatasetAdapter ABC with 2 abstract methods

Template-method pattern with `load_raw()` and `cache_path()` as the
only abstract methods. Concrete `load()` on the ABC handles chemistry
classification, schema validation, filtering, and caching. This is the
portfolio convention (finetune-bench, demandops-lite).

Chemistry classification is a standalone function, not an abstract
method, because the 80% anion-purity rule is domain logic independent
of data source.

## 12. MLflow as development tool, flat files as public interface

MLflow logs params and metrics during iteration. Results also written
to `results/tables/*.csv` and `results/figures/*.png`. A reviewer
cloning the repo sees results by opening files, not by running
`mlflow ui`.

## 13. Prediction parquet as interchange format

Training scripts save predictions (material_id, y_true, y_pred,
chemistry_family, split) to parquet. Evaluation scripts read these
files. This enables re-running evaluation without retraining and
establishes a clean seam between the training and evaluation stages.

## 14. Calibration sweep for deployable UQ finding

Beyond showing that conformal coverage breaks under domain shift
(Finding 4), we sweep calibration set size [5, 10, 25, 50, 100] per
OOD family. This answers the practical question: how many DFT
calculations on a new chemistry family does a scientist need before
conformal intervals become reliable?

This curve is the most deployable finding in the repo and is not
reported in standard materials ML benchmarks.
```

**Step 2: Commit**

```bash
git add DECISIONS.md
git commit -m "docs: add DECISIONS.md with rationale for all 14 design choices"
```

---

### Task 25: MODEL_CARD.md

**Files:**
- Create: `MODEL_CARD.md`

**Step 1: Write MODEL_CARD.md**

```markdown
# Model Card: crystal-prop-bench

Following Mitchell et al. (2019) model card framework.

## Model Details

- **Model type:** LightGBM gradient boosted trees (Tier 1: composition features; Tier 2: composition + structure features)
- **Training data:** Materials Project (~150K crystals with DFT-computed properties)
- **Targets:** Formation energy per atom (eV/atom), band gap (eV)
- **Features:** Tier 1 uses ~150 Magpie composition descriptors; Tier 2 adds ~100-150 structural descriptors (Voronoi tessellation)
- **Framework:** LightGBM 4.3+, matminer 0.9+, pymatgen 2024.2+

## Intended Use

- Benchmarking materials property prediction under domain shift
- Evaluating uncertainty quantification methods on crystal property data
- Baseline comparison for GNN-based crystal property predictors
- NOT intended as a production property predictor for materials discovery

## Training Data

- Source: Materials Project (Creative Commons Attribution 4.0)
- Size: ~150K crystals after filtering
- Chemistry families: oxide, sulfide, nitride, halide (80% anion-purity threshold)
- Properties: DFT-computed (PBE-GGA functional)

## Evaluation

- Standard 80/10/10 split (3 seeds)
- Domain-shift split: train on oxides, test on sulfides/nitrides/halides
- Metrics: MAE, RMSE, R² (overall and per chemistry family)
- Uncertainty: Split conformal regression intervals
- Explainability: SHAP TreeExplainer

## Performance

[TO BE FILLED after running experiments]

## Limitations

- Trained on DFT-computed properties, not experimental measurements
- Domain shift to chemistries outside {oxide, sulfide, nitride, halide} is untested
- Conformal coverage guarantee is marginal, not conditional on chemistry family
- Voronoi featurization fails on some structures (see DECISIONS.md #8)
- No generative model evaluation in Phase A

## Ethical Considerations

- Training data is publicly available under permissive license
- No personal or sensitive data involved
- Predictions should not replace experimental validation in materials discovery
```

**Step 2: Commit**

```bash
git add MODEL_CARD.md
git commit -m "docs: add MODEL_CARD.md following Mitchell et al. 2019"
```

---

### Task 26: README.md

**Files:**
- Create: `README.md`

**Step 1: Write README.md**

```markdown
# crystal-prop-bench

Materials property prediction with calibrated uncertainty and chemical domain-shift evaluation.

## Motivation

AI-driven materials discovery depends on property predictors that are accurate,
well-calibrated, and honest about their limitations. Most materials ML benchmarks
report accuracy on random test splits — they don't ask what happens when the model
encounters a chemistry it wasn't trained on.

This benchmark evaluates tabular models (composition-only and structure-aware) on
Materials Project crystals, with a focus on:

1. **Domain-shift degradation** — how much does prediction quality degrade when
   moving from oxides (training domain) to sulfides, nitrides, and halides?
2. **Calibrated uncertainty** — do conformal prediction intervals maintain their
   coverage guarantees under chemistry shift?
3. **Calibration efficiency** — how many samples from a new chemistry family
   does a scientist need before uncertainty estimates become reliable?

## Key Findings

[TO BE FILLED after experiments]

1. **Composition baseline strength:** [result]
2. **Structure helps band gap more than formation energy:** [result]
3. **Domain-shift degradation pattern:** [result]
4. **Conditional coverage breaks under shift:** [result]
5. **Mixed training as domain randomization:** [result]
6. **Calibration efficiency curve:** [result]

## Benchmark Results

[TO BE FILLED — Tier x Split x Target x MAE +/- std]

## Domain-Shift Analysis

[TO BE FILLED — figure: per-family degradation bars]

## Uncertainty Quantification

[TO BE FILLED — figure: calibration sweep (coverage vs. budget)]

## Explainability

[TO BE FILLED — figure: SHAP summary]

## Connection to Conditional Crystal Generation

Property predictors play four roles inside a conditional crystal generation pipeline:

- **Conditioning oracle.** A generator (e.g., CDVAE, DiffCSP) conditions on target
  properties like band gap = 1.5 eV. The property predictor validates whether
  generated candidates actually hit those targets, closing the loop between
  generation and evaluation.

- **Validity filter.** Physically unreasonable predictions (e.g., negative band gap,
  formation energy far outside the training distribution) serve as a structural
  validity proxy — flagging generated structures that are likely unphysical before
  expensive DFT validation.

- **Ranking function.** In a discovery campaign generating thousands of candidates,
  the predictor ranks them by predicted proximity to target properties. This
  prioritization determines which candidates proceed to synthesis or simulation.

- **Selective prediction for triage.** Conformal prediction intervals identify which
  candidates the predictor is confident about (narrow intervals → route to synthesis)
  versus uncertain about (wide intervals → route to DFT validation first). The
  calibration efficiency curve from this benchmark directly informs how many DFT
  calculations are needed to trust the predictor on a new chemistry.

## Relationship to Other Benchmarks

This is part of a cross-portfolio methodological arc:

| Repo | Domain | Shared methodology |
|------|--------|--------------------|
| diffusion-physics | PDE surrogates | Conformal coverage, multi-regime eval |
| sim-to-data | Ultrasonic inspection | Domain shift, selective prediction |
| finetune-bench | Multimodal NLP | Ablation tiers, DatasetAdapter, model card |
| demandops-lite | Demand forecasting | Multi-dataset adapter, LightGBM baseline |
| **crystal-prop-bench** | **Materials science** | **All of the above** |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Download Materials Project data (requires MP_API_KEY)
export MP_API_KEY=your_key_here
make download-data

# Run Tier 1 (composition features)
make run-tier1

# Run Tier 2 (structural features)
make run-tier2

# Run evaluation
make run-evaluation

# Run SHAP analysis
make run-shap

# Generate figures
make run-plots

# Or run everything
make run-all
```

## Limitations

- **Materials Project only.** No cross-database generalization (JARVIS, OQMD)
  in this version.
- **Tabular models only.** GNN evaluation (CGCNN) planned for Phase B.
- **DFT properties, not experimental.** All target values are computed, not measured.
- **Four chemistry families.** Domain shift is evaluated across oxide/sulfide/
  nitride/halide — other chemistries are filtered out.
- **Marginal coverage only.** Conformal guarantee is marginal, not conditional
  on chemistry family (this is a finding, not a limitation to hide).

## License

MIT
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with motivation, findings placeholders, and quick start"
```

---

### Task 27: Run Fixture Creation and Set Thresholds

**Depends on:** All previous tasks. This is the final setup step.

**Step 1: Run fixture creation**

```bash
export MP_API_KEY=your_key_here
python scripts/create_fixture.py
```

**Step 2: Update regression thresholds**

After running the fixture pipeline, check the actual MAE values and set
thresholds at ~2x:

```bash
python -c "
import pandas as pd
from crystal_prop_bench.data.splits import standard_split
from crystal_prop_bench.models.lgbm_baseline import train_lgbm
from crystal_prop_bench.evaluation.metrics import compute_metrics
import json

df = pd.read_parquet('tests/fixtures/fixture_crystals.parquet')
features = pd.read_parquet('tests/fixtures/fixture_magpie_features.parquet')
feature_cols = [c for c in features.columns if c != 'material_id']
thresholds = {}

for target in ['formation_energy_per_atom', 'band_gap']:
    train, cal, test = standard_split(df, seed=42)
    train_m = train.merge(features, on='material_id')
    cal_m = cal.merge(features, on='material_id')
    test_m = test.merge(features, on='material_id')
    model, _ = train_lgbm(
        train_m[feature_cols].values, train_m[target].values,
        cal_m[feature_cols].values, cal_m[target].values, seed=42,
    )
    preds = model.predict(test_m[feature_cols].values)
    metrics = compute_metrics(test_m[target].values, preds)
    thresholds[target] = round(metrics['mae'] * 2, 4)
    print(f'{target}: MAE={metrics[\"mae\"]:.4f}, threshold={thresholds[target]}')

with open('tests/fixtures/regression_thresholds.json', 'w') as f:
    json.dump(thresholds, f, indent=2)
print('Thresholds saved.')
"
```

**Step 3: Commit fixtures and thresholds**

```bash
git add tests/fixtures/
git commit -m "feat: add 100-crystal test fixture with regression thresholds"
```

---

## Execution Order Summary

```
Block 1: Data Foundation
  Task 1  → Init repo
  Task 2  → Chemistry classifier (TDD)
  Task 3  → DatasetAdapter ABC
  Task 4  → Pandera schemas (TDD)
  Task 5  → MPAdapter (TDD)
  Task 6  → Splits (TDD)
  Task 7  → Fixture creation script
  Task 8  → conftest.py fixture loaders

Block 2: Featurizers & Models
  Task 9  → Magpie featurizer (TDD)
  Task 10 → Voronoi featurizer (TDD)
  Task 11 → LightGBM training (TDD)
  Task 12 → Metrics module (TDD)
  Task 13 → Training scripts

Block 3: UQ & Domain Shift
  Task 14 → Conformal regression (TDD)
  Task 15 → Domain shift analysis (TDD)
  Task 16 → Evaluation script

Block 4: Explainability
  Task 17 → SHAP + failure cases (TDD)
  Task 18 → SHAP script

Block 5: Visualization, Docs & CI
  Task 19 → Plots + run_plots.py
  Task 20 → Integration test
  Task 21 → Regression gate
  Task 22 → CI pipeline
  Task 23 → Dockerfile
  Task 24 → DECISIONS.md
  Task 25 → MODEL_CARD.md
  Task 26 → README.md
  Task 27 → Fixture creation + threshold setting (requires MP_API_KEY)
```

**Parallelizable tasks within blocks:**
- Block 1: Tasks 2, 4 can run in parallel once Task 1 is done
- Block 2: Tasks 9, 10 can run in parallel; Task 11, 12 can run in parallel
- Block 3: Tasks 14, 15 can run in parallel
- Block 5: Tasks 20-26 can run in parallel after Block 4
