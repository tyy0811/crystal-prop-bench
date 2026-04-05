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
