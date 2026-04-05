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
