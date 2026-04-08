# tests/test_featurizers.py
import pandas as pd

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


from crystal_prop_bench.data.featurizers import compute_voronoi_features  # noqa: E402


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
        df = pd.DataFrame({
            "material_id": ["good-1", "bad-1"],
            "formula_pretty": ["Fe2O3", "ZnS"],
            "chemistry_family": ["oxide", "sulfide"],
        })
        # Provide a valid structure for good-1 and a broken one for bad-1
        from pymatgen.core import Lattice, Structure
        good_struct = Structure(
            Lattice.cubic(3.0),
            ["Fe", "Fe", "O", "O", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        )
        structures = {"good-1": good_struct}  # bad-1 missing → dropped

        result = compute_voronoi_features(df, structures)
        assert len(result) == 1
        assert result.iloc[0]["material_id"] == "good-1"

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
