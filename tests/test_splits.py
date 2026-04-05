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
