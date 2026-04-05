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
