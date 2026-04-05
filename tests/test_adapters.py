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
        doc.structure = "mock_structure"  # picklable stand-in
        return doc

    @patch("crystal_prop_bench.data.mp_adapter.MPRester")
    def test_load_raw_returns_dataframe(
        self, mock_rester_cls: MagicMock, tmp_path: Path,
    ) -> None:
        mock_ctx = MagicMock()
        mock_rester_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_rester_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_ctx.materials.summary.search.return_value = [
            self._make_mock_doc("mp-1", "Fe2O3", -1.5, 2.0, 10, 167),
            self._make_mock_doc("mp-2", "ZnS", -0.8, 3.5, 4, 216),
        ]

        adapter = MPAdapter(api_key="fake_key", cache_dir=tmp_path / "mp")
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
    def test_load_raw_correct_values(
        self, mock_rester_cls: MagicMock, tmp_path: Path,
    ) -> None:
        mock_ctx = MagicMock()
        mock_rester_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_rester_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_ctx.materials.summary.search.return_value = [
            self._make_mock_doc("mp-1", "Fe2O3", -1.5, 2.0, 10, 167),
        ]

        adapter = MPAdapter(api_key="fake_key", cache_dir=tmp_path / "mp")
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
