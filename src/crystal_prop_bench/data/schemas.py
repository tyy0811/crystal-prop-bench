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
