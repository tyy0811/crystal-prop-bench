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
