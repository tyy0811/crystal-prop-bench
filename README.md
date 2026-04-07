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

1. **Composition baseline strength.** Magpie + LightGBM achieves MAE = 0.122 eV/atom
   on formation energy (standard split) — confirming that composition alone explains
   most of the variance in DFT formation energies. Band gap is harder: MAE = 0.52 eV.

2. **Structure helps on standard splits, hurts under domain shift.** On the standard
   split, Tier 2 (Voronoi) improves formation energy MAE from 0.122 to 0.105 and
   band gap from 0.520 to 0.443. But under domain shift, Tier 2 *degrades*: band gap
   MAE rises to 0.574 (vs. Tier 1's 0.533), suggesting structural features overfit
   to oxide geometry and transfer poorly to other chemistries.

3. **Domain-shift degradation pattern.** Formation energy degrades severely under
   chemistry shift: MAE ratios of 2.3× (sulfide), 4.7× (nitride), and 6.0× (halide)
   relative to in-distribution oxide performance. Band gap degrades more moderately
   (1.2–2.4×), suggesting composition features transfer better for electronic properties.

4. **Conditional coverage breaks under shift.** Conformal prediction maintains ~90%
   marginal coverage on in-distribution oxides but collapses on OOD families.
   Formation energy coverage drops to 54% (sulfide), 24% (nitride), 27% (halide).
   Band gap coverage degrades less sharply: 89% (sulfide), 82% (nitride), 60% (halide).
   This is the central UQ finding — marginal guarantees do not imply conditional safety.

5. **Mixed training as domain randomization.** Training on all chemistry families
   recovers 70–86% of the formation energy degradation and 46–58% for band gap.
   Halides benefit most (86% recovery on formation energy), consistent with
   domain-randomization effects seen in sim-to-data.

6. **Calibration efficiency curve.** Despite the coverage collapse in Finding 4,
   conformal intervals can be *recalibrated* with surprisingly few OOD samples.
   Using 25 OOD calibration points restores coverage to 95–99% across all
   chemistry families for both targets (at alpha = 0.10). Even 10 samples
   recover >90% coverage for most families. At 5 samples, coverage is
   unreliable (std up to ±31%), making this the practical lower bound.
   As calibration sets grow to 50–100 samples, coverage settles to the
   nominal 90% level — the initial overcoverage at 10–25 samples reflects
   the conservatism of quantile estimates from small samples. **Actionable
   recommendation: a scientist deploying this predictor on a new chemistry
   needs ~25 DFT-validated samples before trusting the uncertainty estimates.**

7. **Failure modes are nearly disjoint between tiers.** The 50 worst-predicted
   crystals share only **4% overlap** (2/50) between Tier 1 and Tier 2. Tier 1
   failures concentrate in oxides (43/50), while Tier 2 failures distribute
   across families (18 oxide, 13 halide, 11 sulfide, 8 nitride). The tiers
   fail on different crystals for different structural reasons — an ensemble
   combining both could reduce worst-case errors substantially.

## Benchmark Results

| Tier | Split | Target | MAE | R² |
|------|-------|--------|-----|-----|
| Tier 1 (Magpie) | Standard | Formation Energy | 0.122 ± 0.002 eV/atom | 0.883 ± 0.003 |
| Tier 1 (Magpie) | Standard | Band Gap | 0.520 ± 0.005 eV | 0.770 ± 0.006 |
| Tier 1 (Magpie) | Domain-Shift (ID) | Formation Energy | 0.123 ± 0.004 eV/atom | 0.798 ± 0.023 |
| Tier 1 (Magpie) | Domain-Shift (ID) | Band Gap | 0.533 ± 0.003 eV | 0.746 ± 0.005 |
| Tier 2 (Voronoi) | Standard | Formation Energy | 0.105 ± 0.001 eV/atom | 0.949 ± 0.006 |
| Tier 2 (Voronoi) | Standard | Band Gap | 0.443 ± 0.005 eV | 0.830 ± 0.004 |
| Tier 2 (Voronoi) | Domain-Shift (ID) | Formation Energy | 0.130 ± 0.002 eV/atom | 0.834 ± 0.040 |
| Tier 2 (Voronoi) | Domain-Shift (ID) | Band Gap | 0.574 ± 0.016 eV | 0.741 ± 0.004 |

## Domain-Shift Analysis

Models trained on oxides degrade substantially on other chemistry families.
Formation energy is most affected — halides show 6.0× MAE degradation:

| Target | OOD Family | ID MAE | OOD MAE | Degradation |
|--------|-----------|--------|---------|-------------|
| Formation Energy | Sulfide | 0.123 | 0.279 | 2.3× |
| Formation Energy | Nitride | 0.123 | 0.580 | 4.7× |
| Formation Energy | Halide | 0.123 | 0.735 | 6.0× |
| Band Gap | Sulfide | 0.533 | 0.626 | 1.2× |
| Band Gap | Nitride | 0.533 | 0.769 | 1.4× |
| Band Gap | Halide | 0.533 | 1.297 | 2.4× |

![Domain-shift degradation](results/figures/domain_shift_bars.png)

## Uncertainty Quantification

Split conformal regression maintains ~90% coverage on in-distribution oxides
but collapses under chemistry shift. At alpha=0.10 (target 90% coverage):

| Target | Test Set | Coverage |
|--------|----------|----------|
| Formation Energy | ID (oxide) | 89.9% |
| Formation Energy | OOD sulfide | 54.0% |
| Formation Energy | OOD nitride | 24.4% |
| Formation Energy | OOD halide | 27.2% |
| Band Gap | ID (oxide) | 90.7% |
| Band Gap | OOD sulfide | 88.6% |
| Band Gap | OOD nitride | 81.9% |
| Band Gap | OOD halide | 59.6% |

Marginal conformal guarantees do not imply conditional safety under domain shift.

![Conformal coverage](results/figures/conformal_coverage.png)

![Calibration sweep](results/figures/calibration_sweep.png)

## Explainability

SHAP analysis reveals that electronegativity features dominate both tiers:
`MagpieData mode Electronegativity` is the top feature for Tier 1 and Tier 2.
In Tier 2, Voronoi coordination number standard deviation (`std_dev CN_VoronoiNN`)
ranks 7th, confirming that structural features contribute but don't dominate.

**The tiers fail on nearly different crystals** (only 4% overlap in the 50
worst-predicted — see Finding 7 above).

![SHAP summary](results/figures/shap_summary.png)

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
| laplace-uq-bench | PDE surrogates | Conformal coverage, multi-regime eval |
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
