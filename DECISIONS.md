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
is the convention across the portfolio (laplace-uq-bench, sim-to-data,
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

## 15. Oxide subsampling for Tier 2 Voronoi featurization

Voronoi tessellation + SiteStatsFingerprint(VoronoiNN) takes ~3 seconds
per structure on a single core. At 110K structures, serial featurization
would require ~80 hours. Even with 8-core parallelism, the full set
takes ~10 hours.

Tier 2 subsamples oxides from ~77K to 15K (stratified random, seed=42)
while keeping all minority families (sulfide, nitride, halide) intact.
Total Tier 2 dataset: ~48K structures. Rationale:

- **LightGBM convergence is unchanged at this scale.** Gradient-boosted
  trees plateau well below 15K training samples for tabular features.
- **Minority families are never subsampled.** These are the OOD test
  sets; subsampling would weaken the domain-shift evaluation.
- **Tier 1 (Magpie) runs on the full 110K dataset.** Only Tier 2
  is subsampled, so the composition-only baseline remains unaffected.
