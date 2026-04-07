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

**Dropped compounds:** 90,757 of 200,487 crystals (45.3%) from the
Materials Project API query are dropped because they lack a dominant
anion family at 80% purity (mixed-anion compounds, metals, intermetallics,
etc.). The 109,730 classified crystals break down as: oxide 76,867 (70.1%),
halide 16,540 (15.1%), sulfide 9,629 (8.8%), nitride 6,694 (6.1%).

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

Tier 2 subsamples oxides from ~77K to 25K (stratified random, seed=42)
while keeping all minority families (sulfide, nitride, halide) intact.
Total Tier 2 dataset: ~58K structures. Rationale:

- **LightGBM convergence is unchanged at this scale.** Gradient-boosted
  trees plateau well below 15K training samples for tabular features.
- **Minority families are never subsampled.** These are the OOD test
  sets; subsampling would weaken the domain-shift evaluation.
- **Tier 1 (Magpie) runs on the full 110K dataset.** Only Tier 2
  is subsampled, so the composition-only baseline remains unaffected.

## 16. Stratified domain-shift split to reduce R² variance

The original `domain_shift_split` did not stratify oxide partitions by
target value — all samples were oxides, so chemistry-family
stratification (used in `standard_split`) was meaningless. This caused
high seed-to-seed R² variance for Tier 2 domain-shift formation energy:
**R² = 0.837 ± 0.103** (per-seed ID MAE ranged from 0.117 to 0.150).

**Root cause:** With only ~15K oxides going through a 70/10/10/10 split,
different seeds placed easy or hard oxides into very different
partitions. R² is especially sensitive because its denominator is total
y-variance in the test set.

**Fix applied in two stages:**

1. **Option A — Stratify by target quartile.** Added `stratify_col`
   parameter to `domain_shift_split`. Bins the target column into
   quartiles via `pd.qcut` and passes them as `stratify=` to all three
   `train_test_split` calls. Falls back to unstratified if any bin has
   fewer than 2 members (handles small fixtures). This is exactly what
   `standard_split` already did via chemistry-family stratification.

   **Result:** R² std dropped from **0.103 → 0.048** (53% reduction).
   MAE std halved (0.017 → 0.007). Improvement was significant but
   R² std remained above the ~0.03 target.

2. **Option B — Increase oxide subsample from 15K to 25K.** Since
   stratification alone did not fully stabilize R², the oxide subsample
   was expanded. Voronoi features for the additional ~10K oxides were
   computed incrementally and merged into the existing cache. Larger
   training and test sets mechanically reduce seed sensitivity.

**Final measured results (Tier 2 domshift formation energy, test_id):**

| Stage                          | R² std | MAE std |
|--------------------------------|--------|---------|
| Original (unstratified, 15K)   | 0.103  | 0.017   |
| + Option A (stratified, 15K)   | 0.048  | 0.007   |
| + Option B (stratified, 25K)   | 0.040  | 0.002   |

R² std reduced 61% overall. MAE std collapsed to 0.002, matching
Tier 1 stability. Remaining R² spread (0.040) is driven by squared-error
sensitivity to a few outlier crystals in one seed's test partition —
the MAE stability confirms the model itself is stable across seeds.

**Alternative considered and rejected:** Adding more seeds (5 instead
of 3) was rejected because it averages over the same broken split
rather than fixing the partition-quality problem.

## 17. Why ALIGNN over vanilla CGCNN

ALIGNN (Atomistic Line Graph Neural Network) adds bond-angle
information via a line graph on top of the standard atom graph.
This is physically motivated: band gap depends on electronic
structure (geometry-sensitive), and angular features capture
coordination geometry more robustly than Voronoi hand-crafted
statistics or the distance-only edge features in CGCNN.

ALIGNN connects to the equivariant architecture family
(MACE, NequIP) used in production crystal generation pipelines.
Vanilla CGCNN (2018) would demonstrate "I learned the baseline"
rather than "I understand the geometric deep learning progression."

The `alignn` package from NIST/JARVIS provides a battle-tested
implementation with published benchmark numbers for comparison.

## 18. Why 8.0 Angstrom cutoff with 12 nearest neighbors

Standard in crystal GNN literature (Xie & Grossman 2018, Choudhary
& DeCost 2021). The 8.0 A radius captures second-shell neighbors for
most crystal structures. Capping at 12 neighbors controls graph
density and keeps memory usage predictable across structures with
varying coordination environments.

## 19. Why mean pooling for intensive properties

Formation energy per atom and band gap are intensive properties
(independent of system size). Sum pooling would create a spurious
correlation with the number of atoms, biasing predictions for
larger unit cells. Mean pooling is the standard choice for
intensive property prediction in crystal GNNs.

## 20. Hybrid implementation: import architecture, own pipeline

The benchmark's contribution is evaluating a GNN under domain shift
with conformal prediction, not reimplementing message passing.
Importing the ALIGNN model class from the `alignn` package while
writing our own graph construction, training loop, and prediction
export keeps the evaluation seam clean: training writes prediction
parquets, evaluation reads them — same contract as Tiers 1-2.

## 21. A100 40GB with batch_size=128 for ALIGNN training

Crystal graphs built with an 8.0 A cutoff and 12 nearest neighbors
are dense — large unit cells produce graphs with thousands of edges.
When batched, the combined graph's edge tensors dominate GPU memory.

**Tested and rejected:**
- A10G (24GB) with batch_size=128: OOM during forward pass (19.5GB
  allocated, needed 2.25GB more for edge update).
- A10G (24GB) with batch_size=32: fits, but ~4x slower per epoch.
  Estimated 8-10 hours for 18 training runs at ~$9-11 total.
- T4 (16GB): OOM at any practical batch size for the full dataset.
- A100 80GB: fits batch_size=256, but the 50% higher hourly rate
  vs 40GB yields no meaningful speedup (compute-bound, not
  memory-bound at batch_size=128).

**Chosen: A100 40GB with batch_size=128.** Estimated 2-3 hours for
18 training runs at ~$6-9 total. Best cost/performance ratio:
the 40GB headroom comfortably fits batch_size=128, and A100's
higher memory bandwidth (1.5 TB/s vs 600 GB/s on A10G) accelerates
the message-passing kernels that dominate ALIGNN training.
