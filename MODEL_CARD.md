# Model Card: crystal-prop-bench

Following Mitchell et al. (2019) model card framework.

## Model Details

- **Model type:** Three-tier benchmark
  - Tier 1: LightGBM with ~150 Magpie composition descriptors
  - Tier 2: LightGBM with composition + ~100-150 Voronoi structural descriptors
  - Tier 3: ALIGNN (Atomistic Line Graph Neural Network) with 3+3 layers
- **Training data:** Materials Project (~47K crystals after oxide subsampling)
- **Targets:** Formation energy per atom (eV/atom), band gap (eV)
- **Framework:** LightGBM 4.3+, PyTorch 2.4+, DGL 2.4+, alignn 2024.5+, matminer 0.9+, pymatgen 2024.2+

## Intended Use

- Benchmarking materials property prediction under domain shift
- Evaluating uncertainty quantification methods on crystal property data
- Comparing tabular vs. graph neural network approaches under distribution shift
- NOT intended as a production property predictor for materials discovery

## Training Data

- Source: Materials Project (Creative Commons Attribution 4.0)
- Size: ~47K crystals after filtering and oxide subsampling
- Chemistry families: oxide, sulfide, nitride, halide (80% anion-purity threshold)
- Properties: DFT-computed (PBE-GGA functional)

## Evaluation

- Standard 70/10/10/10 split (Tiers 1-2: 3 seeds; Tier 3: 1 seed)
- Domain-shift split: train on oxides, test on sulfides/nitrides/halides
- Mixed-train split: train on all families, test per family
- Metrics: MAE, RMSE, R² (overall and per chemistry family)
- Uncertainty: Split conformal regression intervals
- Explainability: SHAP TreeExplainer (Tiers 1-2)

## Performance

### Standard Split

| Tier | Formation Energy MAE | Band Gap MAE |
|------|---------------------|-------------|
| Tier 1 (Magpie) | 0.124 eV/atom | 0.514 eV |
| Tier 2 (Voronoi) | 0.105 eV/atom | 0.440 eV |
| Tier 3 (ALIGNN) | 0.051 eV/atom | 0.337 eV |

### Domain-Shift (Formation Energy)

| Tier | ID (oxide) MAE | OOD sulfide | OOD nitride | OOD halide |
|------|---------------|-------------|-------------|------------|
| Tier 1 | 0.123 | 0.280 (2.3x) | 0.578 (4.7x) | 0.754 (6.1x) |
| Tier 2 | 0.129 | 0.274 (2.1x) | 0.798 (6.2x) | 0.682 (5.3x) |
| Tier 3 | 0.059 | 0.918 (15.7x) | 0.379 (6.5x) | 0.262 (4.5x) |

Tier 2 (Voronoi) shows a similar degradation pattern to Tier 1 — modest on
sulfides (2.1x), severe on nitrides (6.2x) and halides (5.3x). Tier 3 inverts
this: it beats both Tier 1 and Tier 2 OOD on nitrides and halides but
catastrophically fails on sulfides (15.7x) due to a systematic energy bias of
+0.897 eV/atom. The GNN learns oxide-specific energy corrections that
anti-transfer to sulfides, while composition-only features avoid this failure
mode (see Finding 10).

## Limitations

- Trained on DFT-computed properties, not experimental measurements
- Domain shift to chemistries outside {oxide, sulfide, nitride, halide} is untested
- Conformal coverage guarantee is marginal, not conditional on chemistry family
- Voronoi featurization fails on some structures (see DECISIONS.md #8)
- Tier 3 uses a single seed due to GPU training cost (~$70 for 6 runs on A100 80GB)
- Tier 3 uses reduced-depth ALIGNN (3+3 layers vs. published 4+4; see DECISIONS.md #22)
- ALIGNN sulfide predictions show systematic bias — practitioners deploying on new
  chemistry classes should verify prediction distributions before trusting outputs

## Ethical Considerations

- Training data is publicly available under permissive license
- No personal or sensitive data involved
- Predictions should not replace experimental validation in materials discovery
