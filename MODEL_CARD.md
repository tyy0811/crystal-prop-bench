# Model Card: crystal-prop-bench

Following Mitchell et al. (2019) model card framework.

## Model Details

- **Model type:** LightGBM gradient boosted trees (Tier 1: composition features; Tier 2: composition + structure features)
- **Training data:** Materials Project (~150K crystals with DFT-computed properties)
- **Targets:** Formation energy per atom (eV/atom), band gap (eV)
- **Features:** Tier 1 uses ~150 Magpie composition descriptors; Tier 2 adds ~100-150 structural descriptors (Voronoi tessellation)
- **Framework:** LightGBM 4.3+, matminer 0.9+, pymatgen 2024.2+

## Intended Use

- Benchmarking materials property prediction under domain shift
- Evaluating uncertainty quantification methods on crystal property data
- Baseline comparison for GNN-based crystal property predictors
- NOT intended as a production property predictor for materials discovery

## Training Data

- Source: Materials Project (Creative Commons Attribution 4.0)
- Size: ~150K crystals after filtering
- Chemistry families: oxide, sulfide, nitride, halide (80% anion-purity threshold)
- Properties: DFT-computed (PBE-GGA functional)

## Evaluation

- Standard 80/10/10 split (3 seeds)
- Domain-shift split: train on oxides, test on sulfides/nitrides/halides
- Metrics: MAE, RMSE, R² (overall and per chemistry family)
- Uncertainty: Split conformal regression intervals
- Explainability: SHAP TreeExplainer

## Performance

[TO BE FILLED after running experiments]

## Limitations

- Trained on DFT-computed properties, not experimental measurements
- Domain shift to chemistries outside {oxide, sulfide, nitride, halide} is untested
- Conformal coverage guarantee is marginal, not conditional on chemistry family
- Voronoi featurization fails on some structures (see DECISIONS.md #8)
- No generative model evaluation in Phase A

## Ethical Considerations

- Training data is publicly available under permissive license
- No personal or sensitive data involved
- Predictions should not replace experimental validation in materials discovery
