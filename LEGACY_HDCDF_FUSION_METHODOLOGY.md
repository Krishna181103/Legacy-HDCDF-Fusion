# Legacy-HDCDF Fusion Methodology

This project-specific model is designed to pursue higher HyperLeaf2024 validation accuracy without falsely claiming an exact reproduction of the HDCDF paper.

## Motivation

The existing `BEST_HSI_HYBRID_CENTER` model reaches 87.74% validation accuracy, while the pure HDCDF-inspired model underfits and collapses toward one class on early CPU runs. Therefore, the strongest research-safe direction is to retain the proven project backbone and inject selected HDCDF methodology where it is least disruptive.

## Architecture

Input: `48 x 48 x 24 x 1` PCA-reduced hyperspectral crop.

1. Legacy multi-kernel 3D stem:
   - `3 x 3 x 3`, `3 x 3 x 5`, and `1 x 1 x 7` 3D convolutions.
   - Captures local spatial-spectral features at multiple spectral receptive fields.

2. Legacy 3D residual backbone:
   - 3D convolution + batch normalization + Mish.
   - Two residual stages with spatial/spectral pooling.
   - 3D squeeze-excitation preserves the successful baseline inductive bias.

3. HDCDF-inspired 2D refinement:
   - Reshape spectral depth into 2D channels.
   - Apply 2D-DCAttention: depth-wise convolution, BN, Mish, channel attention, spatial attention, point-wise convolution.
   - Apply depth-wise separable 2D residual refinement.

4. Classifier:
   - Global average pooling.
   - Dense layers with Mish and dropout.
   - Softmax cultivar prediction.

## Research Integrity

This is not a direct HDCDF reproduction. It is a new hybrid methodology combining:

- Existing HyperLeaf best 3D PCA backbone.
- HDCDF-style Mish activation.
- HDCDF-style 2D-DCAttention.
- Depth-wise separable 2D refinement.

Report it as `Legacy-HDCDF Fusion`, not as the original HDCDF paper model.

## Recommended Runs

Smoke test:

```bash
python3 training/train_best_hsi_model.py --model-variant legacy_hdcdf_fusion --loss focal --skip-submission --center-only --epoch-budget 1 --run-tag legacy_hdcdf_fusion_focal_smoke
```

Main run:

```bash
python3 training/train_best_hsi_model.py --model-variant legacy_hdcdf_fusion --loss focal --skip-submission --center-only --epoch-budget 60 --run-tag legacy_hdcdf_fusion_focal
```

Comparison baseline:

```bash
python3 training/train_best_hsi_model.py --model-variant legacy_best --eval-only --skip-submission --center-only --run-tag best_hsi_hybrid
```
