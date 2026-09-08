# Spatial Modeling of Myelination-Associated Gene Expression in Mouse Brain Tissue

Pipeline for the study *"Spatial Modeling of Myelination-Associated Gene Expression in Mouse Brain Tissue: A Multi-Gene Benchmark of Exact and Sparse Variational Gaussian Process Regression."*

## Overview
Benchmarks exact and sparse variational Gaussian Process (GP) regression for modeling spatial gene expression in mouse brain spatial transcriptomics data, alongside Moran's I and standard interpolation baselines.

## Dataset
Uses the Slide-seqV2 mouse brain dataset, accessed through Squidpy. Downloaded automatically when the pipeline runs; not stored in this repository.

## Methods
- Exact GP regression on repeated 2,000-cell spatial subsamples
- Sparse variational GP (SVGP) regression on the full dataset
- RBF and Matérn kernel comparison
- Spatially blocked train/validation/test splitting
- k-nearest-neighbors and inverse-distance-weighting baselines
- Moran's I spatial autocorrelation
- Spatial permutation testing, Benjamini-Hochberg FDR correction
- Bootstrap confidence intervals
- Cross-method spatial prediction comparison and calibration checks

## Requirements
Python 3.11. Package versions in `requirements.txt`.

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the full pipeline
```bash
python veritasai.py
```
Runs all 8 genes through both the exact-GP and SVGP pipelines, including robustness checks, permutation testing, and cross-method comparison. Takes roughly 2–3 hours on standard CPU hardware. For a faster check, see Quick reproduction below.

## Quick reproduction (5-10 min)
To verify a single result without running the full pipeline, edit `TARGET_GENES` in `veritasai.py` to include only one gene:
```python
TARGET_GENES = {"Mbp": "myelin marker (primary target)"}
```
Then run `python veritasai.py`. This reproduces the Mbp row of Table 2 (SVGP R² ≈ 0.428, exact-GP R² ≈ 0.040) in a few minutes.

**Note:** run from the repository root — the script writes output to a local `results/` directory using a relative path, so it will fail if that directory can't be created from wherever you run it.

## Output
- `results/gp_results_exactGP_summary.csv`
- `results/gp_results_SVGP_summary.csv`
- `results/gp_results_crossmethod_summary.csv`
- `results/gp_results_combined_significance_summary.csv`
- Per-gene PNG figures (SVGP loss curves, cross-method comparison plots)
