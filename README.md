# Spatial Modeling of Myelination-Associated Gene Expression in Mouse Brain Tissue
Pipeline for the study *"Spatial Modeling of Myelination-Associated Gene Expression in Mouse Brain Tissue: A Multi-Gene Benchmark of Exact and Sparse Variational Gaussian Process Regression."*
## Overview
Benchmarks exact and sparse variational Gaussian Process (GP) regression for modeling spatial gene expression in mouse brain spatial transcriptomics data, alongside Moran's I, SPARK-X, and standard interpolation baselines.
## Dataset
Uses the Slide-seqV2 mouse brain dataset, accessed through Squidpy. Downloaded automatically when the pipeline runs; not stored in this repository.
## Methods
- Exact GP regression on repeated 2,000-cell spatial subsamples
- Sparse variational GP (SVGP) regression on the full dataset
- RBF and Matérn kernel comparison
- Spatially blocked train/validation/test splitting
- k-nearest-neighbors (fixed and tuned k) and inverse-distance-weighting baselines
- Moran's I spatial autocorrelation and SPARK-X
- Permutation tests with full model refit, Benjamini-Hochberg FDR correction
- Bootstrap confidence intervals
- Calibration, interval score, and CRPS
- Cross-method spatial prediction comparison
- Sensitivity analyses (block count, inducing points, subsamples, seeds), permuted-expression control, and low-structure control genes
## Requirements
Python 3.11. Package versions in `requirements.txt`. SPARK-X requires R with the [SPARK](https://github.com/xzhoulab/SPARK) package.
## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Running the full pipeline
```bash
python pipeline.py
```
Runs all 8 genes through both the exact-GP and SVGP pipelines, including robustness checks, permutation testing, and cross-method comparison. Takes roughly 2-3 hours on standard CPU hardware. The permutation p-values and FDR columns written by `pipeline.py` are superseded by the revision analyses below. For a faster check, see Quick reproduction below.
## Quick reproduction (5-10 min)
To verify a single result without running the full pipeline, edit `TARGET_GENES` in `pipeline.py` to include only one gene:
```python
TARGET_GENES = {"Mbp": "myelin marker (primary target)"}
```
Then run `python pipeline.py`. This reproduces the Mbp SVGP R-squared of roughly 0.428 in a few minutes.
**Note:** run from the repository root - the script creates a local `results/` directory for output, and will fail if that directory can't be created from wherever you run it.
## Revision analyses
Scripts run from the repository root and write to `results/`. The `run_*.sh` wrappers run the multi-gene, multi-split sweeps.
- `permutation_tests.py` (`run_perms.sh`): permutation tests with full refit, exact GP and SVGP (`results/perm`)
- `matched_comparison.py` (`run_matched.sh`, `run_blocks.sh`): exact GP vs SVGP at matched n and across block counts (`results/matched`)
- `fixedtest_density.py` (`run_fixedtest.sh`): SVGP R-squared against training density on fixed test blocks (`results/fixedtest`)
- `inducing_sensitivity.py` (`run_inducing.sh`): SVGP sensitivity to the number of inducing points (`results/inducing`)
- `baseline_rerun.py` (`run_baselines.sh`): exact GP, kNN and IDW baselines for all genes (`results/baselines`)
- `fulldata_baselines.py` (`run_fullbase.sh`): kNN and IDW on the full-data SVGP designs (`results/fullbase`)
- `knn_ksweep.py` (`run_knnk.sh`): kNN with tuned k (`results/fullbase`)
- `uncertainty_scores.py` (`run_unc.sh`): interval score and CRPS for GP, kNN and IDW (`results/uncertainty`)
- `moran_all_genes.py`: Moran's I for all genes (`results/moran`)
- `null_control.py` (`run_null.sh`): permuted-expression control (`results/null`)
- `sensitivity_maps.py` (`run_sens.sh`): map stability across subsamples, block partitions, inducing points and seeds (`results/sensitivity`)
- `lowstructure_controls.py`: control genes with low Moran's I (`results/lowstructure`)
- `revision_summaries.py`, `descriptive_tables.py`: kernel counts, hyperparameters, calibration, gene and block descriptives (`results/revision`)
- `sparkx_export_full.py`, `sparkx_run_full.R`: SPARK-X on recovered counts, all genes (`results/sparkx`)

SPARK-X:
```bash
python3 sparkx_export_full.py
Rscript sparkx_run_full.R
```
## Output
- `results/gp_results_exactGP_summary.csv`
- `results/gp_results_SVGP_summary.csv`
- `results/gp_results_crossmethod_summary.csv`
- `results/gp_results_combined_significance_summary.csv` (permutation p-values superseded by `results/perm`)
- Per-gene PNG figures (SVGP loss curves, cross-method comparison plots)
- Revision outputs in the `results/` subfolders listed above
## License
MIT License. See `LICENSE` for details.
