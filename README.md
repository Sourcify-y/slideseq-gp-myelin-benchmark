# Spatial Modeling of Myelin-Pathway Gene Expression in Mouse Hippocampus

Code and results for a benchmark of exact and sparse variational Gaussian process (GP) regression on a Slide-seqV2 mouse hippocampus section (41,786 beads, 4,000 genes). Eight genes are modeled: Mbp, Plp1, Mobp, Mog, Cnp and Sox10 (myelin and oligodendrocyte-lineage genes), and Snap25 and Malat1 (comparison genes). Moran's I, k-nearest-neighbor (kNN) and inverse-distance-weighted (IDW) interpolation, and SPARK-X serve as reference methods.

## Data

The Slide-seqV2 object is loaded with `squidpy.datasets.slideseqv2()`. Expression values are log-normalized and coordinates are two-dimensional bead positions. SPARK-X on raw counts uses puck Puck_200115_08 from the Broad Single Cell Portal. The two raw-puck files are not included in this repository.

## Environment

Python 3.11 with the packages in `requirements.txt`. The code runs on CPU and uses CUDA when it is available. SPARK-X requires R and the [SPARK](https://github.com/xzhoulab/SPARK) package.

The global seed is 42. Block partitions use seeds 42 to 46, and permutations use a seed offset of 10,000.

## Design

**Exact GP.** Five non-overlapping 2,000-cell subsamples per gene. Each subsample is split into 40 spatial blocks (k-means on coordinates) and 20% of the blocks are held out for testing. The remaining cells are re-blocked into 32 blocks, and 20% of those blocks form the validation set. RBF, Matérn 1.5 and Matérn 2.5 kernels, each with a white-noise term, are fitted on the training blocks. The kernel with the lowest validation RMSE is refitted on training and validation cells and scored on the test blocks.

**SVGP.** RBF kernel with a length scale per coordinate axis, Gaussian likelihood, Cholesky variational distribution. 400 inducing points are initialized by k-means on the training coordinates and learned during training. Adam with learning rate 0.01, batch size 1,024, 150 epochs. Convergence is checked from the relative change in epoch loss over the last 20 epochs (tolerance 1%).

**Permutation tests.** Exact GP: labels are shuffled within each subsample, and kernel selection and refitting are repeated in every permutation. The statistic is the mean test R² over the five splits, compared with one empirical null of 100 permutations. SVGP: the training labels are permuted and the model is retrained from scratch for each of 50 permutations on one spatially blocked split of the full data. Benjamini-Hochberg correction is applied across the eight genes within each test.

**Baselines.** kNN (k = 10, with a sweep over k = 10, 30, 100, 300) and IDW (power 2) on the same splits as the GP models. R² is computed on expression standardized by the training mean and standard deviation.

## Scripts

| Script | Analysis | Output |
|---|---|---|
| `pipeline.py` | Exact GP, SVGP, Moran's I, baselines and cross-method maps for the eight genes | `results/gp_results_*` |
| `permutation_tests.py` | Permutation tests with full refitting for exact GP and SVGP | `results/perm` |
| `baseline_rerun.py` | Exact GP, kNN and IDW on the five independent subsamples, with predictive interval coverage and width | `results/baselines` |
| `revision_summaries.py` | Kernel selection counts, hyperparameter ranges, calibration coverage and width across all splits | `results/revision` |
| `uncertainty_scores.py` | Interval score and CRPS of the exact GP against kNN and IDW with constant standard deviation | `results/uncertainty` |
| `matched_comparison.py` | Exact GP and SVGP at matched subsample sizes (2,000, 4,000, 6,000 cells); SVGP on the full data at 10, 20, 40, 80, 150 and 300 blocks | `results/matched` |
| `fixedtest_density.py` | SVGP with a fixed test set and a varying number of training cells | `results/fixedtest` |
| `fulldata_baselines.py` | kNN (k = 10) and IDW on the full-data SVGP designs | `results/fullbase` (`base_*`) |
| `knn_ksweep.py` | kNN with k = 10, 30, 100, 300 on the full-data designs | `results/fullbase` (`knnk_*`) |
| `inducing_sensitivity.py` | SVGP with 200, 800 and 1,600 inducing points | `results/inducing` |
| `null_control.py` | SVGP, IDW and kNN on expression permuted across cells | `results/null` |
| `lowstructure_controls.py` | Eight control genes selected by Moran's I and detection rate | `results/lowstructure` |
| `moran_all_genes.py` | Moran's I for the eight genes and its relation to SVGP R² | `results/moran` |
| `sensitivity_maps.py` | Predicted maps across subsamples, block partitions, inducing-point counts and seeds | `results/sensitivity` |
| `sensitivity_summary_all.py` | Map correlations for all genes | `results/sensitivity/summary_all.csv` |
| `descriptive_tables.py` | Gene detection statistics, block sizes, overlap of predicted maps with annotation labels | `results/revision` |
| `sparkx_export_full.py`, `sparkx_run_full.R` | SPARK-X on counts recovered from the squidpy matrix (4,000 genes) | `results/sparkx` |
| `sparkx_raw/` | SPARK-X on raw counts (14,613 genes) | `results/sparkx_raw` |

Each `run_*.sh` wrapper runs one analysis for all genes.

## Reproduction

```
pip install -r requirements.txt
python3 pipeline.py
sh run_perms.sh
sh run_baselines.sh
sh run_matched.sh
sh run_blocks.sh
sh run_fullbase.sh
sh run_knnk.sh
sh run_fixedtest.sh
sh run_inducing.sh
sh run_null.sh
sh run_unc.sh
sh run_sens.sh
python3 lowstructure_controls.py select
python3 moran_all_genes.py run
python3 descriptive_tables.py gene
python3 descriptive_tables.py blocks
python3 descriptive_tables.py labels
```

Summaries are written by the `summary` subcommand of each script, for example `python3 permutation_tests.py summary`, and by `python3 revision_summaries.py kernels`, `hyper` and `calibration`. `python3 sensitivity_summary_all.py` summarizes the map comparisons.

The SVGP with 400 inducing points takes about 108 s per fit on a laptop CPU at 40 blocks. With 200, 800 and 1,600 inducing points the times are 60 s, 346 s and 1,857 s.

## SPARK-X on raw counts

1. Download `Puck_200115_08.digital_expression.txt.gz` and `Puck_200115_08_bead_locations.csv` from the Broad Single Cell Portal.
2. Build the count matrix and compare its barcodes with the squidpy object:
   `python3 sparkx_raw/01_build_raw_counts.py --dge Puck_200115_08.digital_expression.txt.gz --locs Puck_200115_08_bead_locations.csv --outdir export`
   The overlap is written to `export/match_report.txt`. All 41,786 squidpy beads are present in the puck, which has 53,208 beads.
3. Run SPARK-X on genes detected in at least 0.1% of beads:
   `Rscript sparkx_raw/02_run_sparkx.R export results/sparkx_raw/sparkx_results.csv 4`

`sparkx_export_full.py` and `sparkx_run_full.R` repeat the analysis on counts recovered from the log-normalized squidpy matrix. The export stops if fewer than 99.9% of the rescaled values lie within 0.01 of an integer.

## Results folders

Outputs behind the tables and figures of the paper. Rerunning the scripts reproduces them up to numerical differences between software builds.

| Folder | Contents |
|---|---|
| `results/perm` | Permutation draws and summaries for exact GP and SVGP |
| `results/baselines` | Per-split exact GP, kNN and IDW results; accuracy and calibration summaries |
| `results/revision` | Kernel counts, hyperparameters, calibration coverage and width, block and gene descriptives, label overlap |
| `results/uncertainty` | Interval score and CRPS per gene and split |
| `results/matched` | Exact GP and SVGP at matched sizes and block designs |
| `results/fixedtest` | SVGP R² by number of training cells |
| `results/fullbase` | kNN and IDW on the full-data designs, kNN k sweep |
| `results/inducing` | SVGP R² by number of inducing points |
| `results/null` | Results on expression permuted across cells |
| `results/lowstructure` | Control gene selection and results, Moran's I of all 4,000 genes |
| `results/moran` | Moran's I of the eight genes and its relation to R² |
| `results/sensitivity` | Fits and predicted maps per gene, map correlation summaries |
| `results/sparkx` | SPARK-X on recovered counts (4,000 genes) |
| `results/sparkx_raw` | SPARK-X on raw counts (14,613 genes) and the barcode match report |

## License

See `LICENSE`.
