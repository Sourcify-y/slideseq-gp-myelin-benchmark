# Spatial Modeling of Myelination-Associated Gene Expression in Mouse Brain Tissue

Computational pipeline for:

**Spatial Modeling of Myelination-Associated Gene Expression in Mouse Brain Tissue: A Multi-Gene Benchmark of Exact and Sparse Variational Gaussian Process Regression**

## Overview

This repository contains the complete analysis pipeline used in the study.

The pipeline evaluates Gaussian Process regression for modeling spatial gene expression in mouse brain spatial transcriptomics data using exact GP and sparse variational GP approaches.

## Dataset

The analysis uses the Slide-seqV2 mouse brain dataset, accessed through Squidpy.

The dataset is downloaded automatically when the pipeline is run and is not stored in this repository.

## Methods

The pipeline includes:

- Exact Gaussian Process regression on repeated 2,000-cell spatial subsamples
- Sparse Variational Gaussian Process regression on the full dataset
- RBF and Matérn kernel comparison
- Spatially blocked train/validation/test splitting
- k-nearest-neighbors and inverse-distance-weighting baselines
- Moran's I spatial autocorrelation analysis
- Spatial permutation testing
- Benjamini-Hochberg FDR correction
- Bootstrap confidence intervals
- Cross-method spatial prediction comparison
- Predictive uncertainty calibration

## Requirements

- Python 3.11

Package versions are provided in `requirements.txt`.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
