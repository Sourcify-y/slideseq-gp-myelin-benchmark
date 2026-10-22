"""
Export recovered counts for all genes and cells of the squidpy Slide-seqV2 object for SPARK-X.

Counts are recovered from the log-normalized matrix: values are back-transformed with expm1 and
each cell is divided by its smallest nonzero value. The export stops if fewer than 99.9% of the
rescaled values lie within 0.01 of an integer.

Outputs (results/sparkx): counts_full.mtx, genes_full.txt, cells_full.csv
"""
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import squidpy as sq

a = sq.datasets.slideseqv2()
e = sp.csr_matrix(a.X, dtype=np.float64)
e.eliminate_zeros()
e.data = np.expm1(e.data)
nnz = np.diff(e.indptr)
rows = np.flatnonzero(nnz > 0)
rmin = np.full(e.shape[0], np.nan)
rmin[rows] = np.minimum.reduceat(e.data, e.indptr[rows])
scaled = e.data / np.repeat(rmin, nnz)
share = float(np.mean(np.abs(scaled - np.round(scaled)) < 0.01))
print(f"share of values within 0.01 of an integer: {share:.5f}; max count {np.round(scaled).max():.0f}")
if share < 0.999:
    raise SystemExit("counts not recovered, stop")
e.data = np.round(scaled)
counts = e.T.tocoo().astype(np.int64)
sio.mmwrite("results/sparkx/counts_full.mtx", counts, field="integer")
pd.Series(list(a.var_names)).to_csv("results/sparkx/genes_full.txt", index=False, header=False)
pd.DataFrame({"cell": a.obs_names.astype(str), "x": a.obsm["spatial"][:, 0],
              "y": a.obsm["spatial"][:, 1]}).to_csv("results/sparkx/cells_full.csv", index=False)
