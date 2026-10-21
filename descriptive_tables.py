"""
Descriptive tables for the revision (no model fitting).

gene    sparsity and signal descriptors per gene
blocks  size and area of the k-means blocks at 40 and 150 blocks
labels  overlap of predicted maps with categorical annotations in adata.obs

Usage
    python3 descriptive_tables.py gene
    python3 descriptive_tables.py blocks
    python3 descriptive_tables.py labels
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import squidpy as sq
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import KMeans

import pipeline as P

OUT = Path("results") / "revision"
GENES = ["Mbp", "Plp1", "Mobp", "Mog", "Cnp", "Sox10", "Snap25", "Malat1"]


def counts_from_lognorm(m):
    e = sp.csr_matrix(m, dtype=np.float64)
    e.eliminate_zeros()
    e.data = np.expm1(e.data)
    nnz = np.diff(e.indptr)
    rows = np.flatnonzero(nnz > 0)
    rmin = np.full(e.shape[0], np.nan)
    rmin[rows] = np.minimum.reduceat(e.data, e.indptr[rows])
    scaled = e.data / np.repeat(rmin, nnz)
    if np.mean(np.abs(scaled - np.round(scaled)) < 0.01) < 0.999:
        raise SystemExit("counts not recovered, stop")
    e.data = np.round(scaled)
    return e.tocsc()


def gene_table():
    a = sq.datasets.slideseqv2()
    counts, names = counts_from_lognorm(a.X), list(a.var_names)
    mor = pd.read_csv("results/moran/moran_all.csv").set_index("gene")
    spx = pd.read_csv("results/sparkx/sparkx_full.csv").set_index("gene")
    rows = []
    for g in GENES:
        c = counts[:, names.index(g)].toarray().ravel()
        base = pd.read_csv(f"results/baselines/exact_{g}_n2000.csv")
        svgp = pd.read_csv(f"results/matched/svgp_{g}_n0.csv")["r2"]
        rows.append({"gene": g, "cells_nonzero_pct": round(100 * np.mean(c > 0), 1),
                     "cells_count_ge2_pct": round(100 * np.mean(c >= 2), 1),
                     "mean_count": round(c.mean(), 3),
                     "mean_count_if_nonzero": round(c[c > 0].mean(), 2),
                     "exact_noise_median": f"{base['noise_level'].median():.3g}",
                     "moran_I": round(mor.loc[g, "I"], 3),
                     "svgp_r2_40blocks": round(svgp.mean(), 3),
                     "sparkx_rank_of_4000": int(spx.loc[g, "rank"])})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "gene_descriptives.csv", index=False)
    print(out.to_string(index=False))
    print("exact_noise_median is the fitted white-noise level of the exact GP (normalized y); "
          "values near 0 for Sox10 come from collapsed fits")


def block_table():
    a = sq.datasets.slideseqv2()
    coords = a.obsm["spatial"].astype(float)
    rows = []
    for nb in (40, 150):
        lab = KMeans(n_clusters=nb, random_state=P.SEED, n_init=10).fit(coords).labels_
        n = np.bincount(lab, minlength=nb)
        area = np.array([ConvexHull(coords[lab == b]).volume for b in range(nb)])
        _, te = P.spatial_block_split(coords, test_size=P.TEST_SIZE, n_blocks=nb, seed=P.SEED)
        test_blocks = np.unique(lab[te])
        rows.append({"blocks": nb, "cells_min": int(n.min()), "cells_median": int(np.median(n)),
                     "cells_max": int(n.max()), "cells_cv": round(float(n.std() / n.mean()), 2),
                     "area_median": round(float(np.median(area))), "area_cv": round(float(area.std() / area.mean()), 2),
                     "equiv_diameter_median": round(float(np.median(np.sqrt(4 * area / np.pi)))),
                     "test_blocks_split0": len(test_blocks), "test_cells_split0": int(len(te)),
                     "test_area_pct_split0": round(100 * float(area[test_blocks].sum() / area.sum()), 1)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "block_descriptives.csv", index=False)
    print(out.to_string(index=False))
    print("area and diameter are in coordinate units; split 0 uses the main analysis seed")


def eta2(values, labels):
    v = np.asarray(values, float)
    tot = np.sum((v - v.mean()) ** 2)
    if tot == 0:
        return np.nan
    g = pd.DataFrame({"v": v, "l": labels}).groupby("l")["v"]
    return float(np.sum(g.count() * (g.mean() - v.mean()) ** 2) / tot)


def label_overlap():
    a = sq.datasets.slideseqv2()
    coords = a.obsm["spatial"].astype(float)
    cats = [c for c in a.obs.columns if a.obs[c].dtype.name in ("category", "object")
            and 2 <= a.obs[c].nunique() <= 60]
    print("obs columns:", list(a.obs.columns))
    if not cats:
        print("no categorical annotation in adata.obs, stop")
        return
    _, _, grid_pts, mask = P.make_grid(coords)
    pts = grid_pts[mask]
    nn = cKDTree(coords).query(pts)[1]
    names = list(a.var_names)
    rows = []
    for col in cats:
        lab_cells = a.obs[col].astype(str).to_numpy()
        lab_grid = lab_cells[nn]
        for gene in ("Mbp", "Sox10"):
            mp = np.load(f"results/sensitivity/maps_{gene}.npz")["svgp_p0"]
            x = a.X[:, names.index(gene)]
            x = np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()
            top = pd.Series(mp).groupby(lab_grid).mean().sort_values(ascending=False).head(3)
            rows.append({"annotation": col, "labels": len(np.unique(lab_cells)), "gene": gene,
                         "eta2_svgp_map_by_label": round(eta2(mp, lab_grid), 3),
                         "eta2_cell_expression_by_label": round(eta2(x, lab_cells), 3),
                         "top3_labels_by_mean_map": "; ".join(f"{k} ({v:+.2f})" for k, v in top.items())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "label_overlap.csv", index=False)
    print(out.to_string(index=False))
    print("eta2 = share of variance explained by the annotation (map on the pipeline grid, "
          "each grid point takes the label of its nearest cell)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gene", "blocks", "labels"])
    {"gene": gene_table, "blocks": block_table, "labels": label_overlap}[ap.parse_args().mode]()
