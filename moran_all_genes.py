"""
Moran's I for all 8 genes and its relation to prediction accuracy.

Design
  Same call as pipeline.moran_baseline: squidpy spatial_neighbors (generic, 6 neighbours) on the
  full dataset, spatial_autocorr (Moran, 999 permutations, pipeline seed), expression as loaded
  and normalized by pipeline.load_data. Mbp is checked against the submitted value (0.4886).
  The summary relates Moran's I to the mean R2 of SVGP, IDW and kNN (k = 100) at 40 blocks.

Usage
    python3 moran_all_genes.py run
    python3 moran_all_genes.py summary
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import pipeline as P

OUT_DIR = Path("results") / "moran"
FULLBASE = Path("results") / "fullbase"
MATCHED = Path("results") / "matched"
GENES = ["Mbp", "Plp1", "Mobp", "Mog", "Cnp", "Sox10", "Snap25", "Malat1"]
SUBMITTED_MBP = 0.488613


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adata, _ = P.load_data()
    genes = [g for g in GENES if g in adata.var_names]
    missing = [g for g in GENES if g not in genes]
    if missing:
        print(f"genes not in dataset: {missing}")
    df = P.moran_baseline(adata, genes)
    df.to_csv(OUT_DIR / "moran_all.csv", index=False)

    mbp = float(df.loc[df["gene"] == "Mbp", "I"].iloc[0])
    status = "MATCH" if abs(mbp - SUBMITTED_MBP) < 1e-3 else "MISMATCH"
    print(f"Mbp I = {mbp:.6f}, submitted {SUBMITTED_MBP:.6f}: {status}")


def mean_col(path, col):
    return float(pd.read_csv(path)[col].mean()) if path.exists() else np.nan


def summarize():
    m = pd.read_csv(OUT_DIR / "moran_all.csv").set_index("gene")
    rows = []
    for g in GENES:
        if g not in m.index:
            continue
        svgp = mean_col(MATCHED / f"svgp_{g}_n0.csv", "r2")
        idw = mean_col(FULLBASE / f"base_{g}_n0.csv", "idw_r2")
        knn = mean_col(FULLBASE / f"knnk_{g}_n0.csv", "knn_k100")
        rows.append({"gene": g, "moran_I": m.loc[g, "I"],
                     "pval_sim": m.loc[g, "pval_sim"],
                     "pval_sim_fdr_bh": m.loc[g, "pval_sim_fdr_bh"],
                     "svgp_r2": svgp, "idw_r2": idw, "knn100_r2": knn,
                     "svgp_minus_idw": svgp - idw})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "summary.csv", index=False)
    print(out.round(4).to_string(index=False))

    print(f"\nCorrelation of Moran's I with mean R2 at 40 blocks (n = {len(out)} genes)")
    for c in ["svgp_r2", "idw_r2", "knn100_r2", "svgp_minus_idw"]:
        d = out[["moran_I", c]].dropna()
        rho, p_rho = spearmanr(d["moran_I"], d[c])
        r, p_r = pearsonr(d["moran_I"], d[c])
        print(f"{c:16s} Spearman rho = {rho:+.3f} (p = {p_rho:.3f})   "
              f"Pearson r = {r:+.3f} (p = {p_r:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary"])
    args = ap.parse_args()
    run() if args.mode == "run" else summarize()


if __name__ == "__main__":
    main()
