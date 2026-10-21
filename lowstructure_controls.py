"""
Low-structure control genes.

select   Moran's I (analytic, 6 neighbours, log-normalized X, as in the pipeline) for all genes, in
         chunks. Candidates have a fraction of nonzero cells inside the range of the 8 benchmark
         genes and are not benchmark genes; the 4 with the lowest |I| are the controls.
list     print the control genes
run      SVGP (400 inducing points), IDW and kNN (k = 100) at 40 blocks, 5 splits, one gene
summary  controls next to the 8 benchmark genes at 40 blocks

Usage
    python3 lowstructure_controls.py select
    python3 lowstructure_controls.py run --gene GENE --max-splits 1
    python3 lowstructure_controls.py summary
"""
import argparse
import contextlib
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import squidpy as sq
import torch
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

import pipeline as P
import permutation_tests as T
from matched_comparison import build_splits_n, done_splits, svgp_epochs
from fulldata_baselines import idw_chunked

OUT = Path("results") / "lowstructure"
GENES = ["Mbp", "Plp1", "Mobp", "Mog", "Cnp", "Sox10", "Snap25", "Malat1"]
N_CONTROLS = 4


def select():
    OUT.mkdir(parents=True, exist_ok=True)
    a = sq.datasets.slideseqv2()
    names = list(a.var_names)
    frac = np.diff(sp.csc_matrix(a.X).indptr) / a.n_obs
    sq.gr.spatial_neighbors(a, coord_type="generic", n_neighs=6)
    parts = []
    for s in range(0, len(names), 500):
        sq.gr.spatial_autocorr(a, mode="moran", genes=names[s:s + 500], n_perms=None, n_jobs=1)
        parts.append(a.uns["moranI"][["I"]].copy())
    df = pd.DataFrame({"gene": names, "nonzero_frac": frac}).set_index("gene").join(pd.concat(parts))
    df.to_csv(OUT / "moran_all_4000.csv")

    bench = df.loc[GENES]
    lo, hi = bench["nonzero_frac"].min(), bench["nonzero_frac"].max()
    pool = df[(df["nonzero_frac"] >= lo) & (df["nonzero_frac"] <= hi) & (~df.index.isin(GENES))].copy()
    pool["absI"] = pool["I"].abs()
    sparse = pool[pool["nonzero_frac"] < 0.10].sort_values("absI").head(N_CONTROLS)
    abundant = pool[pool["nonzero_frac"] >= 0.20].sort_values("absI").head(N_CONTROLS)
    sel = pd.concat([sparse.assign(group="control"), abundant.assign(group="control"),
                     bench.assign(group="benchmark")])
    sel[["group", "I", "nonzero_frac"]].to_csv(OUT / "selection.csv")
    print(f"benchmark nonzero fraction range: {lo:.3f} to {hi:.3f}; candidates: {len(pool)}")
    print(sel[["group", "I", "nonzero_frac"]].round(4).to_string())


def controls():
    return list(pd.read_csv(OUT / "selection.csv").query("group == 'control'")["gene"])


def run(gene, max_splits=None):
    path = OUT / f"ls_{gene}.csv"
    done = done_splits(path)
    coords, expr = T.load_gene(gene)
    for i, split in enumerate(build_splits_n(coords, expr, 0, need_val=False, n_blocks=40)):
        if max_splits is not None and i >= max_splits:
            break
        if i in done:
            continue
        t0 = time.time()
        y, X = expr[split["sub"]], split["X"]
        tr, te = split["trainval"], split["test"]
        mu, sd = y[tr].mean(), y[tr].std()
        y_tr, y_te = (y[tr] - mu) / sd, (y[te] - mu) / sd
        idw = float(r2_score(y_te, idw_chunked(X[tr], y_tr, X[te])))
        knn = float(r2_score(y_te, KNeighborsRegressor(n_neighbors=100).fit(X[tr], y_tr).predict(X[te])))
        torch.manual_seed(split["seed"])
        with contextlib.redirect_stdout(io.StringIO()):
            model, lik, _, conv = P.train_svgp(X[tr], y_tr, epochs=svgp_epochs(len(tr)))
        mean, std = P.predict_svgp(model, lik, X[te])
        r2 = float(r2_score(y_te, mean))
        cov2 = float(np.mean(np.abs(y_te - mean) <= 2 * std))
        T.append_row(path, {"split": i, "svgp_r2": r2, "idw_r2": idw, "knn100_r2": knn,
                            "cov2": cov2, "converged": bool(conv), "n_test": int(len(te)),
                            "seconds": round(time.time() - t0, 1)})
        print(f"[{gene}] split {i + 1}/{P.N_SPLITS}  SVGP R2={r2:.4f}  IDW R2={idw:.4f}  "
              f"kNN100 R2={knn:.4f}  cov2={cov2:.3f}  ({time.time() - t0:.0f} s)", flush=True)


def ms(x):
    return f"{x.mean():.3f} ± {x.std(ddof=0):.3f}"


def summarize():
    sel = pd.read_csv(OUT / "selection.csv").set_index("gene")
    rows = []
    for g, r in sel.iterrows():
        if r["group"] == "control":
            f = OUT / f"ls_{g}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            s, i, k, c = d["svgp_r2"], d["idw_r2"], d["knn100_r2"], d["cov2"]
        else:
            s = pd.read_csv(f"results/matched/svgp_{g}_n0.csv")["r2"]
            i = pd.read_csv(f"results/fullbase/base_{g}_n0.csv")["idw_r2"]
            k = pd.read_csv(f"results/fullbase/knnk_{g}_n0.csv")["knn_k100"]
            c = None
        rows.append({"gene": g, "group": r["group"], "moran_I": round(r["I"], 3),
                     "nonzero_frac": round(r["nonzero_frac"], 3), "svgp": ms(s), "idw": ms(i),
                     "knn100": ms(k), "svgp_cov2": "" if c is None else f"{c.mean():.3f}"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "list", "run", "summary"])
    ap.add_argument("--gene")
    ap.add_argument("--max-splits", type=int, default=None)
    a = ap.parse_args()
    if a.mode == "select":
        select()
    elif a.mode == "list":
        print("\n".join(controls()))
    elif a.mode == "summary":
        summarize()
    else:
        if not a.gene:
            ap.error("--gene is required for run")
        run(a.gene, a.max_splits)
