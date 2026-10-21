"""
Uncertainty quality of the exact GP against kNN and IDW.

Design
  Same splits, kernel selection and refit as baseline_rerun.py (n = 2,000, 5 independent splits).
  gp         GP mean and GP predictive sd (includes noise).
  gp_const   GP mean with a constant sd equal to the GP validation RMSE (fit on train only).
  knn_const  kNN (k = 10) mean with a constant sd equal to its validation RMSE.
  idw_const  IDW mean with a constant sd equal to its validation RMSE.
  Each predictive distribution is Gaussian. Test scores: coverage and mean width of the 95%
  interval (z = 1.96), interval score (alpha = 0.05) and CRPS, in log-normalized expression
  units. Lower interval score and CRPS are better.

Usage
    python3 uncertainty_scores.py run --gene Mbp --max-splits 1
    python3 uncertainty_scores.py summary
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from sklearn.neighbors import KNeighborsRegressor

import pipeline as P
import permutation_tests as T
from baseline_rerun import RESTARTS

OUT = Path("results") / "uncertainty"
ALPHA, Z = 0.05, 1.96
METHODS = ["gp", "gp_const", "knn_const", "idw_const"]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def scores(y, mu, sd):
    sd = np.maximum(sd, 1e-12)
    lo, hi = mu - Z * sd, mu + Z * sd
    cov = float(np.mean((y >= lo) & (y <= hi)))
    width = float(np.mean(hi - lo))
    iscore = float(np.mean((hi - lo) + (2 / ALPHA) * (lo - y) * (y < lo)
                           + (2 / ALPHA) * (y - hi) * (y > hi)))
    z = (y - mu) / sd
    crps = float(np.mean(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))
    return cov, width, iscore, crps


def run(gene, max_splits=None):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"unc_{gene}.csv"
    done = set(pd.read_csv(path)["split"].astype(int)) if path.exists() else set()
    coords, expr = T.load_gene(gene)
    splits = T.build_splits(coords, expr)
    for i, sp in enumerate(splits):
        if max_splits is not None and i >= max_splits:
            break
        if i in done:
            continue
        t0 = time.time()
        X, y = sp["X"], expr[sp["sub"]]
        tr, va, tv, te = sp["train"], sp["val"], sp["trainval"], sp["test"]

        best, best_r = None, np.inf
        for name in T.KERNELS:
            gp = T.fit_gp(name, X[tr], y[tr], RESTARTS, P.SEED)
            r = rmse(y[va], gp.predict(X[va]))
            if r < best_r:
                best, best_r = name, r
        gp = T.fit_gp(best, X[tv], y[tv], RESTARTS, sp["seed"])
        mu, sd = gp.predict(X[te], return_std=True)

        knn_val = rmse(y[va], KNeighborsRegressor(n_neighbors=10).fit(X[tr], y[tr]).predict(X[va]))
        knn_pred = KNeighborsRegressor(n_neighbors=10).fit(X[tv], y[tv]).predict(X[te])
        idw_val = rmse(y[va], np.asarray(P.idw_predict(X[tr], y[tr], X[va])).ravel())
        idw_pred = np.asarray(P.idw_predict(X[tv], y[tv], X[te])).ravel()

        n = len(te)
        pred = {"gp": (mu, sd), "gp_const": (mu, np.full(n, best_r)),
                "knn_const": (knn_pred, np.full(n, knn_val)),
                "idw_const": (idw_pred, np.full(n, idw_val))}
        msg = []
        for m in METHODS:
            cov, w, isc, crps = scores(y[te], np.asarray(pred[m][0]).ravel(), pred[m][1])
            T.append_row(path, {"split": i, "method": m, "cov95": cov, "width95": w,
                                "interval_score": isc, "crps": crps, "n_test": int(n)})
            msg.append(f"{m}: cov={cov:.3f} w={w:.3f} IS={isc:.3f} CRPS={crps:.4f}")
        print(f"[{gene}] split {i + 1}/{P.N_SPLITS}  kernel={best}  " + "  |  ".join(msg)
              + f"  ({time.time() - t0:.0f} s)", flush=True)


def fmt(x):
    x = np.asarray(x, float)
    k, m = len(x), x.mean()
    h = t.ppf(0.975, k - 1) * x.std(ddof=1) / np.sqrt(k)
    return f"{m:.3f} [{m - h:.3f}, {m + h:.3f}]"


def summarize():
    d = pd.concat([pd.read_csv(f).assign(gene=f.stem.split("_")[1])
                   for f in sorted(OUT.glob("unc_*.csv"))], ignore_index=True)
    print(f"fits per method: {int((d['method'] == 'gp').sum())}")
    print("\ncoverage of the 95% interval, mean [95% t-interval over fits] (nominal 0.950)")
    print(d.groupby("method")["cov95"].apply(fmt).reindex(METHODS).to_string())

    rows = []
    for metric in ["interval_score", "crps"]:
        w = d.pivot_table(index=["gene", "split"], columns="method", values=metric)
        for m in METHODS[1:]:
            r = w["gp"] / w[m]
            rows.append({"metric": metric, "gp / other": m, "ratio": fmt(r),
                         "gp_better": f"{int((r < 1).sum())}/{len(r)}"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "summary.csv", index=False)
    print("\nratio of GP score to the other method (below 1: GP better)")
    print(out.to_string(index=False))

    w = d.pivot_table(index=["gene", "split"], columns="method", values="interval_score")
    per_gene = pd.DataFrame({m: (w["gp"] / w[m]).groupby(level="gene").mean() for m in METHODS[1:]})
    per_gene.to_csv(OUT / "summary_by_gene.csv")
    print("\ninterval score ratio GP / other, mean over splits, by gene")
    print(per_gene.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary"])
    ap.add_argument("--gene")
    ap.add_argument("--max-splits", type=int, default=None)
    args = ap.parse_args()
    if args.mode == "summary":
        summarize()
        return
    if not args.gene:
        ap.error("--gene is required for run")
    run(args.gene, args.max_splits)


if __name__ == "__main__":
    main()
