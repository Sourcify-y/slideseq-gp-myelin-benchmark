"""
Baseline rerun for exact GP (reviewer items 12, 13, 14).

Design
  Same splits as the exact-GP permutation run (permutation_tests.build_splits):
  n = 2000 per split, 5 splits, 40 blocks, kernel selection on a validation block,
  refit on train+val, evaluation on the test block, 1 optimizer restart.
  Per split and gene the script logs
    kernel chosen, fitted length scale and noise level
    GP: R2, RMSE, Pearson r, coverage of the 1-sigma and 2-sigma predictive intervals,
        mean width of the 95% interval
    kNN (k = 10) and IDW (power 2): R2 and RMSE
    kNN neighbor-spread interval: coverage of 1-sigma and 2-sigma, mean 95% width
  Check: the GP R2 per split is compared with the observed row of results/perm/exact_<gene>.csv.

Usage
    python3 baseline_rerun.py run --gene Mbp
    python3 baseline_rerun.py run --gene Mbp --max-splits 1
    python3 baseline_rerun.py summary
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

import pipeline as P
import permutation_tests as T

OUT_DIR = Path("results") / "baselines"
PERM_DIR = Path("results") / "perm"
GENES = ["Mbp", "Plp1", "Mobp", "Mog", "Cnp", "Sox10", "Snap25", "Malat1"]
RESTARTS = 1
SHORT = {"RBF": "RBF", "Matern(nu=1.5)": "M1.5", "Matern(nu=2.5)": "M2.5"}


def done_splits(path):
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["split"].astype(int))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def coverage(y, mu, sd):
    err = np.abs(y - mu)
    return (float(np.mean(err <= sd)), float(np.mean(err <= 2 * sd)),
            float(np.mean(2 * 1.96 * sd)))


def pearson(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_split(sp, y):
    X = sp["X"]
    best_name, best_rmse = None, np.inf
    for name in T.KERNELS:
        gp = T.fit_gp(name, X[sp["train"]], y[sp["train"]], RESTARTS, P.SEED)
        r = rmse(y[sp["val"]], gp.predict(X[sp["val"]]))
        if r < best_rmse:
            best_name, best_rmse = name, r
    gp = T.fit_gp(best_name, X[sp["trainval"]], y[sp["trainval"]], RESTARTS, sp["seed"])

    te, tv = sp["test"], sp["trainval"]
    y_te = y[te]
    mu, sd = gp.predict(X[te], return_std=True)
    c1, c2, w95 = coverage(y_te, mu, sd)

    knn = KNeighborsRegressor(n_neighbors=10).fit(X[tv], y[tv])
    knn_pred = knn.predict(X[te])
    _, idx = knn.kneighbors(X[te])
    knn_sd = y[tv][idx].std(axis=1, ddof=1)
    k1, k2, kw95 = coverage(y_te, knn_pred, knn_sd)
    idw_pred = P.idw_predict(X[tv], y[tv], X[te])

    params = gp.kernel_.get_params()
    return {
        "kernel": best_name,
        "length_scale": float(np.mean(np.atleast_1d(params["k1__length_scale"]))),
        "noise_level": float(params["k2__noise_level"]),
        "gp_r2": float(r2_score(y_te, mu)), "gp_rmse": rmse(y_te, mu), "gp_pearson": pearson(y_te, mu),
        "gp_cov1": c1, "gp_cov2": c2, "gp_width95": w95,
        "knn_r2": float(r2_score(y_te, knn_pred)), "knn_rmse": rmse(y_te, knn_pred),
        "idw_r2": float(r2_score(y_te, idw_pred)), "idw_rmse": rmse(y_te, idw_pred),
        "knn_cov1": k1, "knn_cov2": k2, "knn_width95": kw95,
        "n_test": int(len(te)),
    }


def run(gene, max_splits=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"exact_{gene}_n2000.csv"
    done = done_splits(path)

    perm_obs = None
    pf = PERM_DIR / f"exact_{gene}.csv"
    if pf.exists():
        d = pd.read_csv(pf)
        obs = d[d["b"] == -1]
        if not obs.empty:
            perm_obs = [float(obs[f"r2_split{k}"].iloc[0]) for k in range(1, P.N_SPLITS + 1)]

    coords, expr = T.load_gene(gene)
    splits = T.build_splits(coords, expr)
    for i, sp in enumerate(splits):
        if max_splits is not None and i >= max_splits:
            break
        if i in done:
            continue
        t0 = time.time()
        row = {"split": i}
        row.update(evaluate_split(sp, expr[sp["sub"]]))
        row["r2_perm_run"] = perm_obs[i] if perm_obs else float("nan")
        row["seconds"] = round(time.time() - t0, 1)
        T.append_row(path, row)
        diff = row["gp_r2"] - row["r2_perm_run"]
        print(f"[{gene}] split {i + 1}/{P.N_SPLITS}  GP R2={row['gp_r2']:.4f} "
              f"(perm run {row['r2_perm_run']:.4f}, diff {diff:+.1e})  "
              f"RMSE gp/idw/knn={row['gp_rmse']:.3f}/{row['idw_rmse']:.3f}/{row['knn_rmse']:.3f}  "
              f"cov1/cov2={row['gp_cov1']:.3f}/{row['gp_cov2']:.3f}  kernel={SHORT[row['kernel']]}  "
              f"({row['seconds']:.0f} s)", flush=True)


def summarize():
    acc, cal = [], []
    for gene in GENES:
        f = OUT_DIR / f"exact_{gene}_n2000.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)

        def ms(c):
            return f"{d[c].mean():.3f} ± {d[c].std(ddof=0):.3f}"

        kern = d["kernel"].map(SHORT).value_counts()
        acc.append({"gene": gene, "k": len(d), "gp_r2": ms("gp_r2"), "knn_r2": ms("knn_r2"),
                    "idw_r2": ms("idw_r2"), "gp_rmse": ms("gp_rmse"), "knn_rmse": ms("knn_rmse"),
                    "idw_rmse": ms("idw_rmse"),
                    "kernels": " ".join(f"{k}:{v}" for k, v in kern.items()),
                    "max|r2-perm|": f"{(d['gp_r2'] - d['r2_perm_run']).abs().max():.1e}"})
        cal.append({"gene": gene, "gp_cov1": ms("gp_cov1"), "gp_cov2": ms("gp_cov2"),
                    "gp_width95": ms("gp_width95"), "knn_cov1": ms("knn_cov1"),
                    "knn_cov2": ms("knn_cov2"), "knn_width95": ms("knn_width95"),
                    "length_scale": ms("length_scale"), "noise": ms("noise_level")})
    if not acc:
        print("no results found")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    acc, cal = pd.DataFrame(acc), pd.DataFrame(cal)
    acc.to_csv(OUT_DIR / "summary_accuracy.csv", index=False)
    cal.to_csv(OUT_DIR / "summary_calibration.csv", index=False)
    print("== accuracy (expected GP coverage: 0.68 at 1 sigma, 0.95 at 2 sigma) ==")
    print(acc.to_string(index=False))
    print()
    print("== calibration and fitted hyperparameters ==")
    print(cal.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary"])
    ap.add_argument("--gene")
    ap.add_argument("--max-splits", type=int, default=None, help="run only the first k splits (smoke test)")
    args = ap.parse_args()

    if args.mode == "summary":
        summarize()
        return
    if not args.gene:
        ap.error("--gene is required for run")
    run(args.gene, args.max_splits)


if __name__ == "__main__":
    main()
