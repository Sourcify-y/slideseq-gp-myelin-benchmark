"""
Map sensitivity for one gene, with Mbp as the reference.

Maps are predicted means on the pipeline grid (P.make_grid, points inside the convex hull).
  exact_s0..s4    exact GP, 5 independent 2,000-cell subsamples, kernel chosen on validation
  svgp_p0..p4     SVGP (400 inducing points), 5 block partitions at 150 blocks (matched designs)
  svgp_M200/M800  SVGP on partition 0 with 200 / 800 inducing points
  svgp_seed1/2    SVGP on partition 0 with torch seed + 1 / + 2
svgp_p0 is the main analysis configuration (Mbp test R2 about 0.428).

Usage
    python3 sensitivity_maps.py run --gene Mbp --smoke
    python3 sensitivity_maps.py run --gene Sox10
    python3 sensitivity_maps.py summary
"""
import argparse
import contextlib
import io
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import pipeline as P
import permutation_tests as T
from matched_comparison import build_splits_n, svgp_epochs
from baseline_rerun import RESTARTS

OUT = Path("results") / "sensitivity"


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def run(gene, smoke=False):
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{gene}_smoke" if smoke else gene
    mpath, fpath = OUT / f"maps_{stem}.npz", OUT / f"fits_{stem}.csv"
    maps = dict(np.load(mpath)) if mpath.exists() else {}

    coords, expr = T.load_gene(gene)
    _, _, grid_pts, mask = P.make_grid(coords)
    pts = grid_pts[mask]
    print(f"[{gene}] grid points inside tissue: {len(pts)}", flush=True)

    def record(key, vec, r2, detail, t0):
        maps[key] = np.asarray(vec, float).ravel()
        np.savez(mpath, **maps)
        T.append_row(fpath, {"fit": key, "test_r2": r2, "detail": detail})
        print(f"[{gene}] {key}  test R2={r2:.4f}  {detail}  ({time.time() - t0:.0f} s)", flush=True)

    # exact GP on independent subsamples
    for i, sp in enumerate(T.build_splits(coords, expr)):
        key = f"exact_s{i}"
        if key in maps or (smoke and i > 0):
            continue
        t0 = time.time()
        y = expr[sp["sub"]]
        sc = next((c for c in (StandardScaler().fit(coords[sp["sub"]]), StandardScaler().fit(coords))
                   if np.allclose(c.transform(coords[sp["sub"]]), sp["X"])), None)
        if sc is None:
            raise SystemExit("split coordinates match neither the subsample scaler nor the global scaler, stop")
        X = sp["X"]
        best, best_r = None, np.inf
        for name in T.KERNELS:
            gp = T.fit_gp(name, X[sp["train"]], y[sp["train"]], RESTARTS, P.SEED)
            r = rmse(y[sp["val"]], gp.predict(X[sp["val"]]))
            if r < best_r:
                best, best_r = name, r
        gp = T.fit_gp(best, X[sp["trainval"]], y[sp["trainval"]], RESTARTS, sp["seed"])
        r2 = float(r2_score(y[sp["test"]], gp.predict(X[sp["test"]])))
        params = gp.kernel_.get_params()
        detail = (f"kernel={best};ls={np.mean(np.atleast_1d(params['k1__length_scale'])):.3g};"
                  f"noise={params['k2__noise_level']:.3g}")
        record(key, gp.predict(sc.transform(pts)), r2, detail, t0)

    # SVGP on block partitions of the full data
    splits = build_splits_n(coords, expr, 0, need_val=False, n_blocks=150)
    sc = StandardScaler().fit(coords)
    Xg = sc.transform(pts)
    jobs = [(f"svgp_p{i}", i, 400, 0) for i in range(5)]
    jobs += [("svgp_M200", 0, 200, 0), ("svgp_M800", 0, 800, 0),
             ("svgp_seed1", 0, 400, 1), ("svgp_seed2", 0, 400, 2)]
    if smoke:
        jobs = jobs[:1]
    for key, i, M, dseed in jobs:
        if key in maps:
            continue
        t0 = time.time()
        sp = splits[i]
        if not np.allclose(sp["X"], sc.transform(coords)):
            raise SystemExit("full-data coordinates differ from the global scaler, stop")
        tr, te = sp["trainval"], sp["test"]
        mu, sd = expr[tr].mean(), expr[tr].std()
        y_tr, y_te = (expr[tr] - mu) / sd, (expr[te] - mu) / sd
        torch.manual_seed(sp["seed"] + dseed)
        with contextlib.redirect_stdout(io.StringIO()):
            model, lik, _, conv = P.train_svgp(sp["X"][tr], y_tr, n_inducing=M,
                                               epochs=svgp_epochs(len(tr)))
        pred_te, _ = P.predict_svgp(model, lik, sp["X"][te])
        pred_g, _ = P.predict_svgp(model, lik, Xg)
        r2 = float(r2_score(y_te, pred_te))
        record(key, pred_g, r2, f"M={M};seed={sp['seed'] + dseed};converged={conv}", t0)


def pr(a, b):
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def describe(vals):
    v = np.array(vals, float)
    ok = v[~np.isnan(v)]
    if len(ok) == 0:
        return f"undefined (0/{len(v)} pairs; a map is constant)"
    return f"median {np.median(ok):+.3f} [{ok.min():+.3f}, {ok.max():+.3f}]  defined {len(ok)}/{len(v)}"


def summarize():
    rows = []
    for gene in ["Sox10", "Mbp"]:
        mp, fp = OUT / f"maps_{gene}.npz", OUT / f"fits_{gene}.csv"
        if not mp.exists():
            continue
        m = dict(np.load(mp))
        print(f"\n== {gene} ==")
        print(pd.read_csv(fp).to_string(index=False))
        ex = [k for k in m if k.startswith("exact_s")]
        parts = [f"svgp_p{i}" for i in range(5) if f"svgp_p{i}" in m]
        msw = [k for k in ["svgp_M200", "svgp_p0", "svgp_M800"] if k in m]
        sds = [k for k in ["svgp_p0", "svgp_seed1", "svgp_seed2"] if k in m]
        print(f"constant exact maps: {sum(np.std(m[k]) < 1e-8 for k in ex)}/{len(ex)}")
        comps = {
            "exact vs exact (independent subsamples)": [pr(m[a], m[b]) for a, b in itertools.combinations(ex, 2)],
            "SVGP vs SVGP (block partitions)": [pr(m[a], m[b]) for a, b in itertools.combinations(parts, 2)],
            "SVGP vs SVGP (200, 400, 800 inducing points)": [pr(m[a], m[b]) for a, b in itertools.combinations(msw, 2)],
            "SVGP vs SVGP (seeds)": [pr(m[a], m[b]) for a, b in itertools.combinations(sds, 2)],
            "exact vs main SVGP": [pr(m[a], m["svgp_p0"]) for a in ex],
            "exact vs SVGP (all exact x all partitions)": [pr(m[a], m[b]) for a in ex for b in parts],
        }
        for name, vals in comps.items():
            print(f"{name:48s} {describe(vals)}")
            rows.append({"gene": gene, "comparison": name, "summary": describe(vals)})
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary"])
    ap.add_argument("--gene")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.mode == "summary":
        summarize()
    else:
        if not a.gene:
            ap.error("--gene is required for run")
        run(a.gene, a.smoke)
