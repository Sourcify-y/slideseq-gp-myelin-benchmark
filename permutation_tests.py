"""
Permutation tests with full refitting (reviewer items 2 and 4).

exact : one gene, 5 independent non-overlapping 2,000-cell subsamples.
        Statistic T = mean test R^2 across the 5 spatially blocked splits.
        Each permutation shuffles the labels inside every subsample, then repeats the
        whole procedure: kernel selection on the validation block, refit on train+val,
        prediction on the test block. The observed T is computed with the same
        procedure and the same settings. One empirical null for T, no Fisher combination.

svgp  : one gene, full dataset, spatially blocked 80/20 split. Each permutation shuffles
        the training labels and retrains the SVGP from scratch, then scores R^2 on the
        real held-out labels.

Results are appended to one CSV per gene and family after every permutation, so a
stopped run resumes where it ended.

Usage:
    python3 permutation_tests.py exact   --gene Mbp --n-perm 100 --restarts 1
    python3 permutation_tests.py svgp    --gene Mbp --n-perm 50
    python3 permutation_tests.py summary
"""
import argparse
import contextlib
import csv
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import squidpy as sq
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import pipeline as P

OUT_DIR = Path("results") / "perm"
KERNELS = ["RBF", "Matern(nu=1.5)", "Matern(nu=2.5)"]
PERM_SEED_OFFSET = 10_000


def load_gene(gene):
    adata = sq.datasets.slideseqv2()
    coords = adata.obsm["spatial"].astype(float)
    expr = P.get_gene_expression(adata, gene)
    del adata
    return coords, expr


def load_done(path):
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["b"].astype(int))


def append_row(path, row):
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def build_splits(coords, expr):
    subsamples = P.draw_disjoint_subsamples(len(expr), P.N_SUBSAMPLE, P.N_SPLITS + P.N_REPEATS)
    splits = []
    for i in range(P.N_SPLITS):
        sub = subsamples[i]
        seed = P.SEED + i
        X_raw = coords[sub]
        X = StandardScaler().fit_transform(X_raw)
        trainval, test = P.spatial_block_split(X_raw, test_size=P.TEST_SIZE,
                                               n_blocks=P.N_BLOCKS_A, seed=seed)
        sub_train, sub_val = P.spatial_block_split(
            X_raw[trainval], test_size=P.VAL_SIZE,
            n_blocks=max(5, int(P.N_BLOCKS_A * (1 - P.TEST_SIZE))), seed=seed + 1000)
        splits.append({"sub": sub, "X": X, "seed": seed, "trainval": trainval, "test": test,
                       "train": trainval[sub_train], "val": trainval[sub_val]})
    return splits


def fit_gp(kernel_name, X, y, restarts, seed):
    gp = GaussianProcessRegressor(kernel=P._kernel_ctor(kernel_name), n_restarts_optimizer=restarts,
                                  normalize_y=True, random_state=seed)
    return gp.fit(X, y)


def exact_statistic(splits, y_list, restarts):
    r2s, chosen = [], []
    for sp, y in zip(splits, y_list):
        X = sp["X"]
        best_name, best_rmse = None, np.inf
        for name in KERNELS:
            gp = fit_gp(name, X[sp["train"]], y[sp["train"]], restarts, P.SEED)
            rmse = float(np.sqrt(np.mean((y[sp["val"]] - gp.predict(X[sp["val"]])) ** 2)))
            if rmse < best_rmse:
                best_name, best_rmse = name, rmse
        gp = fit_gp(best_name, X[sp["trainval"]], y[sp["trainval"]], restarts, sp["seed"])
        r2s.append(float(r2_score(y[sp["test"]], gp.predict(X[sp["test"]]))))
        chosen.append(best_name)
    return float(np.mean(r2s)), r2s, chosen


def exact_row(b, T, r2s, chosen, seconds):
    row = {"b": b, "T": T}
    for i, (r2, k) in enumerate(zip(r2s, chosen), start=1):
        row[f"r2_split{i}"] = r2
        row[f"kernel_split{i}"] = k
    row["seconds"] = round(seconds, 1)
    return row


def run_exact(gene, n_perm, restarts):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"exact_{gene}.csv"
    done = load_done(path)

    coords, expr = load_gene(gene)
    splits = build_splits(coords, expr)
    y_real = [expr[sp["sub"]] for sp in splits]

    if -1 not in done:
        t0 = time.time()
        T, r2s, chosen = exact_statistic(splits, y_real, restarts)
        append_row(path, exact_row(-1, T, r2s, chosen, time.time() - t0))
        print(f"[{gene}] observed T={T:.4f}  split R2={[round(x, 3) for x in r2s]}  "
              f"kernels={chosen}  ({time.time() - t0:.0f} s)", flush=True)

    for b in range(n_perm):
        if b in done:
            continue
        rng = np.random.RandomState(P.SEED + PERM_SEED_OFFSET + b)
        y_perm = [rng.permutation(y) for y in y_real]
        t0 = time.time()
        T, r2s, chosen = exact_statistic(splits, y_perm, restarts)
        append_row(path, exact_row(b, T, r2s, chosen, time.time() - t0))
        print(f"[{gene}] perm {b + 1}/{n_perm}  T={T:.4f}  ({time.time() - t0:.0f} s)", flush=True)


def svgp_setup(coords, expr):
    coord_scaler = StandardScaler().fit(coords)
    X = coord_scaler.transform(coords)
    y = StandardScaler().fit_transform(expr.reshape(-1, 1)).flatten()
    train_idx, test_idx = P.spatial_block_split(coords, test_size=P.TEST_SIZE,
                                                n_blocks=P.N_BLOCKS_B, seed=P.SEED)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def svgp_r2(X_train, y_train, X_test, y_test, seed):
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        model, likelihood, _, converged = P.train_svgp(X_train, y_train)
    pred, _ = P.predict_svgp(model, likelihood, X_test)
    return float(r2_score(y_test, pred)), bool(converged)


def run_svgp(gene, n_perm):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"svgp_{gene}.csv"
    done = load_done(path)

    coords, expr = load_gene(gene)
    X_train, y_train, X_test, y_test = svgp_setup(coords, expr)

    if -1 not in done:
        t0 = time.time()
        r2, conv = svgp_r2(X_train, y_train, X_test, y_test, P.SEED)
        append_row(path, {"b": -1, "r2": r2, "converged": conv, "seconds": round(time.time() - t0, 1)})
        print(f"[{gene}] observed SVGP R2={r2:.4f} converged={conv} ({time.time() - t0:.0f} s)", flush=True)

    for b in range(n_perm):
        if b in done:
            continue
        rng = np.random.RandomState(P.SEED + PERM_SEED_OFFSET + b)
        y_perm = rng.permutation(y_train)
        t0 = time.time()
        r2, conv = svgp_r2(X_train, y_perm, X_test, y_test, P.SEED + b + 1)
        append_row(path, {"b": b, "r2": r2, "converged": conv, "seconds": round(time.time() - t0, 1)})
        print(f"[{gene}] perm {b + 1}/{n_perm}  R2={r2:.4f}  ({time.time() - t0:.0f} s)", flush=True)


def summarize():
    for family, stat in (("exact", "T"), ("svgp", "r2")):
        rows = []
        for f in sorted(OUT_DIR.glob(f"{family}_*.csv")):
            gene = f.stem.split("_", 1)[1]
            df = pd.read_csv(f)
            obs, null = df[df["b"] == -1], df[df["b"] >= 0]
            if obs.empty or null.empty:
                continue
            t_obs = float(obs[stat].iloc[0])
            vals = null[stat].values
            n = len(vals)
            rows.append({"gene": gene, "observed": t_obs, "null_mean": float(np.mean(vals)),
                         "null_sd": float(np.std(vals)), "n_perm": n,
                         "p_value": (np.sum(vals >= t_obs) + 1) / (n + 1), "p_floor": 1 / (n + 1)})
        if not rows:
            continue
        out = pd.DataFrame(rows)
        q, sig = P.benjamini_hochberg(out["p_value"].values, alpha=P.ALPHA_FDR)
        out["q_value"], out["significant_fdr"] = q, sig
        out.to_csv(OUT_DIR / f"summary_{family}.csv", index=False)
        print(f"\n{family} permutation test (full refit)")
        print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["exact", "svgp", "summary"])
    ap.add_argument("--gene")
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--restarts", type=int, default=1)
    args = ap.parse_args()

    if args.mode == "summary":
        summarize()
        return
    if not args.gene:
        ap.error("--gene is required for exact and svgp")
    if args.mode == "exact":
        run_exact(args.gene, args.n_perm, args.restarts)
    else:
        run_svgp(args.gene, args.n_perm)


if __name__ == "__main__":
    main()