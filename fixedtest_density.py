"""
Fixed-test-block density experiment (reviewer items 6 and 7).

Design
  Full dataset, 5 splits, 40 blocks, 20% of blocks held out. The test cells are the same
  for every training size and are those of results/matched/svgp_<gene>_n0.csv. Only the
  number of training cells changes: n cells are drawn at random from the training pool.
  SVGP, 400 inducing points, gradient steps held at the full-data run.

Usage
    python3 fixedtest_density.py run --gene Mbp --n-train 3200
    python3 fixedtest_density.py run --gene Mbp --n-train 1600 --max-splits 1
    python3 fixedtest_density.py summary
"""
import argparse
import contextlib
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

import pipeline as P
import permutation_tests as T
from matched_comparison import build_splits_n, done_splits, svgp_epochs

OUT_DIR = Path("results") / "fixedtest"
MATCHED_DIR = Path("results") / "matched"
GENES = ["Mbp", "Plp1", "Snap25"]
N_TRAIN = [1600, 3200, 4800, 12800, 25600]


def run(gene, n_train, max_splits=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"svgp_{gene}_t{n_train}.csv"
    done = done_splits(path)

    coords, expr = T.load_gene(gene)
    splits = build_splits_n(coords, expr, 0, need_val=False)
    for i, sp in enumerate(splits):
        if max_splits is not None and i >= max_splits:
            break
        if i in done:
            continue
        t0 = time.time()
        y = expr[sp["sub"]]
        pool, te = sp["trainval"], sp["test"]
        rng = np.random.default_rng(P.SEED + 1000 * i + n_train)
        tr = np.sort(rng.choice(pool, size=min(n_train, len(pool)), replace=False))
        mu, sd = y[tr].mean(), y[tr].std()
        y_tr, y_te = (y[tr] - mu) / sd, (y[te] - mu) / sd
        epochs = svgp_epochs(len(tr))

        torch.manual_seed(P.SEED + i)
        with contextlib.redirect_stdout(io.StringIO()):
            model, likelihood, _, converged = P.train_svgp(sp["X"][tr], y_tr, epochs=epochs)
        pred, _ = P.predict_svgp(model, likelihood, sp["X"][te])
        r2 = float(r2_score(y_te, pred))

        T.append_row(path, {"split": i, "r2": r2,
                            "detail": f"n_train={len(tr)};n_test={len(te)};epochs={epochs};converged={converged}",
                            "seconds": round(time.time() - t0, 1)})
        print(f"[{gene}] svgp n_train={len(tr)} split {i + 1}/{P.N_SPLITS}  R2={r2:.4f}  "
              f"epochs={epochs}  converged={converged}  ({time.time() - t0:.0f} s)", flush=True)


def summarize():
    cells = {}
    for gene in GENES:
        for n in N_TRAIN:
            f = OUT_DIR / f"svgp_{gene}_t{n}.csv"
            if f.exists():
                d = pd.read_csv(f)
                cells[(gene, str(n))] = f"{d['r2'].mean():.3f} ± {d['r2'].std(ddof=0):.3f} (k={len(d)})"
        f = MATCHED_DIR / f"svgp_{gene}_n0.csv"
        if f.exists():
            d = pd.read_csv(f)
            cells[(gene, "all")] = f"{d['r2'].mean():.3f} ± {d['r2'].std(ddof=0):.3f} (k={len(d)})"
    if not cells:
        print("no results found")
        return
    cols = [str(n) for n in N_TRAIN] + ["all"]
    table = pd.DataFrame({c: {g: cells.get((g, c), "") for g in GENES} for c in cols})
    table.index.name = "gene"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "summary.csv")
    print(table.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary"])
    ap.add_argument("--gene")
    ap.add_argument("--n-train", type=int, help="number of training cells")
    ap.add_argument("--max-splits", type=int, default=None, help="run only the first k splits (smoke test)")
    args = ap.parse_args()

    if args.mode == "summary":
        summarize()
        return
    if not args.gene or not args.n_train:
        ap.error("--gene and --n-train are required for run")
    run(args.gene, args.n_train, args.max_splits)


if __name__ == "__main__":
    main()
