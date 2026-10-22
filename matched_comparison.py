"""
Matched-data comparison of exact GP and SVGP.

Design
  All models use the same spatially blocked splits: 5 splits, 40 k-means blocks per
  split, 20% of blocks held out for testing. Only the cell density changes with n.
  Subsamples are non-overlapping draws from the full dataset (n = 2000, 4000, 6000);
  n = 0 means the full dataset (41,786 cells), where only SVGP is feasible.

  exact : kernel selection on a validation block (RBF, Matern 1.5, Matern 2.5),
          refit on train+val, R^2 on the test block. 1 optimizer restart, the same
          procedure as permutation_tests.py. Exact GP at n = 2000 is read from the
          permutation run (results/perm/exact_<gene>.csv, observed row).
  svgp  : RBF-ARD SVGP with 400 inducing points, trained on train+val. The number of
          gradient steps is held at the full-data run (about 4,950) at every n, so
          small n is not under-trained.

Results are appended to results/matched/<family>_<gene>_n<n>.csv after every split,
so a stopped run resumes where it ended.

Usage
    python3 matched_comparison.py exact   --gene Mbp --n-sub 4000
    python3 matched_comparison.py svgp    --gene Mbp --n-sub 2000
    python3 matched_comparison.py svgp    --gene Mbp --n-sub 0
    python3 matched_comparison.py svgp    --gene Mbp --n-sub 0 --blocks 150
    python3 matched_comparison.py summary
"""
import argparse
import contextlib
import io
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import pipeline as P
import permutation_tests as T

OUT_DIR = Path("results") / "matched"
PERM_DIR = Path("results") / "perm"
FULL_TRAIN_BATCHES = math.ceil(33379 / P.BATCH_SIZE)
TARGET_STEPS = P.EPOCHS * FULL_TRAIN_BATCHES


def done_splits(path):
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["split"].astype(int))


def build_splits_n(coords, expr, n_sub, need_val, n_blocks=None):
    n_blocks = n_blocks or P.N_BLOCKS_A
    if n_sub == 0:
        sub_list = [np.arange(len(expr))] * P.N_SPLITS
    else:
        sub_list = P.draw_disjoint_subsamples(len(expr), n_sub, P.N_SPLITS)
    splits = []
    for i, sub in enumerate(sub_list):
        seed = P.SEED + i
        X_raw = coords[sub]
        X = StandardScaler().fit_transform(X_raw)
        trainval, test = P.spatial_block_split(X_raw, test_size=P.TEST_SIZE,
                                               n_blocks=n_blocks, seed=seed)
        sp = {"sub": sub, "X": X, "seed": seed, "trainval": trainval, "test": test}
        if need_val:
            sub_train, sub_val = P.spatial_block_split(
                X_raw[trainval], test_size=P.VAL_SIZE,
                n_blocks=max(5, int(P.N_BLOCKS_A * (1 - P.TEST_SIZE))), seed=seed + 1000)
            sp["train"] = trainval[sub_train]
            sp["val"] = trainval[sub_val]
        splits.append(sp)
    return splits


def run_exact(gene, n_sub, restarts, max_splits=None):
    if n_sub == 0:
        raise SystemExit("exact GP is not run on the full dataset")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"exact_{gene}_n{n_sub}.csv"
    done = done_splits(path)

    coords, expr = T.load_gene(gene)
    splits = build_splits_n(coords, expr, n_sub, need_val=True)
    for i, sp in enumerate(splits):
        if max_splits is not None and i >= max_splits:
            break
        if i in done:
            continue
        t0 = time.time()
        _, r2s, chosen = T.exact_statistic([sp], [expr[sp["sub"]]], restarts)
        T.append_row(path, {"split": i, "r2": r2s[0], "detail": chosen[0],
                            "seconds": round(time.time() - t0, 1)})
        print(f"[{gene}] exact n={n_sub} split {i + 1}/{P.N_SPLITS}  R2={r2s[0]:.4f}  "
              f"kernel={chosen[0]}  ({time.time() - t0:.0f} s)", flush=True)


def svgp_epochs(n_train):
    return max(P.EPOCHS, math.ceil(TARGET_STEPS / math.ceil(n_train / P.BATCH_SIZE)))


def run_svgp(gene, n_sub, max_splits=None, n_blocks=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"n{n_sub}" if not n_blocks or n_blocks == P.N_BLOCKS_A else f"n{n_sub}b{n_blocks}"
    path = OUT_DIR / f"svgp_{gene}_{tag}.csv"
    done = done_splits(path)

    coords, expr = T.load_gene(gene)
    splits = build_splits_n(coords, expr, n_sub, need_val=False, n_blocks=n_blocks)
    for i, sp in enumerate(splits):
        if max_splits is not None and i >= max_splits:
            break
        if i in done:
            continue
        t0 = time.time()
        y = expr[sp["sub"]]
        tr, te = sp["trainval"], sp["test"]
        mu, sd = y[tr].mean(), y[tr].std()
        y_tr, y_te = (y[tr] - mu) / sd, (y[te] - mu) / sd
        epochs = svgp_epochs(len(tr))

        torch.manual_seed(P.SEED + i)
        with contextlib.redirect_stdout(io.StringIO()):
            model, likelihood, _, converged = P.train_svgp(sp["X"][tr], y_tr, epochs=epochs)
        pred, _ = P.predict_svgp(model, likelihood, sp["X"][te])
        r2 = float(r2_score(y_te, pred))

        T.append_row(path, {"split": i, "r2": r2, "detail": f"epochs={epochs};converged={converged}",
                            "seconds": round(time.time() - t0, 1)})
        print(f"[{gene}] svgp n={n_sub} blocks={n_blocks or P.N_BLOCKS_A} split {i + 1}/{P.N_SPLITS}  R2={r2:.4f}  "
              f"epochs={epochs}  converged={converged}  ({time.time() - t0:.0f} s)", flush=True)


def summarize():
    rows = []
    for f in sorted(PERM_DIR.glob("exact_*.csv")):
        if f.name.startswith("summary"):
            continue
        gene = f.stem.split("_", 1)[1]
        d = pd.read_csv(f)
        obs = d[d["b"] == -1]
        if obs.empty:
            continue
        vals = [float(obs[f"r2_split{k}"].iloc[0]) for k in range(1, P.N_SPLITS + 1)]
        rows.append({"gene": gene, "design": "exact_2000", "mean_r2": np.mean(vals),
                     "sd_r2": np.std(vals), "n_splits": len(vals)})
    for f in sorted(OUT_DIR.glob("*_*_n*.csv")):
        family, gene, tag = f.stem.split("_")
        n = int(tag[1:].split("b")[0])
        blocks = int(tag.split("b")[1]) if "b" in tag else P.N_BLOCKS_A
        design = f"{family}_{'full' if n == 0 else n}"
        if blocks != P.N_BLOCKS_A:
            design += f"_b{blocks}"
        d = pd.read_csv(f)
        rows.append({"gene": gene, "design": design,
                     "mean_r2": d["r2"].mean(), "sd_r2": d["r2"].std(ddof=0), "n_splits": len(d)})
    if not rows:
        print("no results found")
        return
    long = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT_DIR / "summary_long.csv", index=False)

    long["cell"] = long.apply(lambda r: f"{r.mean_r2:.3f} ± {r.sd_r2:.3f} (k={r.n_splits})", axis=1)
    order = ["exact_2000", "exact_4000", "exact_6000", "svgp_2000", "svgp_4000", "svgp_6000", "svgp_full"]
    wide = long.pivot(index="gene", columns="design", values="cell")
    cols = [c for c in order if c in wide.columns] + [c for c in wide.columns if c not in order]
    wide = wide[cols]
    wide.to_csv(OUT_DIR / "summary_wide.csv")
    print(wide.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["exact", "svgp", "summary"])
    ap.add_argument("--gene")
    ap.add_argument("--n-sub", type=int, default=2000, help="cells per subsample; 0 = full dataset")
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--max-splits", type=int, default=None, help="run only the first k splits (smoke test)")
    ap.add_argument("--blocks", type=int, default=None, help="k-means blocks per split (default 40)")
    args = ap.parse_args()

    if args.mode == "summary":
        summarize()
        return
    if not args.gene:
        ap.error("--gene is required for exact and svgp")
    if args.mode == "exact":
        run_exact(args.gene, args.n_sub, args.restarts, args.max_splits)
    else:
        run_svgp(args.gene, args.n_sub, args.max_splits, args.blocks)


if __name__ == "__main__":
    main()