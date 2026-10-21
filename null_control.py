"""
Synthetic null control on the full-data designs.

Design
  Expression of one gene is permuted across cells (seed = SEED + 100 + split), which keeps the
  marginal distribution and removes spatial structure. Splits, standardization, SVGP training
  (400 inducing points, epochs from matched_comparison.svgp_epochs), IDW and kNN (k = 100)
  follow the real-data runs. Expected: R2 near or below 0 for all methods and 2-sigma
  coverage near 0.95. --real refits the SVGP on the unpermuted gene as a positive control
  and does not write results.

Usage
    python3 null_control.py run --gene Mbp --blocks 150 --real --max-splits 1
    python3 null_control.py run --gene Mbp --blocks 150 --max-splits 1
    python3 null_control.py summary
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

import pipeline as P
import permutation_tests as T
from matched_comparison import build_splits_n, done_splits, svgp_epochs
from fulldata_baselines import GENES, MATCHED_DIR, tag, idw_chunked

OUT_DIR = Path("results") / "null"


def run(gene, blocks, max_splits=None, real_data=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"null_{gene}_{tag(blocks)}.csv"
    done = set() if real_data else done_splits(path)

    svgp_path = MATCHED_DIR / f"svgp_{gene}_{tag(blocks)}.csv"
    ref = pd.read_csv(svgp_path).set_index("split")["r2"] if svgp_path.exists() else {}

    coords, expr = T.load_gene(gene)
    splits = build_splits_n(coords, expr, 0, need_val=False, n_blocks=blocks)
    for i, sp in enumerate(splits):
        if max_splits is not None and i >= max_splits:
            break
        if i in done:
            continue
        t0 = time.time()
        y = expr[sp["sub"]]
        if not real_data:
            y = np.random.RandomState(P.SEED + 100 + i).permutation(y)
        tr, te = sp["trainval"], sp["test"]
        mu, sd = y[tr].mean(), y[tr].std()
        y_tr, y_te = (y[tr] - mu) / sd, (y[te] - mu) / sd
        X = sp["X"]

        r2_idw = float(r2_score(y_te, idw_chunked(X[tr], y_tr, X[te])))
        knn = KNeighborsRegressor(n_neighbors=100).fit(X[tr], y_tr)
        r2_knn = float(r2_score(y_te, knn.predict(X[te])))

        torch.manual_seed(sp["seed"])
        model, lik, _, _ = P.train_svgp(X[tr], y_tr, epochs=svgp_epochs(len(tr)))
        mean, std = P.predict_svgp(model, lik, X[te])
        r2_svgp = float(r2_score(y_te, mean))
        err = np.abs(y_te - mean)
        cov1 = float(np.mean(err <= std))
        cov2 = float(np.mean(err <= 2 * std))
        ref_r2 = float(ref[i]) if i in ref else float("nan")

        label = "real refit" if real_data else "null"
        print(f"[{gene}] blocks={blocks} split {i + 1}/{P.N_SPLITS} {label}  "
              f"SVGP R2={r2_svgp:.4f}  IDW R2={r2_idw:.4f}  kNN100 R2={r2_knn:.4f}  "
              f"cov1={cov1:.3f}  cov2={cov2:.3f}  matched SVGP R2={ref_r2:.4f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
        if not real_data:
            T.append_row(path, {"split": i, "null_svgp_r2": r2_svgp, "null_idw_r2": r2_idw,
                                "null_knn100_r2": r2_knn, "cov1": cov1, "cov2": cov2,
                                "real_svgp_r2": ref_r2, "n_train": int(len(tr)),
                                "n_test": int(len(te)), "seconds": round(time.time() - t0, 1)})


def ms(x):
    return f"{x.mean():.3f} ± {x.std(ddof=0):.3f}"


def summarize():
    rows = []
    for f in OUT_DIR.glob("null_*.csv"):
        _, gene, t = f.stem.split("_")
        blocks = 40 if t == "n0" else int(t[3:])
        d = pd.read_csv(f)
        diff = d["real_svgp_r2"] - d["null_svgp_r2"]
        rows.append({"gene": gene, "blocks": blocks, "k": len(d),
                     "null_svgp": ms(d["null_svgp_r2"]),
                     "null_svgp_max": f"{d['null_svgp_r2'].max():.3f}",
                     "null_idw": ms(d["null_idw_r2"]),
                     "null_knn100": ms(d["null_knn100_r2"]),
                     "cov1": f"{d['cov1'].mean():.3f}",
                     "cov2": f"{d['cov2'].mean():.3f}",
                     "real_svgp": ms(d["real_svgp_r2"]),
                     "real-null": f"{diff.mean():+.3f} ± {diff.std(ddof=0):.3f}",
                     "real>null": f"{int((diff > 0).sum())}/{len(d)}"})
    if not rows:
        print("no results found")
        return
    out = pd.DataFrame(rows)
    out["order"] = out["gene"].map({g: i for i, g in enumerate(GENES)})
    out = out.sort_values(["order", "blocks"]).drop(columns="order")
    out.to_csv(OUT_DIR / "summary.csv", index=False)
    print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary"])
    ap.add_argument("--gene")
    ap.add_argument("--blocks", type=int, default=None, help="k-means blocks per split (default 40)")
    ap.add_argument("--max-splits", type=int, default=None, help="run only the first k splits (smoke test)")
    ap.add_argument("--real", action="store_true", help="refit on unpermuted expression, no output file")
    args = ap.parse_args()

    if args.mode == "summary":
        summarize()
        return
    if not args.gene:
        ap.error("--gene is required for run")
    run(args.gene, args.blocks or P.N_BLOCKS_A, args.max_splits, args.real)


if __name__ == "__main__":
    main()
