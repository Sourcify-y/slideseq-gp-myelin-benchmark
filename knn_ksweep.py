"""
kNN k-sweep on the full-data SVGP designs.

Design
  Same splits, standardization and SVGP reference as fulldata_baselines.py. kNN is fit on the
  train+val pool of each split for k in KS and predicts the test cells. The best k per design
  is chosen by mean test R2 across splits, which favors kNN. SVGP minus best-k kNN is
  reported as a paired difference.

Usage
    python3 knn_ksweep.py run --gene Mbp --blocks 150
    python3 knn_ksweep.py run --gene Mbp --blocks 150 --max-splits 1
    python3 knn_ksweep.py check
    python3 knn_ksweep.py summary
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors

import pipeline as P
import permutation_tests as T
from matched_comparison import build_splits_n, done_splits
from fulldata_baselines import GENES, BLOCKS, MATCHED_DIR, OUT_DIR, tag

KS = [10, 30, 100, 300]


def run(gene, blocks, max_splits=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"knnk_{gene}_{tag(blocks)}.csv"
    done = done_splits(path)

    svgp_path = MATCHED_DIR / f"svgp_{gene}_{tag(blocks)}.csv"
    svgp_r2 = pd.read_csv(svgp_path).set_index("split")["r2"] if svgp_path.exists() else {}

    coords, expr = T.load_gene(gene)
    splits = build_splits_n(coords, expr, 0, need_val=False, n_blocks=blocks)
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
        X = sp["X"]

        nn = NearestNeighbors(n_neighbors=max(KS)).fit(X[tr])
        idx = nn.kneighbors(X[te], return_distance=False)
        row = {"split": i}
        for k in KS:
            row[f"knn_k{k}"] = float(r2_score(y_te, y_tr[idx[:, :k]].mean(axis=1)))
        row.update({"n_train": int(len(tr)), "n_test": int(len(te)),
                    "seconds": round(time.time() - t0, 1)})
        T.append_row(path, row)

        ref = svgp_r2[i] if i in svgp_r2 else float("nan")
        res = "  ".join(f"k={k} R2={row[f'knn_k{k}']:.4f}" for k in KS)
        print(f"[{gene}] blocks={blocks} split {i + 1}/{P.N_SPLITS}  {res}  "
              f"SVGP R2={ref:.4f}  ({time.time() - t0:.0f} s)", flush=True)


def ms(x):
    return f"{x.mean():.3f} ± {x.std(ddof=0):.3f}"


def check():
    files = sorted(OUT_DIR.glob("knnk_*.csv"))
    print(f"{len(files)} files (target 23)")
    short = [f.name for f in files if pd.read_csv(f).split.nunique() < 5]
    print("files with fewer than 5 splits:", short if short else "none")


def summarize():
    rows = []
    cols = [f"knn_k{k}" for k in KS]
    for gene in GENES:
        for b in BLOCKS:
            f = OUT_DIR / f"knnk_{gene}_{tag(b)}.csv"
            g = OUT_DIR / f"base_{gene}_{tag(b)}.csv"
            s = MATCHED_DIR / f"svgp_{gene}_{tag(b)}.csv"
            if not (f.exists() and g.exists() and s.exists()):
                continue
            d = pd.read_csv(f).set_index("split")
            d["svgp_r2"] = pd.read_csv(s).set_index("split")["r2"]
            d["idw_r2"] = pd.read_csv(g).set_index("split")["idw_r2"]
            d = d.dropna(subset=["svgp_r2"])
            best = d[cols].mean().idxmax()
            diff = d["svgp_r2"] - d[best]
            row = {"gene": gene, "blocks": b, "svgp": ms(d["svgp_r2"]), "idw": ms(d["idw_r2"])}
            for c in cols:
                row[c] = f"{d[c].mean():.3f}"
            row["best_k"] = best.replace("knn_k", "")
            row["svgp-bestk"] = f"{diff.mean():+.3f} ± {diff.std(ddof=0):.3f}"
            row["svgp>bestk"] = f"{int((diff > 0).sum())}/{len(d)}"
            rows.append(row)
    if not rows:
        print("no results found")
        return
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "summary_knnk.csv", index=False)
    print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "check", "summary"])
    ap.add_argument("--gene")
    ap.add_argument("--blocks", type=int, default=None, help="k-means blocks per split (default 40)")
    ap.add_argument("--max-splits", type=int, default=None, help="run only the first k splits (smoke test)")
    args = ap.parse_args()

    if args.mode == "check":
        check()
        return
    if args.mode == "summary":
        summarize()
        return
    if not args.gene:
        ap.error("--gene is required for run")
    run(args.gene, args.blocks or P.N_BLOCKS_A, args.max_splits)


if __name__ == "__main__":
    main()
