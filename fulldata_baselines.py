"""
kNN and IDW baselines on the full-data SVGP designs.

Design
  Same splits as results/matched/svgp_<gene>_n0*.csv: full dataset, 5 splits, k-means
  blocks, 20% of blocks held out, seeds as in matched_comparison.py. Training cells are the
  train+val pool of each split. kNN (k = 10) and IDW (power 2, pipeline.idw_predict) predict
  the test cells. R2 is computed on expression standardized by the training mean and SD,
  as in the SVGP runs. The SVGP R2 of the same split is read from the matched results.

Usage
    python3 fulldata_baselines.py run --gene Mbp --blocks 150
    python3 fulldata_baselines.py run --gene Mbp --blocks 150 --max-splits 1
    python3 fulldata_baselines.py summary
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
from matched_comparison import build_splits_n, done_splits

OUT_DIR = Path("results") / "fullbase"
MATCHED_DIR = Path("results") / "matched"
GENES = ["Mbp", "Plp1", "Mobp", "Mog", "Cnp", "Sox10", "Snap25", "Malat1"]
BLOCKS = [10, 20, 40, 80, 150, 300]
CHUNK = 1000


def tag(blocks):
    return "n0" if blocks == P.N_BLOCKS_A else f"n0b{blocks}"


def idw_chunked(X_train, y_train, X_query):
    out = np.empty(len(X_query))
    for s in range(0, len(X_query), CHUNK):
        out[s:s + CHUNK] = np.asarray(P.idw_predict(X_train, y_train, X_query[s:s + CHUNK])).ravel()
    return out


def run(gene, blocks, max_splits=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"base_{gene}_{tag(blocks)}.csv"
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

        knn = KNeighborsRegressor(n_neighbors=10).fit(X[tr], y_tr)
        r2_knn = float(r2_score(y_te, knn.predict(X[te])))
        r2_idw = float(r2_score(y_te, idw_chunked(X[tr], y_tr, X[te])))

        T.append_row(path, {"split": i, "knn_r2": r2_knn, "idw_r2": r2_idw,
                            "n_train": int(len(tr)), "n_test": int(len(te)),
                            "seconds": round(time.time() - t0, 1)})
        ref = svgp_r2[i] if i in svgp_r2 else float("nan")
        print(f"[{gene}] blocks={blocks} split {i + 1}/{P.N_SPLITS}  kNN R2={r2_knn:.4f}  "
              f"IDW R2={r2_idw:.4f}  SVGP R2={ref:.4f}  ({time.time() - t0:.0f} s)", flush=True)


def summarize():
    rows = []
    for gene in GENES:
        for b in BLOCKS:
            f = OUT_DIR / f"base_{gene}_{tag(b)}.csv"
            s = MATCHED_DIR / f"svgp_{gene}_{tag(b)}.csv"
            if not (f.exists() and s.exists()):
                continue
            d = pd.read_csv(f).set_index("split")
            d["svgp_r2"] = pd.read_csv(s).set_index("split")["r2"]
            d = d.dropna(subset=["svgp_r2"])
            diff = d["svgp_r2"] - d["idw_r2"]

            def ms(c):
                return f"{d[c].mean():.3f} ± {d[c].std(ddof=0):.3f}"

            rows.append({"gene": gene, "blocks": b, "k": len(d), "svgp": ms("svgp_r2"),
                         "idw": ms("idw_r2"), "knn": ms("knn_r2"),
                         "svgp-idw": f"{diff.mean():+.3f} ± {diff.std(ddof=0):.3f}",
                         "svgp>idw": f"{int((diff > 0).sum())}/{len(d)}"})
    if not rows:
        print("no results found")
        return
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "summary.csv", index=False)
    print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary"])
    ap.add_argument("--gene")
    ap.add_argument("--blocks", type=int, default=None, help="k-means blocks per split (default 40)")
    ap.add_argument("--max-splits", type=int, default=None, help="run only the first k splits (smoke test)")
    args = ap.parse_args()

    if args.mode == "summary":
        summarize()
        return
    if not args.gene:
        ap.error("--gene is required for run")
    run(args.gene, args.blocks or P.N_BLOCKS_A, args.max_splits)


if __name__ == "__main__":
    main()
