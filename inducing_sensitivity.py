"""
Inducing-point sensitivity for the SVGP (reviewer item 11).

Design
  Full dataset, 5 splits, 40 blocks, 20% of blocks held out. These are the splits and
  seeds of results/matched/svgp_<gene>_n0.csv, which holds the M = 400 result.
  Only the number of inducing points changes. Gradient steps are held at the M = 400 run.

Usage
    python3 inducing_sensitivity.py run --gene Mbp --m 800
    python3 inducing_sensitivity.py run --gene Mbp --m 1600 --max-splits 1
    python3 inducing_sensitivity.py summary
"""
import argparse
import contextlib
import io
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import r2_score

import pipeline as P
import permutation_tests as T
from matched_comparison import build_splits_n, done_splits, svgp_epochs

OUT_DIR = Path("results") / "inducing"
MATCHED_DIR = Path("results") / "matched"
GENES = ["Mbp", "Plp1", "Snap25"]
M_VALUES = [200, 400, 800, 1600]


def run(gene, m, max_splits=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"svgp_{gene}_m{m}.csv"
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
        tr, te = sp["trainval"], sp["test"]
        mu, sd = y[tr].mean(), y[tr].std()
        y_tr, y_te = (y[tr] - mu) / sd, (y[te] - mu) / sd
        epochs = svgp_epochs(len(tr))

        torch.manual_seed(P.SEED + i)
        with contextlib.redirect_stdout(io.StringIO()):
            model, likelihood, _, converged = P.train_svgp(
                sp["X"][tr], y_tr, n_inducing=m, epochs=epochs)
        pred, _ = P.predict_svgp(model, likelihood, sp["X"][te])
        r2 = float(r2_score(y_te, pred))

        T.append_row(path, {"split": i, "r2": r2,
                            "detail": f"M={m};epochs={epochs};converged={converged}",
                            "seconds": round(time.time() - t0, 1)})
        print(f"[{gene}] svgp M={m} split {i + 1}/{P.N_SPLITS}  R2={r2:.4f}  "
              f"epochs={epochs}  converged={converged}  ({time.time() - t0:.0f} s)", flush=True)


def summarize():
    rows = []
    for gene in GENES:
        base_f = MATCHED_DIR / f"svgp_{gene}_n0.csv"
        if not base_f.exists():
            continue
        base = pd.read_csv(base_f).set_index("split")
        for m in M_VALUES:
            f = base_f if m == 400 else OUT_DIR / f"svgp_{gene}_m{m}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f).set_index("split")
            delta = (d["r2"] - base["r2"]).dropna()
            rows.append({
                "gene": gene, "M": m, "k": len(d),
                "r2": f"{d['r2'].mean():.3f} ± {d['r2'].std(ddof=0):.3f}",
                "delta_vs_400": f"{delta.mean():+.3f} ± {delta.std(ddof=0):.3f}",
                "converged": f"{d['detail'].str.contains('converged=True').sum()}/{len(d)}",
                "sec": round(d["seconds"].mean()),
            })
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
    ap.add_argument("--m", type=int, help="number of inducing points")
    ap.add_argument("--max-splits", type=int, default=None, help="run only the first k splits (smoke test)")
    args = ap.parse_args()

    if args.mode == "summary":
        summarize()
        return
    if not args.gene or not args.m:
        ap.error("--gene and --m are required for run")
    run(args.gene, args.m, args.max_splits)


if __name__ == "__main__":
    main()
