"""
Summaries from existing result files.

Reads results/baselines/exact_<gene>_n2000.csv (8 genes x 5 independent splits) and
results/perm/exact_<gene>.csv (kernel_split columns, b = -1 is the real-label fit).

Usage
    python3 revision_summaries.py kernels
    python3 revision_summaries.py hyper
    python3 revision_summaries.py calibration
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path("results/baselines")
PERM = Path("results/perm")
OUT = Path("results/revision")
GENES = ["Mbp", "Plp1", "Mobp", "Mog", "Cnp", "Sox10", "Snap25", "Malat1"]
SHORT = {"RBF": "RBF", "Matern(nu=1.5)": "M15", "Matern(nu=2.5)": "M25"}


def load(g):
    f = BASE / f"exact_{g}_n2000.csv"
    return pd.read_csv(f).sort_values("split") if f.exists() else None


def fmt(x):
    x = np.asarray(x, float)
    k, m = len(x), x.mean()
    h = stats.t.ppf(0.975, k - 1) * x.std(ddof=1) / np.sqrt(k) if k > 1 else np.nan
    return f"{m:.3f} [{m - h:.3f}, {m + h:.3f}]"


def kernels():
    rows = []
    for g in GENES:
        d, pf = load(g), PERM / f"exact_{g}.csv"
        if d is None or not pf.exists():
            continue
        p = pd.read_csv(pf)
        kc = [c for c in p.columns if c.startswith("kernel_split")]
        obs = p[p["b"] == -1][kc].iloc[0].tolist()
        null = p[p["b"] >= 0][kc].to_numpy().ravel()
        real = d["kernel"].tolist()
        row = {"gene": g, "real_fits": len(real)}
        row.update({f"real_{s}": real.count(k) for k, s in SHORT.items()})
        row["null_fits"] = len(null)
        row.update({f"null_{s}_pct": round(100 * float(np.mean(null == k))) for k, s in SHORT.items()})
        row["real_equals_perm_obs"] = real == obs
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "kernel_counts.csv", index=False)
    print(out.to_string(index=False))
    print("total real fits:", int(out["real_fits"].sum()),
          {s: int(out[f"real_{s}"].sum()) for s in SHORT.values()})


def hyper():
    rows = []
    for g in GENES:
        d = load(g)
        if d is None:
            continue
        ls, nz = d["length_scale"], d["noise_level"]
        rows.append({"gene": g, "k": len(d), "ls_median": f"{ls.median():.3g}",
                     "ls_range": f"{ls.min():.3g} to {ls.max():.3g}",
                     "noise_median": f"{nz.median():.3g}",
                     "noise_range": f"{nz.min():.3g} to {nz.max():.3g}",
                     "ls_below_1e-3": int((ls < 1e-3).sum()), "ls_above_10": int((ls > 10).sum())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "hyperparameters.csv", index=False)
    print("length scale: mean over the 2 ARD dimensions, standardized coordinates")
    print(out.to_string(index=False))


def calibration():
    ds = {g: load(g) for g in GENES if load(g) is not None}
    ds["ALL"] = pd.concat(ds.values(), ignore_index=True)
    cov, wid = [], []
    for g, d in ds.items():
        cov.append({"gene": g, "k": len(d), "gp_cov1": fmt(d["gp_cov1"]), "knn_cov1": fmt(d["knn_cov1"]),
                    "gp_cov2": fmt(d["gp_cov2"]), "knn_cov2": fmt(d["knn_cov2"])})
        wid.append({"gene": g, "gp_width95": f"{d['gp_width95'].mean():.3f}",
                    "knn_width95": f"{d['knn_width95'].mean():.3f}",
                    "gp/knn": fmt(d["gp_width95"] / d["knn_width95"])})
    cov, wid = pd.DataFrame(cov), pd.DataFrame(wid)
    cov.to_csv(OUT / "calibration_coverage.csv", index=False)
    wid.to_csv(OUT / "calibration_width.csv", index=False)
    print("coverage, mean [95% t-interval over splits]; nominal 0.683 (1 sigma), 0.954 (2 sigma)")
    print(cov.to_string(index=False))
    print("\nmean 95% interval width (standardized units) and per-split ratio GP / kNN")
    print(wid.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["kernels", "hyper", "calibration"])
    {"kernels": kernels, "hyper": hyper, "calibration": calibration}[ap.parse_args().mode]()
