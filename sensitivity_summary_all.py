import numpy as np, pandas as pd, glob, itertools, os
def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])
def fmt(v):
    v = np.array(v, float); ok = v[np.isfinite(v)]
    if len(ok) == 0:
        return f"no defined pairs 0/{len(v)}"
    return f"median {np.median(ok):+.3f} [{ok.min():+.3f}, {ok.max():+.3f}]  defined {len(ok)}/{len(v)}"
rows = []
for f in sorted(glob.glob("results/sensitivity/maps_*.npz")):
    g = os.path.basename(f)[5:-4]
    z = np.load(f); k = z.files
    ex = [x for x in k if x.startswith("exact_s")]
    pa = [x for x in k if x.startswith("svgp_p")]
    M = [x for x in ["svgp_p0", "svgp_M200", "svgp_M800"] if x in k]
    sd = [x for x in ["svgp_p0", "svgp_seed1", "svgp_seed2"] if x in k]
    C = lambda pairs: [corr(z[a], z[b]) for a, b in pairs]
    const = sum(1 for x in ex if np.nanstd(z[x]) < 1e-9)
    comps = {
      "exact vs exact (independent subsamples)": C(itertools.combinations(ex, 2)),
      "SVGP vs SVGP (block partitions)": C(itertools.combinations(pa, 2)),
      "SVGP vs SVGP (200, 400, 800 inducing points)": C(itertools.combinations(M, 2)),
      "SVGP vs SVGP (seeds)": C(itertools.combinations(sd, 2)),
      "exact vs main SVGP": C([(a, "svgp_p0") for a in ex]),
      "exact vs SVGP (all exact x all partitions)": C(itertools.product(ex, pa)),
    }
    print(f"\n== {g} ==  keys={len(k)}  constant exact maps: {const}/{len(ex)}")
    fp = f"results/sensitivity/fits_{g}.csv"
    if os.path.exists(fp):
        d = pd.read_csv(fp)
        print("exact R2:", [round(x, 3) for x in d[d.fit.str.startswith("exact")].test_r2])
        print("SVGP partition R2:", [round(x, 3) for x in d[d.fit.str.startswith("svgp_p")].test_r2])
    for name, v in comps.items():
        print(f"{name:<46} {fmt(v)}")
        rows.append({"gene": g, "comparison": name, "summary": fmt(v)})
pd.DataFrame(rows).to_csv("results/sensitivity/summary_all.csv", index=False)
print("\nsaved results/sensitivity/summary_all.csv")
