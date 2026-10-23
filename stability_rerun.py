from pathlib import Path

import numpy as np
import pandas as pd

import pipeline as pl

BASE = Path("results/baselines")
OUT = Path("results/revision")
GENES = ["Mbp", "Plp1", "Mobp", "Mog", "Cnp", "Sox10", "Snap25", "Malat1"]


def modal_kernels(gene):
    counts = pd.read_csv(BASE / f"exact_{gene}_n2000.csv")["kernel"].value_counts()
    return counts[counts == counts.max()].index.tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    adata, coords = pl.load_data()
    subsamples = pl.draw_disjoint_subsamples(len(coords), pl.N_SUBSAMPLE, pl.N_SPLITS + pl.N_REPEATS)
    stability = subsamples[pl.N_SPLITS:]

    fit_rows, summary_rows = [], []
    for gene in GENES:
        expr = pl.get_gene_expression(adata, gene)
        for kernel in modal_kernels(gene):
            res = pl.robustness_check(coords, expr, gene, kernel, stability)
            ls = np.array(res["length_scales"])
            nz = np.array(res["noise_levels"])
            for i, (a, b) in enumerate(zip(ls, nz)):
                fit_rows.append({"gene": gene, "kernel": kernel, "subsample": i + 1,
                                 "length_scale": a, "noise_level": b})
            summary_rows.append({"gene": gene, "kernel": kernel, "fits": len(ls),
                                 "ls_median": f"{np.median(ls):.3g}",
                                 "ls_range": f"{ls.min():.3g} to {ls.max():.3g}",
                                 "noise_median": f"{np.median(nz):.3g}",
                                 "noise_range": f"{nz.min():.3g} to {nz.max():.3g}",
                                 "ls_below_1e-3": int((ls < 1e-3).sum()),
                                 "ls_above_10": int((ls > 10).sum())})

    pd.DataFrame(fit_rows).to_csv(OUT / "stability_per_fit.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "stability_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
