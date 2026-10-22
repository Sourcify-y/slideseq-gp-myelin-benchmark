"""
Build the raw Slide-seqV2 count matrix for puck Puck_200115_08, compare its barcodes with the
squidpy object, and export the counts for SPARK-X.

Inputs (Broad Single Cell Portal, Slide-seqV2 study):
    Puck_200115_08.digital_expression.txt.gz
    Puck_200115_08_bead_locations.csv

Usage:
    python3 01_build_raw_counts.py --dge <digital_expression.txt.gz> --locs <bead_locations.csv> --outdir export

Outputs (in --outdir):
    counts.mtx        beads x genes sparse counts (MatrixMarket)
    barcodes.txt      bead barcodes, row order of counts.mtx
    genes.txt         gene symbols, column order of counts.mtx
    locations.csv     barcode, x, y in the order of barcodes.txt
    match_report.txt  barcode overlap between the raw puck and the squidpy object
"""

import argparse
import gzip
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def sniff_sep(path, n_lines=2):
    """Guess the delimiter of a (possibly gzipped) text table."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        head = [fh.readline() for _ in range(n_lines)]
    sample = "".join(head)
    counts = {",": sample.count(","), "\t": sample.count("\t"), " ": sample.count(" ")}
    sep = max(counts, key=counts.get)
    if counts[sep] == 0:
        raise ValueError(f"Could not determine delimiter for {path}")
    return sep


def load_dge(path):
    """
    Load a Slide-seq digital expression matrix.

    Slide-seq DGE files are conventionally GENES x BEADS: first column is the
    gene symbol, header row is bead barcodes. We return beads x genes, which
    is the AnnData convention and what the rest of this script assumes.
    """
    sep = sniff_sep(path)
    print(f"Reading DGE {path} (delimiter {sep!r})... this can take minutes.")
    df = pd.read_csv(path, sep=sep, index_col=0)
    print(f"  raw shape (as stored): {df.shape}")

    # Orientation check. Slide-seqV2 pucks have tens of thousands of beads and
    # ~20-30k genes, so shape alone is ambiguous. Use the index contents:
    # gene symbols are alphabetic, barcodes are long uppercase ACGT strings.
    idx_sample = [str(i) for i in df.index[:50]]
    looks_like_barcodes = all(
        set(s.upper()) <= set("ACGTN") and len(s) >= 10 for s in idx_sample
    )
    if looks_like_barcodes:
        print("  index looks like BARCODES -> file is beads x genes, no transpose")
        mat = df
    else:
        print("  index looks like GENE SYMBOLS -> file is genes x beads, transposing")
        mat = df.T

    print(f"  final shape (beads x genes): {mat.shape}")
    return mat


def load_locations(path):
    """Load bead locations, tolerating the several column namings in the wild.
    Handles files where the header row uses a different delimiter than the
    data rows (observed: tab-separated header, comma-separated body)."""
    with open(path, "r") as fh:
        header_line = fh.readline()
    header_sep = "\t" if header_line.count("\t") >= header_line.count(",") else ","
    col_names = [c.strip() for c in header_line.strip().split(header_sep)]

    body_sep = sniff_sep(path)  # sniffs across first 2 lines; body dominates at scale
    df = pd.read_csv(path, sep=body_sep, skiprows=1, header=None, names=col_names)
    print(f"Reading locations {path}: shape={df.shape}")
    print(f"  header delimiter={header_sep!r}  body delimiter={body_sep!r}")
    print(f"  columns: {list(df.columns)}")

    cols_lower = {c.lower().strip(): c for c in df.columns}

    bc_col = next(
        (cols_lower[k] for k in ("barcode", "barcodes", "bead", "cell", "index", "unnamed: 0")
         if k in cols_lower),
        df.columns[0],
    )
    x_col = next((cols_lower[k] for k in ("xcoord", "x", "x_coord", "row") if k in cols_lower), None)
    y_col = next((cols_lower[k] for k in ("ycoord", "y", "y_coord", "col") if k in cols_lower), None)

    if x_col is None or y_col is None:
        sys.exit(f"Could not find x/y columns in {path}. Columns were: {list(df.columns)}\n"
                 "Edit load_locations() to name them explicitly.")

    print(f"  using barcode={bc_col!r}  x={x_col!r}  y={y_col!r}")
    out = df[[bc_col, x_col, y_col]].copy()
    out.columns = ["barcode", "x", "y"]
    out["barcode"] = out["barcode"].astype(str)
    return out.set_index("barcode")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dge", required=True, help="digital expression matrix (.txt/.txt.gz/.csv)")
    ap.add_argument("--locs", required=True, help="bead locations csv")
    ap.add_argument("--outdir", default="export")
    ap.add_argument("--squidpy-barcodes", default="squidpy_barcodes.txt",
                    help="barcode list written by script 00")
    ap.add_argument("--restrict-to-squidpy", action="store_true",
                    help="keep only beads also present in the squidpy object "
                         "(restricts SPARK-X to the cells used in the GP analyses)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    counts = load_dge(args.dge)          # beads x genes
    locs = load_locations(args.locs)

    # ---- join counts to coordinates -------------------------------------
    common = counts.index.intersection(locs.index)
    print(f"\nBeads in DGE: {len(counts.index)}")
    print(f"Beads in locations: {len(locs.index)}")
    print(f"Beads in both: {len(common)}")
    if len(common) == 0:
        sys.exit("No barcode overlap between DGE and locations. Wrong file pair?")
    counts = counts.loc[common]
    locs = locs.loc[common]

    # ---- verify this is the same puck squidpy used ----------------------
    report = []
    if os.path.exists(args.squidpy_barcodes):
        sq_bc = pd.Index(np.loadtxt(args.squidpy_barcodes, dtype=str))
        overlap = counts.index.intersection(sq_bc)
        frac_of_squidpy = len(overlap) / max(len(sq_bc), 1)
        report.append(f"squidpy beads                     : {len(sq_bc)}")
        report.append(f"raw-puck beads (DGE n locations)  : {len(counts.index)}")
        report.append(f"beads in both                     : {len(overlap)}")
        report.append(f"fraction of squidpy beads matched : {frac_of_squidpy:.4f}")
        print("\n--- PUCK MATCH ---")
        print("\n".join(report))

        if frac_of_squidpy < 0.90:
            print("\n*** WARNING: under 90% of squidpy beads matched. ***")
            print("This is probably a different puck, or barcodes are formatted")
            print("differently (suffixes like '-1'). Resolve before reporting.")
            print("Examples -- squidpy:", list(sq_bc[:3]), " raw:", list(counts.index[:3]))
        else:
            print("Match looks good.")

        if args.restrict_to_squidpy:
            counts = counts.loc[overlap]
            locs = locs.loc[overlap]
            print(f"\nRestricted to squidpy beads: {counts.shape}")
            report.append(f"RESTRICTED to squidpy beads: final n = {counts.shape[0]}")
    else:
        print(f"\n(No {args.squidpy_barcodes}; skipping puck-match check. "
              "Run script 00 first to enable it.)")

    # ---- sanity checks on the counts ------------------------------------
    X = sp.csr_matrix(counts.values.astype(np.float64))
    nz = X.data
    print(f"\nCount matrix: {X.shape[0]} beads x {X.shape[1]} genes, "
          f"{X.nnz} non-zero ({100 * X.nnz / (X.shape[0] * X.shape[1]):.3f}% dense)")
    print(f"  min={nz.min()} max={nz.max()} all_integer={np.allclose(nz, np.round(nz))}")
    if not np.allclose(nz, np.round(nz)):
        print("  *** These are NOT integers. SPARK-X expects raw counts. ***")

    report.append(f"final matrix: {X.shape[0]} beads x {X.shape[1]} genes")
    report.append(f"median UMIs/bead: {np.median(np.asarray(X.sum(axis=1)).ravel()):.1f}")
    report.append(f"median genes/bead: {np.median(np.diff(X.indptr)):.1f}")

    # ---- write ----------------------------------------------------------
    sio.mmwrite(os.path.join(args.outdir, "counts.mtx"), X, field="integer")
    np.savetxt(os.path.join(args.outdir, "barcodes.txt"), counts.index.values, fmt="%s")
    np.savetxt(os.path.join(args.outdir, "genes.txt"), counts.columns.values, fmt="%s")
    locs.reset_index().to_csv(os.path.join(args.outdir, "locations.csv"), index=False)
    with open(os.path.join(args.outdir, "match_report.txt"), "w") as fh:
        fh.write("\n".join(report) + "\n")

    print(f"\nWrote counts.mtx, barcodes.txt, genes.txt, locations.csv, "
          f"match_report.txt to {args.outdir}/")
    print("Next: 02_run_sparkx.R")


if __name__ == "__main__":
    main()
