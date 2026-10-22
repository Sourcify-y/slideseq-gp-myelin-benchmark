# SPARK-X on raw Slide-seqV2 counts for all genes passing a detection filter.
#
# Usage:
#   Rscript 02_run_sparkx.R <input_dir> <output_csv> [cores]
#   input_dir: directory written by 01_build_raw_counts.py
#
# Requires the SPARK package: devtools::install_github("xzhoulab/SPARK")

suppressPackageStartupMessages({
  library(Matrix)
  library(SPARK)
})

args    <- commandArgs(trailingOnly = TRUE)
indir   <- if (length(args) >= 1) args[1] else "export"
outfile <- if (length(args) >= 2) args[2] else "sparkx_results.csv"
ncores  <- if (length(args) >= 3) as.integer(args[3]) else 4

# Genes detected in fewer than this fraction of beads are dropped
# (42 beads at n = 41,786).
MIN_FRAC_BEADS <- 0.001

cat("Reading", indir, "\n")
counts   <- readMM(file.path(indir, "counts.mtx"))
barcodes <- readLines(file.path(indir, "barcodes.txt"))
genes    <- readLines(file.path(indir, "genes.txt"))
locs     <- read.csv(file.path(indir, "locations.csv"), stringsAsFactors = FALSE)

stopifnot(nrow(counts) == length(barcodes),
          ncol(counts) == length(genes))

rownames(counts) <- barcodes
colnames(counts) <- genes

# Align locations to the row order of the count matrix.
rownames(locs) <- locs$index
locs <- locs[barcodes, , drop = FALSE]
stopifnot(!any(is.na(locs$x)), !any(is.na(locs$y)))
stopifnot(identical(rownames(locs), rownames(counts)))

cat(sprintf("Loaded %d beads x %d genes\n", nrow(counts), ncol(counts)))

# Drop sparsely detected genes.
detected <- Matrix::colSums(counts > 0)
keep <- detected >= (MIN_FRAC_BEADS * nrow(counts))
cat(sprintf("Keeping %d of %d genes detected in >= %.3f%% of beads\n",
            sum(keep), length(keep), 100 * MIN_FRAC_BEADS))
counts <- counts[, keep, drop = FALSE]

# SPARK-X takes a genes x cells matrix.
cmat <- Matrix::t(counts)
info <- as.matrix(locs[, c("x", "y")])
storage.mode(info) <- "double"

cat(sprintf("Running SPARK-X on %d genes x %d cells, %d cores\n",
            nrow(cmat), ncol(cmat), ncores))
t0 <- Sys.time()

res <- sparkx(cmat, info, numCores = ncores, option = "mixture")

cat("Elapsed:", format(Sys.time() - t0), "\n")

out <- data.frame(
  gene         = rownames(res$res_mtest),
  combinedPval = res$res_mtest$combinedPval,
  adjustedPval = res$res_mtest$adjustedPval,
  stringsAsFactors = FALSE
)
out <- out[order(out$adjustedPval), ]
write.csv(out, outfile, row.names = FALSE)

cat("Wrote", outfile, "\n")
cat(sprintf("Genes with adjusted p < 0.05: %d of %d (%.1f%%)\n",
            sum(out$adjustedPval < 0.05, na.rm = TRUE), nrow(out),
            100 * mean(out$adjustedPval < 0.05, na.rm = TRUE)))

# Panel genes with their rank among all tested genes.
panel <- c("Mbp", "Plp1", "Mobp", "Mog", "Cnp", "Sox10", "Snap25", "Malat1")
sub <- out[match(panel, out$gene), ]
sub$rank_of_all <- match(panel, out$gene)
print(sub, row.names = FALSE)

missing <- panel[is.na(sub$gene)]
if (length(missing)) {
  cat("Panel genes not found in genes.txt:", paste(missing, collapse = ", "), "\n")
}
