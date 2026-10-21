suppressPackageStartupMessages({library(SPARK); library(Matrix)})
m <- as(readMM("results/sparkx/counts_full.mtx"), "CsparseMatrix")
genes <- readLines("results/sparkx/genes_full.txt")
cells <- read.csv("results/sparkx/cells_full.csv")
rownames(m) <- genes
colnames(m) <- cells$cell
loc <- as.matrix(cells[, c("x", "y")])
rownames(loc) <- cells$cell
res <- sparkx(m, loc, numCores = 3, option = "mixture", verbose = FALSE)
out <- data.frame(gene = rownames(res$res_mtest), res$res_mtest)
out$rank <- rank(out$combinedPval, ties.method = "min")
write.csv(out, "results/sparkx/sparkx_full.csv", row.names = FALSE)
cat("genes tested:", nrow(out), "| adjusted p < 0.05:", sum(out$adjustedPval < 0.05, na.rm = TRUE), "\n")
bench <- c("Mbp", "Plp1", "Mobp", "Mog", "Cnp", "Sox10", "Snap25", "Malat1")
print(out[out$gene %in% bench, ], row.names = FALSE)
