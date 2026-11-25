# Bootstrap on validation data set A with Benchmarking::dea.boot, for comparison
# with PLEAP's bootstrap_dea.  BCC (VRS), input orientation, NREP = 2000,
# seeds 1-5.  For each seed and DMU it writes Benchmarking's own bias-corrected
# score (the additive correction on the Shephard scale) and, from the same
# replicates, the estimator PLEAP reports: the mean over replicates of
# theta_hat^2 / theta*_b.  Run from the repository root:
#   Rscript docs/external_reference/bootstrap_benchmarking.R 1 2 3 4 5
# A comes from the Korean Innovation Survey (STEPI) and is not redistributed:
# it is read from, and the per-unit results written to, the folder named by
# PLEAP_PRIVATE_DATA.  The same comparison on public data is reference_sa.R.
suppressPackageStartupMessages(library(Benchmarking))
args <- commandArgs(trailingOnly = TRUE)
seeds <- if (length(args)) as.integer(args) else 1:5
priv <- Sys.getenv("PLEAP_PRIVATE_DATA", "tests/data_private")
d <- read.csv(file.path(priv, "ds_a_dea.csv"))
X <- as.matrix(d[, c("i1", "i2")]); Y <- as.matrix(d[, c("o1")])
out <- NULL
for (s in seeds) {
  set.seed(s)
  t0 <- proc.time()[["elapsed"]]
  b <- dea.boot(X, Y, NREP = 2000, RTS = "vrs", ORIENTATION = "in")
  ratio <- rowMeans(outer(b$eff, rep(1, ncol(b$boot)))^2 / b$boot, na.rm = TRUE)
  out <- rbind(out, data.frame(seed = s, dmu = d$dmu, theta = b$eff,
                               bc_benchmarking = b$eff.bc, bc_ratio = ratio,
                               mean_boot = rowMeans(b$boot, na.rm = TRUE)))
  cat(sprintf("seed %d: %.0f s\n", s, proc.time()[["elapsed"]] - t0))
}
write.csv(out, file.path(priv, sprintf("bootstrap_benchmarking_seeds_%s.csv", paste(seeds, collapse = "-"))), row.names = FALSE)
cat("Benchmarking", as.character(packageVersion("Benchmarking")), "|", R.version.string, "\n")
