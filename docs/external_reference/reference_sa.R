# Reference values for the synthetic data set SA (tests/data/sa_dea.csv and
# sa_mpi.csv; tests/data/make_synthetic_data.py) from R's Benchmarking package.
# Run from the repository root:
#   Rscript docs/external_reference/reference_sa.R            # DEA and Malmquist
#   Rscript docs/external_reference/reference_sa.R boot       # also the bootstrap, ~1 min
#
# Writes
#   tests/data/sa_benchmarking_dea.csv        CCR and BCC, input and output orientation:
#                                             score, slacks, projections, peers
#   tests/data/sa_benchmarking_mpi.csv        the four distance functions of the
#                                             Malmquist index, CRS and VRS, both orientations
#   tests/data/sa_benchmarking_bootstrap.csv  dea.boot, BCC input orientation, NREP = 2000,
#                                             seeds 1-5, and Benchmarking's bandwidth (with "boot")
suppressPackageStartupMessages(library(Benchmarking))
args <- commandArgs(trailingOnly = TRUE)
num <- function(x) sprintf("%.17g", x)

# ---- DEA: scores, second-phase slacks, projections, peers ----------------------
d <- read.csv("tests/data/sa_dea.csv")
X <- as.matrix(d[, c("i1", "i2")]); Y <- as.matrix(d[, "o1", drop = FALSE])
out <- NULL
for (rts in c("crs", "vrs")) for (ori in c("in", "out")) {
  e <- dea(X, Y, RTS = rts, ORIENTATION = ori, SLACK = TRUE)
  # Benchmarking reports Farrell efficiencies: theta <= 1 for input orientation,
  # phi >= 1 for output orientation.  PLEAP and MaxDEA report 1 / phi.
  score <- if (ori == "in") e$eff else 1 / e$eff
  sx <- as.matrix(e$sx); sy <- as.matrix(e$sy)
  px <- if (ori == "in") X * e$eff - sx else X - sx
  py <- if (ori == "in") Y + sy else Y * e$eff + sy
  lam <- e$lambda                      # from the slack (second-phase) program
  peers <- apply(lam, 1, function(l) {
    j <- which(l > 1e-9)
    paste(sprintf("%s:%.15g", d$dmu[j], l[j]), collapse = ";")
  })
  out <- rbind(out, data.frame(
    model = ifelse(rts == "crs", "CCR", "BCC"), orient = ifelse(ori == "in", "input", "output"),
    dmu = d$dmu, score = num(score),
    slack_i1 = num(sx[, 1]), slack_i2 = num(sx[, 2]), slack_o1 = num(sy[, 1]),
    proj_i1 = num(px[, 1]), proj_i2 = num(px[, 2]), proj_o1 = num(py[, 1]),
    peers = peers))
}
write.csv(out, "tests/data/sa_benchmarking_dea.csv", row.names = FALSE, quote = FALSE)

# ---- Malmquist: the four distances for each pair of adjacent periods ----------
# malmq's own index is the input-oriented Faere-Grosskopf-Norris-Zhang index on
# Farrell efficiencies; the file keeps the distances, so the index can be formed
# under either orientation (tests/test_malmquist.py does so).  An infeasible
# cross-period program comes back as Inf.
p <- read.csv("tests/data/sa_mpi.csv")
periods <- sort(unique(p$period))
mo <- NULL
for (k in 2:length(periods)) {
  a <- p[p$period == periods[k - 1], ]; b <- p[p$period == periods[k], ]
  stopifnot(identical(a$dmu, b$dmu))
  for (rts in c("crs", "vrs")) for (ori in c("in", "out")) {
    m <- malmq(as.matrix(a[, c("i1", "i2")]), as.matrix(a[, "o1", drop = FALSE]), a$dmu,
               as.matrix(b[, c("i1", "i2")]), as.matrix(b[, "o1", drop = FALSE]), b$dmu,
               RTS = rts, ORIENTATION = ori)
    stopifnot(identical(as.character(m$id), as.character(a$dmu)))
    mo <- rbind(mo, data.frame(
      model = ifelse(rts == "crs", "CCR", "BCC"), orient = ifelse(ori == "in", "input", "output"),
      period = periods[k], dmu = a$dmu,
      e00 = num(m$e00), e01 = num(m$e01), e10 = num(m$e10), e11 = num(m$e11),
      m_benchmarking = num(m$m)))
  }
}
write.csv(mo, "tests/data/sa_benchmarking_mpi.csv", row.names = FALSE, quote = FALSE)

# ---- Bootstrap ------------------------------------------------------------------
# For each seed and unit: Benchmarking's own bias-corrected score (its additive
# correction on the Shephard scale) and, from the same replicates, the estimator
# PLEAP reports, the mean over replicates of theta_hat^2 / theta*_b.
if ("boot" %in% args) {
  # Benchmarking's bandwidth, as dea.boot itself prints it with LP = TRUE
  options(digits = 15)
  set.seed(1)
  lp_out <- capture.output(dea.boot(X, Y, NREP = 2, RTS = "vrs", ORIENTATION = "in", LP = TRUE))
  h <- as.numeric(sub(".*Bandwidth = *", "", grep("Bandwidth", lp_out, value = TRUE)))
  bo <- NULL
  for (s in 1:5) {
    set.seed(s)
    b <- dea.boot(X, Y, NREP = 2000, RTS = "vrs", ORIENTATION = "in")
    ratio <- rowMeans(outer(b$eff, rep(1, ncol(b$boot)))^2 / b$boot, na.rm = TRUE)
    bo <- rbind(bo, data.frame(seed = s, dmu = d$dmu, theta = num(b$eff),
                               bc_benchmarking = num(b$eff.bc), bc_ratio = num(ratio),
                               h_benchmarking = num(h)))
  }
  write.csv(bo, "tests/data/sa_benchmarking_bootstrap.csv", row.names = FALSE, quote = FALSE)
}
cat("Benchmarking", as.character(packageVersion("Benchmarking")), "|", R.version.string, "\n")
