# Reference scores for the numerical-fault case of the paper's Section 4.3
# (tests/data/scaling_case.csv: 30 synthetic units, input x2 spanning eleven
# decades), from R's Benchmarking at its default settings.
#
#   Rscript docs/external_reference/reference_scaling.R
#
# writes tests/data/scaling_case_benchmarking.csv: the Farrell score of every
# unit for CCR and BCC under both orientations (1/phi for output orientation).
# The exact optimum is in tests/data/scaling_case_exact.csv
# (docs/external_reference/exact_dea.py); the comparison of the two is in
# docs/validation.md, section 4.

suppressMessages(library(Benchmarking))
num <- function(x) sprintf("%.17g", x)

d <- read.csv("tests/data/scaling_case.csv")
X <- as.matrix(d[, c("x1", "x2")])
Y <- as.matrix(d[, "y", drop = FALSE])

rows <- list()
for (model in c("CCR", "BCC")) {
  for (orient in c("input", "output")) {
    e <- dea(X, Y, RTS = if (model == "CCR") "crs" else "vrs",
             ORIENTATION = if (orient == "input") "in" else "out")$eff
    if (orient == "output") e <- 1 / e
    rows[[length(rows) + 1]] <- data.frame(model = model, orient = orient,
                                           dmu = d$dmu, efficiency = num(e))
  }
}
out <- do.call(rbind, rows)
write.csv(out, "tests/data/scaling_case_benchmarking.csv",
          row.names = FALSE, quote = FALSE)

cat(R.version.string, "\n")
cat("Benchmarking", as.character(packageVersion("Benchmarking")),
    "| lpSolveAPI", as.character(packageVersion("lpSolveAPI")), "\n")
