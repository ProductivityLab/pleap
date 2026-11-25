# Reference values for the example of the paper's Section 3 (examples/sample_data.xlsx,
# as tests/data/example_panel.csv; tests/data/make_synthetic_data.py), from R.
# Run from the repository root:
#   Rscript docs/external_reference/reference_example.R
#
# Writes
#   tests/data/example_benchmarking_dea.csv  Benchmarking: CCR and BCC scores, input and
#                                            output orientation, last period
#   tests/data/example_bcc_input.csv         the last period's BCC input-oriented scores
#                                            (Benchmarking) with the two environmental
#                                            variables: the sample of Algorithm 1
#   tests/data/example_truncreg_R.csv        truncreg(theta ~ z1 + z2, point = 1,
#                                            direction = "right", method = "nr") on the
#                                            units strictly inside the frontier
suppressPackageStartupMessages({ library(Benchmarking); library(truncreg) })
num <- function(x) sprintf("%.17g", x)
p <- read.csv("tests/data/example_panel.csv")
d <- p[p$Period == max(p$Period), ]
X <- as.matrix(d[, c("x1", "x2")]); Y <- as.matrix(d[, c("y1", "y2")])

out <- NULL
for (rts in c("crs", "vrs")) for (ori in c("in", "out")) {
  e <- dea(X, Y, RTS = rts, ORIENTATION = ori)
  out <- rbind(out, data.frame(model = ifelse(rts == "crs", "CCR", "BCC"),
                               orient = ifelse(ori == "in", "input", "output"),
                               dmu = d$DMU_name, score = num(if (ori == "in") e$eff else 1 / e$eff)))
}
write.csv(out, "tests/data/example_benchmarking_dea.csv", row.names = FALSE, quote = FALSE)

theta <- dea(X, Y, RTS = "vrs", ORIENTATION = "in")$eff
write.csv(data.frame(DMU = d$DMU_name, theta = num(theta), z1 = num(d$z1), z2 = num(d$z2)),
          "tests/data/example_bcc_input.csv", row.names = FALSE, quote = FALSE)

inside <- data.frame(theta = theta, z1 = d$z1, z2 = d$z2)[theta < 1 - 1e-9, ]
f <- truncreg(theta ~ z1 + z2, data = inside, point = 1, direction = "right",
              method = "nr", iterlim = 200)
b <- coef(f); V <- vcov(f)
r <- data.frame(name = c("n", "const", "z1", "z2", "sigma", "loglik", "se_const", "se_z1", "se_z2",
                         "se_sigma", "se_logsigma", "gradient_max"),
                value = num(c(nrow(inside), b[1:3], b["sigma"], as.numeric(logLik(f)),
                              sqrt(diag(V))[1:3], sqrt(V["sigma", "sigma"]),
                              sqrt(V["sigma", "sigma"]) / b["sigma"], max(abs(f$gradient)))))
write.csv(r, "tests/data/example_truncreg_R.csv", row.names = FALSE, quote = FALSE)
cat("n inside", nrow(inside), "| Benchmarking", as.character(packageVersion("Benchmarking")),
    "| truncreg", as.character(packageVersion("truncreg")), "|", R.version.string, "\n")
