# Validation data set D: the Tobit and truncated regressions that
# tests/test_secondstage_reference.py pins, re-run with packages anyone can
# install.  D comes from the Korean Innovation Survey (STEPI) and is not
# redistributed; the script reads it from the folder named by PLEAP_PRIVATE_DATA.
# reference_sd.R does the same on the public synthetic data set SD.  The workbook values the tests store came from AER::tobit (run
# through SPSS's R extension) and from Stata's truncreg; this script shows that
# plain survival::survreg and truncreg reproduce them.
#
#   Rscript docs/external_reference/secondstage_ds_d.R
#
# AER::tobit(y ~ x1 + x2, left = 0, right = 1) is survreg on interval-coded
# censoring: status 2 (left-censored at 0), 0 (right-censored at 1), 1 (exact).
suppressPackageStartupMessages({library(survival); library(truncreg)})
d <- read.csv(file.path(Sys.getenv("PLEAP_PRIVATE_DATA", "tests/data_private"), "ds_d.csv"))
y <- d$vrs_i
tobit <- function(left, right) {
  status <- ifelse(y <= left, 2, ifelse(y >= right, 0, 1))
  time <- pmin(pmax(y, left), right)
  survreg(Surv(time, time, event = status, type = "interval") ~ x1 + x2, data = d,
          dist = "gaussian")
}
show <- function(label, fit) {
  s <- summary(fit)$table
  cat(sprintf("\n== %s ==\n", label))
  print(format(s, digits = 15))
  b <- coef(fit)[-1]; V <- vcov(fit)[2:3, 2:3]
  wald <- as.numeric(t(b) %*% solve(V) %*% b)
  cat(sprintf("loglik %.10f   scale %.10f   Wald %.10f (p %.10f)\n", fit$loglik[2],
              fit$scale, wald, pchisq(wald, 2, lower.tail = FALSE)))
}
show("Tobit [0, 1]", tobit(0, 1))
show("Tobit [0, Inf)", tobit(0, Inf))
# Left truncation at 0 only (Stata: truncreg vrs_i x1 x2, ll(0)).  truncreg
# handles one truncation point; Stata's ll(0) ul(1) has no counterpart here.
tr <- truncreg(vrs_i ~ x1 + x2, data = d, point = 0, direction = "left")
cat("\n== truncreg, left-truncated at 0 ==\n")
print(format(summary(tr)$coefficients, digits = 15))
cat(sprintf("loglik %.10f\n", as.numeric(logLik(tr))))
