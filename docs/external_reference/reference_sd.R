# Reference values for the synthetic data set SD (tests/data/sd.csv;
# tests/data/make_synthetic_data.py), from R.  Run from the repository root:
#   Rscript docs/external_reference/reference_sd.R
#
# Writes
#   tests/data/sd_benchmarking_dea.csv   Benchmarking: CCR, BCC and NIRS ("drs") scores,
#                                        input and output orientation
#   tests/data/sd_r_secondstage.json     second-stage regressions and group tests on vrs_i,
#                                        the BCC input-oriented score rounded to eight
#                                        decimals as an exported column would be:
#     tobit_01, tobit_0inf   AER::tobit(vrs_i ~ x1 + x2, left = 0, right = 1 | Inf)
#     trunc_0                truncreg(vrs_i ~ x1 + x2, point = 0, direction = "left")
#     trunc_1                truncreg(..., point = 1, direction = "right") on vrs_i < 1
#                            (both with method = "nr": truncreg's default BFGS stops with a
#                            gradient of 3.5e-3 here, its Newton-Raphson at 2e-9)
#     ols                    lm(vrs_i ~ x1 + x2)
#     mwu                    wilcox.test(vrs_i ~ group1, exact = FALSE, correct = FALSE)
#     kw, dunn               kruskal.test and rstatix::dunn_test (Bonferroni) by group2
#     t_test, levene_2       t.test (Student and Welch) and car::leveneTest(center = mean) by group1
#     anova, welch, levene_4 oneway.test (both forms) and car::leveneTest by group2
#     tukey, games_howell    TukeyHSD by group2, and Games-Howell from ptukey with Welch's
#                            degrees of freedom (rstatix::games_howell_test computes the same
#                            but rounds its p-values to three decimals)
suppressPackageStartupMessages({
  library(Benchmarking); library(AER); library(truncreg); library(car)
  library(rstatix); library(jsonlite)
})
num <- function(x) sprintf("%.17g", x)
d <- read.csv("tests/data/sd.csv")

# ---- DEA scores -------------------------------------------------------------------
X <- as.matrix(d[, c("i1", "i2")]); Y <- as.matrix(d[, "o1", drop = FALSE])
out <- NULL
for (rts in c("crs", "vrs", "drs")) for (ori in c("in", "out")) {
  e <- dea(X, Y, RTS = rts, ORIENTATION = ori)
  score <- if (ori == "in") e$eff else 1 / e$eff
  out <- rbind(out, data.frame(model = c(crs = "CCR", vrs = "BCC", drs = "NIRS")[[rts]],
                               orient = ifelse(ori == "in", "input", "output"),
                               dmu = d$dmu, score = num(score)))
}
write.csv(out, "tests/data/sd_benchmarking_dea.csv", row.names = FALSE, quote = FALSE)

# ---- Second stage -----------------------------------------------------------------
res <- list()
coefs <- function(cf, se, p = NULL) {
  r <- list(coef = as.list(cf), se = as.list(se))
  if (!is.null(p)) r$p <- as.list(p)
  r
}
nm <- function(v) setNames(v, c("const", "x1", "x2", names(v)[-(1:3)]))

for (spec in c("tobit_01", "tobit_0inf")) {
  f <- tobit(vrs_i ~ x1 + x2, left = 0, right = if (spec == "tobit_01") 1 else Inf, data = d)
  s <- summary(f)$coefficients            # const, x1, x2, Log(scale)
  b <- coef(f)[2:3]; V <- vcov(f)[2:3, 2:3]
  wald <- as.numeric(t(b) %*% solve(V) %*% b)
  r <- coefs(nm(s[, 1]), nm(s[, 2]), nm(s[1:3, 4]))
  names(r$coef)[4] <- names(r$se)[4] <- "Log(scale)"
  r$loglik <- as.numeric(logLik(f)); r$scale <- f$scale
  r$wald <- wald; r$wald_p <- pchisq(wald, 2, lower.tail = FALSE)
  r$n <- nrow(d); r$censored_upper <- sum(d$vrs_i >= 1); r$censored_lower <- sum(d$vrs_i <= 0)
  res[[spec]] <- r
}

trunc_fit <- function(dd, point, direction) {
  f <- truncreg(vrs_i ~ x1 + x2, data = dd, point = point, direction = direction,
                method = "nr", iterlim = 200)
  V <- vcov(f); b <- coef(f)
  r <- coefs(nm(b[1:3]), nm(sqrt(diag(V))[1:3]))
  r$sigma <- unname(b["sigma"]); r$sigma_se <- sqrt(V["sigma", "sigma"])
  r$loglik <- as.numeric(logLik(f)); r$n <- nrow(dd)
  r$gradient_max <- max(abs(f$gradient))
  r
}
res$trunc_0 <- trunc_fit(d, 0, "left")
res$trunc_1 <- trunc_fit(d[d$vrs_i < 1, ], 1, "right")

f <- lm(vrs_i ~ x1 + x2, data = d); s <- summary(f)
res$ols <- coefs(nm(s$coefficients[, 1]), nm(s$coefficients[, 2]), nm(s$coefficients[, 4]))
res$ols$r2 <- s$r.squared; res$ols$adj_r2 <- s$adj.r.squared
res$ols$f <- unname(s$fstatistic[1])
res$ols$f_p <- pf(s$fstatistic[1], s$fstatistic[2], s$fstatistic[3], lower.tail = FALSE)[[1]]

g1 <- factor(d$group1); g2 <- factor(d$group2)
w <- wilcox.test(vrs_i ~ g1, data = d, exact = FALSE, correct = FALSE)
res$mwu <- list(W = unname(w$statistic), p = w$p.value, n1 = sum(d$group1 == 1), n2 = sum(d$group1 == 2))
k <- kruskal.test(vrs_i ~ g2, data = d)
res$kw <- list(H = unname(k$statistic), df = unname(k$parameter), p = k$p.value)
dn <- dunn_test(d, vrs_i ~ group2, p.adjust.method = "bonferroni")
res$dunn <- lapply(seq_len(nrow(dn)), function(i) list(
  a = dn$group1[i], b = dn$group2[i], z = dn$statistic[i], p = dn$p[i], p_adj = dn$p.adj[i]))

tt <- lapply(c(TRUE, FALSE), function(eq) t.test(vrs_i ~ g1, data = d, var.equal = eq))
res$t_test <- list(student = list(t = unname(tt[[1]]$statistic), df = unname(tt[[1]]$parameter), p = tt[[1]]$p.value),
                   welch = list(t = unname(tt[[2]]$statistic), df = unname(tt[[2]]$parameter), p = tt[[2]]$p.value))
lv <- leveneTest(vrs_i ~ g1, data = d, center = mean)
res$levene_2 <- list(F = lv[1, "F value"], p = lv[1, "Pr(>F)"])
lv <- leveneTest(vrs_i ~ g2, data = d, center = mean)
res$levene_4 <- list(F = lv[1, "F value"], p = lv[1, "Pr(>F)"])
a <- oneway.test(vrs_i ~ g2, data = d, var.equal = TRUE)
res$anova <- list(F = unname(a$statistic), df1 = unname(a$parameter[1]), df2 = unname(a$parameter[2]), p = a$p.value)
a <- oneway.test(vrs_i ~ g2, data = d, var.equal = FALSE)
res$welch <- list(F = unname(a$statistic), df1 = unname(a$parameter[1]), df2 = unname(a$parameter[2]), p = a$p.value)

tk <- TukeyHSD(aov(vrs_i ~ g2, data = d))$g2           # rows "b-a": mean_b - mean_a
res$tukey <- lapply(rownames(tk), function(rn) {
  ab <- strsplit(rn, "-")[[1]]
  list(a = ab[2], b = ab[1], diff_b_minus_a = tk[rn, "diff"], p_adj = tk[rn, "p adj"])
})
gh <- games_howell_test(d, vrs_i ~ group2)              # estimate: mean_group2 - mean_group1
lv4 <- sort(unique(d$group2))
res$games_howell <- lapply(seq_len(nrow(gh)), function(i) {
  a <- gh$group1[i]; b <- gh$group2[i]
  ya <- d$vrs_i[d$group2 == a]; yb <- d$vrs_i[d$group2 == b]
  va <- var(ya) / length(ya); vb <- var(yb) / length(yb)
  se <- sqrt(va + vb); df <- (va + vb)^2 / (va^2 / (length(ya) - 1) + vb^2 / (length(yb) - 1))
  diff <- mean(yb) - mean(ya)
  list(a = a, b = b, diff_b_minus_a = diff, se = se, df = df,
       p_adj = ptukey(abs(diff / se) * sqrt(2), length(lv4), df, lower.tail = FALSE),
       p_adj_rstatix = gh$p.adj[i], diff_rstatix = gh$estimate[i])
})

res$versions <- list(R = R.version.string,
                     packages = lapply(c(Benchmarking = "Benchmarking", AER = "AER", survival = "survival",
                                         truncreg = "truncreg", car = "car", rstatix = "rstatix"),
                                       function(p) as.character(packageVersion(p))))
writeLines(toJSON(res, auto_unbox = TRUE, digits = NA, pretty = TRUE), "tests/data/sd_r_secondstage.json")
cat("done |", R.version.string, "\n")
