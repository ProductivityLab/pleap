"""Second-stage analysis: Tobit / truncated / OLS regression and group tests.

v1.0.8 - Tobit and truncated regressions standardise the covariates internally
         and maximise the Gaussian likelihood with analytic gradients, so they
         converge and match R survreg(dist="gaussian") / SPSS even when the
         environmental variables span many orders of magnitude.  The truncated
         regression truncates only at the efficient frontier (Simar-Wilson).
         The two-group t-test follows Levene's test (SPSS convention): equal
         variances assumed when Levene p >= 0.05 (Student), otherwise Welch.
"""

# PLEAP - Productivity Lab Efficiency Analysis Program
# Copyright (C) 2025-2026 Changhee Kim, Productivity Lab,
#                         Incheon National University
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats
import statsmodels.api as sm
from scipy.optimize import minimize

LOG2PI = np.log(2 * np.pi)

# A score within this distance of a censoring or truncation point is treated
# as lying on it.  The linear programs return frontier units as 1 only up to
# rounding (1 - 4e-15 is typical), so a strict ``theta < 1`` lets some of them
# into a truncated sample at random.  The Tobit has always used this value for
# its censoring rule; the truncated models now use the same one.
BOUND_TOL = 1e-9


def stars(p):
    if not np.isfinite(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def _num_hessian(grad_f, x, eps=1e-5):
    # Hessian from CENTRAL differences of the (analytic) gradient, on the
    # standardised covariates.  Forward differences, used up to v1.1.0, carry an
    # O(eps) error: on the Tobit reference data that put the standard error of
    # one slope 2.4e-5 (relative) away from R's survreg and from a 60-digit
    # computation.  Central differences cut the error to O(eps^2).
    n = len(x); H = np.zeros((n, n))
    for i in range(n):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        H[i, :] = (grad_f(xp)[1] - grad_f(xm)[1]) / (2.0 * eps)
    return 0.5 * (H + H.T)


def _format_table(coefs, se, names):
    coefs = np.asarray(coefs, float); se = np.asarray(se, float)
    bad = ~np.isfinite(se) | (se <= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(bad, np.nan, coefs / np.where(bad, 1.0, se))
    # Upper-tail functions (sf) rather than 1 - cdf: the subtraction loses the
    # relative precision of a small p-value (it rounds to 1e-16 absolute).
    p = np.where(np.isfinite(z), 2 * stats.norm.sf(np.abs(z)), np.nan)
    se_disp = np.where(bad, np.nan, se)
    out = pd.DataFrame({
        "coef": np.round(coefs, 6), "std_err": np.round(se_disp, 6),
        "z": np.round(z, 4), "P>|z|": np.round(p, 6),
        "sig": [stars(pp) for pp in p],
    }, index=names)
    # Full-precision copies for the publication tables (pleap.viz).  These are
    # metadata only: the table contents printed above are unchanged.
    out.attrs["coef_raw"] = [float(v) for v in np.asarray(coefs, float)]
    out.attrs["se_raw"] = [float(v) for v in np.asarray(se_disp, float)]
    out.attrs["p_raw"] = [float(v) for v in np.asarray(p, float)]
    return out


def ols_regression(y, Z, names=None):
    Zc = sm.add_constant(np.asarray(Z, float))
    res = sm.OLS(np.asarray(y, float), Zc, missing="drop").fit()
    cols = ["const"] + (names if names else [f"x{i+1}" for i in range(Zc.shape[1]-1)])
    df = pd.DataFrame({
        "coef": np.round(res.params, 6), "std_err": np.round(res.bse, 6),
        "t": np.round(res.tvalues, 4), "P>|t|": np.round(res.pvalues, 6),
        "sig": [stars(pp) for pp in res.pvalues],
    }, index=cols)
    # Full-precision copies for the publication tables (metadata only).
    df.attrs["coef_raw"] = [float(v) for v in np.asarray(res.params, float)]
    df.attrs["se_raw"] = [float(v) for v in np.asarray(res.bse, float)]
    df.attrs["p_raw"] = [float(v) for v in np.asarray(res.pvalues, float)]
    return {"summary": df, "n": int(res.nobs), "r2": float(res.rsquared),
            "adj_r2": float(res.rsquared_adj), "fvalue": float(res.fvalue),
            "f_pvalue": float(res.f_pvalue)}


def _censored_negll_grad(par, y, Zc, lower, upper):
    # Gaussian Tobit (type-I) negative log-likelihood and gradient w.r.t.
    # (beta, log sigma).  Matches survreg(dist="gaussian") / AER::tobit.
    k = Zc.shape[1]; b = par[:k]; g = par[k]; s = np.exp(g); mu = Zc @ b
    upp = (upper is not None) & (y >= (upper if upper is not None else np.inf) - BOUND_TOL)
    low = (lower is not None) & (y <= (lower if lower is not None else -np.inf) + BOUND_TOL)
    mid = ~(upp | low)
    ll = 0.0; gb = np.zeros(k); gg = 0.0
    if mid.any():
        z = (y[mid] - mu[mid]) / s
        ll += np.sum(-g - 0.5 * LOG2PI - 0.5 * z * z)
        gb += Zc[mid].T @ (z / s); gg += np.sum(z * z - 1.0)
    if upper is not None and upp.any():
        a = (upper - mu[upp]) / s
        Phi = np.clip(stats.norm.cdf(-a), 1e-300, None)
        lam = stats.norm.pdf(a) / Phi
        ll += np.sum(np.log(Phi)); gb += Zc[upp].T @ (lam / s); gg += np.sum(a * lam)
    if lower is not None and low.any():
        bb = (lower - mu[low]) / s
        Phi = np.clip(stats.norm.cdf(bb), 1e-300, None)
        lamL = stats.norm.pdf(bb) / Phi
        ll += np.sum(np.log(Phi)); gb += Zc[low].T @ (-lamL / s); gg += np.sum(-bb * lamL)
    return -ll, -np.concatenate([gb, [gg]])


def _truncated_negll_grad(par, y, Zc, lower, upper):
    # Gaussian truncated regression negative log-likelihood and gradient.
    k = Zc.shape[1]; b = par[:k]; g = par[k]; s = np.exp(g); mu = Zc @ b
    z = (y - mu) / s
    a = (lower - mu) / s if lower is not None else np.full_like(mu, -np.inf)
    bU = (upper - mu) / s if upper is not None else np.full_like(mu, np.inf)
    phi_a = np.where(np.isfinite(a), stats.norm.pdf(a), 0.0)
    phi_b = np.where(np.isfinite(bU), stats.norm.pdf(bU), 0.0)
    a0 = np.where(np.isfinite(a), a, 0.0); b0 = np.where(np.isfinite(bU), bU, 0.0)
    # Probability of the truncation window, Phi(bU) - Phi(a).  When the
    # window lies in the upper tail (a > 0) the difference of two cdf values
    # near 1 cancels; the same difference of upper-tail probabilities does not.
    # The cancelling form made BFGS stop after two iterations on a model
    # truncated at 0 only (loglik 88.79 against Stata's 105.53).
    D = np.where(a > 0, stats.norm.sf(a) - stats.norm.sf(bU),
                 stats.norm.cdf(bU) - stats.norm.cdf(a))
    D = np.clip(D, 1e-300, None)
    ll = np.sum(-g - 0.5 * LOG2PI - 0.5 * z * z - np.log(D))
    dmu = z / s + (phi_b - phi_a) / (D * s)
    gb = Zc.T @ dmu
    gg = np.sum(-1.0 + z * z + (b0 * phi_b - a0 * phi_a) / D)
    return -ll, -np.concatenate([gb, [gg]])


def _fit_mle(negll_grad, y, Z, lower, upper, names):
    # Covariates are standardised internally so the optimiser is well-scaled
    # even when the environmental variables span many orders of magnitude;
    # coefficients and standard errors are transformed back to the original
    # scale at the end.
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    m_ = Z.mean(0); sd_ = Z.std(0, ddof=0); sd_[sd_ == 0] = 1.0
    Zs = (Z - m_) / sd_
    Zc = np.column_stack([np.ones(len(Zs)), Zs]); k = Zc.shape[1]
    b0 = np.linalg.lstsq(Zc, y, rcond=None)[0]
    s0 = max(np.std(y - Zc @ b0), 1e-3)
    par0 = np.concatenate([b0, [np.log(s0)]])
    f = lambda p: negll_grad(p, y, Zc, lower, upper)
    opt = minimize(f, par0, jac=True, method="BFGS",
                   options={"maxiter": 5000, "gtol": 1e-8})
    gnorm = float(np.linalg.norm(f(opt.x)[1]))
    if not (opt.success or gnorm < 1e-4):
        # BFGS depends on a line search, and a line search can fail ("precision
        # loss") far from the optimum.  Newton's method in a trust region, on
        # the Hessian of the analytic gradient, needs no line search.  It is
        # only tried when BFGS has not converged, so results that BFGS already
        # reaches are untouched; the better end point is kept.
        hess = lambda p: _num_hessian(f, p)
        for start in (opt.x, par0):
            try:
                alt = minimize(f, start, jac=True, hess=hess, method="trust-exact",
                               options={"maxiter": 1000, "gtol": 1e-8})
            except Exception:
                continue
            if np.isfinite(alt.fun) and alt.fun < opt.fun:
                opt = alt
        gnorm = float(np.linalg.norm(f(opt.x)[1]))
    # Polish with Newton steps on the analytic gradient.  BFGS stops once the
    # gradient is below gtol; where the likelihood is flat that can leave the
    # estimate some 1e-9 from the maximum, and where in that band it stops
    # depends on the platform's floating point: in CI the model truncated at 0
    # and 1 ended 2.2e-9 from the 60-digit maximum on Windows and macOS and
    # 9e-12 on Linux.  Newton's method converges quadratically from there, so
    # every platform reaches the maximum itself.  A step is kept only if it
    # shrinks the gradient without lowering the likelihood.
    if np.all(np.isfinite(opt.x)):
        x = np.array(opt.x, float)
        fx, gx = f(x)
        for _ in range(6):
            try:
                step = np.linalg.solve(_num_hessian(f, x), gx)
            except Exception:
                break
            xn = x - step
            fn, gn = f(xn)
            if (not np.isfinite(fn) or not np.all(np.isfinite(gn))
                    or np.linalg.norm(gn) >= np.linalg.norm(gx)
                    or fn > fx + 1e-12 * max(1.0, abs(fx))):
                break
            x, fx, gx = xn, fn, gn
            if np.linalg.norm(gx) < 1e-12:
                break
        opt.x, opt.fun = x, fx
        gnorm = float(np.linalg.norm(gx))
    converged = bool(opt.success or gnorm < 1e-4)
    a = opt.x[:k]; sigma = float(np.exp(opt.x[k]))
    try:
        cov_full = np.linalg.inv(_num_hessian(f, opt.x))
        cov_s = cov_full[:k, :k]
        var_logscale = float(cov_full[k, k])
    except Exception:
        cov_s = np.full((k, k), np.nan); var_logscale = np.nan
    # standardized -> original scale: beta_j = a_j / sd_j ,
    # const = a_0 - sum_j a_j m_j / sd_j ; cov transforms as T cov T'.
    # (sigma is in y-units, so log(scale) is unaffected by the Z standardisation.)
    T = np.zeros((k, k)); T[0, 0] = 1.0
    for j in range(1, k):
        T[0, j] = -m_[j - 1] / sd_[j - 1]
        T[j, j] = 1.0 / sd_[j - 1]
    coefs = T @ a
    cov = T @ cov_s @ T.T
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    cols = ["const"] + (names if names else [f"x{i+1}" for i in range(k - 1)])
    summary = _format_table(coefs, se, cols)
    # Log(scale) row, mirroring R survreg / AER::tobit output.
    g_hat = float(opt.x[k]); se_g = float(np.sqrt(var_logscale)) if np.isfinite(var_logscale) and var_logscale > 0 else np.nan
    ls = _format_table(np.array([g_hat]), np.array([se_g]), ["Log(scale)"])
    meta = {kk: list(summary.attrs.get(kk, [])) + list(ls.attrs.get(kk, []))
            for kk in ("coef_raw", "se_raw", "p_raw")}
    summary = pd.concat([summary, ls])
    summary.attrs.update(meta)      # concat drops attrs that differ
    # Wald test that all slope coefficients are jointly zero.
    wald = wald_df = wald_p = np.nan
    if k > 1:
        sl = coefs[1:]; cov_sl = cov[1:, 1:]
        try:
            wald = float(sl @ np.linalg.solve(cov_sl, sl))
            wald_df = int(k - 1)
            wald_p = float(stats.chi2.sf(wald, wald_df))
        except Exception:
            pass
    return {"summary": summary, "sigma": sigma, "n": int(len(y)),
            "loglik": float(-opt.fun), "converged": converged,
            "wald": wald, "wald_df": wald_df, "wald_p": wald_p}


def tobit_regression(y, Z, lower=None, upper=1.0, names=None):
    y = np.asarray(y, float); Z = np.asarray(Z, float)
    Zf = Z if Z.ndim == 2 else Z[:, None]
    mask = np.isfinite(y) & np.all(np.isfinite(Zf), axis=1)
    return _fit_mle(_censored_negll_grad, y[mask], Zf[mask], lower, upper, names)


def truncated_regression(y, Z, lower=None, upper=1.0, names=None):
    y = np.asarray(y, float); Z = np.asarray(Z, float)
    Zf = Z if Z.ndim == 2 else Z[:, None]
    mask = np.isfinite(y) & np.all(np.isfinite(Zf), axis=1)
    if upper is not None: mask &= (y < upper - BOUND_TOL)
    if lower is not None: mask &= (y > lower + BOUND_TOL)
    return _fit_mle(_truncated_negll_grad, y[mask], Zf[mask], lower, upper, names)


def simarwilson_algorithm1(theta, Z, L=2000, alpha=0.05, seed=None,
                           names=None, upper=1.0):
    """Simar and Wilson (2007), Algorithm 1: truncated regression with a
    parametric bootstrap for the second stage.

    Why this exists.  DEA efficiency scores are serially correlated: every
    score is measured against a frontier estimated from the same sample, so
    the standard errors an ordinary Tobit or truncated regression reports on
    them are not valid.  Algorithm 1 replaces those standard errors with
    bootstrap ones that account for the dependence, which is the usual minimum
    for a defensible second stage.

    Convention.  PLEAP reports Farrell efficiency ``theta`` in (0, 1], so the
    regression is truncated from ABOVE at ``upper`` (= 1), and only the
    inefficient units enter the fit.  Simar and Wilson write the model in
    Shephard units ``delta = 1/theta >= 1``, truncated from below at 1; the
    two are the same model and the coefficients differ only in sign
    convention.

    Steps, following the paper:

      1. Fit the truncated regression of ``theta`` on ``Z`` by maximum
         likelihood, using the m units strictly inside the frontier
         (``theta < upper - BOUND_TOL``, so frontier units that the linear
         program returns as 1 - 4e-15 stay out).  This gives
         ``beta_hat`` and ``sigma_hat``.
      2. Repeat L times: for each of those m units draw an error from
         ``N(0, sigma_hat^2)`` truncated to keep ``z_i beta_hat + eps`` below
         ``upper``, form the pseudo-response, and refit the truncated
         regression.
      3. Report the bootstrap standard deviation of the L coefficient vectors,
         and the basic (pivotal) bootstrap interval
         ``[2 beta_hat - q_(1-alpha/2), 2 beta_hat - q_(alpha/2)]``.

    What it does NOT do.  This is Algorithm 1, not Algorithm 2: the efficiency
    scores are used as estimated, without the bias correction that Algorithm 2
    applies first.  Algorithm 1 is the one that fixes the inference; Algorithm
    2 additionally corrects the dependent variable.

    Returns a dict with ``summary`` (coefficients, bootstrap SEs and interval
    bounds), ``sigma``, ``n`` (units in the fit), ``L``, and ``boot`` (the
    L x k matrix of bootstrap coefficients).
    """
    theta = np.asarray(theta, float)
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    ok = np.isfinite(theta) & np.all(np.isfinite(Z), axis=1)
    theta, Z = theta[ok], Z[ok]
    inef = theta < upper - BOUND_TOL      # frontier units are 1 only up to rounding
    if inef.sum() < Z.shape[1] + 2:
        raise ValueError(
            f"only {int(inef.sum())} units lie strictly inside the frontier; "
            "the truncated regression needs more than there are parameters")

    base = truncated_regression(theta, Z, lower=None, upper=upper, names=names)
    beta = np.asarray(base["summary"].attrs["coef_raw"], float)[:-1]  # drop Log(scale)
    sigma = float(base["sigma"])

    Zf = Z[inef]
    Zc = np.column_stack([np.ones(len(Zf)), Zf])
    mu = Zc @ beta
    rng = np.random.default_rng(seed)
    # Errors are drawn truncated so the pseudo-response stays inside the
    # frontier; otherwise step 3 would refit on points the model says cannot
    # occur.
    hi = (upper - mu) / sigma
    boot = np.full((L, len(beta)), np.nan)
    for b in range(L):
        eps = stats.truncnorm.rvs(-np.inf, hi, loc=0.0, scale=sigma,
                                  random_state=rng)
        y_star = mu + eps
        try:
            fit = truncated_regression(y_star, Zf, lower=None, upper=upper,
                                       names=names)
            boot[b] = np.asarray(fit["summary"].attrs["coef_raw"], float)[:-1]
        except Exception:
            continue

    good = np.all(np.isfinite(boot), axis=1)
    bs = boot[good]
    if len(bs) < 2:
        raise RuntimeError("the parametric bootstrap did not converge")
    se = bs.std(axis=0, ddof=1)
    lo_q = np.percentile(bs, 100 * (alpha / 2), axis=0)
    hi_q = np.percentile(bs, 100 * (1 - alpha / 2), axis=0)
    ci_low, ci_high = 2 * beta - hi_q, 2 * beta - lo_q      # basic bootstrap

    with np.errstate(divide="ignore", invalid="ignore"):
        zstat = np.where(se > 0, beta / se, np.nan)
    p = 2 * stats.norm.sf(np.abs(zstat))
    cols = ["const"] + (list(names) if names else
                        [f"x{i+1}" for i in range(len(beta) - 1)])
    summary = pd.DataFrame({
        "Coef.": [f"{v:.4f}" for v in beta],
        "Boot.SE": [f"{v:.4f}" for v in se],
        "z": [f"{v:.3f}" if np.isfinite(v) else "" for v in zstat],
        "p": [f"{v:.4f}" if np.isfinite(v) else "" for v in p],
        f"CI{int((1-alpha)*100)}%_low": [f"{v:.4f}" for v in ci_low],
        f"CI{int((1-alpha)*100)}%_high": [f"{v:.4f}" for v in ci_high],
    }, index=cols)
    summary.attrs.update(coef_raw=list(beta), se_raw=list(se), p_raw=list(p),
                         ci_low=list(ci_low), ci_high=list(ci_high))
    return {"summary": summary, "sigma": sigma, "n": int(inef.sum()),
            "L": int(good.sum()), "boot": bs, "alpha": alpha,
            "method": "Simar-Wilson (2007) Algorithm 1"}


def _mwu_pair(ya, yb):
    n1, n2 = len(ya), len(yb)
    all_y = np.concatenate([ya, yb])
    ranks = stats.rankdata(all_y)
    R1 = ranks[:n1].sum()
    U1 = R1 - n1*(n1+1)/2.0
    U2 = n1*n2 - U1
    U  = min(U1, U2)
    _, counts = np.unique(all_y, return_counts=True)
    N = n1+n2
    tie_term = (counts**3 - counts).sum()
    mu = n1*n2/2.0
    sigma2 = (n1*n2/12.0) * ((N+1) - tie_term/(N*(N-1)))
    sigma = np.sqrt(max(sigma2, 1e-12))
    z = (U1 - mu) / sigma
    p = 2 * stats.norm.sf(abs(z))
    return U, U1, U2, z, p


def _dunn_pair(y, g, a, b, N, tie_term, mean_rank):
    na = int(np.sum(g==a)); nb = int(np.sum(g==b))
    diff = mean_rank[a] - mean_rank[b]
    se = np.sqrt(((N*(N+1) - tie_term/(N-1))/12.0) * (1.0/na + 1.0/nb))
    z = diff / se if se > 0 else np.nan
    p_raw = 2 * stats.norm.sf(abs(z))
    return z, p_raw


def _snap_ties(y, tol=BOUND_TOL):
    """Treat values that agree to within ``tol`` (relative) as one value.

    The linear programs return a frontier unit's score as 1 only up to
    rounding (1 - 4e-15, 1.0 and 1 + 2e-16 all occur), and units on the same
    facet can differ in the last bits.  Ranked as they stand, those scores
    are distinct: the rank tests lose their tie correction and the mid-ranks
    move.  SPSS, R or Stata, given the same scores exported to a file, see
    ties.  On the 60-unit data set A, the Kruskal-Wallis statistic was 6.381
    instead of SPSS's 6.356 for this reason alone.

    Each run of sorted values within ``tol`` of the run's first value is set
    to that first value, so a run never stretches further than ``tol``.
    """
    y = np.asarray(y, float)
    if y.size < 2:
        return y.copy()
    order = np.argsort(y, kind="mergesort")
    ys = y[order].copy()
    anchor = ys[0]
    for i in range(1, ys.size):
        if ys[i] - anchor <= tol * max(1.0, abs(anchor)):
            ys[i] = anchor
        else:
            anchor = ys[i]
    out = np.empty_like(ys)
    out[order] = ys
    return out


def group_tests_nonparam(y, groups):
    y = np.asarray(y, float); g = np.asarray(groups)
    mask = np.isfinite(y) & pd.notna(g)
    y, g = _snap_ties(y[mask]), g[mask]
    levels = list(pd.unique(g))
    samples = [y[g==lv] for lv in levels]
    out = {"groups": levels, "n": [len(x) for x in samples],
           "median": [float(np.median(x)) for x in samples]}
    if len(levels) < 2:
        out.update({"test":"n/a","stat":np.nan,"p":np.nan,"sig":"","pairwise":pd.DataFrame()})
        return out
    if len(levels) == 2:
        U, U1, U2, z, p = _mwu_pair(samples[0], samples[1])
        out["test"] = "Mann-Whitney U"
        out["stat"] = float(U); out["p"] = float(p); out["sig"] = stars(p)
        ma, mb = float(np.median(samples[0])), float(np.median(samples[1]))
        dirn = "n.s." if p>=0.10 else (
               f"{levels[0]} > {levels[1]}" if ma>mb else (
               f"{levels[0]} < {levels[1]}" if ma<mb else f"{levels[0]} = {levels[1]}"))
        out["pairwise"] = pd.DataFrame([{
            "group_A": levels[0], "group_B": levels[1],
            "median_A": ma, "median_B": mb,
            "U": U, "U1": U1, "U2": U2, "z": z, "p": p,
            "sig": stars(p), "direction": dirn}])
        return out
    H, p_kw = stats.kruskal(*samples)
    out["test"]="Kruskal-Wallis"; out["stat"]=float(H); out["p"]=float(p_kw); out["sig"]=stars(p_kw)
    ranks = stats.rankdata(y)
    mean_rank = {lv: ranks[g==lv].mean() for lv in levels}
    _, counts = np.unique(y, return_counts=True)
    N = len(y); tie_term = float(((counts**3 - counts)).sum())
    npairs = len(levels)*(len(levels)-1)//2
    rows = []
    for a, b in combinations(levels, 2):
        z, p_raw = _dunn_pair(y, g, a, b, N, tie_term, mean_rank)
        p_adj = min(p_raw * npairs, 1.0)
        ma = float(np.median(y[g==a])); mb = float(np.median(y[g==b]))
        dirn = "n.s." if p_adj>=0.10 else (
               f"{a} > {b}" if ma>mb else (f"{a} < {b}" if ma<mb else f"{a} = {b}"))
        rows.append({"group_A":a, "group_B":b, "median_A":ma, "median_B":mb,
                     "z (Dunn)": z, "p_raw": p_raw, "p_bonf": p_adj,
                     "sig": stars(p_adj), "direction": dirn})
    out["pairwise"] = pd.DataFrame(rows)
    return out


def _games_howell(samples, levels):
    rows = []
    for (i, a), (j, b) in combinations(enumerate(levels), 2):
        xi, xj = samples[i], samples[j]
        ni, nj = len(xi), len(xj)
        mi, mj = np.mean(xi), np.mean(xj)
        vi, vj = np.var(xi, ddof=1), np.var(xj, ddof=1)
        se = np.sqrt(vi/ni + vj/nj)
        if se <= 0: continue
        t = (mi - mj) / se
        num = (vi/ni + vj/nj)**2
        den = (vi/ni)**2/(ni-1) + (vj/nj)**2/(nj-1)
        df = num/den if den > 0 else np.nan
        q = np.sqrt(2) * abs(t); k = len(levels)
        try: p = float(stats.studentized_range.sf(q, k, df))
        except Exception: p = 2 * stats.t.sf(abs(t), df)
        dirn = "n.s." if p>=0.10 else (
               f"{a} > {b}" if mi>mj else (f"{a} < {b}" if mi<mj else f"{a} = {b}"))
        rows.append({"group_A":a, "group_B":b, "mean_A":mi, "mean_B":mj,
                     "t":t, "df":df, "p (GH)":p, "sig":stars(p), "direction":dirn})
    return pd.DataFrame(rows)


def _tukey_hsd(samples, levels):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    y = np.concatenate(samples)
    g = np.concatenate([[lv]*len(s) for lv,s in zip(levels, samples)])
    res = pairwise_tukeyhsd(y, g, alpha=0.05)
    # The numbers come from the result arrays, not from res.summary(): the
    # summary table rounds to four decimals, which printed p = 3.9e-6 as
    # 0.0000 and p = 0.08914 as 0.0891 (SPSS: 3.8988e-6 and 0.089137).
    # The arrays follow the pairs (i, j), i < j, of res.groupsunique; the
    # table's labels are kept and checked against that order.
    table = res.summary().data
    labels = [(r[0], r[1]) for r in table[1:]]
    gu = list(res.groupsunique)
    pairs = list(combinations(range(len(gu)), 2))
    assert [(str(gu[i]), str(gu[j])) for i, j in pairs] == \
           [(str(a), str(b)) for a, b in labels], "Tukey pair order changed"
    out = []
    for (a, b), md, p in zip(labels, np.asarray(res.meandiffs, float),
                             np.asarray(res.pvalues, float)):
        p = float(p)
        mi = float(np.mean([x for x,gg in zip(y,g) if gg==a]))
        mj = float(np.mean([x for x,gg in zip(y,g) if gg==b]))
        dirn = "n.s." if p>=0.10 else (
               f"{a} > {b}" if mi>mj else (f"{a} < {b}" if mi<mj else f"{a} = {b}"))
        out.append({"group_A":a,"group_B":b,"mean_A":mi,"mean_B":mj,
                    "meandiff":float(md),"p (Tukey)":p,
                    "sig":stars(p),"direction":dirn})
    return pd.DataFrame(out)


def group_tests_param(y, groups):
    y = np.asarray(y, float); g = np.asarray(groups)
    mask = np.isfinite(y) & pd.notna(g)
    y, g = y[mask], g[mask]
    levels = list(pd.unique(g))
    samples = [y[g==lv] for lv in levels]
    out = {"groups": levels, "n": [len(x) for x in samples],
           "mean": [float(np.mean(x)) for x in samples]}
    if len(levels) < 2:
        out.update({"test":"n/a","stat":np.nan,"p":np.nan,"sig":"","pairwise":pd.DataFrame()})
        return out
    if len(levels) == 2:
        # SPSS convention: Levene's test (centred on the mean) decides whether
        # to report the equal-variance (Student) or unequal-variance (Welch) t.
        lev_stat, lev_p = stats.levene(samples[0], samples[1], center="mean")
        equal_var = lev_p >= 0.05
        t_eq, p_eq = stats.ttest_ind(samples[0], samples[1], equal_var=True)
        t_we, p_we = stats.ttest_ind(samples[0], samples[1], equal_var=False)
        n1, n2 = len(samples[0]), len(samples[1])
        df_eq = n1 + n2 - 2
        v1, v2 = np.var(samples[0], ddof=1), np.var(samples[1], ddof=1)
        df_we = ((v1/n1 + v2/n2)**2 /
                 ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))) if (n1>1 and n2>1) else np.nan
        if equal_var:
            s, p, used = t_eq, p_eq, "Student t-test (equal variances assumed)"
        else:
            s, p, used = t_we, p_we, "Welch t-test (equal variances not assumed)"
        out["test"] = used
        out["stat"] = float(s); out["p"] = float(p); out["sig"] = stars(p)
        out["levene"] = {"stat": float(lev_stat), "p": float(lev_p)}
        ma, mb = float(np.mean(samples[0])), float(np.mean(samples[1]))
        dirn = "n.s." if p>=0.10 else (
               f"{levels[0]} > {levels[1]}" if ma>mb else (
               f"{levels[0]} < {levels[1]}" if ma<mb else f"{levels[0]} = {levels[1]}"))
        out["pairwise"] = pd.DataFrame([
            {"variances":"equal (Student)", "t":float(t_eq),"df":float(df_eq),
             "p":float(p_eq),"sig":stars(p_eq)},
            {"variances":"unequal (Welch)", "t":float(t_we),"df":float(df_we),
             "p":float(p_we),"sig":stars(p_we)},
            {"variances":"-> reported","t":float(s),"df":float(df_eq if equal_var else df_we),
             "p":float(p),"sig":stars(p)},
        ])
        out["direction"] = dirn
        return out
    lev_stat, lev_p = stats.levene(*samples, center="mean")
    heterosced = lev_p < 0.05
    if heterosced:
        ks=[len(s) for s in samples]; means=[np.mean(s) for s in samples]
        vars_=[np.var(s, ddof=1) for s in samples]
        w=[ki/vi if vi>0 else 0 for ki,vi in zip(ks, vars_)]
        gm = sum(wi*mi for wi,mi in zip(w, means)) / sum(w)
        kk = len(samples)
        num = sum(wi*(mi-gm)**2 for wi,mi in zip(w, means)) / (kk-1)
        den_term = sum((1 - wi/sum(w))**2 / (ki-1) for wi, ki in zip(w, ks))
        denom = 1 + (2*(kk-2)/(kk*kk - 1)) * den_term
        F = num/denom
        df1 = kk - 1; df2 = (kk*kk - 1) / (3*den_term)
        p = float(stats.f.sf(F, df1, df2))
        out["test"] = f"Welch ANOVA (Levene p={lev_p:.4f}, heteroscedastic)"
        out["stat"] = float(F); out["p"] = float(p); out["sig"] = stars(p)
        out["levene"] = {"stat": float(lev_stat), "p": float(lev_p)}
        out["pairwise"] = _games_howell(samples, levels)
    else:
        F, p = stats.f_oneway(*samples)
        out["test"] = f"One-way ANOVA (Levene p={lev_p:.4f}, equal variances assumed)"
        out["stat"] = float(F); out["p"] = float(p); out["sig"] = stars(p)
        out["levene"] = {"stat": float(lev_stat), "p": float(lev_p)}
        try: out["pairwise"] = _tukey_hsd(samples, levels)
        except Exception: out["pairwise"] = _games_howell(samples, levels)
    return out
