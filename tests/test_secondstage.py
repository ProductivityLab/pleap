"""Second-stage regression and group tests.

The estimators here were developed against R's ``survreg(dist="gaussian")`` and
SPSS.  Neither is available in continuous integration, so correctness is pinned
by cross-checks that need nothing but numpy and scipy:

* the Tobit log-likelihood and its analytic gradient are checked against each
  other by finite differences;
* the Tobit MLE is checked against the same likelihood maximised by a
  derivative-free optimiser - an independent code path, so agreement is
  evidence about the estimator and not about one implementation;
* with no censored observations the Tobit must collapse onto OLS, which has a
  closed form;
* on simulated data with known coefficients the estimators must recover them;
* the non-parametric and parametric group tests are checked against scipy's
  own implementations, and the Levene-driven choice between Student and Welch
  is pinned to the SPSS convention.
"""

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import minimize

from pleap.core.secondstage import (
    _censored_negll_grad, _truncated_negll_grad, group_tests_nonparam,
    group_tests_param, ols_regression, stars, tobit_regression,
    truncated_regression,
)


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def coefs(res):
    """Full-precision regression coefficients, without the Log(scale) row."""
    raw = np.asarray(res["summary"].attrs["coef_raw"], float)
    keep = [i for i, name in enumerate(res["summary"].index)
            if str(name) != "Log(scale)"]
    return raw[keep]


def ses(res):
    raw = np.asarray(res["summary"].attrs["se_raw"], float)
    keep = [i for i, name in enumerate(res["summary"].index)
            if str(name) != "Log(scale)"]
    return raw[keep]


def reported_row(res):
    """The t-test row PLEAP marks as the one it reports."""
    tbl = res["pairwise"]
    mask = tbl["variances"].astype(str).str.contains("reported")
    assert mask.sum() == 1, tbl
    return tbl[mask].iloc[0]


@pytest.fixture
def censored_sample():
    """y = 0.4 + 0.3 z1 - 0.2 z2 + e, censored from above at 1."""
    rng = np.random.default_rng(11)
    n = 400
    z1 = rng.normal(0, 1, n)
    z2 = rng.normal(0, 1, n)
    y = 0.4 + 0.3 * z1 - 0.2 * z2 + rng.normal(0, 0.25, n)
    return np.minimum(y, 1.0), np.column_stack([z1, z2]), np.array([0.4, 0.3, -0.2])


# --------------------------------------------------------------------------- #
#  stars                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p,expected", [
    (0.0001, "***"), (0.009, "***"), (0.011, "**"), (0.04, "**"),
    (0.051, "*"), (0.08, "*"), (0.11, ""), (0.5, ""), (np.nan, ""),
])
def test_significance_stars_use_the_one_five_ten_convention(p, expected):
    assert stars(p) == expected


# --------------------------------------------------------------------------- #
#  Likelihood and gradient                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("negll", [_censored_negll_grad, _truncated_negll_grad])
def test_analytic_gradient_matches_finite_differences(negll, censored_sample):
    y, Z, _ = censored_sample
    if negll is _truncated_negll_grad:
        keep = y < 1.0 - 1e-9
        y, Z = y[keep], Z[keep]
    Zc = np.column_stack([np.ones(len(Z)), Z])
    par = np.array([0.35, 0.28, -0.18, np.log(0.26)])
    _, g = negll(par, y, Zc, None, 1.0)
    eps = 1e-6
    num = np.empty_like(g)
    for j in range(len(par)):
        hi = par.copy(); hi[j] += eps
        lo = par.copy(); lo[j] -= eps
        num[j] = (negll(hi, y, Zc, None, 1.0)[0]
                  - negll(lo, y, Zc, None, 1.0)[0]) / (2 * eps)
    np.testing.assert_allclose(g, num, rtol=2e-5, atol=2e-5)


def test_tobit_matches_an_independent_derivative_free_fit(censored_sample):
    """Same likelihood, different optimiser and no standardisation."""
    y, Z, _ = censored_sample
    Zc = np.column_stack([np.ones(len(Z)), Z])

    def f(p):
        return _censored_negll_grad(p, y, Zc, None, 1.0)[0]

    start = np.append(np.linalg.lstsq(Zc, y, rcond=None)[0], np.log(0.3))
    ref = minimize(f, start, method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 60000,
                            "maxfev": 60000})
    assert ref.success
    got = tobit_regression(y, Z, lower=None, upper=1.0, names=["z1", "z2"])
    assert got["converged"]
    np.testing.assert_allclose(coefs(got), ref.x[:3], rtol=1e-4, atol=1e-5)
    assert got["sigma"] == pytest.approx(np.exp(ref.x[3]), rel=1e-4)
    assert got["loglik"] == pytest.approx(-ref.fun, rel=1e-8)


def test_tobit_recovers_known_coefficients(censored_sample):
    y, Z, truth = censored_sample
    got = tobit_regression(y, Z, lower=None, upper=1.0, names=["z1", "z2"])
    np.testing.assert_allclose(coefs(got), truth, atol=0.06)


def test_tobit_without_censoring_equals_ols():
    """The analytic identity that anchors the whole estimator."""
    rng = np.random.default_rng(5)
    n = 300
    Z = rng.normal(0, 1, (n, 2))
    y = 0.2 + 0.5 * Z[:, 0] - 0.3 * Z[:, 1] + rng.normal(0, 0.2, n)
    tb = tobit_regression(y, Z, lower=None, upper=10.0, names=["a", "b"])
    ol = ols_regression(y, Z, names=["a", "b"])
    np.testing.assert_allclose(coefs(tb), coefs(ol), rtol=1e-5, atol=1e-7)
    # the MLE scale uses 1/n, OLS uses 1/(n-k): rescale before comparing
    resid_ml = tb["sigma"]
    assert resid_ml == pytest.approx(
        np.std(y - np.column_stack([np.ones(n), Z]) @ coefs(ol), ddof=0),
        rel=1e-4)


def test_tobit_is_invariant_to_the_scale_of_a_covariate():
    """Internal standardisation must cancel exactly on the way out.

    The MLE failed to converge in v1.0.7 when a covariate ran into the
    millions; the back-transform is what fixed it, and this is its test.
    """
    rng = np.random.default_rng(8)
    n = 250
    z = rng.uniform(1.0, 5.0, (n, 1))
    y = np.minimum(0.3 + 0.08 * z[:, 0] + rng.normal(0, 0.15, n), 1.0)
    base = tobit_regression(y, z, lower=None, upper=1.0, names=["z"])
    big = tobit_regression(y, z * 1e6, lower=None, upper=1.0, names=["z"])
    assert base["converged"] and big["converged"]
    assert coefs(big)[0] == pytest.approx(coefs(base)[0], rel=1e-4)
    assert coefs(big)[1] == pytest.approx(coefs(base)[1] / 1e6, rel=1e-4)
    assert ses(big)[1] == pytest.approx(ses(base)[1] / 1e6, rel=1e-3)
    assert big["loglik"] == pytest.approx(base["loglik"], rel=1e-6)


def test_tobit_reports_the_scale_row_and_the_wald_statistic(censored_sample):
    y, Z, _ = censored_sample
    r = tobit_regression(y, Z, lower=None, upper=1.0, names=["z1", "z2"])
    idx = [str(i) for i in r["summary"].index]
    assert "Log(scale)" in idx
    row = float(r["summary"].loc["Log(scale)", "coef"])
    assert row == pytest.approx(np.log(r["sigma"]), abs=1e-5)
    assert r["wald_df"] == 2
    assert 0.0 <= r["wald_p"] <= 1.0
    assert r["n"] == len(y)


def test_truncated_regression_recovers_known_coefficients():
    """Simar-Wilson truncation: the sample is observed only below 1."""
    rng = np.random.default_rng(13)
    n = 700
    z = rng.normal(0, 1, (n, 1))
    y = 0.55 + 0.12 * z[:, 0] + rng.normal(0, 0.2, n)
    keep = y < 1.0
    r = truncated_regression(y[keep], z[keep], lower=None, upper=1.0, names=["z"])
    assert r["converged"]
    np.testing.assert_allclose(coefs(r), [0.55, 0.12], atol=0.06)


def test_standard_errors_are_reported_or_flagged_not_fabricated(censored_sample):
    """A singular Hessian must give nan, never a huge finite number.

    v1.0.7 printed z = 1e134 when the doubly-truncated likelihood diverged.
    """
    y, Z, _ = censored_sample
    r = tobit_regression(y, Z, lower=None, upper=1.0, names=["z1", "z2"])
    se = ses(r)
    assert np.all(np.isnan(se) | ((se > 0) & (se < 1e6)))


# --------------------------------------------------------------------------- #
#  OLS                                                                        #
# --------------------------------------------------------------------------- #
def test_ols_matches_the_normal_equations():
    rng = np.random.default_rng(21)
    n = 120
    Z = rng.normal(0, 1, (n, 3))
    y = 1.0 + Z @ np.array([0.5, -0.25, 0.75]) + rng.normal(0, 0.3, n)
    r = ols_regression(y, Z, names=["a", "b", "c"])
    Zc = np.column_stack([np.ones(n), Z])
    beta = np.linalg.solve(Zc.T @ Zc, Zc.T @ y)
    np.testing.assert_allclose(coefs(r), beta, rtol=1e-10)
    resid = y - Zc @ beta
    s2 = resid @ resid / (n - 4)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(Zc.T @ Zc)))
    np.testing.assert_allclose(ses(r), se, rtol=1e-8)
    assert r["r2"] == pytest.approx(1 - resid @ resid /
                                    ((y - y.mean()) @ (y - y.mean())))
    assert r["n"] == n


# --------------------------------------------------------------------------- #
#  Group tests - parametric                                                   #
# --------------------------------------------------------------------------- #
@pytest.fixture
def equal_variance_groups():
    rng = np.random.default_rng(31)
    a = rng.normal(0.50, 0.10, 70)
    b = rng.normal(0.54, 0.10, 65)
    y = np.concatenate([a, b])
    g = np.array(["A"] * 70 + ["B"] * 65)
    return y, g, a, b


@pytest.fixture
def unequal_variance_groups():
    rng = np.random.default_rng(32)
    a = rng.normal(0.50, 0.05, 60)
    b = rng.normal(0.54, 0.30, 60)
    y = np.concatenate([a, b])
    g = np.array(["A"] * 60 + ["B"] * 60)
    return y, g, a, b


def test_levene_uses_the_mean_as_centre(equal_variance_groups):
    """SPSS centres Levene's test on the mean, scipy defaults to the median."""
    y, g, a, b = equal_variance_groups
    r = group_tests_param(y, g)
    expected = stats.levene(a, b, center="mean")
    assert r["levene"]["stat"] == pytest.approx(expected.statistic, rel=1e-9)
    assert r["levene"]["p"] == pytest.approx(expected.pvalue, rel=1e-9)


def test_equal_variances_report_the_student_test(equal_variance_groups):
    """The v1.0.7 bug: Welch was reported whatever Levene said."""
    y, g, a, b = equal_variance_groups
    r = group_tests_param(y, g)
    assert r["levene"]["p"] >= 0.05
    student = stats.ttest_ind(a, b, equal_var=True)
    row = reported_row(r)
    assert float(row["t"]) == pytest.approx(student.statistic, rel=1e-9)
    assert float(row["df"]) == pytest.approx(len(a) + len(b) - 2, abs=1e-9)
    assert float(row["p"]) == pytest.approx(student.pvalue, rel=1e-9)
    # the headline statistic follows the same decision
    assert r["stat"] == pytest.approx(student.statistic, rel=1e-9)
    assert "Student" in r["test"] or "equal" in r["test"]


def test_unequal_variances_report_welch(unequal_variance_groups):
    y, g, a, b = unequal_variance_groups
    r = group_tests_param(y, g)
    assert r["levene"]["p"] < 0.05
    welch = stats.ttest_ind(a, b, equal_var=False)
    row = reported_row(r)
    assert float(row["t"]) == pytest.approx(welch.statistic, rel=1e-9)
    assert float(row["p"]) == pytest.approx(welch.pvalue, rel=1e-9)
    assert r["stat"] == pytest.approx(welch.statistic, rel=1e-9)


def test_both_t_tests_are_always_shown(equal_variance_groups):
    """The reader must be able to see the test that was not reported."""
    y, g, a, b = equal_variance_groups
    tbl = group_tests_param(y, g)["pairwise"]
    kinds = " ".join(tbl["variances"].astype(str))
    assert "Student" in kinds and "Welch" in kinds
    student = stats.ttest_ind(a, b, equal_var=True)
    welch = stats.ttest_ind(a, b, equal_var=False)
    byname = {str(k): v for k, v in zip(tbl["variances"], tbl["t"])}
    assert float(byname["equal (Student)"]) == pytest.approx(student.statistic,
                                                             rel=1e-9)
    assert float(byname["unequal (Welch)"]) == pytest.approx(welch.statistic,
                                                             rel=1e-9)


def test_anova_matches_scipy_for_four_groups():
    rng = np.random.default_rng(33)
    samples = [rng.normal(m, 0.1, 40) for m in (0.45, 0.50, 0.55, 0.60)]
    y = np.concatenate(samples)
    g = np.repeat(["A", "B", "C", "D"], 40)
    r = group_tests_param(y, g)
    f, p = stats.f_oneway(*samples)
    assert r["levene"]["p"] >= 0.05, "fixture should have equal variances"
    assert r["stat"] == pytest.approx(f, rel=1e-9)
    assert r["p"] == pytest.approx(p, rel=1e-9)
    assert "ANOVA" in r["test"]
    assert len(r["pairwise"]) == 6          # all pairs of four groups


def test_welch_anova_is_used_when_levene_rejects():
    rng = np.random.default_rng(36)
    samples = [rng.normal(0.5, s, 40) for s in (0.03, 0.30, 0.03)]
    y = np.concatenate(samples)
    g = np.repeat(["A", "B", "C"], 40)
    r = group_tests_param(y, g)
    assert r["levene"]["p"] < 0.05
    assert "Welch" in r["test"]
    assert r["stat"] != pytest.approx(stats.f_oneway(*samples).statistic,
                                      rel=1e-6)


def test_group_means_and_sizes_are_reported(equal_variance_groups):
    y, g, a, b = equal_variance_groups
    r = group_tests_param(y, g)
    assert list(r["n"]) == [len(a), len(b)]
    np.testing.assert_allclose(r["mean"], [a.mean(), b.mean()], rtol=1e-12)


# --------------------------------------------------------------------------- #
#  Group tests - non-parametric                                               #
# --------------------------------------------------------------------------- #
def test_mann_whitney_follows_the_spss_asymptotic_convention(equal_variance_groups):
    """Tie-corrected normal approximation, no continuity correction.

    This is what SPSS prints as "Asymp. Sig. (2-tailed)", and it is the number
    a Korean reviewer will have in front of them.  scipy's default differs: it
    applies a continuity correction, which moves the p-value by about 1% here
    (0.0010599 instead of 0.0010516).  The convention is deliberate, so it is
    pinned rather than left to the default.
    """
    y, g, a, b = equal_variance_groups
    r = group_tests_nonparam(y, g)
    spss = stats.mannwhitneyu(a, b, alternative="two-sided",
                              method="asymptotic", use_continuity=False)
    row = r["pairwise"].iloc[0]
    assert float(row["U1"]) == pytest.approx(spss.statistic, rel=1e-12)
    assert float(row["p"]) == pytest.approx(spss.pvalue, rel=1e-12)
    assert r["p"] == pytest.approx(spss.pvalue, rel=1e-12)
    # U1 + U2 = n_a * n_b, and the reported U is the smaller of the two
    assert float(row["U1"]) + float(row["U2"]) == pytest.approx(
        len(a) * len(b), rel=1e-12)
    assert float(row["U"]) == min(float(row["U1"]), float(row["U2"]))


def test_kruskal_matches_scipy():
    rng = np.random.default_rng(34)
    samples = [rng.normal(m, 0.1, 35) for m in (0.4, 0.5, 0.6)]
    y = np.concatenate(samples)
    g = np.repeat(["A", "B", "C"], 35)
    r = group_tests_nonparam(y, g)
    h, p = stats.kruskal(*samples)
    assert r["stat"] == pytest.approx(h, rel=1e-9)
    assert r["p"] == pytest.approx(p, rel=1e-9)
    assert "Kruskal" in r["test"]


def test_dunn_p_values_are_bonferroni_adjusted_and_bounded():
    rng = np.random.default_rng(35)
    samples = [rng.normal(m, 0.1, 30) for m in (0.3, 0.5, 0.7, 0.9)]
    y = np.concatenate(samples)
    g = np.repeat(["A", "B", "C", "D"], 30)
    tbl = group_tests_nonparam(y, g)["pairwise"]
    assert len(tbl) == 6
    npairs = 6
    for _, row in tbl.iterrows():
        assert 0.0 <= float(row["p_raw"]) <= 1.0
        assert 0.0 <= float(row["p_bonf"]) <= 1.0
        assert float(row["p_bonf"]) == pytest.approx(
            min(float(row["p_raw"]) * npairs, 1.0), rel=1e-12)
        assert float(row["p_bonf"]) >= float(row["p_raw"]) - 1e-15


def test_dunn_direction_labels_follow_the_medians():
    rng = np.random.default_rng(37)
    y = np.concatenate([rng.normal(0.3, 0.05, 40), rng.normal(0.8, 0.05, 40)])
    g = np.repeat(["low", "high"], 40)
    row = group_tests_nonparam(y, g)["pairwise"].iloc[0]
    assert float(row["median_A"]) < float(row["median_B"])
    assert row["direction"] == "low < high"


def test_group_tests_handle_a_single_group_gracefully():
    y = np.linspace(0.3, 0.9, 20)
    g = np.array(["only"] * 20)
    for fn in (group_tests_param, group_tests_nonparam):
        try:
            fn(y, g)
        except Exception as exc:        # a clear error is acceptable
            assert isinstance(exc, (ValueError, KeyError, IndexError)), exc


# --------------------------------------------------------------------------- #
#  Simar-Wilson (2007) Algorithm 1                                            #
# --------------------------------------------------------------------------- #
def _sw_sample(rng, n=250, beta=(0.55, 0.20, -0.10), sigma=0.12):
    """Draw from the model Algorithm 1 assumes: truncated-normal responses."""
    from scipy import stats
    beta = np.asarray(beta, float)
    Z = rng.uniform(0.0, 1.0, (n, len(beta) - 1))
    mu = np.column_stack([np.ones(n), Z]) @ beta
    th = stats.truncnorm.rvs(-np.inf, (1.0 - mu) / sigma, loc=mu, scale=sigma,
                             random_state=rng)
    return th, Z, beta, sigma


def test_algorithm1_recovers_known_coefficients():
    """On data generated from its own model, the fit must find the truth."""
    from pleap.core.secondstage import simarwilson_algorithm1
    rng = np.random.default_rng(0)
    th, Z, beta, sigma = _sw_sample(rng)
    r = simarwilson_algorithm1(th, Z, L=150, seed=1, names=["z1", "z2"])
    got = np.asarray(r["summary"].attrs["coef_raw"], float)
    se = np.asarray(r["summary"].attrs["se_raw"], float)
    assert np.all(np.abs(got - beta) < 4 * se), f"{got} vs {beta} (se {se})"
    assert abs(r["sigma"] - sigma) < 0.02
    assert r["n"] == len(th)


def test_algorithm1_is_reproducible_from_its_seed():
    from pleap.core.secondstage import simarwilson_algorithm1
    rng = np.random.default_rng(2)
    th, Z, _, _ = _sw_sample(rng, n=120)
    a = simarwilson_algorithm1(th, Z, L=60, seed=7, names=["z1", "z2"])
    b = simarwilson_algorithm1(th, Z, L=60, seed=7, names=["z1", "z2"])
    np.testing.assert_allclose(a["summary"].attrs["se_raw"],
                               b["summary"].attrs["se_raw"], rtol=0, atol=0)
    c = simarwilson_algorithm1(th, Z, L=60, seed=8, names=["z1", "z2"])
    assert not np.allclose(a["summary"].attrs["se_raw"],
                           c["summary"].attrs["se_raw"]), \
        "a different seed must give a different bootstrap"


def test_algorithm1_point_estimates_match_the_truncated_fit():
    """Algorithm 1 changes the inference, not the point estimate.

    The coefficients are the maximum-likelihood truncated-regression ones;
    only the standard errors and intervals come from the bootstrap.  If these
    ever diverge, one of the two is wrong.
    """
    from pleap.core.secondstage import (simarwilson_algorithm1,
                                        truncated_regression)
    rng = np.random.default_rng(5)
    th, Z, _, _ = _sw_sample(rng, n=150)
    a = simarwilson_algorithm1(th, Z, L=50, seed=3, names=["z1", "z2"])
    t = truncated_regression(th, Z, lower=None, upper=1.0, names=["z1", "z2"])
    np.testing.assert_allclose(
        np.asarray(a["summary"].attrs["coef_raw"], float),
        np.asarray(t["summary"].attrs["coef_raw"], float)[:-1],
        rtol=0, atol=1e-10)


def test_frontier_units_a_rounding_error_below_one_stay_out():
    """A frontier unit is 1 only up to rounding; that must not decide the sample.

    On the example used up to v1.11 the BCC input-oriented run put 11 of 60
    units on the frontier, five of them at 1 - 4e-15.  A strict ``theta < 1``
    let those five into the truncated sample (54 units instead of 49).  The fit must equal
    the fit on the units genuinely inside the frontier.
    """
    from pleap.core.secondstage import (simarwilson_algorithm1,
                                        truncated_regression)
    rng = np.random.default_rng(11)
    th, Z, _, _ = _sw_sample(rng, n=80)
    noisy = th.copy()
    noisy[:6] = [1 - 4e-15, 1 - 2.8e-15, 1 - 1.1e-16, 1.0, 1 + 2.2e-16, 1 + 3.6e-14]
    inside = noisy < 1 - 1e-6
    a = simarwilson_algorithm1(noisy, Z, L=20, seed=4, names=["z1", "z2"])
    assert a["n"] == int(inside.sum()) == 74
    t_noisy = truncated_regression(noisy, Z, lower=None, upper=1.0, names=["z1", "z2"])
    t_clean = truncated_regression(noisy[inside], Z[inside], lower=None, upper=1.0,
                                   names=["z1", "z2"])
    assert t_noisy["n"] == t_clean["n"] == 74
    np.testing.assert_allclose(t_noisy["summary"].attrs["coef_raw"],
                               t_clean["summary"].attrs["coef_raw"], rtol=0, atol=0)


def test_algorithm1_refuses_when_nothing_is_inefficient():
    """Every unit on the frontier leaves the truncated likelihood with no data."""
    from pleap.core.secondstage import simarwilson_algorithm1
    rng = np.random.default_rng(9)
    Z = rng.uniform(0, 1, (40, 2))
    with pytest.raises(ValueError, match="inside the frontier"):
        simarwilson_algorithm1(np.ones(40), Z, L=10, seed=1)
