"""Second-stage regressions and group tests against R, on public synthetic data.

Data set SD (``tests/data/sd.csv``, 400 synthetic units; conftest.py) and R's
output on it (``tests/data/sd_r_secondstage.json``, written by
``docs/external_reference/reference_sd.R``: R 4.3.3 with truncreg 0.2-5,
car 3.1-2 and rstatix 0.7.2).  The dependent variable is ``vrs_i``, the BCC
input-oriented score rounded to eight decimals; 26 units are on the frontier,
so the rank tests have a block of ties at 1.

These are the public counterparts of ``test_secondstage_reference.py``, which
checks the same statistics against Stata and SPSS on the private validation
data.  The Tobit model is in ``test_tobit_reference.py``.

Measured agreement: truncated regression 2e-13 (at 0) and 3e-10 (at 1, where
R's Newton-Raphson stops at a gradient of 8e-9) on the coefficients; OLS,
t-tests, Levene, ANOVA, Welch's ANOVA, Mann-Whitney, Kruskal-Wallis and Dunn
5e-14 relative or better; Games-Howell 8e-9; Tukey 8e-10, and 2.2e-5 for the
one p-value below 1e-5 (8.4e-7), where R's ``ptukey`` is the less precise of
the two: a 30-digit integration of the studentized range puts SciPy's value
1.4e-7 from it and R's 2.2e-5.
"""

import numpy as np
import pytest
from scipy import stats

from pleap.core import secondstage
from pleap.core.dea import dea
from pleap.core.secondstage import (_games_howell, _snap_ties, _tukey_hsd,
                                    group_tests_nonparam, group_tests_param,
                                    ols_regression, truncated_regression)

NAMES = ("const", "x1", "x2")


def _get(fit, key, name):
    raw = np.asarray(fit["summary"].attrs[key], float)
    return float(raw[list(map(str, fit["summary"].index)).index(name)])


def _pair(df, a, b):
    for _, r in df.iterrows():
        if (str(r["group_A"]), str(r["group_B"])) == (str(a), str(b)):
            return r, 1.0
        if (str(r["group_A"]), str(r["group_B"])) == (str(b), str(a)):
            return r, -1.0
    raise KeyError((a, b))


@pytest.fixture(scope="module")
def yz(sd):
    return sd["vrs_i"].to_numpy(float), sd[["x1", "x2"]].to_numpy(float)


# --------------------------------------------------------------------------- #
#  Truncated regression                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key,lower,upper,coef_rel,coef_abs", [
    ("trunc_0", 0.0, None, 1e-9, 0.0),
    ("trunc_1", None, 1.0, 1e-6, 1e-9),
])
def test_truncated_regression_matches_r_truncreg(yz, sd_r, key, lower, upper, coef_rel, coef_abs):
    """``truncreg(vrs_i ~ x1 + x2, point = 0, direction = "left")`` on all 400
    units, and ``point = 1, direction = "right"`` on the 374 strictly inside
    the frontier."""
    y, Z = yz
    r = sd_r[key]
    fit = truncated_regression(y, Z, lower=lower, upper=upper, names=["x1", "x2"])
    assert fit["converged"] and fit["n"] == r["n"]
    for name in NAMES:
        assert _get(fit, "coef_raw", name) == pytest.approx(r["coef"][name], rel=coef_rel, abs=coef_abs)
        assert _get(fit, "se_raw", name) == pytest.approx(r["se"][name], rel=1e-8)
    assert fit["sigma"] == pytest.approx(r["sigma"], rel=coef_rel)
    # sigma's standard error by the delta method from Log(scale)'s
    assert fit["sigma"] * _get(fit, "se_raw", "Log(scale)") == pytest.approx(r["sigma_se"], rel=1e-8)
    assert fit["loglik"] == pytest.approx(r["loglik"], abs=1e-10)


def test_truncation_at_zero_and_one_matches_60_digit_arithmetic(yz, exact_mle):
    """Two truncation points, which R's truncreg cannot fit (Stata's ``ll(0)
    ul(1)`` is the reference on the private data): the 60-digit maximum of the
    same likelihood instead.  Measured: 9e-12 on the coefficients, 2e-10
    relative on the standard errors."""
    y, Z = yz
    e = exact_mle["sd_trunc_01"]
    fit = truncated_regression(y, Z, lower=0.0, upper=1.0, names=["x1", "x2"])
    assert fit["converged"] and fit["n"] == e["n"] == 374
    for name, want in zip(NAMES, e["coef"]):
        assert _get(fit, "coef_raw", name) == pytest.approx(float(want), abs=1e-10)
    assert _get(fit, "coef_raw", "Log(scale)") == pytest.approx(float(e["log_sigma"]), abs=1e-10)
    for name, want in zip(NAMES + ("Log(scale)",), e["se"]):
        assert _get(fit, "se_raw", name) == pytest.approx(float(want), rel=1e-8)
    assert fit["loglik"] == pytest.approx(float(e["loglik"]), abs=1e-10)


# --------------------------------------------------------------------------- #
#  OLS                                                                        #
# --------------------------------------------------------------------------- #
def test_ols_matches_lm(yz, sd_r):
    y, Z = yz
    r, o = sd_r["ols"], ols_regression(y, Z, names=["x1", "x2"])
    for name in NAMES:
        assert _get(o, "coef_raw", name) == pytest.approx(r["coef"][name], rel=1e-10)
        assert _get(o, "se_raw", name) == pytest.approx(r["se"][name], rel=1e-10)
        assert _get(o, "p_raw", name) == pytest.approx(r["p"][name], rel=1e-9, abs=0)
    assert o["r2"] == pytest.approx(r["r2"], rel=1e-10)
    assert o["adj_r2"] == pytest.approx(r["adj_r2"], rel=1e-10)
    assert o["fvalue"] == pytest.approx(r["f"], rel=1e-10)
    assert o["f_pvalue"] == pytest.approx(r["f_p"], rel=1e-9, abs=0)


# --------------------------------------------------------------------------- #
#  Rank tests                                                                 #
# --------------------------------------------------------------------------- #
def _check_rank_tests(y, sd, sd_r):
    m = group_tests_nonparam(y, sd["group1"].to_numpy())
    r = sd_r["mwu"]
    assert m["test"] == "Mann-Whitney U"
    # PLEAP's U is the statistic of the first group, as R's W
    assert m["stat"] == pytest.approx(r["W"], abs=1e-9)
    assert m["p"] == pytest.approx(r["p"], rel=1e-12, abs=0)
    # R gives no z; the normal quantile of its two-sided p is |z|
    z = float(stats.norm.isf(r["p"] / 2.0))
    assert abs(m["pairwise"].iloc[0]["z"]) == pytest.approx(z, rel=1e-10)

    k = group_tests_nonparam(y, sd["group2"].to_numpy())
    assert k["stat"] == pytest.approx(sd_r["kw"]["H"], rel=1e-12)
    assert k["p"] == pytest.approx(sd_r["kw"]["p"], rel=1e-10, abs=0)
    for e in sd_r["dunn"]:
        row, _ = _pair(k["pairwise"], int(e["a"]), int(e["b"]))
        assert abs(row["z (Dunn)"]) == pytest.approx(abs(e["z"]), rel=1e-12)
        assert row["p_raw"] == pytest.approx(e["p"], rel=1e-10, abs=0)
        assert row["p_bonf"] == pytest.approx(e["p_adj"], rel=1e-10, abs=0)


def test_rank_tests_match_r_on_the_exported_scores(sd, sd_r):
    _check_rank_tests(sd["vrs_i"].to_numpy(float), sd, sd_r)


def test_rank_tests_on_pleaps_own_scores_match_r(sd, sd_xy, sd_r):
    """What a user of the GUI gets: unrounded scores, the 26 frontier units at
    1 up to rounding.  They must rank exactly as the exported column."""
    own = dea(*sd_xy, model="BCC", orient="input")["efficiency"]
    assert np.max(np.abs(own - sd["vrs_i"])) < 5.0e-9 + 1e-12
    _check_rank_tests(own, sd, sd_r)


def test_scores_equal_to_rounding_are_ties():
    y = np.array([1 - 4e-15, 1.0, 1 + 2e-16, 0.5, 0.25, 0.25 + 1e-12, 0.75, 0.1])
    g = np.array([1, 2, 1, 2, 1, 2, 1, 2])
    snapped = _snap_ties(y)
    assert len(np.unique(snapped)) == 5
    clean = np.array([1.0, 1.0, 1.0, 0.5, 0.25, 0.25, 0.75, 0.1])
    a = group_tests_nonparam(y, g)
    b = group_tests_nonparam(clean, g)
    assert a["stat"] == b["stat"] and a["p"] == pytest.approx(b["p"], rel=1e-14)
    # a genuine difference larger than the tolerance is not merged
    assert len(np.unique(_snap_ties(np.array([0.3, 0.3 + 2e-9])))) == 2


# --------------------------------------------------------------------------- #
#  t-test, Levene, ANOVA, Welch's ANOVA, post hoc                             #
# --------------------------------------------------------------------------- #
def test_t_tests_and_levene_match_r(sd, sd_r):
    t = group_tests_param(sd["vrs_i"].to_numpy(float), sd["group1"].to_numpy())
    pw = t["pairwise"].set_index("variances")
    r = sd_r["t_test"]
    assert t["levene"]["stat"] == pytest.approx(sd_r["levene_2"]["F"], rel=1e-10)
    assert t["levene"]["p"] == pytest.approx(sd_r["levene_2"]["p"], rel=1e-10)
    assert abs(pw.loc["equal (Student)", "t"]) == pytest.approx(abs(r["student"]["t"]), rel=1e-10)
    assert pw.loc["equal (Student)", "p"] == pytest.approx(r["student"]["p"], rel=1e-10)
    assert abs(pw.loc["unequal (Welch)", "t"]) == pytest.approx(abs(r["welch"]["t"]), rel=1e-10)
    assert pw.loc["unequal (Welch)", "df"] == pytest.approx(r["welch"]["df"], rel=1e-10)
    assert pw.loc["unequal (Welch)", "p"] == pytest.approx(r["welch"]["p"], rel=1e-10)
    assert t["test"].startswith("Student")                # Levene p = 0.094


@pytest.mark.parametrize("forced_levene_p,key,label", [
    (None, "anova", "One-way ANOVA"),     # SD's own Levene p, 0.20: equal variances
    (0.01, "welch", "Welch ANOVA"),       # Levene forced below 0.05: Welch's branch
])
def test_anova_and_welch_anova_match_oneway_test(sd, sd_r, monkeypatch, forced_levene_p, key, label):
    """PLEAP reports one of the two, by Levene's test; both are checked, the
    second by forcing the Levene p-value (Welch's F is PLEAP's own code)."""
    y, g = sd["vrs_i"].to_numpy(float), sd["group2"].to_numpy()
    if forced_levene_p is not None:
        real = stats.levene
        monkeypatch.setattr(secondstage.stats, "levene",
                            lambda *a, **k: (real(*a, **k)[0], forced_levene_p))
    a = group_tests_param(y, g)
    assert a["test"].startswith(label)
    assert a["stat"] == pytest.approx(sd_r[key]["F"], rel=1e-10)
    assert a["p"] == pytest.approx(sd_r[key]["p"], rel=1e-9, abs=0)
    if forced_levene_p is None:
        assert a["levene"]["stat"] == pytest.approx(sd_r["levene_4"]["F"], rel=1e-10)
        assert a["levene"]["p"] == pytest.approx(sd_r["levene_4"]["p"], rel=1e-10)


def _samples(sd):
    y, g = sd["vrs_i"].to_numpy(float), sd["group2"].to_numpy()
    return [y[g == k] for k in (1, 2, 3, 4)]


def test_tukey_matches_r(sd, sd_r):
    tk = _tukey_hsd(_samples(sd), [1, 2, 3, 4])
    for e in sd_r["tukey"]:
        row, sign = _pair(tk, int(e["a"]), int(e["b"]))
        # R's ptukey is the less precise of the two below p = 1e-5
        tol = 1e-8 if e["p_adj"] > 1e-5 else 1e-4
        assert row["p (Tukey)"] == pytest.approx(e["p_adj"], rel=tol, abs=0)
        # statsmodels' meandiff is mean_B - mean_A; R's diff is mean_b - mean_a
        assert sign * row["meandiff"] == pytest.approx(e["diff_b_minus_a"], rel=1e-10)


def test_games_howell_matches_r(sd, sd_r):
    gh = _games_howell(_samples(sd), [1, 2, 3, 4])
    for e in sd_r["games_howell"]:
        row, sign = _pair(gh, int(e["a"]), int(e["b"]))
        diff = sign * (row["mean_B"] - row["mean_A"])
        assert diff == pytest.approx(e["diff_b_minus_a"], rel=1e-10)
        assert abs(diff / row["t"]) == pytest.approx(e["se"], rel=1e-12)
        assert row["df"] == pytest.approx(e["df"], rel=1e-12)
        assert row["p (GH)"] == pytest.approx(e["p_adj"], rel=1e-7, abs=0)
