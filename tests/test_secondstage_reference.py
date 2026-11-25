"""Second-stage regressions and group tests against Stata, R and SPSS, on the
private validation data.

Data set D (453 firms) and its 207-firm subset, with the reference output the
author produced in Stata 16, R and SPSS 25; data set A (60 firms) for SPSS's
rank tests, where the SPSS printout has three decimals.  A and D come from the
Korean Innovation Survey (KIS) of STEPI and are not redistributed, so every
test here is skipped unless the files are present (conftest.py).  The same
statistics are checked against R on the public synthetic data set SD in
``test_secondstage_r_reference.py``.

Each PLEAP call gets exactly the dependent variable the reference program was
given: ``vrs_i`` (the BCC input-oriented score rounded to eight decimals) for
D's regressions and rank tests, and ``y`` for the subset's OLS, t-test and
ANOVA.  ``y`` is a bias-corrected score from an earlier PLEAP release; here it
is only a fixed vector, so these tests check the statistics and nothing
upstream of them.

What v1.12 changed, and which test pins it:

* rank tests treat scores that agree to 1e-9 as ties.  The linear programs
  return frontier scores as 1 only up to rounding, so on PLEAP's own scores
  the tie correction was lost: Kruskal-Wallis H = 6.381 on data set A where
  SPSS, reading the exported scores, reports 6.356
  (``test_rank_tests_on_pleaps_own_scores_match_spss_*``);
* Tukey p-values are no longer read off statsmodels' summary table, which
  rounds to four decimals (p = 3.9e-6 printed as 0.0000)
  (``test_tukey_matches_spss_to_full_precision``);
* the truncated likelihood computes the truncation probability without
  cancellation, and the fit falls back to a trust-region Newton step when
  BFGS's line search fails.  Truncated at 0 only, v1.11 stopped after two
  iterations at log-likelihood 88.79 against Stata's 105.53
  (``test_truncated_at_zero_only_matches_stata_and_r``);
* p-values use upper-tail functions, so a p-value of 1e-14 keeps its
  relative precision.
"""

import math

import numpy as np
import pandas as pd
import pytest

from pleap.core.dea import dea
from pleap.core.secondstage import (_games_howell, _tukey_hsd,
                                    group_tests_nonparam, group_tests_param,
                                    ols_regression, truncated_regression)


pytestmark = pytest.mark.private


def _get(fit, key, name):
    raw = np.asarray(fit["summary"].attrs[key], float)
    return float(raw[list(map(str, fit["summary"].index)).index(name)])


def _pair(df, a, b):
    for _, r in df.iterrows():
        if (r["group_A"], r["group_B"]) == (a, b):
            return r, 1.0
        if (r["group_A"], r["group_B"]) == (b, a):
            return r, -1.0
    raise KeyError((a, b))


# --------------------------------------------------------------------------- #
#  Truncated regression - Stata truncreg (7 significant digits) and R truncreg #
# --------------------------------------------------------------------------- #
STATA_LL0_UL1 = dict(n=437, coef={"const": -.9525321, "x1": .2394901, "x2": -.1329138},
                     se={"const": .814265, "x1": .1893737, "x2": .1541708},
                     sigma=.5966882, sigma_se=.1779407, loglik=151.63698)
STATA_LL0 = dict(n=453, coef={"const": -.8999378, "x1": .1643142, "x2": -.0429942},
                 se={"const": .4834559, "x1": .124706, "x2": .1163285},
                 sigma=.595835, sigma_se=.1002752, loglik=105.52705)
# R 4.3.3, truncreg 0.2-5: truncreg(vrs_i ~ x1 + x2, point = 0, direction = "left")
R_LL0 = dict(coef={"const": -0.899937681717337, "x1": 0.164314230595188,
                   "x2": -0.0429959474214148},
             se={"const": 0.483456656945402, "x1": 0.124706305498928,
                 "x2": 0.116328816687072},
             sigma=0.595835697026286, sigma_se=0.100275605019977, loglik=105.5270540409)


def _check_trunc(fit, ref, rel, loglik_abs, coef_abs=0.0):
    for name in ("const", "x1", "x2"):
        assert _get(fit, "coef_raw", name) == pytest.approx(ref["coef"][name], rel=rel, abs=coef_abs)
        assert _get(fit, "se_raw", name) == pytest.approx(ref["se"][name], rel=rel)
    assert fit["sigma"] == pytest.approx(ref["sigma"], rel=rel)
    # sigma's standard error by the delta method from Log(scale)'s
    assert fit["sigma"] * _get(fit, "se_raw", "Log(scale)") == pytest.approx(ref["sigma_se"], rel=rel)
    assert fit["loglik"] == pytest.approx(ref["loglik"], abs=loglik_abs)


def test_truncated_at_zero_and_one_matches_stata(ds_d):
    """``truncreg vrs_i x1 x2, ll(0) ul(1)``: Stata prints seven significant
    digits and stops at its own convergence tolerance, so agreement is to
    about 1e-5 relative; PLEAP's estimate has the higher likelihood and a
    gradient 300 times smaller than Stata's printed one."""
    fit = truncated_regression(ds_d["vrs_i"].to_numpy(float), ds_d[["x1", "x2"]].to_numpy(float),
                               lower=0.0, upper=1.0, names=["x1", "x2"])
    assert fit["converged"] and fit["n"] == STATA_LL0_UL1["n"]
    _check_trunc(fit, STATA_LL0_UL1, rel=1e-5, loglik_abs=5e-6)


def test_truncated_at_zero_only_matches_stata_and_r(ds_d):
    """Truncated at 0 only.  v1.11 stopped here after two BFGS iterations at
    log-likelihood 88.79.

    Agreement with R's full-precision output is ~3e-6 relative, not better,
    and the reason is R's stopping point, not PLEAP's: the gradient of the
    log-likelihood is 3e-4 at R's estimate and 1e-6 at PLEAP's, and the
    standard errors, evaluated where each program stopped, move by that much.
    The last assertion states it: PLEAP's optimum is not below R's."""
    y = ds_d["vrs_i"].to_numpy(float); Z = ds_d[["x1", "x2"]].to_numpy(float)
    fit = truncated_regression(y, Z, lower=0.0, upper=None, names=["x1", "x2"])
    assert fit["converged"] and fit["n"] == STATA_LL0["n"]
    _check_trunc(fit, STATA_LL0, rel=1e-5, loglik_abs=5e-6)
    _check_trunc(fit, R_LL0, rel=1e-5, loglik_abs=1e-8, coef_abs=5e-6)
    from pleap.core.secondstage import _truncated_negll_grad
    Zc = np.column_stack([np.ones(len(Z)), Z])
    r_par = np.r_[[R_LL0["coef"][k] for k in ("const", "x1", "x2")], np.log(R_LL0["sigma"])]
    assert fit["loglik"] >= -_truncated_negll_grad(r_par, y, Zc, 0.0, None)[0] - 1e-12


# --------------------------------------------------------------------------- #
#  Rank tests - SPSS (asymptotic significance)                                #
# --------------------------------------------------------------------------- #
SPSS_D_MWU = dict(U=12927.0, z=2.95465731752133, p=0.00313016520457321)  # SPSS U = 19469 = 89*364 - 12927
SPSS_D_KW_H = 76.682
SPSS_D_DUNN = {(3, 4): (-3.67701664219457, 0.000235977650644204, 0.00141586590386522),
               (3, 1): (6.07413865153363, 1.24654997435414e-09, 7.47929984612483e-09),
               (3, 2): (7.69188807102381, 1.46549439250521e-14, 8.79296635503124e-14),
               (4, 1): (2.74094693055639, 0.00612624014887508, 0.0367574408932505),
               (4, 2): (4.73921368995351, 2.14549152044441e-06, 1.28729491226665e-05),
               (1, 2): (-2.15441583675, 0.0312075757228261, 0.187245454336956)}


def _check_d_rank_tests(y, ds_d):
    m = group_tests_nonparam(y, ds_d["group1"].to_numpy())
    pw = m["pairwise"].iloc[0]
    assert m["test"] == "Mann-Whitney U"
    assert m["stat"] == pytest.approx(SPSS_D_MWU["U"], abs=1e-9)
    assert abs(pw["z"]) == pytest.approx(SPSS_D_MWU["z"], rel=1e-12)
    assert m["p"] == pytest.approx(SPSS_D_MWU["p"], rel=1e-10, abs=0)
    k = group_tests_nonparam(y, ds_d["group2"].to_numpy())
    assert k["stat"] == pytest.approx(SPSS_D_KW_H, abs=5e-4)
    for (a, b), (z, p, padj) in SPSS_D_DUNN.items():
        row, sign = _pair(k["pairwise"], a, b)
        assert abs(row["z (Dunn)"]) == pytest.approx(abs(z), rel=1e-12)   # SPSS orders the pair by mean rank
        # PLEAP's p is the exact normal tail; SPSS's is 1 - cdf, which loses
        # relative precision as p falls (5.7e-8 at p = 1.2e-9, 1.1e-2 at
        # p = 1.5e-14), so the comparison with SPSS loosens accordingly.
        exact = math.erfc(abs(row["z (Dunn)"]) / math.sqrt(2.0))
        assert row["p_raw"] == pytest.approx(exact, rel=1e-12, abs=0)
        tol = 1e-9 if p > 1e-6 else (1e-6 if p > 1e-12 else 2e-2)
        assert row["p_raw"] == pytest.approx(p, rel=tol, abs=0)
        assert row["p_bonf"] == pytest.approx(padj, rel=tol, abs=0)


def test_rank_tests_match_spss_on_the_exported_scores(ds_d):
    _check_d_rank_tests(ds_d["vrs_i"].to_numpy(float), ds_d)


@pytest.fixture(scope="module")
def own_scores_d(ds_d):
    return dea(ds_d[["i1", "i2"]].to_numpy(float), ds_d[["o1"]].to_numpy(float),
               model="BCC", orient="input")["efficiency"]


def test_rank_tests_on_pleaps_own_scores_match_spss_d(ds_d, own_scores_d):
    """What a user of the GUI gets: PLEAP's unrounded scores, 16 of them on
    the frontier at 1 +/- a few ulp.  Before v1.12, U was off by 4.5 and H by
    0.20 here."""
    assert np.max(np.abs(own_scores_d - ds_d["vrs_i"])) < 5e-9
    assert len(np.unique(own_scores_d[own_scores_d > 1 - 1e-9])) >= 1
    _check_d_rank_tests(own_scores_d, ds_d)


# Data set A, SPSS printout (three decimals).
SPSS_A_KW = dict(H=6.356, p=.096)
SPSS_A_DUNN = {(4, 3): (.666, .505, 1.000), (4, 2): (1.730, .084, .502), (4, 1): (2.321, .020, .122),
               (3, 2): (.961, .336, 1.000), (3, 1): (1.553, .121, .723), (2, 1): (.724, .469, 1.000)}


def test_rank_tests_on_pleaps_own_scores_match_spss_a(ds_a, ds_a_xy):
    theta = dea(*ds_a_xy, model="BCC", orient="input")["efficiency"]
    m = group_tests_nonparam(theta, ds_a["g2"].to_numpy())
    assert m["stat"] == 252.0
    assert abs(m["pairwise"].iloc[0]["z"]) == pytest.approx(2.321, abs=5e-4)
    # SPSS's tie-corrected standard error (63.762) implies p = 0.0202789
    assert m["p"] == pytest.approx(0.0202789, abs=5e-8)
    k = group_tests_nonparam(theta, ds_a["g4"].to_numpy())
    assert k["stat"] == pytest.approx(SPSS_A_KW["H"], abs=5e-4)
    assert k["p"] == pytest.approx(SPSS_A_KW["p"], abs=5e-4)
    for (a, b), (z, p, padj) in SPSS_A_DUNN.items():
        row, _ = _pair(k["pairwise"], a, b)
        assert abs(row["z (Dunn)"]) == pytest.approx(z, abs=5e-4)
        assert row["p_raw"] == pytest.approx(p, abs=5e-4)
        assert row["p_bonf"] == pytest.approx(padj, abs=5e-4)


# --------------------------------------------------------------------------- #
#  OLS, t-test, ANOVA, post hoc - SPSS on the 207-firm subset                 #
# --------------------------------------------------------------------------- #
SPSS_OLS = {"const": (0.32295011044038613, 0.10722964148240702, 0.00292577855240177),
            "x1": (0.06935691164377392, 0.06502786729775696, 0.28742571253634247),
            "x2": (-0.0908403720728099, 0.07027581706499679, 0.19760246240698942)}


def test_ols_matches_spss(ds_d207):
    r = ols_regression(ds_d207["y"].to_numpy(float), ds_d207[["x1", "x2"]].to_numpy(float),
                       names=["x1", "x2"])
    for name, (b, se, p) in SPSS_OLS.items():
        assert _get(r, "coef_raw", name) == pytest.approx(b, rel=1e-8)
        assert _get(r, "se_raw", name) == pytest.approx(se, rel=1e-8)
        assert _get(r, "p_raw", name) == pytest.approx(p, rel=1e-8)
    assert r["r2"] == pytest.approx(0.008994669468498362, rel=1e-8)
    assert r["adj_r2"] == pytest.approx(-0.0007210690661242029, rel=1e-6)
    assert r["fvalue"] == pytest.approx(0.925783401482591, rel=1e-8)
    assert r["f_pvalue"] == pytest.approx(0.3978790143669346, rel=1e-8)


def test_t_test_matches_spss(ds_d207):
    t = group_tests_param(ds_d207["y"].to_numpy(float), ds_d207["group1"].to_numpy())
    pw = t["pairwise"].set_index("variances")
    assert t["levene"]["stat"] == pytest.approx(0.31455003417655425, rel=1e-8)
    assert t["levene"]["p"] == pytest.approx(0.5755139174268228, rel=1e-8)
    assert abs(pw.loc["equal (Student)", "t"]) == pytest.approx(0.899323069541706, rel=1e-8)
    assert pw.loc["equal (Student)", "p"] == pytest.approx(0.3695357492823247, rel=1e-8)
    assert abs(pw.loc["unequal (Welch)", "t"]) == pytest.approx(0.9217441077731198, rel=1e-8)
    assert pw.loc["unequal (Welch)", "df"] == pytest.approx(97.36567585675613, rel=1e-8)
    assert pw.loc["unequal (Welch)", "p"] == pytest.approx(0.35894041167969104, rel=1e-8)
    # Levene p >= 0.05, so the equal-variance row is the one reported
    assert t["test"].startswith("Student")


def test_anova_matches_spss(ds_d207):
    a = group_tests_param(ds_d207["y"].to_numpy(float), ds_d207["group2"].to_numpy())
    assert a["test"].startswith("One-way ANOVA")          # Levene p = 0.131
    assert a["stat"] == pytest.approx(9.003538743820403, rel=1e-8)
    assert a["p"] == pytest.approx(1.2610984149909027e-05, rel=1e-8, abs=0)


SPSS_TUKEY_P = {(1, 2): 0.08913729570127771, (1, 3): 0.8620258442159799,
                (1, 4): 0.003895540291648847, (2, 3): 0.04903806784662279,
                (2, 4): 3.898760528198153e-06, (3, 4): 0.13460719970190393}
SPSS_DIFF = {(1, 2): -0.19772680376407997, (1, 3): 0.07775067226016247,
             (2, 3): 0.27547747602424244, (3, 4): 0.25043441415315315}
SPSS_GH = {(1, 2): (0.0857502192408153, 0.10282639507989355),
           (1, 3): (0.10956005500360676, 0.892823114147842),
           (1, 4): (0.08222617334596462, 0.0007882244564425545),
           (2, 3): (0.11754587431454291, 0.09917901920535876),
           (2, 4): (0.09260005669057446, 1.0100415966940446e-06),
           (3, 4): (0.11500033065884224, 0.14199758612383417)}


def _samples(ds_d207):
    y = ds_d207["y"].to_numpy(float); g = ds_d207["group2"].to_numpy()
    return [y[g == k] for k in (1, 2, 3, 4)]


def test_tukey_matches_spss_to_full_precision(ds_d207):
    tk = _tukey_hsd(_samples(ds_d207), [1, 2, 3, 4])
    for (a, b), p in SPSS_TUKEY_P.items():
        row, sign = _pair(tk, a, b)
        assert row["p (Tukey)"] == pytest.approx(p, rel=1e-6, abs=0)
        if (a, b) in SPSS_DIFF:   # statsmodels' meandiff is mean_B - mean_A
            assert -sign * row["meandiff"] == pytest.approx(SPSS_DIFF[(a, b)], rel=1e-9)


def test_games_howell_matches_spss(ds_d207):
    gh = _games_howell(_samples(ds_d207), [1, 2, 3, 4])
    for (a, b), (se, p) in SPSS_GH.items():
        row, sign = _pair(gh, a, b)
        diff = sign * (row["mean_A"] - row["mean_B"])
        assert abs(diff / row["t"]) == pytest.approx(se, rel=1e-9)
        assert row["p (GH)"] == pytest.approx(p, rel=1e-6, abs=0)
        if (a, b) in SPSS_DIFF:
            assert diff == pytest.approx(SPSS_DIFF[(a, b)], rel=1e-9)
