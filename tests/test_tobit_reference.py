"""Tobit regression against R, and against the likelihood maximum computed in
60-digit arithmetic.

Two censoring specifications are pinned:

* ``[0, 1]``  - censored at 0 and at the frontier, PLEAP's default for an
  efficiency score;
* ``[0, inf)`` - censored at 0 only.

The dependent variable is ``vrs_i``, a BCC input-oriented score rounded to
eight decimals as an exported column would be, regressed on two
environmental variables.  Two data sets:

* SD (public, synthetic, 400 units): R ``AER::tobit(vrs_i ~ x1 + x2, left = 0,
  right = 1 | Inf)`` (AER 1.2-12 on survival 3.5-8, R 4.3.3) at full precision
  (``tests/data/sd_r_secondstage.json``, ``docs/external_reference/
  reference_sd.R``), and the 60-digit maximum of the same likelihood
  (``tests/data/exact_mle.json``, ``docs/external_reference/exact_mle.py``).
  Measured agreement: 9e-12 on the coefficients and 6e-11 relative on the
  standard errors against R, 6e-11 on the coefficients against the 60-digit
  maximum.
* D (private: 453 firms of the Korean Innovation Survey, not redistributed,
  skipped without it): R ``AER::tobit`` as run by the author through SPSS's R
  extension, full-precision coefficients and standard errors; log-likelihood,
  scale and Wald statistic from ``docs/external_reference/secondstage_ds_d.R``
  (survreg), which agrees with that run to 1e-11.

Up to v1.11 this file pinned a 135-plant data set (B) whose release was never
agreed; it is no longer distributed.  The behaviour that data set guarded -
covariates in the millions, which made v1.0.7 collapse to zero coefficients -
is kept by ``test_rescaling_a_covariate_moves_only_its_own_slope`` below.
"""

import numpy as np
import pandas as pd
import pytest

from pleap.core.secondstage import tobit_regression

R = {
    "[0,1]": dict(
        lower=0.0, upper=1.0,
        coef={"const": 0.25859328154815986, "x1": 0.029727307572200583,
              "x2": -0.006418046399022957, "Log(scale)": -1.3347317604477806},
        se={"const": 0.03479642661427655, "x1": 0.02188776705124604,
            "x2": 0.021612741061160495, "Log(scale)": 0.03427240168637602},
        p={"const": 1.0728816881044883e-13, "x1": 0.174409762085515,
           "x2": 0.7664996329852303},
        loglik=-62.3289709685, scale=0.2632287744, wald=2.2337103662,
        wald_p=0.3273075000),
    "[0,inf)": dict(
        lower=0.0, upper=None,
        coef={"const": 0.2573911049852861, "x1": 0.029806779396835213,
              "x2": -0.007550117253392389, "Log(scale)": -1.3680540351978805},
        se={"const": 0.03359063580710902, "x1": 0.021147066004005604,
            "x2": 0.020881827795682754, "Log(scale)": 0.03322277470781851},
        p={"const": 1.8222732412157827e-14, "x1": 0.1586875150859152,
           "x2": 0.7176778684252305},
        loglik=-23.0506776212, scale=0.2546019242, wald=2.3271335754,
        wald_p=0.3123700339),
}
SPECS = list(R)
SD_KEYS = {"[0,1]": ("tobit_01", 1.0), "[0,inf)": ("tobit_0inf", None)}

# data fixture, number of units, number censored at 1
CASES = {"SD (public)": ("sd", 400, 26), "D (private)": ("ds_d", 453, 16)}


def _sd_reference(sd_r):
    out = {}
    for spec, (key, upper) in SD_KEYS.items():
        r = sd_r[key]
        out[spec] = dict(lower=0.0, upper=upper, coef=r["coef"], se=r["se"], p=r["p"],
                         loglik=r["loglik"], scale=r["scale"], wald=r["wald"],
                         wald_p=r["wald_p"])
    return out


@pytest.fixture(scope="module", params=list(CASES))
def case(request):
    data, n, n_up = CASES[request.param]
    ds = request.getfixturevalue(data)
    assert len(ds) == n
    ref = _sd_reference(request.getfixturevalue("sd_r")) if data == "sd" else R
    y = ds["vrs_i"].to_numpy(float)
    Z = ds[["x1", "x2"]].to_numpy(float)
    fits = {s: tobit_regression(y, Z, lower=ref[s]["lower"], upper=ref[s]["upper"],
                                names=["x1", "x2"]) for s in SPECS}
    return dict(ds=ds, ref=ref, n=n, n_up=n_up, fits=fits)


def _get(fit, key, name):
    raw = np.asarray(fit["summary"].attrs[key], float)
    return float(raw[list(map(str, fit["summary"].index)).index(name)])


@pytest.mark.parametrize("spec", SPECS)
def test_it_converges(case, spec):
    assert case["fits"][spec]["converged"]
    assert case["fits"][spec]["n"] == case["n"]


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("name", ["const", "x1", "x2", "Log(scale)"])
def test_coefficients_and_standard_errors_match_r(case, spec, name):
    f, r = case["fits"][spec], case["ref"][spec]
    assert _get(f, "coef_raw", name) == pytest.approx(r["coef"][name], rel=1e-7, abs=1e-10)
    assert _get(f, "se_raw", name) == pytest.approx(r["se"][name], rel=1e-7)


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("name", ["const", "x1", "x2"])
def test_p_values_match_r_including_the_tail(case, spec, name):
    """The constant's p-value is ~1e-13 on D and ~1e-15 on SD: computed as
    1 - cdf it would carry an error of ~1e-16 absolute, i.e. ~1e-3 relative or
    worse.  v1.12 uses the upper tail function, so it matches R to 1e-6
    relative even there."""
    got = _get(case["fits"][spec], "p_raw", name)
    assert got == pytest.approx(case["ref"][spec]["p"][name], rel=1e-6, abs=0)


@pytest.mark.parametrize("spec", SPECS)
def test_scale_log_likelihood_and_wald_match_r(case, spec):
    f, r = case["fits"][spec], case["ref"][spec]
    assert f["sigma"] == pytest.approx(r["scale"], abs=5e-10)
    assert f["loglik"] == pytest.approx(r["loglik"], abs=5e-9)
    assert f["wald"] == pytest.approx(r["wald"], abs=5e-8)
    assert f["wald_df"] == 2
    assert f["wald_p"] == pytest.approx(r["wald_p"], abs=5e-9)


@pytest.mark.parametrize("spec", SPECS)
def test_z_statistics_are_consistent_with_the_coefficients(case, spec):
    f = case["fits"][spec]
    for name in ("const", "x1", "x2", "Log(scale)"):
        z = float(f["summary"].loc[name, "z"])
        assert z == pytest.approx(_get(f, "coef_raw", name) / _get(f, "se_raw", name), rel=1e-3)


def test_censoring_counts_follow_the_bounds(case):
    """Frontier scores are censored at 1, not treated as interior points."""
    y = case["ds"]["vrs_i"].to_numpy(float)
    assert int(np.sum(y >= 1.0 - 1e-9)) == case["n_up"]
    assert int(np.sum(y <= 0.0)) == 0


@pytest.mark.parametrize("spec", SPECS)
def test_the_maximum_matches_60_digit_arithmetic(sd, exact_mle, spec):
    """The same likelihood maximised in 60-digit arithmetic, independently of
    R and of PLEAP: measured agreement 6e-11 on the coefficients, 6e-11
    relative on the standard errors, 5e-14 on the log-likelihood."""
    key, upper = SD_KEYS[spec]
    e = exact_mle["sd_" + key]
    f = tobit_regression(sd["vrs_i"].to_numpy(float), sd[["x1", "x2"]].to_numpy(float),
                         lower=0.0, upper=upper, names=["x1", "x2"])
    want = [float(v) for v in e["coef"]] + [float(e["log_sigma"])]
    for name, w, se in zip(("const", "x1", "x2", "Log(scale)"), want, e["se"]):
        assert _get(f, "coef_raw", name) == pytest.approx(w, abs=1e-9)
        assert _get(f, "se_raw", name) == pytest.approx(float(se), rel=1e-8)
    assert f["loglik"] == pytest.approx(float(e["loglik"]), abs=1e-10)
    assert f["wald"] == pytest.approx(float(e["wald"]), rel=1e-8)


def test_rescaling_a_covariate_moves_only_its_own_slope(sd):
    """The back-transform must be exact, not approximate.

    Multiplying x2 by 1e6 puts it in the range that made v1.0.7's unscaled
    optimiser stall and return coefficients of ~0.  It must divide x2's
    coefficient and standard error by 1e6 and leave everything else alone.
    """
    y = sd["vrs_i"].to_numpy(float)
    Z = sd[["x1", "x2"]].to_numpy(float)
    base = tobit_regression(y, Z, lower=0.0, upper=1.0, names=["x1", "x2"])
    Zb = Z.copy(); Zb[:, 1] *= 1e6
    big = tobit_regression(y, Zb, lower=0.0, upper=1.0, names=["x1", "x2"])
    assert big["converged"]
    assert _get(big, "coef_raw", "x2") == pytest.approx(_get(base, "coef_raw", "x2") / 1e6, rel=1e-6)
    assert _get(big, "se_raw", "x2") == pytest.approx(_get(base, "se_raw", "x2") / 1e6, rel=1e-5)
    assert _get(big, "coef_raw", "x1") == pytest.approx(_get(base, "coef_raw", "x1"), rel=1e-6)
    assert _get(big, "coef_raw", "const") == pytest.approx(_get(base, "coef_raw", "const"), rel=1e-6)
    assert big["sigma"] == pytest.approx(base["sigma"], rel=1e-7)
    assert big["loglik"] == pytest.approx(base["loglik"], rel=1e-9)
    assert big["wald"] == pytest.approx(base["wald"], rel=1e-5)
    # and the v1.0.7 symptom, stated so it cannot come back silently
    assert abs(_get(big, "coef_raw", "x2")) > 1e-12
    assert _get(big, "se_raw", "x2") > 1e-12
