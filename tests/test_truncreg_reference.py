"""Truncated regression against R's ``truncreg`` and a 60-digit computation.

``tests/data/example_bcc_input.csv`` holds the last-period BCC input-oriented
scores of the example in Section 3 of the paper (``examples/sample_data.xlsx``,
synthetic), computed by R Benchmarking, and the two environmental variables.
Twelve of the 60 units are on the frontier; the truncated likelihood must use
the 48 units strictly inside it, which is also the sample Simar-Wilson
Algorithm 1 fits.

Reference 1: R 4.3.3, truncreg 0.2-5, ``truncreg(theta ~ z1 + z2, point = 1,
direction = "right", method = "nr")`` on those 48 units
(``docs/external_reference/reference_example.R``).  With truncreg's default
BFGS the fit stops early; its Newton-Raphson ends with a gradient of 9e-7,
and the coefficients agree with PLEAP's to 1.4e-9, which is truncreg's own
distance from the maximum.

Reference 2: the same likelihood maximised by Newton's method in 60-digit
arithmetic (``docs/external_reference/exact_mle.py``,
``tests/data/exact_mle.json``): PLEAP agrees to 2e-16 on the coefficients
and 5e-10 relative on the standard errors; the test asks 1e-12 of the
coefficients, which holds on every platform since the Newton polish after
BFGS.

Earlier versions of this file pinned the previous example, which was drawn
from real data and is no longer distributed.
"""

import numpy as np
import pandas as pd
import pytest

from pleap.core.secondstage import simarwilson_algorithm1, truncated_regression

NAMES = ["const", "z1", "z2"]


@pytest.fixture(scope="module")
def data(request):
    d = pd.read_csv(request.path.parent / "data" / "example_bcc_input.csv")
    assert len(d) == 60
    return d["theta"].to_numpy(float), d[["z1", "z2"]].to_numpy(float)


@pytest.fixture(scope="module")
def r_truncreg(request):
    r = pd.read_csv(request.path.parent / "data" / "example_truncreg_R.csv")
    return dict(zip(r["name"], r["value"].astype(float)))


@pytest.fixture(scope="module")
def fit(data):
    theta, Z = data
    t = truncated_regression(theta, Z, lower=None, upper=1.0, names=["z1", "z2"])
    c = t["summary"].attrs["coef_raw"]
    s = t["summary"].attrs["se_raw"]
    return dict(n=t["n"], const=c[0], z1=c[1], z2=c[2], sigma=t["sigma"], loglik=t["loglik"],
                se_const=s[0], se_z1=s[1], se_z2=s[2], se_logsigma=s[3],
                converged=t["converged"])


def test_only_units_strictly_inside_the_frontier_enter(fit, data, r_truncreg):
    theta, _ = data
    assert int((theta >= 1 - 1e-6).sum()) == 12
    assert fit["n"] == 48 == int(r_truncreg["n"])
    assert fit["converged"]


def test_the_stored_scores_are_pleaps_dea_scores(data, example_panel):
    """The stored scores are Benchmarking's; PLEAP's DEA gives the same."""
    from pleap.core.dea import dea
    last = example_panel[example_panel["Period"] == example_panel["Period"].max()]
    e = dea(last[["x1", "x2"]].to_numpy(float), last[["y1", "y2"]].to_numpy(float),
            model="BCC", orient="input")["efficiency"]
    assert np.max(np.abs(e - data[0])) < 1e-12


@pytest.mark.parametrize("name", ["const", "z1", "z2", "sigma"])
def test_estimates_match_r_truncreg(fit, r_truncreg, name):
    assert fit[name] == pytest.approx(r_truncreg[name], abs=1e-7)


def test_log_likelihood_matches_r_truncreg(fit, r_truncreg):
    assert fit["loglik"] == pytest.approx(r_truncreg["loglik"], abs=1e-10)


@pytest.mark.parametrize("name", ["se_const", "se_z1", "se_z2", "se_logsigma"])
def test_standard_errors_match_r_truncreg(fit, r_truncreg, name):
    assert fit[name] == pytest.approx(r_truncreg[name], rel=1e-7)


def test_estimates_and_standard_errors_match_the_60_digit_fit(fit, exact_mle):
    e = exact_mle["example_trunc_1"]
    assert e["n"] == fit["n"]
    for name, want in zip(NAMES, e["coef"]):
        assert fit[name] == pytest.approx(float(want), abs=1e-12)
    assert np.log(fit["sigma"]) == pytest.approx(float(e["log_sigma"]), abs=1e-12)
    for name, want in zip(["se_const", "se_z1", "se_z2", "se_logsigma"], e["se"]):
        assert fit[name] == pytest.approx(float(want), rel=1e-8)
    assert fit["loglik"] == pytest.approx(float(e["loglik"]), abs=1e-11)


def test_algorithm1_fits_the_same_sample_and_point_estimates(fit, data):
    theta, Z = data
    a = simarwilson_algorithm1(theta, Z, L=20, seed=1, names=["z1", "z2"])
    assert a["n"] == 48
    np.testing.assert_allclose(a["summary"].attrs["coef_raw"],
                               [fit["const"], fit["z1"], fit["z2"]], rtol=0, atol=1e-12)
