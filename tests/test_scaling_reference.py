"""The numerical-fault case of the paper's Section 4.3, against two references.

``tests/data/scaling_case.csv`` holds 30 synthetic units whose input ``x2``
spans 11.3 decades (``make_synthetic_data.py::make_scaling_case``).  Dividing
that column by its arithmetic mean, as v1.0.9-v1.1.0 did, puts its smallest
entry at 1.2e-10, below HiGHS's feasibility tolerance of 1e-7.

References:

* ``scaling_case_exact.csv`` - every unit's optimum in exact rational
  arithmetic (``docs/external_reference/exact_dea.py``), a reference with no
  tolerances at all;
* ``scaling_case_benchmarking.csv`` - R's Benchmarking 0.33 at its default
  settings (``docs/external_reference/reference_scaling.R``).

The exact reference is what exposed the case the first form of the v1.12 fix
missed: U20 under BCC input orientation, theta = 1 from both the scaled and
the raw solve against an optimum of 0.977706.  The third candidate of
``pleap.core.lp`` exists because of it.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pleap.core.bootstrap import _score_point
from pleap.core.dea import dea, super_efficiency
from pleap.core.lp import envelopment
from pleap.core.malmquist import _theta_against
from pleap.core.scaling import column_scales, scaling_is_risky

DATA = Path(__file__).parent / "data"
ROOT = Path(__file__).resolve().parents[1]
MODELS = [("CCR", "input"), ("CCR", "output"), ("BCC", "input"), ("BCC", "output")]


@pytest.fixture(scope="module")
def case():
    d = pd.read_csv(DATA / "scaling_case.csv", float_precision="round_trip")
    return d[["x1", "x2"]].to_numpy(float), d[["y"]].to_numpy(float)


def _ref(name, model, orient):
    df = pd.read_csv(DATA / name, float_precision="round_trip")
    sel = (df["model"] == model) & (df["orient"] == orient)
    return df.loc[sel, "efficiency"].to_numpy(float)


def test_the_case_is_flagged_risky(case):
    X, _ = case
    am, gm = X.mean(axis=0), column_scales(X)
    assert X[:, 1].min() / am[1] < 1e-7          # 1.2e-10: below the tolerance
    assert X[:, 1].min() / gm[1] > 1e-7          # 3.8e-5: above it
    assert scaling_is_risky(X, am)
    assert scaling_is_risky(X)                   # near the limits either way


@pytest.mark.parametrize("model,orient", MODELS)
def test_pleap_matches_the_exact_optimum(case, model, orient):
    X, Y = case
    got = dea(X, Y, model=model, orient=orient)["efficiency"]
    np.testing.assert_allclose(got, _ref("scaling_case_exact.csv", model, orient),
                               rtol=0, atol=1e-7)


@pytest.mark.parametrize("model,orient", MODELS)
def test_pleap_matches_benchmarking_where_benchmarking_is_exact(case, model, orient):
    """Benchmarking agrees with the exact optimum to 2.1e-9 under CRS.  Under
    VRS it is off on U24, by 6.9e-7 (input) and 2.5e-3 (output); everywhere
    else PLEAP and Benchmarking agree."""
    X, Y = case
    got = dea(X, Y, model=model, orient=orient)["efficiency"]
    ex = _ref("scaling_case_exact.csv", model, orient)
    be = _ref("scaling_case_benchmarking.csv", model, orient)
    ok = np.abs(be - ex) <= 1e-6
    assert ok.sum() >= len(ex) - 1
    np.testing.assert_allclose(got[ok], be[ok], rtol=0, atol=1e-6)


def test_arithmetic_mean_scaling_reproduces_the_fault(case):
    """One solve on arithmetic-mean-scaled data, as in v1.0.9-v1.1.0.

    With SciPy's HiGHS at the time of writing this moves 9 of the 30 CCR
    input-oriented scores, by up to 0.153, and puts U18 - on the frontier - at
    0.847.  The assertion is looser than that, so that a different HiGHS build
    does not fail it while the fault still reproduces.
    """
    X, Y = case
    ex = _ref("scaling_case_exact.csv", "CCR", "input")
    ax, ay = X.mean(axis=0), Y.mean(axis=0)
    got = np.array([envelopment(X[k] / ax, Y[k] / ay, X / ax, Y / ay,
                                "CCR", "input")[0] for k in range(len(X))])
    gap = np.abs(got - ex)
    assert (gap > 1e-4).sum() >= 3
    assert gap.max() > 0.1


@pytest.mark.parametrize("model,orient", MODELS)
def test_the_other_score_paths_share_the_rule(case, model, orient):
    """The bootstrap and Malmquist distances go through the same rule, and
    super-efficiency equals the ordinary score for every inefficient unit."""
    X, Y = case
    ex = _ref("scaling_case_exact.csv", model, orient)
    boot = np.array([_score_point(X[k], Y[k], X, Y, model, orient)
                     for k in range(len(X))])
    malm = np.array([_theta_against(X, Y, X[k], Y[k], model, orient)
                     for k in range(len(X))])
    np.testing.assert_allclose(boot, ex, rtol=0, atol=1e-7)
    np.testing.assert_allclose(malm, ex, rtol=0, atol=1e-7)
    sup = super_efficiency(X, Y, model=model, orient=orient)["score"]
    inner = ex < 1 - 1e-6
    np.testing.assert_allclose(sup[inner], ex[inner], rtol=0, atol=1e-7)


def _exact_module():
    path = ROOT / "docs" / "external_reference" / "exact_dea.py"
    spec = importlib.util.spec_from_file_location("exact_dea", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(5))
def test_random_columns_spanning_fourteen_decades(seed):
    """A slice of docs/scaling_stress.py, which runs 100 such draws."""
    exact = _exact_module()
    rng = np.random.default_rng(seed)
    n = 30
    x1 = rng.uniform(10.0, 400.0, n)
    x2 = np.exp(rng.uniform(np.log(30.0), np.log(1e15), n))
    y = (x1 ** 0.5 * x2 ** 0.1 * rng.uniform(0.4, 1.0, n)).reshape(-1, 1)
    X = np.column_stack([x1, x2])
    for model, orient in MODELS:
        ref = np.array([float(v) for v in exact.exact_efficiency(
            X.tolist(), y.tolist(), model, orient)])
        got = dea(X, y, model=model, orient=orient)["efficiency"]
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6,
                                   err_msg=f"seed {seed}, {model} {orient}")
