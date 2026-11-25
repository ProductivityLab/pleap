"""Unit invariance of the DEA linear programs.

Radial DEA scores, Malmquist indices and bootstrap statistics are invariant to
the unit of measurement of each input and output.  The solver is not, so the
code rescales the columns before solving (``pleap.core.scaling``).

These tests exist because the arithmetic-mean scaling used in v1.0.9-v1.1.0
silently broke that invariance: on a column spanning many orders of magnitude
the mean sits near the top of the range, the smallest observations scale below
the solver's feasibility tolerance, and the corresponding constraint stops
binding.  Any future change to the scaling has to keep every assertion in this
file passing.  The public case of the paper, with its exact and external
references, is in tests/test_scaling_reference.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pleap.core.dea import dea
from pleap.core.malmquist import malmquist
from pleap.core.scaling import column_scales, feasible


# --------------------------------------------------------------------------- #
#  column_scales                                                              #
# --------------------------------------------------------------------------- #
def test_column_scales_is_geometric_mean():
    A = np.array([[1.0, 10.0], [100.0, 1000.0]])
    got = column_scales(A)
    assert got[0] == pytest.approx(10.0)              # sqrt(1 * 100)
    assert got[1] == pytest.approx(100.0)             # sqrt(10 * 1000)


def _skewed_column():
    """Input x2 of tests/data/scaling_case.csv: 30 values over 11.3 decades."""
    d = pd.read_csv(Path(__file__).parent / "data" / "scaling_case.csv",
                    float_precision="round_trip")
    return d[["x2"]].to_numpy(float)


def test_column_scales_is_not_dragged_by_the_upper_tail():
    """The failure mode of arithmetic-mean scaling, stated as a property."""
    A = _skewed_column()
    gm = column_scales(A)[0]
    # The arithmetic mean puts the smallest observation below 1e-7, where the
    # solver's feasibility tolerance swallows it.  The geometric mean does not.
    assert A.min() / A.mean() < 1e-7
    assert A.min() / gm > 1e-7


@pytest.mark.parametrize("bad", [
    np.zeros((4, 2)),
    np.array([[0.0, 1.0], [0.0, 2.0]]),
    np.array([[-1.0, 1.0], [-2.0, 2.0]]),
])
def test_column_scales_always_returns_usable_positive_scales(bad):
    s = column_scales(bad)
    assert s.shape == (bad.shape[1],)
    assert np.all(np.isfinite(s)) and np.all(s > 0)


def test_column_scales_rejects_non_2d():
    with pytest.raises(ValueError):
        column_scales(np.arange(5.0))


# --------------------------------------------------------------------------- #
#  Invariance of the estimator                                                #
# --------------------------------------------------------------------------- #
SCALE_SETS = [
    np.array([1.0, 1.0]),
    np.array([1e3, 1e-3]),
    np.array([1e6, 1e6]),
    np.array([1e-4, 1e9]),
]


@pytest.mark.parametrize("model", ["CCR", "BCC"])
@pytest.mark.parametrize("orient", ["input", "output"])
@pytest.mark.parametrize("cx", SCALE_SETS)
def test_dea_scores_are_unit_invariant(sa_xy, model, orient, cx):
    X, Y = sa_xy
    base = dea(X, Y, model=model, orient=orient)["efficiency"]
    other = dea(X * cx, Y * 7.3e5, model=model, orient=orient)["efficiency"]
    np.testing.assert_allclose(other, base, rtol=0, atol=1e-9)


@pytest.mark.parametrize("model", ["CCR", "BCC"])
def test_malmquist_is_unit_invariant(sa_panel, model):
    ins, outs = ["i1", "i2"], ["o1"]
    base = malmquist(sa_panel, "dmu", "period", ins, outs, model=model)
    scaled = sa_panel.copy()
    scaled["i1"] *= 1e-5
    scaled["i2"] *= 4.2e7
    scaled["o1"] *= 3.3e4
    got = malmquist(scaled, "dmu", "period", ins, outs, model=model)
    for col in ["MI(t-1, t)", "EC(t-1, t)", "TC(t-1, t)"]:
        np.testing.assert_allclose(got[col].to_numpy(float),
                                   base[col].to_numpy(float),
                                   rtol=0, atol=1e-8)


@pytest.mark.parametrize("model", ["CCR", "BCC"])
@pytest.mark.parametrize("orient", ["input", "output"])
def test_dea_survives_a_column_spanning_twelve_orders_of_magnitude(model, orient):
    """The regression case: one input column with an extreme spread.

    Scores must be identical to the same technology expressed in different
    units.  Arithmetic-mean scaling failed this test.
    """
    rng = np.random.default_rng(7)
    n = 40
    x1 = rng.uniform(10.0, 400.0, n)
    x2 = np.exp(rng.uniform(np.log(30.0), np.log(1e13), n))   # 12 decades
    y = x1 ** 0.5 * x2 ** 0.1 * rng.uniform(0.4, 1.0, n)
    X = np.column_stack([x1, x2]); Y = y.reshape(-1, 1)

    base = dea(X, Y, model=model, orient=orient)["efficiency"]
    assert np.all(np.isfinite(base))
    assert np.all(base <= 1.0 + 1e-9) and np.all(base > 0.0)
    for cx in [np.array([1.0, 1e-9]), np.array([1e5, 1e4]), np.array([3.7, 0.019])]:
        got = dea(X * cx, Y, model=model, orient=orient)["efficiency"]
        np.testing.assert_allclose(got, base, rtol=0, atol=1e-7)


def test_scores_agree_with_the_unscaled_problem(sa_xy):
    """Scaling is a conditioning device, not a modelling choice."""
    from pleap.core.dea import _solve_one
    X, Y = sa_xy
    for model in ("CCR", "BCC"):
        scaled = dea(X, Y, model=model, orient="input")["efficiency"]
        raw = np.array([_solve_one(k, X, Y, model, "input")[0]
                        for k in range(len(X))])
        np.testing.assert_allclose(scaled, raw, rtol=0, atol=1e-8)


# --------------------------------------------------------------------------- #
#  The feasibility audit                                                      #
# --------------------------------------------------------------------------- #
def test_feasible_accepts_a_genuine_solution(sa_xy):
    X, Y = sa_xy
    r = dea(X, Y, model="BCC", orient="input")
    for k in range(len(X)):
        assert feasible(X[k], Y[k], X, Y, r["lambdas"][k],
                        r["radial"][k], "BCC", "input"), f"DMU {k}"


def test_feasible_rejects_an_understated_score(sa_xy):
    X, Y = sa_xy
    r = dea(X, Y, model="BCC", orient="input")
    k = int(np.argmax(r["radial"]))
    assert not feasible(X[k], Y[k], X, Y, r["lambdas"][k],
                        r["radial"][k] * 0.5, "BCC", "input")


def test_feasible_rejects_broken_lambdas(sa_xy):
    X, Y = sa_xy
    n = len(X)
    assert not feasible(X[0], Y[0], X, Y, np.full(n, np.nan), 1.0, "BCC", "input")
    bad = np.zeros(n); bad[0] = -1.0
    assert not feasible(X[0], Y[0], X, Y, bad, 1.0, "BCC", "input")
    # VRS convexity violated
    assert not feasible(X[0], Y[0], X, Y, np.zeros(n), 1.0, "BCC", "input")


# --------------------------------------------------------------------------- #
#  Feasibility is not optimality                                              #
# --------------------------------------------------------------------------- #
def test_a_primal_feasibility_check_cannot_catch_the_scaling_fault():
    """States the limitation that motivated the two-solve rule.

    Under input orientation ``theta = 1`` with the DMU as its own and only
    peer satisfies every constraint exactly.  That is the shape the
    arithmetic-mean scaling failure took, so checking the constraints of a
    single returned solution cannot detect it - which is why the decision is
    made from the data instead, by ``scaling_is_risky``.

    This is a statement about PRIMAL feasibility only.  A dual test does
    detect the fault; see the companion test below.
    """
    rng = np.random.default_rng(5)
    n = 25
    X = rng.uniform(1.0, 50.0, (n, 2))
    Y = rng.uniform(1.0, 20.0, (n, 1))
    k = 0
    lam = np.zeros(n); lam[k] = 1.0          # the DMU as its own peer
    assert feasible(X[k], Y[k], X, Y, lam, 1.0, "BCC", "input")
    assert feasible(X[k], Y[k], X, Y, lam, 1.0, "CCR", "input")


def test_a_dual_check_does_detect_the_scaling_fault():
    """The limitation above is about primal feasibility, and only that.

    An earlier draft of the paper claimed that no post-hoc test of a returned
    solution could separate feasibility from optimality.  That is too strong:
    linear programming supplies optimality certificates.  Mapping the dual the
    solver returned for the rescaled problem back into the original units and
    checking the reduced costs shows a clear violation.

    The test exists so the narrower, true claim stays true, and so nobody
    re-broadens it.  The two-solve rule is preferred because it repairs rather
    than only diagnoses, not because a dual test is impossible.
    """
    from scipy.optimize import linprog

    rng = np.random.default_rng(11)
    n = 30
    x1 = rng.uniform(10.0, 400.0, n)
    x2 = np.exp(rng.uniform(np.log(30.0), np.log(1e13), n))
    Y = (x1 ** 0.5 * x2 ** 0.1 * rng.uniform(0.4, 1.0, n)).reshape(-1, 1)
    X = np.column_stack([x1, x2])

    def build(k, X, Y):
        n, m = X.shape
        _, s = Y.shape
        nv = 1 + n
        A = np.zeros((m + s, nv)); b = np.zeros(m + s)
        c = np.zeros(nv); c[0] = 1.0
        for i in range(m):
            A[i, 0] = -X[k, i]; A[i, 1:] = X[:, i]
        for r in range(s):
            A[m + r, 1:] = -Y[:, r]; b[m + r] = -Y[k, r]
        return A, b, c

    am = X.mean(axis=0)                                   # the faulty scaler
    gm = column_scales(X)

    bad = None
    for k in range(n):
        Aa, ba, c = build(k, X / am, Y)
        ra = linprog(c, A_ub=Aa, b_ub=ba, bounds=[(0, None)] * (n + 1),
                     method="highs")
        Ag, bg, c = build(k, X / gm, Y)
        rg = linprog(c, A_ub=Ag, b_ub=bg, bounds=[(0, None)] * (n + 1),
                     method="highs")
        if ra.success and rg.success and abs(ra.x[0] - rg.x[0]) > 1e-4:
            bad = (k, ra, rg)
            break
    assert bad is not None, "the fault must still reproduce"
    k, ra, rg = bad

    Ao, bo, co = build(k, X, Y)
    xo = np.concatenate([[ra.x[0]], np.clip(ra.x[1:], 0.0, None)])

    # Primal feasibility in the original units: passes, catching nothing.
    assert (Ao @ xo - bo).max() <= 1e-6

    # The same solution's dual, mapped back: row i was divided by am[i].
    y = np.asarray(ra.ineqlin.marginals, float).copy()
    y[:X.shape[1]] /= am
    reduced = co - Ao.T @ y
    assert reduced.min() < -1e-6, (
        "a dual check must detect what the primal check missed; "
        f"min reduced cost {reduced.min():.3e}")


def test_scaling_is_risky_flags_the_pathological_case():
    from pleap.core.scaling import scaling_is_risky
    A = _skewed_column()
    assert scaling_is_risky(A, A.mean(axis=0)), "arithmetic mean must be flagged"
    assert scaling_is_risky(A), "this column is near the solver's limits either way"


def test_scaling_is_risky_leaves_ordinary_data_alone(sa_xy):
    from pleap.core.scaling import scaling_is_risky
    X, Y = sa_xy
    assert not scaling_is_risky(X)
    assert not scaling_is_risky(Y)


def test_scores_survive_a_deliberately_bad_scaling(monkeypatch):
    """The two-solve rule must rescue the answer even under the old bug.

    This is the guarantee that replaced the post-solve audit: with the
    arithmetic mean forced back in, the cross-solve still returns the right
    score, because between two feasible points the better objective wins.
    """
    import pleap.core.dea as D
    rng = np.random.default_rng(11)
    n = 30
    x1 = rng.uniform(10.0, 400.0, n)
    x2 = np.exp(rng.uniform(np.log(30.0), np.log(1e13), n))
    y = (x1 ** 0.5 * x2 ** 0.1 * rng.uniform(0.4, 1.0, n)).reshape(-1, 1)
    X = np.column_stack([x1, x2])

    good = D.dea(X, y, model="CCR", orient="input")["efficiency"]
    monkeypatch.setattr(D, "column_scales",
                        lambda A: np.asarray(A, float).mean(axis=0))
    bad = D.dea(X, y, model="CCR", orient="input")["efficiency"]
    np.testing.assert_allclose(bad, good, rtol=0, atol=1e-6)
