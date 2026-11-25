"""PLEAP's DEA scores, slacks, projections and peers against reference programs.

Two references, the same tests:

* public - R's ``Benchmarking`` package (0.33, lp_solve) on the synthetic data
  set SA (60 units) for scores, second-phase slacks, projections and peers,
  and on SD (400 units) for scores (``tests/data/sa_benchmarking_dea.csv``,
  ``sd_benchmarking_dea.csv``; ``docs/external_reference/reference_sa.R``,
  ``reference_sd.R``).  Benchmarking writes full double precision; the
  measured agreement is 1.3e-12 on scores, 3.1e-10 relative on slacks and
  projections and 9e-11 on intensity weights, with identical peer sets;
* private - MaxDEA on the validation data sets A and D, which come from the
  Korean Innovation Survey and are not redistributed (``conftest.py``); the
  tests are skipped when the files are absent.  MaxDEA prints peer weights to
  six decimals, so that comparison is to its display resolution.

The agreement is the central empirical claim of the software, so these are
golden tests: a change that moves any score breaks the build.
"""

import numpy as np
import pytest

from pleap.core.dea import dea, rts_classify

INS, OUTS = ["i1", "i2"], ["o1"]
MODELS = [("BCC", "input"), ("BCC", "output"),
          ("CCR", "input"), ("CCR", "output")]

# data fixture, reference fixture, and the resolution of the reference
CASES_A = {
    "SA vs Benchmarking": dict(data="sa", ref="sa_ref_dea", slack_rel=1e-8,
                               peer_min=1e-9, peer_abs=1e-8),
    "A vs MaxDEA (private)": dict(data="ds_a", ref="ref_dea", slack_rel=1e-6,
                                  peer_min=5e-7, peer_abs=5e-7),
}
CASES_D = {"SD vs Benchmarking": ("sd", "sd_ref_dea"),
           "D vs MaxDEA (private)": ("ds_d", "ref_d_dea")}


@pytest.fixture(scope="module", params=list(CASES_A))
def case_a(request):
    c = dict(CASES_A[request.param])
    ds = request.getfixturevalue(c["data"])
    c.update(ds=ds, ref=request.getfixturevalue(c["ref"]),
             X=ds[INS].to_numpy(float), Y=ds[OUTS].to_numpy(float))
    return c


@pytest.fixture(scope="module")
def scores(sa_xy):
    X, Y = sa_xy
    return {(m, o): dea(X, Y, model=m, orient=o)["efficiency"]
            for m, o in MODELS}


@pytest.mark.parametrize("model,orient", MODELS)
def test_scores_match_the_reference(case_a, model, orient):
    ds, ref = case_a["ds"], case_a["ref"]
    e = dea(case_a["X"], case_a["Y"], model=model, orient=orient)["efficiency"]
    ref = ref[(ref.model == model) & (ref.orient == orient)].set_index("dmu")["score"]
    got = dict(zip(ds["dmu"], e))
    assert set(ref.index) == set(got)
    diff = np.array([abs(got[d] - ref[d]) for d in ref.index])
    # Both references carry full double precision; the measured gap is ~1e-12.
    assert diff.max() < 1.0e-9, (
        f"{model}/{orient}: max |PLEAP - reference| = {diff.max():.3e} "
        f"at {ref.index[int(diff.argmax())]}")


@pytest.mark.parametrize("model,orient", MODELS)
def test_scores_are_in_the_unit_interval(scores, model, orient):
    e = scores[(model, orient)]
    assert np.all(np.isfinite(e))
    assert np.all(e > 0.0) and np.all(e <= 1.0 + 1e-9)


def test_ccr_never_exceeds_bcc(scores):
    """CRS is a restriction of VRS, so the CRS score can only be smaller."""
    for orient in ("input", "output"):
        assert np.all(scores[("CCR", orient)] <= scores[("BCC", orient)] + 1e-9)


def test_at_least_one_dmu_is_efficient(scores):
    for key, e in scores.items():
        assert np.isclose(e.max(), 1.0, atol=1e-7), key


def test_rts_classification_is_consistent(sa_xy, scores):
    labels, ccr, bcc = rts_classify(*sa_xy, orient="input")
    assert set(labels) <= {"CRS", "IRS", "DRS", "NA"}
    np.testing.assert_allclose(ccr, scores[("CCR", "input")], atol=1e-7)
    np.testing.assert_allclose(bcc, scores[("BCC", "input")], atol=1e-7)
    # scale efficiency 1 <=> CRS
    for lab, c, b in zip(labels, ccr, bcc):
        if lab == "CRS":
            assert c / b == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("model,orient", MODELS)
def test_slacks_and_projections_match_the_reference(case_a, model, orient):
    """Second-phase slacks and the projected point, not only the score.

    Both references maximise the plain sum of the slacks in the second phase.
    MaxDEA writes input slacks as (negative) movements; its fixture stores them
    as the non-negative slacks PLEAP reports.  Against Benchmarking the
    measured agreement is 3.1e-10 relative (23 of SA's 60 units carry a slack
    under BCC input orientation); against MaxDEA 7e-11, 3e-8 for one output
    slack under CCR output orientation."""
    X, Y, ds = case_a["X"], case_a["Y"], case_a["ds"]
    r = dea(X, Y, model=model, orient=orient)
    eff, sm, sp = r["efficiency"], r["slack_input"], r["slack_output"]
    if orient == "input":
        px, py = eff[:, None] * X - sm, Y + sp
    else:
        px, py = X - sm, Y / eff[:, None] + sp
    ref = case_a["ref"]
    ref = ref[(ref.model == model) & (ref.orient == orient)].set_index("dmu").loc[ds["dmu"]]
    got = {"slack_i1": sm[:, 0], "slack_i2": sm[:, 1], "slack_o1": sp[:, 0],
           "proj_i1": px[:, 0], "proj_i2": px[:, 1], "proj_o1": py[:, 0]}
    for col, v in got.items():
        want = ref[col].to_numpy(float)
        bad = np.abs(v - want) > case_a["slack_rel"] * np.maximum(1.0, np.abs(want))
        assert not bad.any(), f"{model}/{orient} {col}: {list(ref.index[bad])}"


@pytest.mark.parametrize("model,orient", MODELS)
def test_peers_and_weights_match_the_reference(case_a, model, orient):
    """Same reference set for every DMU, and the same intensity weights (to
    MaxDEA's six printed decimals; to 1e-8 against Benchmarking)."""
    lam = dea(case_a["X"], case_a["Y"], model=model, orient=orient)["lambdas"]
    dmus = list(case_a["ds"]["dmu"])
    ref = case_a["ref"]
    ref = ref[(ref.model == model) & (ref.orient == orient)].set_index("dmu")
    for k, d in enumerate(dmus):
        want = {}
        if isinstance(ref.loc[d, "peers"], str) and ref.loc[d, "peers"]:
            for item in ref.loc[d, "peers"].split(";"):
                name, val = item.split(":")
                want[name] = float(val)
        got = {dmus[j]: lam[k, j] for j in np.flatnonzero(lam[k] > case_a["peer_min"])}
        assert set(got) == set(want), f"{model}/{orient} {d}"
        for p in want:
            assert got[p] == pytest.approx(want[p], abs=case_a["peer_abs"])


@pytest.mark.parametrize("case", list(CASES_D))
@pytest.mark.parametrize("model,orient", MODELS)
def test_scores_match_the_reference_on_the_large_data_set(request, case, model, orient):
    """400 synthetic units against Benchmarking; 453 firms against MaxDEA."""
    data, refname = CASES_D[case]
    ds, ref = request.getfixturevalue(data), request.getfixturevalue(refname)
    e = dea(ds[INS].to_numpy(float), ds[OUTS].to_numpy(float), model=model, orient=orient)["efficiency"]
    ref = ref[(ref.model == model) & (ref.orient == orient)]
    want = ref.set_index("dmu").loc[ds["dmu"], "score"].to_numpy(float)
    assert np.max(np.abs(e - want)) < 1.0e-9


# --------------------------------------------------------------------------- #
#  Analytic cases                                                             #
# --------------------------------------------------------------------------- #
def test_one_input_one_output_crs_has_a_closed_form(one_in_one_out):
    X, Y, expected = one_in_one_out
    got = dea(X, Y, model="CCR", orient="input")["efficiency"]
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_output_orientation_equals_input_orientation_under_crs(one_in_one_out):
    """A property of CRS technologies only."""
    X, Y, _ = one_in_one_out
    a = dea(X, Y, model="CCR", orient="input")["efficiency"]
    b = dea(X, Y, model="CCR", orient="output")["efficiency"]
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_vrs_frontier_endpoints_are_efficient(one_in_one_out):
    """Under VRS the smallest and the largest DMU are always on the frontier."""
    X, Y, _ = one_in_one_out
    e = dea(X, Y, model="BCC", orient="input")["efficiency"]
    assert e[0] == pytest.approx(1.0, abs=1e-9)    # smallest input
    assert e[-1] == pytest.approx(1.0, abs=1e-9)   # largest output


def test_unit_output_example(two_in_one_out):
    """C(8,1), D(4,2) and E(2,4) span the frontier; A, B and G are interior."""
    X, Y = two_in_one_out
    e = dea(X, Y, model="CCR", orient="input")["efficiency"]
    for k in (2, 3, 4):
        assert e[k] == pytest.approx(1.0, abs=1e-9)
    for k in (0, 1, 6):
        assert e[k] < 1.0 - 1e-9


def test_weak_efficiency_is_separated_from_strong_efficiency(two_in_one_out):
    """F(10,1) is radially efficient but Koopmans-inefficient.

    Its second input already equals the frontier minimum, so no radial
    contraction is possible (theta = 1), yet it uses two units more of the
    first input than C(8,1).  Phase 2 has to expose that as a slack - this is
    precisely what distinguishes the two-phase procedure from a bare radial
    solve, and what the Visualization tab draws as the non-radial step.
    """
    X, Y = two_in_one_out
    r = dea(X, Y, model="CCR", orient="input")
    assert r["efficiency"][5] == pytest.approx(1.0, abs=1e-9)
    assert r["slack_input"][5, 0] == pytest.approx(2.0, abs=1e-6)
    assert r["slack_input"][5, 1] == pytest.approx(0.0, abs=1e-6)
    # a strongly efficient DMU has no slack at all
    assert np.allclose(r["slack_input"][2], 0.0, atol=1e-6)


# --------------------------------------------------------------------------- #
#  Projections and slacks                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model,orient", MODELS)
def test_projection_lands_on_the_frontier(sa_xy, model, orient):
    """x_hat = theta*x - s^-, y_hat = y + s^+ must itself be efficient."""
    X, Y = sa_xy
    r = dea(X, Y, model=model, orient=orient)
    th = r["radial"]
    xh = (th[:, None] if orient == "input" else 1.0) * X - r["slack_input"]
    yh = (th[:, None] if orient == "output" else 1.0) * Y + r["slack_output"]
    assert np.all(xh > 0) and np.all(yh > 0)
    # the projection is weakly dominated by the convex combination of peers
    lam = r["lambdas"]
    np.testing.assert_allclose(lam @ X, xh, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(lam @ Y, yh, rtol=1e-6, atol=1e-6)


def test_efficient_dmus_are_their_own_peer(sa_xy):
    X, Y = sa_xy
    r = dea(X, Y, model="BCC", orient="input")
    for k in np.flatnonzero(r["efficiency"] > 1 - 1e-9):
        assert r["slack_input"][k] == pytest.approx(0.0, abs=1e-6)
        assert r["slack_output"][k] == pytest.approx(0.0, abs=1e-6)


def test_vrs_lambdas_sum_to_one(sa_xy):
    X, Y = sa_xy
    for orient in ("input", "output"):
        r = dea(X, Y, model="BCC", orient=orient)
        np.testing.assert_allclose(r["lambdas"].sum(axis=1), 1.0, atol=1e-7)


def test_no_negative_slacks_or_lambdas(sa_xy):
    """Slacks and intensity weights are bounded at zero by construction.

    A solver can return -1e-17 for a variable that is exactly zero, which the
    results table then prints as "-0.000000" - noise that looks like a defect
    to anyone copying the table into a paper.
    """
    X, Y = sa_xy
    for model in ("CCR", "BCC"):
        for orient in ("input", "output"):
            r = dea(X, Y, model=model, orient=orient)
            for key in ("lambdas", "slack_input", "slack_output"):
                v = r[key]
                assert np.all(v >= 0.0), (
                    f"{model}/{orient} {key}: min {v.min():.3e}")
