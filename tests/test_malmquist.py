"""Malmquist productivity index against reference programs, plus identities.

References:

* public - R's ``Benchmarking::malmq`` on the synthetic panel SA (60 units x
  three periods; ``tests/data/sa_benchmarking_mpi.csv``,
  ``docs/external_reference/reference_sa.R``).  The file holds the four
  distance functions, so the index is formed here under either orientation;
  under VRS Benchmarking returns an infinite distance where a cross-period
  program is infeasible, which pins both of PLEAP's conventions: ``omit``
  must leave exactly those rows missing, and ``substitute`` must equal the
  index with each such distance set to 1, MaxDEA's convention.  Measured
  agreement: 7e-12 on every index, all eight model x orientation x
  convention cases.
* private - MaxDEA on the validation panel A (VRS, input orientation, MaxDEA's
  default and therefore PLEAP's), skipped when the data are absent
  (conftest.py).
"""

import numpy as np
import pytest
import pandas as pd

from pleap.core.malmquist import malmquist

INS, OUTS = ["i1", "i2"], ["o1"]
COLS = ["MI(t-1, t)", "EC(t-1, t)", "TC(t-1, t)"]
MODELS = [("CCR", "input"), ("CCR", "output"), ("BCC", "input"), ("BCC", "output")]


@pytest.fixture(scope="module")
def mpi(sa_panel):
    return malmquist(sa_panel, "dmu", "period", INS, OUTS,
                     model="BCC", orient="input")


def test_shape_and_periods(sa_panel, mpi):
    periods = sorted(sa_panel["period"].unique())
    n_dmu = sa_panel["dmu"].nunique()
    assert len(mpi) == n_dmu * (len(periods) - 1)
    assert sorted(mpi["t(Period)"].unique()) == periods[1:]
    assert list(mpi.columns)[:5] == ["t(Period)", "DMU"] + COLS
    # the substitution column appears only when a substitution happened
    if "Infeasible_substitutions" in mpi.columns:
        assert mpi.attrs["infeasible_substitutions"] > 0
    else:
        assert mpi.attrs["infeasible_substitutions"] == 0


def _benchmarking_index(ref, orient, substitute):
    """The Fare-Grosskopf-Norris-Zhang index from Benchmarking's Farrell
    efficiencies.  In Benchmarking's naming e10 is the later period's data
    against the earlier period's technology, e01 the reverse, and e00 and e11
    the own-period efficiencies."""
    e = {c: ref[c].to_numpy(float).copy() for c in ("e00", "e01", "e10", "e11")}
    if substitute:
        for v in e.values():
            v[~np.isfinite(v)] = 1.0
    with np.errstate(invalid="ignore", divide="ignore"):   # infinite rows: not compared
        if orient == "input":       # Farrell input efficiencies, <= 1 on the own frontier
            mi = np.sqrt(e["e10"] / e["e00"] * e["e11"] / e["e01"])
            ec = e["e11"] / e["e00"]
        else:                       # Farrell output efficiencies, >= 1
            mi = np.sqrt(e["e00"] / e["e10"] * e["e01"] / e["e11"])
            ec = e["e00"] / e["e11"]
        return mi, ec, mi / ec


@pytest.mark.parametrize("mode", ["omit", "substitute"])
@pytest.mark.parametrize("model,orient", MODELS)
def test_matches_benchmarking(sa_panel, sa_ref_mpi, model, orient, mode):
    ref = sa_ref_mpi[(sa_ref_mpi.model == model) & (sa_ref_mpi.orient == orient)]
    got = malmquist(sa_panel, "dmu", "period", INS, OUTS, model=model, orient=orient,
                    on_infeasible=mode)
    j = ref.merge(got, left_on=["period", "dmu"], right_on=["t(Period)", "DMU"], how="left")
    assert len(j) == len(got) == 120
    infeasible = ~np.isfinite(j[["e00", "e01", "e10", "e11"]].to_numpy(float)).all(axis=1)
    assert got.attrs["infeasible_substitutions"] == int(
        (~np.isfinite(j[["e00", "e01", "e10", "e11"]].to_numpy(float))).sum())
    if model == "CCR":
        assert not infeasible.any()
    else:                       # SA has 3 such rows under input, 11 under output orientation
        assert infeasible.sum() == {"input": 3, "output": 11}[orient]
    mi, ec, tc = _benchmarking_index(j, orient, substitute=(mode == "substitute"))
    rows = ~infeasible if mode == "omit" else np.ones(len(j), bool)
    for want, col in zip((mi, ec, tc), COLS):
        d = np.abs(j[col].to_numpy(float)[rows] - want[rows])
        assert d.max() < 1e-9, f"{col}: max |PLEAP - Benchmarking| = {d.max():.3e}"
    if mode == "omit":
        assert (j["MI(t-1, t)"].isna().to_numpy() == infeasible).all()


def test_matches_maxdea(ds_a_panel, ref_mpi):
    """Private: validation panel A against MaxDEA, VRS input orientation."""
    got = malmquist(ds_a_panel, "dmu", "period", INS, OUTS, model="BCC", orient="input")
    got["DMU"] = got["DMU"].astype(str)
    got["t(Period)"] = got["t(Period)"].astype(int)
    ref = ref_mpi.copy()
    ref["dmu"] = ref["dmu"].astype(str)
    j = ref.merge(got, left_on=["period", "dmu"],
                  right_on=["t(Period)", "DMU"], how="inner")
    assert len(j) == len(ref), "reference rows missing from the PLEAP output"
    for short, col in zip(["MI", "EC", "TC"], COLS):
        d = (j[short] - j[col]).abs()
        assert d.max() < 5.0e-7, f"{short}: max |PLEAP - MaxDEA| = {d.max():.3e}"


def test_decomposition_is_exact(mpi):
    """MI = EC x TC holds identically, not approximately."""
    np.testing.assert_allclose(mpi["MI(t-1, t)"],
                               mpi["EC(t-1, t)"] * mpi["TC(t-1, t)"],
                               rtol=1e-12, atol=0)


def test_all_components_are_finite_and_positive(mpi):
    for c in COLS:
        v = mpi[c].to_numpy(float)
        assert np.all(np.isfinite(v)), f"{c} has missing values"
        assert np.all(v > 0), f"{c} has non-positive values"


@pytest.mark.parametrize("orient", ["input", "output"])
def test_crs_indices_do_not_depend_on_orientation(sa_panel, orient):
    """Under CRS the input- and output-oriented Malmquist indices coincide.

    This is also how validation data set B was reproduced: its MaxDEA sheet
    had been run under CRS, where both orientations agree.
    """
    a = malmquist(sa_panel, "dmu", "period", INS, OUTS,
                  model="CCR", orient="input")
    b = malmquist(sa_panel, "dmu", "period", INS, OUTS,
                  model="CCR", orient=orient)
    for c in COLS:
        np.testing.assert_allclose(b[c], a[c], rtol=1e-9, atol=1e-9)


def test_no_change_panel_gives_unit_indices(sa_panel):
    """A panel that repeats itself has MI = EC = TC = 1 for every DMU."""
    one = sa_panel[sa_panel["period"] == sa_panel["period"].min()].copy()
    frames = []
    for t in (1, 2, 3):
        f = one.copy(); f["period"] = t
        frames.append(f)
    import pandas as pd
    panel = pd.concat(frames, ignore_index=True)
    r = malmquist(panel, "dmu", "period", INS, OUTS, model="BCC", orient="input")
    for c in COLS:
        np.testing.assert_allclose(r[c].to_numpy(float), 1.0, atol=1e-9)


def test_uniform_output_growth_is_pure_technical_change(sa_panel):
    """Scaling every output by g between periods shifts the frontier only.

    Efficiency change must stay at 1 and the Malmquist index must equal the
    growth factor for every DMU.
    """
    import pandas as pd
    g = 1.25
    one = sa_panel[sa_panel["period"] == sa_panel["period"].min()].copy()
    a = one.copy(); a["period"] = 1
    b = one.copy(); b["period"] = 2; b["o1"] = b["o1"] * g
    panel = pd.concat([a, b], ignore_index=True)
    r = malmquist(panel, "dmu", "period", INS, OUTS, model="CCR", orient="input")
    np.testing.assert_allclose(r["EC(t-1, t)"].to_numpy(float), 1.0, atol=1e-7)
    np.testing.assert_allclose(r["MI(t-1, t)"].to_numpy(float), g, atol=1e-7)
    np.testing.assert_allclose(r["TC(t-1, t)"].to_numpy(float), g, atol=1e-7)



def test_infeasible_substitutions_are_counted_and_disclosed(sa_panel):
    """Replacing an infeasible cross-period distance by 1 must be visible.

    An infeasible distance and a distance of 1 are different things, and the
    substitution propagates into MI and TC.  A reader cannot tell from the
    index that it happened, so the count is reported and the column appears
    whenever it is non-zero.
    """
    r = malmquist(sa_panel, "dmu", "period", INS, OUTS,
                  model="BCC", orient="input")
    n = r.attrs["infeasible_substitutions"]
    assert isinstance(n, int) and n >= 0
    if n:
        assert "Infeasible_substitutions" in r.columns
        assert r["Infeasible_substitutions"].sum() == n
        assert r.attrs["rows_affected"] == int(
            (r["Infeasible_substitutions"] > 0).sum())
        assert "MaxDEA convention" in r.attrs["substitution_note"]
    else:
        assert "Infeasible_substitutions" not in r.columns
        assert r.attrs["substitution_note"] == ""


def test_crs_never_needs_a_substitution(sa_panel):
    """Under CRS the cross-period programs are always feasible."""
    r = malmquist(sa_panel, "dmu", "period", INS, OUTS,
                  model="CCR", orient="input")
    assert r.attrs["infeasible_substitutions"] == 0
    assert "Infeasible_substitutions" not in r.columns


def test_omit_mode_reports_nan_instead_of_substituting(sa_panel):
    """``on_infeasible='omit'`` must not invent a distance.

    The default substitutes 1 for MaxDEA compatibility.  That convention can
    move a reported average - on validation panel A three substituted rows
    lower a transition's mean productivity growth from 9.8% to 4.4%, and on
    the example used up to v1.11 they reversed its sign - so the
    alternative has to be real, and has to leave the unaffected rows
    untouched.
    """
    sub = malmquist(sa_panel, "dmu", "period", INS, OUTS,
                    model="BCC", orient="input", on_infeasible="substitute")
    omit = malmquist(sa_panel, "dmu", "period", INS, OUTS,
                     model="BCC", orient="input", on_infeasible="omit")

    assert omit.attrs["on_infeasible"] == "omit"
    assert omit.attrs["infeasible_substitutions"] == \
        sub.attrs["infeasible_substitutions"]

    n = sub.attrs["infeasible_substitutions"]
    if n:
        bad = sub["Infeasible_substitutions"] > 0
        assert omit.loc[bad, "MI(t-1, t)"].isna().all(), \
            "affected rows must be NaN, not a substituted index"
        np.testing.assert_allclose(
            omit.loc[~bad, "MI(t-1, t)"].to_numpy(float),
            sub.loc[~bad, "MI(t-1, t)"].to_numpy(float), rtol=0, atol=1e-12)
    else:
        pd.testing.assert_frame_equal(sub, omit)


def test_the_cross_period_denominator_is_two_per_row(sa_panel):
    """The rate must be quoted against cross-period distances, not all four.

    Four distance functions are computed per unit-period row, but only two of
    them are cross-period; the own-period pair is always feasible.  An earlier
    manuscript draft quoted the substitutions against 480 rather than 240 and
    so understated the rate by half.
    """
    r = malmquist(sa_panel, "dmu", "period", INS, OUTS,
                  model="BCC", orient="input")
    assert r.attrs["cross_period_distances"] == 2 * len(r)
    assert r.attrs["infeasible_substitutions"] <= r.attrs["cross_period_distances"]
    if r.attrs["infeasible_substitutions"]:
        assert f"of {2 * len(r)} cross-period" in r.attrs["substitution_note"]


def test_on_infeasible_rejects_an_unknown_mode(sa_panel):
    with pytest.raises(ValueError):
        malmquist(sa_panel, "dmu", "period", INS, OUTS,
                  on_infeasible="pretend_it_is_fine")


def test_scale_decomposition_splits_ec_exactly(sa_panel):
    """EC = PEC x SEC must hold identically, like MI = EC x TC.

    PEC comes from the VRS own-period distances and SEC is the residual, so
    the identity holds by construction.  Asserting it guards against a future
    change that computes SEC separately and lets the two drift.
    """
    r = malmquist(sa_panel, "dmu", "period", INS, OUTS, model="CCR",
                  orient="input", scale_decomposition=True)
    for c in ("PEC(t-1, t)", "SEC(t-1, t)"):
        assert c in r.columns
    ec = r["EC(t-1, t)"].to_numpy(float)
    pec = r["PEC(t-1, t)"].to_numpy(float)
    sec = r["SEC(t-1, t)"].to_numpy(float)
    np.testing.assert_allclose(ec, pec * sec, rtol=0, atol=1e-12)
    # and the outer identity still holds alongside it
    np.testing.assert_allclose(r["MI(t-1, t)"].to_numpy(float),
                               ec * r["TC(t-1, t)"].to_numpy(float),
                               rtol=0, atol=1e-12)


def test_pec_equals_the_vrs_efficiency_change(sa_panel):
    """PEC must be the VRS efficiency change, not a rescaling of the CRS one."""
    crs = malmquist(sa_panel, "dmu", "period", INS, OUTS, model="CCR",
                    orient="input", scale_decomposition=True)
    vrs = malmquist(sa_panel, "dmu", "period", INS, OUTS, model="BCC",
                    orient="input")
    key = ["t(Period)", "DMU"]
    merged = crs[key + ["PEC(t-1, t)"]].merge(
        vrs[key + ["EC(t-1, t)"]], on=key, how="inner")
    assert len(merged) == len(crs)
    np.testing.assert_allclose(merged["PEC(t-1, t)"].to_numpy(float),
                               merged["EC(t-1, t)"].to_numpy(float),
                               rtol=0, atol=1e-9)


def test_scale_decomposition_is_refused_under_vrs(sa_panel):
    """Under VRS the efficiency change is already pure; SEC would be noise."""
    with pytest.raises(ValueError, match="CRS"):
        malmquist(sa_panel, "dmu", "period", INS, OUTS, model="BCC",
                  scale_decomposition=True)


def test_the_columns_appear_only_when_asked_for(sa_panel):
    """A CRS run without the flag must not carry two empty columns.

    The column list used to add PEC and SEC for every CRS run while nothing
    filled them, so the frame came back with two all-NaN columns.
    """
    r = malmquist(sa_panel, "dmu", "period", INS, OUTS, model="CCR",
                  orient="input")
    assert "PEC(t-1, t)" not in r.columns
    assert "SEC(t-1, t)" not in r.columns
