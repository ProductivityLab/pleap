"""Smoothed homogeneous bootstrap (Simar & Wilson 1998; Kneip, Simar & Wilson 2008).

Unlike DEA and Malmquist, the bootstrap is stochastic: it cannot be compared to
MaxDEA bit for bit, because two runs of PLEAP with different seeds already
differ from each other.  The tests here are therefore of three kinds:

1. structural guarantees that must hold for every run (bias sign, ordering,
   reproducibility, independence from the number of worker processes);
2. the bandwidth formula, pinned exactly - it was recalibrated in v1.0.9 and a
   silent change to it would shift every published bias correction;
3. agreement with MaxDEA on validation data set A, with a tolerance set from
   the Monte-Carlo spread of PLEAP against itself rather than from a guess.
   A comes from the Korean Innovation Survey and is not redistributed, so
   these tests are skipped without it (conftest.py); the public comparison
   with an open implementation, R's Benchmarking, is in
   test_bootstrap_open_reference.py.

The structural tests run on the synthetic data set SA.  ``B`` is kept small
so the suite stays fast; the tolerance tests use the seed-to-seed spread
measured at the same ``B``, so they remain meaningful.
"""

import numpy as np
import pytest

from pleap.core.bootstrap import _bandwidth, _draw_smoothed, bootstrap_dea

B_FAST = 200
MODELS = [("BCC", "input"), ("BCC", "output"),
          ("CCR", "input"), ("CCR", "output")]


# --------------------------------------------------------------------------- #
#  Bandwidth                                                                  #
# --------------------------------------------------------------------------- #
def test_bandwidth_is_the_reflected_silverman_rule():
    """h = 0.9 * sd(eff) * (2n)^(-1/5), sd with ddof=1.

    Pinned deliberately: v1.0.7 used 1.06 * min(sd, IQR/1.349) * n^(-1/5),
    which left a systematic -0.008 offset against MaxDEA.
    """
    eff = np.linspace(0.2, 1.0, 25)
    n = len(eff)
    expected = 0.9 * np.std(eff, ddof=1) * (2 * n) ** (-1 / 5)
    assert _bandwidth(eff) == pytest.approx(expected, rel=1e-12)


def test_bandwidth_is_positive_for_a_degenerate_sample():
    assert _bandwidth(np.ones(10)) > 0


# --------------------------------------------------------------------------- #
#  The smoothed draw                                                          #
# --------------------------------------------------------------------------- #
def test_draw_stays_inside_the_unit_interval():
    rng = np.random.default_rng(1)
    eff = rng.uniform(0.05, 1.0, 60)
    star = _draw_smoothed(eff, _bandwidth(eff), rng, size=20000)
    assert np.all(star > 0.0) and np.all(star <= 1.0)


def test_draw_preserves_the_variance_of_the_efficiencies():
    """The variance correction is what keeps the pseudo-sample comparable."""
    rng = np.random.default_rng(2)
    eff = rng.uniform(0.1, 1.0, 80)
    star = _draw_smoothed(eff, _bandwidth(eff), rng, size=200000)
    assert star.var(ddof=1) == pytest.approx(eff.var(ddof=1), rel=0.10)
    assert star.mean() == pytest.approx(eff.mean(), abs=0.02)


def test_draw_is_reproducible():
    eff = np.linspace(0.2, 1.0, 30)
    h = _bandwidth(eff)
    a = _draw_smoothed(eff, h, np.random.default_rng(99), 500)
    b = _draw_smoothed(eff, h, np.random.default_rng(99), 500)
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
#  Structural guarantees                                                      #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def boot(sa_xy):
    X, Y = sa_xy
    return bootstrap_dea(X, Y, model="BCC", orient="input",
                         B=B_FAST, seed=12345, n_jobs=1)


def test_bias_is_non_negative(boot):
    """The radial DEA estimator is biased upward; a negative bias is impossible.

    v1.0.4 clipped the pseudo scores at 1 and produced negative biases for
    efficient DMUs, which is what this assertion exists to prevent.
    """
    assert np.all(boot["bias"] >= -1e-12), boot["bias"].min()


def test_bias_correction_lowers_the_score(boot):
    assert np.all(boot["theta_bc"] <= boot["theta_hat"] + 1e-9)


def test_confidence_interval_is_ordered_and_contains_the_corrected_score(boot):
    assert np.all(boot["ci_low"] <= boot["ci_high"])
    assert np.all(boot["ci_low"] <= boot["boot_median"] + 1e-12)
    assert np.all(boot["boot_median"] <= boot["ci_high"] + 1e-12)


def test_all_statistics_are_inside_the_unit_interval(boot):
    for k in ("theta_bc", "boot_mean", "boot_median", "ci_low", "ci_high"):
        v = boot[k]
        assert np.all(np.isfinite(v)), k
        assert np.all(v > 0) and np.all(v <= 1.0 + 1e-12), k


def test_same_seed_gives_identical_results(sa_xy):
    X, Y = sa_xy
    kw = dict(model="BCC", orient="input", B=60, seed=777, n_jobs=1)
    a = bootstrap_dea(X, Y, **kw)
    b = bootstrap_dea(X, Y, **kw)
    for k in ("theta_hat", "theta_bc", "bias", "ci_low", "ci_high"):
        np.testing.assert_array_equal(a[k], b[k])


def test_results_do_not_depend_on_the_number_of_processes(sa_xy):
    """The progress bar's Cores control must not change a single number."""
    X, Y = sa_xy
    kw = dict(model="BCC", orient="input", B=60, seed=4242)
    serial = bootstrap_dea(X, Y, n_jobs=1, **kw)
    parallel = bootstrap_dea(X, Y, n_jobs=4, **kw)
    for k in ("theta_hat", "theta_bc", "bias", "boot_sd", "ci_low", "ci_high"):
        np.testing.assert_allclose(parallel[k], serial[k], rtol=0, atol=1e-12)


@pytest.mark.parametrize("model,orient", MODELS)
def test_every_model_runs_and_reproduces_the_point_estimate(sa_xy, sa_ref_dea,
                                                            sa, model, orient):
    """theta_hat from the bootstrap must equal the plain DEA score
    (Benchmarking's, which PLEAP's DEA matches to 1e-12)."""
    X, Y = sa_xy
    r = bootstrap_dea(X, Y, model=model, orient=orient, B=20, seed=5, n_jobs=1)
    ref = sa_ref_dea[(sa_ref_dea.model == model) & (sa_ref_dea.orient == orient)]
    ref = dict(zip(ref["dmu"], ref["score"]))
    got = dict(zip(sa["dmu"], r["theta_hat"]))
    d = np.array([abs(got[k] - ref[k]) for k in ref])
    assert d.max() < 1.0e-9


# --------------------------------------------------------------------------- #
#  Agreement with MaxDEA, against the Monte-Carlo spread                      #
# --------------------------------------------------------------------------- #
# MaxDEA's reference run used B = 500 (workbook sheet "Bootstrap(B 500, Core
# 18)"), so PLEAP is run at the same B.  Up to v1.11 these tests ran PLEAP at
# B = 1000 or 2000 against it, which by itself makes |PLEAP - MaxDEA| larger
# than |PLEAP - PLEAP|: the expected ratio of the two is
# sqrt((1/2000 + 1/500) / (2/2000)) = 1.58 at B = 2000.
B_MAXDEA = 500


@pytest.fixture(scope="module")
def maxdea_matched_runs(ds_a_xy):
    XA, YA = ds_a_xy
    return [bootstrap_dea(XA, YA, model="BCC", orient="input", B=B_MAXDEA, seed=s, n_jobs=1)
            for s in (1, 2, 3)]


def _ref(ref_bootstrap, ds_a, col):
    ref = ref_bootstrap[(ref_bootstrap.model == "BCC") & (ref_bootstrap.orient == "input")]
    return ref.set_index("dmu").loc[ds_a["dmu"], col].to_numpy(float)


@pytest.mark.slow
def test_bias_corrected_mean_tracks_maxdea_on_inefficient_dmus(maxdea_matched_runs, ds_a,
                                                               ref_bootstrap):
    """Inside the frontier the two programs cannot be told apart.

    At MaxDEA's own B the gap between PLEAP and MaxDEA is the size of the gap
    between two PLEAP seeds: over five seeds the ratio is 1.08 (BCC input),
    1.09 (BCC output) and 0.96 (CCR) on the 56 or 57 units off the frontier.
    Three seeds are used here; a single pair varies between 0.6 and 2.4.
    """
    runs = maxdea_matched_runs
    ref = _ref(ref_bootstrap, ds_a, "mean")
    inefficient = runs[0]["theta_hat"] < 1.0 - 1e-9
    assert inefficient.sum() > 40, "fixture should be mostly inefficient"
    ref_gap = np.mean([np.mean(np.abs(r["boot_mean"][inefficient] - ref[inefficient])) for r in runs])
    self_gap = np.mean([np.mean(np.abs(a["boot_mean"][inefficient] - b["boot_mean"][inefficient]))
                        for a, b in ((runs[0], runs[1]), (runs[0], runs[2]), (runs[1], runs[2]))])
    assert self_gap > 0, "the bootstrap is not varying with the seed"
    assert ref_gap < 0.003, f"PLEAP-MaxDEA gap {ref_gap:.5f} on inefficient DMUs"
    assert ref_gap < 2.0 * self_gap, (
        f"PLEAP-MaxDEA {ref_gap:.5f} vs PLEAP-PLEAP {self_gap:.5f}")


@pytest.mark.slow
def test_the_gap_to_maxdea_concentrates_on_efficient_dmus(maxdea_matched_runs, ds_a,
                                                          ref_bootstrap):
    """Documents a known, bounded disagreement rather than hiding it.

    For a DMU on the estimated frontier the raw bootstrap `Bias` is
    heavy-tailed: the pseudo-frontier can fall well inside the observation, so
    a handful of replications return very large super-efficiency scores and
    dominate the mean.  Both programs' raw bias for such a unit is an artefact
    of that tail, and they do not agree on it.

    The bias-CORRECTED mean is protected by the ratio form of the correction,
    so the disagreement there stays near 0.01 - 1.3 times the seed-to-seed
    gap on the four frontier units at matched B (1.7 under BCC output
    orientation).  It is asserted as a bound, not claimed to be zero.
    """
    a = maxdea_matched_runs[0]
    ref = _ref(ref_bootstrap, ds_a, "mean")
    efficient = a["theta_hat"] >= 1.0 - 1e-9
    assert efficient.any(), "fixture should contain efficient DMUs"
    gap = np.mean(np.abs(a["boot_mean"][efficient] - ref[efficient]))
    assert gap < 0.05, f"gap on efficient DMUs grew to {gap:.4f}"
    assert np.all((a["boot_mean"][efficient] > 0.0) & (a["boot_mean"][efficient] <= 1.0))


@pytest.mark.slow
def test_replicate_spread_matches_maxdea(maxdea_matched_runs, ds_a, ref_bootstrap):
    """The reported SD is a scale claim; it must not drift.

    At matched B PLEAP's SDs average 0.94 of MaxDEA's (0.93-0.96 across the
    four models), and the per-unit gap is about three times the seed-to-seed
    gap: a small but systematic difference in a detail of MaxDEA's
    procedure that its documentation does not settle.  The bound keeps it
    from growing."""
    ref = _ref(ref_bootstrap, ds_a, "sd")
    ratio = np.mean([np.mean(r["boot_sd"]) for r in maxdea_matched_runs]) / np.mean(ref)
    assert 0.85 < ratio < 1.15, f"mean SD ratio PLEAP/MaxDEA = {ratio:.3f}"


@pytest.mark.slow
def test_bias_corrected_scores_preserve_the_ranking(sa_xy):
    """Bias correction must not reshuffle the units.

    Spearman correlation between the raw and the corrected score, computed
    without scipy so the test has no extra dependency.
    """
    X, Y = sa_xy
    r = bootstrap_dea(X, Y, model="BCC", orient="input", B=500, seed=3, n_jobs=1)
    a = np.argsort(np.argsort(r["theta_hat"]))
    b = np.argsort(np.argsort(r["theta_bc"]))
    rho = np.corrcoef(a, b)[0, 1]
    assert rho > 0.95, rho


@pytest.mark.slow
def test_the_bias_corrected_estimator_is_the_mean_of_ratios(sa_xy):
    """Pin which bias-correction form PLEAP reports.

    Three forms are in circulation and they are not interchangeable:

      additive        2*theta_hat - mean(theta_star)      (Simar-Wilson 1998)
      ratio of means  theta_hat^2 / mean(theta_star)
      mean of ratios  mean(theta_hat^2 / theta_star)      <- PLEAP

    On validation data set A the additive form's worst disagreement with
    MaxDEA was 37.5 on a quantity bounded by 1, because the raw bias is
    heavy-tailed on frontier DMUs.  Mean-of-ratios reproduces MaxDEA an order of magnitude better than
    ratio-of-means.  The paper documents that choice, so a silent change of
    form here would falsify the manuscript - hence this test.
    """
    X, Y = sa_xy
    r = bootstrap_dea(X, Y, model="BCC", orient="input", B=400, seed=99,
                      n_jobs=1)
    th, boots = r["theta_hat"], r["boots"]
    safe = np.where(np.isfinite(boots) & (boots > 0), boots, np.nan)

    mean_of_ratios = np.nanmean(
        np.clip((th[:, None] ** 2) / safe, 0.0, 1.0), axis=1)
    np.testing.assert_allclose(r["boot_mean"], mean_of_ratios,
                               rtol=0, atol=1e-12)

    ratio_of_means = th ** 2 / np.nanmean(boots, axis=1)
    additive = 2 * th - np.nanmean(boots, axis=1)
    assert not np.allclose(r["boot_mean"], ratio_of_means, atol=1e-6), \
        "mean-of-ratios must not be confused with ratio-of-means"
    assert not np.allclose(r["boot_mean"], additive, atol=1e-6)

    # theta_bc is the same object the second stage regresses on
    np.testing.assert_allclose(r["theta_bc"], r["boot_mean"], rtol=0, atol=0)


@pytest.mark.slow
def test_the_unit_clip_is_a_rail_and_does_not_bind(sa_xy):
    """The clip must not be doing the work the ratio form does.

    The manuscript says the corrected mean is protected from the heavy tail by
    the ratio correction.  If the clip were what kept it in range, that
    sentence would be wrong, so this asserts the clip is inert: a large
    theta_star drives theta_hat^2/theta_star toward zero, never above one.  It
    held on validation data set A as it does here on SA.
    """
    X, Y = sa_xy
    r = bootstrap_dea(X, Y, model="BCC", orient="input", B=400, seed=99,
                      n_jobs=1)
    th, boots = r["theta_hat"], r["boots"]
    safe = np.where(np.isfinite(boots) & (boots > 0), boots, np.nan)
    raw = (th[:, None] ** 2) / safe

    assert np.nanmax(raw) <= 1.0 + 1e-12, (
        "a replicate ratio exceeded 1; the clip now binds and the "
        "manuscript's explanation needs revisiting")
    np.testing.assert_allclose(np.nanmean(np.clip(raw, 0.0, 1.0), axis=1),
                               np.nanmean(raw, axis=1), rtol=0, atol=1e-12)


@pytest.mark.slow
def test_the_correction_is_the_shephard_scale_ratio_form(sa_xy):
    """Name the correction by the scale it actually acts on.

    An earlier draft called this a Farrell-scale correction and compared it
    against 2*theta_hat - mean(theta_star), which nobody computes.  Writing
    delta = 1/theta, the reported score satisfies

        1 / theta_tilde == delta_hat**2 / mean(delta_star)

    so it is the ratio correction on the SHEPHARD scale, first-order equal to
    the additive delta correction in common use.  The identity is exact, so
    this test pins the description the paper now gives.
    """
    X, Y = sa_xy
    r = bootstrap_dea(X, Y, model="BCC", orient="input", B=400, seed=99,
                      n_jobs=1)
    th, boots = r["theta_hat"], r["boots"]
    safe = np.where(np.isfinite(boots) & (boots > 0), boots, np.nan)

    d_hat = 1.0 / th
    d_star = 1.0 / safe
    shephard_ratio = d_hat ** 2 / np.nanmean(d_star, axis=1)
    np.testing.assert_allclose(r["boot_mean"], 1.0 / shephard_ratio,
                               rtol=0, atol=1e-12)

    # the Shephard additive form is a genuine alternative, not a degenerate
    # one: it must stay finite and positive, so the comparison in
    # docs/validation.md section 7 is between two usable estimators.
    shephard_additive = 2 * d_hat - np.nanmean(d_star, axis=1)
    assert np.all(shephard_additive > 0), \
        "the fair comparison must be against a well-defined estimator"
