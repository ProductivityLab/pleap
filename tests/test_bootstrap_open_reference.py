"""PLEAP's bootstrap against an open-source implementation, R ``Benchmarking``.

MaxDEA, the main bootstrap reference, is closed, so the residual gap to it
(docs/validation.md section 7) cannot be traced into its code.
``Benchmarking::dea.boot`` (Bogetoft and Otto, version 0.33) is open, and its
procedure is documented line by line: it smooths on the Shephard scale
delta = 1/theta, sampling from the reflected set {delta, 2 - delta} with a
robust Silverman bandwidth, and evaluates each unit against each pseudo-frontier
without joining it - as PLEAP does.  PLEAP smooths on the Farrell scale with its
own bandwidth, so the two conventions give different corrected means: 1.3e-2
apart on average on the synthetic data set SA, about 3e-2 on validation data
set A (scores down to 0.0094, delta up to 106).

That difference is a convention, not an error, and this test shows it: with
Benchmarking's draw and bandwidth swapped into PLEAP, everything else - the
pseudo-data, the evaluation, the correction from the replicates - is PLEAP's,
and the corrected means must agree with Benchmarking's to within Monte-Carlo
noise.  The reference is Benchmarking's mean over seeds 1-5 (B = 2000) of the
estimator PLEAP reports, computed from its own replicates
(``docs/external_reference/reference_sa.R``; for A,
``bootstrap_benchmarking.R``).  On SA, one PLEAP run at B = 1000 is 1.2e-3 from
it off the frontier, where two PLEAP seeds are 1.6e-3 apart and two
Benchmarking seeds 1.0e-3; on A the non-frontier gap was 4.6e-4 against
seed-to-seed variability of 5.0e-4 (PLEAP) and 6.7e-4 (Benchmarking)
(docs/bootstrap_benchmarking_aligned.py).

A is not redistributed (conftest.py), so its case is skipped without it.  The
patch runs serially (n_jobs=1) so that it cannot miss a worker process on
platforms that spawn rather than fork.
"""
import numpy as np
import pandas as pd
import pytest

from pleap.core import bootstrap as bt


def bw_benchmarking(eff):
    """Benchmarking 0.33, dea.boot: robust Silverman rule rescaled to delta."""
    dist = 1.0 / eff
    zeff = eff[dist > 1 + 1e-6]
    neff = np.r_[zeff, 2 - zeff]
    adjust = np.std(dist, ddof=1) / np.std(neff, ddof=1) * (len(neff) / len(dist)) ** 0.2
    std = np.std(neff, ddof=1)
    iqr = (np.percentile(neff, 75) - np.percentile(neff, 25)) / 1.349
    if 1e-6 < iqr < std:
        std = iqr
    return adjust * 0.9 * std * len(neff) ** (-1 / 5)


def draw_benchmarking(eff, h, rng, size):
    """Benchmarking 0.33, dea.sample, on the Shephard scale; returns Farrell scores."""
    e = 1.0 / eff
    espejl = np.r_[e, 2 - e]
    beta = espejl[rng.integers(0, len(espejl), size=size)]
    etilde = beta + h * rng.normal(0.0, 1.0, size=size)
    estar = beta.mean() + (etilde - beta.mean()) / np.sqrt(1 + h * h / np.var(espejl, ddof=1))
    estar = np.where(estar < 1, 2 - estar, estar)
    return 1.0 / estar


def _sa_reference(sa_ref_boot, dmus):
    piv = sa_ref_boot.pivot(index="dmu", columns="seed", values="bc_ratio").loc[dmus]
    return piv.mean(axis=1).to_numpy(float), float(sa_ref_boot["h_benchmarking"].iloc[0])


def test_the_bandwidth_replica_is_benchmarkings(sa, sa_xy, sa_ref_boot):
    """The replica reproduces the bandwidth dea.boot itself prints, from the
    DEA scores alone (Benchmarking's own, here)."""
    theta = sa_ref_boot[sa_ref_boot.seed == 1].set_index("dmu").loc[sa["dmu"], "theta"]
    _, h = _sa_reference(sa_ref_boot, list(sa["dmu"]))
    assert bw_benchmarking(theta.to_numpy(float)) == pytest.approx(h, rel=1e-11)


# data, reference, Benchmarking's bandwidth, and the gap bounds off and on the
# frontier (expected gaps: 1.2e-3 and 2.2e-3 on SA, 5e-4 off the frontier on A)
CASES = {"SA (public)": ("sa", 2.0e-3, 5e-3),
         "A (private)": ("ds_a", 1.5e-3, 6e-3)}


@pytest.mark.slow
@pytest.mark.parametrize("case", list(CASES))
def test_with_benchmarkings_smoothing_pleap_reproduces_benchmarking(request, monkeypatch, case):
    data, off_bound, on_bound = CASES[case]
    ds = request.getfixturevalue(data)
    if data == "sa":
        want, h = _sa_reference(request.getfixturevalue("sa_ref_boot"), list(ds["dmu"]))
    else:
        ref = request.getfixturevalue("ref_a_benchmarking_boot")
        assert list(ref["dmu"]) == list(ds["dmu"])
        want, h = ref["bc_ratio_mean5"].to_numpy(float), 7.165   # Benchmarking's h on A
    X, Y = ds[["i1", "i2"]].to_numpy(float), ds[["o1"]].to_numpy(float)

    monkeypatch.setattr(bt, "_bandwidth", bw_benchmarking)
    monkeypatch.setattr(bt, "_draw_smoothed", draw_benchmarking)
    r = bt.bootstrap_dea(X, Y, model="BCC", orient="input", B=1000, seed=1, n_jobs=1)

    assert r["bandwidth"] == pytest.approx(h, abs=1e-3)
    frontier = r["theta_hat"] > 1 - 1e-6
    gap = np.abs(r["boot_mean"] - want)
    assert gap[~frontier].mean() < off_bound, gap[~frontier].mean()
    assert gap[frontier].mean() < on_bound, gap[frontier].mean()
