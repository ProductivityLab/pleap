"""PLEAP's bootstrap with Benchmarking's smoothing convention, against Benchmarking.

    Rscript docs/external_reference/bootstrap_benchmarking.R 1 2 3 4 5
    python docs/bootstrap_benchmarking_aligned.py

Benchmarking::dea.boot (Bogetoft and Otto) smooths on the Shephard scale
delta = 1/theta: it samples from the reflected set {delta, 2 - delta}, adds
Gaussian noise with a robust Silverman bandwidth, variance-corrects with the
mean of the resample and the variance of the reflected set, and reflects at 1.
PLEAP smooths on the Farrell scale with its own bandwidth.  On data set A,
where scores reach 0.0094 (delta = 106), the two conventions give different
bootstrap distributions, so their corrected scores differ by design.

To test the rest of PLEAP's machinery against an open implementation - the
pseudo-data, the evaluation of each unit against each pseudo-frontier without
joining it, and the correction from the replicates - this script swaps in
Benchmarking's draw and bandwidth, runs seeds 1-3 (serially, so the patch
cannot miss a worker) and compares with Benchmarking's replicates under the
same estimator.  Writes docs/bootstrap_benchmarking_aligned.json.
"""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from pleap.core import bootstrap as bt  # noqa: E402


def bw_benchmarking(eff):
    """Benchmarking 0.33, dea.boot: robust Silverman rule, rescaled to delta."""
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
    """Benchmarking 0.33, dea.sample, on the Shephard scale; returns Farrell."""
    e = 1.0 / eff
    espejl = np.r_[e, 2 - e]
    beta = espejl[rng.integers(0, len(espejl), size=size)]
    etilde = beta + h * rng.normal(0.0, 1.0, size=size)
    estar = beta.mean() + (etilde - beta.mean()) / np.sqrt(1 + h * h / np.var(espejl, ddof=1))
    estar = np.where(estar < 1, 2 - estar, estar)
    return 1.0 / estar


PRIVATE = Path(os.environ.get("PLEAP_PRIVATE_DATA", "tests/data_private"))   # KIS data: not redistributed
ds = pd.read_csv(PRIVATE / "ds_a_dea.csv")
X, Y = ds[["i1", "i2"]].to_numpy(float), ds[["o1"]].to_numpy(float)
bt._bandwidth, bt._draw_smoothed = bw_benchmarking, draw_benchmarking
runs = []
for s in (1, 2, 3):
    r = bt.bootstrap_dea(X, Y, model="BCC", orient="input", B=2000, seed=s, n_jobs=1)
    runs.append(np.asarray(r["boot_mean"], float))
    print(f"seed {s}: h = {r['bandwidth']:.4f}", flush=True)
theta = np.asarray(r["theta_hat"], float)
frontier = theta > 1 - 1e-6

bm = pd.read_csv(PRIVATE / "bootstrap_benchmarking_seeds_1-2-3-4-5.csv")
B = np.vstack([bm[bm.seed == s]["bc_ratio"].to_numpy(float) for s in range(1, 6)])
P = np.vstack(runs)


def pair_gap(M):
    return np.mean([np.abs(M[i] - M[j]) for i, j in itertools.combinations(range(len(M)), 2)], axis=0)


gap = np.abs(P.mean(0) - B.mean(0))
out = {}
for name, m in (("nonfrontier", ~frontier), ("frontier", frontier)):
    out[name] = dict(gap_pleap_aligned_vs_benchmarking=round(float(gap[m].mean()), 6),
                     seed_to_seed_pleap_aligned=round(float(pair_gap(P)[m].mean()), 6),
                     seed_to_seed_benchmarking=round(float(pair_gap(B)[m].mean()), 6),
                     max_gap=round(float(gap[m].max()), 6))
out["h"] = round(float(r["bandwidth"]), 4)
print(json.dumps(out, indent=1))
json.dump(out, open("docs/bootstrap_benchmarking_aligned.json", "w"), indent=1)
