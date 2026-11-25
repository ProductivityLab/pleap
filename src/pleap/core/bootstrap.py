"""
Bootstrap DEA with bias correction (Kneip, Simar & Wilson 2008).

v1.0.5  - fixed the bias computation (original observation re-evaluated against
          the pseudo-frontier, no clipping; bias is non-negative by construction).
v1.0.6  - performance rewrite:
            * the LP is solved in-process with scipy / HiGHS instead of spawning
              an external CBC process per solve (pulp).  For a typical problem
              this is roughly 10-50x faster.
            * bootstrap replications are run in parallel across CPU cores.
            * results are reproducible and independent of the number of cores
              (each replication draws from its own spawned random stream).
          The statistical method is unchanged from v1.0.5.
v1.0.7  - corrected the smoothed-bootstrap draw so it matches the MaxDEA
          reference on highly inefficient data:
            * the bandwidth is computed on the ORIGINAL efficiencies (v1.0.6
              used the reflected 2n sample, whose inflated spread distorted it);
            * draws are resampled from the original efficiencies and
              variance-corrected toward the resample mean (v1.0.6 recentred them
              toward 1.0) and reflected at BOTH boundaries (0 and 1).
          This removes the occasional explosive bias values produced by v1.0.6
          and reproduces MaxDEA's Mean / SD / CI across all four models.
          Output columns now follow the MaxDEA layout.
"""

# PLEAP - Productivity Lab Efficiency Analysis Program
# Copyright (C) 2025-2026 Changhee Kim, Productivity Lab,
#                         Incheon National University
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
import numpy as np

from pleap.core.lp import best_point
from pleap.core.scaling import column_scales, scaling_is_risky
from concurrent.futures import ProcessPoolExecutor, as_completed


# --------------------------------------------------------------------------- #
#  Evaluate an EXTERNAL point against a reference technology (scipy / HiGHS).  #
#  The point need not belong to the reference set, so the returned Farrell     #
#  score may exceed 1 (super-efficiency).  It must never be clipped here.      #
# --------------------------------------------------------------------------- #
def _score_point(xk, yk, Xref, Yref, model="BCC", orient="input"):
    """Farrell score of point (xk, yk) against the technology (Xref, Yref).

    Column scaling keeps the LP well conditioned without changing the
    (scale-invariant) score; see pleap.core.scaling for why the geometric mean
    is used rather than the arithmetic mean, and pleap.core.lp for the
    candidate rule applied when the data are close to the solver's limits.
    """
    Xref = np.asarray(Xref, float); Yref = np.asarray(Yref, float)
    sx = column_scales(Xref); sy = column_scales(Yref)
    risky = scaling_is_risky(Xref, sx) or scaling_is_risky(Yref, sy)
    v, _, _ = best_point(xk, yk, Xref, Yref, sx, sy, model, orient, risky)
    if v is None or not np.isfinite(v) or v <= 0:
        return np.nan
    return v if orient == "input" else 1.0 / v


# --------------------------------------------------------------------------- #
#  Smoothed-bootstrap helpers (reflection method, Silverman-type bandwidth).  #
# --------------------------------------------------------------------------- #
def _bandwidth(eff):
    # Silverman rule-of-thumb bandwidth for the reflection method: the scale is
    # the standard deviation of the ORIGINAL efficiency scores, while the
    # sample-size factor uses the reflected sample size 2n,
    #     h = 0.9 * sd(eff) * (2n)^(-1/5).
    # Calibrated against the MaxDEA reference on three independent datasets
    # (highly inefficient, mixed, and highly efficient score distributions);
    # it removes the small systematic offset of the plain 1.06*min(sd,IQR/1.349)
    # *n^(-1/5) rule while remaining a standard normal-reference bandwidth.
    n = len(eff)
    sd = np.std(eff, ddof=1)
    if sd <= 0:
        sd = 1e-3
    return 0.9 * sd * (2 * n) ** (-1 / 5)


def _draw_smoothed(eff, h, rng, size):
    # Simar & Wilson (1998) smoothed homogeneous bootstrap draw.
    #   * resample from the ORIGINAL efficiencies,
    #   * add Gaussian noise (bandwidth h),
    #   * reflect at the efficient boundary (1) and at 0 so the draw stays in
    #     (0, 1],
    #   * variance-correct toward the resample mean (NOT toward 1.0) so the
    #     bootstrap variance matches the empirical variance.
    n = len(eff)
    samp = eff[rng.integers(0, n, size=size)]
    pert = samp + h * rng.normal(0.0, 1.0, size=size)
    pert = np.where(pert > 1.0, 2.0 - pert, pert)          # reflect at 1
    ebar = samp.mean()
    var_e = np.var(eff, ddof=1)
    if var_e > 0:
        star = ebar + (pert - ebar) / np.sqrt(1.0 + h * h / var_e)
    else:
        star = pert
    star = np.where(star > 1.0, 2.0 - star, star)          # reflect at 1
    star = np.abs(star)                                    # reflect at 0
    star = np.where(star > 1.0, 2.0 - star, star)
    return np.clip(star, 1e-4, 1.0)


# --------------------------------------------------------------------------- #
#  One bootstrap replication and a chunk of replications (module level so they #
#  can be pickled and dispatched to worker processes).                        #
# --------------------------------------------------------------------------- #
def _one_replication(seed, X, Y, eff_hat, h, model, orient):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    star = _draw_smoothed(eff_hat, h, rng, size=n)
    if orient == "input":
        Xref = X * (eff_hat / star).reshape(-1, 1)
        Yref = Y
    else:
        Yref = Y * (star / eff_hat).reshape(-1, 1)
        Xref = X
    col = np.empty(n)
    for k in range(n):
        col[k] = _score_point(X[k], Y[k], Xref, Yref, model, orient)
    return col


def _boot_chunk(seeds, X, Y, eff_hat, h, model, orient):
    return np.column_stack([_one_replication(sd, X, Y, eff_hat, h, model, orient)
                            for sd in seeds])


# --------------------------------------------------------------------------- #
#  Main entry point.                                                          #
# --------------------------------------------------------------------------- #
def bootstrap_dea(X, Y, model="BCC", orient="input", B=2000, alpha=0.05,
                  seed=12345, n_jobs=None, progress=None, **_legacy):
    """
    Kneip-Simar-Wilson (2008) bootstrap with bias correction.

    n_jobs : number of worker processes.  None or <=0 -> use all CPU cores.
             1 -> serial.  Results are identical regardless of n_jobs.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = X.shape[0]

    # 1. original efficiencies (Farrell scale, in (0, 1])
    eff_hat = np.array([_score_point(X[k], Y[k], X, Y, model, orient)
                        for k in range(n)])
    eff_hat = np.clip(eff_hat, 1e-6, 1.0)

    # 2. smoothed-bootstrap bandwidth and per-replication random streams
    h = _bandwidth(eff_hat)
    seeds = np.random.SeedSequence(seed).spawn(B)   # reproducible, core-independent

    # 3. bootstrap replications (parallel where possible, serial fallback)
    if n_jobs is None or n_jobs <= 0:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, min(n_jobs, B))

    boots = np.empty((n, B))
    done = 0

    def _report():
        if progress is not None:
            progress(min(done, B), B)

    ran_parallel = False
    if n_jobs > 1:
        # split into several chunks per worker for smoother progress
        n_chunks = min(B, n_jobs * 4)
        chunks = [list(c) for c in np.array_split(np.arange(B), n_chunks) if len(c)]
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                fut2chunk = {
                    ex.submit(_boot_chunk, [seeds[b] for b in ch],
                              X, Y, eff_hat, h, model, orient): ch
                    for ch in chunks
                }
                for fut in as_completed(fut2chunk):
                    ch = fut2chunk[fut]
                    boots[:, np.array(ch)] = fut.result()
                    done += len(ch); _report()
            ran_parallel = True
        except Exception:
            ran_parallel = False  # fall back to serial below

    if not ran_parallel:
        done = 0
        for b in range(B):
            boots[:, b] = _one_replication(seeds[b], X, Y, eff_hat, h, model, orient)
            done += 1; _report()

    # 4. bias = mean(theta*) - theta_hat   (non-negative by construction)
    bias = np.nanmean(boots, axis=1) - eff_hat

    # 5. KSW2008 ratio bias-correction applied to every replicate
    safe = np.where(np.isfinite(boots) & (boots > 0), boots, np.nan)
    bc = np.clip((eff_hat[:, None] ** 2) / safe, 0.0, 1.0)

    boot_mean = np.nanmean(bc, axis=1)
    boot_median = np.nanmedian(bc, axis=1)
    boot_sd = np.nanstd(bc, axis=1, ddof=1)
    ci_low = np.nanpercentile(bc, 100 * (alpha / 2), axis=1)
    ci_high = np.nanpercentile(bc, 100 * (1 - alpha / 2), axis=1)
    theta_bc = boot_mean

    return {"theta_hat": eff_hat, "theta_bc": theta_bc, "bias": bias,
            "boot_mean": boot_mean, "boot_median": boot_median,
            "boot_sd": boot_sd, "ci_low": ci_low, "ci_high": ci_high,
            "bandwidth": h, "boots": boots, "B": B, "alpha": alpha,
            "method": "KSW2008", "n_jobs": n_jobs}
