"""
Radial DEA (CCR / BCC, input- or output-oriented) with second-phase slacks.

v1.12  - the linear programs are now solved in-process with scipy / HiGHS
          instead of writing an LP file and spawning an external CBC process
          per solve (pulp / PULP_CBC_CMD).  Three reasons:
            * PuLP >= 4.0 removed `PULP_CBC_CMD` and no longer ships a bundled
              CBC binary, so the previous code could not run at all on a fresh
              install;
            * `bootstrap.py` and `malmquist.py` already solve their LPs with
              HiGHS, so the three modules now share one solver and one
              formulation;
            * it removes the last external-solver dependency, making the
              package installable from wheels alone on every platform.
          The formulation, the column scaling and the public API are unchanged,
          and the scores / lambdas / slacks reproduce the previous CBC results
          and the MaxDEA reference (see tests/test_dea_reference.py).

Formulation (envelopment form).  For DMU k, input orientation:

    min  theta
    s.t. sum_j lam_j x_ji <= theta x_ki     i = 1..m
         sum_j lam_j y_jr >=       y_kr     r = 1..s
         sum_j lam_j = 1                    (BCC / VRS only)
         lam >= 0

Output orientation maximises phi with the inequalities exchanged; the reported
efficiency is 1 / phi so that it is always in (0, 1].  Phase 2 then maximises
the sum of the slacks at the optimal radial score, which identifies the
strongly efficient projection point and Koopmans efficiency.
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


import numpy as np
from scipy.optimize import linprog

from pleap.core.lp import TIGHT, best_point, envelopment
from pleap.core.scaling import column_scales, scaling_is_risky


# --------------------------------------------------------------------------- #
#  Phase 1 - radial score.                                                    #
#  Variables: [theta|phi, lam_1 .. lam_n]                                     #
# --------------------------------------------------------------------------- #
def _solve_one(k, X, Y, model="BCC", orient="input", options=None):
    """Phase 1 for DMU ``k`` on data exactly as given (see pleap.core.lp)."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    return envelopment(X[k], Y[k], X, Y, model, orient, options=options)


# --------------------------------------------------------------------------- #
#  Phase 2 - maximise the slacks at the phase-1 radial score.                 #
#  Variables: [lam_1 .. lam_n, sm_1 .. sm_m, sp_1 .. sp_s]                    #
# --------------------------------------------------------------------------- #
def _phase2_slacks(k, X, Y, score, model, orient, options=None):
    n, m = X.shape
    _, s = Y.shape
    nv = n + m + s
    c = np.zeros(nv)
    c[n:] = -1.0                                 # maximise sum(sm) + sum(sp)

    rows = m + s + (1 if model == "BCC" else 0)
    A_eq = np.zeros((rows, nv))
    b_eq = np.zeros(rows)

    for i in range(m):
        A_eq[i, :n] = X[:, i]
        A_eq[i, n + i] = 1.0                     # sum lam x + sm == rhs
        b_eq[i] = (score if orient == "input" else 1.0) * X[k, i]
    for r in range(s):
        A_eq[m + r, :n] = Y[:, r]
        A_eq[m + r, n + m + r] = -1.0            # sum lam y - sp == rhs
        b_eq[m + r] = (1.0 if orient == "input" else score) * Y[k, r]
    if model == "BCC":
        A_eq[m + s, :n] = 1.0
        b_eq[m + s] = 1.0

    res = linprog(c, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * nv, method="highs",
                  options=dict(options or {}))
    if not res.success:
        return np.zeros(n), np.zeros(m), np.zeros(s)
    # Same clamp as phase 1: slacks are non-negative by construction, and a
    # value of -1e-17 would otherwise be printed as "-0.000000".
    x = np.clip(np.asarray(res.x, float), 0.0, None)
    return x[:n], x[n:n + m], x[n + m:]


def _audited_solve(k, X, Y, sx, sy, model, orient, cross=True):
    """Phase 1 for DMU ``k`` under the candidate rule of pleap.core.lp.

    Column scaling keeps the LP well conditioned, but a rescaled problem can be
    ill-posed even when the original is not: if a scaled coefficient falls
    below the solver's feasibility tolerance, the corresponding constraint
    silently stops binding and the solver returns the optimum of a *different*
    problem (see ``pleap.core.scaling``).

    A *primal feasibility* check cannot catch every case.  Under input
    orientation ``theta = 1`` with ``lambda_k = 1`` is always feasible - a DMU
    is its own reference - so an overstated score can satisfy every constraint
    exactly.  Feasibility is not optimality.  A *dual* test would detect it
    (``docs/validation.md`` section 9 measures one), but it diagnoses without
    repairing and needs a tolerance of its own.

    What works is comparing independently conditioned solves.  When ``cross``
    is set - decided by the callers from ``scaling_is_risky``, a test of the
    data rather than of the solution - the program is solved on the scaled
    data, on the raw data, and on the scaled data at tightened tolerances.
    All admissible answers are feasible points of the SAME original problem,
    so the one with the better objective is the one nearest the optimum.  On
    well-conditioned data one solve is enough and nothing extra is paid.

    Returns ``(score, lambdas, source)``; ``source`` says which problem the
    answer came from, so the caller solves phase 2 on the same one.
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    return best_point(X[k], Y[k], X, Y, sx, sy, model, orient, risky=cross)


def dea(X, Y, model="BCC", orient="input", progress=None):
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    sx = column_scales(X); sy = column_scales(Y)
    Xs = X / sx; Ys = Y / sy
    cross = scaling_is_risky(X, sx) or scaling_is_risky(Y, sy)
    n = X.shape[0]
    radial = np.full(n, np.nan); eff = np.full(n, np.nan)
    lams = np.zeros((n, n))
    sm = np.zeros_like(X); sp = np.zeros_like(Y)

    for k in range(n):
        score, _, source = _audited_solve(k, X, Y, sx, sy, model,
                                          orient, cross)
        if score is None:
            if progress is not None: progress(k + 1, n)
            continue
        radial[k] = score
        eff[k] = score if orient == "input" else (
            1.0 / score if score > 0 else np.nan)
        # Phase 2 is solved on whichever problem phase 1 was answered from, so
        # that the fixed radial score is consistent with the data it came from.
        if source != "raw":
            lam_v, sm_v, sp_v = _phase2_slacks(
                k, Xs, Ys, score, model, orient,
                TIGHT if source == "tight" else None)
            sm[k] = sm_v * sx; sp[k] = sp_v * sy      # slacks back to raw units
        else:
            lam_v, sm_v, sp_v = _phase2_slacks(k, X, Y, score, model, orient)
            sm[k] = sm_v; sp[k] = sp_v
        lams[k] = lam_v
        if progress is not None: progress(k + 1, n)

    return {"efficiency": eff, "radial": radial, "lambdas": lams,
            "slack_input": sm, "slack_output": sp,
            "model": model, "orient": orient}


def _solve_super(k, X, Y, model="BCC", orient="input", options=None):
    """Phase-1 LP with DMU ``k`` barred from its own reference set.

    This is the Andersen-Petersen super-efficiency program.  It is the
    ordinary envelopment problem with ``lambda_k`` fixed at zero, so an
    efficient unit is measured against the frontier the other units span and
    can score better than 1.  Inefficient units are unaffected: their
    ``lambda_k`` is already zero at the optimum.  Returns theta or phi, or
    None when the program is infeasible (possible under VRS).
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    r = envelopment(X[k], Y[k], X, Y, model, orient, exclude=k,
                    options=options)
    return None if r is None else r[0]


def super_efficiency(X, Y, model="BCC", orient="input", progress=None):
    """Andersen-Petersen super-efficiency scores.

    Each unit is measured against the technology the *other* units span, so a
    unit on the estimated frontier is no longer capped at 1 and the units that
    ordinary DEA ties at 1 can be ranked.  Scores for inefficient units are
    identical to the ordinary ones.

    Two cautions travel with the measure and are reported rather than hidden:

      * Under VRS the program can be **infeasible** for a unit at the edge of
        the input or output range.  Those units get ``nan`` and are counted in
        ``attrs["infeasible"]``; under CRS the program is always feasible.
      * Super-efficiency is sensitive to a single outlying unit, which is what
        makes it useful for outlier screening and risky as a ranking.

    Returns a dict with ``score`` (the Farrell-type value, > 1 for a
    super-efficient unit under input orientation) and ``infeasible``.
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    sx = column_scales(X); sy = column_scales(Y)
    risky = scaling_is_risky(X, sx) or scaling_is_risky(Y, sy)
    n = X.shape[0]
    out = np.full(n, np.nan)

    for k in range(n):
        best, _, _ = best_point(X[k], Y[k], X, Y, sx, sy, model, orient,
                                risky=risky, exclude=k)
        if best is not None:
            out[k] = best if orient == "input" else (
                1.0 / best if best > 0 else np.nan)
        if progress is not None:
            progress(k + 1, n)

    n_inf = int(np.sum(~np.isfinite(out)))
    res = {"score": out, "infeasible": n_inf, "model": model, "orient": orient}
    return res


def rts_classify(X, Y, orient="input", progress=None):
    """Returns to scale of each unit's radial projection, in either orientation.

    Rule (Seiford and Zhu 1999, Omega 27(1):1-11): if the CCR and BCC scores
    are equal the unit is scale efficient, CRS.  Otherwise the sum of the
    CCR intensity weights decides, and it does so in the SAME direction in both
    orientations: sum(lambda) < 1 is IRS, > 1 is DRS.  Seiford and Zhu show
    that for a scale-inefficient unit every alternate optimum gives the same
    side of 1, so one optimal solution is enough.

    v1.12 corrects the output orientation, which had the two labels swapped
    (every non-CRS output-oriented unit was mislabelled).  The rule is checked
    against the independent non-increasing-returns (NIRS) test of Fare,
    Grosskopf and Lovell on all 453 units of the D fixture, in both
    orientations (tests/test_rts.py).

    Returns (labels, ccr, bcc), the scores on the Farrell scale.
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    sx = column_scales(X); sy = column_scales(Y)
    cross = scaling_is_risky(X, sx) or scaling_is_risky(Y, sy)
    n = X.shape[0]
    out = []; ccr = np.zeros(n); bcc = np.zeros(n); sumlam = np.zeros(n)
    for k in range(n):
        score_c, lam_c, _ = _audited_solve(k, X, Y, sx, sy, "CCR", orient, cross)
        score_b, _, _ = _audited_solve(k, X, Y, sx, sy, "BCC", orient, cross)
        if score_c is None or score_b is None:
            out.append("NA")
            if progress is not None: progress(k+1, n)
            continue
        if orient == "input":
            ccr[k] = score_c; bcc[k] = score_b
        else:
            ccr[k] = 1.0/score_c if score_c > 0 else np.nan
            bcc[k] = 1.0/score_b if score_b > 0 else np.nan
        sumlam[k] = lam_c.sum()
        se = ccr[k]/bcc[k] if bcc[k] > 0 else np.nan
        if not np.isfinite(se):
            out.append("NA")
        elif abs(se - 1.0) < 1e-5:
            out.append("CRS")
        else:
            # Same direction for both orientations (see the docstring).
            out.append("IRS" if sumlam[k] < 1 - 1e-5
                       else "DRS" if sumlam[k] > 1 + 1e-5 else "CRS")
        if progress is not None: progress(k+1, n)
    return out, ccr, bcc
