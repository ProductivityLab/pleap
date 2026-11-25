"""
Column scaling for the DEA linear programs.

Why scale at all
----------------
DEA scores and intensity weights are invariant to the unit of measurement of
each input and output: replacing column j by c_j * column j (c_j > 0) leaves
every radial score unchanged.  Floating-point LP solvers, however, are not
invariant.  With raw magnitudes around 1e5-1e6 HiGHS was observed to terminate
about 1e-3 short of the optimum on a handful of DMUs, so v1.0.9 began solving
on rescaled columns.

Why NOT the arithmetic mean
---------------------------
v1.0.9-v1.1.0 divided each column by its arithmetic mean.  That is safe only
when the column is not strongly skewed.  On a column whose values span many
orders of magnitude the mean sits near the top of the range, so the smallest
observations are pushed below the solver's feasibility tolerance and the
corresponding constraint silently stops binding - the solver then reports a
feasible-looking but wrong optimum.

The synthetic case of the paper, tests/data/scaling_case.csv, shows it: its
input x2 spans 11.3 decades, the arithmetic mean puts the smallest entry at
1.2e-10, and under CCR input orientation nine of the thirty scores move, by up
to 0.15 (tests/test_scaling_reference.py, against the optimum in exact
arithmetic).

The geometric mean is the centre of the column *in log space*, so it is not
dragged by the upper tail: after scaling, the smallest and largest entries sit
symmetrically around 1.  Dividing by a constant cannot change a column's
largest-to-smallest ratio; what it changes is where that range sits relative
to the solver's absolute tolerance.  On data that are still close to the
solver's limits after centring, ``scaling_is_risky`` says so and the callers
apply the candidate rule of pleap.core.lp.
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

__all__ = ["column_scales", "feasible", "scaling_is_risky"]

# HiGHS's default primal feasibility tolerance.  A scaled coefficient this
# small is indistinguishable from zero to the solver.
SOLVER_TOL = 1e-7


def column_scales(A):
    """Per-column geometric mean of the strictly positive entries of ``A``.

    Returns an array of length ``A.shape[1]``, every element strictly positive
    and finite, so that ``A / column_scales(A)`` is always well defined.
    Columns that are empty, all non-positive, or otherwise degenerate get a
    scale of 1.0 (i.e. they are left alone).
    """
    A = np.asarray(A, float)
    if A.ndim != 2:
        raise ValueError("column_scales expects a 2-D array")
    pos = A > 0
    s = np.ones(A.shape[1], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        for j in range(A.shape[1]):
            col = A[pos[:, j], j]
            if col.size:
                s[j] = float(np.exp(np.log(col).mean()))
    s[~np.isfinite(s)] = 1.0
    s[s <= 0] = 1.0
    return s


def feasible(xk, yk, X, Y, lam, score, model="BCC", orient="input", tol=1e-6):
    """Check a phase-1 solution against the envelopment constraints.

    The check is carried out in the ORIGINAL units with a relative tolerance,
    which is what makes it able to catch a solution that only looked feasible
    on the rescaled problem.  Returns True when the solution is admissible.
    """
    xk = np.asarray(xk, float); yk = np.asarray(yk, float)
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    lam = np.asarray(lam, float)
    if not np.all(np.isfinite(lam)) or not np.isfinite(score):
        return False
    if np.any(lam < -tol):
        return False
    if model == "BCC" and abs(lam.sum() - 1.0) > 1e-5:
        return False

    xr = lam @ X
    yr = lam @ Y
    tx = (score if orient == "input" else 1.0) * xk
    ty = (1.0 if orient == "input" else score) * yk
    # relative slack, guarded for zero targets
    sx = np.where(np.abs(tx) > 0, (xr - tx) / np.abs(tx), xr - tx)
    sy = np.where(np.abs(ty) > 0, (ty - yr) / np.abs(ty), ty - yr)
    return bool(np.all(sx <= tol) and np.all(sy <= tol))


def scaling_is_risky(A, scales=None, tol=SOLVER_TOL, margin=1e4):
    """Is a single solve on ``A / scales`` safe to trust on its own?

    This is a test of the DATA, made before solving, and that is the point.
    The failure it guards against is hard to detect afterwards from the
    solution: when a scaled coefficient sinks below the solver's feasibility
    tolerance the constraint stops binding, and the answer that comes back can
    be an overstated score that is feasible, with zero violation, and wrong -
    under input orientation ``theta = 1`` with the DMU as its own and only peer
    is always such a point.  A primal feasibility check cannot catch it; a
    dual check can, but does not repair it.  See ``docs/validation.md``
    sections 4 and 9.

    So the decision has to be made from the data: if rescaling puts any
    non-zero entry within ``margin`` of the tolerance, or the same factor above
    1, the problem is close enough to the solver's limits that one solve is not
    enough and the caller applies the candidate rule of ``pleap.core.lp``.
    Well-behaved data is reported as safe, so the common case pays nothing.

    The margin is four decades: entries outside [1e-3, 1e3] after centring.
    Constraints vanish only at the primal tolerance, but the solver can stop
    at an overstated score well before that, at its dual tolerance.  In
    ``docs/scaling_stress.py`` a single solve first missed the exact optimum
    (by 3e-5) on a column spanning 7.5 decades, whose centred entries reached
    5.8e-4 and 1.8e4; none missed on the 100 draws spanning 4 to 7 decades.
    The two-decade margin of v1.12's first form did not flag that column.
    """
    A = np.asarray(A, float)
    if A.ndim != 2:
        raise ValueError("scaling_is_risky expects a 2-D array")
    s = column_scales(A) if scales is None else np.asarray(scales, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.abs(A / s)
    nz = Z[np.isfinite(Z) & (Z > 0)]
    if nz.size == 0:
        return True                      # degenerate: never trust one solve
    return bool(nz.min() < tol * margin or nz.max() > 1.0 / (tol * margin))
