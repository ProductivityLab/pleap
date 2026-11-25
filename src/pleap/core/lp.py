"""
The radial envelopment linear program, shared by every score in PLEAP.

DEA, super-efficiency, returns-to-scale, the Malmquist distances and the
bootstrap all evaluate a point (x_k, y_k) against a reference technology
spanned by (X, Y).  They share this one formulation and this one rule for
deciding which solve to trust.

Formulation.  Input orientation:

    min  theta
    s.t. sum_j lam_j x_ji <= theta x_ki     i = 1..m
         sum_j lam_j y_jr >=       y_kr     r = 1..s
         sum_j lam_j = 1                    (BCC / VRS only)
         lam >= 0

Output orientation maximises phi with the inequalities exchanged.  The value
returned is theta or phi itself; callers convert to the Farrell scale.

Which solve to trust.  The columns are rescaled before solving because the
solver is not unit invariant (pleap.core.scaling).  On data that
``scaling_is_risky`` flags, one solve is not enough, and the program is solved
three times:

    1. on the rescaled data, at HiGHS's default tolerances (1e-7);
    2. on the raw data, at the default tolerances;
    3. on the rescaled data, with the primal and dual feasibility tolerances
       tightened to 1e-10.

Each candidate is checked against the constraints in the ORIGINAL units
(``feasible``), and among the admissible ones the smaller theta, or the larger
phi, wins: every admissible point bounds the optimum, so a further candidate
can only improve the answer.

The third solve was added when an exact reference exposed a case the first
two both missed.  On the public synthetic data set of tests/data/
scaling_case.csv (one input column spanning 11.3 decades), unit 20 under BCC
input orientation came back as theta = 1 from both the scaled and the raw
solve, while the optimum computed in exact rational arithmetic is 0.977706.
HiGHS stopped there because, at its default dual feasibility tolerance, the
improving direction looked like rounding.  At 1e-10 it does not.  On 15,000
unit-model combinations drawn from columns spanning 8 to 14 decades the
three-candidate rule agrees with the exact optimum to 3.5e-8; the
two-candidate rule, with its narrower test of the data, missed 32 of them,
by up to 0.51 (docs/scaling_stress.py, docs/validation.md section 4).

The rule picks the best admissible point among the candidates; it does not
certify optimality.  On data that ``scaling_is_risky`` does not flag, only the
first solve is run and the common case pays nothing.
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

from pleap.core.scaling import feasible

__all__ = ["TIGHT", "envelopment", "best_point"]

# The tolerances of the third candidate.  HiGHS's defaults are 1e-7.
TIGHT = {"primal_feasibility_tolerance": 1e-10,
         "dual_feasibility_tolerance": 1e-10}


def envelopment(xk, yk, Xref, Yref, model="BCC", orient="input",
                exclude=None, options=None):
    """Solve the envelopment LP of point (xk, yk) against (Xref, Yref).

    ``exclude`` is the index of a reference unit whose intensity weight is
    fixed at zero (the Andersen-Petersen super-efficiency program).
    ``options`` is passed to HiGHS.  Returns ``(value, lambdas)`` - theta for
    input orientation, phi for output orientation - or ``None`` when the
    solver reports no optimum (a VRS cross-period or super-efficiency program
    can be infeasible).
    """
    Xref = np.asarray(Xref, float); Yref = np.asarray(Yref, float)
    xk = np.asarray(xk, float); yk = np.asarray(yk, float)
    n, m = Xref.shape
    _, s = Yref.shape
    nv = 1 + n
    A = np.zeros((m + s, nv)); b = np.zeros(m + s); c = np.zeros(nv)

    if orient == "input":
        c[0] = 1.0                               # minimise theta
        for i in range(m):
            A[i, 0] = -xk[i]
            A[i, 1:] = Xref[:, i]                # sum lam x - theta x_k <= 0
        for r in range(s):
            A[m + r, 1:] = -Yref[:, r]
            b[m + r] = -yk[r]                    # -sum lam y <= -y_k
    else:
        c[0] = -1.0                              # maximise phi
        for i in range(m):
            A[i, 1:] = Xref[:, i]
            b[i] = xk[i]                         # sum lam x <= x_k
        for r in range(s):
            A[m + r, 0] = yk[r]
            A[m + r, 1:] = -Yref[:, r]           # phi y_k - sum lam y <= 0

    if model == "BCC":
        A_eq = np.zeros((1, nv)); A_eq[0, 1:] = 1.0
        b_eq = [1.0]
    else:
        A_eq = None; b_eq = None

    bounds = [(0, None)] * nv
    if exclude is not None:
        bounds[1 + int(exclude)] = (0, 0)        # the unit is not its own peer

    res = linprog(c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method="highs", options=dict(options or {}))
    if not res.success or res.x is None:
        return None
    # Intensity weights are bounded below by zero; a solver may return a tiny
    # negative value for one that is exactly zero.  Clamp, so the noise never
    # reaches a peer list or a results table.
    lam = np.clip(np.asarray(res.x[1:], float), 0.0, None)
    return float(res.x[0]), lam


def best_point(xk, yk, Xref, Yref, sx, sy, model="BCC", orient="input",
               risky=False, exclude=None):
    """Apply the candidate rule of the module docstring to one point.

    ``sx`` and ``sy`` are the column scales of ``Xref`` and ``Yref``;
    ``risky`` is the caller's ``scaling_is_risky`` verdict on the reference
    data.  Returns ``(value, lambdas, source)`` with ``source`` one of
    ``"scaled"``, ``"raw"`` or ``"tight"`` - the problem the answer came from,
    so that a caller solving a second phase can solve it on the same one - or
    ``(None, None, "scaled")`` when no solve produced an optimum.
    """
    Xref = np.asarray(Xref, float); Yref = np.asarray(Yref, float)
    xk = np.asarray(xk, float); yk = np.asarray(yk, float)
    sx = np.asarray(sx, float); sy = np.asarray(sy, float)
    scaled = (xk / sx, yk / sy, Xref / sx, Yref / sy)

    tries = [("scaled", envelopment(*scaled, model, orient, exclude))]
    if risky:
        tries.append(("raw", envelopment(xk, yk, Xref, Yref, model, orient,
                                         exclude)))
        tries.append(("tight", envelopment(*scaled, model, orient, exclude,
                                           options=TIGHT)))

    admissible = [(r[0], r[1], src) for src, r in tries
                  if r is not None
                  and feasible(xk, yk, Xref, Yref, r[1], r[0], model, orient)]
    if admissible:
        # input orientation minimises theta, output orientation maximises phi
        if orient == "input":
            return min(admissible, key=lambda t: t[0])
        return max(admissible, key=lambda t: t[0])

    # No solve produced an admissible point; report whatever exists rather
    # than failing outright, preferring the raw problem.
    found = dict(tries)
    for src in ("raw", "scaled", "tight"):
        r = found.get(src)
        if r is not None:
            return r[0], r[1], src
    return None, None, "scaled"
