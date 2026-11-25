"""Radial DEA scores in exact rational arithmetic, as a reference without tolerances.

Every floating-point LP solver, PLEAP's included, decides optimality and
feasibility against tolerances.  On data near the solver's limits that is where
answers go wrong, and a second floating-point program shares the weakness.  This
script does not: it reads the data as the double-precision numbers they are,
converts each to an exact fraction, and solves the envelopment program of every
unit with a primal simplex in rational arithmetic (Bland's rule, so it cannot
cycle).  The optimum it returns is the exact optimum of the program posed on
those doubles; the only rounding is the final conversion to a printed number.

It is slow and meant for small reference sets, which is all it is used for:
the 30-unit case of tests/data/scaling_case.csv and the random stress draws of
docs/scaling_stress.py.

    python docs/external_reference/exact_dea.py tests/data/scaling_case.csv \\
        x1,x2 y tests/data/scaling_case_exact.csv

writes, for CCR and BCC under both orientations, the Farrell score of every unit
(theta for input orientation, 1/phi for output orientation): as a double, and
as a 40-digit decimal of the exact value.
"""
import sys
from decimal import Decimal, getcontext
from fractions import Fraction as F

import pandas as pd


def _simplex(A, b, c, basis):
    """min c.z s.t. A z = b, z >= 0, from a feasible basis; all Fractions."""
    R, N = len(A), len(A[0])
    T = [row[:] + [b[i]] for i, row in enumerate(A)]
    for r, j in enumerate(basis):                 # canonical form for the basis
        p = next(i for i in range(r, R) if T[i][j] != 0)
        T[r], T[p] = T[p], T[r]
        piv = T[r][j]
        T[r] = [v / piv for v in T[r]]
        for i in range(R):
            if i != r and T[i][j] != 0:
                f = T[i][j]
                T[i] = [a - f * bb for a, bb in zip(T[i], T[r])]
    basis = list(basis)
    while True:
        cb = [c[j] for j in basis]
        enter = None
        for j in range(N):                        # Bland: smallest improving index
            if j in basis:
                continue
            if c[j] - sum(cb[i] * T[i][j] for i in range(R)) < 0:
                enter = j
                break
        if enter is None:
            z = [F(0)] * N
            for i, j in enumerate(basis):
                z[j] = T[i][-1]
            return sum(c[j] * z[j] for j in range(N)), z
        rows = [(T[i][-1] / T[i][enter], basis[i], i)
                for i in range(R) if T[i][enter] > 0]
        if not rows:
            raise ValueError("unbounded")
        _, _, r = min(rows)                       # ties: smallest leaving index
        piv = T[r][enter]
        T[r] = [v / piv for v in T[r]]
        for i in range(R):
            if i != r and T[i][enter] != 0:
                f = T[i][enter]
                T[i] = [a - f * bb for a, bb in zip(T[i], T[r])]
        basis[r] = enter


def exact_score(X, Y, k, model="BCC", orient="input"):
    """Exact theta (input) or phi (output) of unit k; X, Y nested lists of floats.

    Starts from the feasible basis theta = 1 (phi = 1), lambda_k = 1, which
    exists for every unit, so no phase 1 is needed.
    """
    n, m, s = len(X), len(X[0]), len(Y[0])
    Xf = [[F(v) for v in row] for row in X]
    Yf = [[F(v) for v in row] for row in Y]
    vrs = model == "BCC"
    # z = [theta|phi, lambda_1..lambda_n, input slacks (m), output surpluses (s)]
    N = 1 + n + m + s
    S0, T0 = 1 + n, 1 + n + m
    A, b = [], []
    if orient == "input":
        for i in range(m):        # sum lam x_i - theta x_ik + slack_i = 0
            row = [F(0)] * N; row[0] = -Xf[k][i]
            for j in range(n):
                row[1 + j] = Xf[j][i]
            row[S0 + i] = F(1); A.append(row); b.append(F(0))
        for r in range(s):        # sum lam y_r - surplus_r = y_rk
            row = [F(0)] * N
            for j in range(n):
                row[1 + j] = Yf[j][r]
            row[T0 + r] = F(-1); A.append(row); b.append(Yf[k][r])
        c = [F(0)] * N; c[0] = F(1)
        basis = [0, 1 + k] + [S0 + i for i in range(1, m)]
        basis += [T0 + r for r in range(s)] if vrs else [T0 + r for r in range(1, s)]
    else:
        for i in range(m):        # sum lam x_i + slack_i = x_ik
            row = [F(0)] * N
            for j in range(n):
                row[1 + j] = Xf[j][i]
            row[S0 + i] = F(1); A.append(row); b.append(Xf[k][i])
        for r in range(s):        # sum lam y_r - phi y_rk - surplus_r = 0
            row = [F(0)] * N; row[0] = -Yf[k][r]
            for j in range(n):
                row[1 + j] = Yf[j][r]
            row[T0 + r] = F(-1); A.append(row); b.append(F(0))
        c = [F(0)] * N; c[0] = F(-1)
        basis = [0, 1 + k] + ([S0 + i for i in range(m)] if vrs
                              else [S0 + i for i in range(1, m)])
        basis += [T0 + r for r in range(1, s)]
    if vrs:
        row = [F(0)] * N
        for j in range(n):
            row[1 + j] = F(1)
        A.append(row); b.append(F(1))
    obj, _ = _simplex(A, b, c, basis)
    return obj if orient == "input" else -obj


def exact_efficiency(X, Y, model="BCC", orient="input"):
    """Exact Farrell scores of every unit, as Fractions."""
    out = []
    for k in range(len(X)):
        v = exact_score(X, Y, k, model, orient)
        out.append(v if orient == "input" else 1 / v)
    return out


def main(path, inputs, outputs, dest):
    df = pd.read_csv(path, float_precision="round_trip")
    X = df[inputs].to_numpy(float).tolist()
    Y = df[outputs].to_numpy(float).tolist()
    getcontext().prec = 40
    rows = []
    for model in ("CCR", "BCC"):
        for orient in ("input", "output"):
            for dmu, v in zip(df["dmu"], exact_efficiency(X, Y, model, orient)):
                rows.append({"model": model, "orient": orient, "dmu": dmu,
                             "efficiency": float(v),
                             "exact_40": str(Decimal(v.numerator) / Decimal(v.denominator))})
    pd.DataFrame(rows).to_csv(dest, index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","), sys.argv[4])
