"""60-digit maximum-likelihood references for the Tobit and truncated regressions.

Independent of PLEAP and of R: the log-likelihood is written out here and
evaluated in 60-digit arithmetic (mpmath); the maximum is found by a damped
Newton iteration in that precision, with the gradient and the Hessian from
central differences (steps 1e-25 and 1e-12), so truncation and rounding error
lie far below the digits compared.  Standard errors come from the inverse of
the observed information, on the scale (const, slopes, log sigma) that PLEAP
and R's survreg report.  Run from the repository root:

    python docs/external_reference/exact_mle.py               # writes tests/data/exact_mle.json
    python docs/external_reference/exact_mle.py PATH.csv y z1 z2 [tobit|trunc] LOWER UPPER
                                                              # one fit, printed (LOWER/UPPER may be "none")

The JSON holds, for the public data sets,

    example_trunc_1   example_bcc_input.csv, theta truncated above at 1 (Algorithm 1's sample)
    sd_trunc_01       sd.csv, vrs_i truncated at 0 and at 1 (R's truncreg has one truncation point)
    sd_tobit_01       sd.csv, vrs_i censored at 0 and at 1
    sd_tobit_0inf     sd.csv, vrs_i censored at 0 only
"""
import csv
import json
import sys
from pathlib import Path

import mpmath as mp

mp.mp.dps = 60
ROOT = Path(__file__).resolve().parents[2]
TOL = mp.mpf("1e-9")          # a score within 1e-9 of a bound is on it, as in PLEAP
LOG2PI = mp.log(2 * mp.pi)


def load(path, ycol, zcols):
    rows = list(csv.DictReader(open(path)))
    return [mp.mpf(r[ycol]) for r in rows], [[mp.mpf(r[c]) for c in zcols] for r in rows]


def make_ll(y, Z, kind, lower, upper):
    if kind == "trunc":          # only the observations strictly inside the bounds enter
        keep = [i for i, v in enumerate(y)
                if (lower is None or v > lower + TOL) and (upper is None or v < upper - TOL)]
        y, Z = [y[i] for i in keep], [Z[i] for i in keep]

    def ll(p):
        *b, g = p
        s = mp.e ** g
        tot = mp.mpf(0)
        for yy, zz in zip(y, Z):
            mu = b[0] + mp.fsum(bj * zj for bj, zj in zip(b[1:], zz))
            if kind == "tobit":
                if upper is not None and yy >= upper - TOL:
                    tot += mp.log(mp.ncdf((mu - upper) / s))
                    continue
                if lower is not None and yy <= lower + TOL:
                    tot += mp.log(mp.ncdf((lower - mu) / s))
                    continue
            u = (yy - mu) / s
            tot += -g - LOG2PI / 2 - u * u / 2
            if kind == "trunc":
                hi = mp.ncdf((upper - mu) / s) if upper is not None else mp.mpf(1)
                lo = mp.ncdf((lower - mu) / s) if lower is not None else mp.mpf(0)
                tot -= mp.log(hi - lo)
        return tot
    return ll, len(y)


def grad(ll, p, h=mp.mpf("1e-25")):
    k = len(p)
    return [(ll([p[j] + (h if j == i else 0) for j in range(k)])
             - ll([p[j] - (h if j == i else 0) for j in range(k)])) / (2 * h) for i in range(k)]


def hess(ll, p, h=mp.mpf("1e-12")):
    k = len(p)
    H = mp.matrix(k, k)
    for i in range(k):
        for j in range(i, k):
            def f(di, dj):
                q = list(p); q[i] += di; q[j] += dj
                return ll(q)
            H[i, j] = H[j, i] = (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4 * h * h)
    return H


def ols_start(y, Z):
    n, k = len(y), len(Z[0]) + 1
    A = mp.matrix([[1] + z for z in Z])
    b = mp.lu_solve(A.T * A, A.T * mp.matrix(y))
    r = [y[i] - mp.fsum(A[i, j] * b[j] for j in range(k)) for i in range(n)]
    s = mp.sqrt(mp.fsum(v * v for v in r) / (n - k))
    return [b[j] for j in range(k)] + [mp.log(s)]


def fit(y, Z, kind, lower, upper):
    ll, n = make_ll(y, Z, kind, lower, upper)
    p = ols_start(y, Z)
    f0 = ll(p)
    for it in range(80):
        g = mp.matrix(grad(ll, p)); H = hess(ll, p)
        try:
            mp.cholesky(-H)                       # Newton only where the model is concave
            step = [-v for v in mp.lu_solve(H, g)]
        except ValueError:
            step = [v * mp.mpf("1e-3") for v in g]
        t = mp.mpf(1)
        while True:
            q = [p[i] + t * step[i] for i in range(len(p))]
            fq = ll(q)
            if fq >= f0 or t < mp.mpf("1e-30"):
                break
            t /= 2
        size = mp.norm(mp.matrix(step)) * t
        p, f0 = q, fq
        if size < mp.mpf("1e-40"):
            break
    H = hess(ll, p)
    V = -mp.inverse(H)
    k = len(p)
    out = dict(n=n, iterations=it + 1, gradient_max=mp.nstr(max(abs(v) for v in grad(ll, p)), 5),
               coef=[mp.nstr(v, 20) for v in p[:-1]], log_sigma=mp.nstr(p[-1], 20),
               sigma=mp.nstr(mp.e ** p[-1], 20), loglik=mp.nstr(f0, 20),
               se=[mp.nstr(mp.sqrt(V[i, i]), 20) for i in range(k)])
    if kind == "tobit":
        idx = list(range(1, k - 1))
        Vs = mp.matrix([[V[i, j] for j in idx] for i in idx]); w = mp.matrix([p[i] for i in idx])
        out["wald"] = mp.nstr((w.T * mp.inverse(Vs) * w)[0], 20)
    return out


def bound(s):
    return None if s.lower() in ("none", "inf", "-inf") else mp.mpf(s)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path, ycol, z1, z2, kind, lo, hi = sys.argv[1:8]
        y, Z = load(path, ycol, [z1, z2])
        print(json.dumps(fit(y, Z, kind, bound(lo), bound(hi)), indent=1))
        sys.exit()
    D = ROOT / "tests" / "data"
    res = {}
    y, Z = load(D / "example_bcc_input.csv", "theta", ["z1", "z2"])
    res["example_trunc_1"] = fit(y, Z, "trunc", None, mp.mpf(1))
    y, Z = load(D / "sd.csv", "vrs_i", ["x1", "x2"])
    res["sd_trunc_01"] = fit(y, Z, "trunc", mp.mpf(0), mp.mpf(1))
    res["sd_tobit_01"] = fit(y, Z, "tobit", mp.mpf(0), mp.mpf(1))
    res["sd_tobit_0inf"] = fit(y, Z, "tobit", mp.mpf(0), None)
    res["mpmath"] = mp.__version__
    (D / "exact_mle.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
