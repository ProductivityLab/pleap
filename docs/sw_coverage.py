"""Monte Carlo coverage of the Simar-Wilson Algorithm 1 bootstrap intervals.

    python docs/sw_coverage.py N [REPS] [L]

Data are drawn from the model Algorithm 1 assumes: theta = z'beta + eps with
eps ~ N(0, sigma^2) truncated so that theta < 1 (Farrell scale, truncated from
above), beta = (0.55, 0.20, -0.10), sigma = 0.12, z ~ U(0, 1)^2.  Each
replication fits Algorithm 1 with L bootstrap draws and records whether the
95% basic bootstrap interval covers the true coefficient; the naive
truncated-regression interval (estimate +/- 1.96 s.e.) is recorded beside it.

The design has no estimated frontier, so it checks that the implementation is
sound (coverage near nominal, truth recovered), not that Algorithm 1 beats the
naive interval: here the naive interval is already valid.  See
docs/validation.md section 10.  Writes docs/sw_coverage_n<N>.json.
"""
import json
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, "src")
from pleap.core.secondstage import simarwilson_algorithm1  # noqa: E402

TRUE_B = np.array([0.55, 0.20, -0.10])
TRUE_S = 0.12
N = int(sys.argv[1])
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
L = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
rng = np.random.default_rng(20261112 + N)

cover = np.zeros(3)
naive = np.zeros(3)
width = np.zeros(3)
done = failed = 0
t0 = time.time()
for r in range(REPS):
    Z = rng.uniform(0.0, 1.0, (N, 2))
    mu = np.column_stack([np.ones(N), Z]) @ TRUE_B
    th = stats.truncnorm.rvs(-np.inf, (1.0 - mu) / TRUE_S, loc=mu,
                             scale=TRUE_S, random_state=rng)
    try:
        f = simarwilson_algorithm1(th, Z, L=L, seed=int(rng.integers(1 << 31)),
                                   names=["z1", "z2"])
    except Exception:
        failed += 1
        continue
    lo = np.asarray(f["summary"].attrs["ci_low"], float)
    hi = np.asarray(f["summary"].attrs["ci_high"], float)
    cover += (lo <= TRUE_B) & (TRUE_B <= hi)
    width += hi - lo
    b = np.asarray(f["summary"].attrs["coef_raw"], float)
    # the naive interval uses the truncated-regression standard errors, which
    # Algorithm 1 does not report; refit cheaply from the same point estimates
    from pleap.core.secondstage import truncated_regression
    t = truncated_regression(th, Z, lower=None, upper=1.0, names=["z1", "z2"])
    s = np.asarray(t["summary"].attrs["se_raw"], float)[:-1]
    naive += (b - 1.96 * s <= TRUE_B) & (TRUE_B <= b + 1.96 * s)
    done += 1
    if done % 20 == 0:
        print(f"n={N}: {done}/{REPS} ({time.time() - t0:.0f}s)", flush=True)

mcse = 100 * np.sqrt(0.95 * 0.05 / done)
out = dict(n=N, reps_requested=REPS, reps_used=done, failed=failed, L=L,
           nominal=0.95, mc_standard_error_points=round(float(mcse), 2),
           coverage_algorithm1={k: round(100 * v / done, 1)
                                for k, v in zip(["const", "z1", "z2"], cover)},
           coverage_naive={k: round(100 * v / done, 1)
                           for k, v in zip(["const", "z1", "z2"], naive)},
           mean_interval_width={k: round(float(v / done), 4)
                                for k, v in zip(["const", "z1", "z2"], width)},
           seconds=round(time.time() - t0, 1))
json.dump(out, open(f"docs/sw_coverage_n{N}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
