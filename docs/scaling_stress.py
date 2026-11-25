"""How often does each scaling rule miss the optimum on ill-conditioned columns?

Draws 100 synthetic data sets with one input column spanning 8 to 14 decades
(the structure of tests/data/scaling_case.csv, varied in seed, spread, size and
number of inputs), computes every unit's optimum in exact rational arithmetic
(docs/external_reference/exact_dea.py), and counts the unit-model combinations
(CCR and BCC, both orientations) on which each rule is off by more than 1e-6,
and those on which it returns no score at all:

  * arithmetic-mean scaling, one solve   - PLEAP v1.0.9-v1.1.0;
  * geometric-mean scaling, one solve;
  * geometric-mean scaling cross-checked against the raw data, best admissible
    of the two, on data its test flagged - the first form of the v1.12 fix;
  * PLEAP v1.12 as shipped               - pleap.core.dea.dea, which adds a
    third candidate at tightened tolerances (pleap.core.lp).

A second part sweeps the spread of the column from 4 to 10 decades (25 draws
each) and records where a single geometric-mean solve starts to miss, and
which draws ``scaling_is_risky`` flags; that is the evidence behind its margin.

    python docs/scaling_stress.py        # about 10 minutes; writes scaling_stress.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "docs" / "external_reference"))

from exact_dea import exact_efficiency                      # noqa: E402
from pleap.core.dea import dea                              # noqa: E402
from pleap.core.lp import envelopment                       # noqa: E402
from pleap.core.scaling import (column_scales, feasible,    # noqa: E402
                                scaling_is_risky)

CONFIGS = {
    "30 units, 2 inputs, 11.5 decades": dict(n=30, hi=1e13),
    "30 units, 2 inputs, 14 decades": dict(n=30, hi=1e15),
    "30 units, 2 inputs, 8 decades": dict(n=30, hi=3e9),
    "60 units, 3 inputs, 11.5 decades": dict(n=60, hi=1e13, third=True),
}
SEEDS = range(25)
TOL = 1e-6


def draw(seed, n=30, hi=1e13, third=False):
    """The generator of tests/data/make_synthetic_data.py::make_scaling_case,
    with the spread, the size and an optional third input as parameters."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(10.0, 400.0, n)
    x2 = np.exp(rng.uniform(np.log(30.0), np.log(hi), n))
    cols = [x1, x2]
    if third:
        cols.append(rng.uniform(1e5, 1e9, n))
    y = x1 ** 0.5 * x2 ** 0.1 * rng.uniform(0.4, 1.0, n)
    if third:
        y = y * cols[2] ** 0.05
    return np.column_stack(cols), y.reshape(-1, 1)


def one_solve(X, Y, sx, sy, model, orient):
    """A single solve on X/sx, Y/sy, with no check: the pre-v1.12 behaviour."""
    out = np.full(len(X), np.nan)
    for k in range(len(X)):
        r = envelopment(X[k] / sx, Y[k] / sy, X / sx, Y / sy, model, orient)
        if r is not None:
            out[k] = r[0] if orient == "input" else 1.0 / r[0]
    return out


def two_candidates(X, Y, model, orient):
    """The first form of the fix: on data its narrower test flagged (entries
    outside [1e-5, 1e5] after centring), scaled and raw solves, best
    admissible; otherwise one scaled solve."""
    sx, sy = column_scales(X), column_scales(Y)
    flagged = (scaling_is_risky(X, sx, margin=100.0)
               or scaling_is_risky(Y, sy, margin=100.0))
    out = np.full(len(X), np.nan)
    for k in range(len(X)):
        tries = [envelopment(X[k] / sx, Y[k] / sy, X / sx, Y / sy, model, orient)]
        if flagged:
            tries.append(envelopment(X[k], Y[k], X, Y, model, orient))
        ok = [r[0] for r in tries if r is not None
              and feasible(X[k], Y[k], X, Y, r[1], r[0], model, orient)]
        if not ok:
            ok = [r[0] for r in tries if r is not None]
        v = min(ok) if orient == "input" else max(ok)
        out[k] = v if orient == "input" else 1.0 / v
    return out


RULES = ["arithmetic mean, one solve", "geometric mean, one solve",
         "two candidates (first form of the fix)", "PLEAP v1.12"]


def main():
    t0 = time.time()
    result = {"tolerance": TOL, "seeds": list(SEEDS), "configs": {}}
    total = {r: {"cases": 0, "failed": 0, "wrong": 0, "max_gap": 0.0} for r in RULES}
    for name, kw in CONFIGS.items():
        tally = {r: {"cases": 0, "failed": 0, "wrong": 0, "max_gap": 0.0} for r in RULES}
        for seed in SEEDS:
            X, Y = draw(seed, **kw)
            am_x, am_y = X.mean(axis=0), Y.mean(axis=0)
            gm_x, gm_y = column_scales(X), column_scales(Y)
            for model in ("CCR", "BCC"):
                for orient in ("input", "output"):
                    exact = np.array([float(v) for v in exact_efficiency(
                        X.tolist(), Y.tolist(), model, orient)])
                    got = {
                        RULES[0]: one_solve(X, Y, am_x, am_y, model, orient),
                        RULES[1]: one_solve(X, Y, gm_x, gm_y, model, orient),
                        RULES[2]: two_candidates(X, Y, model, orient),
                        RULES[3]: dea(X, Y, model=model, orient=orient)["efficiency"],
                    }
                    for rule, v in got.items():
                        ok = np.isfinite(v)
                        gap = np.abs(v[ok] - exact[ok])
                        for t in (tally[rule], total[rule]):
                            t["cases"] += len(v)
                            t["failed"] += int((~ok).sum())
                            t["wrong"] += int((gap > TOL).sum())
                            if gap.size:
                                t["max_gap"] = max(t["max_gap"], float(gap.max()))
        result["configs"][name] = tally
        print(name, {r: (v["wrong"], v["failed"], f'{v["max_gap"]:.2e}') for r, v in tally.items()},
              f"({time.time() - t0:.0f}s)", flush=True)
    result["total"] = total

    # Part 2: where does a single solve start to miss?
    sweep = {}
    for decades in range(4, 11):
        rec = {"draws": 0, "flagged": 0, "single_wrong": 0, "single_max_gap": 0.0,
               "pleap_wrong": 0, "pleap_max_gap": 0.0, "misses": []}
        for seed in SEEDS:
            X, Y = draw(seed, n=30, hi=30.0 * 10 ** decades)
            gm_x, gm_y = column_scales(X), column_scales(Y)
            rec["draws"] += 1
            rec["flagged"] += int(scaling_is_risky(X, gm_x) or scaling_is_risky(Y, gm_y))
            for model in ("CCR", "BCC"):
                for orient in ("input", "output"):
                    exact = np.array([float(v) for v in exact_efficiency(
                        X.tolist(), Y.tolist(), model, orient)])
                    single = one_solve(X, Y, gm_x, gm_y, model, orient)
                    pleap = dea(X, Y, model=model, orient=orient)["efficiency"]
                    for key, v in (("single", single), ("pleap", pleap)):
                        gap = np.abs(v - exact)
                        rec[f"{key}_wrong"] += int((gap > TOL).sum())
                        rec[f"{key}_max_gap"] = max(rec[f"{key}_max_gap"], float(np.nanmax(gap)))
                    gap = np.abs(single - exact)
                    for k in np.where(gap > TOL)[0]:
                        Z = X / gm_x
                        rec["misses"].append({
                            "seed": seed, "model": model, "orient": orient,
                            "unit": int(k) + 1, "gap": float(gap[k]),
                            "spread_decades": float(np.log10(X[:, 1].max() / X[:, 1].min())),
                            "scaled_min": float(Z.min()), "scaled_max": float(Z.max())})
        sweep[f"{decades} decades"] = rec
        print(f"{decades} decades:", {k: v for k, v in rec.items() if k != "misses"},
              rec["misses"], f"({time.time() - t0:.0f}s)", flush=True)
    result["sweep"] = sweep
    out = ROOT / "docs" / "scaling_stress.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(total, indent=2))


if __name__ == "__main__":
    main()
