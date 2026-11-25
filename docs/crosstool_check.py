"""Is the scaling fault a PLEAP defect or a hazard any DEA tool can hit?

Puts the public case of docs/validation.md section 4 - tests/data/scaling_case.csv,
one input column spanning 11.3 decades - through several floating-point
solvers, CCR input orientation, and compares each with the optimum computed in
exact rational arithmetic (tests/data/scaling_case_exact.csv), the one
reference that shares no tolerance with any of them.

pyDEA is optional and is compared only if it imports; it needs PuLP before 4.0
(see crosstool_pydea.py).  R's Benchmarking is in
tests/data/scaling_case_benchmarking.csv.

    python docs/crosstool_check.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pleap.core.dea import dea
from pleap.core.lp import envelopment
from pleap.core.scaling import column_scales

DATA = Path(__file__).resolve().parents[1] / "tests" / "data"


def main():
    d = pd.read_csv(DATA / "scaling_case.csv", float_precision="round_trip")
    X = d[["x1", "x2"]].to_numpy(float); Y = d[["y"]].to_numpy(float)
    ex = pd.read_csv(DATA / "scaling_case_exact.csv", float_precision="round_trip")
    be = pd.read_csv(DATA / "scaling_case_benchmarking.csv", float_precision="round_trip")
    sel = lambda df: df.loc[(df.model == "CCR") & (df.orient == "input"),  # noqa: E731
                            "efficiency"].to_numpy(float)
    truth = sel(ex)
    print(f"{len(X)} units; input 2 spans "
          f"{np.log10(X[:, 1].max() / X[:, 1].min()):.1f} decades")

    def one(sx, sy):
        out = np.full(len(X), np.nan)
        for k in range(len(X)):
            r = envelopment(X[k] / sx, Y[k] / sy, X / sx, Y / sy, "CCR", "input")
            if r is not None:
                out[k] = r[0]
        return out

    got = {"no scaling": one(np.ones(2), np.ones(1)),
           "arithmetic mean, one solve": one(X.mean(axis=0), Y.mean(axis=0)),
           "geometric mean, one solve": one(column_scales(X), column_scales(Y)),
           "PLEAP v1.12": dea(X, Y, model="CCR", orient="input")["efficiency"],
           "R Benchmarking": sel(be)}
    try:
        from crosstool_pydea import pydea_ccr_input      # optional helper
        got["pyDEA"] = np.asarray(pydea_ccr_input(X, Y), float)
    except Exception as exc:                              # pragma: no cover
        print(f"(pyDEA not compared: {exc})")

    print(f"\n{'method':28s}{'max |gap|':>12s}{'mean |gap|':>12s}{'# > 1e-4':>10s}")
    for name, v in got.items():
        g = np.abs(v - truth)
        print(f"{name:28s}{g.max():12.3e}{g.mean():12.3e}{int((g > 1e-4).sum()):10d}")


if __name__ == "__main__":
    main()
