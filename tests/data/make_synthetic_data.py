"""Synthetic data for the example and the public reference tests.

No value in the files this script writes comes from a survey or any other
real source: every number is drawn from the data-generating processes below,
with fixed seeds, so the files can be rebuilt bit for bit.  Run from the
repository root:

    python tests/data/make_synthetic_data.py

It writes

    examples/sample_data.xlsx        the example of the paper's Section 3
    tests/data/example_panel.csv     the same panel, for R
    tests/data/sa_mpi.csv            data set SA, 60 units x 2020, 2022, 2024
    tests/data/sa_dea.csv            the 2024 cross-section of SA
    tests/data/sd.csv                data set SD, 400 units, one period

SA and SD have the layout of the validation data sets A and D described in
docs/validation.md.  A and D come from the Korean Innovation Survey (KIS) of
the Science and Technology Policy Institute (STEPI), whose terms do not allow
them to be redistributed; SA and SD let the tests written for them run on
public data, against reference values from R (docs/external_reference/).

The technology, shared by the three data sets
--------------------------------------------
Inputs enter through a Cobb-Douglas aggregate of degree one,
v = x1^a * x2^(1 - a), and the frontier output is

    q*(x) = A_t * (v - v0)^rho        (v > v0),

concave in x, so the technology is convex.  The fixed requirement v0 gives
increasing returns below the most productive scale size v0 / (1 - rho) and
decreasing returns above it, so every returns-to-scale class occurs.  With two
outputs, q* bounds the Euclidean norm of the output vector, which keeps the
output set convex.

Each unit draws a frontier input aggregate v, an input mix and (with two
outputs) an output mix, produces the frontier output, and then uses 1/theta
times the frontier inputs: theta is its input-oriented (Farrell) efficiency
against the true frontier.  DEA estimates that frontier from the sample, so
the DEA scores are not theta; they are biased upward.

theta follows a normal regression on the environmental variables, truncated
to [theta_min, 1]:

    theta = b0 + b1 * z1 + b2 * z2 + group effect + unit effect + e,

with a group-specific standard deviation of e.  The parameters are listed with
each data set below.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data"


def _truncated_normal(rng, mean, sd, lo, hi):
    """Draw from N(mean, sd^2) restricted to [lo, hi] by rejection."""
    mean = np.asarray(mean, float)
    sd = np.broadcast_to(np.asarray(sd, float), mean.shape)
    out = np.full(mean.shape, np.nan)
    todo = np.ones(mean.shape, bool)
    while todo.any():
        draw = rng.normal(mean[todo], sd[todo])
        ok = (draw >= lo) & (draw <= hi)
        idx = np.flatnonzero(todo)
        out[idx[ok]] = draw[ok]
        todo[idx[ok]] = False
    return out


def _frontier_inputs(v, m, a):
    """Inputs with aggregate v and log-ratio m = ln(x1 / x2)."""
    return v * np.exp((1 - a) * m), v * np.exp(-a * m)


def _bcc_input_scores(X, Y):
    """BCC input-oriented scores with SciPy's HiGHS, independently of PLEAP."""
    n = len(X)
    out = np.empty(n)
    for k in range(n):
        c = np.r_[1.0, np.zeros(n)]
        A_ub = np.vstack([np.c_[-X[k][:, None], X.T],          # X'lam <= theta x_k
                          np.c_[np.zeros((Y.shape[1], 1)), -Y.T]])   # Y'lam >= y_k
        b_ub = np.r_[np.zeros(X.shape[1]), -Y[k]]
        A_eq = np.r_[0.0, np.ones(n)][None, :]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                      bounds=[(None, None)] + [(0, None)] * n, method="highs")
        assert res.status == 0, res.message
        out[k] = res.x[0]
    return out


# --------------------------------------------------------------------------- #
#  The example: 60 units x 2021-2023, two inputs, two outputs                 #
# --------------------------------------------------------------------------- #
EXAMPLE = dict(
    seed=20211, periods=(2021, 2022, 2023), n=60,
    a=0.5, v0=2.0, rho=0.75,                 # most productive scale size v = 8
    level=(1.00, 1.03, 1.08),                # A_t: the frontier moves out over time
    log_excess=(np.log(6.0), 0.6),           # ln(v - v0): unit scale, persistent
    scale_growth=0.02, scale_noise=0.08,     # per year; period-to-period
    mix_in_sd=0.2, mix_out=(0.55, 1.02),     # sd of ln(x1/x2); output angle, radians
    b0=0.70, b1=0.10, b2=0.0,                # z1 raises efficiency, z2 does not
    group=dict(A=(0.06, 0.05), B=(0.0, 0.10), C=(-0.10, 0.18)),   # effect, sd of e
    unit_sd=0.03, theta_min=0.25,
    column_scale=dict(x1=6.0, x2=4.0, y1=25.0, y2=15.0),   # units of measurement
)


def make_example():
    p = EXAMPLE
    rng = np.random.default_rng(p["seed"])
    n, T = p["n"], len(p["periods"])
    groups = np.repeat(["A", "B", "C"], n // 3)
    lnexc_unit = rng.normal(*p["log_excess"], n)
    m_unit = rng.normal(0.0, p["mix_in_sd"], n)
    w_unit = rng.uniform(*p["mix_out"], n)       # output mix angle, radians
    zeta = rng.normal(0.0, 1.0, n)               # persistent part of z1
    u = rng.normal(0.0, p["unit_sd"], n)
    rows = []
    for t, year in enumerate(p["periods"]):
        v = p["v0"] + np.exp(lnexc_unit + p["scale_growth"] * t + rng.normal(0, p["scale_noise"], n))
        m = m_unit + rng.normal(0, 0.05, n)
        w = np.clip(w_unit + rng.normal(0, 0.05, n), 0.1, 1.47)
        z1 = 0.8 * zeta + 0.6 * rng.normal(0, 1, n)
        z2 = rng.uniform(0.0, 1.0, n)
        eff = np.array([p["group"][g][0] for g in groups])
        sd = np.array([p["group"][g][1] for g in groups])
        theta = _truncated_normal(rng, p["b0"] + p["b1"] * z1 + p["b2"] * z2 + eff + u,
                                  sd, p["theta_min"], 1.0)
        q = p["level"][t] * (v - p["v0"]) ** p["rho"]
        x1s, x2s = _frontier_inputs(v, m, p["a"])
        cs = p["column_scale"]
        rows.append(pd.DataFrame({
            "DMU_name": [f"DMU{i + 1:02d}" for i in range(n)], "Period": year, "Group": groups,
            "x1": np.round(cs["x1"] * x1s / theta, 1), "x2": np.round(cs["x2"] * x2s / theta, 1),
            "y1": np.round(cs["y1"] * q * np.cos(w), 1), "y2": np.round(cs["y2"] * q * np.sin(w), 1),
            "z1": np.round(z1, 3), "z2": np.round(z2, 3)}))
    d = pd.concat(rows, ignore_index=True)
    return d.sort_values(["Period", "DMU_name"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  SA: 60 units x 2020, 2022, 2024, two inputs, one output (the layout of A)  #
# --------------------------------------------------------------------------- #
SA = dict(
    seed=20202, periods=(2020, 2022, 2024), n=60,
    a=0.6, v0=2.0, rho=0.7,                  # most productive scale size v = 6.7
    level=(1.00, 1.06, 1.10),
    log_excess=(np.log(5.0), 0.8), scale_growth=0.04, scale_noise=0.12,
    b0=0.55, b1=0.006, b2=0.0,               # per unit of z1 - 50 and z2 - 20
    group=dict(g1=(0.0, 0.20), g2=(-0.05, 0.22)),
    unit_sd=0.08, theta_min=0.05,
    column_scale=dict(i1=40.0, i2=3.0, o1=500.0),
)


def make_sa():
    p = SA
    rng = np.random.default_rng(p["seed"])
    n = p["n"]
    g2 = np.repeat([1, 2], n // 2)
    g4 = np.tile(np.repeat([1, 2, 3, 4], n // 4), 1)
    rng.shuffle(g4)
    lnexc_unit = rng.normal(*p["log_excess"], n)
    m_unit = rng.normal(0.0, 0.5, n)
    u = rng.normal(0.0, p["unit_sd"], n)
    z1_unit = rng.normal(50.0, 6.0, n)
    rows = []
    for t, year in enumerate(p["periods"]):
        v = p["v0"] + np.exp(lnexc_unit + p["scale_growth"] * t + rng.normal(0, p["scale_noise"], n))
        m = m_unit + rng.normal(0, 0.1, n)
        z1 = z1_unit + rng.normal(0, 3.0, n)
        z2 = rng.normal(20.0, 5.0, n)
        eff = np.where(g2 == 1, p["group"]["g1"][0], p["group"]["g2"][0])
        sd = np.where(g2 == 1, p["group"]["g1"][1], p["group"]["g2"][1])
        theta = _truncated_normal(rng, p["b0"] + p["b1"] * (z1 - 50) + p["b2"] * (z2 - 20) + eff + u,
                                  sd, p["theta_min"], 1.0)
        q = p["level"][t] * (v - p["v0"]) ** p["rho"]
        x1s, x2s = _frontier_inputs(v, m, p["a"])
        cs = p["column_scale"]
        rows.append(pd.DataFrame({
            "dmu": [f"DMU{i + 1:02d}" for i in range(n)], "period": year,
            "i1": np.round(cs["i1"] * x1s / theta, 1), "i2": np.round(cs["i2"] * x2s / theta, 1),
            "o1": np.round(cs["o1"] * q, 1),
            "z1": np.round(z1, 4), "z2": np.round(z2, 4), "g2": g2, "g4": g4}))
    return pd.concat(rows, ignore_index=True).sort_values(["dmu", "period"], kind="stable") \
             .reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  SD: 400 units, two inputs, one output, two covariates, two groupings       #
#  (the layout of D)                                                          #
# --------------------------------------------------------------------------- #
SD = dict(
    seed=20183, n=400,
    a=0.7, v0=1.5, rho=0.8,                  # most productive scale size v = 7.5
    log_excess=(np.log(6.0), 0.9),
    b0=0.48, b1=0.03, b2=0.0,                # per point of x1 - 3 and x2 - 3
    group1=(0.0, 0.04),                      # effect of group1 = 1, 2
    group2=(0.0, -0.06, 0.08, 0.02),         # effect of group2 = 1..4
    sd2=(0.16, 0.20, 0.18, 0.24),            # sd of e by group2
    theta_min=0.03,
    column_scale=dict(i1=12.0, i2=0.8, o1=150.0),
)


def make_sd():
    p = SD
    rng = np.random.default_rng(p["seed"])
    n = p["n"]
    group1 = rng.choice([1, 2], size=n, p=[0.7, 0.3])
    group2 = rng.choice([1, 2, 3, 4], size=n, p=[0.3, 0.3, 0.25, 0.15])
    # survey-style covariates: means of four and of five five-point items
    x1 = np.round(np.clip(rng.normal(3.0, 0.8, n), 1, 5) * 4) / 4
    x2 = np.round(np.clip(rng.normal(3.2, 0.7, n), 1, 5) * 5) / 5
    v = p["v0"] + np.exp(rng.normal(*p["log_excess"], n))
    m = rng.normal(0.0, 0.6, n)
    mean = (p["b0"] + p["b1"] * (x1 - 3) + p["b2"] * (x2 - 3)
            + np.asarray(p["group1"])[group1 - 1] + np.asarray(p["group2"])[group2 - 1])
    theta = _truncated_normal(rng, mean, np.asarray(p["sd2"])[group2 - 1], p["theta_min"], 1.0)
    q = (v - p["v0"]) ** p["rho"]
    x1s, x2s = _frontier_inputs(v, m, p["a"])
    cs = p["column_scale"]
    d = pd.DataFrame({
        "dmu": [f"DMU{i + 1:03d}" for i in range(n)],
        "i1": np.round(cs["i1"] * x1s / theta, 0), "i2": np.round(cs["i2"] * x2s / theta, 2),
        "o1": np.round(cs["o1"] * q, 1), "x1": x1, "x2": x2,
        "group1": group1, "group2": group2})
    X = d[["i1", "i2"]].to_numpy(float); Y = d[["o1"]].to_numpy(float)
    # the BCC input-oriented score, rounded to eight decimals as an exported
    # column would be: the dependent variable the second-stage references read
    d["vrs_i"] = np.round(_bcc_input_scores(X, Y), 8)
    return d


def make_scaling_case(n=30, seed=11):
    """The numerical-fault case of the paper's Section 4.3 (``scaling_case.csv``).

    Thirty units, two inputs and one output.  Input ``x2`` is log-uniform over
    eleven decades, the structure on which column scaling by the arithmetic
    mean breaks: the mean sits near the top of the range and pushes the
    smallest entries below the solver's feasibility tolerance.  Nothing here
    comes from real data.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(10.0, 400.0, n)
    x2 = np.exp(rng.uniform(np.log(30.0), np.log(1e13), n))
    y = x1 ** 0.5 * x2 ** 0.1 * rng.uniform(0.4, 1.0, n)
    return pd.DataFrame({"dmu": [f"U{k:02d}" for k in range(1, n + 1)],
                         "x1": x1, "x2": x2, "y": y})


def main():
    ex = make_example()
    ex.to_excel(ROOT / "examples" / "sample_data.xlsx", index=False, sheet_name="Sheet1")
    ex.to_csv(DATA / "example_panel.csv", index=False)
    sa = make_sa()
    sa.to_csv(DATA / "sa_mpi.csv", index=False)
    sa[sa["period"] == 2024].to_csv(DATA / "sa_dea.csv", index=False)
    make_sd().to_csv(DATA / "sd.csv", index=False)
    make_scaling_case().to_csv(DATA / "scaling_case.csv", index=False)


if __name__ == "__main__":
    main()
