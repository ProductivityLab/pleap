"""Where does the remaining bootstrap gap to MaxDEA come from?

    python docs/bootstrap_decomposition.py no_reflect_0 shephard   # one variant set per process
    python docs/bootstrap_decomposition.py baseline h_x0.8 h_x1.2
    python docs/bootstrap_decomposition.py --summarise

Validation data set A, BCC input-oriented, B = 2000.  PLEAP's bias-corrected
mean sits about twice as far from MaxDEA as from itself under another seed
(docs/validation.md section 7).  Three of PLEAP's documented choices could
contribute: the bandwidth rule, the reflection at 0, and the ratio form of the
correction.  Each is changed in turn, everything else held fixed, and the gap
to MaxDEA is measured on the same seeds:

    baseline        PLEAP as shipped, seeds 1-2 (seeds 1-5 in docs/seed_study.py)
    no_reflect_0    Farrell-scale smoothing, but a draw that would fall below 0
                    is redrawn instead of reflected (seeds 1-3)
    shephard        Simar-Wilson (1998) as published: smoothing on the Shephard
                    scale delta = 1/theta, reflected at 1 only, bandwidth
                    0.9 sd(delta) (2n)^(-1/5) (seeds 1-3)
    h_x0.8, h_x1.2  the shipped bandwidth scaled by 0.8 and 1.2 (seeds 1-2)

The correction form needs no new run: the additive (Shephard) and
ratio-of-means estimators are computed from the baseline replicates.

The variants patch the module's helpers and run serially (n_jobs=1), so the
patch cannot miss a worker process.  Each variant writes its summary to
docs/bootstrap_decomposition_<name>.json and its per-unit runs to the private
folder (A comes from the Korean Innovation Survey and is not redistributed;
PLEAP_PRIVATE_DATA); --summarise combines the summaries with the five-seed
baseline of docs/seed_study.py into docs/bootstrap_decomposition.json.
"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from pleap.core import bootstrap as bt  # noqa: E402

PRIVATE = Path(os.environ.get("PLEAP_PRIVATE_DATA", "tests/data_private"))   # KIS data: not redistributed
ds = pd.read_csv(PRIVATE / "ds_a_dea.csv")
X, Y = ds[["i1", "i2"]].to_numpy(float), ds[["o1"]].to_numpy(float)
ref = pd.read_csv(PRIVATE / "ds_a_maxdea_bootstrap.csv")
ref = ref[(ref.model == "BCC") & (ref.orient == "input")]
ref = ds[["dmu"]].astype(str).merge(ref.assign(dmu=ref.dmu.astype(str))[["dmu", "mean"]],
                                    on="dmu", how="left")["mean"].to_numpy(float)

ORIG_BW, ORIG_DRAW = bt._bandwidth, bt._draw_smoothed


def draw_no_reflect_0(eff, h, rng, size):
    """The shipped draw, except that draws below 0 are redrawn, not reflected."""
    out = np.empty(size)
    todo = np.arange(size)
    n = len(eff)
    var_e = np.var(eff, ddof=1)
    while todo.size:
        samp = eff[rng.integers(0, n, size=todo.size)]
        pert = samp + h * rng.normal(0.0, 1.0, size=todo.size)
        pert = np.where(pert > 1.0, 2.0 - pert, pert)
        ebar = samp.mean()
        star = ebar + (pert - ebar) / np.sqrt(1.0 + h * h / var_e)
        star = np.where(star > 1.0, 2.0 - star, star)
        ok = star > 0
        out[todo[ok]] = star[ok]
        todo = todo[~ok]
    return np.clip(out, 1e-4, 1.0)


def bw_shephard(eff):
    d = 1.0 / eff
    return 0.9 * np.std(d, ddof=1) * (2 * len(d)) ** (-1 / 5)


def draw_shephard(eff, h, rng, size):
    """Simar-Wilson (1998): smooth delta = 1/theta, reflect at 1, return 1/delta*."""
    d = 1.0 / eff
    n = len(d)
    samp = d[rng.integers(0, n, size=size)]
    pert = samp + h * rng.normal(0.0, 1.0, size=size)
    pert = np.where(pert < 1.0, 2.0 - pert, pert)
    dbar = samp.mean()
    var_d = np.var(d, ddof=1)
    star = dbar + (pert - dbar) / np.sqrt(1.0 + h * h / var_d)
    star = np.where(star < 1.0, 2.0 - star, star)
    return np.clip(1.0 / star, 1e-4, 1.0)


VARIANTS = {
    "baseline": (ORIG_BW, ORIG_DRAW, [1, 2]),
    "no_reflect_0": (ORIG_BW, draw_no_reflect_0, [1, 2, 3]),
    "shephard": (bw_shephard, draw_shephard, [1, 2, 3]),
    "h_x0.8": (lambda e: 0.8 * ORIG_BW(e), ORIG_DRAW, [1, 2]),
    "h_x1.2": (lambda e: 1.2 * ORIG_BW(e), ORIG_DRAW, [1, 2]),
}


def summarise_runs(M, theta, frontier):
    M = np.asarray(M)
    pairs = list(itertools.combinations(range(len(M)), 2))
    seed_gap = np.mean([np.abs(M[i] - M[j]) for i, j in pairs], axis=0) if pairs else np.full(len(ref), np.nan)
    gap = np.abs(M.mean(axis=0) - ref)
    return {"gap_nonfrontier": round(float(np.nanmean(gap[~frontier])), 6),
            "gap_frontier": round(float(np.nanmean(gap[frontier])), 6),
            "seed_to_seed_nonfrontier": round(float(np.nanmean(seed_gap[~frontier])), 6),
            "seed_to_seed_frontier": round(float(np.nanmean(seed_gap[frontier])), 6),
            "signed_gap_nonfrontier": round(float(np.nanmean((M.mean(axis=0) - ref)[~frontier])), 6)}


if sys.argv[1:] == ["--summarise"]:
    # the five-seed baseline measured by docs/seed_study.py (docs/validation.md section 7)
    out = {"baseline_5_seeds": {"seeds": [1, 2, 3, 4, 5], "h": 0.09395,
                                "gap_nonfrontier": 0.001081, "gap_frontier": 0.009564,
                                "seed_to_seed_nonfrontier": 0.000525, "seed_to_seed_frontier": 0.004174}}
    for name in VARIANTS:
        try:
            out[name] = json.load(open(f"docs/bootstrap_decomposition_{name}.json"))["summary"]
        except FileNotFoundError:
            pass
    print(json.dumps(out, indent=1))
    json.dump(out, open("docs/bootstrap_decomposition.json", "w"), indent=1)
    raise SystemExit

for name in sys.argv[1:]:
    bw, draw, seeds = VARIANTS[name]
    bt._bandwidth, bt._draw_smoothed = bw, draw
    runs, t0 = [], time.time()
    for s in seeds:
        r = bt.bootstrap_dea(X, Y, model="BCC", orient="input", B=2000, seed=s, n_jobs=1)
        runs.append(np.asarray(r["boot_mean"], float))
        print(f"{name} seed {s}: h={r['bandwidth']:.5f} ({time.time() - t0:.0f}s)", flush=True)
    theta = np.asarray(r["theta_hat"], float)
    frontier = theta > 1 - 1e-6
    summary = dict(seeds=seeds, h=round(float(r["bandwidth"]), 5), **summarise_runs(runs, theta, frontier))
    print(name, json.dumps(summary), flush=True)
    json.dump({"summary": summary}, open(f"docs/bootstrap_decomposition_{name}.json", "w"))
    json.dump({"summary": summary, "runs": [x.tolist() for x in runs]},
              open(PRIVATE / f"bootstrap_decomposition_{name}_runs.json", "w"))
    bt._bandwidth, bt._draw_smoothed = ORIG_BW, ORIG_DRAW
