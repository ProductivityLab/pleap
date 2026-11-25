"""Seed-to-seed variability of the bias-corrected mean, on validation data set A.

Section 4.2 says the MaxDEA gap on non-frontier DMUs is "the order of PLEAP's
own seed-to-seed variability".  That is a quantitative claim and this measures
both sides of it on the same data, the same model and the same B.

A comes from the Korean Innovation Survey (STEPI) and is not redistributed: the
script reads it from PLEAP_PRIVATE_DATA, writes only aggregates to
docs/seed_study.json and keeps the per-unit gaps in the private folder.
"""
import itertools, json, os
import numpy as np, pandas as pd
from pathlib import Path
from pleap.core.dea import dea
from pleap.core.bootstrap import bootstrap_dea

D = Path(os.environ.get("PLEAP_PRIVATE_DATA", "tests/data_private"))   # KIS data: not redistributed
ds = pd.read_csv(D / "ds_a_dea.csv")
INS, OUTS = ["i1", "i2"], ["o1"]
X, Y = ds[INS].to_numpy(float), ds[OUTS].to_numpy(float)

SEEDS = [1, 2, 3, 4, 5]
B = 2000
runs = {}
for s in SEEDS:
    r = bootstrap_dea(X, Y, model="BCC", orient="input", B=B, seed=s, n_jobs=2)
    runs[s] = np.asarray(r["boot_mean"], float)
    print(f"seed {s} done", flush=True)

theta = dea(X, Y, model="BCC", orient="input")["efficiency"]
M = np.vstack([runs[s] for s in SEEDS])

ref = pd.read_csv(D / "ds_a_maxdea_bootstrap.csv")
ref = ref[(ref.model == "BCC") & (ref.orient == "input")].copy()
ref["dmu"] = ref["dmu"].astype(str)
ds["dmu"] = ds["dmu"].astype(str)
ref = ds[["dmu"]].merge(ref[["dmu", "mean"]], on="dmu", how="left")
ref_mean = ref["mean"].to_numpy(float)

frontier = theta > 1 - 1e-6
pairs = list(itertools.combinations(range(len(SEEDS)), 2))
seed_gap = np.mean([np.abs(M[i] - M[j]) for i, j in pairs], axis=0)
maxdea_gap = np.abs(M.mean(axis=0) - ref_mean)

print(f"\ndata set A, BCC input, B={B}, {len(SEEDS)} seeds, {len(theta)} DMUs "
      f"({int(frontier.sum())} on the frontier)")
print(f"{'':14s}{'seed-to-seed':>14s}{'vs MaxDEA':>12s}{'ratio':>8s}")
for name, m in (("non-frontier", ~frontier), ("frontier", frontier)):
    if not m.any():
        continue
    a, b = float(np.nanmean(seed_gap[m])), float(np.nanmean(maxdea_gap[m]))
    print(f"{name:14s}{a:14.6f}{b:12.6f}{b/a:8.2f}")
a, b = float(np.nanmean(seed_gap)), float(np.nanmean(maxdea_gap))
print(f"{'all':14s}{a:14.6f}{b:12.6f}{b/a:8.2f}")
json.dump({"seeds": SEEDS, "B": B,
           "seed_gap": seed_gap.tolist(), "maxdea_gap": maxdea_gap.tolist(),
           "frontier": frontier.tolist()},
          open(D / "seed_study_per_unit.json", "w"))
summary = {"seeds": SEEDS, "B": B, "units": int(len(theta)), "frontier_units": int(frontier.sum())}
for name, m in (("nonfrontier", ~frontier), ("frontier", frontier)):
    summary[f"seed_to_seed_{name}"] = round(float(np.nanmean(seed_gap[m])), 6)
    summary[f"gap_to_maxdea_{name}"] = round(float(np.nanmean(maxdea_gap[m])), 6)
json.dump(summary, open("docs/seed_study.json", "w"), indent=1)
print("WROTE")
