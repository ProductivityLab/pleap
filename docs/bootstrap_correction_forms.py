"""Gap to MaxDEA of three bias-correction forms on the same replicates.

    python docs/bootstrap_correction_forms.py

Validation data set A, BCC input-oriented, B = 2000, seed 1.  One PLEAP run,
three estimators computed from its replicates: the ratio form PLEAP reports,
the additive correction on the Shephard scale, and the ratio of means.  Gaps
are split into non-frontier and frontier DMUs, as in docs/bootstrap_decomposition.py.
Writes docs/bootstrap_correction_forms.json.
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from pleap.core.bootstrap import bootstrap_dea

PRIVATE = Path(os.environ.get("PLEAP_PRIVATE_DATA", "tests/data_private"))   # KIS data: not redistributed
ds = pd.read_csv(PRIVATE / "ds_a_dea.csv")
X, Y = ds[["i1", "i2"]].to_numpy(float), ds[["o1"]].to_numpy(float)
ref = pd.read_csv(PRIVATE / "ds_a_maxdea_bootstrap.csv")
ref = ref[(ref.model == "BCC") & (ref.orient == "input")]
ref = ds[["dmu"]].merge(ref[["dmu", "mean"]], on="dmu", how="left")["mean"].to_numpy(float)
r = bootstrap_dea(X, Y, model="BCC", orient="input", B=2000, seed=1, n_jobs=2)
th = r["theta_hat"]; bo = np.where(r["boots"] > 0, r["boots"], np.nan); fr = th > 1 - 1e-6
forms = {"ratio (PLEAP)": r["boot_mean"],
         "additive, Shephard scale": np.clip(1 / (2 / th - np.nanmean(1 / bo, axis=1)), 0, 1),
         "ratio of means": np.clip(th ** 2 / np.nanmean(bo, axis=1), 0, 1)}
out = {}
for k, v in forms.items():
    g = np.abs(v - ref)
    out[k] = {"gap_nonfrontier": round(float(g[~fr].mean()), 6), "gap_frontier": round(float(g[fr].mean()), 6)}
    print(f"{k:26s} non-frontier {g[~fr].mean():.6f}  frontier {g[fr].mean():.6f}")
json.dump(out, open("docs/bootstrap_correction_forms.json", "w"), indent=1)
