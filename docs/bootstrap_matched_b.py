"""Section 7.1 of docs/validation.md: PLEAP's bootstrap against MaxDEA at
MaxDEA's own number of replications.

    python docs/bootstrap_matched_b.py A      # B = 500, five seeds  (~10 min)
    python docs/bootstrap_matched_b.py D      # B = 1000, two seeds  (~1.5 h on 2 cores)

If the two procedures were statistically identical, the mean |PLEAP - MaxDEA|
over the units off the frontier would equal the mean |PLEAP seed s - seed t|.
"""
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pleap.core.bootstrap import bootstrap_dea

# A and D come from the Korean Innovation Survey (STEPI) and are not redistributed
DATA = Path(os.environ.get("PLEAP_PRIVATE_DATA",
                           Path(__file__).resolve().parents[1] / "tests" / "data_private"))
SPEC = {"A": ("ds_a_dea.csv", "ds_a_maxdea_bootstrap.csv", 500, (1, 2, 3, 4, 5)),
        "D": ("ds_d.csv", "ds_d_maxdea_bootstrap.csv", 1000, (1, 2))}


def main(name):
    data_file, ref_file, B, seeds = SPEC[name]
    d = pd.read_csv(DATA / data_file)
    ref = pd.read_csv(DATA / ref_file)
    X = d[["i1", "i2"]].to_numpy(float); Y = d[["o1"]].to_numpy(float)
    rows = []
    for model, orient in (("BCC", "input"), ("BCC", "output"), ("CCR", "input")):
        runs = [bootstrap_dea(X, Y, model=model, orient=orient, B=B, seed=s) for s in seeds]
        M = ref[(ref.model == model) & (ref.orient == orient)].set_index("dmu").loc[d["dmu"], "mean"].to_numpy()
        th = runs[0]["theta_hat"]
        out = {}
        for part, mask in (("off frontier", th < 1 - 1e-9), ("on frontier", th >= 1 - 1e-9)):
            gap = np.mean([np.mean(np.abs(r["boot_mean"][mask] - M[mask])) for r in runs])
            s2s = np.mean([np.mean(np.abs(a["boot_mean"][mask] - b["boot_mean"][mask]))
                           for a, b in itertools.combinations(runs, 2)])
            out[part] = (gap, s2s, gap / s2s)
        signed = np.mean([np.mean(r["boot_mean"][th < 1 - 1e-9] - M[th < 1 - 1e-9]) for r in runs])
        rel = np.mean([np.mean((r["boot_mean"][th < 1 - 1e-9] - M[th < 1 - 1e-9]) / th[th < 1 - 1e-9]) for r in runs])
        rows.append(dict(model=f"{model} {orient}", B=B, gap=out["off frontier"][0], seed_to_seed=out["off frontier"][1],
                         ratio=out["off frontier"][2], signed=signed, relative=rel,
                         frontier_ratio=out["on frontier"][2], n_frontier=int((th >= 1 - 1e-9).sum())))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.5g}"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A")
