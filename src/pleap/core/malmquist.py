"""
Malmquist Productivity Index (Fare, Grosskopf, Norris & Zhang 1994).

v1.0.7 - two corrections so results match the MaxDEA reference:
           * the distance functions are now solved under the VRS (BCC)
             technology, input-oriented, by default.  v1.0.6 hard-coded CRS
             (CCR), which produced different EC / TC / MI values.
           * the linear programs are solved in-process with scipy / HiGHS
             instead of spawning an external CBC solver (pulp).
         Output columns follow the MaxDEA layout:
             t(Period), DMU, MI(t-1, t), EC(t-1, t), TC(t-1, t)
"""

# PLEAP - Productivity Lab Efficiency Analysis Program
# Copyright (C) 2025-2026 Changhee Kim, Productivity Lab,
#                         Incheon National University
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import pandas as pd

from pleap.core.lp import best_point
from pleap.core.scaling import column_scales, scaling_is_risky


def _theta_against(Xref, Yref, xk, yk, model="BCC", orient="input"):
    """Distance of point (xk, yk) against the technology spanned by (Xref, Yref).

    Input orientation returns the Farrell input efficiency (theta);
    output orientation returns 1/phi.  The evaluated point need not belong to
    the reference set, so the value may exceed 1 (cross-period comparisons).

    Column scaling keeps the LP well conditioned without changing the
    scale-invariant distance, but on data close to the solver's tolerance a
    single scaled solve cannot be trusted (see pleap.core.scaling).  Where
    `scaling_is_risky` says so, the candidate rule of pleap.core.lp applies:
    the program is solved on the scaled data, on the raw data and on the
    scaled data at tightened tolerances, and the best admissible answer wins.
    """
    Xref = np.asarray(Xref, float); Yref = np.asarray(Yref, float)
    sx = column_scales(Xref); sy = column_scales(Yref)
    risky = scaling_is_risky(Xref, sx) or scaling_is_risky(Yref, sy)
    v, _, _ = best_point(xk, yk, Xref, Yref, sx, sy, model, orient, risky)
    if v is None or not np.isfinite(v) or v <= 0:
        return np.nan
    return v if orient == "input" else 1.0 / v


def malmquist(df, dmu_col, period_col, input_cols, output_cols,
              model="BCC", orient="input", progress=None,
              on_infeasible="substitute", scale_decomposition=False):
    """Malmquist productivity index, decomposed as MI = EC x TC.

    With ``scale_decomposition=True`` the efficiency-change term is split
    further into pure efficiency change and scale efficiency change,

        EC = PEC x SEC,

    which is the usual Fare-Grosskopf-Norris-Zhang presentation: the index
    itself is built from CRS distances, while PEC is the ratio of the VRS
    own-period distances and SEC is what is left over.  The split is defined
    only for a CRS index, so this requires ``model="CCR"``; asking for it
    under BCC raises, because EC is then already a pure efficiency change and
    an "SEC" column would be meaningless rather than merely redundant.

    ``on_infeasible`` decides what happens when a cross-period program is
    infeasible under VRS, and the choice is not cosmetic.

    ``"substitute"`` (default) replaces the infeasible distance by 1, which is
    what MaxDEA does and what keeps PLEAP's reference agreement meaningful.
    An index is then always reported.

    ``"omit"`` returns NaN for the affected row instead, with the reason in
    ``Infeasible_reason``.  Nothing is invented, but the affected units drop
    out of any average.

    Neither is neutral.  On the panel shipped with PLEAP the substitution
    changes the sign of one period's mean productivity change (see
    ``docs/validation.md`` section 8), so the count is always reported and the
    column always appears when it is non-zero.
    """
    if on_infeasible not in ("substitute", "omit"):
        raise ValueError("on_infeasible must be 'substitute' or 'omit'")
    if scale_decomposition and model != "CCR":
        raise ValueError(
            "scale_decomposition needs a CRS index (model='CCR'): under VRS "
            "the efficiency change is already a pure efficiency change, so "
            "there is no scale component to separate")
    periods = sorted(df[period_col].unique())
    pairs = list(zip(periods[:-1], periods[1:]))

    pre = []
    total = 0
    for t1, t2 in pairs:
        a = df[df[period_col] == t1].set_index(dmu_col)
        b = df[df[period_col] == t2].set_index(dmu_col)
        common = sorted(set(a.index) & set(b.index))
        pre.append((t1, t2, a, b, common))
        total += len(common)

    rows = []
    done = 0
    for t1, t2, a, b, common in pre:
        Xa = a.loc[common, input_cols].values.astype(float)
        Ya = a.loc[common, output_cols].values.astype(float)
        Xb = b.loc[common, input_cols].values.astype(float)
        Yb = b.loc[common, output_cols].values.astype(float)
        for k in range(len(common)):
            d_t_t   = _theta_against(Xa, Ya, Xa[k], Ya[k], model, orient)
            d_t1_t1 = _theta_against(Xb, Yb, Xb[k], Yb[k], model, orient)
            d_t_t1  = _theta_against(Xa, Ya, Xb[k], Yb[k], model, orient)
            d_t1_t  = _theta_against(Xb, Yb, Xa[k], Ya[k], model, orient)
            # Under VRS the cross-period LPs can be infeasible (e.g. the unit's
            # output lies outside the other period's frontier span, or an input
            # is zero).  MaxDEA replaces an infeasible cross-period score with
            # 1; we follow the same convention so MI / TC remain defined and
            # match the MaxDEA reference.  Own-period scores are always
            # feasible.
            subs = 0
            if not np.isfinite(d_t_t1):
                d_t_t1 = 1.0; subs += 1
            if not np.isfinite(d_t1_t):
                d_t1_t = 1.0; subs += 1
            if subs and on_infeasible == "omit":
                EC = TC = MI = np.nan
            else:
                try:
                    EC = d_t1_t1 / d_t_t
                    TC = np.sqrt((d_t_t1 / d_t1_t1) * (d_t_t / d_t1_t))
                    MI = EC * TC
                except Exception:
                    EC = TC = MI = np.nan
            row = {"t(Period)": t2, "DMU": common[k],
                   "MI(t-1, t)": MI, "EC(t-1, t)": EC, "TC(t-1, t)": TC,
                   "Infeasible_substitutions": subs}
            if scale_decomposition:
                # PEC from the VRS own-period distances; SEC is the residual,
                # so EC = PEC x SEC holds by construction rather than by
                # a second subtraction that could drift from it.
                v_t_t = _theta_against(Xa, Ya, Xa[k], Ya[k], "BCC", orient)
                v_t1_t1 = _theta_against(Xb, Yb, Xb[k], Yb[k], "BCC", orient)
                try:
                    PEC = v_t1_t1 / v_t_t
                    SEC = EC / PEC
                except Exception:
                    PEC = SEC = np.nan
                if not np.isfinite(EC):
                    PEC = SEC = np.nan
                row["PEC(t-1, t)"] = PEC
                row["SEC(t-1, t)"] = SEC
            if model == "CCR":
                # Färe et al. decompose catching-up into a pure efficiency
                # change measured against the VRS frontier and the scale
                # change that is left over:
                #     EC  = PEC x SEC
                #     PEC = Dv^{t+1} / Dv^t
                #     SEC = (Dc^{t+1}/Dv^{t+1}) / (Dc^t/Dv^t)
                # Only the own-period VRS distances are needed, and those are
                # always feasible, so this adds no infeasibility of its own.
                v_t_t = _theta_against(Xa, Ya, Xa[k], Ya[k], "BCC", orient)
                v_t1_t1 = _theta_against(Xb, Yb, Xb[k], Yb[k], "BCC", orient)
                try:
                    PEC = v_t1_t1 / v_t_t
                    SEC = (d_t1_t1 / v_t1_t1) / (d_t_t / v_t_t)
                except Exception:
                    PEC = SEC = np.nan
                if subs and on_infeasible == "omit":
                    PEC = SEC = np.nan
                row["PEC(t-1, t)"] = PEC
                row["SEC(t-1, t)"] = SEC
            rows.append(row)
            done += 1
            if progress is not None:
                progress(done, total)

    cols = ["t(Period)", "DMU", "MI(t-1, t)", "EC(t-1, t)", "TC(t-1, t)"]
    # This branch used to key off model == "CCR" while nothing ever wrote the
    # two columns, so every CRS run produced a frame with two all-NaN columns.
    # It is now tied to the flag that actually computes them.
    if scale_decomposition:
        cols += ["PEC(t-1, t)", "SEC(t-1, t)"]
    cols += ["Infeasible_substitutions"]
    out = pd.DataFrame(rows, columns=cols)
    # Substituting 1 for an infeasible cross-period distance is a
    # MaxDEA-compatibility default, not a neutral one: it propagates straight
    # into MI and TC.  Record how often it happened so a reader can tell
    # whether any index depends on it, and drop the column when it never did.
    n_sub = int(out["Infeasible_substitutions"].sum())
    n_rows = int((out["Infeasible_substitutions"] > 0).sum())
    if n_sub == 0:
        out = out.drop(columns=["Infeasible_substitutions"])
    out.attrs["infeasible_substitutions"] = n_sub
    out.attrs["rows_affected"] = n_rows
    out.attrs["on_infeasible"] = on_infeasible
    # Two cross-period distances are computed per unit-period row; the other
    # two are own-period and always feasible.  Report that denominator, not
    # the total number of distance functions, so the rate is not understated.
    out.attrs["cross_period_distances"] = 2 * len(out)
    if n_sub == 0:
        out.attrs["substitution_note"] = ""
    elif on_infeasible == "substitute":
        out.attrs["substitution_note"] = (
            f"{n_sub} of {2 * len(out)} cross-period distances, on {n_rows} "
            f"unit-period row(s), were infeasible and replaced by 1 (MaxDEA "
            f"convention). The affected MI / TC values are conditional on that "
            f"substitution; re-run with on_infeasible='omit' to exclude them. "
            f"Under CRS the cross-period programs are always feasible.")
    else:
        out.attrs["substitution_note"] = (
            f"{n_sub} of {2 * len(out)} cross-period distances, on {n_rows} "
            f"unit-period row(s), were infeasible; those rows are reported as "
            f"NaN rather than substituted, so they drop out of any average.")
    return out
