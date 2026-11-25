"""Figure builders for the PLEAP Visualization tab.

Pure matplotlib - no PyQt dependency - so every figure can be produced and
tested headlessly.  This module never calls the analysis routines: it only
renders results that have already been computed by pleap.core.

Appearance is controlled by :func:`set_theme` (colour palette, visual style
and font scale); every builder reads the current theme when it draws.
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
from matplotlib.figure import Figure

__all__ = ["set_korean_font", "set_theme", "current_theme", "PALETTES",
           "STYLES", "fig_frontier", "fig_eff_hist", "fig_eff_bar",
           "fig_boot_ci", "fig_malm_trend", "fig_malm_quadrant",
           "fig_malm_bar", "fig_malm_box", "figure_to_png"]

_FONT_READY = False

# --------------------------------------------------------------------------- #
#  Themes                                                                     #
# --------------------------------------------------------------------------- #
#   eff   - efficient DMUs / primary series      proj  - radial projections
#   ineff - inefficient DMUs / secondary series  acc   - third series
#   front - frontier and reference lines
PALETTES = {
    "INU Blue (default)": dict(eff="#0B4F9E", ineff="#7F8C99", front="#B4241F",
                               proj="#E08A2E", acc="#1B7F5A"),
    "Colour-blind safe":  dict(eff="#0072B2", ineff="#999999", front="#D55E00",
                               proj="#E69F00", acc="#009E73"),
    "Grayscale (print)":  dict(eff="#1A1A1A", ineff="#B5B5B5", front="#000000",
                               proj="#6E6E6E", acc="#4D4D4D"),
    "Teal & Coral":       dict(eff="#1F7A73", ineff="#9AA5A3", front="#E2553D",
                               proj="#F0A202", acc="#3D5A80"),
    "Nature":             dict(eff="#2D6A4F", ineff="#A3B18A", front="#BC4749",
                               proj="#DDA15E", acc="#386641"),
    "Journal Navy":       dict(eff="#1D3557", ineff="#A8B6C4", front="#C1121F",
                               proj="#F4A261", acc="#457B9D"),
    "Vivid":              dict(eff="#3A0CA3", ineff="#ADB5BD", front="#F72585",
                               proj="#FF9E00", acc="#4CC9F0"),
}

STYLES = ["Academic (white)", "Grid", "Minimal", "Dark", "Classic box"]

_PALETTE_NAME = "INU Blue (default)"
_PALETTE = dict(PALETTES[_PALETTE_NAME])
_STYLE = "Academic (white)"
_FSCALE = 1.0


def set_theme(palette=None, style=None, font_scale=None):
    """Select the colour palette / visual style used by every figure."""
    global _PALETTE, _PALETTE_NAME, _STYLE, _FSCALE
    if palette and palette in PALETTES:
        _PALETTE = dict(PALETTES[palette]); _PALETTE_NAME = palette
    if style and style in STYLES:
        _STYLE = style
    if font_scale:
        _FSCALE = max(0.6, min(2.0, float(font_scale)))
    return current_theme()


def current_theme():
    return {"palette": _PALETTE_NAME, "style": _STYLE, "font_scale": _FSCALE}


def _c(key):        return _PALETTE.get(key, "#333333")
def _dark():        return _STYLE == "Dark"
def _bg():          return "#1E222A" if _dark() else "white"
def _fg():          return "#E8EAED" if _dark() else "#111111"
def _muted():       return "#9AA0A6" if _dark() else "#666666"
def _fs(pt):        return pt * _FSCALE


def set_korean_font():
    """Pick a font that can render Korean DMU names, if one is installed."""
    global _FONT_READY
    if _FONT_READY:
        return
    try:
        from matplotlib import font_manager, rcParams
        have = {f.name for f in font_manager.fontManager.ttflist}
        for name in ("Malgun Gothic", "AppleGothic", "NanumGothic",
                     "Noto Sans CJK KR", "Noto Sans KR", "Gulim", "Batang"):
            if name in have:
                rcParams["font.family"] = name
                break
        rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    _FONT_READY = True


def _new_fig(w=7.6, h=5.0):
    set_korean_font()
    fig = Figure(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(_bg())
    return fig


def _style(ax, title=None, xlabel=None, ylabel=None, legend=True):
    ax.set_facecolor(_bg())
    grid_c = "#3A3F4A" if _dark() else "#B9BEC7"
    if _STYLE == "Grid":
        ax.grid(True, ls="-", lw=0.6, alpha=0.45, color=grid_c)
    elif _STYLE == "Minimal":
        ax.grid(False)
    else:
        ax.grid(True, ls=":", lw=0.6, alpha=0.55, color=grid_c)
    ax.set_axisbelow(True)
    if _STYLE == "Classic box":
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True); ax.spines[side].set_color(_fg())
    else:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_muted())
    if title:
        ax.set_title(title, fontsize=_fs(11.5), fontweight="bold", pad=10,
                     color=_fg())
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=_fs(10), color=_fg())
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=_fs(10), color=_fg())
    ax.tick_params(labelsize=_fs(9), colors=_fg())
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_color(_fg())
    if legend and ax.get_legend_handles_labels()[0]:
        leg = ax.legend(fontsize=_fs(8.5), frameon=(_STYLE == "Classic box"))
        for t in leg.get_texts():
            t.set_color(_fg())
        if _STYLE == "Classic box":
            leg.get_frame().set_facecolor(_bg())
            leg.get_frame().set_edgecolor(_muted())


def figure_to_png(fig, dpi=300):
    """Render a Figure to PNG bytes (used for the appendix export)."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    return buf.getvalue()


# --------------------------------------------------------------------------- #
#  DEA frontier                                                               #
# --------------------------------------------------------------------------- #
def _aggregate(X, Y, ins=None, outs=None):
    """Map the DMUs into a 2-D (input, output) plane.

    With one input and one output this is the data itself, so the frontier that
    is drawn is exact.  Otherwise each column is divided by its mean and
    averaged, giving a 2-D projection of the multi-dimensional technology.
    Because the aggregation is linear and radial projection scales every input
    (or output) by the same factor, projected points still lie on the true
    frontier in this plane.
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    if X.shape[1] == 1 and Y.shape[1] == 1:
        xl = ins[0] if ins else "Input"
        yl = outs[0] if outs else "Output"
        return X[:, 0], Y[:, 0], xl, yl, True
    mx = X.mean(0).copy(); mx[mx == 0] = 1.0
    my = Y.mean(0).copy(); my[my == 0] = 1.0
    ai = (X / mx).sum(1) / X.shape[1]
    ao = (Y / my).sum(1) / Y.shape[1]
    return (ai, ao, "Aggregate input (mean-normalised)",
            "Aggregate output (mean-normalised)", False)


def _upper_hull(pts):
    """Upper convex-hull chain of 2-D points, left to right."""
    P = np.asarray(pts, float)
    P = P[np.all(np.isfinite(P), axis=1)]
    if len(P) == 0:
        return P
    P = P[np.lexsort((-P[:, 1], P[:, 0]))]
    _, first = np.unique(P[:, 0], return_index=True)
    P = P[np.sort(first)]                      # highest y at each x
    if len(P) <= 2:
        return P
    hull = []
    for p in P:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            cross = (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)
            if cross >= 0:                     # hull[-1] lies below -> drop it
                hull.pop()
            else:
                break
        hull.append((p[0], p[1]))
    return np.array(hull)


def _vrs_frontier(front_pts, x_hi):
    """VRS (BCC) frontier: concave, non-decreasing, free-disposal extension."""
    hull = _upper_hull(front_pts)
    if len(hull) == 0:
        return hull
    imax = int(np.argmax(hull[:, 1]))
    chain = hull[:imax + 1]
    if x_hi > chain[-1, 0]:                    # free disposability of input
        chain = np.vstack([chain, [x_hi, chain[-1, 1]]])
    return chain


def fig_frontier(X, Y, eff, model="BCC", orient="input", labels=None,
                 ins=None, outs=None, show_labels=True, show_proj=True,
                 title=None):
    """Efficiency frontier with the DMUs and their radial projections."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    eff = np.asarray(eff, float)
    ai, ao, xl, yl, exact = _aggregate(X, Y, ins, outs)
    n = len(ai)
    if labels is None:
        labels = [f"DMU{i+1}" for i in range(n)]
    ok = np.isfinite(eff) & np.isfinite(ai) & np.isfinite(ao)
    is_eff = ok & (eff >= 1 - 1e-6)

    # Radial projection onto the frontier (eff <= 1 for both orientations).
    with np.errstate(divide="ignore", invalid="ignore"):
        if orient == "input":
            pai = np.where(ok, ai * eff, np.nan); pao = np.where(ok, ao, np.nan)
        else:
            pai = np.where(ok, ai, np.nan)
            pao = np.where(ok & (eff > 0), ao / eff, np.nan)

    fig = _new_fig()
    ax = fig.add_subplot(111)

    # ---- frontier line ----------------------------------------------------
    fp = np.column_stack([np.concatenate([ai[is_eff], pai[ok & ~is_eff]]),
                          np.concatenate([ao[is_eff], pao[ok & ~is_eff]])])
    fp = fp[np.all(np.isfinite(fp), axis=1)]
    x_hi = np.nanmax(ai) * 1.06 if n else 1.0
    if len(fp):
        if model.upper() == "CCR":
            good = fp[:, 0] > 0
            slope = np.max(fp[good, 1] / fp[good, 0]) if good.any() else 0.0
            ax.plot([0, x_hi], [0, slope * x_hi], "-", color=_c("front"), lw=2.0,
                    zorder=2, label="Efficiency frontier (CRS)")
        else:
            chain = _vrs_frontier(fp, x_hi)
            if len(chain):
                ax.plot(chain[:, 0], chain[:, 1], "-", color=_c("front"), lw=2.0,
                        zorder=2, label="Efficiency frontier (VRS)")
                ax.plot([chain[0, 0], chain[0, 0]], [0, chain[0, 1]], "-",
                        color=_c("front"), lw=2.0, zorder=2)

    # ---- projections ------------------------------------------------------
    if show_proj:
        idx = np.where(ok & ~is_eff)[0]
        for i in idx:
            ax.plot([ai[i], pai[i]], [ao[i], pao[i]], "-", color=_c("proj"),
                    lw=0.7, alpha=0.55, zorder=1)
        if len(idx):
            ax.scatter(pai[idx], pao[idx], s=26, facecolors="none",
                       edgecolors=_c("proj"), lw=1.0, zorder=3,
                       label="Radial projection")

    # ---- DMUs -------------------------------------------------------------
    m_in = ok & ~is_eff
    if m_in.any():
        ax.scatter(ai[m_in], ao[m_in], s=34, c=_c("ineff"), alpha=0.85,
                   edgecolors=_bg(), lw=0.5, zorder=4,
                   label=f"Inefficient DMU (n={int(m_in.sum())})")
    if is_eff.any():
        ax.scatter(ai[is_eff], ao[is_eff], s=62, c=_c("eff"), marker="o",
                   edgecolors=_bg(), lw=0.8, zorder=5,
                   label=f"Efficient DMU (n={int(is_eff.sum())})")

    if show_labels:
        show = np.where(is_eff)[0] if n > 25 else np.where(ok)[0]
        for i in show[:40]:
            ax.annotate(str(labels[i]), (ai[i], ao[i]), fontsize=_fs(7.5),
                        xytext=(4, 4), textcoords="offset points", color=_fg())

    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    _style(ax, title or f"Efficiency frontier - {model.upper()} ({orient}-oriented)",
           xl, yl)
    if not exact:
        fig.text(0.5, -0.005,
                 "2-D projection of a multi-dimensional technology: axes are "
                 "mean-normalised aggregates of all inputs / outputs.",
                 ha="center", fontsize=_fs(7.8), color=_muted())
    fig.tight_layout()
    return fig


def fig_eff_hist(eff, model="BCC", orient="input", bins=20):
    eff = np.asarray(eff, float); eff = eff[np.isfinite(eff)]
    fig = _new_fig(7.2, 4.6); ax = fig.add_subplot(111)
    ax.hist(eff, bins=bins, range=(0, 1), color=_c("eff"), alpha=0.82,
            edgecolor=_bg(), lw=0.7)
    mu = float(np.mean(eff)) if len(eff) else np.nan
    ax.axvline(mu, color=_c("front"), ls="--", lw=1.4, label=f"Mean = {mu:.4f}")
    n_eff = int((eff >= 1 - 1e-6).sum())
    ax.axvline(1.0, color=_c("acc"), ls=":", lw=1.4,
               label=f"Efficient (score = 1): {n_eff} DMU")
    _style(ax, f"Distribution of efficiency scores - {model.upper()} ({orient})",
           "Efficiency score", "Number of DMUs")
    fig.tight_layout()
    return fig


def fig_eff_bar(eff, labels=None, top=None, model="BCC", orient="input"):
    eff = np.asarray(eff, float)
    n = len(eff)
    if labels is None:
        labels = [f"DMU{i+1}" for i in range(n)]
    ok = np.isfinite(eff)
    idx = np.argsort(-np.where(ok, eff, -np.inf))
    idx = idx[: (top or n)]
    vals = eff[idx]; nm = [str(labels[i]) for i in idx]
    h = max(4.2, 0.26 * len(idx) + 1.4)
    fig = _new_fig(7.6, min(h, 16)); ax = fig.add_subplot(111)
    colors = [_c("eff") if v >= 1 - 1e-6 else _c("ineff") for v in vals]
    ypos = np.arange(len(idx))[::-1]
    ax.barh(ypos, vals, color=colors, edgecolor=_bg(), lw=0.5, height=0.72)
    ax.set_yticks(ypos); ax.set_yticklabels(nm, fontsize=_fs(7.5))
    ax.axvline(1.0, color=_c("front"), ls="--", lw=1.2)
    ax.set_xlim(0, max(1.02, float(np.nanmax(vals)) * 1.02) if len(vals) else 1.02)
    ttl = f"Efficiency by DMU - {model.upper()} ({orient})"
    if top and top < n:
        ttl += f"  [top {top}]"
    _style(ax, ttl, "Efficiency score", None, legend=False)
    fig.tight_layout()
    return fig


def fig_boot_ci(boot_df, top=None, sort=True):
    """Original vs bias-corrected efficiency with bootstrap confidence bands."""
    df = boot_df.copy()
    need = {"DMU", "Score(Original)", "Mean", "CI_LowerBound", "CI_UpperBound"}
    if not need.issubset(df.columns):
        raise ValueError("Bootstrap result table is missing required columns.")
    if sort:
        df = df.sort_values("Mean", ascending=False)
    if top:
        df = df.head(top)
    n = len(df)
    x = np.arange(n)
    mean = df["Mean"].values.astype(float)
    lo = df["CI_LowerBound"].values.astype(float)
    hi = df["CI_UpperBound"].values.astype(float)
    orig = df["Score(Original)"].values.astype(float)
    fig = _new_fig(max(7.6, min(0.22 * n + 2.4, 18)), 5.0)
    ax = fig.add_subplot(111)
    ax.vlines(x, lo, hi, color=_c("ineff"), lw=1.3, alpha=0.8,
              label="95% confidence interval")
    ax.plot(x, orig, "o", ms=4.6, color=_c("front"), label="Original score", zorder=4)
    ax.plot(x, mean, "s", ms=4.2, color=_c("eff"), label="Bias-corrected (Mean)",
            zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in df["DMU"].values], rotation=90, fontsize=_fs(7))
    _style(ax, "Bootstrap bias-corrected efficiency with confidence intervals",
           None, "Efficiency score")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  Malmquist                                                                  #
# --------------------------------------------------------------------------- #
MI_COL, EC_COL, TC_COL = "MI(t-1, t)", "EC(t-1, t)", "TC(t-1, t)"
_IDX = {"MI": MI_COL, "EC": EC_COL, "TC": TC_COL}


def _gmean(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.exp(np.mean(np.log(v)))) if len(v) else np.nan


def fig_malm_trend(mdf, show=("MI", "EC", "TC")):
    """Geometric mean of MI / EC / TC by period."""
    per = sorted(mdf["t(Period)"].unique())
    fig = _new_fig(7.6, 4.8); ax = fig.add_subplot(111)
    style = {"MI": (_c("front"), "o", 2.2), "EC": (_c("eff"), "s", 1.6),
             "TC": (_c("acc"), "^", 1.6)}
    for key in show:
        col = _IDX[key]
        vals = [_gmean(mdf.loc[mdf["t(Period)"] == p, col]) for p in per]
        c, mk, lw = style[key]
        ax.plot(range(len(per)), vals, marker=mk, color=c, lw=lw, ms=6,
                label=f"{key} (geometric mean)")
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.3f}", (i, v), fontsize=_fs(7),
                            xytext=(0, 7), textcoords="offset points",
                            ha="center", color=c)
    ax.axhline(1.0, color=_muted(), ls="--", lw=1.1)
    ax.set_xticks(range(len(per))); ax.set_xticklabels([str(p) for p in per])
    _style(ax, "Malmquist productivity index by period",
           "Period (t)", "Index (1.0 = no change)")
    fig.tight_layout()
    return fig


def fig_malm_quadrant(mdf, period=None, show_labels=True):
    """EC-TC scatter: catch-up vs innovation, with the MI = 1 iso-curve."""
    d = mdf if period in (None, "All") else mdf[mdf["t(Period)"] == period]
    ec = d[EC_COL].values.astype(float); tc = d[TC_COL].values.astype(float)
    lab = d["DMU"].values
    ok = np.isfinite(ec) & np.isfinite(tc)
    ec, tc, lab = ec[ok], tc[ok], lab[ok]
    mi = ec * tc
    fig = _new_fig(7.2, 5.6); ax = fig.add_subplot(111)
    up = mi >= 1
    ax.scatter(ec[up], tc[up], s=42, c=_c("eff"), alpha=0.85, edgecolors="white",
               lw=0.5, label=f"MI >= 1 (progress, n={int(up.sum())})")
    ax.scatter(ec[~up], tc[~up], s=42, c=_c("ineff"), alpha=0.85,
               edgecolors=_bg(), lw=0.5,
               label=f"MI < 1 (regress, n={int((~up).sum())})")
    ax.axhline(1.0, color=_muted(), ls="--", lw=1.1)
    ax.axvline(1.0, color=_muted(), ls="--", lw=1.1)
    if len(ec):
        lo = max(1e-3, min(ec.min(), tc.min()) * 0.9)
        hi = max(ec.max(), tc.max()) * 1.08
        xs = np.linspace(lo, hi, 200)
        ax.plot(xs, 1.0 / xs, ":", color=_c("front"), lw=1.3, label="MI = 1")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        if show_labels and len(ec) <= 40:
            for i in range(len(ec)):
                ax.annotate(str(lab[i]), (ec[i], tc[i]), fontsize=_fs(7),
                            xytext=(4, 4), textcoords="offset points")
    ttl = "Efficiency change vs technical change"
    if period not in (None, "All"):
        ttl += f"  (t = {period})"
    _style(ax, ttl, "EC  (catch-up effect)", "TC  (frontier shift / innovation)")
    fig.tight_layout()
    return fig


def fig_malm_bar(mdf, index="MI", period=None, top=None):
    col = _IDX[index]
    d = mdf if period in (None, "All") else mdf[mdf["t(Period)"] == period]
    if period in (None, "All"):
        g = d.groupby("DMU")[col].apply(_gmean).reset_index()
        d = g.rename(columns={col: col})
    d = d[["DMU", col]].dropna().sort_values(col, ascending=False)
    if top:
        d = d.head(top)
    vals = d[col].values.astype(float)
    nm = [str(v) for v in d["DMU"].values]
    h = max(4.2, 0.26 * len(d) + 1.4)
    fig = _new_fig(7.6, min(h, 16)); ax = fig.add_subplot(111)
    colors = [_c("eff") if v >= 1 else _c("ineff") for v in vals]
    ypos = np.arange(len(d))[::-1]
    ax.barh(ypos, vals, color=colors, edgecolor=_bg(), lw=0.5, height=0.72)
    ax.set_yticks(ypos); ax.set_yticklabels(nm, fontsize=_fs(7.5))
    ax.axvline(1.0, color=_c("front"), ls="--", lw=1.2)
    ttl = f"{index} by DMU"
    ttl += "  (geometric mean over all periods)" if period in (None, "All") \
        else f"  (t = {period})"
    _style(ax, ttl, index, None, legend=False)
    fig.tight_layout()
    return fig


def fig_malm_box(mdf, index="MI"):
    col = _IDX[index]
    per = sorted(mdf["t(Period)"].unique())
    data = [mdf.loc[mdf["t(Period)"] == p, col].dropna().values for p in per]
    fig = _new_fig(7.6, 4.8); ax = fig.add_subplot(111)
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color=_c("front"), lw=1.6),
                    flierprops=dict(marker="o", ms=3, mfc=_c("ineff"),
                                    mec="none", alpha=0.6))
    for key in ("whiskers", "caps"):
        for art in bp[key]:
            art.set_color(_muted())
    for patch in bp["boxes"]:
        patch.set_facecolor(_c("eff")); patch.set_alpha(0.35)
        patch.set_edgecolor(_c("eff"))
    ax.axhline(1.0, color=_muted(), ls="--", lw=1.1)
    ax.set_xticklabels([str(p) for p in per])
    _style(ax, f"Distribution of {index} by period", "Period (t)", index,
           legend=False)
    fig.tight_layout()
    return fig
