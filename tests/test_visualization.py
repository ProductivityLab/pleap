"""Visualization and publication-table layer.

The Visualization tab renders results that ``pleap.core`` has already
computed; it must never change a number.  These tests therefore check two
things: that every figure and table builds head-less under every theme, and
that the numbers travelling into a publication table are the full-precision
ones from the estimator rather than the rounded display column.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from pleap.core.bootstrap import bootstrap_dea
from pleap.core.dea import dea
from pleap.core.malmquist import malmquist
from pleap.core.secondstage import ols_regression, tobit_regression
from pleap.viz import figures as F
from pleap.viz import tables as T

INS, OUTS = ["i1", "i2"], ["o1"]


@pytest.fixture(scope="module")
def solved(sa):
    X = sa[INS].to_numpy(float)
    Y = sa[OUTS].to_numpy(float)
    r = dea(X, Y, model="BCC", orient="input")
    return dict(X=X, Y=Y, eff=r["efficiency"], labels=sa["dmu"].tolist(),
                ds=sa)


@pytest.fixture(scope="module")
def mdf(sa_panel):
    return malmquist(sa_panel, "dmu", "period", INS, OUTS)


@pytest.fixture(scope="module")
def boot(sa):
    X = sa[INS].to_numpy(float); Y = sa[OUTS].to_numpy(float)
    import pandas as pd
    r = bootstrap_dea(X, Y, model="BCC", orient="input", B=40, seed=1, n_jobs=1)
    return pd.DataFrame({
        "DMU": sa["dmu"], "Score(Original)": r["theta_hat"],
        "Bias": r["bias"], "Mean": r["boot_mean"],
        "Median": r["boot_median"], "SD": r["boot_sd"],
        "CI_LowerBound": r["ci_low"], "CI_UpperBound": r["ci_high"]})


# --------------------------------------------------------------------------- #
#  Themes                                                                     #
# --------------------------------------------------------------------------- #
def test_every_palette_and_style_is_usable():
    assert len(F.PALETTES) >= 5
    assert "Grayscale (print)" in F.PALETTES
    assert "Colour-blind safe" in F.PALETTES
    for name, pal in F.PALETTES.items():
        for key in ("eff", "ineff", "front"):
            assert key in pal, (name, key)
            assert str(pal[key]).startswith("#"), (name, key)


@pytest.mark.parametrize("palette", sorted(F.PALETTES))
def test_frontier_builds_under_every_palette(solved, palette):
    F.set_theme(palette=palette, style="Academic (white)", font_scale=1.0)
    fig = F.fig_frontier(solved["X"], solved["Y"], solved["eff"],
                         ins=INS, outs=OUTS, labels=solved["labels"])
    assert fig.axes
    png = F.figure_to_png(fig)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 5000


@pytest.mark.parametrize("style", F.STYLES)
def test_frontier_builds_under_every_style(solved, style):
    F.set_theme(palette="INU Blue (default)", style=style, font_scale=1.1)
    fig = F.fig_frontier(solved["X"], solved["Y"], solved["eff"],
                         ins=INS, outs=OUTS)
    assert F.figure_to_png(fig)[:4] == b"\x89PNG"


def test_current_theme_round_trips():
    F.set_theme(palette="Nature", style="Grid", font_scale=1.3)
    cur = F.current_theme()
    assert "Nature" in str(cur) and "Grid" in str(cur)


# --------------------------------------------------------------------------- #
#  Figures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("orient", ["input", "output"])
@pytest.mark.parametrize("model", ["CCR", "BCC"])
def test_frontier_builds_for_every_model(sa, model, orient):
    X = sa[INS].to_numpy(float); Y = sa[OUTS].to_numpy(float)
    eff = dea(X, Y, model=model, orient=orient)["efficiency"]
    fig = F.fig_frontier(X, Y, eff, model=model, orient=orient,
                         ins=INS, outs=OUTS)
    assert fig.axes


def test_frontier_is_exact_for_one_input_one_output(one_in_one_out):
    """With a single input and output the picture is the real technology."""
    X, Y, _ = one_in_one_out
    eff = dea(X, Y, model="BCC", orient="input")["efficiency"]
    fig = F.fig_frontier(X, Y, eff, ins=["x"], outs=["y"])
    ax = fig.axes[0]
    pts = np.vstack([c.get_offsets() for c in ax.collections if
                     len(c.get_offsets())])
    # every DMU appears at its own raw coordinates
    for x, y in zip(X[:, 0], Y[:, 0]):
        assert np.any(np.hypot(pts[:, 0] - x, pts[:, 1] - y) < 1e-6), (x, y)


def test_distribution_and_ranking_figures(solved):
    assert F.fig_eff_hist(solved["eff"]).axes
    assert F.fig_eff_bar(solved["eff"], labels=solved["labels"], top=15).axes


def test_bootstrap_figure(boot):
    fig = F.fig_boot_ci(boot, top=20)
    assert fig.axes
    assert F.figure_to_png(fig)[:4] == b"\x89PNG"


@pytest.mark.parametrize("index", ["MI", "EC", "TC"])
def test_malmquist_figures(mdf, index):
    assert F.fig_malm_bar(mdf, index=index).axes
    assert F.fig_malm_box(mdf, index=index).axes


def test_malmquist_trend_and_quadrant(mdf):
    assert F.fig_malm_trend(mdf).axes
    assert F.fig_malm_quadrant(mdf).axes


def test_upper_hull_is_monotone_and_concave():
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(1, 10, 60), rng.uniform(1, 10, 60)])
    hull = np.asarray(F._upper_hull(pts), float)
    assert len(hull) >= 2
    assert np.all(np.diff(hull[:, 0]) > 0), "hull must advance in x"
    slopes = np.diff(hull[:, 1]) / np.diff(hull[:, 0])
    assert np.all(np.diff(slopes) <= 1e-9), "hull must be concave"


# --------------------------------------------------------------------------- #
#  Publication tables                                                         #
# --------------------------------------------------------------------------- #
def test_small_coefficients_are_not_printed_as_zero():
    """A 1e-8 coefficient must survive as text, not collapse to 0.000.

    This is what large-scale environmental variables produce, and it was the
    v1.1.0 display bug.
    """
    assert T.fmt_coef(0.0000000123, digits=3) not in ("0.000", "0", "")
    assert "e" in T.fmt_coef(0.0000000123, digits=3).lower()
    assert T.fmt_coef(0.456, digits=3) == "0.456"


def test_regression_table_uses_full_precision_coefficients(sa):
    """The table must read coef_raw, not the rounded `coef` column."""
    y = dea(sa[INS].to_numpy(float), sa[OUTS].to_numpy(float),
            model="BCC", orient="input")["efficiency"]
    Z = sa[["z1", "z2"]].to_numpy(float) * 1e7      # forces tiny coefficients
    res = tobit_regression(y, Z, lower=None, upper=1.0, names=["z1", "z2"])
    raw = np.asarray(res["summary"].attrs["coef_raw"], float)
    assert np.any(np.abs(raw[1:3]) < 1e-4), "fixture should give tiny coefs"
    df, caption, note = T.regression_table(
        [{"method": "Tobit", "kind": "Raw efficiency (DEA)",
          "envs": ["z1", "z2"], "result": res}])
    text = T.table_to_text(df, caption, note)
    assert "0.000" not in text.replace("(0.000)", ""), text
    for v in raw[1:3]:
        mant = f"{abs(v):.3e}".split("e")[0][:3]
        assert mant in text or "e-" in text.lower()


def test_regression_table_holds_several_models_side_by_side(sa):
    y = dea(sa[INS].to_numpy(float), sa[OUTS].to_numpy(float))["efficiency"]
    Z = sa[["z1", "z2"]].to_numpy(float)
    runs = [
        {"method": "OLS", "kind": "Raw efficiency (DEA)", "envs": ["z1", "z2"],
         "result": ols_regression(y, Z, names=["z1", "z2"])},
        {"method": "Tobit", "kind": "Raw efficiency (DEA)", "envs": ["z1", "z2"],
         "result": tobit_regression(y, Z, lower=None, upper=1.0,
                                    names=["z1", "z2"])},
    ]
    df, caption, note = T.regression_table(runs, table_no=2,
                                           caption="Determinants")
    assert df.shape[1] == 2, "one column per model"
    assert list(df.columns) == ["(1)", "(2)"]
    assert "Table 2. Determinants" == caption
    assert "p < 0.01" in note
    idx = [str(i) for i in df.index]
    assert "Constant" in idx and "z1" in idx and "Observations" in idx
    # the Tobit column carries a scale row, the OLS column an R-squared row
    joined = " ".join(idx)
    assert "R-squared" in joined


@pytest.mark.parametrize("fmt", ["html", "markdown", "latex", "text"])
def test_every_table_export_format_runs(sa, fmt):
    y = dea(sa[INS].to_numpy(float), sa[OUTS].to_numpy(float))["efficiency"]
    Z = sa[["z1", "z2"]].to_numpy(float)
    df, caption, note = T.regression_table(
        [{"method": "OLS", "kind": "Raw efficiency (DEA)",
          "envs": ["z1", "z2"],
          "result": ols_regression(y, Z, names=["z1", "z2"])}])
    out = getattr(T, f"table_to_{fmt}")(df, caption, note)
    assert isinstance(out, str) and len(out) > 50
    if fmt == "html":
        assert "<table" in out.lower()
    if fmt == "latex":
        assert "tabular" in out


def test_malmquist_summary_table(mdf):
    df, caption, note = T.malmquist_summary_table(mdf)
    assert len(df) >= 1
    assert "Table" in caption
    assert "<table" in T.table_to_html(df, caption, note).lower()


def test_appendix_export_is_self_contained(tmp_path, solved, mdf):
    """One file, figures embedded, openable in Word without assets."""
    y = solved["eff"]
    Z = solved["ds"][["z1", "z2"]].to_numpy(float)
    df, caption, note = T.regression_table(
        [{"method": "OLS", "kind": "Raw efficiency (DEA)",
          "envs": ["z1", "z2"],
          "result": ols_regression(y, Z, names=["z1", "z2"])}])
    fig = F.fig_eff_hist(y)
    blocks = [
        {"kind": "table", "title": "Determinants", "df": df,
         "caption": caption, "note": note},
        {"kind": "figure", "title": "Distribution",
         "png": F.figure_to_png(fig), "caption": "Figure 1"},
    ]
    out = tmp_path / "appendix.html"
    T.export_appendix_html(str(out), blocks, title="Appendix",
                           version="1.12")
    html = out.read_text(encoding="utf-8")
    assert "<table" in html.lower()
    assert "data:image/png;base64," in html, "figure must be embedded"
    assert "src=\"appendix" not in html, "no external file references"
    assert "1.12" in html


def test_appendix_excel_export(tmp_path, mdf):
    out = tmp_path / "appendix.xlsx"
    df, caption, note = T.malmquist_summary_table(mdf)
    T.export_appendix_excel(str(out), [
        {"kind": "table", "title": "Malmquist", "df": df,
         "caption": caption, "note": note}], version="1.12")
    assert out.exists() and out.stat().st_size > 3000
    import openpyxl
    wb = openpyxl.load_workbook(out)
    assert wb.sheetnames
