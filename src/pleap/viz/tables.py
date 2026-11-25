"""Publication-ready tables and appendix export for PLEAP.

Everything here only formats results that pleap.core has already produced; no
analysis is performed or altered.  Output formats:

  * a three-line (top / header / bottom rule) academic table,
  * HTML that Microsoft Word opens directly,
  * Excel, Markdown and LaTeX.
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


import datetime as _dt
import html as _html

import numpy as np
import pandas as pd

__all__ = ["fmt_coef", "regression_table", "descriptive_table",
           "malmquist_summary_table", "group_test_table",
           "table_to_html", "table_to_markdown", "table_to_latex",
           "export_appendix_html", "export_appendix_excel"]

NOTE_SIG = ("*** p < 0.01, ** p < 0.05, * p < 0.10.")


def _stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt_coef(v, digits=3):
    """Format a coefficient without hiding very small values as 0.000.

    Environmental variables are often measured on huge scales, so the
    coefficients can be of order 1e-9; plain fixed-point formatting would
    print them all as 0.000 and destroy the table.
    """
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    a = abs(float(v))
    if a == 0:
        return f"{0.0:.{digits}f}"
    if a >= 10 ** (-digits):
        return f"{float(v):.{digits}f}"
    return f"{float(v):.{max(2, digits - 1)}e}"


def _coef_cell(coef, se, p, se_below, digits):
    star = _stars(p)
    c = f"{fmt_coef(coef, digits)}{star}"
    s = f"({fmt_coef(se, digits)})"
    return (c, s) if se_below else (f"{c} {s}", None)


def regression_table(runs, se_below=False, digits=3, table_no=1,
                     caption="Second-stage regression results"):
    """Build a publication table from one or more stored regression runs.

    ``runs`` is a list of dicts as recorded by the GUI::

        {"method": "Tobit", "kind": "Raw efficiency (DEA)",
         "envs": [...], "result": <dict returned by pleap.core.secondstage>}

    Returns ``(DataFrame, caption, note)``.
    """
    if not runs:
        raise ValueError("No regression results have been stored yet.")

    cols = [f"({i + 1})" for i in range(len(runs))]

    # ---- collect the union of variable names, in first-seen order ---------
    order, seen = [], set()
    for r in runs:
        for nm in list(r["result"]["summary"].index):
            if nm not in seen:
                seen.add(nm); order.append(nm)
    body_vars = [v for v in order if v != "Log(scale)"]
    tail_vars = [v for v in order if v == "Log(scale)"]

    rows, index = [], []

    def add(label, values):
        index.append(label); rows.append(values)

    add("Dependent variable", [r.get("kind", "") for r in runs])
    add("Estimator", [r.get("method", "") for r in runs])

    for nm in body_vars + tail_vars:
        main, sub = [], []
        for r in runs:
            s = r["result"]["summary"]
            if nm not in s.index:
                main.append(""); sub.append("")
                continue
            # Prefer the full-precision metadata: the displayed summary is
            # rounded to 6 dp, which would print coefficients of order 1e-9
            # as 0.000 and destroy the table.
            pos = list(s.index).index(nm)
            raw = s.attrs
            pcol = "P>|z|" if "P>|z|" in s.columns else "P>|t|"
            row = s.loc[nm]
            def pick(key, fallback):
                arr = raw.get(key)
                try:
                    if arr is not None and len(arr) > pos:
                        return float(arr[pos])
                except Exception:
                    pass
                return float(fallback)
            coef = pick("coef_raw", row["coef"])
            se = pick("se_raw", row["std_err"])
            pv = pick("p_raw", row[pcol])
            c, t = _coef_cell(coef, se, pv, se_below, digits)
            main.append(c); sub.append(t or "")
        disp = "Constant" if nm == "const" else nm
        add(disp, main)
        if se_below and any(sub):
            add("", sub)

    # ---- diagnostics ------------------------------------------------------
    add("Observations", [str(r["result"].get("n", "")) for r in runs])

    def diag(label, key, fmt="{:.4f}"):
        vals, any_ = [], False
        for r in runs:
            v = r["result"].get(key)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                vals.append("")
            else:
                vals.append(fmt.format(v)); any_ = True
        if any_:
            add(label, vals)

    diag("Log likelihood", "loglik")
    diag("Sigma", "sigma", "{:.5f}")
    diag("R-squared", "r2")
    diag("Adjusted R-squared", "adj_r2")

    # Wald / F carry their own degrees of freedom, so build them by hand.
    wald, fstat, any_w, any_f = [], [], False, False
    for r in runs:
        res = r["result"]
        w, wdf, wp = res.get("wald"), res.get("wald_df"), res.get("wald_p")
        if w is not None and np.isfinite(w) and wdf:
            wald.append(f"{w:.3f}{_stars(wp)} (df={wdf})"); any_w = True
        else:
            wald.append("")
        f, fp = res.get("fvalue"), res.get("f_pvalue")
        if f is not None and np.isfinite(f):
            fstat.append(f"{f:.3f}{_stars(fp)}"); any_f = True
        else:
            fstat.append("")
    if any_w:
        add("Wald chi-square", wald)
    if any_f:
        add("F-statistic", fstat)

    conv = [("Yes" if r["result"].get("converged") else "No")
            if "converged" in r["result"] else "" for r in runs]
    if any(conv):
        add("Converged", conv)

    df = pd.DataFrame(rows, index=index, columns=cols)
    df.index.name = "Variable"
    note = ("Note: " + NOTE_SIG + "  Standard errors in parentheses"
            + (" below the coefficients." if se_below else ".")
            + "  Coefficients smaller than 1e-3 are shown in scientific notation.")
    return df, f"Table {table_no}. {caption}", note


def descriptive_table(dea_df, table_no=1, model="", orient=""):
    """Summary statistics of the DEA efficiency scores (paper-ready)."""
    e = pd.to_numeric(dea_df["Efficiency"], errors="coerce").dropna()
    rows = [("Number of DMUs", f"{len(e):d}"),
            ("Efficient DMUs (score = 1)", f"{int((e >= 1 - 1e-6).sum()):d}"),
            ("Mean", f"{e.mean():.4f}"),
            ("Standard deviation", f"{e.std(ddof=1):.4f}"),
            ("Minimum", f"{e.min():.4f}"),
            ("1st quartile", f"{e.quantile(.25):.4f}"),
            ("Median", f"{e.median():.4f}"),
            ("3rd quartile", f"{e.quantile(.75):.4f}"),
            ("Maximum", f"{e.max():.4f}")]
    if "RTS" in dea_df.columns:
        for lab, cnt in dea_df["RTS"].value_counts().items():
            rows.append((f"Returns to scale: {lab}", f"{int(cnt):d}"))
    df = pd.DataFrame(rows, columns=["Statistic", "Value"]).set_index("Statistic")
    cap = f"Table {table_no}. Descriptive statistics of efficiency scores"
    if model:
        cap += f" ({model.upper()}, {orient}-oriented)"
    return df, cap, ""


def malmquist_summary_table(mdf, table_no=1):
    """Geometric means of MI / EC / TC by period."""
    def g(v):
        v = pd.to_numeric(v, errors="coerce").dropna()
        v = v[v > 0]
        return float(np.exp(np.log(v).mean())) if len(v) else np.nan

    per = sorted(mdf["t(Period)"].unique())
    rows = []
    for p in per:
        d = mdf[mdf["t(Period)"] == p]
        mi = g(d["MI(t-1, t)"])
        rows.append({"Period (t)": str(p),
                     "EC": f"{g(d['EC(t-1, t)']):.4f}",
                     "TC": f"{g(d['TC(t-1, t)']):.4f}",
                     "MI": f"{mi:.4f}",
                     "DMUs": f"{len(d):d}",
                     "MI >= 1": f"{int((pd.to_numeric(d['MI(t-1, t)'], errors='coerce') >= 1).sum()):d}"})
    rows.append({"Period (t)": "Overall (geometric mean)",
                 "EC": f"{g(mdf['EC(t-1, t)']):.4f}",
                 "TC": f"{g(mdf['TC(t-1, t)']):.4f}",
                 "MI": f"{g(mdf['MI(t-1, t)']):.4f}",
                 "DMUs": f"{len(mdf):d}",
                 "MI >= 1": f"{int((pd.to_numeric(mdf['MI(t-1, t)'], errors='coerce') >= 1).sum()):d}"})
    df = pd.DataFrame(rows).set_index("Period (t)")
    note = ("Note: values are geometric means. MI = EC x TC; "
            "a value above 1 indicates productivity growth.")
    return df, f"Table {table_no}. Malmquist productivity index by period", note


def group_test_table(runs, table_no=1):
    """One row per stored group-comparison test."""
    if not runs:
        raise ValueError("No group-test results have been stored yet.")
    rows = []
    for r in runs:
        res = r["result"]
        stat_lbl = "Mean" if "mean" in res else "Median"
        vals = res.get("mean", res.get("median", []))
        summary = "; ".join(f"{lv} (n={n}): {v:.4f}"
                            for lv, n, v in zip(res["groups"], res["n"], vals))
        row = {"Dependent variable": r.get("kind", ""),
               "Test": res.get("test", ""),
               "Statistic": f"{res.get('stat', float('nan')):.4f}",
               "p-value": f"{res.get('p', float('nan')):.4f}{_stars(res.get('p', np.nan))}",
               f"Group {stat_lbl.lower()}s": summary}
        if "levene" in res:
            row["Levene p"] = f"{res['levene']['p']:.4f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.index = [f"({i+1})" for i in range(len(df))]
    df.index.name = "No."
    return (df, f"Table {table_no}. Group comparison tests",
            "Note: " + NOTE_SIG)


# --------------------------------------------------------------------------- #
#  Renderers                                                                  #
# --------------------------------------------------------------------------- #
_CSS = """
body { font-family: 'Times New Roman', serif; font-size: 11pt; color:#111;
       margin: 28px 34px; }
h1 { font-size: 16pt; margin: 0 0 2px 0; }
h2 { font-size: 12.5pt; margin: 26px 0 8px 0; border-bottom: 1px solid #bbb;
     padding-bottom: 3px; }
p.meta { color:#666; font-size: 9pt; margin: 0 0 18px 0; }
p.caption { font-weight: bold; margin: 0 0 5px 0; font-size: 10.5pt; }
p.note { font-size: 9pt; color:#333; margin: 5px 0 16px 0; }
table.pleap { border-collapse: collapse; margin: 0 0 6px 0; font-size: 10pt; }
table.pleap th, table.pleap td { padding: 4px 11px; text-align: right;
     border: none; white-space: nowrap; }
table.pleap th:first-child, table.pleap td:first-child { text-align: left; }
table.pleap thead tr { border-top: 1.6px solid #000; border-bottom: 1px solid #000; }
table.pleap tbody tr:last-child { border-bottom: 1.6px solid #000; }
table.pleap tbody tr.rule td { border-bottom: 0.8px solid #666; }
pre { font-family: Consolas, monospace; font-size: 9.5pt; background:#f7f7f9;
      padding: 10px 12px; border:1px solid #e3e3e8; white-space: pre-wrap; }
img { max-width: 100%; margin: 6px 0 4px 0; }
"""


def table_to_html(df, caption=None, note=None, rule_after=None):
    """Three-line academic table as HTML."""
    out = []
    if caption:
        out.append(f'<p class="caption">{_html.escape(str(caption))}</p>')
    out.append('<table class="pleap"><thead><tr>')
    first = df.index.name or ""
    out.append(f"<th>{_html.escape(str(first))}</th>")
    for c in df.columns:
        out.append(f"<th>{_html.escape(str(c))}</th>")
    out.append("</tr></thead><tbody>")
    rule_after = set(rule_after or [])
    for idx, row in df.iterrows():
        cls = ' class="rule"' if idx in rule_after else ""
        out.append(f"<tr{cls}><td>{_html.escape(str(idx))}</td>")
        for v in row:
            out.append(f"<td>{_html.escape('' if v is None else str(v))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    if note:
        out.append(f'<p class="note">{_html.escape(str(note))}</p>')
    return "\n".join(out)


def table_to_markdown(df, caption=None, note=None):
    lines = []
    if caption:
        lines += [f"**{caption}**", ""]
    head = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] * len(head)) + "|")
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] +
                                       ["" if v is None else str(v) for v in row]) + " |")
    if note:
        lines += ["", f"_{note}_"]
    return "\n".join(lines)


def table_to_latex(df, caption=None, note=None, label="tab:pleap"):
    def esc(s):
        s = str(s)
        for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
                     ("{", r"\{"), ("}", r"\}"), ("^", r"\^{}"),
                     ("~", r"\~{}"), (">=", r"$\geq$"), ("<", r"$<$")]:
            s = s.replace(a, b)
        return s

    ncol = len(df.columns)
    out = [r"\begin{table}[htbp]", r"\centering"]
    if caption:
        out.append(r"\caption{%s}" % esc(caption))
    out.append(r"\label{%s}" % label)
    out.append(r"\begin{tabular}{l%s}" % ("r" * ncol))
    out.append(r"\hline\hline")
    out.append(" & ".join([esc(df.index.name or "")] +
                          [esc(c) for c in df.columns]) + r" \\")
    out.append(r"\hline")
    for idx, row in df.iterrows():
        out.append(" & ".join([esc(idx)] + [esc(v) for v in row]) + r" \\")
    out.append(r"\hline\hline")
    out.append(r"\end{tabular}")
    if note:
        out.append(r"\begin{tablenotes}\footnotesize %s \end{tablenotes}" % esc(note))
    out.append(r"\end{table}")
    return "\n".join(out)


def table_to_text(df, caption=None, note=None):
    """Fixed-width three-line table for on-screen preview."""
    body = df.copy().astype(str)
    heads = [str(df.index.name or "")] + [str(c) for c in df.columns]
    widths = [max(len(heads[0]), *(len(str(i)) for i in body.index))]
    for j, c in enumerate(body.columns):
        widths.append(max(len(heads[j + 1]), *(len(v) for v in body[c])))
    def line(cells, ljust0=True):
        out = [cells[0].ljust(widths[0]) if ljust0 else cells[0].rjust(widths[0])]
        out += [cells[j + 1].rjust(widths[j + 1]) for j in range(len(body.columns))]
        return "  ".join(out).rstrip()
    total = sum(widths) + 2 * len(widths) - 2
    lines = []
    if caption:
        lines += [caption, ""]
    lines.append("=" * total)
    lines.append(line(heads))
    lines.append("-" * total)
    for idx, row in body.iterrows():
        lines.append(line([str(idx)] + [str(v) for v in row]))
    lines.append("=" * total)
    if note:
        lines += ["", note]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Appendix export                                                            #
# --------------------------------------------------------------------------- #
def _stamp(version=""):
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    v = f"PLEAP v{version}" if version else "PLEAP"
    return (f"Generated by {v} - Incheon National University, Productivity Lab "
            f"- {ts}")


def export_appendix_html(path, blocks, title="Appendix", version="",
                         subtitle=""):
    """Write a self-contained HTML appendix (opens directly in MS Word).

    ``blocks`` is a list of dicts:
        {"kind": "table",  "title": str, "df": DataFrame,
                           "caption": str, "note": str}
        {"kind": "text",   "title": str, "text": str}
        {"kind": "figure", "title": str, "png": bytes, "caption": str}
    """
    import base64
    parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
             f"<title>{_html.escape(title)}</title>",
             f"<style>{_CSS}</style></head><body>",
             f"<h1>{_html.escape(title)}</h1>"]
    if subtitle:
        parts.append(f"<p class='meta'>{_html.escape(subtitle)}</p>")
    parts.append(f"<p class='meta'>{_html.escape(_stamp(version))}</p>")

    for b in blocks:
        if b.get("title"):
            parts.append(f"<h2>{_html.escape(str(b['title']))}</h2>")
        kind = b.get("kind", "table")
        if kind == "table":
            parts.append(table_to_html(b["df"], b.get("caption"), b.get("note"),
                                       b.get("rule_after")))
        elif kind == "text":
            parts.append(f"<pre>{_html.escape(str(b.get('text', '')))}</pre>")
        elif kind == "figure":
            b64 = base64.b64encode(b["png"]).decode("ascii")
            if b.get("caption"):
                parts.append(f'<p class="caption">{_html.escape(str(b["caption"]))}</p>')
            parts.append(f'<img src="data:image/png;base64,{b64}" alt="figure">')
    parts.append("</body></html>")
    html_text = "\n".join(parts)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html_text)
    return path


def export_appendix_excel(path, blocks, version=""):
    """Write the same appendix as a formatted Excel workbook."""
    used = set()

    def sheet_name(t):
        base = "".join(ch for ch in str(t) if ch not in "[]:*?/\\")[:28] or "Sheet"
        nm, k = base, 2
        while nm in used:
            nm = f"{base[:26]}_{k}"; k += 1
        used.add(nm)
        return nm

    with pd.ExcelWriter(path, engine="openpyxl") as wr:
        info = pd.DataFrame({"PLEAP": [_stamp(version)]})
        info.to_excel(wr, sheet_name="Info", index=False)
        for b in blocks:
            if b.get("kind", "table") == "table":
                sn = sheet_name(b.get("title", "Table"))
                cap = b.get("caption") or b.get("title") or ""
                pd.DataFrame({cap: []}).to_excel(wr, sheet_name=sn, index=False)
                b["df"].to_excel(wr, sheet_name=sn, startrow=2)
                if b.get("note"):
                    ws = wr.sheets[sn]
                    ws.cell(row=len(b["df"]) + 5, column=1, value=b["note"])
            elif b.get("kind") == "text":
                sn = sheet_name(b.get("title", "Text"))
                pd.DataFrame({"": str(b.get("text", "")).split("\n")}).to_excel(
                    wr, sheet_name=sn, index=False)
    return path
