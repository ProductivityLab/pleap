"""The Visualization tab of PLEAP.

Implemented as a mixin so that main_window.py stays readable.  Everything here
consumes results that pleap.core has already produced - no analysis is run,
repeated or modified in this file.
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


import os
import traceback

import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

from pleap import __version__
from pleap.viz import figures as VF
from pleap.viz import tables as VT

try:
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavToolbar)
    MPL_OK = True
except Exception:                                    # pragma: no cover
    MPL_OK = False

CH_FRONTIER = "DEA - Efficiency frontier & DMU positions"
CH_HIST     = "DEA - Distribution of efficiency scores"
CH_BAR      = "DEA - Efficiency by DMU (bar)"
CH_BOOT     = "Bootstrap - Bias-corrected efficiency with CI"
CH_MTREND   = "Malmquist - MI / EC / TC trend by period"
CH_MQUAD    = "Malmquist - EC vs TC quadrant"
CH_MBAR     = "Malmquist - Index by DMU (bar)"
CH_MBOX     = "Malmquist - Distribution by period (box)"
CHARTS = [CH_FRONTIER, CH_HIST, CH_BAR, CH_BOOT,
          CH_MTREND, CH_MQUAD, CH_MBAR, CH_MBOX]

TB_REG_ALL = "Second-stage regression - all stored models"
TB_REG_ONE = "Second-stage regression - most recent model"
TB_DESC    = "Descriptive statistics of efficiency scores"
TB_MALM    = "Malmquist productivity index by period"
TB_GROUP   = "Group comparison tests"
TABLES = [TB_REG_ALL, TB_REG_ONE, TB_DESC, TB_MALM, TB_GROUP]

FMTS = ["Plain text (preview)", "HTML (opens in Word)", "Markdown", "LaTeX"]


class VizTabMixin:
    # ------------------------------------------------------------------ #
    #  Tab construction                                                   #
    # ------------------------------------------------------------------ #
    def _tab_viz(self):
        self._appendix_figs = []
        self._last_fig = None
        self._last_table = None
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        inner = QtWidgets.QTabWidget()
        inner.addTab(self._viz_figures_tab(), "A. Figures")
        inner.addTab(self._viz_tables_tab(), "B. Publication tables")
        inner.addTab(self._viz_appendix_tab(), "C. Appendix export")
        v.addWidget(inner, 1)
        return w

    # ---------------------------- A. Figures --------------------------- #
    def _viz_figures_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)

        h = QtWidgets.QHBoxLayout()
        self.cmb_chart = QtWidgets.QComboBox(); self.cmb_chart.addItems(CHARTS)
        self.cmb_chart.setMinimumWidth(320)
        self.cmb_chart.currentIndexChanged.connect(self._viz_sync)
        h.addWidget(QtWidgets.QLabel("Chart:")); h.addWidget(self.cmb_chart)

        self.cmb_vperiod = QtWidgets.QComboBox(); self.cmb_vperiod.addItem("All")
        h.addWidget(QtWidgets.QLabel("Period:")); h.addWidget(self.cmb_vperiod)

        self.cmb_vindex = QtWidgets.QComboBox(); self.cmb_vindex.addItems(["MI", "EC", "TC"])
        h.addWidget(QtWidgets.QLabel("Index:")); h.addWidget(self.cmb_vindex)

        self.spn_top = QtWidgets.QSpinBox(); self.spn_top.setRange(0, 2000)
        self.spn_top.setValue(30)
        self.spn_top.setToolTip("Number of DMUs to show (0 = all).")
        h.addWidget(QtWidgets.QLabel("Top N:")); h.addWidget(self.spn_top)

        self.chk_labels = QtWidgets.QCheckBox("DMU labels"); self.chk_labels.setChecked(True)
        h.addWidget(self.chk_labels)
        h.addStretch(1)
        v.addLayout(h)

        # ---- appearance -------------------------------------------------
        h3 = QtWidgets.QHBoxLayout()
        self.cmb_palette = QtWidgets.QComboBox()
        self.cmb_palette.addItems(list(VF.PALETTES.keys()))
        self.cmb_palette.setMinimumWidth(180)
        h3.addWidget(QtWidgets.QLabel("Colour:")); h3.addWidget(self.cmb_palette)

        self.cmb_style = QtWidgets.QComboBox(); self.cmb_style.addItems(VF.STYLES)
        h3.addWidget(QtWidgets.QLabel("Style:")); h3.addWidget(self.cmb_style)

        self.spn_fscale = QtWidgets.QDoubleSpinBox()
        self.spn_fscale.setRange(0.7, 1.8); self.spn_fscale.setSingleStep(0.05)
        self.spn_fscale.setValue(1.0)
        self.spn_fscale.setToolTip("Font size multiplier for every chart.")
        h3.addWidget(QtWidgets.QLabel("Font scale:")); h3.addWidget(self.spn_fscale)

        self.chk_live = QtWidgets.QCheckBox("Apply instantly")
        self.chk_live.setChecked(True)
        self.chk_live.setToolTip("Redraw the current chart as soon as the "
                                 "appearance is changed.")
        h3.addWidget(self.chk_live)
        for wdg in (self.cmb_palette, self.cmb_style):
            wdg.currentIndexChanged.connect(self._theme_changed)
        self.spn_fscale.valueChanged.connect(self._theme_changed)
        h3.addStretch(1)
        v.addLayout(h3)

        h2 = QtWidgets.QHBoxLayout()
        for text, slot in [("Draw chart", self.viz_draw),
                           ("Save figure...", self.viz_save_fig),
                           ("Add to appendix", self.viz_add_fig)]:
            b = QtWidgets.QPushButton(text); b.clicked.connect(slot); h2.addWidget(b)
        h2.addStretch(1)
        v.addLayout(h2)

        self.viz_holder = QtWidgets.QVBoxLayout()
        box = QtWidgets.QWidget(); box.setLayout(self.viz_holder)
        v.addWidget(box, 1)
        if not MPL_OK:
            self.viz_holder.addWidget(QtWidgets.QLabel(
                "matplotlib is not available - charts are disabled."))
        else:
            self._viz_msg = QtWidgets.QLabel(
                "Run an analysis, then choose a chart and press <b>Draw chart</b>.")
            self._viz_msg.setTextFormat(QtCore.Qt.RichText)
            self._viz_msg.setAlignment(QtCore.Qt.AlignCenter)
            self._viz_msg.setStyleSheet("color:#666;")
            self.viz_holder.addWidget(self._viz_msg)
        self._viz_sync()
        return w

    def _apply_theme(self):
        """Push the current appearance selection into the figure module."""
        try:
            VF.set_theme(palette=self.cmb_palette.currentText(),
                         style=self.cmb_style.currentText(),
                         font_scale=self.spn_fscale.value())
        except Exception:
            pass

    def _theme_changed(self, *_):
        self._apply_theme()
        if getattr(self, "chk_live", None) is not None and \
                self.chk_live.isChecked() and self._last_fig is not None:
            self.viz_draw()

    def _viz_sync(self):
        try:
            ch = self.cmb_chart.currentText()
        except Exception:
            return
        self.cmb_vperiod.setEnabled(ch in (CH_MQUAD, CH_MBAR))
        self.cmb_vindex.setEnabled(ch in (CH_MBAR, CH_MBOX))
        self.spn_top.setEnabled(ch in (CH_BAR, CH_BOOT, CH_MBAR))
        self.chk_labels.setEnabled(ch in (CH_FRONTIER, CH_MQUAD))
        if ch.startswith("Malmquist") and "malm" in self.results:
            cur = self.cmb_vperiod.currentText()
            per = ["All"] + [str(p) for p in
                             sorted(self.results["malm"]["df"]["t(Period)"].unique())]
            if [self.cmb_vperiod.itemText(i)
                    for i in range(self.cmb_vperiod.count())] != per:
                self.cmb_vperiod.blockSignals(True)
                self.cmb_vperiod.clear(); self.cmb_vperiod.addItems(per)
                k = self.cmb_vperiod.findText(cur)
                self.cmb_vperiod.setCurrentIndex(k if k >= 0 else 0)
                self.cmb_vperiod.blockSignals(False)

    def _need(self, key, label):
        if key not in self.results:
            raise ValueError(f"Run the {label} analysis first.")
        return self.results[key]

    def _period_value(self):
        txt = self.cmb_vperiod.currentText()
        if txt in ("", "All"):
            return "All"
        per = self.results["malm"]["df"]["t(Period)"]
        for p in per.unique():
            if str(p) == txt:
                return p
        return "All"

    def viz_draw(self):
        if not MPL_OK:
            QtWidgets.QMessageBox.warning(self, "Charts", "matplotlib is not installed.")
            return
        self._apply_theme()
        ch = self.cmb_chart.currentText()
        top = self.spn_top.value() or None
        labels_on = self.chk_labels.isChecked()
        try:
            if ch == CH_FRONTIER:
                r = self._need("dea", "DEA")
                if "X" not in r:
                    raise ValueError("Please re-run the DEA analysis (tab 2) once "
                                     "so the chart can access the input/output data.")
                fig = VF.fig_frontier(r["X"], r["Y"],
                                      r["df"]["Efficiency"].values,
                                      model=r["model"], orient=r["orient"],
                                      labels=list(r["df"]["DMU"]),
                                      ins=r.get("ins"), outs=r.get("outs"),
                                      show_labels=labels_on)
            elif ch == CH_HIST:
                r = self._need("dea", "DEA")
                fig = VF.fig_eff_hist(r["df"]["Efficiency"].values,
                                      r["model"], r["orient"])
            elif ch == CH_BAR:
                r = self._need("dea", "DEA")
                fig = VF.fig_eff_bar(r["df"]["Efficiency"].values,
                                     list(r["df"]["DMU"]), top=top,
                                     model=r["model"], orient=r["orient"])
            elif ch == CH_BOOT:
                r = self._need("boot", "bootstrap")
                fig = VF.fig_boot_ci(r["df"], top=top)
            elif ch == CH_MTREND:
                r = self._need("malm", "Malmquist")
                fig = VF.fig_malm_trend(r["df"])
            elif ch == CH_MQUAD:
                r = self._need("malm", "Malmquist")
                fig = VF.fig_malm_quadrant(r["df"], period=self._period_value(),
                                           show_labels=labels_on)
            elif ch == CH_MBAR:
                r = self._need("malm", "Malmquist")
                fig = VF.fig_malm_bar(r["df"], index=self.cmb_vindex.currentText(),
                                      period=self._period_value(), top=top)
            else:
                r = self._need("malm", "Malmquist")
                fig = VF.fig_malm_box(r["df"], index=self.cmb_vindex.currentText())
        except Exception as e:
            if isinstance(e, ValueError):
                QtWidgets.QMessageBox.information(self, "Visualization", str(e))
            else:
                QtWidgets.QMessageBox.critical(self, "Error", traceback.format_exc())
            return
        self._show_fig(fig)

    def _show_fig(self, fig):
        while self.viz_holder.count():
            item = self.viz_holder.takeAt(0)
            wd = item.widget()
            if wd is not None:
                wd.setParent(None)
        canvas = FigureCanvas(fig)
        toolbar = NavToolbar(canvas, self)
        self.viz_holder.addWidget(toolbar)
        self.viz_holder.addWidget(canvas, 1)
        canvas.draw()
        self._last_fig = fig

    def viz_save_fig(self):
        if self._last_fig is None:
            QtWidgets.QMessageBox.information(self, "Save figure", "Draw a chart first.")
            return
        name = self.cmb_chart.currentText().split(" - ")[-1][:40].replace(" ", "_")
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save figure", f"{name}.png",
            "PNG image (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tif)")
        if not fn:
            return
        try:
            self._last_fig.savefig(fn, dpi=300, bbox_inches="tight",
                                   facecolor=self._last_fig.get_facecolor())
            QtWidgets.QMessageBox.information(
                self, "Save figure", f"Saved at 300 dpi:\n{fn}")
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Error", traceback.format_exc())

    def viz_add_fig(self):
        if self._last_fig is None:
            QtWidgets.QMessageBox.information(self, "Appendix", "Draw a chart first.")
            return
        title = self.cmb_chart.currentText()
        try:
            png = VF.figure_to_png(self._last_fig, dpi=200)
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Error", traceback.format_exc()); return
        self._appendix_figs.append({"title": title, "png": png})
        self._refresh_fig_list()
        QtWidgets.QMessageBox.information(
            self, "Appendix",
            f"Added to the appendix ({len(self._appendix_figs)} figure(s) collected).")

    # ------------------------ B. Publication tables -------------------- #
    def _viz_tables_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        h = QtWidgets.QHBoxLayout()
        self.cmb_ptable = QtWidgets.QComboBox(); self.cmb_ptable.addItems(TABLES)
        self.cmb_ptable.setMinimumWidth(300)
        h.addWidget(QtWidgets.QLabel("Table:")); h.addWidget(self.cmb_ptable)
        self.cmb_se = QtWidgets.QComboBox()
        self.cmb_se.addItems(["SE in parentheses", "SE below coefficient"])
        h.addWidget(QtWidgets.QLabel("Style:")); h.addWidget(self.cmb_se)
        self.cmb_fmt = QtWidgets.QComboBox(); self.cmb_fmt.addItems(FMTS)
        h.addWidget(QtWidgets.QLabel("Format:")); h.addWidget(self.cmb_fmt)
        self.spn_tno = QtWidgets.QSpinBox(); self.spn_tno.setRange(1, 99); self.spn_tno.setValue(1)
        h.addWidget(QtWidgets.QLabel("Table no.:")); h.addWidget(self.spn_tno)
        h.addStretch(1)
        v.addLayout(h)

        h2 = QtWidgets.QHBoxLayout()
        for text, slot in [("Build table", self.viz_build_table),
                           ("Copy to clipboard", self.viz_copy_table),
                           ("Save as...", self.viz_save_table)]:
            b = QtWidgets.QPushButton(text); b.clicked.connect(slot); h2.addWidget(b)
        self.lbl_runs = QtWidgets.QLabel("")
        h2.addWidget(self.lbl_runs); h2.addStretch(1)
        v.addLayout(h2)

        self.txt_ptable = QtWidgets.QPlainTextEdit(); self.txt_ptable.setReadOnly(True)
        self.txt_ptable.setFont(QtGui.QFont("Consolas", 10))
        self.txt_ptable.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        v.addWidget(self.txt_ptable, 1)
        return w

    def _build_table_df(self, which):
        """Return (DataFrame, caption, note) for the requested table."""
        no = self.spn_tno.value()
        se_below = self.cmb_se.currentIndex() == 1
        if which in (TB_REG_ALL, TB_REG_ONE):
            runs = self.results.get("reg_runs", [])
            if not runs:
                raise ValueError("Run a 2nd-stage regression first (tab 5).")
            if which == TB_REG_ONE:
                runs = runs[-1:]
            return VT.regression_table(runs, se_below=se_below, table_no=no)
        if which == TB_DESC:
            r = self._need("dea", "DEA")
            return VT.descriptive_table(r["df"], table_no=no,
                                        model=r["model"], orient=r["orient"])
        if which == TB_MALM:
            r = self._need("malm", "Malmquist")
            return VT.malmquist_summary_table(r["df"], table_no=no)
        runs = self.results.get("group_runs", [])
        if not runs:
            raise ValueError("Run a group test first (tab 6).")
        return VT.group_test_table(runs, table_no=no)

    def viz_build_table(self):
        try:
            df, cap, note = self._build_table_df(self.cmb_ptable.currentText())
        except Exception as e:
            if isinstance(e, ValueError):
                QtWidgets.QMessageBox.information(self, "Publication table", str(e))
            else:
                QtWidgets.QMessageBox.critical(self, "Error", traceback.format_exc())
            return
        self._last_table = (df, cap, note)
        fmt = self.cmb_fmt.currentText()
        if fmt.startswith("HTML"):
            text = VT.table_to_html(df, cap, note)
        elif fmt.startswith("Markdown"):
            text = VT.table_to_markdown(df, cap, note)
        elif fmt.startswith("LaTeX"):
            text = VT.table_to_latex(df, cap, note)
        else:
            text = VT.table_to_text(df, cap, note)
        self.txt_ptable.setPlainText(text)
        n = len(self.results.get("reg_runs", []))
        m = len(self.results.get("group_runs", []))
        self.lbl_runs.setText(f"stored: {n} regression run(s), {m} group test(s)")

    def viz_copy_table(self):
        if not self.txt_ptable.toPlainText():
            QtWidgets.QMessageBox.information(self, "Copy", "Build a table first.")
            return
        QtWidgets.QApplication.clipboard().setText(self.txt_ptable.toPlainText())
        self.lbl_status.setText("Table copied to clipboard.")

    def viz_save_table(self):
        if self._last_table is None:
            QtWidgets.QMessageBox.information(self, "Save", "Build a table first.")
            return
        df, cap, note = self._last_table
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save table", "table.html",
            "HTML for Word (*.html);;Excel (*.xlsx);;Markdown (*.md);;"
            "LaTeX (*.tex);;Text (*.txt)")
        if not fn:
            return
        ext = os.path.splitext(fn)[1].lower()
        try:
            if ext == ".xlsx":
                VT.export_appendix_excel(
                    fn, [{"kind": "table", "title": cap, "df": df,
                          "caption": cap, "note": note}], version=__version__)
            else:
                if ext == ".md":
                    text = VT.table_to_markdown(df, cap, note)
                elif ext == ".tex":
                    text = VT.table_to_latex(df, cap, note)
                elif ext == ".txt":
                    text = VT.table_to_text(df, cap, note)
                else:
                    text = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
                            f"<style>{VT._CSS}</style></head><body>"
                            + VT.table_to_html(df, cap, note) + "</body></html>")
                with open(fn, "w", encoding="utf-8") as fh:
                    fh.write(text)
            QtWidgets.QMessageBox.information(self, "Save table", f"Saved:\n{fn}")
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Error", traceback.format_exc())

    # -------------------------- C. Appendix ---------------------------- #
    def _viz_appendix_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        form = QtWidgets.QFormLayout()
        self.ed_apx_title = QtWidgets.QLineEdit("Appendix")
        self.ed_apx_sub = QtWidgets.QLineEdit("")
        self.ed_apx_sub.setPlaceholderText("Optional subtitle, e.g. the paper title")
        form.addRow("Title:", self.ed_apx_title)
        form.addRow("Subtitle:", self.ed_apx_sub)
        v.addLayout(form)

        box = QtWidgets.QGroupBox("Sections to include")
        g = QtWidgets.QGridLayout(box)
        self.chk_apx = {}
        items = [("desc",  "Descriptive statistics of efficiency scores"),
                 ("dea",   "DEA results (full table: scores, RTS, peers, slacks)"),
                 ("boot",  "Bootstrap results (bias-corrected scores and CI)"),
                 ("malms", "Malmquist summary by period"),
                 ("malm",  "Malmquist results (full table)"),
                 ("reg",   "2nd-stage regression (publication format)"),
                 ("group", "Group comparison tests"),
                 ("figs",  "Figures collected from the Figures tab"),
                 ("log",   "Analysis settings log")]
        for i, (key, text) in enumerate(items):
            cb = QtWidgets.QCheckBox(text); cb.setChecked(True)
            self.chk_apx[key] = cb
            g.addWidget(cb, i // 2, i % 2)
        v.addWidget(box)

        h = QtWidgets.QHBoxLayout()
        for text, slot in [("Export appendix (HTML / Word)...", self.viz_export_html),
                           ("Export appendix (Excel)...", self.viz_export_xlsx),
                           ("Clear collected figures", self.viz_clear_figs)]:
            b = QtWidgets.QPushButton(text); b.clicked.connect(slot); h.addWidget(b)
        h.addStretch(1)
        v.addLayout(h)

        v.addWidget(QtWidgets.QLabel("Collected figures:"))
        self.lst_apx_figs = QtWidgets.QListWidget()
        v.addWidget(self.lst_apx_figs, 1)
        hint = QtWidgets.QLabel(
            "The HTML appendix embeds the figures, so it is a single self-contained "
            "file: open it in Word and save as .docx to paste into a manuscript.")
        hint.setWordWrap(True); hint.setStyleSheet("color:#666;")
        v.addWidget(hint)
        return w

    def _refresh_fig_list(self):
        self.lst_apx_figs.clear()
        for i, f in enumerate(self._appendix_figs, 1):
            self.lst_apx_figs.addItem(f"Figure {i}. {f['title']}")

    def viz_clear_figs(self):
        self._appendix_figs = []
        self._refresh_fig_list()
        self.lbl_status.setText("Collected figures cleared.")

    def _settings_log(self):
        lines = []
        if getattr(self, "df", None) is not None:
            lines.append(f"Observations in data file: {len(self.df)}")
        try:
            ins = [x.text() for x in self.lst_inputs.selectedItems()]
            outs = [x.text() for x in self.lst_outputs.selectedItems()]
            envs = [x.text() for x in self.lst_envs.selectedItems()]
            if ins:  lines.append("Inputs: " + ", ".join(ins))
            if outs: lines.append("Outputs: " + ", ".join(outs))
            if envs: lines.append("Environmental variables: " + ", ".join(envs))
        except Exception:
            pass
        r = self.results.get("dea")
        if r:
            lines.append(f"DEA model: {r['model']}, {r['orient']}-oriented, "
                         f"{len(r['df'])} DMUs")
        r = self.results.get("boot")
        if r:
            lines.append(f"Bootstrap: {r['model']}, {r['orient']}-oriented "
                         f"(Simar-Wilson smoothed homogeneous bootstrap)")
        r = self.results.get("malm")
        if r:
            per = sorted(r["df"]["t(Period)"].unique())
            lines.append(f"Malmquist periods: {', '.join(str(p) for p in per)}")
        for run in self.results.get("reg_runs", []):
            lines.append(f"Regression: {run['method']} on {run['kind']} "
                         f"[{', '.join(run.get('envs', []))}]")
        return "\n".join(lines) or "(no analysis has been run yet)"

    def _appendix_blocks(self):
        blocks = []
        letters = iter("ABCDEFGHIJ")
        tno = [0]

        def nxt():
            tno[0] += 1
            return tno[0]

        if self.chk_apx["desc"].isChecked() and "dea" in self.results:
            r = self.results["dea"]
            df, cap, note = VT.descriptive_table(r["df"], table_no=nxt(),
                                                 model=r["model"], orient=r["orient"])
            blocks.append({"kind": "table", "df": df, "caption": cap, "note": note,
                           "title": f"Appendix {next(letters)}. Descriptive statistics"})
        if self.chk_apx["dea"].isChecked() and "dea" in self.results:
            d = self.results["dea"]["df"].copy()
            for c in d.columns:
                if pd.api.types.is_float_dtype(d[c]):
                    d[c] = d[c].round(6)
            blocks.append({"kind": "table", "df": d.set_index(d.columns[0]),
                           "caption": f"Table {nxt()}. DEA efficiency scores by DMU",
                           "note": "", "title": f"Appendix {next(letters)}. DEA results"})
        if self.chk_apx["boot"].isChecked() and "boot" in self.results:
            d = self.results["boot"]["df"].copy()
            for c in d.columns:
                if pd.api.types.is_float_dtype(d[c]):
                    d[c] = d[c].round(6)
            blocks.append({"kind": "table", "df": d.set_index("NO"),
                           "caption": f"Table {nxt()}. Bootstrap bias-corrected "
                                      "efficiency and confidence intervals",
                           "note": "Note: Mean is the bias-corrected efficiency score.",
                           "title": f"Appendix {next(letters)}. Bootstrap results"})
        if self.chk_apx["malms"].isChecked() and "malm" in self.results:
            df, cap, note = VT.malmquist_summary_table(self.results["malm"]["df"],
                                                       table_no=nxt())
            blocks.append({"kind": "table", "df": df, "caption": cap, "note": note,
                           "title": f"Appendix {next(letters)}. Malmquist summary"})
        if self.chk_apx["malm"].isChecked() and "malm" in self.results:
            d = self.results["malm"]["df"].copy()
            for c in d.columns:
                if pd.api.types.is_float_dtype(d[c]):
                    d[c] = d[c].round(6)
            blocks.append({"kind": "table", "df": d.set_index(d.columns[0]),
                           "caption": f"Table {nxt()}. Malmquist index by DMU and period",
                           "note": "", "title": f"Appendix {next(letters)}. Malmquist results"})
        if self.chk_apx["reg"].isChecked() and self.results.get("reg_runs"):
            df, cap, note = VT.regression_table(
                self.results["reg_runs"],
                se_below=(self.cmb_se.currentIndex() == 1), table_no=nxt())
            blocks.append({"kind": "table", "df": df, "caption": cap, "note": note,
                           "title": f"Appendix {next(letters)}. Second-stage regression"})
        if self.chk_apx["group"].isChecked() and self.results.get("group_runs"):
            df, cap, note = VT.group_test_table(self.results["group_runs"],
                                                table_no=nxt())
            blocks.append({"kind": "table", "df": df, "caption": cap, "note": note,
                           "title": f"Appendix {next(letters)}. Group comparison tests"})
        if self.chk_apx["figs"].isChecked() and self._appendix_figs:
            head = f"Appendix {next(letters)}. Figures"
            for i, f in enumerate(self._appendix_figs, 1):
                blocks.append({"kind": "figure", "png": f["png"],
                               "caption": f"Figure {i}. {f['title']}",
                               "title": head if i == 1 else None})
        if self.chk_apx["log"].isChecked():
            blocks.append({"kind": "text", "text": self._settings_log(),
                           "title": f"Appendix {next(letters)}. Analysis settings"})
        return blocks

    def _export_appendix(self, kind):
        try:
            blocks = self._appendix_blocks()
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Error", traceback.format_exc()); return
        if not blocks:
            QtWidgets.QMessageBox.information(
                self, "Appendix",
                "Nothing to export: run an analysis and tick at least one section.")
            return
        if kind == "html":
            fn, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export appendix", "PLEAP_Appendix.html",
                "HTML for Word (*.html)")
        else:
            fn, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export appendix", "PLEAP_Appendix.xlsx", "Excel (*.xlsx)")
        if not fn:
            return
        try:
            if kind == "html":
                VT.export_appendix_html(fn, blocks,
                                        title=self.ed_apx_title.text() or "Appendix",
                                        version=__version__,
                                        subtitle=self.ed_apx_sub.text())
            else:
                VT.export_appendix_excel(fn, blocks, version=__version__)
            n_t = sum(1 for b in blocks if b.get("kind", "table") == "table")
            n_f = sum(1 for b in blocks if b.get("kind") == "figure")
            QtWidgets.QMessageBox.information(
                self, "Appendix",
                f"Exported {n_t} table(s) and {n_f} figure(s) to:\n{fn}")
            self.lbl_status.setText(f"Appendix exported: {fn}")
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Error", traceback.format_exc())

    def viz_export_html(self):
        self._export_appendix("html")

    def viz_export_xlsx(self):
        self._export_appendix("xlsx")
