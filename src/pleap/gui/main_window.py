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

import os, sys, traceback
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

from pleap import APP_NAME, APP_FULL, __version__, __author__, __email__
from pleap.core.dea import dea, rts_classify
from pleap.core.bootstrap import bootstrap_dea
from pleap.core.malmquist import malmquist
from pleap.core import secondstage as ss
from pleap.gui.viz_tab import VizTabMixin


def resource_path(rel):
    """Locate a bundled resource, e.g. ``resource_path("assets/lab_logo.png")``.

    ``rel`` is relative to the ``pleap`` package, so the assets travel with the
    package and are found whether PLEAP is run from a checkout, from a wheel
    installed into site-packages, or from a PyInstaller bundle.  Before v1.12
    the base was computed two directories up from this file, which pointed
    outside the package once the source moved under ``src/`` - the window then
    fell back to showing the text "[LOGO]", and an installed wheel had no
    assets at all.
    """
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        for candidate in (os.path.join(frozen, "pleap", rel),
                          os.path.join(frozen, rel)):   # pre-1.12 bundles
            if os.path.exists(candidate):
                return candidate
        return os.path.join(frozen, "pleap", rel)
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(pkg, rel)


def _set_app_user_model_id():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "INU.ProductivityLab.PLEAP.1")
        except Exception:
            pass


class WorkerSignals(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(object)
    error    = QtCore.pyqtSignal(str)


class FnRunner(QtCore.QThread):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn; self.signals = WorkerSignals()
    def run(self):
        try:
            res = self.fn(self.signals.progress.emit)
        except Exception:
            self.signals.error.emit(traceback.format_exc()); return
        self.signals.finished.emit(res)


REPO_URL = "https://github.com/ProductivityLab/pleap"

LICENSE_KO = (
    "PLEAP은 자유 소프트웨어입니다.\n\n"
    "Copyright (C) 2025-2026 Changhee Kim,\n"
    "인천대학교 생산성연구실(Incheon National University, Productivity Lab)\n\n"
    "이 프로그램은 자유 소프트웨어입니다. 자유 소프트웨어 재단이 발표한\n"
    "GNU 일반 공중 사용 허가서(GNU General Public License) 제3판 또는\n"
    "그 이후 판의 조건에 따라 누구나 자유롭게 사용, 복제, 배포, 수정할 수\n"
    "있습니다.\n\n"
    "이 프로그램은 유용하게 쓰이기를 바라는 마음으로 배포되지만,\n"
    "어떠한 보증도 제공되지 않습니다. 상품성이나 특정 목적에 대한\n"
    "적합성에 대한 묵시적 보증도 포함되지 않습니다.\n"
    "자세한 내용은 GNU 일반 공중 사용 허가서를 참조하십시오.\n\n"
    "허가서 전문: https://www.gnu.org/licenses/gpl-3.0.html\n"
    "소스 코드: " + REPO_URL + "\n\n"
    "연구에 사용하신 경우 인용해 주시면 감사하겠습니다.\n"
    "인용 정보는 저장소의 CITATION.cff 파일에 있습니다.\n\n"
    "문의: Changhee Kim (ckim@inu.ac.kr)"
)
LICENSE_EN = (
    "PLEAP is free software.\n\n"
    "Copyright (C) 2025-2026 Changhee Kim,\n"
    "Productivity Lab, Incheon National University\n\n"
    "This program is free software: you can redistribute it and/or modify\n"
    "it under the terms of the GNU General Public License as published by\n"
    "the Free Software Foundation, either version 3 of the License, or\n"
    "(at your option) any later version.\n\n"
    "This program is distributed in the hope that it will be useful,\n"
    "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
    "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
    "GNU General Public License for more details.\n\n"
    "You should have received a copy of the GNU General Public License\n"
    "along with this program.  If not, see https://www.gnu.org/licenses/.\n\n"
    "Source code: " + REPO_URL + "\n\n"
    "If PLEAP contributed to published work, a citation is appreciated;\n"
    "see CITATION.cff in the repository.\n\n"
    "Contact: Changhee Kim (ckim@inu.ac.kr)"
)


class MainWindow(VizTabMixin, QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} - {APP_FULL}  v{__version__}")
        self.resize(1240, 820)
        ico = resource_path("assets/lab_logo.ico")
        if os.path.exists(ico):
            self.setWindowIcon(QtGui.QIcon(ico))
        self.df = None
        self.results = {}
        self.thread = None
        self._build_ui()

    def _build_ui(self):
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        v = QtWidgets.QVBoxLayout(cw)
        v.addLayout(self._header())
        self.tabs = QtWidgets.QTabWidget()
        v.addWidget(self.tabs, 1)
        self.tabs.addTab(self._tab_data(),    "1. Data")
        self.tabs.addTab(self._tab_dea(),     "2. DEA (CCR/BCC)")
        self.tabs.addTab(self._tab_boot(),    "3. Bootstrap DEA")
        self.tabs.addTab(self._tab_malm(),    "4. Malmquist")
        self.tabs.addTab(self._tab_reg(),     "5. 2nd-stage Regression")
        self.tabs.addTab(self._tab_group(),   "6. Group Tests")
        self.tabs.addTab(self._tab_viz(),     "7. Visualization")
        self.tabs.addTab(self._tab_license(), "8. License")
        self.pb = QtWidgets.QProgressBar(); self.pb.setRange(0,1); self.pb.setValue(0)
        self.lbl_status = QtWidgets.QLabel(f"Ready  -  {APP_NAME} by {__author__} <{__email__}>")
        bar = QtWidgets.QHBoxLayout(); bar.addWidget(self.lbl_status, 1); bar.addWidget(self.pb, 1)
        v.addLayout(bar)

    def _header(self):
        h = QtWidgets.QHBoxLayout()
        logo = QtWidgets.QLabel()
        png = resource_path("assets/lab_logo.png")
        if os.path.exists(png):
            pm = QtGui.QPixmap(png).scaledToHeight(56, QtCore.Qt.SmoothTransformation)
            logo.setPixmap(pm)
        else:
            logo.setText("[LOGO]"); logo.setFixedSize(56, 56)
        h.addWidget(logo)
        title = QtWidgets.QLabel(
            f"<b style='font-size:18pt'>{APP_NAME}</b> "
            f"<span style='font-size:10pt;color:#555'>- {APP_FULL}  v{__version__}</span><br>"
            f"<span style='color:#777'>by {__author__} &lt;{__email__}&gt; &nbsp;|&nbsp; "
            f"(c) Incheon National University, Productivity Lab</span>")
        title.setTextFormat(QtCore.Qt.RichText)
        h.addWidget(title); h.addStretch(1)
        return h

    def _tab_data(self):
        w = QtWidgets.QWidget(); g = QtWidgets.QGridLayout(w)
        btn = QtWidgets.QPushButton("Load Excel..."); btn.clicked.connect(self.load_excel)
        g.addWidget(btn, 0, 0)
        self.lbl_file = QtWidgets.QLabel("(no file)"); g.addWidget(self.lbl_file, 0, 1, 1, 5)
        self.tbl_preview = QtWidgets.QTableWidget(); g.addWidget(self.tbl_preview, 1, 0, 1, 6)
        box = QtWidgets.QGroupBox("Column mapping")
        f = QtWidgets.QFormLayout(box)
        self.cmb_dmu_name = QtWidgets.QComboBox()
        self.cmb_period   = QtWidgets.QComboBox()
        self.cmb_group    = QtWidgets.QComboBox()
        f.addRow("DMU name:", self.cmb_dmu_name)
        f.addRow("Period variable:", self.cmb_period)
        f.addRow("Group variable:", self.cmb_group)
        self.lst_inputs  = QtWidgets.QListWidget(); self.lst_inputs.setSelectionMode(QtWidgets.QListWidget.MultiSelection)
        self.lst_outputs = QtWidgets.QListWidget(); self.lst_outputs.setSelectionMode(QtWidgets.QListWidget.MultiSelection)
        self.lst_envs    = QtWidgets.QListWidget(); self.lst_envs.setSelectionMode(QtWidgets.QListWidget.MultiSelection)
        f.addRow("Inputs (<=5):",  self.lst_inputs)
        f.addRow("Outputs (<=5):", self.lst_outputs)
        f.addRow("Environmental vars:", self.lst_envs)
        g.addWidget(box, 2, 0, 1, 6)
        return w

    def load_excel(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Excel", "", "Excel (*.xlsx *.xls)")
        if not fn: return
        try: self.df = pd.read_excel(fn)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e)); return
        self.lbl_file.setText(fn)
        cols = list(self.df.columns)
        self.tbl_preview.setColumnCount(len(cols))
        self.tbl_preview.setHorizontalHeaderLabels(cols)
        self.tbl_preview.setRowCount(min(20, len(self.df)))
        for i in range(min(20, len(self.df))):
            for j, c in enumerate(cols):
                self.tbl_preview.setItem(i, j, QtWidgets.QTableWidgetItem(str(self.df.iloc[i, j])))
        for cb in (self.cmb_dmu_name, self.cmb_period, self.cmb_group):
            cb.clear(); cb.addItem(""); cb.addItems(cols)
        for lst in (self.lst_inputs, self.lst_outputs, self.lst_envs):
            lst.clear()
            for c in cols: lst.addItem(c)

    def _selected(self, lst, cap=None):
        items = [x.text() for x in lst.selectedItems()]
        if cap and len(items) > cap:
            raise ValueError(f"Select at most {cap} columns.")
        return items

    def _XY(self):
        if self.df is None: raise ValueError("Load data first.")
        ins  = self._selected(self.lst_inputs, 5)
        outs = self._selected(self.lst_outputs, 5)
        if not ins or not outs: raise ValueError("Select inputs and outputs.")
        X = self.df[ins].astype(float).values
        Y = self.df[outs].astype(float).values
        return X, Y, ins, outs

    def _dmu_labels(self):
        col = self.cmb_dmu_name.currentText()
        if col: return self.df[col].astype(str).tolist()
        return [f"DMU{i+1}" for i in range(len(self.df))]

    def _tab_dea(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        h = QtWidgets.QHBoxLayout()
        self.cmb_model  = QtWidgets.QComboBox(); self.cmb_model.addItems(["CCR","BCC"])
        self.cmb_orient = QtWidgets.QComboBox(); self.cmb_orient.addItems(["input","output"])
        h.addWidget(QtWidgets.QLabel("Model:")); h.addWidget(self.cmb_model)
        h.addWidget(QtWidgets.QLabel("Orientation:")); h.addWidget(self.cmb_orient)
        run = QtWidgets.QPushButton("Run DEA"); run.clicked.connect(self.run_dea)
        save = QtWidgets.QPushButton("Save results..."); save.clicked.connect(lambda: self._save_result("dea","DEA"))
        h.addWidget(run); h.addWidget(save); h.addStretch(1)
        v.addLayout(h)
        self.tbl_dea = QtWidgets.QTableWidget(); v.addWidget(self.tbl_dea, 1)
        return w

    def _start_job(self, fn, on_done, label="Running..."):
        if self.thread is not None and self.thread.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy", "Another job is running."); return
        self.pb.setRange(0, 0); self.lbl_status.setText(label)
        self.thread = FnRunner(fn)
        self.thread.signals.progress.connect(self._on_progress)
        self.thread.signals.finished.connect(lambda r: self._on_done(r, on_done))
        self.thread.signals.error.connect(self._on_error)
        self.thread.start()

    def _on_progress(self, cur, total, msg):
        if total > 0:
            self.pb.setRange(0, total); self.pb.setValue(cur)
        if msg: self.lbl_status.setText(msg)

    def _on_done(self, res, cb):
        self.pb.setRange(0, 1); self.pb.setValue(1)
        try: cb(res)
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Error", traceback.format_exc())
        self.lbl_status.setText("Done.")

    def _on_error(self, tb):
        self.pb.setRange(0, 1); self.pb.setValue(0)
        self.lbl_status.setText("Error.")
        QtWidgets.QMessageBox.critical(self, "Error", tb)

    def run_dea(self):
        try: X, Y, ins, outs = self._XY()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Input", str(e)); return
        model = self.cmb_model.currentText(); orient = self.cmb_orient.currentText()
        labels = self._dmu_labels()
        def job(prog):
            def p1(c, t): prog(c, t*2, f"DEA scores {c}/{t}")
            def p2(c, t): prog(t + c, t*2, f"RTS {c}/{t}")
            res = dea(X, Y, model=model, orient=orient, progress=p1)
            rts, ccr, bcc = rts_classify(X, Y, orient=orient, progress=p2)
            return res, rts, ccr, bcc
        def done(payload):
            res, rts, ccr, bcc = payload
            eff = res["efficiency"]; lam = res["lambdas"]
            sm = res["slack_input"]; sp = res["slack_output"]
            peers = []
            for k in range(len(eff)):
                ps = [(labels[j], lam[k, j]) for j in range(len(eff)) if lam[k, j] > 1e-6]
                ps.sort(key=lambda x: -x[1])
                peers.append("; ".join(f"{n}({w:.3f})" for n,w in ps[:8]))
            df = pd.DataFrame({"DMU": labels, "Efficiency": eff, "RTS": rts, "Peers": peers})
            for i,c in enumerate(ins):  df[f"slack_{c}"] = sm[:, i]
            for i,c in enumerate(outs): df[f"slack_{c}"] = sp[:, i]
            self.results["dea"] = {"df": df, "model": model, "orient": orient,
                                   "raw": res, "X": X, "Y": Y,
                                   "ins": ins, "outs": outs}
            self._fill_table(self.tbl_dea, df)
        self._start_job(job, done, label=f"Running DEA ({model}/{orient})...")

    def _save_result(self, key, label):
        if key not in self.results:
            QtWidgets.QMessageBox.information(self, label, f"No {label} results yet."); return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, f"Save {label}", f"{label}.xlsx", "Excel (*.xlsx)")
        if not fn: return
        with pd.ExcelWriter(fn, engine="openpyxl") as wr:
            for k, v in self.results.items():
                if isinstance(v, dict) and "df" in v: v["df"].to_excel(wr, sheet_name=k[:31], index=False)
        QtWidgets.QMessageBox.information(self, label, f"Saved to {fn}")

    def _tab_boot(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        h = QtWidgets.QHBoxLayout()
        self.cmb_bmodel = QtWidgets.QComboBox(); self.cmb_bmodel.addItems(["BCC","CCR"])
        self.cmb_borient = QtWidgets.QComboBox(); self.cmb_borient.addItems(["input","output"])
        self.spn_B = QtWidgets.QSpinBox(); self.spn_B.setRange(50, 5000); self.spn_B.setValue(2000)
        self.spn_alpha = QtWidgets.QDoubleSpinBox(); self.spn_alpha.setRange(0.001, 0.5)
        self.spn_alpha.setSingleStep(0.01); self.spn_alpha.setValue(0.05)
        import os as _os
        self.spn_jobs = QtWidgets.QSpinBox(); self.spn_jobs.setRange(1, max(1, _os.cpu_count() or 1))
        self.spn_jobs.setValue(max(1, _os.cpu_count() or 1))
        self.spn_jobs.setToolTip("CPU cores used for the bootstrap (more = faster).")
        for lab, wd in [("Model:",self.cmb_bmodel),("Orient:",self.cmb_borient),
                        ("B:",self.spn_B),("alpha:",self.spn_alpha),("Cores:",self.spn_jobs)]:
            h.addWidget(QtWidgets.QLabel(lab)); h.addWidget(wd)
        run = QtWidgets.QPushButton("Run Bootstrap"); run.clicked.connect(self.run_boot)
        save = QtWidgets.QPushButton("Save..."); save.clicked.connect(lambda: self._save_result("boot","Bootstrap"))
        h.addWidget(run); h.addWidget(save); h.addStretch(1)
        v.addLayout(h)
        self.tbl_boot = QtWidgets.QTableWidget(); v.addWidget(self.tbl_boot, 1)
        return w

    def run_boot(self):
        try: X, Y, ins, outs = self._XY()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Input", str(e)); return
        model = self.cmb_bmodel.currentText(); orient = self.cmb_borient.currentText()
        B = self.spn_B.value(); alpha = self.spn_alpha.value(); n_jobs = self.spn_jobs.value()
        labels = self._dmu_labels()
        def job(prog):
            def p(c, t): prog(c, t, f"Bootstrap {c}/{t}")
            return bootstrap_dea(X, Y, model=model, orient=orient, B=B,
                                 alpha=alpha, n_jobs=n_jobs, progress=p)
        def done(res):
            nB = len(labels)
            df = pd.DataFrame({"NO": np.arange(1, nB + 1),
                               "DMU": labels,
                               "DMU Order in Data": np.arange(1, nB + 1),
                               "Score(Original)": res["theta_hat"],
                               "Bias": res["bias"],
                               "Mean": res["boot_mean"],
                               "Median": res["boot_median"],
                               "SD": res["boot_sd"],
                               "CI_LowerBound": res["ci_low"],
                               "CI_UpperBound": res["ci_high"]})
            self.results["boot"] = {"df": df, "raw": res, "model": model,
                                    "orient": orient, "X": X, "Y": Y,
                                    "ins": ins, "outs": outs}
            self._fill_table(self.tbl_boot, df)
        self._start_job(job, done, label=f"Bootstrap (KSW2008, B={B}, {n_jobs} core)...")

    def _tab_malm(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        h = QtWidgets.QHBoxLayout()
        self.cmb_mmodel = QtWidgets.QComboBox(); self.cmb_mmodel.addItems(["BCC","CCR"])
        self.cmb_morient = QtWidgets.QComboBox(); self.cmb_morient.addItems(["input","output"])
        h.addWidget(QtWidgets.QLabel("Model:")); h.addWidget(self.cmb_mmodel)
        h.addWidget(QtWidgets.QLabel("Orient:")); h.addWidget(self.cmb_morient)
        # Under VRS a cross-period program can be infeasible.  Which convention
        # is applied can change the direction of a reported index, so it is a
        # visible control and not a hidden default.
        self.cmb_minfeas = QtWidgets.QComboBox()
        self.cmb_minfeas.addItems(["substitute 1 (MaxDEA)", "omit (report NaN)"])
        self.cmb_minfeas.setToolTip(
            "What to do when a cross-period distance is infeasible under VRS.\n"
            "Substituting 1 keeps an index defined and matches MaxDEA, but it\n"
            "propagates into MI and TC. Omitting reports NaN instead.")
        h.addWidget(QtWidgets.QLabel("If infeasible:")); h.addWidget(self.cmb_minfeas)
        run = QtWidgets.QPushButton("Run Malmquist"); run.clicked.connect(self.run_malm)
        save = QtWidgets.QPushButton("Save..."); save.clicked.connect(lambda: self._save_result("malm","Malmquist"))
        h.addWidget(run); h.addWidget(save); h.addStretch(1)
        v.addLayout(h)
        self.lbl_malm_note = QtWidgets.QLabel("")
        self.lbl_malm_note.setWordWrap(True)
        self.lbl_malm_note.setStyleSheet("color: #8a5000;")
        self.lbl_malm_note.hide()
        v.addWidget(self.lbl_malm_note)
        self.tbl_malm = QtWidgets.QTableWidget(); v.addWidget(self.tbl_malm, 1)
        return w

    def run_malm(self):
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "Data","Load data first."); return
        per = self.cmb_period.currentText()
        if not per:
            QtWidgets.QMessageBox.warning(self, "Period","Set period variable in Data tab."); return
        try:
            ins  = self._selected(self.lst_inputs, 5)
            outs = self._selected(self.lst_outputs, 5)
            dmu_col = self.cmb_dmu_name.currentText()
            if not dmu_col: raise ValueError("Set DMU name.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Input", str(e)); return
        df_full = self.df.copy()
        mmodel = self.cmb_mmodel.currentText(); morient = self.cmb_morient.currentText()
        minfeas = ("omit" if self.cmb_minfeas.currentIndex() == 1
                   else "substitute")
        def job(prog):
            def p(c, t): prog(c, t, f"Malmquist {c}/{t}")
            return malmquist(df_full, dmu_col, per, ins, outs,
                             model=mmodel, orient=morient, progress=p,
                             on_infeasible=minfeas)
        def done(df):
            self.results["malm"] = {"df": df}
            self._fill_table(self.tbl_malm, df)
            # Say so on screen.  A reader cannot tell from an index that a
            # distance was substituted, and the substitution can move the
            # reported direction of productivity change.
            note = df.attrs.get("substitution_note", "")
            if note:
                self.lbl_malm_note.setText("⚠  " + note)
                self.lbl_malm_note.show()
            else:
                self.lbl_malm_note.clear()
                self.lbl_malm_note.hide()
        self._start_job(job, done, label="Running Malmquist...")

    def _tab_reg(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        h = QtWidgets.QHBoxLayout()
        self.cmb_dep = QtWidgets.QComboBox()
        self.cmb_dep.addItems(["Raw efficiency (DEA)",
                               "Bias-corrected efficiency (Bootstrap)",
                               "Malmquist EC","Malmquist TC","Malmquist MI"])
        self.cmb_regtype = QtWidgets.QComboBox()
        self.cmb_regtype.addItems(["Tobit", "Truncated", "OLS",
                                   "Simar-Wilson Alg.1"])
        self.cmb_regtype.setToolTip(
            "Tobit, Truncated and OLS report conventional standard errors,\n"
            "which are not valid for DEA scores because every score is\n"
            "measured against a frontier estimated from the same sample.\n"
            "Simar-Wilson Algorithm 1 replaces them with bootstrap standard\n"
            "errors that account for that dependence.")
        h.addWidget(QtWidgets.QLabel("Dependent:")); h.addWidget(self.cmb_dep)
        h.addWidget(QtWidgets.QLabel("Method:")); h.addWidget(self.cmb_regtype)
        run = QtWidgets.QPushButton("Run regression"); run.clicked.connect(self.run_reg)
        h.addWidget(run); h.addStretch(1)
        v.addLayout(h)
        self.txt_reg = QtWidgets.QPlainTextEdit(); self.txt_reg.setReadOnly(True)
        self.txt_reg.setFont(QtGui.QFont("Consolas", 10))
        v.addWidget(self.txt_reg, 1)
        return w

    def _get_dep(self, kind):
        if kind == "Raw efficiency (DEA)":
            if "dea" not in self.results: raise ValueError("Run DEA first.")
            return self.results["dea"]["df"]["Efficiency"].values, np.arange(len(self.df))
        if kind == "Bias-corrected efficiency (Bootstrap)":
            if "boot" not in self.results: raise ValueError("Run bootstrap first.")
            return self.results["boot"]["df"]["Mean"].values, np.arange(len(self.df))
        if "malm" not in self.results: raise ValueError("Run Malmquist first.")
        mdf = self.results["malm"]["df"]
        col = {"Malmquist EC":"EC(t-1, t)","Malmquist TC":"TC(t-1, t)","Malmquist MI":"MI(t-1, t)"}[kind]
        dmu_col = self.cmb_dmu_name.currentText()
        per = self.cmb_period.currentText()
        if not dmu_col or not per: raise ValueError("Set DMU name and period in Data tab.")
        last_period = sorted(self.df[per].unique())[-1]
        last_pair_to = sorted(mdf["t(Period)"].unique())[-1]
        m_last = mdf[mdf["t(Period)"] == last_pair_to].set_index("DMU")
        sub = self.df[self.df[per]==last_period].copy().set_index(dmu_col)
        common = [d for d in sub.index if d in m_last.index]
        m_last = m_last.loc[common]
        idx = self.df.index[(self.df[per]==last_period) & (self.df[dmu_col].isin(common))].to_numpy()
        order = {d:i for i,d in enumerate(common)}
        idx = np.array(sorted(idx, key=lambda i: order[self.df.loc[i, dmu_col]]))
        return m_last[col].values, idx

    def run_reg(self):
        envs = self._selected(self.lst_envs)
        if not envs:
            QtWidgets.QMessageBox.warning(self,"Env","Select environmental variables in Data tab."); return
        method = self.cmb_regtype.currentText(); kind = self.cmb_dep.currentText()
        try:
            y, idx = self._get_dep(kind)
            Z = self.df.loc[idx, envs].astype(float).values
            is_eff = "efficiency" in kind.lower()
            if method == "Simar-Wilson Alg.1":
                if not is_eff:
                    raise ValueError(
                        "Algorithm 1 is for efficiency scores: the model is "
                        "truncated at the frontier, so the dependent variable "
                        "has to be an efficiency score.")
                r = ss.simarwilson_algorithm1(y, Z, L=2000, seed=12345,
                                              names=envs)
                hdr = (f"Simar-Wilson (2007) Algorithm 1  ({kind})\n"
                       f" n={r['n']} inefficient units  L={r['L']} bootstrap "
                       f"replications  sigma={r['sigma']:.5f}\n"
                       " Standard errors and intervals are bootstrap; the "
                       "coefficients are the truncated-regression estimates.\n"
                       " Algorithm 1 uses the scores as estimated. It does not "
                       "bias-correct them; that is Algorithm 2.\n\n")
                self.txt_reg.setPlainText(hdr + r["summary"].to_string())
            elif method == "OLS":
                r = ss.ols_regression(y, Z, names=envs)
                hdr = (f"OLS  ({kind})\n n={r['n']}  R^2={r['r2']:.4f}  adjR^2={r['adj_r2']:.4f}  "
                       f"F={r['fvalue']:.3f} (p={r['f_pvalue']:.4g})\n"
                       "Significance: *** p<0.01, ** p<0.05, * p<0.10\n\n")
                self.txt_reg.setPlainText(hdr + r["summary"].to_string())
            else:
                upper = 1.0 if is_eff else None
                if method == "Tobit":
                    lower = 0.0 if is_eff else None
                    r = ss.tobit_regression(y, Z, lower=lower, upper=upper, names=envs)
                else:
                    # Simar-Wilson truncated regression truncates only at the
                    # efficient frontier (upper); a lower truncation at 0 makes
                    # the doubly-truncated likelihood ill-posed.
                    r = ss.truncated_regression(y, Z, lower=None, upper=upper, names=envs)
                hdr = (f"{method}  ({kind})\n n={r['n']}  sigma={r['sigma']:.5f}  "
                       f"loglik={r['loglik']:.4f}  converged={r['converged']}\n")
                if r.get("wald_df"):
                    hdr += (f" Wald statistic={r['wald']:.4f}  d.f.={r['wald_df']}  "
                            f"(p={r['wald_p']:.4g})\n")
                hdr += "Significance: *** p<0.01, ** p<0.05, * p<0.10\n\n"
                self.txt_reg.setPlainText(hdr + r["summary"].to_string())
            self._record_run("reg_runs", {"method": method, "kind": kind,
                                          "envs": list(envs), "result": r})
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Error", traceback.format_exc())

    def _record_run(self, key, entry):
        """Keep analysis runs so the Visualization tab can tabulate them.

        Purely a record of results already computed and displayed; it does not
        re-run or alter any analysis.
        """
        runs = self.results.setdefault(key, [])
        same = [i for i, r in enumerate(runs)
                if r.get("kind") == entry.get("kind")
                and r.get("method") == entry.get("method")]
        if same:
            runs[same[0]] = entry          # replace an identical earlier run
        else:
            runs.append(entry)
        if len(runs) > 8:
            del runs[0]

    def _tab_group(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        h = QtWidgets.QHBoxLayout()
        self.cmb_gdep = QtWidgets.QComboBox()
        self.cmb_gdep.addItems([
            "Raw efficiency (DEA)  -> Mann-Whitney / Kruskal-Wallis (Dunn)",
            "Bias-corrected efficiency (Bootstrap)  -> t-test / ANOVA (Levene+Tukey/GH)",
            "Malmquist EC  -> Mann-Whitney / Kruskal-Wallis (Dunn)",
            "Malmquist TC  -> Mann-Whitney / Kruskal-Wallis (Dunn)",
            "Malmquist MI  -> Mann-Whitney / Kruskal-Wallis (Dunn)",
        ])
        h.addWidget(QtWidgets.QLabel("Test set:")); h.addWidget(self.cmb_gdep)
        run = QtWidgets.QPushButton("Run"); run.clicked.connect(self.run_group)
        h.addWidget(run); h.addStretch(1)
        v.addLayout(h)
        self.txt_grp = QtWidgets.QPlainTextEdit(); self.txt_grp.setReadOnly(True)
        self.txt_grp.setFont(QtGui.QFont("Consolas", 10))
        v.addWidget(self.txt_grp, 1)
        return w

    def run_group(self):
        gcol = self.cmb_group.currentText()
        if not gcol:
            QtWidgets.QMessageBox.warning(self,"Group","Set group variable in Data tab."); return
        kind = self.cmb_gdep.currentText()
        try:
            if kind.startswith("Raw"):
                y, idx = self._get_dep("Raw efficiency (DEA)")
                r = ss.group_tests_nonparam(y, self.df.loc[idx, gcol].values)
            elif kind.startswith("Bias"):
                y, idx = self._get_dep("Bias-corrected efficiency (Bootstrap)")
                r = ss.group_tests_param(y, self.df.loc[idx, gcol].values)
            else:
                key = kind.split("  ")[0]
                y, idx = self._get_dep(key)
                r = ss.group_tests_nonparam(y, self.df.loc[idx, gcol].values)
            stat_label = "Mean" if "mean" in r else "Median"
            vals = r.get("mean", r.get("median", []))
            lines = [f"Test: {r['test']}",
                     f"Overall stat: {r['stat']:.4f}    p-value: {r['p']:.6g}  {r['sig']}"]
            if "levene" in r:
                lev = r["levene"]
                lines.append(f"Levene test:  F={lev['stat']:.4f}  p={lev['p']:.6g}")
            lines += ["Significance: *** p<0.01, ** p<0.05, * p<0.10","",
                      "Group summary:"]
            for lv, n, mv in zip(r["groups"], r["n"], vals):
                lines.append(f"  {lv}:  n={n}  {stat_label}={mv:.6f}")
            lines += ["", "Pairwise comparisons (direction shown when p<0.10):"]
            if isinstance(r.get("pairwise"), pd.DataFrame) and not r["pairwise"].empty:
                lines.append(r["pairwise"].round(6).to_string(index=False))
            self.txt_grp.setPlainText("\n".join(lines))
            self._record_run("group_runs",
                             {"kind": kind.split("  ->")[0].strip(),
                              "method": r.get("test", ""), "group": gcol,
                              "result": r})
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Error", traceback.format_exc())

    def _tab_license(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        head = QtWidgets.QLabel(
            f"<b>{APP_NAME}</b> v{__version__} &nbsp;&mdash;&nbsp; "
            f"free software under the <b>GNU GPL v3 or later</b><br>"
            f"Copyright &copy; 2025&ndash;2026 Changhee Kim, "
            f"Productivity Lab, Incheon National University<br>"
            f"Source code: <a href='{REPO_URL}'>{REPO_URL}</a>")
        head.setTextFormat(QtCore.Qt.RichText)
        head.setOpenExternalLinks(True)
        v.addWidget(head)
        txt = QtWidgets.QPlainTextEdit(); txt.setReadOnly(True)
        txt.setFont(QtGui.QFont("Malgun Gothic", 11))
        body = ("[한국어]\n" + LICENSE_KO + "\n\n" +
                "============================================================\n\n" +
                "[English]\n" + LICENSE_EN)
        txt.setPlainText(body)
        v.addWidget(txt, 1)
        foot = QtWidgets.QLabel(f"Author: {__author__}  &lt;{__email__}&gt;")
        foot.setTextFormat(QtCore.Qt.RichText)
        v.addWidget(foot)
        return w

    def _fill_table(self, tbl, df):
        tbl.clear()
        tbl.setColumnCount(len(df.columns))
        tbl.setHorizontalHeaderLabels(list(df.columns))
        tbl.setRowCount(len(df))
        for i in range(len(df)):
            for j, c in enumerate(df.columns):
                v = df.iloc[i, j]
                s = f"{v:.6f}" if isinstance(v, float) else str(v)
                tbl.setItem(i, j, QtWidgets.QTableWidgetItem(s))
        tbl.resizeColumnsToContents()


def main():
    _set_app_user_model_id()
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    ico = resource_path("assets/lab_logo.ico")
    if os.path.exists(ico):
        app.setWindowIcon(QtGui.QIcon(ico))
    splash_pix = QtGui.QPixmap(560, 280); splash_pix.fill(QtGui.QColor("#082878"))
    sp = QtWidgets.QSplashScreen(splash_pix)
    sp.showMessage(f"{APP_NAME}\n{APP_FULL}\n\nby {__author__}\n<{__email__}>\n\n"
                   f"(c) Incheon National University, Productivity Lab\nv{__version__}",
                   QtCore.Qt.AlignCenter, QtGui.QColor("white"))
    sp.show(); app.processEvents()
    QtCore.QTimer.singleShot(1200, sp.close)
    win = MainWindow(); win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
