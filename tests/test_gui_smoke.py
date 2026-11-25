"""The interface builds and wires up.

These tests construct the real main window off-screen.  They do not drive an
analysis - that is what the rest of the suite is for - but they do catch the
failures that are invisible until someone launches the program: a missing tab,
a renamed signal, a module that the package restructuring left unreachable.

Skipped when PyQt5 is not installed, so ``pip install -e .`` without the
``gui`` extra still gives a green suite.
"""

import os

import pytest

pytest.importorskip("PyQt5", reason="GUI extra not installed")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")

from PyQt5 import QtWidgets        # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    a.setStyle("Windows")
    yield a


@pytest.fixture(scope="module")
def win(app):
    from pleap.gui.main_window import MainWindow
    w = MainWindow()
    yield w
    w.close()


def test_entry_point_is_importable():
    """``pleap`` on the command line and ``python -m pleap`` resolve here."""
    from pleap.__main__ import main
    assert callable(main)


def test_window_is_titled_with_the_version(win):
    import pleap
    assert pleap.__version__ in win.windowTitle()
    assert "PLEAP" in win.windowTitle()


def test_every_analysis_tab_is_present(win):
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    joined = " ".join(titles)
    for expected in ("Data", "DEA", "Bootstrap", "Malmquist",
                     "Visualization", "License"):
        assert expected in joined, (expected, titles)


def test_visualization_comes_before_the_license_tab(win):
    """The tab order is the order an analysis is carried out."""
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    viz = next(i for i, t in enumerate(titles) if "Visualization" in t)
    lic = next(i for i, t in enumerate(titles) if "License" in t)
    assert viz == lic - 1


def test_appearance_controls_are_wired(win):
    """Palette, style and font scale must offer exactly what viz supports."""
    from pleap.viz import figures as F
    assert set(win.cmb_palette.itemText(i)
               for i in range(win.cmb_palette.count())) == set(F.PALETTES)
    assert [win.cmb_style.itemText(i)
            for i in range(win.cmb_style.count())] == list(F.STYLES)
    assert win.spn_fscale.minimum() > 0


def test_malmquist_model_and_orientation_are_selectable(win):
    """Specification drift is what these controls exist to prevent."""
    models = [win.cmb_mmodel.itemText(i)
              for i in range(win.cmb_mmodel.count())]
    orients = [win.cmb_morient.itemText(i)
               for i in range(win.cmb_morient.count())]
    assert set(models) == {"BCC", "CCR"}
    assert len(orients) == 2
    assert any("input" in o.lower() for o in orients)
    assert any("output" in o.lower() for o in orients)


def test_results_store_starts_empty(win):
    assert isinstance(win.results, dict)
    assert not win.results.get("reg_runs")
    assert not win.results.get("group_runs")


def test_the_license_tab_states_the_gpl_and_not_a_prohibition(win):
    """The License tab must agree with LICENSE.txt.

    Before v1.12 it forbade copying, redistribution and modification, which
    is the opposite of the licence the software is now released under.  A
    disagreement between the two is a licensing problem, not a typo.
    """
    from pleap.gui.main_window import LICENSE_EN, LICENSE_KO
    both = LICENSE_EN + LICENSE_KO
    assert "GNU General Public License" in LICENSE_EN
    assert "GNU 일반 공중 사용 허가서" in LICENSE_KO
    assert "free software" in LICENSE_EN.lower()
    for forbidden in ("All rights reserved", "strictly prohibited",
                      "Unauthorized copying", "무단 복제", "무단 배포",
                      "일체 금지"):
        assert forbidden not in both, forbidden
    assert "github.com" in both


def test_the_license_tab_renders(win):
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    i = next(i for i, t in enumerate(titles) if "License" in t)
    page = win.tabs.widget(i)
    texts = [c.toPlainText() for c in page.findChildren(QtWidgets.QPlainTextEdit)]
    assert texts, "the licence text must be shown, not just linked"
    assert "GNU" in " ".join(texts)


def test_bundled_assets_resolve_and_load():
    """The logo must be found from a checkout, a wheel and a frozen bundle.

    The src/ move broke this once: resource_path pointed two directories up
    from the module, which landed outside the package, so the window silently
    fell back to the text "[LOGO]" and an installed wheel shipped no assets at
    all.  Resolution is now relative to the package, and the assets live
    inside it.
    """
    import os
    from PyQt5 import QtGui
    from pleap.gui.main_window import resource_path

    for rel in ("assets/lab_logo.png", "assets/lab_logo.ico"):
        path = resource_path(rel)
        assert os.path.exists(path), f"{rel} not found at {path}"
        assert "src" + os.sep + "pleap" in path or "pleap" in path
        pm = QtGui.QPixmap(path)
        assert not pm.isNull(), f"{rel} did not load as an image"
        assert pm.width() >= 16 and pm.height() >= 16


def test_assets_are_inside_the_package_so_a_wheel_carries_them():
    import pathlib
    import pleap
    pkg = pathlib.Path(pleap.__file__).parent
    assert (pkg / "assets" / "lab_logo.png").is_file()
    assert (pkg / "assets" / "lab_logo.ico").is_file()


def test_the_window_shows_the_logo_not_the_fallback_text(win):
    """[LOGO] on screen means the icon failed to load."""
    labels = win.findChildren(QtWidgets.QLabel)
    assert labels
    assert not any(lb.text().strip() == "[LOGO]" for lb in labels), \
        "the header fell back to the [LOGO] placeholder"
    assert any(lb.pixmap() is not None and not lb.pixmap().isNull()
               for lb in labels), "no logo pixmap in the header"


def test_the_window_icon_is_set(win):
    assert not win.windowIcon().isNull()
    assert win.windowIcon().availableSizes(), "icon has no usable sizes"


def test_the_malmquist_tab_exposes_the_infeasibility_convention(win):
    """The convention has to be visible, because it can change the answer.

    Substituting 1 for an infeasible cross-period distance lowers validation
    panel A's first-transition productivity growth from 9.8% to 4.4%, and it
    reversed the direction of that transition on the example used up to
    v1.11.  A choice with that much leverage is a control, not a hidden
    default, and the paper claims the software reports when it was applied.
    """
    assert hasattr(win, "cmb_minfeas"), "no control for the convention"
    items = [win.cmb_minfeas.itemText(i)
             for i in range(win.cmb_minfeas.count())]
    assert len(items) == 2
    assert any("MaxDEA" in t for t in items)
    assert any("omit" in t.lower() for t in items)
    assert win.cmb_minfeas.currentIndex() == 0, \
        "the MaxDEA-compatible default must stay the default"
    assert hasattr(win, "lbl_malm_note")
    assert win.lbl_malm_note.isHidden(), "the note shows only after a run"


def test_the_regression_tab_offers_simar_wilson(win):
    """Algorithm 1 has to be reachable without writing Python.

    It sat in ``secondstage.py`` untested and unreferenced for a while, which
    is the same as not shipping it: the manuscript said PLEAP did not
    implement it, and from a user's point of view that was true.
    """
    items = [win.cmb_regtype.itemText(i)
             for i in range(win.cmb_regtype.count())]
    assert any("Simar" in t for t in items), items
    assert items[0] == "Tobit", "the conventional default should not change"
    tip = win.cmb_regtype.toolTip()
    assert "not valid" in tip and "bootstrap" in tip, \
        "the control must say why the conventional errors are not enough"
