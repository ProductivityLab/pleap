"""Optional helper: run pyDEA's CCR input-oriented model headlessly.

Imported by ``crosstool_check.py`` if present.  Two obstacles, both of them
pyDEA's packaging rather than its mathematics:

  * ``pyDEA.core.utils.dea_utils`` does ``from tkinter import StringVar, ALL``
    at module scope, so the whole GUI toolkit is pulled in even to reach the
    LP models.  Nothing on the solver path calls either name, so a small
    stand-in is installed before the import.
  * pyDEA needs the pre-4.0 PuLP API (``LpStatus``), which newer PuLP removed
    - the same breakage that forced PLEAP off PuLP (see CHANGELOG).  Install
    it in its own environment:

        python -m venv pydea-env
        pydea-env/bin/pip install pyDEA "pulp==2.7.0" "setuptools<81"
"""
import sys
import types

import numpy as np

if "tkinter" not in sys.modules:                       # must precede pyDEA
    _tk = types.ModuleType("tkinter")

    class _Var:
        def __init__(self, *a, **k): self._v = ""
        def set(self, v): self._v = v
        def get(self): return self._v

    _tk.StringVar = _tk.IntVar = _tk.DoubleVar = _Var
    _tk.ALL = "all"
    sys.modules["tkinter"] = _tk

from pyDEA.core.data_processing.input_data import InputData          # noqa: E402
from pyDEA.core.models.bound_generators import (                     # noqa: E402
    generate_upper_bound_for_efficiency_score)
from pyDEA.core.models.envelopment_model import (                    # noqa: E402
    EnvelopmentModelInputOriented)
from pyDEA.core.models.envelopment_model_base import (               # noqa: E402
    EnvelopmentModelBase)
from pyDEA.core.models.envelopment_model_decorators import (         # noqa: E402
    DefaultConstraintCreator)


def pydea_ccr_input(X, Y):
    """CCR input-oriented efficiency from pyDEA, in the row order of ``X``."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    data = InputData()
    n, m = X.shape
    s = Y.shape[1]
    for i in range(n):
        name = f"D{i:03d}"
        for j in range(m):
            data.add_coefficient(name, f"in{j}", float(X[i, j]))
        for r in range(s):
            data.add_coefficient(name, f"out{r}", float(Y[i, r]))
    for j in range(m):
        data.add_input_category(f"in{j}")
    for r in range(s):
        data.add_output_category(f"out{r}")

    model = EnvelopmentModelBase(
        data,
        EnvelopmentModelInputOriented(generate_upper_bound_for_efficiency_score),
        DefaultConstraintCreator())
    sol = model.run()
    code = {v: k for k, v in data.DMU_code_to_user_name.items()}
    return np.array([sol.get_efficiency_score(code[f"D{i:03d}"])
                     for i in range(n)], float)
