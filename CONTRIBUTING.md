# Contributing to PLEAP

Thank you for looking. PLEAP is used to produce numbers that end up in
published papers, so the bar for changes to the analysis layer is deliberately
high — but bug reports, documentation fixes and new validation cases are always
welcome.

## Reporting a problem with a result

This is the most valuable kind of report. Please include:

1. **A small data set that reproduces it.** Ten or twenty DMUs is usually
   enough. If the data cannot be shared, a rescaled or synthetic version that
   still shows the problem works just as well — DEA scores are invariant to the
   units of each column, so multiplying every input and output by a constant
   changes nothing but the numbers on the page.
2. **The model, orientation and any other settings** you used.
3. **What another program reports** for the same data, if you have it (MaxDEA,
   R, Stata, SPSS). A disagreement with an established program is treated as a
   bug in PLEAP until shown otherwise.
4. **Your PLEAP version** (shown in the window title) and how you installed it.

## Development setup

```bash
git clone https://github.com/ProductivityLab/pleap.git
cd pleap
python -m venv .venv && source .venv/bin/activate
pip install -e ".[gui,test]"
pytest -m "not slow"
```

On a head-less machine, set `QT_QPA_PLATFORM=offscreen` and `MPLBACKEND=Agg`.

## The rules for the analysis layer

`src/pleap/core/` computes; `src/pleap/viz/` and `src/pleap/gui/` only display.
A change to a figure or a table must never need a change under `core/`.

**Every change that touches an estimator has to keep the reference tests
passing.** If a change deliberately moves a published number — a corrected
formula, a different convention — then update the reference value in the same
commit and say in the commit message what the new value was checked against.
Never relax a tolerance to make a test pass; a tolerance is a claim about the
method's accuracy, and loosening one silently is how a wrong result ships.

If you add an estimator, add with it:

- a comparison against something outside PLEAP (reference software, a published
  worked example, or a closed form), and
- at least one property that must hold for every input — an identity, an
  invariance, a sign restriction. These catch the failures that reference
  comparisons miss, because they hold on data nobody has tried yet.

`tests/test_scaling.py` is the example to follow. The arithmetic-mean scaling
bug it guards against reproduced every reference value correctly on two data
sets and was wrong by 0.32 on a third; what exposed it was an invariance test,
not a comparison.

## Conventions worth knowing

PLEAP follows SPSS where statistical packages disagree, because that is what
most of its users will be checking against:

- Levene's test is centred on the **mean**, not the median (scipy's default).
- The independent-samples *t*-test reports Student when Levene gives
  *p* ≥ 0.05 and Welch otherwise; both rows are always shown.
- Mann–Whitney uses the tie-corrected normal approximation **without** a
  continuity correction.
- Under VRS, an infeasible cross-period Malmquist distance is replaced by 1,
  as MaxDEA does.

Each of these is pinned by a test with the reason in its docstring. If you
change one, change the docstring too.

## Style

Follow the surrounding code. Comments should say *why*, and in this codebase
that usually means recording what went wrong before — those notes are the
reason the bugs did not come back.

## License

Contributions are accepted under the GPL v3 or later, the licence PLEAP is
distributed under.
