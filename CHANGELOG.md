# Changelog

## v1.12 — first open-source release

**Licence and distribution.** PLEAP is released under the GNU General Public
License v3.0 or later. The source lives at
https://github.com/ProductivityLab/pleap; a packaged Windows build remains
available for users who do not have Python, and is now built from the same public
source.

### Fixed: returns to scale under output orientation had IRS and DRS swapped

`rts_classify(..., orient="output")` labelled a unit whose CCR intensity
weights sum to less than 1 as DRS and one whose weights sum to more than 1 as
IRS. The rule is the same in both orientations (Seiford and Zhu, 1999): below 1
is IRS, above 1 is DRS. Every scale-inefficient unit was therefore mislabelled
under output orientation — 448 of 453 on data set D; CRS units and the
input orientation were right. Found while comparing with MaxDEA's RTS column
on data set D. That column could not serve as the reference (it agrees with the
standard test, run on MaxDEA's own scores, for 310 and 82 of the 453 units), so
the labels are now checked against the independent NIRS test of Färe,
Grosskopf and Lovell, computed from MaxDEA's CRS, VRS and NIRS scores: all 453
units agree in both orientations, as do all 400 units of the public synthetic
data set SD against Benchmarking's scores (`tests/test_rts.py`). Earlier releases from
v1.0.4 on reported, in the output column, the input-oriented labels, which are
wrong for 208 of the 453 units. **If you have reported output-oriented
returns to scale from any earlier PLEAP, re-run them.**

### Fixed: rank tests lost their tie correction on PLEAP's own scores

The linear programs return a frontier unit's score as 1 only up to rounding
(1 − 4 × 10⁻¹⁵, 1.0 and 1 + 2 × 10⁻¹⁶ all occur). Mann–Whitney,
Kruskal–Wallis and Dunn's test ranked those as distinct values, so the tie
correction disappeared and mid-ranks moved: Kruskal–Wallis *H* was 6.381 on
data set A where SPSS, reading the exported scores, reports 6.356, and the
Mann–Whitney *U* on data set D was 4.5 away from SPSS's. Values that agree to
10⁻⁹ (relative) are now ties, and PLEAP's own scores give SPSS's results on
both data sets.

### Fixed: Tukey HSD p-values were rounded to four decimals

The Tukey table was read off statsmodels' summary, which rounds: p = 3.9 × 10⁻⁶
was reported as 0.0000 and 0.08914 as 0.0891. The values now come from the
result arrays and match SPSS to 10⁻⁶ relative.

### Fixed: truncated regression with a lower truncation point only

Truncated at 0 with no upper bound, BFGS stopped after two iterations with a
line-search failure, at log-likelihood 88.79 against Stata's and R's 105.527, and
reported `converged=False`. The truncation probability is now computed without
cancellation, and when BFGS has not converged the fit tries a trust-region
Newton step on the same likelihood and keeps the better end point. Results
BFGS already reached are unchanged. The GUI never fits this specification (it
truncates at the frontier only); the library does.

### Changed: p-values in the far tail

p-values are computed from upper-tail functions (`sf`) rather than as
1 − cdf, which loses relative precision below about 10⁻¹⁰.

### Changed: test data and the example

**Public tests run on synthetic data, against R.** `tests/data/make_synthetic_data.py`
draws three data sets from stated data-generating processes with fixed seeds:
the example (below), SA (60 units × three periods) and SD (400 units), laid
out like the validation data sets A and D so the same tests run on them.
Reference values come from R — Benchmarking 0.33 for scores, slacks,
projections, peers, NIRS scores, Malmquist distances and the bootstrap;
AER, survival, truncreg, car, rstatix and base R for the second stage and the
group tests — and from likelihood maxima computed in 60-digit arithmetic
(`docs/external_reference/`). Measured agreement: DEA scores 1.3 × 10⁻¹²,
Malmquist 7 × 10⁻¹², Tobit and truncated regression 10⁻⁹ or better, group
tests 10⁻¹⁴ or better.

**The validation data sets A and D are no longer distributed.** They come
from the Korean Innovation Survey (KIS) of the Science and Technology Policy
Institute (STEPI), whose terms do not allow the data to be passed on. The
tests that compare PLEAP with MaxDEA, SPSS 25, Stata 16 and R on them stay in
the suite and run where the data are available (`PLEAP_PRIVATE_DATA`);
elsewhere they are skipped. `docs/validation.md` reports them in aggregate.
This research used the Korean Innovation Survey (KIS) data provided by the
Science and Technology Policy Institute (STEPI).

**The example is synthetic.** `examples/sample_data.xlsx`, the example of the
paper, was drawn from real data up to v1.11; it is replaced by a synthetic
panel with the same layout (60 units × 2021–2023, two inputs, two outputs,
two environmental variables, three groups), and the paper's figures and
numbers are regenerated from it.

The 135-plant second-stage fixture (`ds_b_tobit.csv`) is withdrawn because its
release was not agreed. The MaxDEA bootstrap comparisons run PLEAP at
MaxDEA's own B = 500.

### Fixed: second-stage estimates now reach the maximum on every platform

The Tobit and truncated regressions are fitted by BFGS, which stops once the
gradient falls below its tolerance. Where the likelihood is flat that can leave
the estimate a few 10⁻⁹ from the maximum, at a point that depends on the
platform's floating point: in continuous integration the fit truncated at 0
and 1 on the synthetic data set SD ended 2.2 × 10⁻⁹ from its 60-digit maximum
on Windows and macOS and 9 × 10⁻¹² on Linux. A few Newton steps on the
analytic gradient now follow BFGS, and each is kept only if it shrinks the
gradient without lowering the likelihood. Every fit with a 60-digit reference
is now within 3 × 10⁻¹⁶ of that maximum (measured on Linux), and the public
tests ask 10⁻¹² of it on Windows, macOS and Linux instead of 10⁻⁹–10⁻¹⁰.
What remains against R is R's own distance from the maximum, now reported in
`docs/validation.md` §5. Displayed results, and the example's results and
figures, do not change.

### Fixed: DEA, Malmquist and bootstrap scores on strongly skewed data

v1.0.9 and v1.1.0 divided each input and output column by its **arithmetic
mean** before solving, to keep the linear programs well conditioned. On a
column whose values span many orders of magnitude the arithmetic mean sits near
the top of the range, so the smallest observations scale below the solver's
feasibility tolerance, their constraints stop binding, and the solver returns a
feasible-looking optimum for a problem that is no longer the one posed.

The public data set `tests/data/scaling_case.csv` shows it: one input column
spans 11.3 decades, and the arithmetic mean moves 8 to 10 of the 30 scores in
each of the four radial models, by up to 0.16, in both directions.

Each column is now divided by its **geometric mean**, which the upper tail
cannot drag. Before solving, `scaling_is_risky` asks whether rescaling leaves
any non-zero entry outside [10⁻³, 10³]; where it does, every program is solved
three times — on the scaled data, on the raw data, and on the scaled data with
HiGHS's feasibility tolerances tightened to 10⁻¹⁰ — and the best of the
solutions admissible in the original units is kept (`pleap.core.lp`). DEA,
super-efficiency, returns to scale, the Malmquist distances and the bootstrap
all use this rule.

The third solve and the wider test were added late, when the scaling case was
checked against its optimum computed in **exact rational arithmetic**
(`docs/external_reference/exact_dea.py`). The first form of the fix, with two
solves and a test at [10⁻⁵, 10⁵], still returned θ = 1 for unit U20 under BCC
input orientation, whose optimum is 0.977706, and on 100 random draws
(`docs/scaling_stress.py`, 15,000 unit-model combinations) it missed 32, by up
to 0.51. As released, v1.12 matches the exact optimum on all of them, to
3.5 × 10⁻⁸. R's Benchmarking, run on the same case, misses one VRS unit by
2.5 × 10⁻³.

A post-solve feasibility audit was tried first and does **not** work on its
own: under input orientation `theta = 1` with the DMU as its own peer
satisfies every constraint exactly, which is the shape the fault took. That is
a limitation of checking *primal* feasibility only — a dual test does detect it
(`docs/validation.md` §9) — but it diagnoses without repairing, so the
candidate rule is used instead. Agreement with MaxDEA is unchanged on the
validation data sets that could be re-run, and unit invariance is asserted by
the test suite, including on synthetic technologies whose input column spans
twelve and fourteen orders of magnitude. See `docs/validation.md` §4.

**If you have published results from v1.0.9 or v1.1.0**, re-run them. The fault
only bites when an input or output column is strongly skewed — if the ratio of
the largest to the smallest value in every column is below roughly 10⁵, the two
versions agree to machine precision.

### Fixed: frontier units entered the truncated regression and Algorithm 1

The truncated regression, and Simar–Wilson Algorithm 1 which builds on it,
kept the units with `theta < 1`. The linear programs return a frontier unit as
1 only up to rounding, and on the example used at the time five of the
eleven frontier units came back as 1 − 4 × 10⁻¹⁵. They passed the filter, so the fit used 54
units instead of the 49 strictly inside the frontier, and the coefficient on
*z*₁ came out as 0.143 instead of 0.107. Both models now treat a score within
10⁻⁹ of the truncation point as on it — the tolerance the Tobit has always
used for its censoring rule. Pinned by
`test_frontier_units_a_rounding_error_below_one_stay_out` and by
`tests/test_truncreg_reference.py`, which fits the units strictly inside the
frontier of the (now synthetic) example in R's `truncreg`.

### Fixed: standard errors of the Tobit and truncated regressions

These standard errors come from a Hessian built by differencing the analytic
gradient. Forward differences leave an error of the order of the step: up to
2.4 × 10⁻⁵ (relative) on the Tobit reference data and 4.5 × 10⁻⁵ on the
truncated example — invisible at the three or four digits a table shows, but
real. Central differences reduce it to O(step²); both models now agree with a
60-digit evaluation of the same likelihood to 5 × 10⁻¹⁰
(`docs/external_reference/exact_tobit.py`, `exact_truncreg.py`).

Re-running the Tobit reference in R showed why the forward-difference error had
gone unnoticed: the standard errors and Wald statistic stored as R's values in
`tests/test_tobit_reference.py` were not R's. They matched PLEAP's own
forward-difference Hessian to 4.5 × 10⁻¹¹ and missed R by up to 8.1 × 10⁻⁶
(standard error) and 1.2 × 10⁻⁴ (Wald). They are now taken from R 4.3.3 with
survival 3.5-8 (`docs/external_reference/tobit_survreg.R`, removed in v1.12 with
the data set B fixture), which the 60-digit
computation reproduces to every printed digit, and PLEAP agrees with them to
7.3 × 10⁻⁹. The coefficients, scale and log-likelihood were R's all along. The
test docstring also gave the wrong R call (`type = "left"`); the call that
reproduces the reference is `type = "right"` with scores within 10⁻⁹ of 1
censored.

### Added: open-source references for the truncated regression and the bootstrap

- Truncated regression against R `truncreg` 0.2-5 on the example's units
  inside the frontier, and against the 60-digit likelihood maximum
  (`tests/test_truncreg_reference.py`).
- Bootstrap against Benchmarking 0.33 `dea.boot` on the synthetic data set SA
  and on validation data set A (`docs/external_reference/reference_sa.R`,
  `bootstrap_benchmarking.R`, `docs/bootstrap_benchmarking_aligned.py`;
  `docs/validation.md` §7).

### Added: a choice about infeasible Malmquist distances

Under VRS a cross-period distance can be infeasible. PLEAP substitutes 1, which
is what MaxDEA does. That convention is not neutral: on validation panel A,
three substituted rows out of sixty lower the first transition's mean
productivity growth from 9.8% to 4.4%, and on the example used up to v1.11
they reversed its direction (`docs/validation.md` §8).

`malmquist(..., on_infeasible="omit")` now returns NaN for those rows instead.
The default stays `"substitute"` so the MaxDEA reference comparisons remain
meaningful. The count, the affected rows and the denominator are reported
either way — and the denominator is now the number of **cross-period**
distances (two per unit-period row), not all four, which earlier reporting
overstated by a factor of two.

### Fixed: `dea.py` could not run on a fresh install

The radial DEA module solved its linear programs by writing an LP file and
spawning an external CBC process through `pulp`. PuLP 4.0 removed
`PULP_CBC_CMD` and stopped bundling a CBC binary, so `pip install -r
requirements.txt` followed by any DEA run raised `AttributeError` on a
current environment. (Packaged Windows builds were unaffected, having been
frozen with an older PuLP.)

Every linear program is now solved in-process through SciPy's HiGHS interface,
as `bootstrap.py` and `malmquist.py` already did. There is no external solver
dependency left, the package installs from wheels alone on every platform, and
DEA is substantially faster. The formulation and the public API are unchanged,
and the scores, intensity weights and slacks reproduce the previous CBC results
and the MaxDEA reference.

### Fixed: negative zeros in the results table

Slacks and intensity weights are bounded at zero by the linear program, but
HiGHS can return `-1e-17` for one that is exactly zero, which the results
table printed as `-0.000000`. Both are now clamped at their lower bound, so
the noise never reaches a peer list or a table a researcher copies into a
paper. Scores are unaffected.

### Fixed: the scaling safeguard actually guards something now

v1.12's first cut paired the geometric-mean scaling with a post-solve audit
that re-checked each solution against the constraints in the original units.
That audit cannot detect the failure it was written for. Under input
orientation `theta = 1` with the DMU as its own and only peer is feasible with
zero violation, which is precisely the answer the broken scaling returned —
feasibility is not optimality.

The safeguard is now `scaling_is_risky`, a test of the *data* made before
solving: does rescaling put any non-zero entry within two decades of the
solver's tolerance, or the same factor above one? Where it does, the linear
program is solved on both the scaled and the raw data and the better objective
is kept, since both are feasible points of the same problem. Applied in `dea`,
`malmquist` and `bootstrap`. Well-conditioned data is detected as such and pays
nothing; the affected 135-DMU data set takes 1.05 s instead of 0.5 s.

Verified by restoring the old arithmetic-mean scaling deliberately: the
cross-solve recovers every MaxDEA value even then.

### Added: Malmquist infeasible-substitution reporting

Replacing an infeasible cross-period distance with 1 is a MaxDEA-compatibility
default, and it propagates into MI and TC where a reader cannot see it. The
result now carries an `Infeasible_substitutions` column whenever any occurred,
with the count and a note in `DataFrame.attrs`. Under CRS the cross-period
programs are always feasible and the column is absent.

### Added: automated validation

`tests/` now holds the reference comparisons that were previously carried out by
hand at release time. Those on the public synthetic data run in continuous
integration; those on the validation data that cannot be redistributed run
where the data are held and are skipped elsewhere:

- CCR and BCC scores against MaxDEA for both orientations on the validation
  data;
- Malmquist MI / EC / TC against MaxDEA under both returns-to-scale
  assumptions;
- Tobit against R `survreg(dist = "gaussian")` — every coefficient, standard
  error, the scale row, the log-likelihood and the Wald statistic;
- Levene, *t*-test, Mann–Whitney and Kruskal–Wallis against the SPSS
  conventions;
- bootstrap agreement with MaxDEA, stated against its own seed-to-seed
  variability, since a stochastic procedure cannot be reproduced exactly.

Alongside those, properties that must hold on any data: unit invariance,
MI = EC × TC identically, Tobit collapsing onto OLS without censoring,
non-negative bootstrap bias, results independent of the worker-process count,
closed-form scores on textbook technologies, and weak versus strong efficiency.
Continuous integration runs the suite on Linux, Windows and macOS for Python
3.10 and 3.12, and installs the built wheel into a clean environment.

### Fixed: the window logo and icon

The lab logo was replaced with the current mark, squared up with even margins
so it stays legible down to 16 px.

Fixing it exposed a second problem. `resource_path` resolved two directories
up from the module, which pointed outside the package once the source moved
under `src/`: running from a checkout showed the text `[LOGO]` instead of the
logo, and a wheel install shipped no assets at all. The assets now live inside
the package (`src/pleap/assets/`), are declared as package data, and are
resolved relative to the package, so they travel with a checkout, a wheel and
a PyInstaller bundle alike. Pre-1.12 bundle layouts are still accepted.

### Changed

- The package moved to a `src/` layout; `python -m pleap` and the installed
  `pleap` command replace `pleap_main.py`.
- Packaging moved to `pyproject.toml`. The interface is an optional extra, so
  `pip install pleap` gives the analysis layer with no Qt dependency and no
  need for a display.
- The sample data sets moved from `sample/` to `examples/`, with a guide.
- Documentation added: `README.md` (English), `docs/validation.md`,
  `CONTRIBUTING.md`, `CITATION.cff`.

---

## v1.1.0

- **New "Visualization" tab** (tab 7, before License). It only renders results
  that have already been computed — no analysis routine was changed, and every
  DEA / bootstrap / Malmquist / regression number is identical to v1.0.9.
  - **Figures.** Efficiency frontier with the DMUs and their radial projections
    (exact for one input and one output; a mean-normalised 2-D projection of the
    technology otherwise), efficiency distribution, efficiency ranking,
    bootstrap bias-corrected scores with confidence intervals, and four
    Malmquist charts (MI / EC / TC trend by period, EC–TC quadrant with the
    MI = 1 iso-curve, index by DMU, distribution by period).
  - **Appearance.** Seven colour palettes (including a grayscale set for print
    and a colour-blind-safe set), five visual styles (Academic, Grid, Minimal,
    Dark, Classic box) and an adjustable font scale, applied instantly.
    Figures export to PNG / PDF / SVG / TIFF at 300 dpi.
  - **Publication tables.** Second-stage regression results in journal format
    (several models side by side, significance stars, standard errors in
    parentheses or below the coefficient, Log(scale), Wald / F, log likelihood),
    plus descriptive statistics, the Malmquist summary and group tests.
    Output as text, HTML for Word, Markdown or LaTeX.
  - **Appendix export.** One click writes every selected result — tables and
    figures — to a self-contained HTML file that opens directly in MS Word, or
    to a formatted Excel workbook.
- Coefficients smaller than 1e-3 are shown in scientific notation in the
  publication tables, so results for large-scale environmental variables are no
  longer displayed as 0.000.
- matplotlib is now bundled with the Windows build.

## v1.0.9

- **MPI: infeasible cross-period distances handled like MaxDEA.** Under VRS the
  cross-period LPs can be infeasible (a unit's output lies outside the other
  period's frontier span, or an input is zero); previously the affected MI / TC
  cells were blank. Following the MaxDEA convention the infeasible cross-period
  score is replaced with 1, so MI / TC are always reported and reproduce the
  MaxDEA reference exactly.
- **Numerical stability: column scaling for every DEA linear program.** With
  raw magnitudes around 1e5–1e6 the LP solver could terminate ~1e-3 short of
  the optimum (observed on a handful of DMUs). All inputs / outputs are now
  mean-normalised before solving — DEA scores and lambdas are invariant to
  units of measurement, and slacks are mapped back to the original units.
  *(Superseded in v1.12: see the fault described above. The intent was right,
  the arithmetic mean was the wrong centre.)*
- **Bootstrap bandwidth recalibrated to the MaxDEA reference:**
  h = 0.9 · sd(eff) · (2n)^(−1/5) (Silverman rule with the reflected sample
  size). Verified on three independent datasets (highly efficient, mixed and
  highly inefficient score distributions); the small systematic offset of the
  previous rule disappears and the bias-corrected Mean now agrees with MaxDEA
  to ~0.2–0.9% (median), within Monte-Carlo error.

## v1.0.8

- **Tobit regression now matches R `survreg(dist="gaussian")` / SPSS.** The MLE
  failed to converge when environmental variables spanned many orders of
  magnitude (e.g. recycling volumes in the millions), collapsing the
  coefficients to ~0. The covariates are now standardised internally, the
  likelihood is maximised with analytic gradients, and the coefficients and
  standard errors are transformed back to the original scale. Validated:
  with no censoring the Tobit reproduces OLS to machine precision. The result
  table now also reports the `Log(scale)` row and the Wald statistic, mirroring
  R's `survreg` / `AER::tobit` output.
- **Independent-samples t-test follows the Levene decision (SPSS convention).**
  Previously the unequal-variance (Welch) result was always reported. Now
  Levene's test (centred on the mean) decides: p ≥ 0.05 reports the
  equal-variance (Student) t-test, p < 0.05 reports Welch. Both rows are shown.
- **Truncated regression** now truncates only at the efficient frontier
  (Simar–Wilson convention), which is numerically stable and converges
  reliably; standard-error reporting is guarded against singular Hessians.
- The bootstrap is a stochastic procedure: its Mean / SD / CI match MaxDEA
  within Monte-Carlo error (two PLEAP runs with different seeds differ from
  each other by about as much as PLEAP differs from MaxDEA). Unlike the
  deterministic DEA / MPI results it cannot be reproduced bit-for-bit.

## v1.0.7

- **Bootstrap now matches the MaxDEA reference.** Corrected the smoothed-bootstrap
  draw (Simar & Wilson 1998): the bandwidth is computed on the original
  efficiencies, draws are variance-corrected toward the resample mean, and they
  are reflected at both the 0 and 1 boundaries. This removes the occasional
  explosive `Bias` values seen on highly inefficient data and reproduces
  MaxDEA's Mean / SD / CI for all four models (IBCCB, OBCCB, ICCRB, OCCRB).
- **Malmquist (MPI) fixed.** The distance functions are now solved under the
  **VRS (BCC), input-oriented** technology by default (v1.0.6 used CRS, giving
  different EC / TC / MI). Reproduces MaxDEA exactly. Model and orientation are
  now selectable on the Malmquist tab.
- **MaxDEA-style output.** Bootstrap and Malmquist result tables now use the
  same column layout as MaxDEA.

## v1.0.6

- **Faster Bootstrap.** The bootstrap LP is now solved in-process with
  `scipy`/HiGHS instead of launching an external CBC solver for every single
  linear program (the old `pulp` path). On a typical problem this alone is
  about 10–50× faster.
- **Parallel replications.** Bootstrap replications now run across multiple CPU
  cores (see the new **Cores** control on the Bootstrap tab). Results are fully
  reproducible and independent of the number of cores used.
- The statistical method is unchanged from v1.0.5 (identical bias / mean /
  CI values).

## v1.0.5

- **Bootstrap DEA bias fix.** The bootstrap now follows the correct
  Simar–Wilson data-generating process: in every replication the *original*
  observation is re-evaluated against the pseudo-frontier (the observation is
  not a member of the reference set, so its score may exceed 1 and is no longer
  clipped). v1.0.4 evaluated the rescaled pseudo point inside the pseudo
  reference set and clipped every score to ≤ 1, which forced the estimated
  bias of efficient DMUs to be **negative** — impossible for the radial DEA
  estimator. Bias is now non-negative by construction and matches reference
  software (e.g. MaxDEA).
- **Single method.** The Simar–Wilson (1998) / KSW2008 selector was removed.
  The program now uses only **Kneip, Simar & Wilson (2008)**, with the ratio
  bias-correction applied per replicate (theta_bc_b = theta_hat² / theta*_b).
- **Output columns.** Bootstrap results now report
  `Score, Bias, Bias_corrected, Mean, Median, SD, CI_low, CI_high`
  (the bootstrap Mean/Median/SD/CI are computed from the bias-corrected
  replicate distribution).
- Default replications B raised from 1000 to 2000.
