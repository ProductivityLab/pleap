# Validation record

Every estimator in `pleap.core` is checked against software outside PLEAP. This
document records what was compared, against what, how closely it agreed, and —
where reference programs disagree with each other — which convention PLEAP
follows and why.

The comparisons are not a one-off claim made at release. Where the data are
public they live in `tests/` and run in continuous integration, so a change
that moves one of those numbers fails the tests; where they are not, the tests
run where the data are held, or the comparison is recorded from an earlier run.
Two kinds of data are involved:

- **Public, synthetic.** The example of the paper (E), two data sets laid out
  like A and D (SA, SD) and the case of the scaling fault (SC) are drawn by
  `tests/data/make_synthetic_data.py` from stated data-generating processes,
  and checked against R (Benchmarking, AER, survival, truncreg, car, rstatix,
  base R), against likelihood maxima computed in 60-digit arithmetic and
  against linear-programming optima computed in exact rational arithmetic.
  Anyone can re-run these comparisons from the repository; they run in
  continuous integration on every push to the main branch and every pull request.
- **Real, not distributed.** Data sets A and D come from the Korean Innovation
  Survey (KIS), 2018 manufacturing, provided by the Science and Technology
  Policy Institute (STEPI), whose terms of use do not allow the data to be
  passed on. The comparisons with MaxDEA, SPSS, Stata and R on them are
  recorded below in aggregate; their tests are in the suite and run where the
  data are available (`PLEAP_PRIVATE_DATA`, `tests/data/README.md`), and are
  skipped elsewhere. Data sets B and C are confidential as well; their MaxDEA
  comparisons are recorded below in aggregate, from earlier runs, and have no
  tests.

This research used the Korean Innovation Survey (KIS) data provided by the
Science and Technology Policy Institute (STEPI).

| Reference | Version | Used for |
|---|---|---|
| MaxDEA | MaxDEA 6.6 Ultra (MaxDEA Software Ltd.) | CCR / BCC scores, slacks, projections and peers; NIRS scores; Malmquist indices; bootstrap statistics |
| R `AER::tobit`, `survival::survreg(dist = "gaussian")` | R 4.3.3, AER 1.2-12, survival 3.5-8 | Tobit regression (SD; data set D) |
| R `truncreg` | R 4.3.3, truncreg 0.2-5 | Truncated regression (the example's Algorithm 1 sample; SD truncated at 0 and at 1; data set D truncated at 0) |
| Stata `truncreg` | Stata 16 | Truncated regression on data set D, truncated at 0 and 1, and at 0 only |
| R `Benchmarking` (`dea`, `slack`, `malmq`, `dea.boot`) | R 4.3.3, Benchmarking 0.33 | On the public data: scores, slacks, projections, peers, NIRS scores, Malmquist distances; the bootstrap as an open-source second reference (Section 7) |
| R `car::leveneTest`, `rstatix::dunn_test`, `stats` (`wilcox.test`, `kruskal.test`, `t.test`, `oneway.test`, `TukeyHSD`, `ptukey`, `lm`) | R 4.3.3, car 3.1-2, rstatix 0.7.2 | Group tests and OLS on the public data set SD |
| 60-digit maximum likelihood (`docs/external_reference/exact_mle.py`) | mpmath 1.4 | Tobit and truncated regression, including truncation at 0 and 1 |
| SPSS | IBM SPSS Statistics 25 for data set D; data set A's printouts come from version 27 or later (their *t*-test output carries effect sizes, which SPSS added in version 27) | Tobit (through SPSS's R extension), OLS, independent-samples *t*-test, Levene, Mann–Whitney, Kruskal–Wallis with Dunn's test, one-way and Welch ANOVA, Tukey HSD, Games–Howell |
| SciPy | 1.10+ (pinned by the test run) | Levene, *t*-tests, Mann–Whitney, Kruskal–Wallis, ANOVA |

---

## 1. Data sets used

| Tag | DMUs | Inputs / outputs | Periods | Character |
|---|---|---|---|---|
| **E** | 60 | 2 / 2 | 3 | The example of the paper, `examples/sample_data.xlsx`. Synthetic. Three groups, two environmental variables. **Public.** |
| **SA** | 60 | 2 / 1 | 3 | Synthetic, laid out like A. R Benchmarking references: DEA with slacks and peers, Malmquist distances, bootstrap. **Public.** |
| **SD** | 400 | 2 / 1 | — | Synthetic, laid out like D: two environmental variables, a two- and a four-group coding, the BCC input-oriented score as an eight-decimal column. Benchmarking's CCR, BCC and NIRS scores; R's second-stage regressions and group tests. **Public.** |
| **A** | 60 | 2 / 1 | 3 | Korean Innovation Survey (STEPI). Well-conditioned columns (input spread ≤ 10⁴). Reference sheet carries full double precision. **Not distributed.** |
| **SC** | 30 | 2 / 1 | — | Synthetic: input x2 log-uniform over 11.3 decades, the structure of the scaling fault of Section 4. The optimum in exact rational arithmetic and Benchmarking's scores. **Public.** |
| **B** | 135 | 3 / 1 | 3 | Industrial environmental data, with one strongly skewed input column. **Not distributed.** |
| **C** | 61 | 3 / 1 | — | Retirement-pension data. Magnitudes 10⁵ … 10⁹. **Not distributed.** |
| **D** | 453 | 2 / 1 | — | Korean Innovation Survey (STEPI), 2018 manufacturing. Second-stage references in R, Stata and SPSS; a 207-firm subset (**D207**) with its own group coding carries SPSS's OLS, *t*-test and ANOVA. **Not distributed.** |

The synthetic data, their generating processes and the R scripts that produce
the reference values are described in `tests/data/README.md`. Their tests
(`tests/test_*_reference.py`) compare with R at full double precision, so the
agreement they assert is tighter than MaxDEA's six printed decimals allow.

A and D are not distributed because STEPI's terms do not allow the KIS data to
be passed on, and this record reports them only in aggregate: no unit's data
or results appear here. Up to v1.11 the test fixtures included data set A,
with each column multiplied by a constant; from v1.12 neither A nor D is kept
in the repository. A 135-plant
second-stage file derived from B, shipped up to v1.11 as `ds_b_tobit.csv`, is
withdrawn because its release was not agreed. The example used up to v1.11
was drawn from real data and has been replaced by E.

---

## 2. Radial DEA — MaxDEA and Benchmarking

**On the public data, against R Benchmarking 0.33** (lp_solve), at full
double precision (`tests/data/sa_benchmarking_dea.csv`,
`sd_benchmarking_dea.csv`, `example_benchmarking_dea.csv`):

| | SA (60 units) | SD (400 units) | E, last period (60 units, 2 outputs) |
|---|---:|---:|---:|
| Scores, all four models | 1.3 × 10⁻¹² | 1.3 × 10⁻¹² | 2.8 × 10⁻¹⁴ |
| Second-phase slacks and projections, relative to max(1, \|value\|) | 3.1 × 10⁻¹⁰ | — | — |
| Reference sets | 60 of 60 identical, all four models | — | — |
| Intensity weights | 9 × 10⁻¹¹ | — | — |

On SA 23 of the 60 units carry a slack under BCC input orientation, so the
second phase is exercised, not only the radial step. Benchmarking and MaxDEA
maximise the plain sum of the slacks in the second phase; PLEAP maximises it
on the rescaled problem (each column divided by its geometric mean, Section 4),
which weights the slacks differently and could select a different solution
where the second phase has more than one optimum. On SA and on A it selects
the same slacks as both references.

**On the validation data, against MaxDEA.** Maximum absolute difference
between PLEAP's efficiency score and MaxDEA's, over every DMU. MaxDEA exports six decimals, so **5 × 10⁻⁷ is its display
resolution** — any figure at that level means the two programs agree on every
digit MaxDEA reports.

| Model | Data set A | Data set B | Data set C | Data set D |
|---|---|---|---|---|
| BCC, input-oriented | 3.96 × 10⁻¹³ | 5.00 × 10⁻⁷ | 4.96 × 10⁻⁷ | 8.19 × 10⁻¹³ |
| BCC, output-oriented | 4.86 × 10⁻¹³ | 4.98 × 10⁻⁷ | 4.84 × 10⁻⁷ | 7.20 × 10⁻¹³ |
| CCR, input-oriented | 3.90 × 10⁻¹³ | 4.98 × 10⁻⁷ | 4.67 × 10⁻⁷ | 6.26 × 10⁻¹³ |
| CCR, output-oriented | 3.90 × 10⁻¹³ | 4.98 × 10⁻⁷ | 4.67 × 10⁻⁷ | 6.26 × 10⁻¹³ |

No DMU in any of the sixteen comparisons differs by more than 10⁻⁵. MaxDEA's
NIRS scores for data set D, which Section 2a uses, are reproduced by the same
linear program with Σλ ≤ 1 to 4.8 × 10⁻¹³.

**Beyond the score, on data set A.** The second-phase slacks and the projected
points agree with MaxDEA to 7 × 10⁻¹¹ of max(1, |value|) in all four models,
except one output slack under CCR output orientation at 2.5 × 10⁻⁸. Every DMU
has the same reference set in both programs (60 of 60, all four models), and
the intensity weights agree to MaxDEA's six printed decimals (largest gap
5.0 × 10⁻⁷).

Tests: `tests/test_dea_reference.py` (Benchmarking on the public data; MaxDEA
where the validation data are available).

---

## 2a. Returns to scale

PLEAP classifies the radial projection of each unit: CRS when the CCR and BCC
scores agree; otherwise IRS when the CCR intensity weights sum to less than 1
and DRS when they sum to more. Seiford and Zhu (1999, *Omega* 27(1):1–11,
https://doi.org/10.1016/S0305-0483(98)00025-5) show that for a
scale-inefficient unit every alternate optimum falls on the same side of 1, so
one optimal solution decides, and that the rule runs in the **same direction
under both orientations**.

**Fault found in v1.12 development.** Up to v1.11 the output-oriented branch
had the labels swapped, so every scale-inefficient unit was mislabelled under
output orientation (448 of 453 on data set D). Releases from v1.0.4 on had
reported the input-oriented labels in the output column instead, wrong for
208 of the 453. The input orientation was right throughout.

**How it was found, and why MaxDEA is not the reference here.** Comparing with
MaxDEA's RTS column on data set D gave 310 of 453 agreements under input
orientation and 376 under output orientation, which looked like noise in both.
The test that settles it shares no code with either rule: Färe, Grosskopf and
Lovell's comparison with the non-increasing-returns score — CRS if
CCR = BCC, DRS if NIRS = BCC, IRS otherwise — computed from **MaxDEA's own**
CCR, BCC and NIRS scores. MaxDEA's RTS column agrees with that test, on its own
scores, for 310 of 453 units under input orientation and 82 of 453 under
output orientation:

| NIRS test → | input: CRS | DRS | IRS | output: CRS | DRS | IRS |
|---|---:|---:|---:|---:|---:|---:|
| MaxDEA says CRS | 5 | 0 | 0 | 5 | 0 | 0 |
| MaxDEA says DRS | 0 | 0 | 139 | 0 | 0 | 159 |
| MaxDEA says IRS | 0 | 4 | 305 | 0 | 212 | 77 |

The rule behind MaxDEA's column is not documented in anything we could
consult, and it matches neither the sum-of-weights rule nor the NIRS test, so
it is not used as a reference. PLEAP v1.12 agrees with the NIRS test on all 453
units in both orientations (input: 5 CRS, 4 DRS, 444 IRS; output: 5 CRS,
212 DRS, 236 IRS), and on a one-input example whose answer is known by
construction, including a unit that is IRS under input orientation and DRS
under output orientation.

The same test on the public data set SD, with **Benchmarking's** CCR, BCC
and NIRS (`RTS = "drs"`) scores, agrees with PLEAP on all 400 units in both
orientations (input: 4 CRS, 229 DRS, 167 IRS; output: 5 CRS, 333 DRS,
62 IRS). SD's technology has a fixed input requirement, so all three classes
occur in numbers.

Tests: `tests/test_rts.py`.

---

## 3. Malmquist index — MaxDEA and Benchmarking

| Component | Data set A (VRS, input) | Data set B (CRS) | SA, against Benchmarking (CRS and VRS, both orientations) |
|---|---|---|---|
| MI | 7.68 × 10⁻¹¹ | 4.99 × 10⁻⁷ | 6.8 × 10⁻¹² |
| EC | 1.30 × 10⁻¹¹ | 4.99 × 10⁻⁷ | 6.8 × 10⁻¹² |
| TC | 5.34 × 10⁻¹² | 4.98 × 10⁻⁷ | 6.8 × 10⁻¹² |

`Benchmarking::malmq` returns the four distance functions; the index is
formed from them under each orientation (`tests/data/sa_benchmarking_mpi.csv`).
Under VRS it returns an infinite distance where a cross-period program is
infeasible — 3 of 240 cross-period distances on SA under input orientation,
11 under output orientation — and PLEAP flags exactly the same ones. With
`on_infeasible="omit"` those rows are left empty and every other row matches;
with the default, the index equals the one formed after setting each such
distance to 1, MaxDEA's convention.

Two notes on specification, both of which caused apparent disagreements during
development and neither of which was a fault:

- **The returns-to-scale assumption has to match the reference run.** Data set
  A's MaxDEA sheet was produced under VRS, data set B's under CRS. Compared
  under the wrong assumption the indices differ by up to 2.7 — which is not a
  bug but is indistinguishable from one if the setting is not recorded. PLEAP
  therefore exposes the model and orientation on the Malmquist tab and prints
  both in the output.
- **Under CRS the input- and output-oriented indices coincide**, which is why
  data set B agrees under both orientations. This identity is asserted in
  `tests/test_malmquist.py` rather than left as a remark.

`MI = EC × TC` holds identically (to 10⁻¹²), not approximately.

Under VRS a cross-period program can be infeasible. Following MaxDEA, PLEAP
replaces the infeasible distance with 1 so the index is always reported; before
v1.0.9 the affected cells were blank. Section 8 measures what that does.

Tests: `tests/test_malmquist.py`.

---

## 4. The arithmetic-mean scaling fault, and a gap in its first fix

This section records a fault in PLEAP v1.0.9–v1.1.0 that the test suite found
after those versions had been released, and a gap in the first form of its
fix that an exact reference exposed while the paper was being revised. Both
are corrected in v1.12.

DEA scores are invariant to the unit of measurement of each input and output.
LP solvers are not: with raw magnitudes around 10⁵–10⁶, HiGHS was observed to
terminate about 10⁻³ short of the optimum on a handful of DMUs, so v1.0.9 began
dividing each column by its **arithmetic mean** before solving.

That is safe for a well-behaved column, and it was verified against MaxDEA on
data sets A and C. It is not safe for a skewed one: on a column spanning many
orders of magnitude the arithmetic mean puts the smallest entries below HiGHS's
default primal feasibility tolerance of 10⁻⁷, their constraints stop binding,
and the solver returns the optimum of a different problem. Everything below is
measured on the public data set SC (`tests/data/scaling_case.csv`, Section 1),
which has such a column.

### 4.1 The fault, on the public case

SC's input x2 runs from 46.9 to 9.10 × 10¹² (11.3 decades). Its
arithmetic mean, 3.82 × 10¹¹, puts the smallest entry at 1.2 × 10⁻¹⁰; its
geometric mean, 1.24 × 10⁶, at 3.8 × 10⁻⁵, and the largest at 7.3 × 10⁶.

One solve per unit after dividing by the arithmetic mean, against the optimum
computed in exact rational arithmetic (Section 4.3):

| Model | units off by more than 10⁻⁴ | largest error |
|---|---|---|
| CCR, input | 9 of 30 | 0.153 |
| CCR, output | 10 of 30 | 0.156 |
| BCC, input | 8 of 30 | 0.134 |
| BCC, output | 10 of 30 | 0.122 |

The errors go both ways. U18, on the frontier under every model, comes back at
0.847049 under CCR input orientation; U03, whose optimum is
0.691022, at 0.697344. The three HiGHS settings (`highs`,
`highs-ds`, `highs-ipm`) return the same two values, so the rescaled problem,
not the algorithm, is at fault.

Checking the returned solution against the constraints in the original units
catches half of this. U18's understatement is not a feasible point of the
problem posed, and the check rejects it. U03's overstatement is feasible, with
zero violation, and the check passes it; a dual check does detect it
(Section 9). Under input orientation θ = 1 with the DMU as its own peer is
always feasible, so an overstatement to 1 is invisible to any check of the
solution's feasibility.

### 4.2 The fix

**Geometric-mean scaling.** Each column is divided by its geometric mean, the
centre of the column in log space, which the upper tail cannot drag: after
scaling, the smallest and largest entries sit more symmetrically around 1.
Dividing by a constant cannot change a column's largest-to-smallest ratio;
what it changes is where that range sits relative to the solver's *absolute*
tolerance, and centring it is the whole effect.

**A test of the data, and three candidates.** Before solving,
`scaling_is_risky` asks whether rescaling leaves any non-zero entry outside
[10⁻³, 10³]. That is a question about the data and, unlike the solution, can
be answered in advance. Where the answer is yes, every program is solved three
times — on the scaled data and on the raw data at HiGHS's default tolerances,
and on the scaled data with the primal and dual feasibility tolerances
tightened from 10⁻⁷ to 10⁻¹⁰ — each candidate is checked against the
constraints in the original units, and among the admissible ones the smaller
θ, or the larger φ, wins (`pleap.core.lp`). Every admissible point bounds the
optimum, so an extra candidate can only improve the answer. DEA,
super-efficiency, returns to scale, the Malmquist distances and the bootstrap
all go through this one rule. On data the test passes, one solve is run and
nothing extra is paid.

**The first form of the fix, and what it missed.** Until this revision the
rule had two candidates, scaled and raw, and the test flagged entries outside
[10⁻⁵, 10⁵]. That restored every MaxDEA value on A, B and C and passed every
invariance test. The exact reference of Section 4.3 found what it still
missed. On SC, U20 under BCC input orientation came back as θ = 1, with U20 as
its own and only peer, from both candidates, while its optimum is
0.977706: at the default dual feasibility tolerance HiGHS took the
improving direction for rounding, and at 10⁻¹⁰ it does not. The random draws of
Section 4.3 also found a single solve missing, by 3.0 × 10⁻⁵, on a column
spanning 7.5 decades whose centred entries reached 5.8 × 10⁻⁴ and
1.8 × 10⁴, which the narrower test did not flag. Hence the third
candidate and the wider test.

**What changed on the validation data.** With the wider test, data set A is
flagged as well (its inputs span about six decades); its MaxDEA comparisons are
unchanged at the precision of Section 2. B could not be re-run for this
revision. The third candidate changes an answer only when it finds an
admissible point with a better objective, so any change on B would be a
correction; whether there is one is not known.

**Verified by putting the bug back.** With the arithmetic mean deliberately
restored, the candidates still recover every score
(`tests/test_scaling.py::test_scores_survive_a_deliberately_bad_scaling`).

### 4.3 Against references without shared tolerances

`docs/external_reference/exact_dea.py` computes the optimum of every unit's
envelopment program on SC in exact rational arithmetic: the doubles in the
file are converted to fractions and each program is solved by a primal simplex
with Bland's rule, so no tolerance enters anywhere. The result is stored as a
double and as a 40-digit decimal (`tests/data/scaling_case_exact.csv`). Against
it, the largest absolute difference over the 30 units:

| | CCR, input | CCR, output | BCC, input | BCC, output |
|---|---|---|---|---|
| PLEAP v1.12 | 4.5 × 10⁻¹¹ | 4.2 × 10⁻¹⁴ | 3.2 × 10⁻⁹ | 2.1 × 10⁻¹⁵ |
| R Benchmarking 0.33 (lp_solve), defaults | 2.1 × 10⁻⁹ | 1.0 × 10⁻¹² | 6.9 × 10⁻⁷ (U24) | 2.5 × 10⁻³ (U24) |

Benchmarking is itself a floating-point program, and on U24 under VRS it stops
short of the optimum.

`docs/scaling_stress.py` repeats the comparison on 100 random draws of the same
design (25 for each row), varied in spread, size and number of inputs:
15,000 unit-model combinations in all. Combinations off the exact optimum by
more than 10⁻⁶, with those that returned no score at all in brackets:

| Draws | arithmetic mean, one solve | geometric mean, one solve | first form of the fix | v1.12 |
|---|---|---|---|---|
| 30 units, 2 inputs, 8 decades | 2 | 1 | 1 | 0 |
| 30 units, 2 inputs, 11.5 decades | 289 (2) | 9 | 9 | 0 |
| 30 units, 2 inputs, 14 decades | 748 (1) | 18 (1) | 19 | 0 |
| 60 units, 3 inputs, 11.5 decades | 640 (2) | 4 | 3 | 0 |
| all 100 draws, 15,000 combinations | 1679 (5) | 32 (1) | 32 | 0 |
| largest error | 0.90 | 0.48 | 0.51 | 3.5 × 10⁻⁸ |

It also sweeps the spread of x2 from 4 to 10 decades, 25 draws (3,000
combinations) at each, to see where a single solve starts to miss and what the
test of the data flags:

| Spread of x2 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|
| draws flagged by `scaling_is_risky` | 0 of 25 | 0 of 25 | 16 of 25 | 24 of 25 | 25 of 25 | 25 of 25 | 25 of 25 |
| one geometric-mean solve: combinations missed | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| v1.12: combinations missed | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### 4.4 What is claimed

That a known conditioning hazard is detected from the data and neutralised,
and that on the public case and the 100 draws the result is the exact optimum
to within 3.5 × 10⁻⁸ — not that no data can defeat the solver. The rule
picks the best admissible point among its candidates; it does not certify
optimality.

**Why it mattered that this was found by a property test, not a comparison.**
The fault reproduced every MaxDEA value on the well-conditioned validation
data. No amount of further reference comparison on those data would have
revealed it. What exposed it was asserting unit invariance — a property of the
estimator that can be checked on any data, including data no reference program
has been run on. What exposed the gap in the fix was a reference that shares no
tolerance with any floating-point solver. `tests/test_scaling.py` checks
invariance on columns spanning twelve decades, and
`tests/test_scaling_reference.py` checks SC against both references in all
four models, and five more draws spanning fourteen decades against the exact
optimum.

---

## 5. Tobit and truncated regression — R, Stata

**Newton polish (v1.12).** BFGS stops at its gradient tolerance, and where the
likelihood is flat the point where it stops depends on the platform: the fit
truncated at 0 and 1 on SD ended 2.2 × 10⁻⁹ from its 60-digit maximum on
Windows and macOS in continuous integration, 9 × 10⁻¹² on Linux. A few Newton
steps on the analytic gradient now follow BFGS. On the public data every
coefficient and log σ is then within 3 × 10⁻¹⁶ of the 60-digit maximum and
the log-likelihood within 3 × 10⁻¹⁴ (SD censored at 0 and 1 and at 0;
SD truncated at 0 and 1; E's Algorithm 1 sample). The tables below were
measured before the polish: their columns against the 60-digit maximum
overstate PLEAP's remaining distance, and its distance from R and Stata is now
the reference programs' own distance from the maximum.

### 5.1 Tobit — R `AER::tobit` / `survreg(dist = "gaussian")`

**On the public data set SD** (400 units, 26 on the frontier;
`tests/data/sd_r_secondstage.json`, `exact_mle.json`), `vrs_i` regressed on
the two environmental variables, censored at 0 and 1 and at 0 only:

| | censored at 0 and 1 | censored at 0 |
|---|---:|---:|
| Coefficients and log σ, \|PLEAP − R\| | 8.7 × 10⁻¹² | 3 × 10⁻¹⁶ |
| Standard errors, relative to R | 6.0 × 10⁻¹¹ | 2.0 × 10⁻¹¹ |
| *p*-values, relative to R | 5.0 × 10⁻⁹ | 1.5 × 10⁻⁹ |
| Wald statistic, relative to R | 3.0 × 10⁻¹⁰ | 4.0 × 10⁻¹¹ |
| Log-likelihood, \|PLEAP − R\| | 9 × 10⁻¹⁴ | 2 × 10⁻¹³ |
| Coefficients, \|PLEAP − 60-digit maximum\| | 5.8 × 10⁻¹¹ | 2 × 10⁻¹⁶ |

**On data set D** (validation data, not distributed).

Dependent variable: `vrs_i`, the BCC input-oriented score of the 453 firms of
data set D as it was exported to R (eight decimals; 16 firms on the frontier at
exactly 1). Covariates: the two environmental variables. Two censoring
specifications, as the author ran them in R's `AER::tobit` through SPSS's R
extension: censored at 0 and 1 (PLEAP's default for an efficiency score) and
at 0 only. `AER::tobit` is `survival::survreg` on interval-coded censoring;
`docs/external_reference/secondstage_ds_d.R` re-runs both with survreg and
reproduces the author's values to 10⁻¹¹. A 60-digit evaluation of the same
likelihood, refined by Newton's method (`docs/external_reference/exact_mle.py`,
run on D's file), settles where the optimum is.

| Quantity | R, censored at 0 and 1 | \|PLEAP − R\| | \|PLEAP − 60-digit\| | R, censored at 0 | \|PLEAP − R\| |
|---|---:|---:|---:|---:|---:|
| Intercept | 0.2585932815 | 4.8 × 10⁻¹⁰ | 4.6 × 10⁻¹⁰ | 0.2573911050 | 1.3 × 10⁻¹¹ |
| *x*₁ | 0.0297273076 | 1.2 × 10⁻¹⁰ | 1.2 × 10⁻¹⁰ | 0.0298067794 | 1.6 × 10⁻¹² |
| *x*₂ | −0.0064180464 | 1.0 × 10⁻¹⁰ | 1.1 × 10⁻¹⁰ | −0.0075501173 | 5.2 × 10⁻¹² |
| `Log(scale)` | −1.3347317604 | 3.7 × 10⁻¹⁰ | 3.8 × 10⁻¹⁰ | −1.3680540352 | 5.3 × 10⁻¹¹ |
| s.e. intercept | 0.0347964266 | 1.3 × 10⁻¹¹ | 1.3 × 10⁻¹¹ | 0.0335906358 | 1.8 × 10⁻¹² |
| s.e. *x*₁ | 0.0218877671 | 8.0 × 10⁻¹² | 8.3 × 10⁻¹² | 0.0211470660 | 1.1 × 10⁻¹² |
| s.e. *x*₂ | 0.0216127411 | 8.0 × 10⁻¹² | 8.2 × 10⁻¹² | 0.0208818278 | 1.1 × 10⁻¹² |
| s.e. `Log(scale)` | 0.0342724017 | 1.5 × 10⁻¹¹ | 1.5 × 10⁻¹¹ | 0.0332227747 | 2.3 × 10⁻¹³ |
| Log-likelihood | −62.3289709685 | < 10⁻¹⁰ | 1 × 10⁻¹³ | −23.0506776212 | < 10⁻¹⁰ |
| Wald statistic (df = 2) | 2.2337103662 | 1.4 × 10⁻⁸ | 1.4 × 10⁻⁸ | 2.3271335754 | 2.3 × 10⁻¹⁰ |

Censored at 0 only, no firm is censored (no score is 0), so the model is the
Gaussian regression and all three agree to 10⁻¹¹ or better. Censored at 1, R
is within 10⁻¹¹ of the 60-digit optimum and PLEAP within 5 × 10⁻¹⁰: BFGS stops
at its gradient tolerance, and the remaining gap is that, not a difference in
the model. `tests/test_tobit_reference.py` asserts 10⁻⁷ relative on every
coefficient and standard error, both specifications. The constant's *p*-value
is about 10⁻¹³; v1.12 computes *p*-values from the upper tail (`sf`) rather
than as 1 − cdf, which had left it 8 × 10⁻⁴ relative away from R's, and now
asserts 10⁻⁶ relative.

**Data set B, up to v1.11.** The Tobit used to be pinned on the 135 plants of
data set B, whose second-stage file is now withdrawn. On it, all twelve
quantities R reports agreed to 7.3 × 10⁻⁹ or better (coefficients, scale and
log-likelihood to 4 × 10⁻¹¹), against R 4.3.3 / survival 3.5-8 and a 60-digit
fit. Two lessons from that data set stand. First, on 12 Nov 2026 the stored
standard errors and Wald statistic turned out not to be R's: they matched
PLEAP's own forward-difference Hessian, and re-running R put them up to
8.1 × 10⁻⁶ (s.e.) and 1.2 × 10⁻⁴ (Wald) away. The Hessian now uses central
differences and the reference values are R's; a reference value is evidence
only if its provenance is. Second, its covariates ran into the millions:

**History.** In v1.0.7 that model did not converge. The unscaled optimiser
stalled and the coefficients collapsed to approximately zero with standard
errors of approximately zero — the output looked like a precise finding of no
effect. The fix was analytic gradients, internal standardisation of the
covariates, and an exact back-transform of the coefficients and the covariance
matrix. `tests/test_tobit_reference.py::test_rescaling_a_covariate_moves_only_its_own_slope`
now multiplies one of SD's covariates by 10⁶ and requires that only its own
coefficient and standard error move, by exactly that factor.

**Supporting checks** that need no external reference, in
`tests/test_secondstage.py`: the analytic gradient agrees with finite
differences of the likelihood; the MLE agrees with the same likelihood maximised
by a derivative-free optimiser on unstandardised data (an independent code
path); with no censored observation the Tobit collapses onto OLS to 10⁻⁵; and on
simulated data with known coefficients the estimator recovers them.

### 5.2 Truncated regression

**What the GUI fits.** Truncation at the frontier only, following Simar and
Wilson [6]: units within 10⁻⁹ of 1 are on it and stay out of the fit. Before
12 Nov 2026 the filter was a strict `theta < 1`, which let frontier units
returned as 1 − 4 × 10⁻¹⁵ into the sample (54 instead of 49 on the example
used at the time). Truncation at 0 as well is available from the library but not
offered in the GUI: on scores that crowd towards 0 the doubly truncated
likelihood can lose its maximum — σ diverges, which is what v1.0.7 produced.
It is not ill-posed in general; on data set D it is well behaved, and Stata
fits it (below).

**On the public data.** R 4.3.3, truncreg 0.2-5, with its Newton–Raphson
method (`method = "nr"`; its default BFGS stops early, at a gradient of
3.5 × 10⁻³ on SD), and the 60-digit maximum of the same likelihood:

| Fit | *n* | Reference | Coefficients and σ | Standard errors (relative) | Log-likelihood |
|---|---:|---|---:|---:|---:|
| E, last period, `truncreg(theta ~ z1 + z2, point = 1, direction = "right")` — Algorithm 1's sample | 48 | R | 1.4 × 10⁻⁹ | 1.9 × 10⁻⁸ | 7 × 10⁻¹⁵ |
| same | 48 | 60-digit | 6.9 × 10⁻¹² | 4.5 × 10⁻¹⁰ | < 10⁻¹⁴ |
| SD, truncated at 0 (`point = 0, direction = "left"`) | 400 | R | 1.7 × 10⁻¹³ | 6.2 × 10⁻¹¹ | 2 × 10⁻¹⁴ |
| SD, truncated at 1, units strictly inside | 374 | R | 2.5 × 10⁻¹⁰ | 2.7 × 10⁻¹⁰ | 7 × 10⁻¹⁴ |
| SD, truncated at 0 and 1 — R's truncreg takes one point | 374 | 60-digit | 8.6 × 10⁻¹² | 1.8 × 10⁻¹⁰ | 4 × 10⁻¹⁴ |

On the example R's Newton–Raphson stops at a gradient of 8.8 × 10⁻⁷, which
is the size of its remaining distance from PLEAP; the 60-digit fit agrees
with PLEAP to 7 × 10⁻¹². Before the example was replaced, the same check on
the previous example gave 3 × 10⁻⁷ against R's default fit and 6 × 10⁻¹⁰
against the 60-digit one.

**Against Stata and R on data set D.** The author fitted `truncreg vrs_i x1 x2`
in Stata truncated at 0 and 1 (`ll(0) ul(1)`, 437 firms) and at 0 only
(`ll(0)`, 453 firms); R's `truncreg(..., point = 0, direction = "left")` fits
the second (R's `truncreg` takes one truncation point).
`tests/test_secondstage_reference.py`:

| | ll(0) ul(1): \|PLEAP − Stata\| | ll(0): \|PLEAP − Stata\| | ll(0): \|PLEAP − R\| |
|---|---:|---:|---:|
| Coefficients | ≤ 2.5 × 10⁻⁶ | ≤ 9.0 × 10⁻⁷ | ≤ 1.8 × 10⁻⁶ |
| σ | 6.4 × 10⁻⁷ | 1.7 × 10⁻⁷ | 8.6 × 10⁻⁷ |
| Standard errors (σ's by the delta method) | ≤ 4.8 × 10⁻⁶ | ≤ 8.6 × 10⁻⁷ | ≤ 1.6 × 10⁻⁶ |
| Log-likelihood | < 10⁻⁵ (151.63698) | < 10⁻⁵ (105.52705) | 2 × 10⁻¹⁰ |
| Gradient norm at the reported estimate | Stata 1.2 × 10⁻⁴, PLEAP 3.7 × 10⁻⁷ | Stata 9.4 × 10⁻⁵, PLEAP 2.1 × 10⁻⁷ | R 3.1 × 10⁻⁴ |

Stata prints seven significant digits and stops at its own tolerance; R stops
at maxLik's. At PLEAP's estimate the likelihood is at least as high and the
gradient two to three orders of magnitude smaller, so the gaps are the
reference programs' stopping points. Standard errors move with the point they
are evaluated at: recomputed at R's estimate with an accurate Hessian, PLEAP's
formula gives R's to 3 × 10⁻⁷ relative.

**Fault found by this comparison (fixed in v1.12).** Truncated at 0 only,
v1.11's BFGS stopped after two iterations with a line-search failure, at
log-likelihood 88.79, and returned `converged=False` with coefficients far from
Stata's. Two causes: the truncation probability Φ(b) − Φ(a) was computed as a
difference of two cdf values near 1, which cancels when the window lies in the
upper tail; and nothing was tried after BFGS failed. The probability is now
computed from upper-tail functions there, and a fit that BFGS has not
converged is retried with a trust-region Newton step on the same likelihood,
keeping the better end point. Fits that BFGS already completes are unchanged.

---

## 6. Group tests — SPSS and R

Where statistical packages differ, PLEAP follows SPSS, because that is what its
users will be checking against. Each convention is pinned by a test whose
docstring records the reason.

| Test | Convention | Note |
|---|---|---|
| Levene | centred on the **mean** | SciPy's default is the median; SPSS uses the mean. |
| Independent-samples *t* | Student when Levene *p* ≥ 0.05, else Welch | Both rows are always shown, with the reported one marked. Before v1.0.8 Welch was reported regardless of Levene. |
| Mann–Whitney *U* | tie-corrected normal approximation, **no** continuity correction | SPSS's "Asymp. Sig. (2-tailed)". SciPy's default applies a continuity correction, which moved *p* from 0.0010516 to 0.0010599 on the test fixture. |
| Kruskal–Wallis | tie-corrected *H* | Matches SciPy exactly. |
| Dunn | Bonferroni over all pairs | Adjusted *p* = min(*p*·*m*, 1). |
| ANOVA | one-way when Levene *p* ≥ 0.05, else Welch ANOVA | Post hoc: Tukey HSD under equal variances, Games–Howell otherwise. |
| Ties in rank tests | scores that agree to 10⁻⁹ (relative) are one value | New in v1.12; see below. |

**Against R, on the public data set SD** (`tests/test_secondstage_r_reference.py`).
`vrs_i` by `group1` (2 groups) and `group2` (4 groups); 26 units are on the
frontier, so the rank tests are tie-corrected throughout:

| Test | R | Largest relative difference |
|---|---|---|
| Mann–Whitney | `wilcox.test(exact = FALSE, correct = FALSE)` | *U* equal to R's *W*; *p* 1.8 × 10⁻¹⁵ |
| Kruskal–Wallis, Dunn (Bonferroni) | `kruskal.test`, `rstatix::dunn_test` | 1.6 × 10⁻¹⁵; 4.0 × 10⁻¹⁵ |
| Student and Welch *t*-tests, Levene (mean) | `t.test`, `car::leveneTest(center = mean)` | 1.1 × 10⁻¹⁴; 9.5 × 10⁻¹⁵ |
| One-way and Welch ANOVA | `oneway.test` | 1.0 × 10⁻¹⁴; 1.4 × 10⁻¹⁴ |
| Tukey HSD | `TukeyHSD` | 8 × 10⁻¹⁰ (2.2 × 10⁻⁵ at *p* = 8.4 × 10⁻⁷) |
| Games–Howell | Welch's df and `ptukey` | *p* 7.6 × 10⁻⁹; s.e. 1.5 × 10⁻¹⁵ |
| OLS | `lm` | 4.9 × 10⁻¹⁴ |

SD's own Levene test does not reject equal variances across the four groups,
so PLEAP reports the one-way ANOVA there; the Welch branch is checked by
forcing the Levene *p*-value. `rstatix::games_howell_test` rounds its
*p*-values to three decimals, so the reference computes them from `ptukey`
directly, as rstatix does. At *p* = 8.4 × 10⁻⁷ R's `ptukey` is the less
precise of the two programs: a 30-digit integration of the studentized range
puts SciPy's value 1.4 × 10⁻⁷ from it and R's 2.2 × 10⁻⁵. PLEAP's own unrounded DEA scores give the same
rank-test results as the eight-decimal column.

**Against SPSS.** The author's SPSS output for data sets A and D, with each test
fed the vector SPSS was given (`tests/test_secondstage_reference.py`, run where
the validation data are available):

| Data | Tests | Reference precision | Largest relative difference |
|---|---|---|---|
| D, 453 firms | Mann–Whitney (89 vs 364); Kruskal–Wallis and Dunn (4 groups) | full; *H* to 3 decimals | *U* exact; *z* 2 × 10⁻¹⁴; *H* equal to SPSS's digits; Dunn *z* 10⁻¹⁴ |
| D207, 207 firms | OLS; *t*-test with Levene; one-way ANOVA; Tukey HSD; Games–Howell | full | OLS 1.1 × 10⁻⁸; *t*-test 1.6 × 10⁻⁹; ANOVA 1.9 × 10⁻⁹; Tukey *p* 1.1 × 10⁻⁷; Games–Howell s.e. 5 × 10⁻¹¹, *p* 5.4 × 10⁻⁷ |
| A, 60 firms | Mann–Whitney, Kruskal–Wallis, Dunn; Tobit; OLS; *t*-test; Welch ANOVA; Games–Howell; Tukey HSD | 3 decimals | every printed digit |

The Tobit on data set A was R's `AER::tobit` run from SPSS, as on data set D.
SPSS's *p*-values are 1 − cdf, which loses relative precision as *p* falls:
5.7 × 10⁻⁸ at *p* = 1.2 × 10⁻⁹ and 1.1 × 10⁻² at *p* = 1.5 × 10⁻¹⁴, measured
against the exact normal tail. PLEAP's are the exact tail (10⁻¹⁴), so the
largest relative "difference" in Dunn's *p* is SPSS's. On the 207-firm subset
the author's notes say Levene rejected equal variances for the four-group
ANOVA; Levene centred on the mean gives *F* = 1.900, *p* = 0.131 on that
vector (centred on the median, 1.137, *p* = 0.335), so PLEAP reports the
one-way ANOVA with Tukey HSD. Both post hoc tables match SPSS's either way.

**Fault found by this comparison: ties among frontier scores (fixed in v1.12).**
The linear programs return a frontier unit's score as 1 only up to rounding;
1 − 4 × 10⁻¹⁵, 1.0 and 1 + 2 × 10⁻¹⁶ all occur. Ranked as they stand, they are
distinct values, the tie correction disappears and the mid-ranks move. SPSS,
reading the exported scores, sees ties. On PLEAP's own scores v1.11 gave
Kruskal–Wallis *H* = 6.381 on data set A against SPSS's 6.356, and on data set
D a Mann–Whitney *U* 4.5 away and an *H* 0.196 away. Values within 10⁻⁹
(relative) of each other are now one value, and PLEAP's own scores reproduce
SPSS on both data sets.

**Fault found by this comparison: Tukey *p*-values (fixed in v1.12).** They were
read off statsmodels' summary table, which rounds to four decimals, so
*p* = 3.9 × 10⁻⁶ came out as 0.0000. They now come from the result arrays.

Tests: `tests/test_secondstage.py`, `tests/test_secondstage_r_reference.py`,
`tests/test_secondstage_reference.py`.

---

## 7. Bootstrap — MaxDEA

The bootstrap is stochastic, so it cannot be reproduced bit for bit: two PLEAP
runs differing only in the seed already differ from each other. The honest
comparison is therefore against the procedure's own sampling variability — and
at the reference program's own number of replications.

### 7.1 At MaxDEA's own B (v1.12)

MaxDEA's reference runs used B = 500 on data set A (the workbook sheet is
"Bootstrap(B 500, Core 18)") and B = 1000 on data set D. PLEAP was run at the
same B, five seeds on A and two on D. If the two procedures were statistically
identical, |PLEAP − MaxDEA| and |PLEAP seed *s* − PLEAP seed *t*| would both be
differences of two independent estimates with the same Monte-Carlo variance,
and their ratio would sit near 1.

Bias-corrected mean, mean over DMUs off the frontier (A: 56–57 of 60; D:
437–448 of 453):

| Data, model | B | \|PLEAP − MaxDEA\| | \|seed − seed\| | ratio | mean signed gap | ratio on frontier DMUs |
|---|---:|---:|---:|---:|---:|---:|
| A, BCC input | 500 | 0.00124 | 0.00116 | 1.08 | +0.00045 | 1.29 (4 DMUs) |
| A, BCC output | 500 | 0.00059 | 0.00054 | 1.09 | −0.00018 | 1.68 (4) |
| A, CCR | 500 | 0.00060 | 0.00062 | 0.96 | −0.00006 | 1.30 (3) |
| D, BCC input | 1000 | 0.00180 | 0.00043 | 4.18 | +0.00173 | 1.73 (16) |
| D, BCC output | 1000 | 0.00080 | 0.00024 | 3.35 | +0.00079 | 2.29 (14) |
| D, CCR | 1000 | 0.00052 | 0.00028 | 1.86 | +0.00052 | 1.57 (5) |

(Under CRS the two orientations give identical bootstrap results for the same
seed, to 6 × 10⁻¹⁴ in PLEAP on D, and identical results in MaxDEA too, so CCR
is one row.)

**Data set A: indistinguishable inside the frontier.** The gap to MaxDEA is the
size of the gap between two PLEAP seeds, and its sign is mixed. On the three or
four frontier units it is 1.3–1.7 times the seed-to-seed gap.

**Data set D: a small, systematic difference.** PLEAP's bias-corrected mean is
above MaxDEA's for nearly every unit, by a fixed fraction of the score: 0.6%
under BCC input orientation, 1.1% under BCC output orientation and 1.0% under
CCR, in every quintile of θ̂ (0.54–0.67% across the BCC-input quintiles). In
the homogeneous bootstrap every unit is re-evaluated against pseudo-frontiers
drawn from one smoothed distribution, so a constant relative offset points at
the scale of that smoothing: PLEAP's estimated bias is a little smaller, as it
is under a smaller bandwidth. A probe agrees: with the bandwidth multiplied by
1.15, PLEAP's offset on D (BCC input, B = 200) falls from 0.73% to 0.07% of the
score. But the same widening would move data set A by about −1.2 × 10⁻³ (the
×1.2 run of 7.2 moved it by −1.7 × 10⁻³), where the two programs now agree, so
MaxDEA is not simply using a wider rule: whatever differs depends on the data,
and its output does not show what. Silverman's robust rule,
0.9 · min(sd, IQR/1.349) · n^(−1/5), is not it either: it would give bandwidths
0.27–0.88 times PLEAP's on these data, the wrong direction. In absolute terms
the offset is 0.0017 on scores averaging 0.30.

**Standard deviations** (data set A only; the D workbook has no SD column).
PLEAP's average 0.94 of MaxDEA's (0.93–0.96 across models), and per unit the
gap is about three times the seed-to-seed gap under BCC: another small,
systematic difference.

**What this changes in the record below.** Up to v1.11 this section compared
PLEAP at B = 2000 with MaxDEA's B = 500 run and reported ratios of 1.9–3.1.
The mismatch alone inflates that ratio by
√((1/2000 + 1/500)/(2/2000)) = 1.58, so on data set A most of what looked like
a residual difference was MaxDEA's own Monte-Carlo noise. The subsections that
follow were measured that way; their comparisons between variants stand, and
their absolute gaps include the mismatch.

### 7.2 Earlier measurements (PLEAP at B = 2000, MaxDEA at B = 500)

Bias-corrected mean, data set A, 60 DMUs:

| Model | mean \|PLEAP − MaxDEA\| | median | max | mean \|seed 1 − seed 2\| | ratio |
|---|---|---|---|---|---|
| BCC, input | 0.00191 | 0.00060 | 0.03332 | 0.00100 | 1.9× |
| BCC, output | 0.00160 | 0.00022 | 0.04167 | 0.00051 | 3.1× |
| CCR, input | 0.00103 | 0.00018 | 0.02633 | 0.00048 | 2.1× |
| CCR, output | 0.00103 | 0.00018 | 0.02633 | 0.00048 | 2.1× |

(The two CCR rows are identical to every digit, as they must be: under CRS the
orientations coincide. That is a useful internal check on the table itself.)

**The ratio is not 1, and that matters.** If MaxDEA's procedure were
statistically identical to PLEAP's, the PLEAP–MaxDEA difference and the
seed-to-seed difference would have the same distribution, so the ratio would
sit near 1. At 1.9–3.1× there is a small residual difference in convention on
top of the Monte-Carlo noise. Saying "within Monte-Carlo error" without
qualification would overstate the agreement. *(Superseded for data set A by
7.1: at MaxDEA's own B the ratio is 0.96–1.09, and most of the 1.9–3.1 was the
mismatch in B. On data set D a residual does show, at matched B.)*

**Where the residual lives.** Almost entirely in the four fully efficient DMUs.
Signed gap by efficiency band (BCC input-oriented, B = 2000):

| θ̂ band | n | mean signed gap | mean \|gap\| | max \|gap\| |
|---|---|---|---|---|
| [0.00, 0.20) | 39 | +0.00058 | 0.00073 | 0.00620 |
| [0.20, 0.40) | 11 | +0.00073 | 0.00149 | 0.00388 |
| [0.40, 0.60) | 3 | +0.00312 | 0.00312 | 0.00724 |
| [0.60, 0.80) | 2 | −0.00088 | 0.00383 | 0.00470 |
| [0.80, 1.00) | 1 | −0.00211 | 0.00211 | 0.00211 |
| **= 1.000** | **4** | **+0.01078** | **0.01273** | **0.03332** |

For the 50 DMUs below 0.6 the mean absolute gap is 0.0007–0.0031, which *is*
the order of the Monte-Carlo noise. The overall figure of 0.0019 is produced by
four observations out of sixty.

**Why efficient DMUs are different.** For a DMU on the estimated frontier the
pseudo-frontier can fall well inside the observation, so its re-evaluated score
may be far above 1 — correctly, since the observation is not a member of the
pseudo reference set and must not be clipped. A handful of replications
therefore return very large super-efficiency values and dominate the raw
`Bias`. The column is heavy-tailed, and the two programs disagree on it
substantially: on the four frontier units PLEAP's `Bias` ranges from 0.47 to
25.2 and MaxDEA's from 0.47 to 10.2, while their corrected `Mean` values
differ by 0.004 to 0.033. (Unit-level values are not reported, because data
set A may not be passed on.)

A "bias" of 25 on a score bounded by 1 is not a meaningful quantity in either
program; it is an artefact of the same heavy tail. The **bias-corrected mean**
is shielded from it by the KSW ratio correction and its clip to [0, 1], which is
why the disagreement there is 0.013 rather than 15. Across all 60 DMUs the two
programs' `Bias` columns correlate at 0.934 but differ in level by a factor of
1.46, while the reported SDs agree to within 5% (PLEAP 0.0386, MaxDEA 0.0408,
ratio 0.946).

**Practical reading.** Report the bias-corrected score, its SD and its
confidence interval — those agree closely. Do not report the raw `Bias` for a
DMU whose uncorrected score is 1, in PLEAP or in MaxDEA; it is not stable
enough to interpret, and the two programs will not agree on it.

Tests (all marked `slow`, all at MaxDEA's B = 500 since v1.12):
`test_bias_corrected_mean_tracks_maxdea_on_inefficient_dmus` runs three seeds
and requires the gap on non-frontier DMUs to be below 0.003 *and* below twice
the seed-to-seed gap; `test_the_gap_to_maxdea_concentrates_on_efficient_dmus`
bounds the frontier-DMU gap at 0.05 so a regression there is caught rather than
excused; `test_replicate_spread_matches_maxdea` holds the mean SD ratio inside
±15%. Stating the bounds separately keeps each one falsifiable instead of
hiding a known disagreement inside a loose average.

**Which bias-corrected point estimator, and why.** The bias correction has more
than one form in circulation, and they are not close to each other. PLEAP
reports the mean over replicates of the ratio θ̂²/θ̂*_b_, clipped to [0, 1].
Measured against MaxDEA's `Mean` column on data set A (BCC input, B = 2000,
seed 12345):

**What PLEAP computes, in both scales.** Writing δ = 1/θ for the Shephard
distance, the reported score mean_b_(θ̂²/θ̂\*_b_) satisfies

    1 / θ̃  =  δ̂² / mean(δ̂*)

exactly — verified to 2.2 × 10⁻¹⁶ in
`test_the_bias_corrected_estimator_is_the_mean_of_ratios`. So PLEAP applies the
**ratio (multiplicative) correction on the Shephard scale**, which agrees to
first order with the additive δ̃ = 2δ̂ − mean(δ̂\*). Naming it a "Farrell-scale"
correction, as an earlier draft did, was wrong.

| Point estimator | mean \|gap\| | max \|gap\| |
|---|---:|---:|
| Additive on the **Farrell** scale, 2θ̂ − mean(θ̂\*) | 0.819339 | 37.508550 |
| Additive on the **Shephard** scale, 1/(2δ̂ − mean(δ̂\*)) | 0.029897 | 0.336018 |
| Ratio of means, θ̂² / mean(θ̂\*) | 0.015906 | 0.204490 |
| **Ratio on the Shephard scale — PLEAP** | **0.001505** | **0.024307** |

The first row is **not** the fair comparison and should not be quoted as one.
Nobody applies the additive correction on the Farrell scale; it blows up
because a frontier DMU's raw bias is heavy-tailed there, and an earlier draft
of the manuscript used its 37.5 as evidence against "the additive correction",
which was a straw man. The honest comparison is the second row: the additive
correction done where it is normally done, on the Shephard scale. It behaves
perfectly sensibly (no undefined values on these data) and is still 20× further
from the reference than the ratio form.

What this does *not* establish is which equation of \[3\] or \[4\] defines the
estimator. It is reported as the convention PLEAP applies and the one that
matches the reference. Reference \[4\] is cited for consistency results, not
for this formula.

**The clip is a rail, not a correction.** On the same run, *no* replicate ratio
exceeds 1: zero of 60 DMUs have any clipped replicate, and the clipped and
unclipped point estimates agree to every digit printed above. The clip cannot
be what protects the corrected mean from the heavy tail — the ratio form is.
A large θ̂\* drives θ̂²/θ̂\* toward zero, not toward infinity. The clip only
guarantees the output stays inside the estimand's range on data where a
replicate does come back below θ̂².

**Independent re-measurement (5 seeds, B = 2000).** `docs/seed_study.py`
re-derives both sides of the comparison from scratch on data set A, BCC
input-oriented, averaging the absolute gap over all ten seed pairs:

| | seed-to-seed | vs MaxDEA | ratio |
|---|---:|---:|---:|
| Non-frontier (56 DMUs) | 0.000525 | 0.001081 | 2.06 |
| Frontier (4 DMUs) | 0.004174 | 0.009564 | 2.29 |
| All 60 | 0.000768 | 0.001646 | 2.14 |

This is the number to quote, and it agrees with the 1.9–3.1× above. The gap is
the same *order* as the seed-to-seed variability and roughly twice it, in both
regimes — so the two procedures are close but not statistically identical, and
"within Monte-Carlo error" remains the wrong description. Run it with:

```bash
python docs/seed_study.py        # ~10 minutes; writes docs/seed_study.json
```

**Bandwidth.** *h* = 0.9 · sd(θ̂) · (2*n*)<sup>−1/5</sup> — the Silverman rule
with the reflected sample size, pinned exactly in
`test_bandwidth_is_the_reflected_silverman_rule`. On data set A this gives
*h* = 0.093947 for sd(θ̂) = 0.271942, *n* = 60. v1.0.7 used
1.06 · min(sd, IQR/1.349) · *n*<sup>−1/5</sup>, which left a systematic offset
of about −0.008 against MaxDEA; that offset was confirmed to be systematic
rather than Monte-Carlo noise by reproducing it across three seeds, and the
recalibration was verified on three data sets with highly efficient, mixed and
highly inefficient score distributions. The residual 2–3× ratio above suggests
MaxDEA's bandwidth or draw convention still differs slightly from the published
Simar–Wilson rule; identifying it exactly would need MaxDEA's source, which is
not available.

**Where the residual gap is not.** Three of PLEAP's documented choices could
contribute to it: the bandwidth rule, the reflection at 0, and the ratio form of
the correction. `docs/bootstrap_decomposition.py` and
`docs/bootstrap_correction_forms.py` change each in turn on data set A, holding
everything else fixed, and measure the mean absolute gap to MaxDEA on the 56
non-frontier DMUs (B = 2000):

| Variant | Seeds | Gap to MaxDEA, non-frontier | Frontier |
|---|---|---:|---:|
| **PLEAP as shipped** | 1–5 | **1.08 × 10⁻³** | **9.6 × 10⁻³** |
| Draws below 0 redrawn instead of reflected | 1–3 | 4.8 × 10⁻³ | 3.3 × 10⁻² |
| Smoothing on the Shephard scale, as in Simar and Wilson (1998) | 1–3 | 5.6 × 10⁻² | 1.9 × 10⁻¹ |
| Bandwidth × 0.8 | 1–2 | 2.6 × 10⁻³ | 1.1 × 10⁻² |
| Bandwidth × 1.2 | 1–2 | 2.2 × 10⁻³ | 1.4 × 10⁻² |
| Additive correction, Shephard scale (same replicates) | 1 | 1.5 × 10⁻² | 2.4 × 10⁻¹ |
| Ratio of means (same replicates) | 1 | 9.2 × 10⁻³ | 1.0 × 10⁻¹ |

The reflection at 0 is not a technicality on this data: half the scores are below
0.1 and the bandwidth is 0.094, so 13.5% of draws fall below 0 and are
reflected. Every alternative moves PLEAP *away* from MaxDEA, so each of the
three choices is closer to MaxDEA's procedure than its alternative, and the
residual lies in a detail of the closed program — or in a bandwidth difference
smaller than the ±20% tried: the mean signed gap is +4.6 × 10⁻⁴ at the shipped
bandwidth, +2.6 × 10⁻³ at × 0.8 and −1.7 × 10⁻³ at × 1.2, so it changes sign
a few per cent above the shipped value.

**An open-source second reference: Benchmarking.** R's
`Benchmarking::dea.boot` (version 0.33) is open, so unlike MaxDEA it can be
read. It smooths on the Shephard scale δ = 1/θ, sampling from the reflected set
{δ, 2 − δ} with a robust Silverman bandwidth (7.165 on data set A, where δ
reaches 106), and reports the additive correction on that scale. As each program
reports by default, the two bias-corrected scores differ by **1.2 × 10⁻²** on
non-frontier DMUs (seeds 1–2 of PLEAP against seeds 1–5 of Benchmarking) — a
difference of convention, as the table above shows for the Shephard-scale
smoothing.

To test the rest of PLEAP's machinery against an open implementation — the
pseudo-data, the evaluation of each unit against each pseudo-frontier without
joining it, and the correction computed from the replicates —
`docs/bootstrap_benchmarking_aligned.py` swaps Benchmarking's draw and
bandwidth into PLEAP and compares with Benchmarking's replicates under the same
estimator (`docs/external_reference/bootstrap_benchmarking.R`):

| | PLEAP with Benchmarking's smoothing vs Benchmarking | seed-to-seed, PLEAP | seed-to-seed, Benchmarking |
|---|---:|---:|---:|
| Non-frontier (56 DMUs) | **4.6 × 10⁻⁴** | 5.0 × 10⁻⁴ | 6.7 × 10⁻⁴ |
| Frontier (4 DMUs) | 1.6 × 10⁻³ | 3.1 × 10⁻³ | 3.7 × 10⁻³ |

The gap is below either program's own seed-to-seed variability: with the
smoothing convention aligned, the two implementations cannot be told apart at
B = 2000.

The same comparison runs on the public data set SA
(`docs/external_reference/reference_sa.R`, Benchmarking's replicates for
seeds 1–5 at NREP = 2000, and the bandwidth `dea.boot` itself prints, which
PLEAP's replica of its rule reproduces to 3 × 10⁻¹³). One PLEAP run at
B = 1000 with Benchmarking's smoothing is 1.2 × 10⁻³ from Benchmarking's
five-seed mean off the frontier and 2.2 × 10⁻³ on it, where two PLEAP seeds at
B = 1000 are 1.6 × 10⁻³ apart and two Benchmarking seeds at NREP = 2000
1.0 × 10⁻³. As each program reports by default the gap is 1.3 × 10⁻².
`tests/test_bootstrap_open_reference.py` asserts both cases (SA on every run,
A where the data are available).

**Structural guarantees**, each asserted on every run: the estimated bias is
non-negative (v1.0.4 clipped the pseudo scores at 1 and produced negative biases
for efficient DMUs, which is impossible for the radial estimator); bias
correction never raises a score; the confidence interval brackets the
bias-corrected score; `theta_hat` equals the plain DEA score (Benchmarking's,
on SA) to 10⁻⁹ for all four models; results are identical for a given seed and **independent of the
number of worker processes**, so the Cores control cannot change a published
number.

---

## 8. The Malmquist substitution, and what it costs

Under VRS a cross-period distance can be infeasible. PLEAP's default follows
MaxDEA and substitutes 1. This section measures what that convention does,
because "we disclosed it" is not the same as "it did not matter".

Four distance functions are computed per unit-period row, but only two of them
are cross-period; the own-period pair is always feasible. On a panel of 60
units and three periods the denominator is therefore **240**, not 480.
Re-running with `on_infeasible="omit"`, which drops the affected rows instead,
gives the geometric-mean index of each transition (input orientation):

| Panel | Infeasible distances (rows) | Transition | substituted | omitted | CRS index |
|---|---:|---|---:|---:|---:|
| E, the example of the paper | 5 (4) | 2021→2022 | 1.0013 | 1.0009 | 0.9983 |
| | | 2022→2023 | 1.0851 | 1.0868 | 1.0878 |
| A, validation (KIS) | 3 (3) | 2020→2022 | **1.0439** | **1.0977** | 1.0452 |
| | | 2022→2024 | 1.2149 | 1.1794 | 1.1655 |
| SA, synthetic | 3 (3) | 2020→2022 | 1.0186 | 1.0182 | 1.0263 |
| | | 2022→2024 | 1.1251 | 1.1275 | 1.1315 |

On the example and on SA the convention moves the averages in the third
decimal. On A, three rows out of sixty carry it: substituting 1 lowers the
first transition's mean productivity growth from 9.8% to 4.4% and raises the
second's from 17.9% to 21.5%. An infeasible program means that no radial
change of the unit's inputs puts it inside the other period's technology, so
no finite distance exists; replacing it by 1 leaves the affected index to the
other three distances, which can move it either way.

On the example used up to v1.11, which was drawn from real data, eight
substituted distances reversed the sign of the first transition (1.0146
substituted, 0.9963 omitted), and the paper used it to make this point.

This is the reason the count is always reported and the alternative always
available. A convention that can halve a reported growth rate, or reverse it,
is not a detail. Under CRS no cross-period program is infeasible, which is
one reason the paper's example reports the CRS index with the VRS split.

---

## 9. Can a post-hoc check catch the scaling fault?

Section 4 claims a feasibility audit cannot detect the scaling fault. That
claim is about **primal** feasibility and nothing wider; an earlier draft of
the manuscript overstated it as "no post-hoc test of a single solution", which
is false — linear programming supplies optimality certificates.

Measured on SC (`tests/data/scaling_case.csv`; 30 units, one input column
spanning 11.3 decades), for U03, the first unit whose score moves between the
two scalers:

| Check, in the original units | Result |
|---|---|
| Score under arithmetic-mean scaling | 0.697344 |
| Score under geometric-mean scaling | 0.691022 |
| Primal feasibility of the faulty solution | max violation 0.0 — **passes, catches nothing** |
| Dual feasibility (reduced costs, dual mapped back) | min reduced cost −1.57 × 10⁻² — **violated, detected** |

So a dual test does work. The two-solve rule is preferred for two reasons, and
neither is that the alternative is impossible:

1. It repairs. A dual check tells the user a number is wrong and stops there;
   the cross-solve returns the right number, and recovers every MaxDEA value
   even with the old scaling forced back in.
2. It needs no decision threshold of its own. A reduced-cost test does, and
   mis-set tolerances are what caused the fault in the first place. (The third
   candidate of Section 4.2 runs the solver at tighter tolerances, but the
   choice among candidates is still made by the objective.)

`tests/test_scaling.py::test_a_dual_check_does_detect_the_scaling_fault` pins
this, so the narrower claim cannot quietly re-broaden.

---

## 10. Simar–Wilson Algorithm 1 — point estimates against R, bootstrap by simulation

Algorithm 1 changes the inference, not the estimate: its coefficients are the
truncated-regression maximum-likelihood ones, and those now have an external
reference — R's `truncreg`, Section 5, on the 48 units the example's
Algorithm 1 fit uses, and the same likelihood maximised in 60-digit
arithmetic. The **bootstrap**, the part that changes the inference,
still has none. Stata's `simarwilson` implements the full Algorithm 1 and is
the natural reference, but was not available; `rDEA::dea.env.robust`
implements Algorithm 2 only. The bootstrap is therefore checked against a known
truth, and the record says so rather than letting the surrounding table imply
otherwise.

Data are drawn from the model Algorithm 1 assumes, with the dependent variable
on the **Farrell** scale and therefore truncated from above at 1 (Simar and
Wilson write it in Shephard units, truncated from below; the two are the same
model) — β = (0.55, 0.20, −0.10), σ = 0.12, *z* ~ U(0, 1)² — and the procedure
is asked to recover it. `docs/sw_coverage.py`, 200 Monte-Carlo replications per
sample size, L = 1000 bootstrap draws, nominal 95%; the Monte-Carlo standard
error on each coverage figure is sqrt(0.95 × 0.05 / 200) = **1.5 percentage
points**:

| Coefficient | *n* = 100 | *n* = 200 | *n* = 400 |
|---|---:|---:|---:|
| const | 93.0% (92.5%) | 96.5% (97.0%) | 94.5% (94.0%) |
| *z*₁ | 98.0% (97.5%) | 95.0% (95.5%) | 95.0% (95.0%) |
| *z*₂ | 92.0% (93.0%) | 96.5% (97.0%) | 94.0% (94.0%) |
| Mean interval width, *z*₁ | 0.164 | 0.115 | 0.082 |

Algorithm 1 bootstrap coverage, with the naive truncated-regression interval
(estimate ± 1.96 s.e.) in parentheses; no replication failed. All nine cells
are within two Monte-Carlo standard errors of nominal; the largest departures,
3.0 points, are both at *n* = 100. The interval width falls by √2 for each
doubling of *n*. The earlier record (n = 200, L = 120) gave 91.0–94.5%; the
naive intervals shifted by about as much between the two runs, so that
difference is Monte-Carlo variation, not the number of bootstrap draws.

**Read this carefully.** It shows the implementation is sound: coverage sits
near nominal and the point estimates recover β. It does **not** show that
Algorithm 1 beats the naive interval, and on this design it cannot. The
responses here are drawn independently from the model, so there is no
estimated frontier and therefore none of the dependence Algorithm 1 exists to
handle; the naive intervals are already valid and the bootstrap only adds
Monte-Carlo noise. Demonstrating the advantage would mean re-running Simar and
Wilson's own experiment, which is their result, not PLEAP's.

Five properties are asserted in `tests/test_secondstage.py` and
`tests/test_truncreg_reference.py`: the coefficients are exactly the
truncated-regression maximum-likelihood ones and match R's `truncreg`; the
bootstrap is reproducible from its seed and differs across seeds; the truth is
recovered within four bootstrap standard errors; units within 10⁻⁹ of the
frontier stay out of the fit; and the procedure refuses a sample with no
inefficient units.

**Before this is relied on in print**, cross-check one fit against Stata's
`simarwilson` if a licence is available. The point estimates are pinned to an
outside program; the bootstrap is not.

---

## 11. Does the scaling fault affect other DEA software?

Section 4's fault was found in PLEAP. The obvious question is whether it is a
PLEAP defect or a hazard any implementation can hit. On SC, CCR input
orientation, against the exact optimum of Section 4.3:

| | max \|gap\| | mean \|gap\| | units off by more than 10⁻⁴ |
|---|---:|---:|---:|
| Solved with no scaling | 0.503540 | 0.016785 | 1 of 30 |
| Divided by the **arithmetic** mean, one solve | 0.152951 | 0.006952 | 9 of 30 |
| Divided by the **geometric** mean, one solve | 5.0 × 10⁻¹¹ | 7.4 × 10⁻¹² | 0 of 30 |
| **PLEAP v1.12** (geometric mean, three candidates) | 4.5 × 10⁻¹¹ | 4.8 × 10⁻¹² | 0 of 30 |
| **R Benchmarking 0.33** (lp_solve), defaults | 2.1 × 10⁻⁹ | 7.5 × 10⁻¹¹ | 0 of 30 |
| **pyDEA 1.6** (PuLP 2.7, CBC) | 4.5 × 10⁻⁹ | 1.7 × 10⁻⁹ | 0 of 30 |

The hazard is real and reproducible: arithmetic-mean scaling gets 9 of 30 units
wrong on synthetic data anyone can rebuild, and several of those
errors are *understatements* — U18's optimum is 1.000000 and the mis-scaled
solve returns 0.847049, which is not a feasible point of the problem posed.
Solving the raw data is no safer: it misses U20 by half.

The conditioning choice was PLEAP's own: pyDEA, with a different formulation
and a different solver (CBC), and Benchmarking, with lp_solve, both get CCR
input orientation right. Under VRS Benchmarking misses U24 (Section 4.3), so
no floating-point program here is a reference beyond doubt, which is why the
exact one exists. `docs/crosstool_check.py` drives the floating-point solvers;
pyDEA needs PuLP before 4.0, in an environment of its own
(`docs/crosstool_pydea.py`). deaR and rDEA were not run on this case.

---

## 12. Reproducing this record

On the public data, from a clone of the repository:

```bash
pip install -e ".[gui,test]"
pytest                       # everything; tests on the validation data are skipped
pytest -m "not slow"         # about three minutes on two cores

python tests/data/make_synthetic_data.py           # rebuilds E, SA, SD and SC bit for bit
Rscript docs/external_reference/reference_sa.R boot    # Benchmarking on SA (~1 min)
Rscript docs/external_reference/reference_sd.R         # Benchmarking and R second stage on SD
Rscript docs/external_reference/reference_example.R    # Benchmarking and truncreg on E
python docs/external_reference/exact_mle.py            # 60-digit likelihood maxima (~1 min)
Rscript docs/external_reference/reference_scaling.R    # Benchmarking on SC
python docs/external_reference/exact_dea.py tests/data/scaling_case.csv x1,x2 y tests/data/scaling_case_exact.csv
python docs/scaling_stress.py                          # section 4.3: 100 draws and the sweep (~10 min)

python docs/make_figures.py  # every figure and number in the paper (~3 min; Figure 5 needs Node.js and Playwright)
python docs/sw_coverage.py 100; python docs/sw_coverage.py 200; python docs/sw_coverage.py 400   # section 10 (~1 hour each)
```

The R scripts need R with Benchmarking, AER, survival, truncreg, car, rstatix
and jsonlite. `make_figures.py` also asserts that the appendix table and the
paper's text report the same regression — a guard added after a draft was
found carrying the two fitted on different scores.

On the validation data, where they are available, with `PLEAP_PRIVATE_DATA`
pointing at the folder that holds them:

```bash
pytest                                                          # now including A and D
python docs/seed_study.py                                       # section 7 (~10 min)
python docs/bootstrap_decomposition.py no_reflect_0 shephard    # section 7 (~1 hour)
python docs/bootstrap_decomposition.py baseline h_x0.8 h_x1.2
python docs/bootstrap_decomposition.py --summarise
python docs/bootstrap_correction_forms.py                       # section 7 (~3 min)
Rscript docs/external_reference/bootstrap_benchmarking.R 1 2 3 4 5
python docs/bootstrap_benchmarking_aligned.py                   # section 7 (~10 min)
python docs/bootstrap_matched_b.py A                            # section 7.1 (~10 min)
python docs/bootstrap_matched_b.py D                            # section 7.1 (~1.5 hours)
Rscript docs/external_reference/secondstage_ds_d.R              # sections 5.1-5.2, data set D
python docs/external_reference/exact_mle.py "$PLEAP_PRIVATE_DATA/ds_d.csv" vrs_i x1 x2 tobit 0 1
```

These scripts write only aggregate results to `docs/`; anything per unit stays
in the private folder.

Data sets A and D come from the Korean Innovation Survey and are available
from STEPI under its terms of use (https://www.stepi.re.kr/kis/), not from this
repository; B and C are confidential. The MaxDEA export workbooks and the
Stata and SPSS output behind the stored reference values are kept in the
author's project archive.
