# Test data

Everything in this folder is synthetic or derived from synthetic data.  No
value comes from a survey or any other real source.

## The data sets

`make_synthetic_data.py` draws them from stated data-generating processes,
with fixed seeds, and rebuilds the CSV files bit for bit (the example workbook
too, apart from the time stamp in its file properties):

    python tests/data/make_synthetic_data.py

| File | Contents |
|---|---|
| `example_panel.csv` | The example of the paper's Section 3, identical to `examples/sample_data.xlsx`: 60 units × 2021–2023, inputs `x1`, `x2`, outputs `y1`, `y2`, environmental variables `z1`, `z2`, groups A/B/C. |
| `sa_mpi.csv` | **SA**: 60 units × 2020, 2022, 2024, inputs `i1`, `i2`, output `o1`, covariates `z1`, `z2`, groupings `g2` (2 groups) and `g4` (4). It has the layout of validation data set A. |
| `sa_dea.csv` | The 2024 cross-section of SA. |
| `scaling_case.csv` | **SC**: the numerical-fault case of the paper's Section 4.3. 30 units, inputs `x1` (uniform) and `x2` (log-uniform over 11.3 decades), output `y`. Dividing `x2` by its arithmetic mean puts its smallest entry below the LP solver's feasibility tolerance. |
| `sd.csv` | **SD**: 400 units, inputs `i1`, `i2`, output `o1`, environmental variables `x1`, `x2` (means of five-point items), `group1` (2 groups), `group2` (4), and `vrs_i`, the BCC input-oriented score rounded to eight decimals as an exported column would be. It has the layout of validation data set D. |

The generating processes of the example, SA and SD share one technology: a fixed input requirement
and a concave frontier, so increasing, constant and decreasing returns all
occur, and units use 1/θ times their frontier inputs, with θ drawn from a
truncated normal regression on the environmental variables and the group.
The parameters are in the script.

## Reference values

Each file is written by a script in `docs/external_reference/`; run it from
the repository root to reproduce it.

| File | Program | Contents |
|---|---|---|
| `sa_benchmarking_dea.csv` | R Benchmarking 0.33 (`reference_sa.R`) | CCR and BCC, input and output orientation: score, second-phase slacks, projections, peers and intensity weights, full precision |
| `sa_benchmarking_mpi.csv` | R Benchmarking 0.33 (`reference_sa.R`) | The four distance functions of the Malmquist index, CRS and VRS, both orientations; `Inf` where a cross-period program is infeasible |
| `sa_benchmarking_bootstrap.csv` | R Benchmarking 0.33 (`reference_sa.R boot`) | `dea.boot`, BCC input orientation, NREP = 2000, seeds 1–5, with the bandwidth Benchmarking prints |
| `sd_benchmarking_dea.csv` | R Benchmarking 0.33 (`reference_sd.R`) | CCR, BCC and NIRS scores, both orientations |
| `sd_r_secondstage.json` | R 4.3.3: AER, survival, truncreg, car, rstatix, stats (`reference_sd.R`) | Tobit, truncated regression, OLS, Mann–Whitney, Kruskal–Wallis, Dunn, t-tests, Levene, ANOVA, Welch's ANOVA, Tukey HSD, Games–Howell |
| `example_benchmarking_dea.csv` | R Benchmarking 0.33 (`reference_example.R`) | The example's last-period CCR and BCC scores |
| `example_bcc_input.csv` | R Benchmarking 0.33 (`reference_example.R`) | The example's last-period BCC input-oriented scores with `z1`, `z2`: the sample of Algorithm 1 |
| `example_truncreg_R.csv` | R truncreg 0.2-5 (`reference_example.R`) | Truncated regression of those scores, units strictly inside the frontier |
| `scaling_case_exact.csv` | Exact rational arithmetic (`exact_dea.py`) | The optimum of every unit's envelopment program on SC, CCR and BCC, both orientations, solved by a primal simplex in fractions: no tolerance anywhere. As a double and as a 40-digit decimal |
| `scaling_case_benchmarking.csv` | R Benchmarking 0.33 at its defaults (`reference_scaling.R`) | The same scores. Off the exact optimum on one VRS unit (U24): 6.9 × 10⁻⁷ input-, 2.5 × 10⁻³ output-oriented |
| `exact_mle.json` | mpmath, 60 digits (`exact_mle.py`) | Likelihood maxima for the truncated regressions and Tobit models above, including truncation at 0 and 1, which R's truncreg cannot fit |

## Validation data that are not here

PLEAP was also checked against MaxDEA 6.6 Ultra, SPSS, Stata 16 and R on two real
data sets, A (60 firms × three periods) and D (453 firms), both from the
Korean Innovation Survey (KIS) of the Science and Technology Policy Institute
(STEPI).  STEPI's terms of use do not allow the data to be passed on, so they
are not distributed.  `docs/validation.md` reports those comparisons in
aggregate.

The tests written for A and D are part of the suite and are skipped when the
files are absent.  To run them, put the files in a folder outside the
repository and point `PLEAP_PRIVATE_DATA` at it (or use `tests/data_private/`,
which `.gitignore` excludes):

    PLEAP_PRIVATE_DATA=/path/to/private pytest

    pytest -m "not private"     # the public suite only

The 135-plant (B) and 61-unit (C) data sets of `docs/validation.md` are not
distributed either; their MaxDEA comparisons are recorded there in aggregate,
and there are no tests for them.

This research used the Korean Innovation Survey (KIS) data provided by the
Science and Technology Policy Institute (STEPI).
