"""Shared fixtures for the PLEAP test suite.

Public data, in ``tests/data``: synthetic, drawn by
``tests/data/make_synthetic_data.py`` from stated data-generating processes,
with reference values from R (``docs/external_reference``):

* SA - 60 units x 2020, 2022, 2024, 2 inputs / 1 output, two covariates, two
  groupings: the layout of validation data set A.  Benchmarking's DEA (scores,
  slacks, projections, peers), Malmquist distances and bootstrap.
* SD - 400 units, 2 inputs / 1 output, two environmental variables, two
  groupings and ``vrs_i``, the BCC input-oriented score as an exported column
  (eight decimals): the layout of data set D.  Benchmarking's CCR, BCC and
  NIRS scores; R's Tobit, truncated regression, OLS and group tests; 60-digit
  likelihood maxima.
* the example of the paper's Section 3, ``examples/sample_data.xlsx``.

Private data: the validation data sets A and D themselves, with the MaxDEA,
SPSS, Stata and R output they were checked against.  They come from the
Korean Innovation Survey (KIS) of the Science and Technology Policy Institute
(STEPI), whose terms do not allow them to be redistributed, so they are not in
the repository.  Tests that need them read the folder named by the environment
variable ``PLEAP_PRIVATE_DATA`` (default ``tests/data_private``, which
``.gitignore`` excludes) and are skipped when the files are not there.
"""

import json
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
PRIVATE = pathlib.Path(os.environ.get("PLEAP_PRIVATE_DATA", HERE / "data_private"))

INS = ["i1", "i2"]
OUTS = ["o1"]


def private_csv(name):
    """A file of the private validation data, or a skip if it is not available."""
    path = PRIVATE / name
    if not path.is_file():
        pytest.skip(f"private validation data not available ({name}); "
                    "see tests/data/README.md")
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
#  Public: synthetic data and R references                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def sa():
    """SA, 2024 cross-section: 60 units, 2 inputs, 1 output."""
    return pd.read_csv(DATA / "sa_dea.csv")


@pytest.fixture(scope="session")
def sa_xy(sa):
    return (sa[INS].to_numpy(float), sa[OUTS].to_numpy(float))


@pytest.fixture(scope="session")
def sa_panel():
    """SA, balanced panel: 60 units x 3 periods."""
    return pd.read_csv(DATA / "sa_mpi.csv")


@pytest.fixture(scope="session")
def sa_ref_dea():
    """Benchmarking on SA: CCR and BCC x input and output orientation, with
    second-phase slacks, projections and peers at full precision."""
    return pd.read_csv(DATA / "sa_benchmarking_dea.csv")


@pytest.fixture(scope="session")
def sa_ref_mpi():
    """Benchmarking::malmq's four distances on SA, CRS and VRS, both orientations."""
    return pd.read_csv(DATA / "sa_benchmarking_mpi.csv")


@pytest.fixture(scope="session")
def sa_ref_boot():
    """Benchmarking::dea.boot on SA, BCC input orientation, NREP 2000, seeds 1-5."""
    return pd.read_csv(DATA / "sa_benchmarking_bootstrap.csv")


@pytest.fixture(scope="session")
def sd():
    """SD: 400 units, 2 inputs, 1 output, x1 and x2, group1 (2) and group2 (4),
    and vrs_i (eight decimals)."""
    return pd.read_csv(DATA / "sd.csv")


@pytest.fixture(scope="session")
def sd_xy(sd):
    return (sd[INS].to_numpy(float), sd[OUTS].to_numpy(float))


@pytest.fixture(scope="session")
def sd_ref_dea():
    """Benchmarking's CCR, BCC and NIRS scores on SD, both orientations."""
    return pd.read_csv(DATA / "sd_benchmarking_dea.csv")


@pytest.fixture(scope="session")
def sd_r():
    """R's second-stage regressions and group tests on SD."""
    return json.loads((DATA / "sd_r_secondstage.json").read_text())


@pytest.fixture(scope="session")
def exact_mle():
    """60-digit likelihood maxima (docs/external_reference/exact_mle.py)."""
    return json.loads((DATA / "exact_mle.json").read_text())


@pytest.fixture(scope="session")
def example_panel():
    """The example of Section 3, as shipped in examples/sample_data.xlsx."""
    return pd.read_csv(DATA / "example_panel.csv")


# --------------------------------------------------------------------------- #
#  Private: the KIS validation data sets A and D (skipped when absent)        #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def ds_a():
    """A, 2024 cross-section: 60 firms, 2 inputs, 1 output."""
    return private_csv("ds_a_dea.csv")


@pytest.fixture(scope="session")
def ds_a_xy(ds_a):
    return (ds_a[INS].to_numpy(float), ds_a[OUTS].to_numpy(float))


@pytest.fixture(scope="session")
def ds_a_panel():
    """A, balanced panel: 60 firms x 3 periods."""
    return private_csv("ds_a_mpi.csv")


@pytest.fixture(scope="session")
def ref_dea():
    return private_csv("ds_a_maxdea_dea.csv")


@pytest.fixture(scope="session")
def ref_mpi():
    return private_csv("ds_a_maxdea_mpi.csv")


@pytest.fixture(scope="session")
def ref_bootstrap():
    return private_csv("ds_a_maxdea_bootstrap.csv")


@pytest.fixture(scope="session")
def ref_a_benchmarking_boot():
    """Benchmarking::dea.boot on A: means over seeds 1-5 of PLEAP's estimator."""
    return private_csv("ds_a_benchmarking_bootstrap.csv")


@pytest.fixture(scope="session")
def ds_d():
    """D: 453 firms, 2 inputs, 1 output, 2 environmental variables, 2 group
    variables, and ``vrs_i``, the BCC input-oriented score as it was exported
    (8 decimals) to R, Stata and SPSS."""
    return private_csv("ds_d.csv")


@pytest.fixture(scope="session")
def ds_d_xy(ds_d):
    return (ds_d[INS].to_numpy(float), ds_d[OUTS].to_numpy(float))


@pytest.fixture(scope="session")
def ref_d_dea():
    """MaxDEA scores for data set D: models CCR, BCC and NIRS, both orientations."""
    return private_csv("ds_d_maxdea_dea.csv")


@pytest.fixture(scope="session")
def ref_d_bootstrap():
    """MaxDEA bootstrap for data set D, B = 1000: score, bias and mean."""
    return private_csv("ds_d_maxdea_bootstrap.csv")


@pytest.fixture(scope="session")
def ds_d207():
    """A 207-firm subset of D with its own group coding and a fixed score y."""
    return private_csv("ds_d207.csv")


@pytest.fixture
def rng():
    return np.random.default_rng(20260501)


# --------------------------------------------------------------------------- #
#  Small textbook technologies with analytically known efficiency scores.      #
# --------------------------------------------------------------------------- #
@pytest.fixture
def one_in_one_out():
    """Five DMUs, one input, one output.

    The CRS frontier is the ray through the most productive unit (D: y/x = 1.0).
    The VRS frontier is the piecewise-linear hull A-D-E.
    """
    X = np.array([[2.0], [4.0], [6.0], [8.0], [10.0]])
    Y = np.array([[1.0], [3.0], [5.0], [8.0], [9.0]])
    #              y/x =  0.5    0.75   0.833  1.0    0.9
    ccr_in = np.array([0.5, 0.75, 5.0 / 6.0, 1.0, 0.9])
    return X, Y, ccr_in


@pytest.fixture
def two_in_one_out():
    """Cooper-Seiford-Tone style unit-output example (CRS == VRS here)."""
    X = np.array([[4.0, 3.0], [7.0, 3.0], [8.0, 1.0], [4.0, 2.0],
                  [2.0, 4.0], [10.0, 1.0], [3.0, 7.0]])
    Y = np.ones((7, 1))
    return X, Y
