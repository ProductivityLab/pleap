"""Returns-to-scale classification.

PLEAP classifies the radial projection of each unit by the rule of Seiford
and Zhu (1999): CRS when the CCR and BCC scores agree, otherwise IRS when the
CCR intensity weights sum to less than 1 and DRS when they sum to more, in
BOTH orientations.  Up to v1.11 the output-oriented branch had the two labels
swapped, so every scale-inefficient unit was mislabelled under output
orientation; the input orientation was right.

The check is against a test that shares no code with the rule: Fare,
Grosskopf and Lovell's comparison with the non-increasing-returns (NIRS)
score - CRS if CCR = BCC, DRS if NIRS = BCC, IRS otherwise - computed from
another program's CCR, BCC and NIRS scores: R Benchmarking's on the synthetic
data set SD (public), and MaxDEA's on the validation data set D (private,
skipped when absent; see conftest.py).

MaxDEA's own RTS column for D is not used: it agrees with that test, run on
MaxDEA's own scores, for 310 of 453 units under input orientation and 82 of
453 under output orientation, and matches neither the sum-of-weights rule nor
the NIRS test, so it is not a usable reference.  The rule behind it is not
documented in anything we could consult.
"""

import numpy as np
import pytest

from pleap.core.dea import rts_classify

TOL = 1e-7


def test_one_input_one_output_toy_with_known_answer():
    """VRS frontier A(1,1) - B(2,3) - C(4,4); B is the most productive scale
    size.  Unit E(3,2) projects left of B under input orientation and right
    of it under output orientation, so its RTS depends on the orientation."""
    X = np.array([[1.0], [2.0], [4.0], [1.5], [3.0]])
    Y = np.array([[1.0], [3.0], [4.0], [1.0], [2.0]])
    lab_in, _, _ = rts_classify(X, Y, orient="input")
    lab_out, _, _ = rts_classify(X, Y, orient="output")
    assert lab_in == ["IRS", "CRS", "DRS", "IRS", "IRS"]
    assert lab_out == ["IRS", "CRS", "DRS", "IRS", "DRS"]


def _nirs_test(ref, dmus, orient):
    r = ref[ref.orient == orient].pivot(index="dmu", columns="model", values="score").loc[dmus]
    ccr, bcc, nirs = r["CCR"].to_numpy(), r["BCC"].to_numpy(), r["NIRS"].to_numpy()
    return np.where(np.abs(ccr - bcc) < TOL, "CRS",
                    np.where(np.abs(nirs - bcc) < TOL, "DRS", "IRS"))


# data, reference scores, and the label counts the NIRS test gives
CASES = {
    "SD vs Benchmarking": ("sd", "sd_ref_dea", {"input": {"CRS": 4, "DRS": 229, "IRS": 167},
                                                "output": {"CRS": 5, "DRS": 333, "IRS": 62}}),
    "D vs MaxDEA (private)": ("ds_d", "ref_d_dea", {"input": {"CRS": 5, "DRS": 4, "IRS": 444},
                                                    "output": {"CRS": 5, "DRS": 212, "IRS": 236}}),
}


def _case(request, case):
    data, ref, counts = CASES[case]
    ds = request.getfixturevalue(data)
    return ds, (ds[["i1", "i2"]].to_numpy(float), ds[["o1"]].to_numpy(float)), \
        request.getfixturevalue(ref), counts


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("orient", ["input", "output"])
def test_labels_agree_with_the_nirs_test_on_every_unit(request, case, orient):
    ds, xy, ref, counts = _case(request, case)
    labels, ccr, bcc = rts_classify(*xy, orient=orient)
    expected = _nirs_test(ref, list(ds["dmu"]), orient)
    assert list(labels) == list(expected)
    got = {k: int(np.sum(np.array(labels) == k)) for k in ("CRS", "DRS", "IRS")}
    assert got == counts[orient]


@pytest.mark.parametrize("case", list(CASES))
def test_the_returned_scores_are_the_references(request, case):
    ds, xy, ref, _ = _case(request, case)
    for orient in ("input", "output"):
        _, ccr, bcc = rts_classify(*xy, orient=orient)
        r = ref[ref.orient == orient].pivot(index="dmu", columns="model",
                                            values="score").loc[list(ds["dmu"])]
        np.testing.assert_allclose(ccr, r["CCR"].to_numpy(), atol=1e-10)
        np.testing.assert_allclose(bcc, r["BCC"].to_numpy(), atol=1e-10)
