"""The solver is tested against known ground truth.

Reprojection error on the fitting set measures fit, not accuracy, which is why
solve() holds 20% out and why these tests check RECOVERY of the true
parameters, not just a small residual (spec 5.3, 5.4).
"""
import json

import numpy as np
import pytest

from stereokit import calibrate as CAL
from stereokit import profiles as P
from tests.synthetic import make_pairs, K_TRUE, BASELINE_TRUE


@pytest.fixture(scope="module")
def solved():
    pairs, truth = make_pairs(n=40, seed=1)
    calib = CAL.solve(pairs, (640, 480), "1280x480", 25.0)
    return calib, truth


def test_recovers_focal_length(solved):
    calib, _ = solved
    fx = calib["left"]["K"][0][0]
    assert abs(fx - K_TRUE[0, 0]) / K_TRUE[0, 0] < 0.01


def test_recovers_principal_point(solved):
    calib, _ = solved
    cx = calib["left"]["K"][0][2]
    cy = calib["left"]["K"][1][2]
    assert abs(cx - K_TRUE[0, 2]) < 5.0
    assert abs(cy - K_TRUE[1, 2]) < 5.0


def test_recovers_baseline(solved):
    calib, _ = solved
    assert abs(calib["baseline_m"] - BASELINE_TRUE) / BASELINE_TRUE < 0.02


def test_meets_every_acceptance_criterion(solved):
    calib, _ = solved
    assert CAL.check_acceptance(calib, profile=P.AR0144) == []


def test_reports_holdout_count(solved):
    calib, _ = solved
    m = calib["metrics"]
    assert m["n_holdout"] == pytest.approx(0.2 * (m["n_pairs"] + m["n_holdout"]),
                                           abs=1)
    assert m["n_holdout"] >= 1
    assert m["f_nominal"] == pytest.approx(669.7, abs=0.1)


def test_epipolar_error_is_subpixel(solved):
    calib, _ = solved
    assert calib["metrics"]["epipolar_px"] < 0.3


def test_p2_translation_term_is_negative(solved):
    """ROS convention: P2[0,3] = -fx*B. Wrong sign yields a mirrored map."""
    calib, _ = solved
    P1 = np.array(calib["rectified"]["P1"])
    P2 = np.array(calib["rectified"]["P2"])
    assert P2[0, 3] < 0
    expected = -P1[0, 0] * calib["baseline_m"]
    assert abs(P2[0, 3] - expected) / abs(expected) < 0.01


def test_noisy_input_fails_acceptance():
    """The gates must actually reject a bad calibration."""
    pairs, _ = make_pairs(n=40, noise_px=3.0, seed=2)
    calib = CAL.solve(pairs, (640, 480), "1280x480", 25.0)
    assert CAL.check_acceptance(calib, profile=P.AR0144) != []


def test_wrong_mode_calibration_is_rejected(solved):
    """Spec 5.4: recovered f vs the mode's prior catches a calibration
    captured at the wrong mode. The 4:3 modes crop, so f does not scale
    with width and a rescaled calibration is quietly wrong."""
    calib, _ = solved
    assert CAL.check_acceptance(calib, profile=P.AR0144) == []          # 1280x480 passes
    wrong = json.loads(json.dumps(calib))             # deep copy
    wrong["mode"] = "1280x360"                        # f_nominal 502.3
    wrong["metrics"]["f_nominal"] = 502.3
    failures = CAL.check_acceptance(wrong, profile=P.AR0144)
    assert any("nominal" in f for f in failures)


def test_records_the_mode_it_was_captured_at(solved):
    calib, _ = solved
    assert calib["mode"] == "1280x480"
    assert calib["eye_size"] == [640, 480]


def test_save_load_round_trip(solved, tmp_path):
    calib, _ = solved
    p = tmp_path / "calibration.json"
    CAL.save(calib, p)
    back = CAL.load(p)
    assert back["mode"] == calib["mode"]
    assert np.allclose(np.array(back["left"]["K"]),
                       np.array(calib["left"]["K"]))


def test_rms_gate_is_angular_not_absolute_pixels():
    """A pixel limit is resolution-dependent: the same physical corner
    accuracy gives proportionally more pixels of error at higher resolution.
    The 2560x720 calibration was rejected at 0.371 px while being angularly
    the BEST of four (311 urad against 350-371), purely for having more
    pixels."""
    base = {"mode": "1280x480", "baseline_m": 0.052,
            "left": {"K": [[669.7, 0, 320], [0, 669.7, 240], [0, 0, 1]]},
            "metrics": {"rms_left": 0.29, "rms_right": 0.29,
                        "rms_stereo": 0.45, "epipolar_px": 0.1,
                        "f_recovered": 669.7, "f_nominal": 669.7}}
    # At the reference resolution the gate is unchanged from the old 0.3 px.
    assert CAL.check_acceptance(base, profile=P.AR0144) == []
    base["metrics"]["rms_left"] = 0.35
    assert any("rms_left" in f for f in CAL.check_acceptance(base, profile=P.AR0144))

    # The SAME angular accuracy at double the focal length must still pass,
    # even though the pixel figure is double.
    hi = json.loads(json.dumps(base))
    hi["baseline_m"] = 0.017          # gated against the SKL, not the AR0144
    hi["left"]["K"] = [[1339.4, 0, 640], [0, 1339.4, 360], [0, 0, 1]]
    hi["metrics"].update(rms_left=0.58, rms_right=0.58, rms_stereo=0.90,
                         f_recovered=1339.4, f_nominal=None)
    assert CAL.check_acceptance(hi, profile=P.SKL2MP220) == []
