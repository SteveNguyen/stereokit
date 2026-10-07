"""The two side-by-side stereo traps, asserted rather than remembered.

Both produce a plausible-looking but wrong map, which is the worst failure
mode: nothing errors, the result is just quietly incorrect (spec 9.1).
"""
import numpy as np
import pytest
import yaml

from stereokit import calibrate as CAL
from stereokit import export as EX
from tests.synthetic import make_pairs


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    pairs, _ = make_pairs(n=40, seed=7)
    calib = CAL.solve(pairs, (640, 480), "1280x480", 25.0)
    d = tmp_path_factory.mktemp("export")
    CAL.save(calib, d / "calibration.json")
    left, right = EX.export_calibration(d / "calibration.json", d)
    return calib, yaml.safe_load(left.read_text()), yaml.safe_load(right.read_text())


def test_image_width_is_per_eye_not_the_driver_width(exported):
    """640, not the 1280 the driver advertises. THE most common
    side-by-side stereo misconfiguration."""
    _, left, right = exported
    assert left["image_width"] == 640
    assert right["image_width"] == 640
    assert left["image_height"] == 480


def test_right_projection_translation_is_negative(exported):
    """ROS convention P2[0,3] = -fx*B. Wrong sign gives a mirrored map."""
    calib, _, right = exported
    P2 = right["projection_matrix"]["data"]
    assert P2[3] < 0


def test_right_projection_translation_equals_minus_fx_baseline(exported):
    calib, left, right = exported
    fx = left["projection_matrix"]["data"][0]
    expected = -fx * calib["baseline_m"]
    assert abs(right["projection_matrix"]["data"][3] - expected) / abs(expected) < 0.01


def test_left_projection_has_no_translation(exported):
    _, left, _ = exported
    assert abs(left["projection_matrix"]["data"][3]) < 1e-9


def test_distortion_model_is_plumb_bob_with_five_coefficients(exported):
    _, left, right = exported
    for info in (left, right):
        assert info["distortion_model"] == "plumb_bob"
        assert info["distortion_coefficients"]["cols"] == 5
        assert len(info["distortion_coefficients"]["data"]) == 5


def test_matrix_shapes_match_ros_camera_info(exported):
    _, left, _ = exported
    assert left["camera_matrix"]["rows"] == 3
    assert left["camera_matrix"]["cols"] == 3
    assert len(left["camera_matrix"]["data"]) == 9
    assert left["rectification_matrix"]["rows"] == 3
    assert len(left["rectification_matrix"]["data"]) == 9
    assert left["projection_matrix"]["rows"] == 3
    assert left["projection_matrix"]["cols"] == 4
    assert len(left["projection_matrix"]["data"]) == 12


def test_camera_names_are_distinct(exported):
    _, left, right = exported
    assert left["camera_name"].endswith("left")
    assert right["camera_name"].endswith("right")


def test_matrix_rejects_shape_mismatch():
    """A camera_info block whose declared shape disagrees with its data is
    misread rather than rejected downstream, so refuse to emit one."""
    with pytest.raises(ValueError, match="values"):
        EX._matrix([1.0, 2.0, 3.0], 3, 3)


def test_export_rejects_calibration_with_short_distortion(tmp_path):
    """calibrate.load() does no schema validation, so a hand-edited
    calibration.json can reach export with the wrong coefficient count."""
    pairs, _ = make_pairs(n=40, seed=11)
    calib = CAL.solve(pairs, (640, 480), "1280x480", 25.0)
    calib["left"]["dist"] = [0.1, 0.2]          # only 2 of 5
    CAL.save(calib, tmp_path / "calibration.json")
    with pytest.raises(ValueError, match="values"):
        EX.export_calibration(tmp_path / "calibration.json", tmp_path)
