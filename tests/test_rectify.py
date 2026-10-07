import numpy as np
import pytest

from stereokit import calibrate as CAL
from stereokit.rectify import Rectifier
from tests.synthetic import make_pairs, BASELINE_TRUE


@pytest.fixture(scope="module")
def rect():
    pairs, _ = make_pairs(n=40, seed=3)
    return Rectifier(CAL.solve(pairs, (640, 480), "1280x480", 25.0))


def test_exposes_rectified_intrinsics(rect):
    assert rect.K.shape == (3, 3)
    assert abs(rect.fx - rect.K[0, 0]) < 1e-9
    assert 600 < rect.fx < 750


def test_baseline_matches_truth(rect):
    assert abs(rect.baseline_m - BASELINE_TRUE) / BASELINE_TRUE < 0.02


def test_rectify_preserves_shape(rect):
    left = np.zeros((480, 640, 3), np.uint8)
    right = np.zeros((480, 640, 3), np.uint8)
    lr, rr = rect.rectify(left, right)
    assert lr.shape == (480, 640, 3)
    assert rr.shape == (480, 640, 3)


def test_rectify_rejects_wrong_frame_size(rect):
    """A calibration is per-mode; the crop moves the principal point, so
    using it against another mode is an error, not a warning (spec 5.4)."""
    bad = np.zeros((720, 1280, 3), np.uint8)
    with pytest.raises(ValueError, match="eye size"):
        rect.rectify(bad, bad)


def test_rectified_corners_land_on_the_same_row(rect):
    """The spec 6 verification: corresponding corners within 0.3 px in y."""
    import cv2
    pairs, _ = make_pairs(n=5, seed=9)
    calib = rect.calib
    Kl = np.array(calib["left"]["K"]); Dl = np.array(calib["left"]["dist"]).ravel()
    Kr = np.array(calib["right"]["K"]); Dr = np.array(calib["right"]["dist"]).ravel()
    R1 = np.array(calib["rectified"]["R1"]); P1 = np.array(calib["rectified"]["P1"])
    R2 = np.array(calib["rectified"]["R2"]); P2 = np.array(calib["rectified"]["P2"])
    errs = []
    for p in pairs:
        ul = cv2.undistortPoints(p.img_left.reshape(-1, 1, 2), Kl, Dl,
                                 R=R1, P=P1).reshape(-1, 2)
        ur = cv2.undistortPoints(p.img_right.reshape(-1, 1, 2), Kr, Dr,
                                 R=R2, P=P2).reshape(-1, 2)
        errs.append(np.abs(ul[:, 1] - ur[:, 1]))
    assert np.concatenate(errs).mean() < 0.3


def test_q_matrix_encodes_the_baseline(rect):
    """Q[3,2] = 1/baseline, so reprojectImageTo3D returns metres."""
    assert abs(abs(rect.Q[3, 2]) - 1.0 / rect.baseline_m) < 0.5
