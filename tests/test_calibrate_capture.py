"""Capture-side logic: what makes a pair worth keeping, and when to stop.

A calibration is only as good as the variety of views fed to it. Coverage of
the image area conditions distortion; tilt variety conditions focal length.
"""
import cv2
import numpy as np

from stereokit import board as B
from stereokit import calibrate as CAL


def test_detect_pair_finds_the_board_in_both_eyes():
    page = B.render_page()
    eye = cv2.resize(page, (640, 480), interpolation=cv2.INTER_AREA)
    board = B.make_board()
    det = cv2.aruco.CharucoDetector(board)
    pair = CAL.detect_pair(det, board, eye, eye)
    assert pair is not None
    assert pair.n == B.n_corners()
    assert pair.obj.shape == (B.n_corners(), 3)
    assert pair.img_left.shape == (B.n_corners(), 2)


def test_detect_pair_returns_none_when_an_eye_has_no_board():
    page = B.render_page()
    eye = cv2.resize(page, (640, 480), interpolation=cv2.INTER_AREA)
    blank = np.full((480, 640), 200, np.uint8)
    board = B.make_board()
    det = cv2.aruco.CharucoDetector(board)
    assert CAL.detect_pair(det, board, eye, blank) is None


def test_detect_pair_intersects_ids_seen_by_both_eyes():
    """Extrinsics need the SAME physical corners in both images."""
    page = B.render_page()
    full = cv2.resize(page, (640, 480), interpolation=cv2.INTER_AREA)
    half = full.copy()
    half[:, 400:] = 255            # hide part of the board from the right eye
    board = B.make_board()
    det = cv2.aruco.CharucoDetector(board)
    pair = CAL.detect_pair(det, board, full, half)
    assert pair is not None
    assert pair.n < B.n_corners()          # the occlusion really cost corners
    assert pair.obj.shape[0] == pair.n
    assert pair.img_left.shape == pair.img_right.shape == (pair.n, 2)


def test_sharpness_drops_with_blur():
    page = B.render_page()
    eye = cv2.resize(page, (640, 480), interpolation=cv2.INTER_AREA)
    board = B.make_board()
    det = cv2.aruco.CharucoDetector(board)
    pair = CAL.detect_pair(det, board, eye, eye)
    sharp = CAL.sharpness(eye, pair.corners_left)
    blurred = cv2.GaussianBlur(eye, (0, 0), 2.0)
    assert CAL.sharpness(blurred, pair.corners_left) < sharp


def test_fov_coverage_starts_empty_and_fills():
    t = CAL.CoverageTracker((640, 480), grid=(4, 3))
    assert t.fov_coverage() == 0.0
    # One pair in the middle covers only part of the grid.
    t.add(np.array([[320.0, 240.0]]), tilt_deg=0.0)
    assert 0.0 < t.fov_coverage() < 1.0
    # Corners spanning every cell fill it.
    pts = np.array([[x, y] for x in np.linspace(5, 635, 12)
                    for y in np.linspace(5, 475, 9)])
    t.add(pts, tilt_deg=10.0)
    assert t.fov_coverage() == 1.0


def test_tilt_coverage_needs_varied_angles():
    t = CAL.CoverageTracker((640, 480), tilt_bins=4)
    pts = np.array([[320.0, 240.0]])
    for _ in range(20):
        t.add(pts, tilt_deg=2.0)        # all flat-on
    assert t.tilt_coverage() < 0.5
    for angle in (15.0, 25.0, 40.0):
        t.add(pts, tilt_deg=angle)
    assert t.tilt_coverage() == 1.0


def _full_coverage_points():
    return np.array([[x, y] for x in np.linspace(5, 635, 12)
                     for y in np.linspace(5, 475, 9)])


def test_done_requires_pairs_coverage_tilt_AND_distance():
    t = CAL.CoverageTracker((640, 480), target_pairs=3)
    pts = _full_coverage_points()
    for angle in (2.0, 15.0, 28.0, 42.0):
        t.add(pts, tilt_deg=angle)
    assert t.n_pairs == 4
    # FOV and tilt are satisfied but every view was at the same distance.
    assert t.done() is False, "distance variety must be required"
    for d in CAL.CoverageTracker.DIST_BINS:
        t.add(pts, tilt_deg=20.0, dist_m=d)
    assert t.dist_coverage() == 1.0
    assert t.done() is True


def test_distance_coverage_reports_what_is_missing():
    """Depth variety is what separates eye-to-eye yaw from baseline. Without
    it the two trade off into a uniform disparity offset that the epipolar
    gate cannot see, because epipolar measures ROW alignment while a yaw
    error shifts disparity in x."""
    t = CAL.CoverageTracker((640, 480))
    pts = _full_coverage_points()
    assert t.dist_coverage() == 0.0
    assert t.missing_distances() == list(CAL.CoverageTracker.DIST_BINS)
    t.add(pts, tilt_deg=10.0, dist_m=0.25)
    t.add(pts, tilt_deg=10.0, dist_m=0.90)
    missing = t.missing_distances()
    assert 0.25 not in missing and 0.90 not in missing
    assert 0.45 in missing


def test_distance_outside_the_useful_range_is_ignored():
    t = CAL.CoverageTracker((640, 480))
    pts = _full_coverage_points()
    t.add(pts, tilt_deg=10.0, dist_m=3.0)      # far beyond the bins
    assert t.dist_coverage() == 0.0


def test_board_tilt_is_zero_when_facing_the_camera():
    assert CAL.board_tilt_deg(np.zeros(3)) < 1e-6
    # 30 degrees about x
    rvec = np.array([np.deg2rad(30.0), 0.0, 0.0])
    assert abs(CAL.board_tilt_deg(rvec) - 30.0) < 1e-3
