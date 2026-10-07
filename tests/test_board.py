"""The board is the measurement standard for everything downstream.

If the generated page and the detector ever disagree, every calibration built
on it is silently wrong, so the round trip is tested, not assumed.
"""
import cv2
import numpy as np

from stereokit import board as B


def test_board_geometry_constants():
    assert B.BOARD["squares_x"] == 7
    assert B.BOARD["squares_y"] == 10
    assert B.BOARD["square_mm"] == 25.0
    assert B.n_corners() == 54


def test_page_is_exactly_a4():
    page = B.render_page()
    h, w = page.shape
    assert w / B.PX_PER_MM == B.A4_W_MM
    assert h / B.PX_PER_MM == B.A4_H_MM


def test_generate_detect_round_trip_finds_every_corner():
    page = B.render_page()
    cc, ci, mc, mi = cv2.aruco.CharucoDetector(B.make_board()).detectBoard(page)
    assert cc is not None and len(cc) == B.n_corners()
    assert mi is not None and len(mi) == 35


def test_detected_squares_are_square_and_exact():
    page = B.render_page()
    cc, ci, _, _ = cv2.aruco.CharucoDetector(B.make_board()).detectBoard(page)
    pts = cc.reshape(-1, 2)
    lut = {int(i): p for i, p in zip(ci.ravel(), pts)}
    per_row = B.BOARD["squares_x"] - 1
    horiz = [np.linalg.norm(lut[i + 1] - lut[i]) for i in lut
             if (i + 1) % per_row != 0 and i + 1 in lut]
    vert = [np.linalg.norm(lut[i + per_row] - lut[i]) for i in lut
            if i + per_row in lut]
    expected = B.BOARD["square_mm"] * B.PX_PER_MM
    assert abs(np.median(horiz) - expected) / expected < 1e-3
    assert abs(np.median(vert) - expected) / expected < 1e-3
    assert abs(np.median(horiz) / np.median(vert) - 1.0) < 1e-3


def test_measured_square_override_scales_the_board():
    """Printers rescale; calibration must use the measured size (spec 5.1)."""
    b = B.make_board(square_mm=24.5)
    corners = b.getChessboardCorners()
    spacing = np.linalg.norm(corners[1] - corners[0])
    assert abs(spacing - 0.0245) < 1e-6


def test_board_fits_a4_with_printable_margins():
    bw = B.BOARD["squares_x"] * B.BOARD["square_mm"]
    assert (B.A4_W_MM - bw) / 2 >= 10.0
