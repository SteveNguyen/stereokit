"""Camera logic that can be tested without the camera.

The width guard is the important one: the driver silently returns ONE eye when
a single-width size is requested, and splitting that frame in half produces two
halves of the left image that look plausibly like a stereo pair (spec 2.1).
"""
import numpy as np
import pytest

from stereokit import camera as C


def test_mode_table_matches_spec_table_2_1():
    m = C.MODES["1280x480"]
    assert (m.combined_w, m.combined_h) == (1280, 480)
    assert (m.eye_w, m.eye_h) == (640, 480)
    assert m.fps == 60
    assert abs(m.f_nominal - 670) < 2


def test_every_mode_is_double_width():
    for name, m in C.MODES.items():
        assert m.eye_w * 2 == m.combined_w, name
        assert m.eye_h == m.combined_h, name


def test_split_eyes_halves_the_frame():
    frame = np.zeros((480, 1280, 3), np.uint8)
    frame[:, :640] = 10
    frame[:, 640:] = 20
    left, right = C.split_eyes(frame)
    assert left.shape == right.shape == (480, 640, 3)
    assert left.mean() == 10 and right.mean() == 20


class FakeCap:
    """Stands in for cv2.VideoCapture so the guard is testable offline."""

    def __init__(self, actual_width, actual_height=480, opened=True):
        self.actual_width = actual_width
        self.actual_height = actual_height
        self._opened = opened
        self.props = {}
        self.released = False

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.actual_width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.actual_height)
        return float(self.props.get(prop, 0))

    def read(self):
        return True, np.zeros((self.actual_height, self.actual_width, 3), np.uint8)

    def release(self):
        self.released = True


def test_guard_rejects_node_that_returns_a_single_eye():
    """1280 requested, 640 delivered => not the stereo node."""
    cap = FakeCap(actual_width=640)
    assert C._accept(cap, 1280, 480) is False
    assert cap.released is True


def test_guard_accepts_node_that_returns_the_full_width():
    cap = FakeCap(actual_width=1280)
    assert C._accept(cap, 1280, 480) is True
    assert cap.released is False


def test_guard_rejects_mono_camera_that_negotiates_to_1280():
    """1280 is the commonest webcam width; a width-only check would bind a
    mono camera and split one image into a fake stereo pair (spec 2.1)."""
    cap = FakeCap(actual_width=1280, actual_height=720)
    assert C._accept(cap, 1280, 480) is False
    assert cap.released is True


def test_restore_puts_auto_flags_back_last():
    """Setting auto flags first would be undone by the manual writes."""
    import cv2
    cap = FakeCap(1280)
    saved = {cv2.CAP_PROP_AUTO_EXPOSURE: 3.0, cv2.CAP_PROP_AUTO_WB: 1.0,
             cv2.CAP_PROP_EXPOSURE: 166.0, cv2.CAP_PROP_GAIN: 32.0}
    order = []
    original_set = cap.set
    cap.set = lambda p, v: (order.append(p), original_set(p, v))[1]
    C.restore_controls(cap, saved)
    assert order[-2:] == [cv2.CAP_PROP_AUTO_WB, cv2.CAP_PROP_AUTO_EXPOSURE]
    assert cap.props[cv2.CAP_PROP_AUTO_WB] == 1.0


def test_restore_falls_back_to_defaults_when_value_unreadable():
    import cv2
    cap = FakeCap(1280)
    C.restore_controls(cap, {cv2.CAP_PROP_AUTO_EXPOSURE: -1.0})
    assert cap.props[cv2.CAP_PROP_AUTO_EXPOSURE] == C.DEFAULT_PROPS[
        cv2.CAP_PROP_AUTO_EXPOSURE]


def test_unknown_mode_is_rejected():
    with pytest.raises(KeyError):
        C.StereoCamera(mode="999x999")
