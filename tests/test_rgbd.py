"""Storage round trip and frame invariants.

16-bit PNG millimetres with 0 = invalid is the TUM/Redwood convention, so
Open3D and friends read the output without a converter (spec 8).
"""
import json

import numpy as np
import pytest

from stereokit import calibrate as CAL
from stereokit import rgbd as R
from stereokit.rectify import Rectifier
from tests.synthetic import make_pairs


@pytest.fixture(scope="module")
def rect():
    pairs, _ = make_pairs(n=40, seed=5)
    return Rectifier(CAL.solve(pairs, (640, 480), "1280x480", 25.0))


def make_frame(rect):
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.3, 3.5, (480, 640)).astype(np.float32)
    depth[:10, :10] = np.nan                     # an invalid patch
    return R.RGBDFrame(
        rgb=rng.integers(0, 255, (480, 640, 3)).astype(np.uint8),
        depth=depth,
        disparity=(rect.fx * rect.baseline_m / depth).astype(np.float32),
        K=rect.K, Q=rect.Q, timestamp=1234.5678)


def test_save_load_preserves_depth_to_millimetre(rect, tmp_path):
    frame = make_frame(rect)
    R.save_frame(tmp_path, frame)
    back = R.load_frame(tmp_path, frame.timestamp)
    valid = np.isfinite(frame.depth) & np.isfinite(back.depth)
    assert valid.sum() > 0
    assert np.abs(frame.depth[valid] - back.depth[valid]).max() < 0.0006


def test_invalid_depth_round_trips_as_nan(rect, tmp_path):
    frame = make_frame(rect)
    R.save_frame(tmp_path, frame)
    back = R.load_frame(tmp_path, frame.timestamp)
    assert np.isnan(back.depth[:10, :10]).all()


def test_rgb_round_trips_exactly(rect, tmp_path):
    frame = make_frame(rect)
    R.save_frame(tmp_path, frame)
    back = R.load_frame(tmp_path, frame.timestamp)
    assert np.array_equal(frame.rgb, back.rgb)


def test_depth_is_written_as_16bit_millimetres(rect, tmp_path):
    import cv2
    frame = make_frame(rect)
    R.save_frame(tmp_path, frame)
    raw = cv2.imread(str(tmp_path / "depth" / f"{frame.timestamp:.6f}.png"),
                     cv2.IMREAD_UNCHANGED)
    assert raw.dtype == np.uint16
    assert raw[:10, :10].max() == 0          # 0 encodes invalid


def test_meta_records_intrinsics_and_mode(rect, tmp_path):
    R.write_meta(tmp_path, rect)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["mode"] == "1280x480"
    assert meta["eye_size"] == [640, 480]
    assert np.allclose(np.array(meta["K"]), rect.K)
    assert abs(meta["baseline_m"] - rect.baseline_m) < 1e-9


def test_rgbd_rejects_calibration_from_another_mode(tmp_path):
    """Spec 5.4: loading a calibration against a different mode is an error."""
    pairs, _ = make_pairs(n=40, seed=6)
    calib = CAL.solve(pairs, (640, 480), "1280x480", 25.0)
    calib["mode"] = "1280x360"               # pretend it came from elsewhere
    calib["eye_size"] = [640, 360]
    path = tmp_path / "calibration.json"
    CAL.save(calib, path)
    with pytest.raises(ValueError, match="mode"):
        R.StereoRGBD(path, mode="1280x480")
