"""The public API: aligned RGB and metric depth from the stereo pair.

RGB is the RECTIFIED LEFT image and depth is computed in that same frame, so
the two are aligned by construction — no warp, no resampling, no interpolation
error at depth discontinuities (spec 4.1).
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import calibrate
from . import profiles
from .camera import StereoCamera
from .depth import DepthEstimator
from .rectify import Rectifier


@dataclass
class RGBDFrame:
    rgb: np.ndarray          # (H,W,3) uint8, rectified left
    depth: np.ndarray        # (H,W) float32 metres, NaN where invalid
    disparity: np.ndarray    # (H,W) float32, NaN where invalid
    K: np.ndarray            # (3,3) rectified intrinsics
    Q: np.ndarray            # (4,4) reprojection matrix
    timestamp: float


class StereoRGBD:
    def __init__(self, calibration_path, mode=None, device=None,
                 **depth_kwargs):
        calib = calibrate.load(calibration_path)
        if mode is not None and calib["mode"] != mode:
            raise ValueError(
                f"calibration is for mode {calib['mode']} but {mode} was "
                "requested. Calibration is per-mode: the 4:3 crop moves the "
                "principal point, so it cannot be rescaled.")
        self.mode = calib["mode"]
        # Older calibrations predate the camera key; they are all AR0144.
        self.camera = calib.get("camera", profiles.DEFAULT_PROFILE.name)
        self.rectifier = Rectifier(calib)
        self.estimator = DepthEstimator(
            fx=self.rectifier.fx, baseline_m=self.rectifier.baseline_m,
            **depth_kwargs)
        self._camera = StereoCamera(mode=self.mode, device=device,
                                    profile=profiles.get(self.camera))

    @property
    def K(self):
        return self.rectifier.K

    @property
    def Q(self):
        return self.rectifier.Q

    def read(self):
        left, right, ts = self._camera.grab()
        lr, rr = self.rectifier.rectify(left, right)
        disp, depth = self.estimator.compute(lr, rr)
        # Kept so callers can save a full rectified PAIR; the public frame
        # carries only the left eye, which is the aligned RGB.
        self._last_right = rr
        return RGBDFrame(rgb=lr, depth=depth, disparity=disp,
                         K=self.K, Q=self.Q, timestamp=ts)

    def __iter__(self):
        while True:
            yield self.read()

    def __enter__(self):
        self._camera.open()
        return self

    def __exit__(self, *exc):
        self._camera.close()
        return False


# --- storage, TUM/Redwood convention (spec 8) ---------------------------------

DEPTH_SCALE = 1000.0          # millimetres; 16-bit caps at 65.535 m


def save_frame(out_dir, frame):
    out_dir = Path(out_dir)
    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    name = f"{frame.timestamp:.6f}.png"
    rgb_path = out_dir / "rgb" / name
    depth_path = out_dir / "depth" / name

    mm = frame.depth * DEPTH_SCALE
    mm[~np.isfinite(mm)] = 0          # 0 encodes invalid
    # round(), not the cast's implicit truncation toward zero: truncating
    # biases every saved depth by up to -1mm in one direction (spec 3).
    mm = np.clip(np.round(mm), 0, 65535).astype(np.uint16)

    cv2.imwrite(str(rgb_path), frame.rgb)
    cv2.imwrite(str(depth_path), mm)
    return rgb_path, depth_path


def load_frame(out_dir, timestamp):
    out_dir = Path(out_dir)
    name = f"{timestamp:.6f}.png"
    rgb = cv2.imread(str(out_dir / "rgb" / name), cv2.IMREAD_COLOR)
    mm = cv2.imread(str(out_dir / "depth" / name), cv2.IMREAD_UNCHANGED)
    depth = mm.astype(np.float32) / DEPTH_SCALE
    depth[mm == 0] = np.nan

    meta_path = out_dir / "meta.json"
    K = Q = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        K = np.array(meta["K"])
        Q = np.array(meta["Q"])
    return RGBDFrame(rgb=rgb, depth=depth, disparity=np.full_like(depth, np.nan),
                     K=K, Q=Q, timestamp=timestamp)


def write_meta(out_dir, rectifier):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "meta.json"
    path.write_text(json.dumps({
        "mode": rectifier.mode,
        "eye_size": list(rectifier.eye_size),
        "K": rectifier.K.tolist(),
        "Q": rectifier.Q.tolist(),
        "baseline_m": rectifier.baseline_m,
        "depth_scale": DEPTH_SCALE,
        "depth_units": "millimetres, 0 = invalid",
    }, indent=2))
    return path
