"""Export calibration in the ROS camera_info format RTAB-Map reads.

Two traps here both produce a plausible-looking but wrong map, so both are
automated rather than remembered (spec 9.1).
"""
from pathlib import Path

import numpy as np
import yaml

from . import calibrate


def _matrix(a, rows, cols):
    data = [float(v) for v in np.asarray(a, float).reshape(-1)]
    if len(data) != rows * cols:
        raise ValueError(
            f"expected {rows}x{cols} = {rows * cols} values, got {len(data)}. "
            "A camera_info block whose declared shape disagrees with its data "
            "is misread rather than rejected by ROS/RTAB-Map parsers.")
    return {"rows": rows, "cols": cols, "data": data}


def camera_info_dict(name, width, height, K, dist, R_rect, P):
    return {
        "image_width": int(width),
        "image_height": int(height),
        "camera_name": name,
        "camera_matrix": _matrix(K, 3, 3),
        "distortion_model": "plumb_bob",
        "distortion_coefficients": _matrix(
            np.asarray(dist, float).reshape(-1)[:5], 1, 5),
        "rectification_matrix": _matrix(R_rect, 3, 3),
        "projection_matrix": _matrix(P, 3, 4),
    }


def export_calibration(calibration_path, out_dir, name="ar0144"):
    """Write <name>_left.yaml and <name>_right.yaml."""
    calib = calibrate.load(calibration_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # image_width is the PER-EYE width (640), not the 1280 the driver
    # advertises for the concatenated frame. This is the single most common
    # side-by-side stereo misconfiguration.
    w, h = calib["eye_size"]
    rec = calib["rectified"]

    # P2[0,3] must be -fx*B. OpenCV's stereoRectify already returns this
    # convention, so it is a direct copy — but a wrong sign yields a mirrored,
    # plausible-looking map, so raise rather than trust. Checked BEFORE
    # writing: a bad pair must never land on disk where RTAB-Map could load
    # it (same defect class already fixed in cmd_solve).
    P2 = np.array(rec["P2"], float)
    if P2[0, 3] >= 0:
        raise ValueError(
            f"P2[0,3] = {P2[0, 3]:.3f} must be negative (ROS convention "
            "-fx*B). A positive value yields a mirrored map that looks "
            "entirely believable.")

    paths = []
    for side, R_key, P_key in (("left", "R1", "P1"), ("right", "R2", "P2")):
        info = camera_info_dict(
            f"{name}_{side}", w, h,
            calib[side]["K"], calib[side]["dist"],
            rec[R_key], rec[P_key])
        path = out_dir / f"{name}_{side}.yaml"
        path.write_text(yaml.safe_dump(info, sort_keys=False))
        paths.append(path)

    return tuple(paths)
