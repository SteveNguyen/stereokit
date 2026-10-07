"""Rectification: raw eyes in, row-aligned eyes out.

Because both eyes are the same colour sensor, depth computed on the rectified
pair lands in the left camera's own frame and the rectified left image IS the
aligned RGB. There is no cross-sensor warp to get wrong (spec 4.1).
"""
import cv2
import numpy as np

from . import calibrate


class Rectifier:
    def __init__(self, calib):
        self.calib = calib
        self.mode = calib["mode"]
        self.eye_size = (int(calib["eye_size"][0]), int(calib["eye_size"][1]))
        self.baseline_m = float(calib["baseline_m"])

        rec = calib["rectified"]
        self.P1 = np.array(rec["P1"], float)
        self.P2 = np.array(rec["P2"], float)
        self.Q = np.array(rec["Q"], float)
        self.roi_left = tuple(rec["roi_left"])
        self.roi_right = tuple(rec["roi_right"])

        # The RECTIFIED intrinsics, not the raw ones: alpha=0 cropping changes
        # the effective FOV, so P1 is what downstream must use.
        self.K = self.P1[:, :3].copy()
        self.fx = float(self.K[0, 0])

        Kl = np.array(calib["left"]["K"], float)
        Dl = np.array(calib["left"]["dist"], float).ravel()
        Kr = np.array(calib["right"]["K"], float)
        Dr = np.array(calib["right"]["dist"], float).ravel()
        R1 = np.array(rec["R1"], float)
        R2 = np.array(rec["R2"], float)
        self._map_l = cv2.initUndistortRectifyMap(
            Kl, Dl, R1, self.P1, self.eye_size, cv2.CV_32FC1)
        self._map_r = cv2.initUndistortRectifyMap(
            Kr, Dr, R2, self.P2, self.eye_size, cv2.CV_32FC1)

    @classmethod
    def from_file(cls, path):
        return cls(calibrate.load(path))

    def rectify(self, left, right):
        expected = (self.eye_size[1], self.eye_size[0])
        for name, img in (("left", left), ("right", right)):
            if img.shape[:2] != expected:
                raise ValueError(
                    f"{name} frame is {img.shape[1]}x{img.shape[0]} but this "
                    f"calibration is for eye size {self.eye_size[0]}x"
                    f"{self.eye_size[1]} (mode {self.mode}). Calibration is "
                    "per-mode; the 4:3 crop moves the principal point.")
        lr = cv2.remap(left, *self._map_l, cv2.INTER_LINEAR)
        rr = cv2.remap(right, *self._map_r, cv2.INTER_LINEAR)
        return lr, rr
