"""Calibration: capture protocol and solve.

Capture quality decides calibration quality. Image-area coverage conditions the
distortion estimate; tilt variety conditions focal length. Both are tracked and
shown so the operator knows when to stop rather than guessing (spec 5.2).
"""
from typing import NamedTuple

import cv2
import numpy as np

from . import board as board_mod
from . import profiles
from .camera import MODES


class DetectedPair(NamedTuple):
    obj: np.ndarray           # (n,3) float32 board coordinates, metres
    img_left: np.ndarray      # (n,2) float32
    img_right: np.ndarray     # (n,2) float32
    corners_left: np.ndarray  # (n,2) float32, for display and sharpness
    corners_right: np.ndarray
    n: int


def detect_pair(detector, board, gray_left, gray_right, min_corners=12):
    """Detect the board in both eyes and keep only corners seen by BOTH.

    Extrinsics are estimated from correspondences, so a corner visible in one
    eye only is useless here; taking the id intersection keeps the object and
    image point arrays aligned.
    """
    cc_l, ci_l, _, _ = detector.detectBoard(gray_left)
    cc_r, ci_r, _, _ = detector.detectBoard(gray_right)
    if cc_l is None or cc_r is None:
        return None
    ids_l = {int(i): p for i, p in zip(ci_l.ravel(), cc_l.reshape(-1, 2))}
    ids_r = {int(i): p for i, p in zip(ci_r.ravel(), cc_r.reshape(-1, 2))}
    shared = sorted(set(ids_l) & set(ids_r))
    if len(shared) < min_corners:
        return None
    all_obj = board.getChessboardCorners()
    obj = np.array([all_obj[i] for i in shared], np.float32)
    pl = np.array([ids_l[i] for i in shared], np.float32)
    pr = np.array([ids_r[i] for i in shared], np.float32)
    return DetectedPair(obj, pl, pr, pl, pr, len(shared))


def sharpness(gray, corners):
    """Scale- and contrast-invariant sharpness over the board region.

    Raw variance-of-Laplacian is NOT comparable across distances: for a
    checkerboard of square size s_px and contrast C blurred by width w, the
    variance goes as C^2 / (w^3 * s_px). So holding the board close inflates
    it, and a running maximum set there then rejects every frame at a normal
    working distance as "blurry" -- which is exactly what happened.

    Dividing out C^2 and multiplying by s_px leaves a quantity depending only
    on blur width, which IS comparable across distances. Same reasoning as the
    edge-rise metric in focus_check.py, but cheap enough to run per frame.
    """
    x0, y0 = np.floor(corners.min(axis=0)).astype(int)
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    patch = gray[y0:y1, x0:x1]
    if patch.size < 64:
        return 0.0
    var = float(cv2.Laplacian(patch, cv2.CV_64F).var())
    contrast = float(patch.std())
    if contrast < 1.0:
        return 0.0
    # Median nearest-neighbour corner spacing: the board's pixel scale.
    pts = np.asarray(corners, float).reshape(-1, 2)
    if len(pts) >= 2:
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        s_px = float(np.median(d.min(axis=1)))
    else:
        s_px = 1.0
    return var * max(s_px, 1.0) / (contrast * contrast)


def pose_delta(a, b):
    """(metres, degrees) between two (rvec, tvec) poses, or None.

    Compare POSES, never corner arrays. `DetectedPair.corners_left` is the
    sorted ID-intersection of the two eyes, so its length and ordering change
    whenever a single edge corner flickers in or out of detection. A
    positional comparison then measures different physical corners against
    each other and reports large motion on a completely stationary board.
    """
    if a is None or b is None:
        return None
    ra, ta = np.asarray(a[0]).reshape(3), np.asarray(a[1]).reshape(3)
    rb, tb = np.asarray(b[0]).reshape(3), np.asarray(b[1]).reshape(3)
    dR = cv2.Rodrigues(cv2.Rodrigues(ra)[0] @ cv2.Rodrigues(rb)[0].T)[0]
    return float(np.linalg.norm(ta - tb)), float(np.degrees(np.linalg.norm(dR)))


def board_tilt_deg(rvec):
    """Angle between the board normal and the camera axis, in degrees."""
    R = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))[0]
    normal = R @ np.array([0.0, 0.0, 1.0])
    cos = abs(float(normal[2]))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class CoverageTracker:
    """Tracks where in the image corners have been seen, and at what tilts."""

    # Distance bins, in metres. Depth variety is what separates eye-to-eye YAW
    # from baseline in stereo calibration: with every view at a similar range
    # the two trade off, and the result is a uniform disparity offset. That
    # offset is invisible to the epipolar gate, which measures ROW alignment
    # while a yaw error shifts disparity in x -- rectification absorbs it into
    # P2 and the rows still align. A 640x480 capture spanning only 0.23-0.52 m
    # passed epipolar at 0.052 px and biased every depth by 2.74%.
    DIST_BINS = (0.25, 0.35, 0.45, 0.55, 0.70, 0.90)

    def __init__(self, eye_size, grid=(4, 3), tilt_bins=4, target_pairs=50,
                 max_tilt_deg=50.0):
        self.eye_w, self.eye_h = eye_size
        self.grid = grid
        self.tilt_bins = tilt_bins
        self.target_pairs = target_pairs
        self.max_tilt_deg = max_tilt_deg
        self._cells = np.zeros(grid[0] * grid[1], bool)
        self._tilts = np.zeros(tilt_bins, bool)
        self._dists = np.zeros(len(self.DIST_BINS), bool)
        self.n_pairs = 0

    def _dist_bin(self, dist_m):
        return int(np.argmin([abs(dist_m - b) for b in self.DIST_BINS]))

    def add(self, corners, tilt_deg, dist_m=None):
        gx, gy = self.grid
        pts = np.asarray(corners, float).reshape(-1, 2)
        cx = np.clip((pts[:, 0] / self.eye_w * gx).astype(int), 0, gx - 1)
        cy = np.clip((pts[:, 1] / self.eye_h * gy).astype(int), 0, gy - 1)
        self._cells[cy * gx + cx] = True
        b = int(np.clip(tilt_deg / self.max_tilt_deg * self.tilt_bins,
                        0, self.tilt_bins - 1))
        self._tilts[b] = True
        if dist_m is not None and self.DIST_BINS[0] - 0.08 <= dist_m <= self.DIST_BINS[-1] + 0.15:
            self._dists[self._dist_bin(dist_m)] = True
        self.n_pairs += 1

    def fov_coverage(self):
        return float(self._cells.mean())

    def tilt_coverage(self):
        return float(self._tilts.mean())

    def dist_coverage(self):
        return float(self._dists.mean())

    def missing_distances(self):
        return [b for b, seen in zip(self.DIST_BINS, self._dists) if not seen]

    def done(self):
        return (self.n_pairs >= self.target_pairs
                and self.fov_coverage() == 1.0
                and self.tilt_coverage() == 1.0
                and self.dist_coverage() == 1.0)


import json
from pathlib import Path


def _as_lists(a):
    return np.asarray(a).tolist()


def solve(pairs, eye_size, mode, square_mm, holdout_frac=0.2, seed=0,
          profile=profiles.DEFAULT_PROFILE):
    """Calibrate from detected pairs and rectify.

    Per-eye intrinsics first, then stereoCalibrate, which either refines them
    jointly with the extrinsics or holds them fixed depending on the camera
    (spec 5.3, CameraProfile.joint_intrinsics). A fraction of pairs is held
    out and never seen by the solver, because residual on the fitting set
    measures fit, not accuracy.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_hold = max(1, int(round(holdout_frac * len(pairs))))
    hold_idx, fit_idx = idx[:n_hold], idx[n_hold:]
    fit = [pairs[i] for i in fit_idx]
    hold = [pairs[i] for i in hold_idx]

    obj = [p.obj for p in fit]
    pl = [p.img_left for p in fit]
    pr = [p.img_right for p in fit]

    rms_l, Kl, Dl, _, _ = cv2.calibrateCamera(obj, pl, eye_size, None, None)
    rms_r, Kr, Dr, _, _ = cv2.calibrateCamera(obj, pr, eye_size, None, None)

    # Intrinsics are REFINED here, not frozen. CALIB_FIX_INTRINSIC is the
    # common tutorial default, but it leaves each eye at whatever point the
    # monocular solve happened to pick in the focal/principal-point/distortion
    # degeneracy. Both eyes then fit their own data while disagreeing with
    # each other, and a rigid (R, T) has no freedom to reconcile that -- the
    # mismatch comes out as epipolar error instead. Measured on the AR0144 at
    # 2560x720: held-out epipolar 0.471 px frozen, 0.113 px refined, with
    # triangulated-scale spread 8.5x tighter and the recovered baseline moving
    # toward the value measured independently in other modes.
    # Whether intrinsics are refined here or frozen is per-camera (see
    # CameraProfile.joint_intrinsics) and must be settled by a far sweep --
    # epipolar error improves under joint refinement on both cameras while
    # one of them loses 4% of its depth scale.
    #
    # flags must be passed EXPLICITLY. OpenCV 5 defaults this argument to
    # CALIB_FIX_INTRINSIC, where 4.x defaulted it to 0, so simply omitting it
    # silently freezes the intrinsics.
    # OpenCV 5 returns 9 values here; 4.x examples show 8.
    stereo_flags = 0 if profile.joint_intrinsics else cv2.CALIB_FIX_INTRINSIC
    ret = cv2.stereoCalibrate(obj, pl, pr, Kl, Dl, Kr, Dr, eye_size,
                              flags=stereo_flags)
    rms_s, Kl, Dl, Kr, Dr, R, T = ret[:7]

    # alpha=0 crops to valid pixels, so the RGBD map has no black border
    # (spec 6). This changes the effective FOV and intrinsics, so the
    # RECTIFIED P1/P2/Q are what the API reports.
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        Kl, Dl, Kr, Dr, eye_size, R, T, alpha=0)

    calib = {
        # Which camera this came from, so every consumer can pick the right
        # profile without being told. Calibration is per-camera AND per-mode,
        # and both cameras define 1280x480 with the same eye size.
        "camera": profile.name,
        "mode": mode,
        "eye_size": [int(eye_size[0]), int(eye_size[1])],
        "square_mm": float(square_mm),
        "board": dict(board_mod.BOARD),
        # ravel(): OpenCV returns dist as (1,5) or (5,1) depending on call
        # path, and a nested list would break every consumer that does
        # np.array(...).ravel() expecting 5 coefficients.
        "left": {"K": _as_lists(Kl), "dist": np.asarray(Dl).ravel().tolist()},
        "right": {"K": _as_lists(Kr), "dist": np.asarray(Dr).ravel().tolist()},
        "R": _as_lists(R),
        "T": _as_lists(T),
        "baseline_m": float(np.linalg.norm(T)),
        "rectified": {"R1": _as_lists(R1), "R2": _as_lists(R2),
                      "P1": _as_lists(P1), "P2": _as_lists(P2),
                      "Q": _as_lists(Q),
                      "roi_left": [int(v) for v in roi1],
                      "roi_right": [int(v) for v in roi2]},
        "metrics": {
            "rms_left": float(rms_l), "rms_right": float(rms_r),
            "rms_stereo": float(rms_s),
            "n_pairs": len(fit), "n_holdout": len(hold),
            "f_recovered": float(Kl[0, 0]),
            # None when the vendor publishes no usable FOV figure. The
            # f-vs-prior gate then has nothing to compare against and skips,
            # rather than failing a good calibration against an invented
            # number (profiles.py).
            "f_nominal": (float(f_nom) if (f_nom := profile.modes[mode].f_nominal)
                          is not None else None),
        },
    }
    calib["metrics"]["epipolar_px"] = epipolar_error(hold, calib)
    # T[0] must be negative: the right camera sits to the RIGHT of the left
    # one. A positive value means the halves are reversed, which inverts
    # disparity. SGBM only searches positive disparity, so it then returns
    # spurious matches -- sparse, noisy depth with a large apparent scale
    # error, rather than an obvious failure. Caught here because export.py's
    # P2[0,3] assertion is downstream of anyone actually using the depth.
    calib["eyes_reversed"] = bool(np.asarray(T).ravel()[0] > 0)
    return calib


def epipolar_error(pairs, calib):
    """Mean |y_left - y_right| after rectification, in pixels.

    This is the quantity disparity actually depends on. A calibration can show
    a flattering reprojection error and still be useless for stereo (spec 5.4).
    """
    if not pairs:
        return float("nan")
    Kl = np.array(calib["left"]["K"])
    Dl = np.array(calib["left"]["dist"]).ravel()
    Kr = np.array(calib["right"]["K"])
    Dr = np.array(calib["right"]["dist"]).ravel()
    rec = calib["rectified"]
    R1, P1 = np.array(rec["R1"]), np.array(rec["P1"])
    R2, P2 = np.array(rec["R2"]), np.array(rec["P2"])
    errs = []
    for p in pairs:
        ul = cv2.undistortPoints(p.img_left.reshape(-1, 1, 2), Kl, Dl,
                                 R=R1, P=P1).reshape(-1, 2)
        ur = cv2.undistortPoints(p.img_right.reshape(-1, 1, 2), Kr, Dr,
                                 R=R2, P=P2).reshape(-1, 2)
        errs.append(np.abs(ul[:, 1] - ur[:, 1]))
    return float(np.concatenate(errs).mean())


# Spec 5.4. The epipolar and baseline checks are the real gates.
ACCEPTANCE = {
    "epipolar_px": 0.3,
}

# Reprojection RMS is gated ANGULARLY (rms / f), not in absolute pixels.
# A pixel limit is resolution-dependent: the same physical corner accuracy
# yields proportionally more pixels of error at higher resolution, so a
# fixed 0.3 px threshold silently penalises the better mode. Measured across
# four calibrations the angular figure is stable at 311-371 urad while the
# pixel figure ranges 0.22-0.37, and the mode the pixel gate REJECTED was
# angularly the best of the set.
#
# The limit is the original 0.3 px expressed at the AR0144's nominal
# f = 669.7 px, so the gate is unchanged for the camera it was tuned on.
RMS_ANGULAR_URAD = 448.0
STEREO_RMS_ANGULAR_URAD = RMS_ANGULAR_URAD * 0.5 / 0.3
BASELINE_TOLERANCE = 0.05          # 5%
F_TOLERANCE = 0.10                 # 10%: closest pair of mode f_nominal values
                                    # differ by 25%, so this separates modes
                                    # cleanly while tolerating prior rounding


def check_acceptance(calib, profile=None):
    """Return a list of failed criteria. Empty means the calibration passes."""
    failures = []
    if calib.get("eyes_reversed"):
        failures.append(
            "eyes are REVERSED (T[0] > 0, P2[0,3] positive): this device "
            "presents [right|left]. Set swap_eyes=True on its profile and "
            "re-capture, or disparity is inverted and depth is nonsense.")
    m = calib["metrics"]
    # Angular reprojection gates.
    # Tolerate a dict without intrinsics (synthetic/partial): the angular
    # gate simply does not apply, rather than raising.
    try:
        f_px = float(np.array(calib["left"]["K"])[0][0])
    except (KeyError, TypeError, IndexError):
        f_px = 0.0
    if f_px > 0:
        for key, limit in (("rms_left", RMS_ANGULAR_URAD),
                           ("rms_right", RMS_ANGULAR_URAD),
                           ("rms_stereo", STEREO_RMS_ANGULAR_URAD)):
            v = m.get(key)
            if v is None or not np.isfinite(v):
                failures.append(f"{key} missing")
                continue
            urad = v / f_px * 1e6
            m[f"{key}_urad"] = urad
            if urad >= limit:
                failures.append(
                    f"{key}={v:.4f}px = {urad:.0f}urad (limit <{limit:.0f}urad, "
                    f"i.e. {limit * f_px / 1e6:.3f}px at this resolution)")
    for key, limit in ACCEPTANCE.items():
        value = m.get(key)
        if value is None or not np.isfinite(value) or value >= limit:
            failures.append(f"{key}={value:.4f} (limit <{limit})")
    # Only gate the baseline when one is actually known for THIS camera.
    # Hardcoding 52 mm rejected the SKL's perfectly good 16.98 mm calibration
    # as "67% off, suspect print scaling" -- the AR0144's number leaking into
    # a different camera.
    nominal = getattr(profile, "baseline_nominal_m", None) if profile else None
    if nominal:
        rel = abs(calib["baseline_m"] - nominal) / nominal
        if rel > BASELINE_TOLERANCE:
            failures.append(
                f"baseline={calib['baseline_m']*1000:.2f}mm vs nominal "
                f"{nominal*1000:.1f}mm ({rel*100:.1f}% off) — suspect print "
                "scaling or board flatness")
    f_nom = calib["metrics"].get("f_nominal")
    f_rec = calib["metrics"].get("f_recovered")
    if f_nom and f_rec:
        rel_f = abs(f_rec - f_nom) / f_nom
        if rel_f > F_TOLERANCE:
            failures.append(
                f"f={f_rec:.1f}px vs nominal {f_nom:.1f}px for mode "
                f"{calib['mode']} ({rel_f*100:.1f}% off) — suspect the wrong "
                "capture mode; the 4:3 modes crop, so f does not scale with width")
    return failures


def save(calib, path):
    Path(path).write_text(json.dumps(calib, indent=2))


def load(path):
    calib = json.loads(Path(path).read_text())
    return calib


def measure_colour_gains(frames, swap=False):
    """Per-channel gains mapping the RIGHT eye onto the LEFT.

    The two sensors differ by a global per-channel gain plus a smaller
    spatial term. On the SKL the global part is R x1.05, G x0.93, B x0.95 --
    a 13.7% difference in R/G, which reads as one eye being visibly warmer.
    Residual spatial variation is 2-4% and would need a per-pixel flat field;
    not worth it for the gain.

    Any scene works: the goal is making the eyes AGREE, not absolute colour,
    so the ratio is valid whatever they are both looking at. Uses the median
    across frames so a transient highlight cannot skew it.
    """
    from .camera import split_eyes
    ratios = []
    for f in frames:
        left, right = split_eyes(f, swap)
        h, w = left.shape[:2]
        sl = left[h // 4:3 * h // 4, w // 4:3 * w // 4].astype(np.float64)
        sr = right[h // 4:3 * h // 4, w // 4:3 * w // 4].astype(np.float64)
        per = []
        for ch in range(3):
            a, b = sl[:, :, ch].mean(), sr[:, :, ch].mean()
            per.append(a / b if b > 1e-6 else 1.0)
        ratios.append(per)
    return np.median(np.asarray(ratios), axis=0).tolist()   # B, G, R


def apply_colour_gains(right, gains):
    """Bring the right eye onto the left eye's colour. Returns a new array."""
    if gains is None:
        return right
    out = right.astype(np.float32)
    for ch, g in enumerate(gains):
        out[:, :, ch] *= g
    return np.clip(out, 0, 255).astype(np.uint8)
