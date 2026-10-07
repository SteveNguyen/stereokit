"""Per-camera profiles, so one pipeline serves several UVC side-by-side
stereo modules.

Everything else in this package is camera-agnostic already: the board, the
solver, rectification, SGBM and the export all work from the calibration, not
from the camera. Only the mode table and the device's identity differ.

Identity matters more than it looks. The AR0144 and the SKL both advertise
1280x480, 2560x720 and 1600x600, so scanning /dev/video* for "the first node
that accepts this size" would bind whichever enumerates first when both are
plugged in — silently, because both genuinely deliver the frame. Profiles are
therefore matched on the V4L2 card name before any mode is requested.
"""
import math
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np


class Mode(NamedTuple):
    name: str
    combined_w: int
    combined_h: int
    eye_w: int
    eye_h: int
    fps: int
    # Prior from a datasheet FOV figure, never a substitute for calibration.
    # None when the vendor gives no usable figure — the f-vs-prior acceptance
    # gate then has nothing to compare against and is skipped rather than
    # failing a good calibration against an invented number.
    f_nominal: Optional[float] = None


class CameraProfile(NamedTuple):
    name: str
    card_match: str          # substring of the V4L2 card name
    modes: dict
    default_mode: str
    notes: str = ""
    # Physical lens focal length in mm, when the vendor states one. Alone this
    # cannot give focal length in PIXELS -- that needs the sensor's pixel
    # pitch, which these vendors never publish. Its use is the inverse: once
    # calibration measures f_px, this yields the implied pitch, and an
    # implausible pitch means the calibration or the mode is wrong.
    lens_mm: Optional[float] = None
    # Native per-eye width in px, for relating a mode's f back to the sensor.
    native_eye_w: Optional[int] = None
    # Expected baseline in metres, when it is actually known. None means the
    # acceptance gate has nothing to compare against and skips -- the SKL's
    # was unknown until calibration measured it, which was the whole point of
    # calibrating. Never seed this from the run you are about to gate.
    baseline_nominal_m: Optional[float] = None
    # True when the device presents its halves as [right|left]. The SKL does.
    # Left uncorrected this inverts the sign of disparity, and since SGBM only
    # searches positive disparity it then latches onto spurious matches --
    # producing sparse, noisy depth with a large apparent scale error rather
    # than an obvious failure.
    swap_eyes: bool = False
    # Let stereoCalibrate refine intrinsics jointly instead of freezing the
    # per-eye monocular fit. Pinning down focal length this way needs stereo
    # leverage: with a long baseline the two eyes see genuinely different
    # views and f is well constrained, but with a short one they see nearly
    # the same view and f can drift. Measured on the far sweep, as f*B scale
    # error against PnP distance out to 3.8 m:
    #   AR0144 (52 mm baseline)  frozen +5.69%   joint -0.17%
    #   SKL    (17 mm baseline)  frozen -1.44%   joint -5.57%
    # So this is per-camera and must be validated against a far sweep, not
    # assumed. Epipolar error does NOT settle it -- joint refinement improves
    # epipolar on both cameras while wrecking the SKL's scale.
    joint_intrinsics: bool = False


# --- Waveshare AR0144 Stereo USB Camera (A) ---------------------------------
# The 4:3 modes are a horizontal CROP (960 of 1280 reference columns), not a
# downscale, which is why f does not simply scale with width. f_nominal is
# derived from the vendor's 65 deg HFOV; calibration later measured 636.5 px
# raw at 1280x480, so the figures are good to about 5%.
AR0144_MODES = {
    "2560x720": Mode("2560x720", 2560, 720, 1280, 720, 30, 1004.6),
    "1600x600": Mode("1600x600", 1600, 600, 800, 600, 30, 837.2),
    "1280x480": Mode("1280x480", 1280, 480, 640, 480, 60, 669.7),
    "1280x360": Mode("1280x360", 1280, 360, 640, 360, 120, 502.3),
}

AR0144 = CameraProfile(
    name="ar0144",
    card_match="CCB Camera",
    modes=AR0144_MODES,
    default_mode="1280x480",
    notes="Global shutter, 52 mm baseline (measured 51.92), colour.",
    lens_mm=2.88,
    native_eye_w=1280,
    baseline_nominal_m=0.052,
    joint_intrinsics=True,
)


# --- SKL-4689-220 / SKL-2MP-220 (SunplusIT SPCA2100) ------------------------
# No usable vendor documentation: the supplied PDF is for a different model
# (SKL-2MP-220), claims 2MP against a 3.3MP device, and never mentions stereo.
# Every mode below was measured by correlating the frame halves; each was
# confirmed side-by-side stereo at 0.98-0.99 correlation with a small offset.
# f_nominal is unknown for all of them because no FOV figure exists; the
# AliExpress baseline claim of 20 mm is unverified and looks closer to 16.
# Capped at 30 fps in every mode, unlike the AR0144.
SKL2MP220_MODES = {
    "3040x1080": Mode("3040x1080", 3040, 1080, 1520, 1080, 30),
    "2560x960": Mode("2560x960", 2560, 960, 1280, 960, 30),
    "2560x720": Mode("2560x720", 2560, 720, 1280, 720, 30),
    "2176x1520": Mode("2176x1520", 2176, 1520, 1088, 1520, 30),
    "1920x1280": Mode("1920x1280", 1920, 1280, 960, 1280, 30),
    "1600x600": Mode("1600x600", 1600, 600, 800, 600, 30),
    "1440x1280": Mode("1440x1280", 1440, 1280, 720, 1280, 30),
    "1280x480": Mode("1280x480", 1280, 480, 640, 480, 30),
    "1200x800": Mode("1200x800", 1200, 800, 600, 800, 30),
    "960x640": Mode("960x640", 960, 640, 480, 640, 30),
}

SKL2MP220 = CameraProfile(
    name="skl2mp220",
    card_match="SPCA2100",
    modes=SKL2MP220_MODES,
    default_mode="1280x480",
    notes="GLOBAL SHUTTER (measured, 955 frames). "
          "Baseline MEASURED 16.98 mm (vendor claimed 20, 18% high). "
          "Implied pixel pitch 2.07 um from the 2.8 mm lens, so a ~3.1 mm "
          "native sensor width. 30 fps cap. "
          "Eyes agree to ~0.2% in R/G under auto or 4600K white balance, but "
          "differ ~3-7% in brightness intrinsically, and the gap opens sharply "
          "at extreme WB (+64% R/G locked at 2800K) because their spectral "
          "responses differ. FLARE-PRONE: a bright source in frame throws "
          "asymmetric veiling glare across one eye -- measured as a spurious "
          "19% colour difference before the source was moved out of frame. "
          "Global normalisation cannot correct that, as flare varies spatially.",
    lens_mm=2.8,          # the ONLY figure the vendor supplies
    native_eye_w=1520,    # largest per-eye width offered (3040x1080)
    # Vendor claims 20 mm, 18% high. Two INDEPENDENT calibrations now agree:
    # 16.9837 mm at 1280x480 and 16.9925 mm at 1600x600, different capture
    # sessions and different modes, 0.05% apart. Deliberately not seeded from
    # a single run -- gating a calibration against a figure derived from that
    # same run proves nothing -- but two agreeing runs earn it.
    baseline_nominal_m=0.016988,
    swap_eyes=True,
)


PROFILES = {p.name: p for p in (AR0144, SKL2MP220)}
DEFAULT_PROFILE = AR0144


def get(name):
    """Look up a profile by name. KeyError on an unknown one, by design."""
    return PROFILES[name]


def video_nodes():
    """[(index, card_name)] for every /dev/videoN, lowest index first.

    Read from sysfs because OpenCV does not expose the card name, and the
    card name is what tells two cameras apart before either is opened.
    """
    out = []
    for p in sorted(Path("/sys/class/video4linux").glob("video*"),
                    key=lambda q: int(q.name.removeprefix("video"))):
        try:
            out.append((int(p.name.removeprefix("video")),
                        (p / "name").read_text().strip()))
        except (OSError, ValueError):
            continue
    return out


def candidate_indices(profile):
    """Video node indices whose card name matches this profile.

    Returns EMPTY when sysfs is readable and nothing matches — that means the
    camera is not attached, and offering the other nodes instead is how you
    bind the wrong camera. An earlier version fell back to every node here,
    and with only the SKL plugged in the AR0144 profile duly bound the SKL:
    both accept 1280x480, so the frame-shape guard had nothing to object to.

    The fallback applies only when sysfs itself is unreadable, where the
    frame-shape guard in camera.py is the remaining backstop.
    """
    nodes = video_nodes()
    if not nodes:
        return list(range(10))
    return [i for i, name in nodes
            if profile.card_match.lower() in name.lower()]


# Assumed horizontal field when a profile has no vendor figure. Only ever used
# for the provisional camera matrix below.
PROVISIONAL_HFOV_DEG = 64.0


def provisional_f(mode):
    """A focal length good enough to estimate board TILT during capture.

    solvePnP needs some camera matrix, but `capture` uses that pose only to bin
    tilt into four coarse buckets for coverage feedback. It never reaches the
    solve, which works from the real board geometry. So when a profile has no
    vendor FOV figure, assume a typical field rather than crash -- which is
    what happened when f_nominal became optional and a None went straight into
    np.array(), producing a dtype=object matrix that solvePnP rejected.
    """
    if mode.f_nominal is not None:
        return float(mode.f_nominal)
    return mode.eye_w / (2.0 * math.tan(math.radians(PROVISIONAL_HFOV_DEG / 2.0)))


def provisional_K(mode):
    """Provisional intrinsics for capture-time tilt estimation only."""
    f = provisional_f(mode)
    return np.array([[f, 0.0, mode.eye_w / 2.0],
                     [0.0, f, mode.eye_h / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


# Plausible pixel pitch for a modern small CMOS sensor, in micrometres.
PITCH_RANGE_UM = (1.0, 6.0)


def implied_pixel_pitch_um(profile, f_px_native):
    """Pixel pitch implied by a measured focal length, in micrometres.

    f_px = f_mm / pitch, so pitch = f_mm / f_px. Pass the focal length scaled
    to the profile's NATIVE per-eye width, not whatever mode was calibrated --
    a downscaled mode has a proportionally smaller f and would imply a
    correspondingly wrong pitch.

    Returns None when the vendor gave no lens figure.
    """
    if profile.lens_mm is None or not f_px_native:
        return None
    return profile.lens_mm * 1000.0 / f_px_native


def pitch_is_plausible(pitch_um):
    """Whether an implied pitch is physically sane for this class of sensor.

    This is the only wrong-mode sanity check available for a profile with no
    vendor FOV figure, so it is weak but not worthless: a badly wrong f lands
    far outside the range.
    """
    if pitch_um is None:
        return None
    lo, hi = PITCH_RANGE_UM
    return lo <= pitch_um <= hi
