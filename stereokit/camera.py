"""Capture from the AR0144 stereo module.

The module is a single UVC node emitting one concatenated [left|right] frame,
so both eyes arrive in the same buffer and are hardware frame-synchronised.
Two hazards drive this code: the driver silently returns ONE eye if a
single-width size is requested, and UVC control values persist on the device
after the process exits.
"""
import time

import cv2
import numpy as np

from . import profiles
from .profiles import DEFAULT_PROFILE, Mode   # re-exported: Mode moved to
                                              # profiles.py so the profile
                                              # table can own it

# The default profile's modes, re-exported so existing callers and tests that
# reach for camera.MODES keep working unchanged.
MODES = DEFAULT_PROFILE.modes
DEFAULT_MODE = DEFAULT_PROFILE.default_mode

# UVC control values live on the DEVICE and persist after the process exits,
# so a tool that changes them silently reconfigures every program run
# afterwards (spec 2.3).
SAVED_PROPS = (cv2.CAP_PROP_AUTO_EXPOSURE, cv2.CAP_PROP_AUTO_WB,
               cv2.CAP_PROP_EXPOSURE, cv2.CAP_PROP_GAIN)
DEFAULT_PROPS = {cv2.CAP_PROP_AUTO_EXPOSURE: 3.0,   # 3 = aperture priority
                 cv2.CAP_PROP_AUTO_WB: 1.0,
                 cv2.CAP_PROP_EXPOSURE: 166.0,
                 cv2.CAP_PROP_GAIN: 32.0}


def feature_score(gray):
    """Feature yield of a frame: mean |horizontal gradient|.

    Cheap stand-in for "how many keypoints will a tracker find". Correlates
    with ORB count and, unlike mean brightness, actually reflects what SLAM
    needs.
    """
    return float(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).mean())


def tune_exposure(cap, mode, settle=8):
    """Exposure that maximises FEATURE YIELD, not brightness.

    Auto-exposure meters for a pleasing picture, which for SLAM is the wrong
    target: a long exposure brightens the frame but clips highlights, and the
    clipped regions carry no gradient. Measured on the SKL, going from 20 ms
    to 180 ms raised mean brightness 64 -> 202 while ORB keypoints collapsed
    181 -> 14, because blown pixels went 0.4% -> 10%.

    Candidates are capped at the frame period, since a longer exposure
    throttles the sensor and costs frame rate on top of features.
    """
    max_exp = int(1000.0 / mode.fps * 10)          # 100us units
    candidates = [e for e in (50, 80, 120, 160, 200, 250, 300, 400, 600)
                  if e <= max_exp] or [max_exp]
    # Below this the frame carries no usable structure at any exposure, and
    # the sweep is choosing between kinds of nothing.
    MIN_USEFUL_GRADIENT = 0.5
    best, best_score = candidates[0], -1.0
    for e in candidates:
        cap.set(cv2.CAP_PROP_EXPOSURE, e)
        for _ in range(settle):
            cap.read()
        ok, frame = cap.read()
        if not ok:
            continue
        left, _ = split_eyes(frame)
        score = feature_score(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY))
        if score > best_score:
            best, best_score = e, score
    if best_score < MIN_USEFUL_GRADIENT:
        # Every candidate was featureless, so the winner is meaningless and
        # candidates[0] is the SHORTEST, i.e. the darkest. Fall back to the
        # longest instead and say so: the scene is too dark to tune in, and
        # silently picking the darkest option makes it worse.
        best = candidates[-1]
    return best, best_score


def split_eyes(frame, swap=False):
    """Split the concatenated frame into (left, right).

    `swap` for devices that present [right|left]; see CameraProfile.swap_eyes.
    """
    mid = frame.shape[1] // 2
    a, b = frame[:, :mid], frame[:, mid:]
    return (b, a) if swap else (a, b)


def restore_controls(cap, saved):
    """Put device controls back as found. Auto flags LAST, or the manual
    writes that follow would silently turn auto off again."""
    for prop in reversed(SAVED_PROPS):
        value = saved.get(prop)
        if value is None or value < 0:
            value = DEFAULT_PROPS[prop]
        cap.set(prop, value)


def _accept(cap, combined_w, combined_h):
    """True if this node really delivers the full double-width frame.

    Checks the returned FRAME, not just the negotiated properties: 1280 is the
    most common webcam width, so a width-only check would bind a mono camera
    and hand back two halves of one image as a 'stereo pair' (spec 2.1).
    """
    if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) != combined_w
            or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) != combined_h):
        cap.release()
        return False
    ok, frame = cap.read()
    if not ok or frame.shape[:2] != (combined_h, combined_w):
        cap.release()
        return False
    return True


class StereoCamera:
    # Frames to let auto-exposure converge before freezing it.
    SETTLE_FRAMES = 45

    def __init__(self, mode=None, device=None, lock_exposure=True,
                 exposure=None, gain=None, profile=DEFAULT_PROFILE):
        if isinstance(profile, str):
            profile = profiles.get(profile)
        self.profile = profile
        mode = profile.default_mode if mode is None else mode
        # KeyError on an unknown mode, by design.
        self.mode = profile.modes[mode]
        self._device = device
        self._lock_exposure = lock_exposure
        self._exposure = exposure
        self._gain = gain
        self._cap = None
        self._saved = {}

    @property
    def eye_size(self):
        return (self.mode.eye_w, self.mode.eye_h)

    def open(self):
        m = self.mode
        # Match on the V4L2 card name BEFORE requesting a mode: two different
        # stereo cameras can advertise the same frame size, so a blind scan
        # would bind whichever enumerates first (profiles.py).
        indices = ([self._device] if self._device is not None
                   else profiles.candidate_indices(self.profile))
        logging = cv2.utils.logging
        previous = logging.getLogLevel()
        logging.setLogLevel(logging.LOG_LEVEL_ERROR)   # mute metadata nodes
        try:
            for index in indices:
                cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
                if not cap.isOpened():
                    continue
                # FOURCC before size, or V4L2 ignores it. MJPG is mandatory:
                # YUYV at 2560x720 caps at 5 fps on USB 2.0 (spec 2.2).
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, m.combined_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, m.combined_h)
                cap.set(cv2.CAP_PROP_FPS, m.fps)
                if not _accept(cap, m.combined_w, m.combined_h):
                    continue
                self._cap = cap
                self._saved = {p: cap.get(p) for p in SAVED_PROPS}
                if self._lock_exposure:
                    # Locked for SLAM: auto-exposure drifting between frames
                    # breaks feature tracking and loop closure (spec 2.3).
                    # What matters is that it is FIXED, not what value it is
                    # fixed at -- so let auto converge on this room first and
                    # freeze whatever it chose. A hardcoded value is wrong in
                    # every room but the one it was tuned in: 300 was picked
                    # for the AR0144 and left the SKL at mean brightness 49
                    # with two thirds of the frame below 60.
                    if self._exposure == "features":
                        # Optimise for what SLAM actually consumes.
                        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                        converged, score = tune_exposure(cap, m)
                        if score < 0.5:
                            print(f"warning: no usable gradient at ANY "
                                  f"exposure (best {score:.2f}) — the scene is "
                                  f"too dark to tune in. Falling back to "
                                  f"{converged/10:.0f} ms. Turn the lights on.")
                        else:
                            print(f"exposure tuned to {converged/10:.0f} ms "
                                  f"(gradient {score:.2f}), keeping {m.fps} fps")
                    elif self._exposure is None:
                        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
                        for _ in range(self.SETTLE_FRAMES):
                            cap.read()
                        converged = cap.get(cv2.CAP_PROP_EXPOSURE)
                    else:
                        converged = self._exposure
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # 1 = manual
                    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                    if converged and converged > 0:
                        cap.set(cv2.CAP_PROP_EXPOSURE, converged)
                        # UVC exposure is in 100us units. An exposure longer
                        # than the frame period throttles the sensor, so a dim
                        # room silently costs frame rate: on the SKL, auto
                        # converged to 1200 (120 ms) and dropped 30 fps to 8.3.
                        # Warn rather than cap -- capping would just make the
                        # image dark without saying why, and the real fix is
                        # light.
                        exp_ms = converged / 10.0
                        frame_ms = 1000.0 / m.fps
                        if exp_ms > frame_ms:
                            print(f"warning: exposure {exp_ms:.0f} ms exceeds "
                                  f"the {frame_ms:.0f} ms frame period, so "
                                  f"{m.fps} fps will throttle to about "
                                  f"{1000.0 / exp_ms:.0f} fps. More light "
                                  f"would recover it.")
                    if self._gain is not None:
                        cap.set(cv2.CAP_PROP_GAIN, self._gain)
                    self.locked_exposure = converged
                return self
        finally:
            logging.setLogLevel(previous)
        raise RuntimeError(
            f"No {self.profile.name} camera delivering "
            f"{m.combined_w}x{m.combined_h} (card match "
            f"{self.profile.card_match!r}). Is it plugged in?")

    def grab(self, retries=3):
        """Return (left, right, timestamp).

        Timestamp is taken immediately after read() returns. OpenCV does not
        expose V4L2 kernel buffer timestamps, so this carries a few ms of USB
        and MJPEG-decode latency with jitter (spec 11).

        Transient read failures are routine on USB 2.0 MJPG at 60 fps, so a
        single dropped frame must not abort a recording session.
        """
        for attempt in range(retries):
            ok, frame = self._cap.read()
            ts = time.monotonic()
            if ok:
                left, right = split_eyes(frame, self.profile.swap_eyes)
                return left, right, ts
        raise RuntimeError(f"Frame grab failed after {retries} attempts")

    def close(self):
        if self._cap is not None:
            restore_controls(self._cap, self._saved)
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False
