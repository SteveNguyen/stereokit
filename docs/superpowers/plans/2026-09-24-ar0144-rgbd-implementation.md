# AR0144 Stereo → Depth → RGBD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Waveshare AR0144 stereo USB camera into a calibrated metric depth source feeding RTAB-Map standalone for SLAM and a debug visual RGBD output for manipulation perception.

**Architecture:** A small Python package `ar0144/` with one responsibility per module: camera I/O, board definition, calibration capture+solve, rectification, SGBM depth, the public RGBD API, and ROS-format calibration export. A thin `cli.py` wires them into subcommands. Everything that can be tested without hardware is tested against synthetic data generated from known ground truth; hardware steps are called out explicitly as manual verification.

**Tech Stack:** Python ≥3.10, OpenCV 5.0.0 (`opencv-python`), numpy ≥2, Pillow (board PDF), pytest, `uv` for dependency management.

**Spec:** `docs/superpowers/specs/2026-09-22-ar0144-rgbd-design.md` — read it alongside this plan. Every threshold and magic number here traces to a numbered section there.

## Global Constraints

Every task's requirements implicitly include all of these.

- **Mode is `1280x480`** (640×480 per eye, 60 fps). Nominal f ≈ 670 px, HFOV 51.1°, VFOV 39.4°. Spec §2.1, §3.1.
- **MJPG is mandatory.** YUYV at 2560×720 caps at 5 fps on USB 2.0. `CAP_PROP_FOURCC` must be set *before* frame size or V4L2 ignores it. Spec §2.2.
- **Requesting a single-width size silently returns ONE eye.** Every capture path asserts the returned width equals the requested combined width. Spec §2.1.
- **Calibration is per-mode.** The 4:3 modes crop, which moves the principal point, so a calibration captured at one mode must never be loaded against another. This is an error, not a warning. Spec §5.4.
- **Baseline nominal 52 mm.** Depth `Z = f·B/d`. Clamp depth to **[0.18, 4.0] m**. `numDisparities = 192`. Spec §3.2.
- **Board: 7×10 squares, 25 mm square, 18.75 mm marker, `DICT_4X4_50`, 54 corners, 35 markers.** `ar0144/board.py` is the single source of truth — never restate these numbers anywhere else. Spec §5.1.
- **UVC control values persist on the device** after the process exits. Any code that changes exposure/gain/WB must save state on open and restore on exit, restoring the *auto* flags last. Spec §2.3.
- **The eyes are NOT photometrically identical** (~6% R/G difference under auto WB, ~22% with WB locked). Stereo matching runs on grayscale with per-eye mean/variance normalisation. Spec §2.3.
- **`cv2.stereoCalibrate` returns 9 values in OpenCV 5**, not the 8 that 4.x examples show. Unpack `ret[:7]`.
- **numpy ≥2 raises on `float()` of a size-1 1-D array.** Use `.item()`. This already bit once.
- **No ROS 2.** Consumers are RTAB-Map standalone and the debug RGBD output. Spec §1.
- Acceptance thresholds (spec §5.4): per-eye RMS < 0.3 px, stereo RMS < 0.5 px, post-rectification epipolar error < 0.3 px, recovered baseline within a few % of 52 mm.

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, pytest config |
| `ar0144/__init__.py` | Package marker, version |
| `ar0144/board.py` | **Moved from `./board.py`.** ChArUco geometry + printable PDF. Single source of truth. |
| `ar0144/camera.py` | Mode table, node detection, MJPG, width guard, UVC control save/restore, `grab()` |
| `ar0144/calibrate.py` | `CoverageTracker`, pair detection, `solve()`, acceptance metrics |
| `ar0144/rectify.py` | Loads `calibration.json`, builds maps, exposes rectified K/Q/baseline |
| `ar0144/depth.py` | Photometric normalisation, SGBM, disparity → metric depth |
| `ar0144/rgbd.py` | `RGBDFrame`, `StereoRGBD` public API, TUM-format save/load |
| `ar0144/export.py` | `calibration.json` → ROS `camera_info` YAML for RTAB-Map |
| `ar0144/cli.py` | `ar0144 {board,capture,solve,view,record,export}` |
| `tests/synthetic.py` | Shared ground-truth generator for offline calibration tests |
| `tests/test_*.py` | One per module |

Kept at top level: `focus_check.py` (spec phase 0, done — its `import board` becomes `from stereokit import board`). `stereo_view.py` is superseded by `ar0144 view --raw` in Task 10.

---

### Task 1: Package scaffold, move board.py, board tests

**Files:**
- Create: `pyproject.toml`, `ar0144/__init__.py`, `tests/test_board.py`
- Move: `board.py` → `ar0144/board.py` (use `git mv`)
- Modify: `focus_check.py` (import path only)

**Interfaces:**
- Consumes: nothing.
- Produces: `ar0144.board.BOARD` (dict with keys `squares_x`, `squares_y`, `square_mm`, `marker_mm`, `dictionary`), `ar0144.board.make_board(square_mm: float | None = None) -> cv2.aruco.CharucoBoard`, `ar0144.board.n_corners() -> int`, `ar0144.board.render_page() -> np.ndarray`, `ar0144.board.PX_PER_MM: int`, `ar0144.board.A4_W_MM: int`, `ar0144.board.A4_H_MM: int`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "ar0144"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["opencv-python>=5.0", "numpy>=2", "pillow>=10", "pyyaml>=6"]

[project.scripts]
ar0144 = "stereokit.cli:main"

[project.optional-dependencies]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Move the module and create the package marker**

```bash
mkdir -p ar0144 tests
git mv board.py ar0144/board.py
printf '"""Calibrated stereo depth for the Waveshare AR0144 module."""\n\n__version__ = "0.1.0"\n' > ar0144/__init__.py
```

Then remove the now-redundant PEP 723 header from `ar0144/board.py` (the first four lines, `# /// script` through `# ///`) — dependencies come from `pyproject.toml` now.

- [ ] **Step 3: Fix the importer**

In `focus_check.py`, change `import board as B` to `from stereokit import board as B`.

- [ ] **Step 4: Write the failing tests**

Create `tests/test_board.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_board.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ar0144'` before the move, then collection errors until steps 1–3 are done.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_board.py -v`
Expected: 6 passed. No implementation needed — `board.py` already satisfies these; the tests lock in behaviour that was previously only checked by throwaway scripts.

- [ ] **Step 7: Verify the phase-0 tool still runs**

Run: `uv run focus_check.py --help`
Expected: usage text, no import error.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml ar0144/ tests/test_board.py focus_check.py
git commit -m "Scaffold ar0144 package and lock board geometry in tests

Moves board.py into the package and promotes the throwaway round-trip checks
into real tests. The board is the measurement standard for everything
downstream: if the generated page and the detector disagree, every calibration
built on it is silently wrong."
```

---

### Task 2: `camera.py` — mode table, width guard, control save/restore

**Files:**
- Create: `ar0144/camera.py`, `tests/test_camera.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ar0144.camera.Mode` — `NamedTuple(name: str, combined_w: int, combined_h: int, eye_w: int, eye_h: int, fps: int, f_nominal: float)`
  - `ar0144.camera.MODES: dict[str, Mode]`
  - `ar0144.camera.StereoCamera(mode: str = "1280x480", device: int | None = None, lock_exposure: bool = True, exposure: int = 300, gain: int = 40)` — context manager; `.grab() -> tuple[np.ndarray, np.ndarray, float]` returning `(left_bgr, right_bgr, timestamp)`; `.mode -> Mode`; `.eye_size -> tuple[int, int]`
  - `ar0144.camera.split_eyes(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
  - `ar0144.camera.SAVED_PROPS`, `ar0144.camera.DEFAULT_PROPS`, `ar0144.camera.restore_controls(cap, saved) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_camera.py`:

```python
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

    def __init__(self, actual_width, opened=True):
        self.actual_width = actual_width
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
        return float(self.props.get(prop, 0))

    def read(self):
        return True, np.zeros((480, self.actual_width, 3), np.uint8)

    def release(self):
        self.released = True


def test_guard_rejects_node_that_returns_a_single_eye():
    """1280 requested, 640 delivered => not the stereo node."""
    cap = FakeCap(actual_width=640)
    assert C._accept(cap, 1280) is False
    assert cap.released is True


def test_guard_accepts_node_that_returns_the_full_width():
    cap = FakeCap(actual_width=1280)
    assert C._accept(cap, 1280) is True
    assert cap.released is False


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_camera.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar0144.camera'`

- [ ] **Step 3: Implement `ar0144/camera.py`**

```python
"""Capture from the AR0144 stereo module.

The module is a single UVC node emitting one concatenated [left|right] frame,
so both eyes arrive in the same buffer and are hardware frame-synchronised.
Two hazards drive this code: the driver silently returns ONE eye if a
single-width size is requested, and UVC control values persist on the device
after the process exits.
"""
import time
from typing import NamedTuple

import cv2
import numpy as np


class Mode(NamedTuple):
    name: str
    combined_w: int
    combined_h: int
    eye_w: int
    eye_h: int
    fps: int
    f_nominal: float      # px, spec table 2.1 — a prior, never a substitute
                          # for calibration


# Spec 2.1. The 4:3 modes are a horizontal CROP (960 of 1280 reference
# columns), not a downscale, which is why f does not simply scale with width.
MODES = {
    "2560x720": Mode("2560x720", 2560, 720, 1280, 720, 30, 1004.6),
    "1600x600": Mode("1600x600", 1600, 600, 800, 600, 30, 837.2),
    "1280x480": Mode("1280x480", 1280, 480, 640, 480, 60, 669.7),
    "1280x360": Mode("1280x360", 1280, 360, 640, 360, 120, 502.3),
}

DEFAULT_MODE = "1280x480"

# UVC control values live on the DEVICE and persist after the process exits,
# so a tool that changes them silently reconfigures every program run
# afterwards (spec 2.3).
SAVED_PROPS = (cv2.CAP_PROP_AUTO_EXPOSURE, cv2.CAP_PROP_AUTO_WB,
               cv2.CAP_PROP_EXPOSURE, cv2.CAP_PROP_GAIN)
DEFAULT_PROPS = {cv2.CAP_PROP_AUTO_EXPOSURE: 3.0,   # 3 = aperture priority
                 cv2.CAP_PROP_AUTO_WB: 1.0,
                 cv2.CAP_PROP_EXPOSURE: 166.0,
                 cv2.CAP_PROP_GAIN: 32.0}


def split_eyes(frame):
    """Split the concatenated frame into (left, right)."""
    mid = frame.shape[1] // 2
    return frame[:, :mid], frame[:, mid:]


def restore_controls(cap, saved):
    """Put device controls back as found. Auto flags LAST, or the manual
    writes that follow would silently turn auto off again."""
    for prop in reversed(SAVED_PROPS):
        value = saved.get(prop)
        if value is None or value < 0:
            value = DEFAULT_PROPS[prop]
        cap.set(prop, value)


def _accept(cap, combined_w):
    """True if this node really delivers the full double-width frame.

    A node that hands back half the requested width is a single-eye mode, and
    splitting that frame would yield two halves of the LEFT image that look
    plausibly like a stereo pair. Release and reject.
    """
    if int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) != combined_w:
        cap.release()
        return False
    return True


class StereoCamera:
    def __init__(self, mode=DEFAULT_MODE, device=None, lock_exposure=True,
                 exposure=300, gain=40):
        self.mode = MODES[mode]          # KeyError on unknown mode, by design
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
        indices = [self._device] if self._device is not None else range(10)
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
                if not _accept(cap, m.combined_w):
                    continue
                self._cap = cap
                self._saved = {p: cap.get(p) for p in SAVED_PROPS}
                if self._lock_exposure:
                    # Locked for SLAM: auto-exposure drifting between frames
                    # breaks feature tracking and loop closure (spec 2.3).
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # 1 = manual
                    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                    cap.set(cv2.CAP_PROP_EXPOSURE, self._exposure)
                    cap.set(cv2.CAP_PROP_GAIN, self._gain)
                return self
        finally:
            logging.setLogLevel(previous)
        raise RuntimeError(
            f"No camera delivering {m.combined_w}x{m.combined_h}. "
            "Is the AR0144 plugged in?")

    def grab(self):
        """Return (left, right, timestamp).

        Timestamp is taken immediately after read() returns. OpenCV does not
        expose V4L2 kernel buffer timestamps, so this carries a few ms of USB
        and MJPEG-decode latency with jitter (spec 11).
        """
        ok, frame = self._cap.read()
        ts = time.monotonic()
        if not ok:
            raise RuntimeError("Frame grab failed")
        left, right = split_eyes(frame)
        return left, right, ts

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_camera.py -v`
Expected: 9 passed.

- [ ] **Step 5: Hardware check**

Run:
```bash
uv run python -c "
from stereokit.camera import StereoCamera
import time
with StereoCamera() as cam:
    for _ in range(20): cam.grab()
    t0 = time.perf_counter()
    for _ in range(120): l, r, ts = cam.grab()
    print(l.shape, r.shape, '%.1f fps' % (120/(time.perf_counter()-t0)))
"
```
Expected: `(480, 640, 3) (480, 640, 3)` and ~60 fps.

Then confirm controls were restored:
```bash
v4l2-ctl -d /dev/video4 --list-ctrls | grep -E "white_balance_automatic|auto_exposure"
```
Expected: `white_balance_automatic ... value=1`, `auto_exposure ... value=3`.

- [ ] **Step 6: Commit**

```bash
git add ar0144/camera.py tests/test_camera.py
git commit -m "Add StereoCamera with width guard and UVC control restore

The width guard matters more than it looks: the driver does not error when a
single-width size is requested, it just hands back one eye, and splitting that
frame yields two halves of the left image that pass for a stereo pair.

Controls are saved on open and restored on close, auto flags last."
```

---

### Task 3: `calibrate.py` part 1 — pair detection and coverage tracking

**Files:**
- Create: `ar0144/calibrate.py`, `tests/test_calibrate_capture.py`

**Interfaces:**
- Consumes: `ar0144.board.make_board`, `ar0144.board.n_corners`.
- Produces:
  - `ar0144.calibrate.DetectedPair` — `NamedTuple(obj: np.ndarray, img_left: np.ndarray, img_right: np.ndarray, corners_left: np.ndarray, corners_right: np.ndarray, n: int)`
  - `ar0144.calibrate.detect_pair(detector, board, gray_left, gray_right, min_corners: int = 12) -> DetectedPair | None`
  - `ar0144.calibrate.sharpness(gray, corners) -> float`
  - `ar0144.calibrate.CoverageTracker(eye_size, grid=(4, 3), tilt_bins=4, target_pairs=50)` with `.add(corners, tilt_deg)`, `.fov_coverage() -> float`, `.tilt_coverage() -> float`, `.done() -> bool`, `.n_pairs: int`
  - `ar0144.calibrate.board_tilt_deg(rvec) -> float`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_calibrate_capture.py`:

```python
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
    if pair is not None:
        assert pair.img_left.shape == pair.img_right.shape
        assert pair.obj.shape[0] == pair.img_left.shape[0]


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
    for angle in (12.0, 25.0, 40.0):
        t.add(pts, tilt_deg=angle)
    assert t.tilt_coverage() == 1.0


def test_done_requires_pairs_and_coverage_and_tilt():
    t = CAL.CoverageTracker((640, 480), target_pairs=3)
    pts = np.array([[x, y] for x in np.linspace(5, 635, 12)
                    for y in np.linspace(5, 475, 9)])
    for angle in (2.0, 15.0, 28.0, 42.0):
        t.add(pts, tilt_deg=angle)
    assert t.n_pairs == 4
    assert t.done() is True


def test_board_tilt_is_zero_when_facing_the_camera():
    assert CAL.board_tilt_deg(np.zeros(3)) < 1e-6
    # 30 degrees about x
    rvec = np.array([np.deg2rad(30.0), 0.0, 0.0])
    assert abs(CAL.board_tilt_deg(rvec) - 30.0) < 1e-3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_calibrate_capture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar0144.calibrate'`

- [ ] **Step 3: Implement the capture half of `ar0144/calibrate.py`**

```python
"""Calibration: capture protocol and solve.

Capture quality decides calibration quality. Image-area coverage conditions the
distortion estimate; tilt variety conditions focal length. Both are tracked and
shown so the operator knows when to stop rather than guessing (spec 5.2).
"""
from typing import NamedTuple

import cv2
import numpy as np

from . import board as board_mod


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
    """Variance of the Laplacian over the board's bounding box.

    Used only to REJECT motion-blurred frames, where a relative comparison
    within one scene is valid. It is not a distance-comparable blur metric —
    that needs edge rise width (see focus_check.py and spec 3.2).
    """
    x0, y0 = np.floor(corners.min(axis=0)).astype(int)
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(cv2.Laplacian(patch, cv2.CV_64F).var())


def board_tilt_deg(rvec):
    """Angle between the board normal and the camera axis, in degrees."""
    R = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))[0]
    normal = R @ np.array([0.0, 0.0, 1.0])
    cos = abs(float(normal[2]))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class CoverageTracker:
    """Tracks where in the image corners have been seen, and at what tilts."""

    def __init__(self, eye_size, grid=(4, 3), tilt_bins=4, target_pairs=50,
                 max_tilt_deg=50.0):
        self.eye_w, self.eye_h = eye_size
        self.grid = grid
        self.tilt_bins = tilt_bins
        self.target_pairs = target_pairs
        self.max_tilt_deg = max_tilt_deg
        self._cells = np.zeros(grid[0] * grid[1], bool)
        self._tilts = np.zeros(tilt_bins, bool)
        self.n_pairs = 0

    def add(self, corners, tilt_deg):
        gx, gy = self.grid
        pts = np.asarray(corners, float).reshape(-1, 2)
        cx = np.clip((pts[:, 0] / self.eye_w * gx).astype(int), 0, gx - 1)
        cy = np.clip((pts[:, 1] / self.eye_h * gy).astype(int), 0, gy - 1)
        self._cells[cy * gx + cx] = True
        b = int(np.clip(tilt_deg / self.max_tilt_deg * self.tilt_bins,
                        0, self.tilt_bins - 1))
        self._tilts[b] = True
        self.n_pairs += 1

    def fov_coverage(self):
        return float(self._cells.mean())

    def tilt_coverage(self):
        return float(self._tilts.mean())

    def done(self):
        return (self.n_pairs >= self.target_pairs
                and self.fov_coverage() == 1.0
                and self.tilt_coverage() == 1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_calibrate_capture.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add ar0144/calibrate.py tests/test_calibrate_capture.py
git commit -m "Add calibration pair detection and coverage tracking

detect_pair takes the id INTERSECTION of the two eyes: extrinsics come from
correspondences, so a corner seen by only one eye is useless and would
misalign the object/image arrays.

CoverageTracker exists because calibration quality is decided at capture time.
Image-area coverage conditions distortion, tilt variety conditions focal
length, and neither is obvious to an operator without feedback."
```

---

### Task 4: `calibrate.py` part 2 — solve, holdout validation, acceptance metrics

**Files:**
- Modify: `ar0144/calibrate.py` (append)
- Create: `tests/synthetic.py`, `tests/test_calibrate_solve.py`

**Interfaces:**
- Consumes: `DetectedPair` from Task 3, `ar0144.board.BOARD`.
- Produces:
  - `ar0144.calibrate.solve(pairs: list[DetectedPair], eye_size, mode: str, square_mm: float, holdout_frac: float = 0.2, seed: int = 0) -> dict` — returns the calibration dict written to `calibration.json`
  - `ar0144.calibrate.epipolar_error(pairs, calib) -> float` — mean |y_left − y_right| in px after rectification
  - `ar0144.calibrate.check_acceptance(calib) -> list[str]` — empty list means pass; each string names a failed criterion
  - `ar0144.calibrate.save(calib, path) -> None`, `ar0144.calibrate.load(path) -> dict`
  - Calibration dict schema (all matrices as nested lists for JSON):
    ```
    {"mode": str, "eye_size": [w, h], "square_mm": float,
     "board": {...BOARD...},
     "left":  {"K": 3x3, "dist": [5]},
     "right": {"K": 3x3, "dist": [5]},
     "R": 3x3, "T": [3][1], "baseline_m": float,
     "rectified": {"R1":3x3,"R2":3x3,"P1":3x4,"P2":3x4,"Q":4x4,
                   "roi_left":[x,y,w,h],"roi_right":[x,y,w,h]},
     "metrics": {"rms_left":f,"rms_right":f,"rms_stereo":f,
                 "epipolar_px":f,"n_pairs":i,"n_holdout":i,
                 "f_nominal":f,"f_recovered":f}}
    ```

- [ ] **Step 1: Write the synthetic ground-truth generator**

Create `tests/synthetic.py`:

```python
"""Ground-truth stereo observations, so the solver is tested against a known
answer rather than against itself.

Projects the real board's 3D corners through known intrinsics and extrinsics.
No image rendering: detection is tested separately in test_board.py, and this
isolates the solver maths.
"""
import cv2
import numpy as np

from stereokit import board as B
from stereokit.calibrate import DetectedPair

K_TRUE = np.array([[670.0, 0.0, 322.0],
                   [0.0, 670.0, 238.0],
                   [0.0, 0.0, 1.0]])
D_TRUE = np.array([-0.05, 0.02, 0.0, 0.0, 0.0])
BASELINE_TRUE = 0.052
RVEC_STEREO = np.array([0.002, -0.004, 0.001])     # slight rig misalignment


def make_pairs(n=40, noise_px=0.05, seed=0, eye_size=(640, 480)):
    """Return (pairs, truth) with `pairs` in the shape solve() consumes."""
    rng = np.random.default_rng(seed)
    W, H = eye_size
    R_true = cv2.Rodrigues(RVEC_STEREO)[0]
    T_true = np.array([[-BASELINE_TRUE], [0.0], [0.0]])
    obj0 = B.make_board().getChessboardCorners()
    obj0 = obj0 - obj0.mean(axis=0)                # centre so tilts stay in view

    pairs = []
    attempts = 0
    while len(pairs) < n and attempts < n * 20:
        attempts += 1
        rvec = rng.normal(0.0, 0.35, 3)
        tvec = np.array([rng.normal(0.0, 0.05), rng.normal(0.0, 0.04),
                         rng.uniform(0.40, 0.70)])
        pl, _ = cv2.projectPoints(obj0, rvec, tvec, K_TRUE, D_TRUE)
        Rb = cv2.Rodrigues(rvec)[0]
        Rr = R_true @ Rb
        tr = R_true @ tvec.reshape(3, 1) + T_true
        pr, _ = cv2.projectPoints(obj0, cv2.Rodrigues(Rr)[0], tr,
                                  K_TRUE, D_TRUE)
        pl = pl.reshape(-1, 2) + rng.normal(0.0, noise_px, (len(obj0), 2))
        pr = pr.reshape(-1, 2) + rng.normal(0.0, noise_px, (len(obj0), 2))
        inside = (pl.min() >= 0 and pr.min() >= 0
                  and pl[:, 0].max() < W and pl[:, 1].max() < H
                  and pr[:, 0].max() < W and pr[:, 1].max() < H)
        if not inside:
            continue
        pairs.append(DetectedPair(obj0.astype(np.float32),
                                  pl.astype(np.float32),
                                  pr.astype(np.float32),
                                  pl.astype(np.float32),
                                  pr.astype(np.float32), len(obj0)))
    truth = dict(K=K_TRUE, dist=D_TRUE, baseline=BASELINE_TRUE)
    return pairs, truth
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_calibrate_solve.py`:

```python
"""The solver is tested against known ground truth.

Reprojection error on the fitting set measures fit, not accuracy, which is why
solve() holds 20% out and why these tests check RECOVERY of the true
parameters, not just a small residual (spec 5.3, 5.4).
"""
import numpy as np
import pytest

from stereokit import calibrate as CAL
from tests.synthetic import make_pairs, K_TRUE, BASELINE_TRUE


@pytest.fixture(scope="module")
def solved():
    pairs, truth = make_pairs(n=40, seed=1)
    calib = CAL.solve(pairs, (640, 480), "1280x480", 25.0)
    return calib, truth


def test_recovers_focal_length(solved):
    calib, _ = solved
    fx = calib["left"]["K"][0][0]
    assert abs(fx - K_TRUE[0, 0]) / K_TRUE[0, 0] < 0.01


def test_recovers_principal_point(solved):
    calib, _ = solved
    cx = calib["left"]["K"][0][2]
    cy = calib["left"]["K"][1][2]
    assert abs(cx - K_TRUE[0, 2]) < 5.0
    assert abs(cy - K_TRUE[1, 2]) < 5.0


def test_recovers_baseline(solved):
    calib, _ = solved
    assert abs(calib["baseline_m"] - BASELINE_TRUE) / BASELINE_TRUE < 0.02


def test_meets_every_acceptance_criterion(solved):
    calib, _ = solved
    assert CAL.check_acceptance(calib) == []


def test_reports_holdout_count(solved):
    calib, _ = solved
    m = calib["metrics"]
    assert m["n_holdout"] == pytest.approx(0.2 * (m["n_pairs"] + m["n_holdout"]),
                                           abs=1)
    assert m["n_holdout"] >= 1


def test_epipolar_error_is_subpixel(solved):
    calib, _ = solved
    assert calib["metrics"]["epipolar_px"] < 0.3


def test_p2_translation_term_is_negative(solved):
    """ROS convention: P2[0,3] = -fx*B. Wrong sign yields a mirrored map."""
    calib, _ = solved
    P1 = np.array(calib["rectified"]["P1"])
    P2 = np.array(calib["rectified"]["P2"])
    assert P2[0, 3] < 0
    expected = -P1[0, 0] * calib["baseline_m"]
    assert abs(P2[0, 3] - expected) / abs(expected) < 0.01


def test_noisy_input_fails_acceptance():
    """The gates must actually reject a bad calibration."""
    pairs, _ = make_pairs(n=40, noise_px=3.0, seed=2)
    calib = CAL.solve(pairs, (640, 480), "1280x480", 25.0)
    assert CAL.check_acceptance(calib) != []


def test_records_the_mode_it_was_captured_at(solved):
    calib, _ = solved
    assert calib["mode"] == "1280x480"
    assert calib["eye_size"] == [640, 480]


def test_save_load_round_trip(solved, tmp_path):
    calib, _ = solved
    p = tmp_path / "calibration.json"
    CAL.save(calib, p)
    back = CAL.load(p)
    assert back["mode"] == calib["mode"]
    assert np.allclose(np.array(back["left"]["K"]),
                       np.array(calib["left"]["K"]))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_calibrate_solve.py -v`
Expected: FAIL with `AttributeError: module 'ar0144.calibrate' has no attribute 'solve'`

- [ ] **Step 4: Append the solve half to `ar0144/calibrate.py`**

```python
import json
from pathlib import Path


def _as_lists(a):
    return np.asarray(a).tolist()


def solve(pairs, eye_size, mode, square_mm, holdout_frac=0.2, seed=0):
    """Calibrate from detected pairs and rectify.

    Per-eye intrinsics first, then stereoCalibrate with CALIB_FIX_INTRINSIC
    (spec 5.3). A fraction of pairs is held out and never seen by the solver,
    because residual on the fitting set measures fit, not accuracy.
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

    # OpenCV 5 returns 9 values here; 4.x examples show 8.
    ret = cv2.stereoCalibrate(obj, pl, pr, Kl, Dl, Kr, Dr, eye_size,
                              flags=cv2.CALIB_FIX_INTRINSIC)
    rms_s, Kl, Dl, Kr, Dr, R, T = ret[:7]

    # alpha=0 crops to valid pixels, so the RGBD map has no black border
    # (spec 6). This changes the effective FOV and intrinsics, so the
    # RECTIFIED P1/P2/Q are what the API reports.
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        Kl, Dl, Kr, Dr, eye_size, R, T, alpha=0)

    calib = {
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
            "f_nominal": float(MODES[mode].f_nominal),
        },
    }
    calib["metrics"]["epipolar_px"] = epipolar_error(hold, calib)
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
    "rms_left": 0.3, "rms_right": 0.3, "rms_stereo": 0.5, "epipolar_px": 0.3,
}
BASELINE_NOMINAL_M = 0.052
BASELINE_TOLERANCE = 0.05          # 5%
F_TOLERANCE = 0.10                 # 10%: closest pair of mode f_nominal values
                                    # differ by 25%, so this separates modes
                                    # cleanly while tolerating prior rounding


def check_acceptance(calib):
    """Return a list of failed criteria. Empty means the calibration passes."""
    failures = []
    m = calib["metrics"]
    for key, limit in ACCEPTANCE.items():
        value = m.get(key)
        if value is None or not np.isfinite(value) or value >= limit:
            failures.append(f"{key}={value:.4f} (limit <{limit})")
    rel = abs(calib["baseline_m"] - BASELINE_NOMINAL_M) / BASELINE_NOMINAL_M
    if rel > BASELINE_TOLERANCE:
        failures.append(
            f"baseline={calib['baseline_m']*1000:.2f}mm vs nominal 52mm "
            f"({rel*100:.1f}% off) — suspect print scaling or board flatness")
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_calibrate_solve.py -v`
Expected: 10 passed.

- [ ] **Step 6: Commit**

```bash
git add ar0144/calibrate.py tests/synthetic.py tests/test_calibrate_solve.py
git commit -m "Add calibration solve with holdout validation and acceptance gates

Tested against synthetic ground truth rather than against itself: the board's
real 3D corners are projected through known intrinsics and extrinsics, and the
tests assert RECOVERY of those values, not merely a small residual.

20% of pairs are held out, because reprojection error on the fitting set
measures fit, not accuracy. The gates that matter are post-rectification
epipolar error (<0.3 px, what disparity actually depends on) and recovered
baseline vs 52 mm (catches print scaling, which nothing downstream detects).
A deliberately noisy input is tested to confirm the gates actually reject."
```

---

### Task 5: `rectify.py`

**Files:**
- Create: `ar0144/rectify.py`, `tests/test_rectify.py`

**Interfaces:**
- Consumes: `ar0144.calibrate.load`.
- Produces:
  - `ar0144.rectify.Rectifier(calib: dict)`; classmethod `from_file(path) -> Rectifier`
  - `.rectify(left, right) -> tuple[np.ndarray, np.ndarray]`
  - `.K -> np.ndarray` (3×3 rectified left, i.e. `P1[:, :3]`), `.Q -> np.ndarray`, `.fx -> float`, `.baseline_m -> float`, `.eye_size -> tuple[int, int]`, `.mode -> str`
  - Raises `ValueError` if asked to rectify a frame whose size differs from the calibrated `eye_size`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rectify.py`:

```python
import numpy as np
import pytest

from stereokit import calibrate as CAL
from stereokit.rectify import Rectifier
from tests.synthetic import make_pairs, BASELINE_TRUE


@pytest.fixture(scope="module")
def rect():
    pairs, _ = make_pairs(n=40, seed=3)
    return Rectifier(CAL.solve(pairs, (640, 480), "1280x480", 25.0))


def test_exposes_rectified_intrinsics(rect):
    assert rect.K.shape == (3, 3)
    assert abs(rect.fx - rect.K[0, 0]) < 1e-9
    assert 600 < rect.fx < 750


def test_baseline_matches_truth(rect):
    assert abs(rect.baseline_m - BASELINE_TRUE) / BASELINE_TRUE < 0.02


def test_rectify_preserves_shape(rect):
    left = np.zeros((480, 640, 3), np.uint8)
    right = np.zeros((480, 640, 3), np.uint8)
    lr, rr = rect.rectify(left, right)
    assert lr.shape == (480, 640, 3)
    assert rr.shape == (480, 640, 3)


def test_rectify_rejects_wrong_frame_size(rect):
    """A calibration is per-mode; the crop moves the principal point, so
    using it against another mode is an error, not a warning (spec 5.4)."""
    bad = np.zeros((720, 1280, 3), np.uint8)
    with pytest.raises(ValueError, match="eye size"):
        rect.rectify(bad, bad)


def test_rectified_corners_land_on_the_same_row(rect):
    """The spec 6 verification: corresponding corners within 0.3 px in y."""
    import cv2
    pairs, _ = make_pairs(n=5, seed=9)
    calib = rect.calib
    Kl = np.array(calib["left"]["K"]); Dl = np.array(calib["left"]["dist"]).ravel()
    Kr = np.array(calib["right"]["K"]); Dr = np.array(calib["right"]["dist"]).ravel()
    R1 = np.array(calib["rectified"]["R1"]); P1 = np.array(calib["rectified"]["P1"])
    R2 = np.array(calib["rectified"]["R2"]); P2 = np.array(calib["rectified"]["P2"])
    errs = []
    for p in pairs:
        ul = cv2.undistortPoints(p.img_left.reshape(-1, 1, 2), Kl, Dl,
                                 R=R1, P=P1).reshape(-1, 2)
        ur = cv2.undistortPoints(p.img_right.reshape(-1, 1, 2), Kr, Dr,
                                 R=R2, P=P2).reshape(-1, 2)
        errs.append(np.abs(ul[:, 1] - ur[:, 1]))
    assert np.concatenate(errs).mean() < 0.3


def test_q_matrix_encodes_the_baseline(rect):
    """Q[3,2] = 1/baseline, so reprojectImageTo3D returns metres."""
    assert abs(abs(rect.Q[3, 2]) - 1.0 / rect.baseline_m) < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_rectify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar0144.rectify'`

- [ ] **Step 3: Implement `ar0144/rectify.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_rectify.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add ar0144/rectify.py tests/test_rectify.py
git commit -m "Add Rectifier

Exposes the RECTIFIED intrinsics (P1[:,:3]), not the raw ones: alpha=0 crops to
valid pixels so the RGBD map has no black border, and that changes the
effective FOV. Rectifying a frame whose size does not match the calibrated mode
raises, because the 4:3 crop moves the principal point and a rescaled
calibration would be quietly wrong."
```

---

### Task 6: `depth.py` — photometric normalisation, SGBM, metric depth

**Files:**
- Create: `ar0144/depth.py`, `tests/test_depth.py`

**Interfaces:**
- Consumes: `Rectifier.fx`, `Rectifier.baseline_m`.
- Produces:
  - `ar0144.depth.normalise_pair(gray_left, gray_right) -> tuple[np.ndarray, np.ndarray]`
  - `ar0144.depth.DepthEstimator(fx, baseline_m, num_disparities=192, block_size=5, min_depth=0.18, max_depth=4.0, uniqueness=10, speckle_window=100, speckle_range=2, disp12_max_diff=1)`
  - `.disparity(left_rect, right_rect) -> np.ndarray` float32, NaN where invalid
  - `.depth_from_disparity(disp) -> np.ndarray` float32 metres, NaN where invalid
  - `.compute(left_rect, right_rect) -> tuple[np.ndarray, np.ndarray]` — `(disparity, depth)`
  - `.min_disparity_for_range -> float`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_depth.py`:

```python
"""Depth maths is tested exactly; SGBM behaviour is tested by property.

The exact test matters because a wrong f*B is a silent scale error: every
depth is wrong by a constant factor and nothing looks broken.
"""
import numpy as np
import pytest

from stereokit import depth as D

FX = 669.7
B = 0.052
FB = FX * B          # 34.83


@pytest.fixture
def est():
    return D.DepthEstimator(fx=FX, baseline_m=B)


def test_depth_from_constant_disparity_is_exact(est):
    disp = np.full((10, 10), 34.83, np.float32)
    depth = est.depth_from_disparity(disp)
    assert np.allclose(depth, FB / 34.83, rtol=1e-5)


def test_known_depths_match_the_spec_table(est):
    """Spec 3: Z = f*B/d, so 1 m needs ~34.8 px at this focal length."""
    for z in (0.5, 1.0, 2.0, 3.0):
        d = FB / z
        depth = est.depth_from_disparity(np.full((4, 4), d, np.float32))
        assert np.allclose(depth, z, rtol=1e-4)


def test_depth_disparity_round_trip(est):
    disp = np.linspace(10.0, 180.0, 64).astype(np.float32).reshape(8, 8)
    depth = est.depth_from_disparity(disp)
    back = FB / depth
    assert np.allclose(back, disp, rtol=1e-5)


def test_invalid_disparity_becomes_nan(est):
    disp = np.array([[0.0, -1.0, 34.83]], np.float32)
    depth = est.depth_from_disparity(disp)
    assert np.isnan(depth[0, 0])
    assert np.isnan(depth[0, 1])
    assert np.isfinite(depth[0, 2])


def test_depth_outside_the_clamp_is_rejected(est):
    """Beyond 4 m the error budget says it is not worth keeping (spec 3)."""
    too_far = FB / 10.0            # 10 m
    too_near = FB / 0.05           # 5 cm
    depth = est.depth_from_disparity(
        np.array([[too_far, too_near]], np.float32))
    assert np.isnan(depth).all()


def test_num_disparities_sets_the_near_limit(est):
    """192 disparities => 0.18 m minimum (spec 3.2)."""
    assert abs(FB / 192 - 0.1814) < 1e-3
    assert est.min_disparity_for_range == pytest.approx(FB / 4.0, rel=1e-3)


def test_normalise_removes_a_gain_difference():
    """The eyes are NOT photometrically identical (spec 2.3); SGBM's cost
    assumes they are, so normalise before matching."""
    rng = np.random.default_rng(0)
    base = rng.integers(40, 200, (64, 64)).astype(np.uint8)
    dim = (base * 0.75).astype(np.uint8)
    nl, nr = D.normalise_pair(base, dim)
    assert abs(float(nl.mean()) - float(nr.mean())) < 2.0
    assert abs(float(nl.std()) - float(nr.std())) < 2.0


def test_normalise_preserves_structure():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (64, 64)).astype(np.uint8)
    out, _ = D.normalise_pair(img, img)
    # Monotonic remap: ordering of pixel values must survive.
    assert np.corrcoef(img.ravel(), out.ravel())[0, 1] > 0.99


def test_synthetic_shifted_pair_yields_the_planted_disparity(est):
    """A right image that is the left shifted by k px must measure ~k."""
    rng = np.random.default_rng(2)
    shift = 40
    w, h = 480, 240
    wide = rng.integers(0, 255, (h, w + shift)).astype(np.uint8)
    wide = np.dstack([wide] * 3)
    left = wide[:, shift:]
    right = wide[:, :-shift]
    disp = est.disparity(left, right)
    mid = disp[h // 4:3 * h // 4, w // 2 - 60:w // 2 + 60]
    valid = mid[np.isfinite(mid)]
    assert valid.size > 0.5 * mid.size
    assert abs(float(np.median(valid)) - shift) < 1.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_depth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar0144.depth'`

- [ ] **Step 3: Implement `ar0144/depth.py`**

```python
"""Dense stereo: rectified pair in, metric depth out.

For SLAM and for collision checking, OUTLIER RATE matters more than density. A
sparse but correct map is useful; a dense one with a few percent of gross
outliers corrupts odometry and drags the occupancy grid with it. Every filter
here is tuned toward rejection (spec 7).
"""
import cv2
import numpy as np


def normalise_pair(gray_left, gray_right):
    """Equalise mean and variance between the eyes before matching.

    The two sensors are not photometrically identical — roughly 6% colour
    difference under auto white balance, 22% with it locked (spec 2.3) — and
    SGBM's cost function assumes they agree. Normalising to a common mean and
    standard deviation removes both the gain and the colour mismatch in one
    step, and costs almost nothing.
    """
    out = []
    target_mean, target_std = 128.0, 50.0
    for g in (gray_left, gray_right):
        g = g.astype(np.float32)
        std = float(g.std())
        if std < 1e-6:
            out.append(np.full(g.shape, target_mean, np.uint8))
            continue
        scaled = (g - float(g.mean())) / std * target_std + target_mean
        out.append(np.clip(scaled, 0, 255).astype(np.uint8))
    return out[0], out[1]


class DepthEstimator:
    def __init__(self, fx, baseline_m, num_disparities=192, block_size=5,
                 min_depth=0.18, max_depth=4.0, uniqueness=10,
                 speckle_window=100, speckle_range=2, disp12_max_diff=1):
        if num_disparities % 16:
            raise ValueError("num_disparities must be a multiple of 16")
        self.fx = float(fx)
        self.baseline_m = float(baseline_m)
        self.fb = self.fx * self.baseline_m
        self.num_disparities = int(num_disparities)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

        channels = 1          # matching runs on normalised grayscale
        self._matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=block_size,
            P1=8 * channels * block_size ** 2,
            P2=32 * channels * block_size ** 2,
            disp12MaxDiff=disp12_max_diff,      # left-right consistency
            uniquenessRatio=uniqueness,
            speckleWindowSize=speckle_window,
            speckleRange=speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    @property
    def min_disparity_for_range(self):
        """Disparity corresponding to max_depth."""
        return self.fb / self.max_depth

    def _gray(self, img):
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def disparity(self, left_rect, right_rect):
        gl, gr = normalise_pair(self._gray(left_rect), self._gray(right_rect))
        raw = self._matcher.compute(gl, gr)       # fixed point, x16
        disp = raw.astype(np.float32) / 16.0
        disp[disp <= 0] = np.nan                  # 0 and negatives are invalid
        return disp

    def depth_from_disparity(self, disp):
        disp = np.asarray(disp, np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = self.fb / disp
        depth[~np.isfinite(depth)] = np.nan
        depth[disp <= 0] = np.nan
        depth[(depth < self.min_depth) | (depth > self.max_depth)] = np.nan
        return depth.astype(np.float32)

    def compute(self, left_rect, right_rect):
        disp = self.disparity(left_rect, right_rect)
        return disp, self.depth_from_disparity(disp)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_depth.py -v`
Expected: 9 passed. If `test_synthetic_shifted_pair_yields_the_planted_disparity` is marginal, widen the sampled region rather than loosening the 1.5 px tolerance — the tolerance is the point of the test.

- [ ] **Step 5: Benchmark against the spec's rate claim**

Run:
```bash
uv run python -c "
import numpy as np, time
from stereokit.depth import DepthEstimator
rng = np.random.default_rng(0)
l = rng.integers(0,255,(480,640,3)).astype(np.uint8)
r = rng.integers(0,255,(480,640,3)).astype(np.uint8)
e = DepthEstimator(669.7, 0.052)
e.compute(l, r)
t0 = time.perf_counter()
for _ in range(10): e.compute(l, r)
print('%.1f fps' % (10/(time.perf_counter()-t0)))
"
```
Expected: ≥10 fps. Spec §3.2 predicts ~17 fps at `numDisparities=192`. Record the actual figure; if it is far below, note it — the spec's near-range/frame-rate trade-off table would need revising.

- [ ] **Step 6: Commit**

```bash
git add ar0144/depth.py tests/test_depth.py
git commit -m "Add SGBM depth with per-eye photometric normalisation

Matching runs on normalised grayscale, not raw colour. The two sensors are not
photometrically identical (~6% colour difference under auto WB, 22% locked) and
SGBM's cost function assumes they agree, so equalising mean and variance first
removes both gain and colour mismatch for almost no cost.

Depth maths is tested exactly rather than approximately: a wrong f*B is a
silent scale error where every depth is off by a constant factor and nothing
looks broken."
```

---

### Task 7: `rgbd.py` — public API and TUM-format storage

**Files:**
- Create: `ar0144/rgbd.py`, `tests/test_rgbd.py`

**Interfaces:**
- Consumes: `StereoCamera`, `Rectifier`, `DepthEstimator`.
- Produces:
  - `ar0144.rgbd.RGBDFrame` — dataclass with `rgb: np.ndarray`, `depth: np.ndarray`, `disparity: np.ndarray`, `K: np.ndarray`, `Q: np.ndarray`, `timestamp: float`
  - `ar0144.rgbd.StereoRGBD(calibration_path, device=None, **depth_kwargs)` — context manager, `.read() -> RGBDFrame`, `__iter__` yielding frames, `.K`, `.Q`, `.mode`
  - `ar0144.rgbd.save_frame(out_dir, frame) -> tuple[Path, Path]`
  - `ar0144.rgbd.load_frame(out_dir, timestamp) -> RGBDFrame`
  - `ar0144.rgbd.write_meta(out_dir, rectifier) -> Path`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rgbd.py`:

```python
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
    assert np.abs(frame.depth[valid] - back.depth[valid]).max() < 0.001


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_rgbd.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar0144.rgbd'`

- [ ] **Step 3: Implement `ar0144/rgbd.py`**

```python
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
from .camera import MODES, StereoCamera
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
        self.rectifier = Rectifier(calib)
        self.estimator = DepthEstimator(
            fx=self.rectifier.fx, baseline_m=self.rectifier.baseline_m,
            **depth_kwargs)
        self._camera = StereoCamera(mode=self.mode, device=device)

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
    mm = np.clip(mm, 0, 65535).astype(np.uint16)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_rgbd.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add ar0144/rgbd.py tests/test_rgbd.py
git commit -m "Add StereoRGBD public API and TUM-format storage

RGB is the rectified left image and depth is computed in that same frame, so
alignment is free — unlike a RealSense there is no cross-sensor extrinsic to
reproject across.

Depth is stored as 16-bit PNG millimetres with 0 = invalid, the TUM/Redwood
convention, so Open3D reads the output without a converter. Loading a
calibration captured at a different mode raises rather than silently rescaling."
```

---

### Task 8: `export.py` — ROS `camera_info` YAML for RTAB-Map

**Files:**
- Create: `ar0144/export.py`, `tests/test_export.py`

**Interfaces:**
- Consumes: `ar0144.calibrate.load`.
- Produces:
  - `ar0144.export.camera_info_dict(name, width, height, K, dist, R_rect, P) -> dict`
  - `ar0144.export.export_calibration(calibration_path, out_dir, name="ar0144") -> tuple[Path, Path]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_export.py`:

```python
"""The two side-by-side stereo traps, asserted rather than remembered.

Both produce a plausible-looking but wrong map, which is the worst failure
mode: nothing errors, the result is just quietly incorrect (spec 9.1).
"""
import numpy as np
import pytest
import yaml

from stereokit import calibrate as CAL
from stereokit import export as EX
from tests.synthetic import make_pairs


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    pairs, _ = make_pairs(n=40, seed=7)
    calib = CAL.solve(pairs, (640, 480), "1280x480", 25.0)
    d = tmp_path_factory.mktemp("export")
    CAL.save(calib, d / "calibration.json")
    left, right = EX.export_calibration(d / "calibration.json", d)
    return calib, yaml.safe_load(left.read_text()), yaml.safe_load(right.read_text())


def test_image_width_is_per_eye_not_the_driver_width(exported):
    """640, not the 1280 the driver advertises. THE most common
    side-by-side stereo misconfiguration."""
    _, left, right = exported
    assert left["image_width"] == 640
    assert right["image_width"] == 640
    assert left["image_height"] == 480


def test_right_projection_translation_is_negative(exported):
    """ROS convention P2[0,3] = -fx*B. Wrong sign gives a mirrored map."""
    calib, _, right = exported
    P2 = right["projection_matrix"]["data"]
    assert P2[3] < 0


def test_right_projection_translation_equals_minus_fx_baseline(exported):
    calib, left, right = exported
    fx = left["projection_matrix"]["data"][0]
    expected = -fx * calib["baseline_m"]
    assert abs(right["projection_matrix"]["data"][3] - expected) / abs(expected) < 0.01


def test_left_projection_has_no_translation(exported):
    _, left, _ = exported
    assert abs(left["projection_matrix"]["data"][3]) < 1e-9


def test_distortion_model_is_plumb_bob_with_five_coefficients(exported):
    _, left, right = exported
    for info in (left, right):
        assert info["distortion_model"] == "plumb_bob"
        assert info["distortion_coefficients"]["cols"] == 5
        assert len(info["distortion_coefficients"]["data"]) == 5


def test_matrix_shapes_match_ros_camera_info(exported):
    _, left, _ = exported
    assert left["camera_matrix"]["rows"] == 3
    assert left["camera_matrix"]["cols"] == 3
    assert len(left["camera_matrix"]["data"]) == 9
    assert left["rectification_matrix"]["rows"] == 3
    assert len(left["rectification_matrix"]["data"]) == 9
    assert left["projection_matrix"]["rows"] == 3
    assert left["projection_matrix"]["cols"] == 4
    assert len(left["projection_matrix"]["data"]) == 12


def test_camera_names_are_distinct(exported):
    _, left, right = exported
    assert left["camera_name"].endswith("left")
    assert right["camera_name"].endswith("right")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar0144.export'`

- [ ] **Step 3: Implement `ar0144/export.py`**

```python
"""Export calibration in the ROS camera_info format RTAB-Map reads.

Two traps here both produce a plausible-looking but wrong map, so both are
automated rather than remembered (spec 9.1).
"""
from pathlib import Path

import numpy as np
import yaml

from . import calibrate


def _matrix(a, rows, cols):
    return {"rows": rows, "cols": cols,
            "data": [float(v) for v in np.asarray(a, float).reshape(-1)]}


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

    paths = []
    for side, R_key, P_key in (("left", "R1", "P1"), ("right", "R2", "P2")):
        info = camera_info_dict(
            f"{name}_{side}", w, h,
            calib[side]["K"], calib[side]["dist"],
            rec[R_key], rec[P_key])
        path = out_dir / f"{name}_{side}.yaml"
        path.write_text(yaml.safe_dump(info, sort_keys=False))
        paths.append(path)

    # P2[0,3] must be -fx*B. OpenCV's stereoRectify already returns this
    # convention, so it is a direct copy — but a wrong sign yields a mirrored,
    # plausible-looking map, so assert rather than trust.
    P2 = np.array(rec["P2"], float)
    assert P2[0, 3] < 0, "P2[0,3] must be negative (ROS convention -fx*B)"
    return tuple(paths)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_export.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add ar0144/export.py tests/test_export.py
git commit -m "Add ROS camera_info export for RTAB-Map

Automates the two side-by-side stereo traps rather than relying on memory:
image_width must be the per-eye 640 rather than the 1280 the driver advertises,
and P2[0,3] must be negative (-fx*B). Both produce a plausible-looking but
wrong map, which is the worst kind of failure — nothing errors."
```

---

### Task 9: `cli.py` — subcommands, live capture and record tools

**Files:**
- Create: `ar0144/cli.py`
- Delete: `stereo_view.py` (superseded by `ar0144 view --raw`)

**Interfaces:**
- Consumes: everything from Tasks 1–8.
- Produces: `ar0144.cli.main(argv=None) -> int`; subcommands `board`, `capture`, `solve`, `view`, `record`, `export`.

- [ ] **Step 1: Implement `ar0144/cli.py`**

```python
"""Command line entry points.

    ar0144 board    --out charuco_A4.pdf
    ar0144 capture  --out captures/ [--square-mm 24.8]
    ar0144 solve    captures/ --out calibration.json
    ar0144 view     [--raw] [--calibration calibration.json]
    ar0144 record   --out rgbd/ --calibration calibration.json
    ar0144 export   calibration.json --out rtabmap/
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import board as board_mod
from . import calibrate, export
from . import rgbd as rgbd_mod
from .camera import DEFAULT_MODE, StereoCamera
from .rectify import Rectifier


def cmd_board(args):
    from PIL import Image
    page = board_mod.render_page()
    Image.fromarray(page).save(args.out, "PDF", resolution=board_mod.DPI)
    print(f"Wrote {args.out}  ({board_mod.BOARD['squares_x']}x"
          f"{board_mod.BOARD['squares_y']} @ {board_mod.BOARD['square_mm']}mm, "
          f"{board_mod.n_corners()} corners)")
    print("Print at 100% on matte paper, mount flat, then MEASURE a square.")
    return 0


def cmd_capture(args):
    """Live capture with coverage feedback. Saves raw pairs so solving can be
    re-run with different settings without repeating the tedious part."""
    out = Path(args.out)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    board = board_mod.make_board(args.square_mm)
    detector = cv2.aruco.CharucoDetector(board)

    with StereoCamera(mode=args.mode) as cam:
        tracker = calibrate.CoverageTracker(cam.eye_size,
                                            target_pairs=args.target)
        K = np.array([[cam.mode.f_nominal, 0, cam.mode.eye_w / 2],
                      [0, cam.mode.f_nominal, cam.mode.eye_h / 2],
                      [0, 0, 1.0]])
        saved, prev, best_sharp = 0, None, 0.0
        print("Sweep the board across the frame at varied tilts. q to stop.")
        while True:
            left, right, ts = cam.grab()
            gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            pair = calibrate.detect_pair(detector, board, gl, gr)

            status = "no board"
            if pair is not None:
                sharp = calibrate.sharpness(gl, pair.corners_left)
                best_sharp = max(best_sharp, sharp)
                still = (prev is not None and prev.shape == pair.corners_left.shape
                         and np.median(np.linalg.norm(
                             pair.corners_left - prev, axis=1)) < 1.5)
                prev = pair.corners_left
                ok, rvec, _ = cv2.solvePnP(pair.obj, pair.img_left, K,
                                           np.zeros(5))
                tilt = calibrate.board_tilt_deg(rvec) if ok else 0.0
                # Reject motion-blurred frames: a blurry pair degrades the
                # solve and there is no way to tell afterwards.
                sharp_enough = sharp > 0.4 * best_sharp
                status = (f"{pair.n:2d} corners  tilt {tilt:4.1f}deg  "
                          f"{'STILL' if still else 'moving'} "
                          f"{'sharp' if sharp_enough else 'BLURRY'}")
                if still and sharp_enough:
                    np.savez(out / "pairs" / f"{saved:03d}.npz",
                             obj=pair.obj, left=pair.img_left,
                             right=pair.img_right)
                    tracker.add(pair.corners_left, tilt)
                    saved += 1
                    prev = None      # force re-settling before the next save
            else:
                prev = None

            view = np.hstack([left, right])
            cv2.putText(view, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 1)
            cv2.putText(view, f"pairs {saved}/{args.target}  "
                              f"fov {tracker.fov_coverage()*100:3.0f}%  "
                              f"tilt {tracker.tilt_coverage()*100:3.0f}%"
                              f"{'   DONE' if tracker.done() else ''}",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1)
            cv2.imshow("ar0144 capture", view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
        cv2.destroyAllWindows()

    meta = {"mode": args.mode, "square_mm": args.square_mm
            or board_mod.BOARD["square_mm"], "n_pairs": saved}
    (out / "capture.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {saved} pairs to {out/'pairs'}")
    return 0


def cmd_solve(args):
    src = Path(args.src)
    meta = json.loads((src / "capture.json").read_text())
    pairs = []
    for f in sorted((src / "pairs").glob("*.npz")):
        z = np.load(f)
        pairs.append(calibrate.DetectedPair(
            z["obj"], z["left"], z["right"], z["left"], z["right"],
            len(z["obj"])))
    if len(pairs) < 10:
        print(f"Only {len(pairs)} pairs; need at least 10.", file=sys.stderr)
        return 1

    from .camera import MODES
    mode = MODES[meta["mode"]]
    calib = calibrate.solve(pairs, (mode.eye_w, mode.eye_h), meta["mode"],
                            meta["square_mm"])
    calibrate.save(calib, args.out)

    m = calib["metrics"]
    print(f"pairs {m['n_pairs']} fit / {m['n_holdout']} held out")
    print(f"  rms left        {m['rms_left']:.4f} px   (limit <0.3)")
    print(f"  rms right       {m['rms_right']:.4f} px   (limit <0.3)")
    print(f"  rms stereo      {m['rms_stereo']:.4f} px   (limit <0.5)")
    print(f"  epipolar (held) {m['epipolar_px']:.4f} px   (limit <0.3)")
    print(f"  baseline        {calib['baseline_m']*1000:.2f} mm  "
          f"(nominal 52.0)")
    print(f"  f recovered     {m['f_recovered']:.1f} px  "
          f"(nominal {mode.f_nominal:.1f})")

    failures = calibrate.check_acceptance(calib)
    if failures:
        print("\nFAILED acceptance:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"\nPASSED. Wrote {args.out}")
    return 0


def cmd_view(args):
    if args.raw:
        with StereoCamera(mode=args.mode, lock_exposure=False) as cam:
            while True:
                left, right, _ = cam.grab()
                cv2.imshow("ar0144 raw", np.hstack([left, right]))
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
        cv2.destroyAllWindows()
        return 0

    with rgbd_mod.StereoRGBD(args.calibration) as cam:
        for frame in cam:
            vis = frame.depth.copy()
            vis[~np.isfinite(vis)] = 0
            vis = cv2.applyColorMap(
                cv2.convertScaleAbs(vis, alpha=255.0 / 4.0), cv2.COLORMAP_TURBO)
            vis[~np.isfinite(frame.depth)] = 0        # invalid stays black
            valid = np.isfinite(frame.depth).mean() * 100
            cv2.putText(vis, f"valid {valid:4.1f}%", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("ar0144 rgbd", np.hstack([frame.rgb, vis]))
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    cv2.destroyAllWindows()
    return 0


def cmd_record(args):
    out = Path(args.out)
    with rgbd_mod.StereoRGBD(args.calibration) as cam:
        rgbd_mod.write_meta(out, cam.rectifier)
        n = 0
        for frame in cam:
            rgbd_mod.save_frame(out, frame)
            n += 1
            if n >= args.frames:
                break
    print(f"Wrote {n} frames to {out}")
    return 0


def cmd_export(args):
    left, right = export.export_calibration(args.calibration, args.out,
                                            args.name)
    print(f"Wrote {left}\n      {right}")
    print("\nIn RTAB-Map: Preferences > Source > Stereo > 'Video Side-by-Side',"
          "\npoint it at the device and load these two files.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="ar0144", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("board"); s.add_argument("--out", default="charuco_A4.pdf")
    s.set_defaults(func=cmd_board)

    s = sub.add_parser("capture")
    s.add_argument("--out", default="captures")
    s.add_argument("--mode", default=DEFAULT_MODE)
    s.add_argument("--square-mm", type=float, default=None)
    s.add_argument("--target", type=int, default=50)
    s.set_defaults(func=cmd_capture)

    s = sub.add_parser("solve")
    s.add_argument("src")
    s.add_argument("--out", default="calibration.json")
    s.set_defaults(func=cmd_solve)

    s = sub.add_parser("view")
    s.add_argument("--raw", action="store_true")
    s.add_argument("--mode", default=DEFAULT_MODE)
    s.add_argument("--calibration", default="calibration.json")
    s.set_defaults(func=cmd_view)

    s = sub.add_parser("record")
    s.add_argument("--out", default="rgbd")
    s.add_argument("--calibration", default="calibration.json")
    s.add_argument("--frames", type=int, default=100)
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("export")
    s.add_argument("calibration")
    s.add_argument("--out", default="rtabmap")
    s.add_argument("--name", default="ar0144")
    s.set_defaults(func=cmd_export)

    args = p.parse_args(argv)
    return args.func(args)
```

- [ ] **Step 2: Verify the CLI wires up**

Run: `uv run ar0144 --help` then `uv run ar0144 board --out /tmp/x.pdf`
Expected: help lists all six subcommands; the board command writes the PDF and prints the geometry line.

- [ ] **Step 3: Verify `view --raw` matches the old script, then retire it**

Run: `uv run ar0144 view --raw`
Expected: a live side-by-side window, `q` quits, and afterwards
`v4l2-ctl -d /dev/video4 --list-ctrls | grep auto_exposure` still shows `value=3`.

```bash
git rm stereo_view.py
```

- [ ] **Step 4: Run the whole suite**

Run: `uv run --extra dev pytest -v`
Expected: all tests pass (≈55 across the seven test files).

- [ ] **Step 5: Commit**

```bash
git add ar0144/cli.py
git commit -m "Add ar0144 CLI and retire stereo_view.py

Six subcommands: board, capture, solve, view, record, export. capture saves raw
detected pairs rather than images, so solving can be re-run with different
settings without repeating the tedious part, and it rejects motion-blurred and
moving frames because a blurry pair degrades the solve with no way to tell
afterwards.

solve prints every acceptance figure against its limit and exits non-zero on
failure, so a bad calibration cannot be used by accident.

stereo_view.py is superseded by 'ar0144 view --raw' (spec 4)."
```

---

### Task 10: Hardware validation — calibrate, wall test, RTAB-Map

This task is manual and needs the printed board and the camera. It is the only
evidence that the whole chain is correct: a print-scale error survives every
automated test above and shows up here as a consistent percentage bias.

**Files:**
- Create: `tests/test_wall_validation.py`, `docs/superpowers/plans/2026-09-24-validation-results.md`

- [ ] **Step 1: Print and measure the board**

Print `charuco_A4.pdf` at exactly 100% on matte paper. Mount on foam board or
glass. **Measure the 100 mm scale bar and one square with calipers.** If the bar
is not 100.0 mm, pass the measured square size to every later command via
`--square-mm`.

- [ ] **Step 2: Capture calibration pairs**

Run: `uv run ar0144 capture --out captures/ --square-mm <measured>`

Work at 0.35–0.8 m, where the board is fully visible and sharpest (spec §3.2).
Push the board into the image corners — that is where distortion lives. Vary
tilt; tilt is what conditions the focal-length estimate. Stop when the display
shows `fov 100% tilt 100%` and ≥50 pairs.

- [ ] **Step 3: Solve and check the gates**

Run: `uv run ar0144 solve captures/ --out calibration.json`

Expected: `PASSED`, with per-eye RMS < 0.3 px, stereo RMS < 0.5 px, epipolar
< 0.3 px, baseline within a few % of 52 mm.

If **baseline** fails but the RMS figures are good, suspect print scaling
first — re-measure the square and re-run `solve` with the corrected value
(no re-capture needed, the pairs are on disk). If **epipolar** fails, the
capture lacked coverage or tilt variety; capture more.

- [ ] **Step 3b: Record per-eye colour gains (spec §2.3 item 2)**

Point the camera at an evenly-lit neutral surface (a grey card, or white paper
under diffuse light) filling both eyes, then:

```bash
uv run python -c "
import json, numpy as np
from pathlib import Path
from stereokit.camera import StereoCamera

with StereoCamera() as cam:
    for _ in range(30): left, right, _ = cam.grab()

def gains(img):
    h, w = img.shape[:2]
    p = img[h//4:3*h//4, w//4:3*w//4].astype(float)
    b, g, r = p[:,:,0].mean(), p[:,:,1].mean(), p[:,:,2].mean()
    return [r/g, 1.0, b/g]

gl, gr = gains(left), gains(right)
print('left  R/G=%.3f B/G=%.3f' % (gl[0], gl[2]))
print('right R/G=%.3f B/G=%.3f' % (gr[0], gr[2]))
print('right vs left: R/G %+.1f%%  B/G %+.1f%%'
      % ((gr[0]/gl[0]-1)*100, (gr[2]/gl[2]-1)*100))

path = Path('calibration.json')
calib = json.loads(path.read_text())
calib['colour_gains'] = {'left': gl, 'right': gr, 'note':
    'measured on a neutral target; diagnostic only, see plan step 3b'}
path.write_text(json.dumps(calib, indent=2))
print('recorded in calibration.json')
"
```

**Why this is diagnostic rather than applied.** The spec says to apply these to
the RGB output, but the RGB output is the rectified *left* image alone — a
per-eye gain ratio has nothing to correct there. What the ratio actually serves
is inter-eye agreement for matching, and `normalise_pair` already handles that
on grayscale, which cancels a colour imbalance more directly than a gain
correction would. So the number is recorded (it is the cheapest way to notice
one sensor drifting, or to confirm the ~6% figure from spec §2.3 against a
genuinely neutral target rather than a room scene) but not applied. **Flag this
in the validation results** — the spec's §2.3 item 2 wording should be narrowed
to match.

- [ ] **Step 4: Write the wall validation test**

Create `tests/test_wall_validation.py`:

```python
"""End-to-end metric validation against a flat wall at measured distances.

This is the only test that validates the whole chain. A print-scale error
survives every other test in the suite and appears here as a consistent
percentage bias, which is also how it is diagnosed.

Not part of the automated suite: needs the camera and a tape measure.
Run with: uv run --extra dev pytest tests/test_wall_validation.py -s -m hardware
"""
import numpy as np
import pytest

from stereokit.rgbd import StereoRGBD

pytestmark = pytest.mark.hardware


@pytest.mark.parametrize("expected_m", [0.5, 1.0, 2.0, 3.0])
def test_flat_wall_depth_and_flatness(expected_m):
    input(f"\nAim squarely at a flat, textured wall at exactly {expected_m} m, "
          "then press Enter...")
    with StereoRGBD("calibration.json") as cam:
        for _ in range(10):
            frame = cam.read()

    h, w = frame.depth.shape
    patch = frame.depth[h // 3:2 * h // 3, w // 3:2 * w // 3]
    valid = patch[np.isfinite(patch)]
    assert valid.size > 0.3 * patch.size, "too few valid pixels to judge"

    measured = float(np.median(valid))
    # Spec 3 error budget, generously: 5% plus the table's noise figure.
    tolerance = 0.05 * expected_m + 0.02
    print(f"\n  expected {expected_m:.2f} m, measured {measured:.3f} m, "
          f"bias {(measured/expected_m - 1)*100:+.1f}%")

    # Plane fit over the patch gives the noise, independent of scale error.
    ys, xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    m = np.isfinite(patch)
    A = np.column_stack([xs[m], ys[m], np.ones(m.sum())])
    coef, *_ = np.linalg.lstsq(A, patch[m], rcond=None)
    residual = patch[m] - A @ coef
    print(f"  plane-fit RMS {np.sqrt((residual**2).mean())*1000:.1f} mm")

    assert abs(measured - expected_m) < tolerance
```

Register the marker by adding to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = ["hardware: needs the camera and a physical setup"]
addopts = "-m 'not hardware'"
```

- [ ] **Step 5: Run the wall test**

Run: `uv run --extra dev pytest tests/test_wall_validation.py -s -m hardware`

Record the measured bias and plane-fit RMS at each distance. **A consistent
percentage bias across all four distances is a scale error** — almost certainly
the printed square size. A bias that grows with distance is a baseline or
rectification problem instead.

- [ ] **Step 6: Export and run RTAB-Map**

Run: `uv run ar0144 export calibration.json --out rtabmap/`

In RTAB-Map standalone: Preferences → Source → Stereo → **"Video Side-by-Side"**,
point it at `/dev/video4`, load `rtabmap/ar0144_left.yaml` and
`ar0144_right.yaml`. Set `Vis/MinDepth ≈ 0.2`, `Vis/MaxDepth ≈ 4.0` (spec §9.4 —
leaving MaxDepth unbounded lets far, noisy points dominate and is the most
likely cause of drift with a 52 mm baseline).

Walk a loop around the room and confirm a map builds and closes the loop.

- [ ] **Step 7: Record results**

Write `docs/superpowers/plans/2026-09-24-validation-results.md` with: the
measured square size, every calibration metric against its limit, the wall test
table (expected / measured / bias / plane RMS), SGBM frame rate, and whether
RTAB-Map built and closed a loop. Note anything that disagrees with the spec's
predictions — particularly the §3 error budget and the §3.2 frame-rate figures.

- [ ] **Step 8: Commit**

```bash
git add tests/test_wall_validation.py pyproject.toml calibration.json rtabmap/ \
        docs/superpowers/plans/2026-09-24-validation-results.md
git commit -m "Add wall validation and record hardware results

The wall test is the only evidence the whole chain is correct: a print-scale
error survives every other test in the suite and shows up here as a consistent
percentage bias across distances, which is also how it is diagnosed. A bias
that GROWS with distance points at the baseline or rectification instead."
```

---

## Notes for the executor

- **Do not reach for a better stereo matcher before calibration passes §5.4.**
  Bad extrinsics look exactly like a bad matcher, and swapping the matcher is
  the more satisfying thing to do, which is why it is the more common mistake
  (spec §12).
- `opencv-contrib-python`'s WLS disparity filter is **not** assumed. Add it only
  if disparity proves too sparse after calibration is verified clean.
- If the RGBD path proves too slow, remember it is perception, not odometry —
  RTAB-Map does its own per-feature correspondence and never consumes this dense
  map (spec §9.3).
