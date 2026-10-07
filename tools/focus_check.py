"""Measure the AR0144's usable near focus limit (spec Phase 0), v2.

v1 used variance-of-Laplacian over a fixed centre crop and produced a
non-physical curve: sharp at 0.30m, collapsed at 0.40-0.60m, sharp again at
1-2m. Defocus is unimodal, so that was the metric failing, not the lens. VoL
conflates blur with how much target texture happens to sit in the crop, with
contrast (auto-exposure was drifting), and with motion blur.

v2 fixes all four:

  * Exposure, gain and white balance are LOCKED for the whole sweep.
  * Distance comes from the board's own pose via solvePnP, so no tape measure
    and no guessed distances.
  * Sharpness is the 10-90% rise width across checker edges, measured IN PIXELS.
    Blur is a fixed pixel width regardless of how large the squares appear, so
    this is scale-invariant in a way VoL is not. ~1.5-2 px is sampling-limited
    (as sharp as the sensor can resolve); >4 px is visibly soft.
  * Frames are only sampled when the board is holding still.

Sweep the board slowly from ~0.15m out to ~0.8m and back. Samples bin by
distance automatically. The near limit is where rise width starts climbing.

Run with:  uv run focus_check.py                    (defaults to the spec board)
           uv run focus_check.py --square-mm 24.8   (if the print scaled)
Keys:      +/- exposure, [/] gain, r reset samples, q quit and report
"""

import argparse
import cv2
from stereokit import board as B
import numpy as np

MODE = (1280, 480)          # 640x480 per eye, the mode the spec settles on
NOMINAL_F = 670.0           # px, from spec table 2.1 for this mode
BIN_M = 0.05                # distance bin width, metres
MIN_CORNER_FRAC = 0.30      # need this fraction of corners to trust a sample
STILL_PX = 1.5              # median corner motion below this = board is still


# UVC control values live on the DEVICE and persist after the process exits,
# so anything run afterwards inherits them. Disabling auto white balance here
# was measured to swell the right-vs-left R/G difference from -6% to +22%,
# which looked like a hardware fault until it was traced back. Always restore.
SAVED_PROPS = (cv2.CAP_PROP_AUTO_EXPOSURE, cv2.CAP_PROP_AUTO_WB,
               cv2.CAP_PROP_EXPOSURE, cv2.CAP_PROP_GAIN)
DEFAULT_PROPS = {cv2.CAP_PROP_AUTO_EXPOSURE: 3,   # 3 = aperture priority (auto)
                 cv2.CAP_PROP_AUTO_WB: 1,
                 cv2.CAP_PROP_EXPOSURE: 166,
                 cv2.CAP_PROP_GAIN: 32}


def restore_camera(cap, saved):
    """Put the device controls back the way we found them."""
    for prop in reversed(SAVED_PROPS):   # auto flags last, so they win
        value = saved.get(prop)
        if value is None or value < 0:   # unreadable: fall back to the default
            value = DEFAULT_PROPS[prop]
        cap.set(prop, value)


def open_camera(exposure, gain):
    """Find the stereo node and pin exposure/gain/WB so contrast cannot drift."""
    logging = cv2.utils.logging
    previous = logging.getLogLevel()
    logging.setLogLevel(logging.LOG_LEVEL_ERROR)
    try:
        for index in range(10):
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, MODE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MODE[1])
            if int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) != MODE[0]:
                cap.release()
                continue
            print(f"Using /dev/video{index}")
            saved = {p: cap.get(p) for p in SAVED_PROPS}
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # 1 = manual, 3 = auto
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            cap.set(cv2.CAP_PROP_GAIN, gain)
            return cap, saved
    finally:
        logging.setLogLevel(previous)
    raise SystemExit("No stereo camera found. Is the AR0144 plugged in?")


def sample_line(gray, p0, direction, half_len, step=0.25):
    """Bilinearly sample a profile through p0 along direction."""
    ts = np.arange(-half_len, half_len + step, step)
    pts = p0[None, :] + ts[:, None] * direction[None, :]
    # cv2.remap wants float32 maps; one row of samples.
    mx = pts[:, 0].astype(np.float32).reshape(1, -1)
    my = pts[:, 1].astype(np.float32).reshape(1, -1)
    prof = cv2.remap(gray, mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)[0]
    return ts, prof.astype(np.float64)


def _crossing(ts, norm, level):
    """Position where the profile first reaches `level`, linearly interpolated.

    np.interp cannot be used directly: the profile is not guaranteed monotonic
    (noise, neighbouring squares), and np.interp silently returns nonsense for
    non-monotonic xp. Find the first crossing explicitly instead.
    """
    above = np.nonzero(norm >= level)[0]
    if len(above) == 0:
        return ts[-1]
    i = above[0]
    if i == 0:
        return ts[0]
    y0, y1 = norm[i - 1], norm[i]
    if y1 == y0:
        return ts[i]
    return ts[i - 1] + (level - y0) / (y1 - y0) * (ts[i] - ts[i - 1])


def edge_rise(gray, corners, ids, squares_x):
    """Median 10-90% edge rise width, in pixels, over the board's checker edges.

    Charuco interior corners sit on a (squares_x-1) x (squares_y-1) grid. Two
    horizontally adjacent corners straddle one square, and their midpoint lies
    on a black/white boundary that runs PARALLEL to the line joining them. So
    the edge normal is perpendicular to A->B, and that is the direction to
    sample along to see the transition.
    """
    per_row = squares_x - 1
    lut = {int(i): c for i, c in zip(ids.ravel(), corners.reshape(-1, 2))}
    rises = []

    for i, a in lut.items():
        if (i + 1) % per_row == 0:       # last corner in its row, no neighbour
            continue
        b = lut.get(i + 1)
        if b is None:
            continue
        a, b = np.asarray(a, float), np.asarray(b, float)
        span = np.linalg.norm(b - a)
        if span < 8:                      # squares too small to resolve an edge
            continue
        along = (b - a) / span
        normal = np.array([-along[1], along[0]])   # perpendicular = edge normal
        mid = (a + b) / 2.0
        ts, prof = sample_line(gray, mid, normal, span / 3.0)

        lo, hi = prof.min(), prof.max()
        if hi - lo < 25:                  # too little local contrast to trust
            continue
        norm = (prof - lo) / (hi - lo)
        if norm[0] > norm[-1]:            # always measure a rising edge
            norm = norm[::-1]
            ts = -ts[::-1]

        rise = _crossing(ts, norm, 0.90) - _crossing(ts, norm, 0.10)
        if 0.2 < rise < span / 2:         # reject degenerate fits
            rises.append(rise)

    return float(np.median(rises)) if len(rises) >= 5 else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--square-mm", type=float, default=None,
                   help="measured square size, if the print scaled (default: "
                        "board.py's nominal value)")
    p.add_argument("--exposure", type=int, default=300)
    p.add_argument("--gain", type=int, default=40)
    args = p.parse_args()

    # board.py is the single source of truth for the target geometry; never
    # restate it here, or the detected board can drift from the printed one.
    board = B.make_board(args.square_mm)
    detector = cv2.aruco.CharucoDetector(board)
    n_corners = B.n_corners()
    squares_x = B.BOARD["squares_x"]

    cap, saved = open_camera(args.exposure, args.gain)
    w, h = MODE[0] // 2, MODE[1]
    K = np.array([[NOMINAL_F, 0, w / 2.0],
                  [0, NOMINAL_F, h / 2.0],
                  [0, 0, 1.0]])
    zero_dist = np.zeros(5)

    print(f"\nBoard {squares_x}x{B.BOARD['squares_y']} @ {B.BOARD['square_mm']}mm, "
          f"{n_corners} corners. Exposure/gain locked.")
    print("Sweep the board slowly from ~0.15 m out to ~0.8 m and back.")
    print("+/- exposure, [/] gain, r reset, q quit and report.\n")

    samples = {}          # (eye, bin) -> list of rise widths
    detects = {}          # (eye, bin) -> list of corner counts
    prev = {}
    exposure, gain = args.exposure, args.gain

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed")
            break
        mid = frame.shape[1] // 2
        eyes = {"L": frame[:, :mid], "R": frame[:, mid:]}
        status = []

        for name, eye in eyes.items():
            gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            cc, ci, _, _ = detector.detectBoard(gray)
            if cc is None or len(cc) < MIN_CORNER_FRAC * n_corners:
                prev[name] = None
                status.append(f"{name}: no board")
                continue

            pts = cc.reshape(-1, 2)
            # Only sample when the board is holding still, so motion blur does
            # not get recorded as defocus.
            moved = None
            if prev.get(name) is not None and prev[name].shape == pts.shape:
                moved = float(np.median(np.linalg.norm(pts - prev[name], axis=1)))
            prev[name] = pts

            obj, img = board.matchImagePoints(cc, ci)
            ok_pnp, _, tvec = cv2.solvePnP(obj, img, K, zero_dist)
            if not ok_pnp:
                status.append(f"{name}: no pose")
                continue
            # .item(): numpy >= 2 refuses float() on a size-1 1-D array.
            dist = float(tvec[2].item())

            rise = edge_rise(gray, cc, ci, squares_x)
            still = moved is not None and moved < STILL_PX
            if rise is not None and still and 0.08 < dist < 1.5:
                b = round(dist / BIN_M) * BIN_M
                samples.setdefault((name, b), []).append(rise)
                detects.setdefault((name, b), []).append(len(cc))
            status.append(f"{name}: {dist:.2f}m {len(cc):>2}/{n_corners} "
                          f"rise={rise:.2f}px" if rise else f"{name}: {dist:.2f}m")

        view = np.hstack([eyes["L"], eyes["R"]]).copy()
        cv2.putText(view, "   ".join(status), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(view, f"exp={exposure} gain={gain} samples={sum(len(v) for v in samples.values())}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("focus check v2", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            samples, detects = {}, {}
        elif key in (ord("+"), ord("=")):
            exposure = min(2047, int(exposure * 1.25)); cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        elif key == ord("-"):
            exposure = max(3, int(exposure / 1.25)); cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        elif key == ord("]"):
            gain = min(130, gain + 5); cap.set(cv2.CAP_PROP_GAIN, gain)
        elif key == ord("["):
            gain = max(0, gain - 5); cap.set(cv2.CAP_PROP_GAIN, gain)

    restore_camera(cap, saved)
    cap.release()
    cv2.destroyAllWindows()

    if not samples:
        print("\nNo samples collected. Was the board detected?")
        return

    bins = sorted({b for _, b in samples})
    print(f"\n{'dist':>6}  {'L rise':>8} {'L n':>5}  {'R rise':>8} {'R n':>5}   corners L/R")
    for b in bins:
        row = [f"{b:6.2f}"]
        cn = []
        for eye in ("L", "R"):
            v = samples.get((eye, b))
            d = detects.get((eye, b))
            row.append(f"{np.median(v):8.2f} {len(v):5d}" if v else f"{'-':>8} {0:5d}")
            cn.append(f"{np.median(d):.0f}" if d else "-")
        print("  ".join(row) + f"   {cn[0]}/{cn[1]}")

    print(f"\nRise width is the 10-90% edge transition in PIXELS "
          f"(lower = sharper).")
    print("Flat curve  => in focus across that range, no near-focus wall.")
    print("Rising at short distance => that is the real near limit.")
    print("Persistent L vs R gap    => the two lenses are focused differently.")


if __name__ == "__main__":
    main()
