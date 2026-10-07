"""Validate stereo depth against the ChArUco board's own PnP geometry.

Better than a wall test in three ways: the board is guaranteed planar, its pose
gives a TRUE reference plane (so residuals measure depth noise rather than
"was that wall flat"), and it self-reports distance, so a bias-versus-range
curve comes free from sweeping the board around.

What this does and does not prove
---------------------------------
PnP distance scales with f; stereo depth scales with f*B. Their ratio is
therefore insensitive to focal-length error and tests the BASELINE and the
triangulation chain. It cannot detect a print-scale error, because a wrong
square size scales the recovered baseline and the PnP distance by the same
factor. Absolute scale is anchored by physically measuring the printed
100 mm bar; this script validates everything downstream of that.

Run with:  stereokit validate [calibration.json]
Sweep the board slowly between ~0.3 m and ~1.5 m. The window shows what the
camera sees, the detected corners, and the live bias. Bins fill as you sweep;
press q when the table below the view looks well covered.
"""
import argparse
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from stereokit import board as B
from stereokit.rgbd import StereoRGBD

BIN_M = 0.1
MIN_CORNERS = 20
TARGET_PER_BIN = 20
# FIXED target range. An earlier version binned whatever distances happened to
# appear, so the denominator grew every time you reached a new one and the
# target ran away from you -- "8/9 filled" with no way to see which.
TARGET_MIN, TARGET_MAX = 0.20, 1.00
# Far target: the calibration board's 18.75 mm markers die past ~1.3 m, so
# longer ranges need the 85 mm grid (charuco_far_A4.pdf).
FAR_MIN, FAR_MAX, FAR_BIN = 1.0, 4.0, 0.5


def board_plane_depth(shape, K, rvec, tvec):
    """Expected depth at every pixel, from the board's PnP plane.

    Board is z=0 in board coordinates, so its normal in camera coordinates is
    R@[0,0,1] and tvec is a point on it. For a ray K^-1[u,v,1] (whose z is 1),
    depth Z satisfies Z*(n.v) = n.p0.
    """
    R = cv2.Rodrigues(rvec)[0]
    n = R @ np.array([0.0, 0.0, 1.0])
    d = float(n @ tvec.reshape(3))

    h, w = shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    Kinv = np.linalg.inv(K)
    rays = Kinv @ np.stack([us.ravel(), vs.ravel(), np.ones(us.size)], axis=0)
    denom = n @ rays
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = d / denom
    Z[~np.isfinite(Z)] = np.nan
    return Z.reshape(h, w)


def _make_tone(path, freq, ms=90, vol=0.25):
    """Write a short mono WAV. Built once, played with paplay, because at
    2-4 m from the laptop the on-screen progress is unreadable and the
    operator has to work by ear."""
    import struct
    import wave
    rate = 22050
    n = int(rate * ms / 1000)
    frames = b"".join(
        struct.pack("<h", int(32767 * vol * math.sin(2 * math.pi * freq * i / rate)
                              * min(1.0, 4.0 * min(i, n - i) / n)))
        for i in range(n))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(frames)
    return path


class Beeper:
    """Sample = low tick, bin complete = rising pair, all done = chord."""

    def __init__(self, enabled=True):
        self.enabled = enabled and shutil.which("paplay") is not None
        if not self.enabled:
            return
        d = Path(tempfile.mkdtemp(prefix="stereokit-beep-"))
        self.tick = _make_tone(d / "tick.wav", 880, 45, 0.18)
        self.bin_done = _make_tone(d / "bin.wav", 1320, 120, 0.3)
        self.all_done = _make_tone(d / "all.wav", 1760, 320, 0.35)

    def play(self, which):
        if not self.enabled:
            return
        subprocess.Popen(["paplay", str(which)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


BINS = {"near": (TARGET_MIN, TARGET_MAX, BIN_M),
        "far": (FAR_MIN, FAR_MAX, FAR_BIN)}
MODE = "near"


def target_bins():
    lo, hi, step = BINS[MODE]
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 2) for i in range(n)]


class NearTarget:
    """The ChArUco calibration board: many corners, limited range."""
    min_corners = MIN_CORNERS

    def __init__(self):
        self.board = B.make_board()
        self.det = cv2.aruco.CharucoDetector(self.board)
        self.total = B.n_corners()

    def detect(self, gray):
        cc, ci, _, _ = self.det.detectBoard(gray)
        if cc is None:
            return 0, None, None, None
        return len(cc), cc.reshape(-1, 2), cc, ci

    def pose_points(self, cc, ci):
        return self.board.matchImagePoints(cc, ci)

    def draw(self, view, cc, ci):
        cv2.aruco.drawDetectedCornersCharuco(
            view, cc.reshape(-1, 1, 2).astype(np.float32),
            ci.reshape(-1, 1).astype(np.int32), (0, 255, 0))


class FarTarget:
    """85 mm ArUco grid: far fewer corners, but decodable to roughly 5 m."""
    min_corners = 8        # two markers

    def __init__(self):
        self.board = B.make_far_board()
        first = B.FAR_BOARD["first_id"]
        n = B.FAR_BOARD["markers_x"] * B.FAR_BOARD["markers_y"]
        self.ids = set(range(first, first + n))
        adict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, B.FAR_BOARD["dictionary"]))
        self.det = cv2.aruco.ArucoDetector(adict)
        self.total = B.FAR_BOARD["markers_x"] * B.FAR_BOARD["markers_y"] * 4

    def detect(self, gray):
        mc, mi, _ = self.det.detectMarkers(gray)
        if mi is None or len(mi) == 0:
            return 0, None, None, None
        # Keep ONLY this board's ids. Both targets use DICT_4X4_50 -- the
        # calibration board is 0-34, this grid is 40-43 -- so any calibration
        # marker in shot would be counted here but silently dropped by
        # matchImagePoints, leaving solvePnP with fewer than 4 points.
        keep = [i for i, mid in enumerate(mi.ravel()) if mid in self.ids]
        if len(keep) < 2:                       # need 2 markers = 8 points
            return 0, None, None, None
        mc = [mc[i] for i in keep]
        mi = mi.reshape(-1, 1)[keep]
        pts = np.concatenate([c.reshape(-1, 2) for c in mc])
        return len(pts), pts, mc, mi

    def pose_points(self, mc, mi):
        return self.board.matchImagePoints(mc, mi)

    def draw(self, view, mc, mi):
        cv2.aruco.drawDetectedMarkers(view, mc, mi, (0, 255, 0))


def draw_progress(panel, per_bin, current):
    """One row per target distance, so it is obvious which still need work."""
    bins = target_bins()
    x0, y0, bw, bh = 10, 76, 150, 13
    done = 0
    for i, b in enumerate(bins):
        n = len(per_bin.get(b, []))
        frac = min(n / TARGET_PER_BIN, 1.0)
        done += frac >= 1.0
        y = y0 + i * (bh + 3)
        colour = (0, 220, 0) if frac >= 1.0 else (0, 165, 255) if frac else (90, 90, 90)
        cv2.rectangle(panel, (x0, y), (x0 + bw, y + bh), (60, 60, 60), -1)
        cv2.rectangle(panel, (x0, y), (x0 + int(bw * frac), y + bh), colour, -1)
        # Marker showing where the board is right now.
        here = current is not None and abs(current - b) < BIN_M / 2
        cv2.putText(panel, f"{b:.2f}m {n:3d}/{TARGET_PER_BIN}"
                           f"{'  <- HERE' if here else ''}",
                    (x0 + bw + 8, y + bh - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 255, 255) if here else (210, 210, 210), 1)
    short = [b for b in bins if len(per_bin.get(b, [])) < TARGET_PER_BIN]
    return done, len(bins), short


def depth_colormap(depth, max_m=4.0):
    vis = depth.copy()
    vis[~np.isfinite(vis)] = 0
    vis = cv2.applyColorMap(
        cv2.convertScaleAbs(vis, alpha=255.0 / max_m), cv2.COLORMAP_TURBO)
    vis[~np.isfinite(depth)] = 0
    return vis


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("calibration", nargs="?", default="calibration.json",
                    help="calibration JSON (default: %(default)s)")
    ap.add_argument("--quiet", action="store_true",
                    help="no audio feedback (on by default: the screen is "
                         "unreadable from several metres away)")
    ap.add_argument("--target", default="near", choices=("near", "far"),
                    help="'near' uses the ChArUco calibration board "
                         "(0.2-1.0 m); 'far' uses the 85 mm ArUco grid "
                         "from charuco_far_A4.pdf (1-4 m)")
    ap.add_argument("--save", default=None, metavar="DIR",
                    help="also save rectified pairs plus the board's true "
                         "pose, so SGBM parameters can be swept offline "
                         "against ground truth (see tools/tune_sgbm.py)")
    return ap.parse_args(argv)


def main(args=None):
    global MODE
    args = args or parse_args()
    MODE = args.target
    target = FarTarget() if MODE == "far" else NearTarget()
    beeper = Beeper(enabled=not args.quiet)
    was_full = set()
    board = target.board
    per_bin = defaultdict(list)

    print(f"using {args.calibration}"
          + (f", saving samples to {args.save}/" if args.save else ""))
    with StereoRGBD(args.calibration) as cam:
        # Rectified frame: distortion is already removed, so PnP uses the
        # rectified intrinsics with zero distortion.
        K = cam.K
        zero = np.zeros(5)
        for _ in range(10):
            cam.read()

        print("Hold the board in view and sweep 0.3 -> 1.5 m. q to finish.")
        while True:
            f = cam.read()
            # The rectified right eye, for saved samples only.
            right_rect = cam._last_right if args.save else None
            gray = cv2.cvtColor(f.rgb, cv2.COLOR_BGR2GRAY)
            view = f.rgb.copy()
            status, colour = "no board in view", (0, 0, 255)
            here_dist = None

            n_corners, pts_flat, raw_a, raw_b = target.detect(gray)
            if n_corners:
                target.draw(view, raw_a, raw_b)

            if n_corners >= target.min_corners:
                obj, img = target.pose_points(raw_a, raw_b)
                if obj is None or len(obj) < 4:
                    status = f"{n_corners} corners but only "\
                             f"{0 if obj is None else len(obj)} matched the board"
                    colour = (0, 200, 255)
                    ok, rvec, tvec = False, None, None
                else:
                    ok, rvec, tvec = cv2.solvePnP(obj, img, K, zero)
                dist_pnp = float(tvec[2].item()) if ok else 0.0
                here_dist = dist_pnp if ok else None
                lo, hi, _ = BINS[MODE]
                if ok and lo - 0.1 < dist_pnp < hi + 0.6:
                    pts = pts_flat.astype(np.int32)
                    mask = np.zeros(gray.shape, np.uint8)
                    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
                    # Erode so pixels where SGBM straddles the board edge do
                    # not pollute the comparison.
                    mask = cv2.erode(mask, np.ones((15, 15), np.uint8))
                    expect = board_plane_depth(gray.shape, K, rvec, tvec)
                    good = (mask > 0) & np.isfinite(f.depth) & np.isfinite(expect)

                    if good.sum() >= 500:
                        err = f.depth[good] - expect[good]
                        bias = float(np.median(err))
                        noise = float(err.std())
                        step = BINS[MODE][2]
                        b = round(round(dist_pnp / step) * step, 2)
                        if args.save and len(per_bin.get(b, [])) < 3:
                            # A few per bin is plenty for a parameter sweep,
                            # and keeps the directory small.
                            sd = Path(args.save)
                            sd.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(
                                sd / f"{b:.2f}_{len(per_bin.get(b, []))}.npz",
                                left=f.rgb, right=right_rect,
                                rvec=rvec, tvec=tvec, K=K)
                        per_bin[b].append((dist_pnp, bias, noise,
                                           good.sum() / max((mask > 0).sum(), 1)))
                        n_now = len(per_bin[b])
                        if n_now % 5 == 0:
                            beeper.play(beeper.tick)
                        if n_now >= TARGET_PER_BIN and b not in was_full:
                            was_full.add(b)
                            remaining = [x for x in target_bins()
                                         if len(per_bin.get(x, [])) < TARGET_PER_BIN]
                            beeper.play(beeper.all_done if not remaining
                                        else beeper.bin_done)
                        status = (f"{dist_pnp:.3f}m  bias {bias*1000:+.0f}mm "
                                  f"({bias/dist_pnp*100:+.1f}%)  "
                                  f"noise {noise*1000:.0f}mm")
                        colour = (0, 255, 0)
                        cv2.drawContours(view, [cv2.convexHull(pts)], -1,
                                         (255, 255, 0), 1)
                    else:
                        status = f"{dist_pnp:.3f}m  too little valid depth on board"
                        colour = (0, 200, 255)
                else:
                    status = (f"board at {dist_pnp:.2f}m — outside "
                              f"{lo-0.1:.1f}-{hi+0.6:.1f}m")
                    colour = (0, 200, 255)
            elif n_corners:
                status = f"only {n_corners}/{target.total} corners — move closer"
                colour = (0, 200, 255)

            panel = np.hstack([view, depth_colormap(f.depth)])
            cv2.putText(panel, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, colour, 2)
            done, total, short = draw_progress(panel, per_bin, here_dist)
            if short:
                nearest = min(short, key=lambda b: abs(b - (here_dist or 0.5)))
                hint = f"MOVE TO {nearest:.2f}m  ({len(short)} bins left)"
            else:
                hint = "ALL BINS FILLED - press q"
            scale = 1.1 if MODE == "far" else 0.6
            cv2.putText(panel, f"{done}/{total} bins   {hint}",
                        (10, 52 if MODE == "near" else 62),
                        cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (0, 255, 0) if not short else (0, 220, 255),
                        2 if MODE == "near" else 3)
            cv2.imshow("board validation — q to finish", panel)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    cv2.destroyAllWindows()

    if not per_bin:
        print("\nNo usable samples collected.")
        return

    print(f"\n{'PnP dist':>9}  {'n':>4}  {'bias':>9}  {'bias %':>7}  "
          f"{'noise':>8}  {'fill':>6}")
    for b in sorted(per_bin):
        rows = per_bin[b]
        d = np.mean([r[0] for r in rows])
        bias = np.mean([r[1] for r in rows])
        noise = np.mean([r[2] for r in rows])
        fill = np.mean([r[3] for r in rows]) * 100
        print(f"{d:8.3f}m  {len(rows):4d}  {bias*1000:+7.1f}mm  "
              f"{bias/d*100:+6.2f}%  {noise*1000:7.1f}mm  {fill:5.0f}%")

    all_rows = [r for rows in per_bin.values() for r in rows]
    biases = np.array([r[1] / r[0] for r in all_rows])
    print(f"\n{len(all_rows)} samples over {len(per_bin)} distance bins")
    print(f"mean relative bias  {biases.mean()*100:+.2f}%  "
          f"(sd {biases.std()*100:.2f}%)")
    print("\nA roughly CONSTANT % bias => baseline/scale offset.")
    print("A bias growing with distance => rectification or disparity offset.")
    print("'noise' is scatter about the board's TRUE plane: real depth noise,")
    print("not scene structure.")


if __name__ == "__main__":
    main()
