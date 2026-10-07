"""Sweep SGBM parameters against the board's true plane, offline.

Capture once with `stereokit validate --save DIR`, then evaluate any number of
parameter sets against the same frames. Live sweeping would need a fresh
board wave per setting and could not hold the scene constant between them.

Ground truth is the board's solvePnP plane, so this measures real ACCURACY.
That matters because the obvious proxies do not: valid-pixel count and
spatial smoothness both improve monotonically with block size, but a larger
block buys smoothness by fattening depth edges, which those metrics reward
and accuracy does not.

Run with:  uv run tools/tune_sgbm.py samples/ calibration.json
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from stereokit import board as B
from stereokit.depth import DepthEstimator
from stereokit.rectify import Rectifier
from stereokit.validate import board_plane_depth


def load(sample_dir):
    out = []
    for f in sorted(Path(sample_dir).glob("*.npz")):
        z = np.load(f)
        out.append((z["left"], z["right"], z["rvec"], z["tvec"], z["K"],
                    float(z["tvec"].ravel()[2])))
    return out


def board_mask(shape, K, rvec, tvec):
    """Pixels covered by the physical board, from its known geometry.

    The plane from solvePnP is INFINITE, so masking by "depth close to the
    board distance" selects most of the image and compares SGBM against a
    plane that is not there. Project the board's real corners instead -- the
    same region stereokit validate masks live, reconstructed from the saved pose
    so no re-capture is needed.
    """
    obj = B.make_board().getChessboardCorners().astype(np.float64)
    pts, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))
    hull = cv2.convexHull(pts.reshape(-1, 2).astype(np.int32))
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, hull, 255)
    return m


def edge_bleed(depth, mask, expect, dist):
    """How far the board's depth leaks PAST its real edge, in pixels.

    The plane metric alone is biased toward large blocks: the board really is
    flat, so smoothing moves the measurement toward truth. The board's border
    is a genuine depth discontinuity, so measuring how far its depth bleeds
    outward captures the cost that the plane metric rewards.
    """
    outside = cv2.dilate(mask, np.ones((31, 31), np.uint8)) & ~mask
    sel = (outside > 0) & np.isfinite(depth) & np.isfinite(expect)
    if sel.sum() < 100:
        return float("nan")
    # Fraction of just-outside pixels still reporting the board's depth.
    return float((np.abs(depth[sel] - expect[sel]) < 0.02 * dist).mean()) * 100


def evaluate(est, samples):
    """Median |error| against the true plane, plus noise and fill."""
    biases, noises, fills, bleeds = [], [], [], []
    for left, right, rvec, tvec, K, dist in samples:
        _, depth = est.compute(left, right)
        expect = board_plane_depth(depth.shape, K, rvec, tvec)
        # Board interior only, eroded so SGBM straddling the edge does not
        # pollute the comparison.
        mask = cv2.erode(board_mask(depth.shape, K, rvec, tvec),
                         np.ones((15, 15), np.uint8))
        good = (mask > 0) & np.isfinite(depth) & np.isfinite(expect)
        if good.sum() < 300:
            continue
        err = depth[good] - expect[good]
        biases.append(abs(float(np.median(err))) / dist)
        noises.append(float(err.std()) / (dist * dist))     # normalise by Z^2
        fills.append(good.sum() / max((mask > 0).sum(), 1))
        bleeds.append(edge_bleed(depth, mask, expect, dist))
    if not noises:
        return None
    return (float(np.mean(biases)) * 100, float(np.mean(noises)),
            float(np.mean(fills)) * 100, float(np.nanmean(bleeds)))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    samples = load(sys.argv[1])
    if not samples:
        print(f"no samples in {sys.argv[1]}")
        return 1
    rect = Rectifier.from_file(sys.argv[2])
    print(f"{len(samples)} samples, "
          f"{min(s[5] for s in samples):.2f}-{max(s[5] for s in samples):.2f} m\n")

    print(f"{'blockSize':>9} {'uniq':>5} {'bias %':>8} {'dd px':>8} "
          f"{'fill %':>7} {'bleed %':>8}")
    best = None
    for bs in (3, 5, 7, 9, 11, 13):
        for uniq in (5, 10, 15):
            est = DepthEstimator(fx=rect.fx, baseline_m=rect.baseline_m,
                                 block_size=bs, uniqueness=uniq)
            r = evaluate(est, samples)
            if r is None:
                continue
            bias, noise_norm, fill, bleed = r
            # noise/Z^2 * f*B is the implied subpixel matching error
            dd = noise_norm * rect.fx * rect.baseline_m
            print(f"{bs:9d} {uniq:5d} {bias:7.2f}% {dd:7.3f} {fill:6.1f}% "
                  f"{bleed:8.1f}%")
            if best is None or dd < best[0]:
                best = (dd, bs, uniq, fill)
    if best:
        print(f"\nbest matching accuracy: blockSize={best[1]} "
              f"uniqueness={best[2]} -> {best[0]:.3f} px, fill {best[3]:.1f}%")
        print("\nCAVEAT: dd alone is biased toward LARGE blocks, because the"
              "\nboard really is flat and smoothing moves a planar"
              " measurement toward"
              "\ntruth. 'bleed %' is the counterweight: the share of pixels"
              " just OUTSIDE"
              "\nthe board still reporting its depth, i.e. edge fattening."
              " Pick a block"
              "\nsize where dd has mostly plateaued but bleed has not yet"
              " climbed.")


if __name__ == "__main__":
    sys.exit(main() or 0)
