# AR0144 hardware validation results

> **Superseded.** This records the state on 2026-09-24, before intrinsics
> were refined per camera. Several figures here — notably f·B, focal
> length and the depth-error table — changed as a result. See
> [RESULTS.md](../../../RESULTS.md) for current numbers. Kept for the
> reasoning and the method, not the values.

Date: 2026-09-24
Hardware: Waveshare AR0144 Stereo USB Camera (A), `/dev/video4`
Mode: `1280x480` (640×480 per eye), MJPG, 60 fps

## Calibration

Board printed at 100% on A4; the 100 mm scale bar measured correct, so squares
are a true 25.00 mm and no `--square-mm` override was needed. 154 pairs
captured, 123 fit / 31 held out.

| Check | Result | Limit | Verdict |
|---|---|---|---|
| RMS reprojection, left | 0.2358 px | < 0.3 | pass |
| RMS reprojection, right | 0.2356 px | < 0.3 | pass |
| RMS stereo | 0.2434 px | < 0.5 | pass |
| **Epipolar error (held out)** | **0.1451 px** | < 0.3 | pass, less than half the limit |
| **Baseline vs 52 mm nominal** | **51.920 mm (−0.15%)** | few % | pass |
| Recovered f vs mode prior | 636.5 vs 669.7 px (−5.0%) | < 10% | pass |

Derived: rectified `fx` 682.83 px, `f·B` 35.453, rectification ROI the full
`[0,0,640,480]` (no invalid border). Near limit at `numDisparities=192`:
0.185 m.

The **baseline check is the one that matters most** — it is the only automated
gate that catches a print-scale error, which survives every other test and
would appear downstream as a uniform depth bias. −0.15% confirms the print.

Two incidental findings: the vendor's 65° horizontal FOV is really 64.0°, and
the fitted distortion (`k1=0.165, k2=−0.356`) is not the "distortion-free"
the datasheet advertises. Neither matters — the standard 5-coefficient model
fits to 0.24 px — but the datasheet figures are marketing, not measurements.

## Depth accuracy — ChArUco plane validation

Method (`validate_board.py`): detect the board in the rectified left image,
recover its pose with `solvePnP`, and compare SGBM depth against the board's
**true** plane, pixel by pixel, over an eroded interior mask.

This is a better instrument than the planned flat-wall test. The board is
guaranteed planar, so plane residual measures genuine depth noise rather than
scene geometry; and it self-reports distance, so a bias-versus-range curve
comes free from sweeping it. The first wall attempt returned a 117 mm plane
RMS that turned out to be a wall *corner* with a storage box in it — the
failure mode this method removes entirely.

560 samples, 7 distance bins:

| PnP distance | n | Bias | Bias % | Noise | Fill |
|---|---|---|---|---|---|
| 0.232 m | 26 | +0.1 mm | +0.04% | 0.8 mm | 79% |
| 0.302 m | 137 | +0.3 mm | +0.09% | 0.9 mm | 85% |
| 0.396 m | 72 | +0.3 mm | +0.07% | 1.2 mm | 86% |
| 0.503 m | 72 | +0.5 mm | +0.10% | 1.8 mm | 94% |
| 0.604 m | 113 | +0.8 mm | +0.13% | 2.4 mm | 94% |
| 0.697 m | 102 | +0.8 mm | +0.11% | 2.8 mm | 94% |
| 0.784 m | 38 | +0.7 mm | +0.09% | 3.2 mm | 99% |

**Mean relative bias +0.10%, sd 0.13%.** Sub-millimetre at every range and
FLAT with distance — a constant percentage would indicate a baseline error, and
growth with range would indicate a rectification or disparity offset. Neither
is present.

### The §3 error model is confirmed

Back-solving subpixel matching accuracy from `Δd = noise·f·B/Z²`:

| Z | implied Δd |
|---|---|
| 0.23 m | 0.53 px |
| 0.50 m | **0.25 px** |
| 0.78 m | 0.19 px |

§3 assumed 0.25 px, which is exactly right mid-range. Accuracy degrades close
in — consistent with the §3.2 focus measurement showing the lens softening
below ~0.35 m — and improves with range as disparity shrinks.

### What this does NOT prove

PnP distance scales with `f` and stereo depth scales with `f·B`, so their ratio
is blind to focal-length error. This validates the **baseline and
triangulation chain to 0.1%**. Absolute scale is anchored by physically
measuring the printed 100 mm bar, independently corroborated by the recovered
baseline.

## Throughput

| Path | Measured |
|---|---|
| Raw stereo grab, 1280×480 | 60.1 fps (nominal 60) |
| Full RGBD (grab + rectify + SGBM + depth) | 56.8 fps |
| SGBM alone, `numDisparities` 128 / 192 / 256 | 75.9 / 69.7 / 60.6 fps |

The spec originally predicted ~17 fps at 192 and has been corrected. There is
no frame-rate-versus-near-range trade-off: `numDisparities=256` yields a
0.136 m near limit at 60.6 fps, still above the camera's own frame rate.

## Density

Valid-pixel fraction is entirely scene-dependent, as expected for passive
stereo with no projector:

- 79–99% on the textured ChArUco board
- 55–69% on a cluttered wall corner
- 27% on a largely blank wall

## Outstanding

- RTAB-Map mapping run (`rtabmap/ar0144_{left,right}.yaml` exported and both
  side-by-side traps verified: per-eye width 640, `P2[0,3] = −35.453`).
  Set `Vis/MinDepth 0.2`, `Vis/MaxDepth 4.0` per §9.4.
- The four §12 deferrals remain open with their triggers.
