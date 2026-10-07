# Measured results: AR0144 and SKL-4689-220

Last updated 2026-10-07. Every figure here is measured, not from a datasheet.
Calibration figures are read from the shipped `calibration*.json`; depth
figures come from board sweeps described under [How this was measured](#how-this-was-measured).

**Recommendation: use the Waveshare AR0144 at 2560x720 for SLAM.** It has
2.4x the depth precision of the SKL and 13 degrees more horizontal field. The
SKL's only advantage is that its short baseline sees closer.

## Head to head at 2560x720

| Measure | AR0144 | SKL-4689-220 |
| --- | --- | --- |
| Baseline, measured | 51.96 mm | 17.01 mm |
| Baseline, vendor claim | 52 mm (0.1% off) | 20 mm (18% high) |
| f·B | 51.66 | 21.97 |
| Horizontal x vertical field | 65.5° x 39.8° | 52.7° x 31.2° |
| Epipolar error, held out | 0.114 px | 0.166 px |
| f·B scale error vs PnP | **−0.17%** | −1.44% |
| Subpixel matching, 1.2–3.8 m | 0.098 px | 0.295 px |
| Depth error at 4 m | **77 mm** | 215 mm |
| Near limit, numDisparities 256 | 20.2 cm | 8.6 cm |
| Shutter | global | global |

`f·B` is focal length in pixels times baseline in metres. It sets depth
precision directly: random depth error is matching error x Z² / f·B. The
AR0144's is 2.4x larger and its matching is 3x better, which compounds to
roughly 7x less noise at 4 m.

## All calibrated modes

| Calibration file | Camera | Eye | Baseline | f·B | HFOV | Epipolar |
| --- | --- | --- | --- | --- | --- | --- |
| `calibration_ar0144_2560x720.json` | AR0144 | 1280x720 | 51.96 mm | 51.66 | 65.5° | 0.114 px |
| `calibration.json` | AR0144 | 640x480 | 51.91 mm | 35.39 | 50.3° | 0.142 px |
| `calibration_skl_2560x720.json` | SKL | 1280x720 | 17.01 mm | 21.97 | 52.7° | 0.166 px |
| `calibration_skl_1600x600.json` | SKL | 800x600 | 16.99 mm | 13.85 | 52.3° | 0.071 px |
| `calibration_skl.json` | SKL | 640x480 | 16.98 mm | 10.62 | 54.2° | 0.052 px |

The AR0144's 480-height modes crop to roughly two-thirds of the sensor width,
so they see 50° rather than 65°. That is a different field, not a rescale.

Each camera's baseline agrees across every mode and session to about 0.1%
(AR0144 51.91/51.96, SKL 16.98/16.99/17.01). That consistency is the main
reason to trust the numbers: baseline is solved independently per mode.

## AR0144 depth accuracy at 2560x720

48 samples in two sweeps, measured against the board's own solvePnP plane.
Valid-pixel fill is 100% across the whole range.

| Distance | Bias | Noise | Matching | Fill |
| --- | --- | --- | --- | --- |
| 0.28 m | −0.03% | 0.9 mm | 0.55 px | 98% |
| 0.46 m | −0.22% | 1.3 mm | 0.31 px | 100% |
| 0.67 m | −0.15% | 2.2 mm | 0.26 px | 100% |
| 0.96 m | −0.23% | 4.0 mm | 0.22 px | 100% |
| 1.32 m | −0.88% | 5 mm | 0.16 px | 100% |
| 1.77 m | −1.22% | 11 mm | 0.18 px | 100% |
| 2.76 m | −2.56% | 13 mm | 0.09 px | 100% |
| 3.76 m | −2.86% | 11 mm | 0.04 px | 100% |

**Matching improves with distance** rather than collapsing: 0.25 px near,
0.098 px median beyond 1.2 m. Random error at 3.76 m is 11 mm.

The bias column is systematic and grows with distance, which is diagnostic. A
scale error would be constant; a bias proportional to Z means a fixed
disparity offset instead. Fitting `bias(Z) = a + bZ` separates them: the
constant term is **−0.17%**, so f·B is essentially exact, and the slope
corresponds to a **0.395 px disparity offset**.

Subtracting that offset would leave −0.17% at every range. It is **not
applied** — it was measured against one target and has not been shown to
transfer to arbitrary scenes.

| Range | Random error | Systematic, uncorrected |
| --- | --- | --- |
| 1 m | 5 mm | −7 mm |
| 2 m | 19 mm | −30 mm |
| 3 m | 44 mm | −67 mm |
| 4 m | 77 mm | −119 mm |

## Intrinsic refinement is a per-camera decision

The AR0144's first 2560x720 calibration failed its epipolar gate at 0.538 px
against a 0.3 px limit. The cause was the solver, not the camera.

`stereoCalibrate` was freezing each eye's intrinsics with
`CALIB_FIX_INTRINSIC`, so only the rigid transform between the eyes could
absorb any left/right inconsistency. Monocular calibration has a known
degeneracy — focal length, principal point and radial distortion trade off
against each other and against board pose — so each eye lands at a slightly
different point in that valley. Both fit their own data well while disagreeing
with each other, and a rigid rotation plus translation cannot reconcile an
intrinsics mismatch. It surfaces as epipolar error.

Refining jointly fixed the AR0144 and broke the SKL:

| Camera | Baseline | Frozen: f·B scale error | Joint: f·B scale error |
| --- | --- | --- | --- |
| AR0144 | 52 mm | +5.69% | **−0.17%** |
| SKL-4689-220 | 17 mm | **−1.44%** | −5.57% |

Pinning down focal length jointly needs stereo leverage. With 52 mm of
baseline the two eyes see genuinely different views and `f` is well
constrained; with 17 mm they see nearly the same view and `f` drifts. The
setting therefore lives on `CameraProfile.joint_intrinsics`, per camera.

**How this was nearly missed matters more than the finding.** Epipolar error
*improved* under joint refinement on **both** cameras — 0.183 to 0.137 px on
the SKL — while that camera's depth scale was getting 4% worse. Epipolar
measures row alignment only. Nothing short of a far sweep against an
independent distance measurement distinguishes a calibration that aligns rows
from one that also has the scale right.

**If you calibrate a new camera, run `stereokit validate --target far`.** A
passing epipolar gate is not sufficient.

An OpenCV 5 trap sits underneath this: `stereoCalibrate` now *defaults* its
`flags` argument to `CALIB_FIX_INTRINSIC`, where 4.x defaulted to 0. Omitting
the argument silently freezes the intrinsics.

A recorded dead end: an 11-coefficient thin-prism plus tilted-sensor
distortion model also cleared the epipolar gate, at 0.160 px, and survived
five-fold cross-validation. It was fitting around the frozen intrinsics, not
describing the lens. Residual analysis confirmed the standard 5-coefficient
model describes these lenses well — spatial coherence of the monocular
residuals was 0.19 to 0.30, i.e. noise.

## Known limits

**Frame rate at 2560x720 is not verified.** The driver advertises 30 fps and
the AR0144 was measured at 60 fps at 1280x480, but neither camera has been
timed at 2560x720 under a realistic exposure. On the SKL, an exposure longer
than the frame period throttled the sensor to 16.6 fps, so the advertised
figure is not safe to assume. This matters because frame rate governs motion
blur while the robot moves.

**Near limit.** The near limit is f·B / numDisparities, so the AR0144's longer
baseline costs it close range: 26.9 cm at the default 192, 20.2 cm at 256. For
work inside 30 cm, raise `numDisparities`. The cost is compute only.

**Both lenses are focused far.** Edges are about 37% wider at 0.2–0.35 m than
at 0.7–1.1 m on the AR0144, with the same pattern on the SKL. Measured as a
relative comparison; the absolute figure is not calibrated.

**The eyes do not match photometrically.** On the AR0144 the right eye runs
5.1% brighter near and 10.8% brighter far, with a small red-to-green
difference. This does not affect SGBM, which works on gradients, but it is
visible in side-by-side RGB. On the SKL most of an apparent 19% mismatch
turned out to be differential lens flare rather than sensor response, so keep
bright sources out of frame — especially during calibration.

**SGBM block size is not optimised.** Bias is flat at 0.18–0.20% and fill at
99.4% across block sizes 5 to 13. The edge-bleed counterweight saturated at
~100% on this data and could not discriminate, so no optimum is claimed.
`blockSize=7` is a reasonable default.

**The AR0144 at 640x480 is unvalidated under the current solver.** It was
re-solved with joint refinement and passes its gates, but no far sweep has
confirmed its scale.

## How this was measured

Ground truth is a printed ChArUco board's own solvePnP pose, not a wall. The
board is guaranteed planar, its pose gives a true reference plane so residuals
measure depth noise rather than wall flatness, and it self-reports distance,
so a bias-versus-range curve comes free from sweeping it around.

PnP distance scales with `f`, while stereo depth scales with `f·B`. Their
ratio is therefore insensitive to focal-length error and tests the baseline
and the triangulation chain end to end. It cannot detect a print-scale error,
because a wrong square size scales the recovered baseline and the PnP distance
by the same factor — absolute scale is anchored by physically measuring the
printed 100 mm bar on the board.

Two targets: the 7x10 ChArUco board at 25 mm squares (`charuco_A4.pdf`) for
0.2–1.0 m, and a 2x2 ArUco grid at 85 mm (`charuco_far_A4.pdf`) for 1.2–3.8 m.

Calibration holds out 20% of pairs that the solver never sees, because
residual on the fitting set measures fit rather than accuracy. Acceptance
gates are expressed in microradians rather than pixels so they compare fairly
across resolutions; a pixel-denominated gate rejected the angularly best
calibration in an earlier round.

`stereokit validate --save DIR` banks frames during each sweep, so any
parameter can be re-evaluated offline against the identical scene. The
numDisparities sweep, the SGBM comparison and the focus curve were all
produced that way, with no re-capture. The frozen-versus-joint comparison was
re-solved from the banked capture corners, likewise without the camera.

Neither the capture corners nor the sample frames are committed here — they
are working data, regenerated per calibration session. What ships is the
solved `calibration*.json` and the numbers above.

## Session history

These dated documents record how the results were arrived at. **Their numbers
predate the intrinsic-refinement fix and are superseded by this file.**

- `docs/superpowers/plans/2026-09-24-validation-results.md` — AR0144 at 640x480
- `docs/superpowers/plans/2026-09-30-skl-validation-results.md` — SKL characterisation
