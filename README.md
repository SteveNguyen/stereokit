# stereokit

Calibrate, validate and visualise depth from side-by-side USB stereo cameras.

Built while evaluating two modules for RTAB-Map SLAM on Reachy. Both are
characterised and ship with working calibrations:

| Camera | Baseline | Best mode | Depth error at 4 m |
| --- | --- | --- | --- |
| **Waveshare AR0144** | 51.96 mm | 2560x720 | **77 mm** |
| SKL-4689-220 | 17.01 mm | 2560x720 | 215 mm |

**The AR0144 is the better choice for SLAM** — 2.4x the depth precision and
13° more field. Full numbers and the reasoning: [RESULTS.md](RESULTS.md).

## Install

```bash
uv sync --extra dev
uv run stereokit --help
```

Plug the camera in and check it is found:

```bash
uv run stereokit view --raw --camera ar0144
```

That shows the raw `[left|right]` frame. If nothing opens, the camera was not
matched — see [Adding another camera](#adding-another-camera).

## See depth right now

No calibration needed; both cameras' calibrations are in the repo.

```bash
# AR0144 at 2560x720 (recommended)
uv run stereokit view --calibration calibration_ar0144_2560x720.json

# SKL-4689-220 at 2560x720
uv run stereokit view --calibration calibration_skl_2560x720.json
```

Left pane is the rectified colour image, right pane is depth, colour-mapped
over the estimator's range with the valid-pixel percentage overlaid. Press `q`
to quit.

Pass the wrong filename and the tool lists every calibration it can find, with
the camera and mode each belongs to. Calibration is per camera **and** per
mode, so `calibration.json` (AR0144 at 640x480) will not work for a camera
running at 2560x720.

## Recalibrate

Needed if you have a different physical unit, a different mode, or the lens
has been refocused or knocked.

**1. Print the board.** `charuco_A4.pdf` is in the repo, or regenerate it:

```bash
uv run stereokit board --out charuco_A4.pdf
```

Print at 100% on matte paper and mount it flat — on foam board or glass, not
held in the hand. Then **measure a square with calipers**: print scaling is
the one error the maths cannot detect, because a wrong square size scales the
recovered baseline and the reference distance by the same factor.

**2. Capture pairs.**

```bash
uv run stereokit capture --camera ar0144 --mode 2560x720 \
    --out captures_mycam --square-mm 25.0
```

Pass the square size you measured. The tool tracks coverage across the frame,
board tilt and distance, and tells you which bins are still empty — fill them
all. Move the board slowly; it rejects blurred and moving frames. Aim for 50+
accepted pairs.

The output directory is gitignored: it holds the detected board corners, about
2 KB a pair. Keep it until the calibration is validated — it lets you re-run
`solve` with different settings without picking the board up again.

**3. Solve.**

```bash
uv run stereokit solve captures_mycam --camera ar0144 \
    --out calibration_mycam.json
```

This prints reprojection errors, the recovered baseline and focal length, and
either `PASSED` or the gates that failed. A failing solve writes
`*.rejected.json` for diagnosis and refuses to write the real file, so nothing
downstream can load it by accident.

**4. Validate.** Do not skip this.

```bash
uv run stereokit validate calibration_mycam.json --target near --save samples_near
uv run stereokit validate calibration_mycam.json --target far  --save samples_far
```

`near` sweeps the ChArUco board from 0.2–1.0 m; `far` uses the 85 mm ArUco
grid (`charuco_far_A4.pdf`) from 1–4 m. Both measure real depth against the
board's true plane and report bias, noise and fill per distance bin. Audio
feedback is on by default, because the screen is unreadable from 4 m away.

`--save` banks the frames so parameters can be swept offline later without
re-capturing.

**A passing solve is not sufficient — run the far sweep.** On the SKL, one
solver change *improved* epipolar error while making depth scale 4% worse.
Epipolar error measures row alignment only; nothing but a far sweep against an
independent distance measurement catches a scale error.
[RESULTS.md](RESULTS.md#intrinsic-refinement-is-a-per-camera-decision) has the
full story.

## Export for RTAB-Map

```bash
uv run stereokit export calibration_ar0144_2560x720.json --out rtabmap/
```

Writes ROS `camera_info` YAML for each eye. Set `Vis/MinDepth` and
`Vis/MaxDepth` to match the camera's working range — leaving MaxDepth
unbounded lets far, noisy points dominate.

## Adding another camera

Only mildly supported, and it needs a code edit. Cameras are described by a
`CameraProfile` in `stereokit/profiles.py`: which `/sys/class/video4linux`
name to match, the stereo modes, whether the halves arrive as `[right|left]`,
and whether intrinsics should be refined jointly. Copy the `SKL2MP220` entry
and adjust.

Two things the existing profiles learned the hard way:

- **Check the eye order.** The SKL presents its halves as `[right|left]`.
  Uncorrected, disparity is inverted; SGBM then latches onto spurious matches
  and reports a plausible-looking 9.6% valid and −51% scale error rather than
  failing outright. `swap_eyes=True` fixes it.
- **Decide `joint_intrinsics` by measurement, not by default.** See
  [RESULTS.md](RESULTS.md#intrinsic-refinement-is-a-per-camera-decision).

Discovering an unknown camera's modes — which are stereo, which are mono, what
the pixel pitch is — was done by hand for the SKL and is not automated.

## Evaluation tools

In `tools/`, not part of the main workflow. Each runs against banked samples
or the camera directly:

| Tool | What it answers |
| --- | --- |
| `tools/tune_sgbm.py` | Which SGBM parameters, measured against the board's true plane |
| `tools/focus_check.py` | Is the lens focused, and at what distance |
| `tools/shutter_test.py` | Rolling or global shutter, measured not assumed |

```bash
uv run tools/tune_sgbm.py samples_near/ calibration_ar0144_2560x720.json
```

## Repository layout

```
stereokit/          the package: camera, board, calibrate, depth, rectify,
                    rgbd, export, validate, profiles, cli
tools/              evaluation instruments
tests/              102 tests, no hardware needed
calibration*.json   solved calibrations, one per camera and mode
charuco_A4.pdf      the calibration board
charuco_far_A4.pdf  the 85 mm far-range target
docs/               design spec, implementation plan, dated session results
```

Capture and validation output is not committed. `stereokit capture` writes
`captures*/` and `stereokit validate --save` writes `samples*/`; both are
gitignored, since you generate your own when calibrating your own unit. The
solved `calibration*.json` is the artifact worth keeping, and those are here.

## Tests

```bash
uv run pytest -q
```

102 tests, none of which need a camera. Geometry is checked against
synthetic scenes with known ground truth.
