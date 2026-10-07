# SKL-4689-220 characterisation and validation

> **Superseded.** This records the state on 2026-09-30, before intrinsics
> were refined per camera. Several figures here — notably f·B, focal
> length and the depth-error table — changed as a result. See
> [RESULTS.md](../../../RESULTS.md) for current numbers. Kept for the
> reasoning and the method, not the values.

Date: 2026-09-30
Hardware: SKL-4689-220 (SunplusIT SPCA2100), `/dev/video4`, USB `2048:2088`
Mode calibrated: `1280x480` (640×480 per eye), MJPG

Vendor documentation is useless: the supplied PDF is for a different model
(SKL-2MP-220), claims 2 MP against a 3.3 MP device, and never mentions stereo.
The only usable figure is a 2.8 mm lens. Everything below is measured.

## Headline

| | SKL | AR0144 |
|---|---|---|
| Baseline | **16.98 mm** (claimed 20, 18% high) | 51.92 mm |
| Epipolar error, held out | **0.052 px** | 0.145 px |
| Depth bias | **+0.08%** (sd 0.61%) | +0.10% (sd 0.13%) |
| Subpixel matching @ 0.5 m | **0.18 px** | 0.25 px |
| f·B | 10.62 | 35.45 |
| Implied pixel pitch | 2.07 µm | 3.0 µm |
| Usable fps for SLAM | **30** (feature-tuned exposure) | 60 |
| Shutter | **global** (measured) | global |

**The baseline was the question this camera was bought to answer: 16.98 mm,
not the 20 mm advertised.**

## Modes

Ten side-by-side stereo modes, five mono — each mono mode is exactly half the
width of a stereo one. All measured by half-correlation (0.98–0.99 at a small
offset for stereo; 0.78–0.84 at nonsense offsets for mono).

| Stereo | Per eye | Mono twin |
|---|---|---|
| 3040×1080 | 1520×1080 | 1520×1080 |
| 2560×960 | 1280×960 | 1280×960 |
| 2560×720 | 1280×720 | 1280×720 |
| 1600×600 | 800×600 | 800×600 |
| 1280×480 | 640×480 | 640×480 |

Plus five portrait-eye modes (2176×1520 → 1088×1520, etc.) that see a taller
field than the 3040 reference — a different sensor crop, not a rescale.

## The eyes are reversed

**This device presents its halves as `[right|left]`.** `T[0]` came out positive
and `P2[0,3]` positive, where both are negative on a conventional rig.

Left uncorrected this inverts the sign of disparity. SGBM only searches
positive disparity, so instead of failing it latched onto spurious matches,
producing **9.6% valid pixels and a −51% apparent scale error** — a plausible
number that reads, by the validation tool's own guidance, as a baseline
problem. After setting `swap_eyes=True`: **54.3% valid pixels and +0.08% bias.**

`export.py` already asserted `P2[0,3] < 0` and would have rejected the
calibration outright — but export runs downstream of anyone actually using the
depth, so the check never fired in time. `solve()` now fails on it directly.
A gate in the wrong place is barely a gate.

## Depth validation

903 samples over 7 distance bins, against the board's own `solvePnP` plane:

| PnP distance | n | Bias | Bias % | Noise | Fill |
|---|---|---|---|---|---|
| 0.198 m | 221 | +0.4 mm | +0.18% | 4.7 mm | 84% |
| 0.300 m | 135 | +0.5 mm | +0.16% | 3.0 mm | 84% |
| 0.393 m | 126 | +0.4 mm | +0.10% | 3.2 mm | 96% |
| 0.498 m | 134 | +0.5 mm | +0.10% | 4.2 mm | 100% |
| 0.599 m | 91 | +0.5 mm | +0.09% | 5.8 mm | 100% |
| 0.702 m | 167 | −0.4 mm | −0.06% | 6.6 mm | 100% |
| 0.771 m | 29 | −2.5 mm | −0.32% | 5.9 mm | 100% |

**Mean relative bias +0.08%, sd 0.61%** — flat with distance, so no baseline
or rectification offset.

### Matching quality beats the AR0144

Back-solving `Δd = noise·f·B/Z²`:

| Z | SKL | AR0144 |
|---|---|---|
| 0.2 m | 1.27 px | 0.53 px |
| 0.5 m | **0.18 px** | 0.25 px |
| 0.77 m | **0.11 px** | 0.19 px |

Past 0.3 m the SKL matches *better* than the AR0144 and beats the model's
0.25 px assumption. Its worse depth precision is entirely the short baseline,
not the optics.

The 0.198 m outlier (1.27 px) is almost certainly **motion blur**: the board is
hand-held at 120 ms exposure and angular velocity peaks when closest. Testable
by repeating the near bins under more light.

## Frame rate: 30 fps IS achievable (earlier claim retracted)

An earlier version of this document concluded the camera was "light-starved"
and limited to 16.6 fps. **That was wrong, because it optimised the wrong
metric.** Mean brightness is what a photographer wants; SLAM consumes
features, and the two point in opposite directions here.

| Exposure | fps | mean | blown % | gradient | ORB kp |
|---|---|---|---|---|---|
| 10 ms | 30.0 | 40 | 0.31% | 3.54 | 170 |
| **20 ms** | **30.0** | 64 | 0.38% | 3.93 | **181** |
| 30 ms | 30.0 | 84 | 0.70% | 4.43 | 172 |
| 60 ms | 16.6 | 127 | 2.09% | 4.26 | 116 |
| 120 ms | 8.3 | 171 | 6.09% | 3.67 | 61 |
| 180 ms | 5.5 | 202 | 10.35% | 3.42 | 14 |

Long exposure brightens the frame by **clipping** it — blown pixels go 0.4% to
10% — and clipped regions carry no gradient. Feature yield peaks at 20–30 ms,
which is exactly the 30 fps frame period. Valid depth coverage is nearly flat
across the whole range (38–44%), so nothing is lost.

Running short is therefore a triple win for SLAM while moving: **30 fps,
~3x the features, and far less motion blur.** The image looks dark to a human,
which is irrelevant.

`StereoCamera(exposure="features")` sweeps the candidates and picks the one
maximising gradient energy, capped at the frame period so it can never
throttle the sensor. Measured: 29.9 fps, 170 keypoints.

## Brightness versus exposure (for reference)

2.07 µm pixels have about half the area of the AR0144's 3.0 µm. UVC exposure is
in 100 µs units, and an exposure longer than the frame period throttles the
sensor:

| Exposure | fps | Mean brightness |
|---|---|---|
| 30 ms | 30.0 | 50 (dark) |
| 60 ms | 16.6 | 81 |
| 120 ms | 8.3 | 118 |
| auto | 16.6 | 151 |

More light would still help the 0.198 m noise outlier, which looks like motion
blur from hand-holding. But it is NOT needed for frame rate.

## Shutter: global, measured

Shaking the camera showed no skew, which is suggestive but weak — hand motion
is often too slow to reveal a row delay. Measured properly with a flashing
fullscreen window at 5 ms exposure, comparing banding WITHIN a frame against
variation BETWEEN frames. A rolling shutter puts the screen's mid-frame toggle
into a hard horizontal band; a global shutter flips whole frames.

```
frames analysed            955
banding WITHIN a frame     2.13 DN    (noise floor)
variation BETWEEN frames  31.28 DN
ratio                      0.07       (rolling would exceed 1.0)
```

**Global shutter**, same as the AR0144. No skew during motion, so geometry
stays valid while the camera moves — the property that matters most for SLAM
and the one no frame rate can compensate for.

A prior pointed the other way and was wrong: the device exposes a
`power_line_frequency` control, which is anti-banding machinery that only
makes sense on a rolling sensor. The AR0144 exposes it too and is also global,
so it carries no information. Worth measuring rather than reasoning about.

## 2560x720: the recommended mode (measured 0.9-3.8 m)

Calibrated from 54 pairs. Baseline 17.01 mm, a THIRD independent confirmation
after 16.98 (1280x480) and 16.99 (1600x600) -- three modes, three sessions,
0.1% spread. Reprojection RMS 292-311 urad, the best of four calibrations
(the others are 350-371), which is why the absolute-pixel RMS gate had to be
replaced with an angular one.

Near validation: bias **+0.14%** (sd 0.27%), the tightest measured.

Far validation, 21 banked samples re-evaluated offline:

| Z | bias | noise | implied dd | fill |
|---|---|---|---|---|
| 0.93 m | −0.13% | 12 mm | 0.317 px | 100% |
| 1.31 m | +0.02% | 24 mm | 0.313 px | 100% |
| 1.77 m | +1.38% | 42 mm | 0.299 px | 100% |
| 2.25 m | −2.37% | 62 mm | 0.269 px | 100% |
| 2.76 m | −1.39% | 54 mm | 0.157 px | 100% |
| 3.28 m | +0.70% | 152 mm | 0.311 px | 100% |
| 3.78 m | +3.44% | 200 mm | 0.307 px | 100% |

**Subpixel matching does not collapse at range.** dd holds at 0.27-0.32 px
from 0.9 to 3.8 m, where disparity is only 5.8 px. That was the open question
and it is settled: this camera degrades smoothly, it does not fall off a cliff.

f·B = 21.97. With the measured dd = 0.29:

| Z | SKL 800×600 | **SKL 1280×720** | Gemini 305 |
|---|---|---|---|
| 1 m | 21 mm | **13 mm** | 24 mm |
| 2 m | 84 mm | **53 mm** | 97 mm |
| 3 m | 188 mm | **119 mm** | 219 mm |
| 4 m | 335 mm | **211 mm** | 389 mm |

At 2560×720 this camera is roughly **twice as precise as the Gemini 305 at
every range**, trading the Gemini's 88° field for 58°.

### Two findings from the offline re-evaluation

`max_depth=4.0` distorts far measurements: at 3.78 m it cut fill to 60% and
reported +0.29% bias where the unclamped truth is 100% fill and +3.44%. It
hides error rather than creating it. Raise it to 6 m for any far work.

`numDisparities=64` gives bit-identical results to 192 across every sample.
It sets only the NEAR limit, so 64 costs nothing and runs 2.3x faster.

## Focus: the lens is focused far

Measured from banked samples, no extra capture:

| Range | Mean edge rise |
|---|---|
| < 0.40 m | 3.46 px |
| 0.40-0.70 m | 2.50 px |
| > 0.70 m | 2.23 px (best 1.95) |

Monotonic, so the near softness is defocus. Edges are 75% wider at 0.24 m than
at best focus, which explains the anomalous close-range depth noise (68.8 mm at
0.24 m) and fill dropping to 63%. **Usable near edge is ~0.4 m**, not the
0.22 m the disparity range allows.

The SKL is softer than the AR0144 overall (2.23 px against 1.66 px at best
focus) -- expected, since 2.07 um pixels resolve more of the same lens's blur
than 3.0 um ones.

## Flare

A bright source in frame throws asymmetric veiling glare across one eye. This
was initially mis-measured as a 19% intrinsic per-eye colour mismatch; with the
source out of frame the eyes agree to 0.2% in R/G. Real and intrinsic is a
3–7% brightness difference, plus a spectral-response difference that opens up
at extreme white balance (+64% R/G locked at 2800 K, ~1% at 4600 K or auto).

Global normalisation cannot correct flare, since it varies spatially. Keep
strong sources out of frame, especially during calibration.

## Recommended next step: recalibrate at 3040×1080

The matching quality makes this compelling. At 1520 px per eye, `f` rises to
roughly 1400 px rectified, taking `f·B` from 10.6 to about **23.8**. With the
measured 0.15 px matching rather than the model's 0.25:

| Z | 640×480 now | 1520×1080 projected |
|---|---|---|
| 0.5 m | ±3.5 mm | **±1.6 mm** |
| 1.0 m | ±14 mm | **±6.3 mm** |
| 2.0 m | ±57 mm | **±25 mm** |

That is AR0144-class depth precision from a third of the baseline, bought
entirely with resolution. SGBM on 5× the pixels runs around 14 fps, which is
below the camera's own exposure-limited 16.6 fps in this lighting — so the
compute cost is effectively free.

Requires its own capture and solve, since calibration is per-mode.
