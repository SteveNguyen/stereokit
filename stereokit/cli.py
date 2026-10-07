"""Command line entry points.

    stereokit board     --out charuco_A4.pdf
    stereokit capture   --out captures/ [--square-mm 24.8]
    stereokit solve     captures/ --out calibration.json
    stereokit validate  calibration.json [--target near|far] [--save DIR]
    stereokit view      [--raw] --calibration calibration.json
    stereokit record    --out rgbd/ --calibration calibration.json
    stereokit colour    calibration.json   (measure per-eye colour gains)
    stereokit export    calibration.json --out rtabmap/

Recalibrating is board -> capture -> solve -> validate. The last step is the
one that says whether the result is usable, so it is a subcommand rather than
a separate script you have to know exists.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import board as board_mod
from . import profiles
from . import calibrate, export
from . import rgbd as rgbd_mod
from .camera import DEFAULT_MODE, StereoCamera


def require_calibration(path):
    """Fail early and usefully when the calibration file is not there.

    Calibration is per camera AND per mode, so a repo holds several and no
    single default is right for everyone. Rather than pick one arbitrarily,
    say what is actually on disk -- the usual mistake is reaching for the
    default name when the mode you want was solved into another file.
    """
    if Path(path).exists():
        return 0
    found = sorted(Path(".").glob("calibration*.json"))
    print(f"No such calibration: {path}", file=sys.stderr)
    if found:
        print("\nAvailable here:", file=sys.stderr)
        for f in found:
            try:
                c = json.loads(f.read_text())
                print(f"  {f}   {c.get('camera', '?')} {c.get('mode', '?')}",
                      file=sys.stderr)
            except (OSError, ValueError):
                print(f"  {f}", file=sys.stderr)
        print("\nPass one with --calibration.", file=sys.stderr)
    else:
        print("Run 'stereokit capture' then 'stereokit solve' to make one.",
              file=sys.stderr)
    return 1


def cmd_board(args):
    board_mod.save_pdf(args.out)
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

    # Auto-exposure throughout: the board is swept across distances and tilts,
    # and auto keeps it well exposed at every one of them, which is exactly
    # what corner detection needs. Locking belongs in the SLAM runtime, where
    # temporal consistency matters -- not in calibration capture.
    with StereoCamera(mode=args.mode, lock_exposure=False,
                      profile=profiles.get(args.camera)) as cam:
        tracker = calibrate.CoverageTracker(cam.eye_size,
                                            target_pairs=args.target)
        # Provisional only: used to bin board tilt for coverage feedback,
        # never for the solve. Handles profiles with no vendor FOV figure.
        K = profiles.provisional_K(cam.mode)
        saved, best_sharp = 0, 0.0
        prev_pose = last_saved_pose = None
        STILL_M, STILL_DEG = 0.004, 0.8      # board is holding still
        NEW_POSE_M, NEW_POSE_DEG = 0.03, 6.0  # far enough to be a new view
        print("Sweep the board across the frame at varied tilts. q to stop.")
        while True:
            left, right, ts = cam.grab()
            gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            pair = calibrate.detect_pair(detector, board, gl, gr)

            status = "no board"
            if pair is not None:
                ok, rvec, tvec = cv2.solvePnP(pair.obj, pair.img_left, K,
                                              np.zeros(5))
                pose = (rvec, tvec) if ok else None
                tilt = calibrate.board_tilt_deg(rvec) if ok else 0.0
                dist = float(tvec[2].item()) if ok else None

                sharp = calibrate.sharpness(gl, pair.corners_left)
                # Decay the reference so it adapts instead of latching onto
                # the single sharpest frame ever seen.
                best_sharp = max(best_sharp * 0.997, sharp)
                thresh = 0.25 * best_sharp
                sharp_enough = sharp >= thresh

                # Both gates compare POSES. Comparing corner arrays by index
                # is wrong: corners_left is the sorted ID-intersection of the
                # two eyes, so one flickering edge corner shifts every later
                # entry and a stationary board reads as moving.
                d_still = calibrate.pose_delta(pose, prev_pose)
                still = (d_still is not None
                         and d_still[0] < STILL_M and d_still[1] < STILL_DEG)
                prev_pose = pose

                # Novelty: without it, a steady hand saves ~30 near-identical
                # pairs a second at one pose, which is ill-conditioned for
                # focal length (spec 5.2).
                d_new = calibrate.pose_delta(pose, last_saved_pose)
                moved_enough = (d_new is None
                                or d_new[0] > NEW_POSE_M
                                or d_new[1] > NEW_POSE_DEG)

                mm = 0.0 if d_still is None else d_still[0] * 1000
                status = (f"{pair.n:2d}c {dist or 0:.2f}m tilt{tilt:4.1f}  "
                          f"move {mm:5.1f}mm "
                          f"{'STILL' if still else 'MOVING'}  "
                          f"sharp {sharp:4.1f}/{thresh:4.1f} "
                          f"{'ok' if sharp_enough else 'BLUR'}  "
                          f"{'NEW' if moved_enough else 'same pose'}")
                if still and sharp_enough and moved_enough:
                    np.savez(out / "pairs" / f"{saved:03d}.npz",
                             obj=pair.obj, left=pair.img_left,
                             right=pair.img_right)
                    tracker.add(pair.corners_left, tilt, dist_m=dist)
                    saved += 1
                    last_saved_pose = pose
            else:
                prev_pose = None

            view = np.hstack([left, right])
            cv2.putText(view, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 1)
            cv2.putText(view, f"pairs {saved}/{args.target}  "
                              f"fov {tracker.fov_coverage()*100:3.0f}%  "
                              f"tilt {tracker.tilt_coverage()*100:3.0f}%  "
                              f"dist {tracker.dist_coverage()*100:3.0f}%"
                              f"{'   DONE' if tracker.done() else ''}",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1)
            missing = tracker.missing_distances()
            if missing:
                cv2.putText(view, "need distances: " +
                            " ".join(f"{m:.2f}" for m in missing),
                            (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 220, 255), 1)
            cv2.imshow("stereokit capture", view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
        cv2.destroyAllWindows()

    meta = {"camera": args.camera, "mode": args.mode, "square_mm": args.square_mm
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

    # Resolve the mode against the profile the capture was TAKEN with, never
    # the default. Both cameras define 1280x480 with the same eye size, so
    # falling back to the default profile would silently solve one camera's
    # pairs against the other's mode entry -- and apply its focal-length gate.
    cam_name = meta.get("camera", profiles.DEFAULT_PROFILE.name)
    try:
        prof = profiles.get(cam_name)
    except KeyError:
        print(f"capture.json names camera {cam_name!r}, not one of "
              f"{sorted(profiles.PROFILES)}.", file=sys.stderr)
        return 1
    if meta["mode"] not in prof.modes:
        print(f"capture.json names mode {meta['mode']!r}, not one of "
              f"{sorted(prof.modes)} for camera {prof.name}.", file=sys.stderr)
        return 1
    mode = prof.modes[meta["mode"]]
    print(f"camera {prof.name}, mode {mode.name} ({mode.eye_w}x{mode.eye_h}/eye)")
    calib = calibrate.solve(pairs, (mode.eye_w, mode.eye_h), meta["mode"],
                            meta["square_mm"], profile=prof)

    m = calib["metrics"]
    print(f"pairs {m['n_pairs']} fit / {m['n_holdout']} held out")
    fpx = calib["left"]["K"][0][0]
    lim = calibrate.RMS_ANGULAR_URAD * fpx / 1e6
    slim = calibrate.STEREO_RMS_ANGULAR_URAD * fpx / 1e6
    print(f"  rms left        {m['rms_left']:.4f} px "
          f"= {m['rms_left']/fpx*1e6:5.0f} urad  (limit <{lim:.3f} px)")
    print(f"  rms right       {m['rms_right']:.4f} px "
          f"= {m['rms_right']/fpx*1e6:5.0f} urad  (limit <{lim:.3f} px)")
    print(f"  rms stereo      {m['rms_stereo']:.4f} px "
          f"= {m['rms_stereo']/fpx*1e6:5.0f} urad  (limit <{slim:.3f} px)")
    print(f"  epipolar (held) {m['epipolar_px']:.4f} px   (limit <0.3)")
    bl_nom = (f"(nominal {prof.baseline_nominal_m*1000:.1f})"
              if prof.baseline_nominal_m else "(no known nominal — gate skipped)")
    print(f"  baseline        {calib['baseline_m']*1000:.2f} mm  {bl_nom}")
    nominal = (f"(nominal {mode.f_nominal:.1f})" if mode.f_nominal is not None
               else "(no vendor FOV figure — gate skipped)")
    print(f"  f recovered     {m['f_recovered']:.1f} px  {nominal}")

    failures = calibrate.check_acceptance(calib, profile=prof)
    if failures:
        rejected = Path(args.out).with_suffix(".rejected.json")
        calibrate.save(calib, rejected)
        print("\nFAILED acceptance:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(f"\nWrote {rejected} for diagnosis. {args.out} was NOT written, "
              "so nothing downstream can load this calibration by accident.",
              file=sys.stderr)
        return 1

    calibrate.save(calib, args.out)
    print(f"\nPASSED. Wrote {args.out}")
    return 0


def cmd_view(args):
    if args.raw:
        with StereoCamera(mode=args.mode, lock_exposure=False,
                          profile=profiles.get(args.camera)) as cam:
            while True:
                left, right, _ = cam.grab()
                cv2.imshow("stereokit raw", np.hstack([left, right]))
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
        cv2.destroyAllWindows()
        return 0

    if require_calibration(args.calibration) != 0:
        return 2

    with rgbd_mod.StereoRGBD(args.calibration) as cam:
        # Scale the colour ramp to the estimator's own range. Hardcoding 4 m
        # washed out every camera whose max_depth differs from it, which made
        # a correct depth map look broken.
        far = cam.estimator.max_depth
        for frame in cam:
            vis = frame.depth.copy()
            vis[~np.isfinite(vis)] = 0
            vis = cv2.applyColorMap(
                cv2.convertScaleAbs(vis, alpha=255.0 / far), cv2.COLORMAP_TURBO)
            vis[~np.isfinite(frame.depth)] = 0        # invalid stays black
            valid = np.isfinite(frame.depth).mean() * 100
            cv2.putText(vis, f"valid {valid:4.1f}%   0-{far:.1f} m", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("stereokit rgbd", np.hstack([frame.rgb, vis]))
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    cv2.destroyAllWindows()
    return 0


def cmd_validate(args):
    """Sweep the board and measure depth against its own PnP plane."""
    if require_calibration(args.calibration) != 0:
        return 2
    from . import validate
    validate.main(args)
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


def cmd_colour(args):
    """Measure per-eye colour gains and store them in the calibration.

    Cosmetic for today's consumers -- the RGB output is the left eye alone,
    and depth matches on normalised grayscale -- but it makes the pair look
    consistent and gives any future colour-dependent consumer a correction.
    """
    calib = calibrate.load(args.calibration)
    prof = profiles.get(calib.get("camera", profiles.DEFAULT_PROFILE.name))
    frames = []
    with StereoCamera(mode=calib["mode"], profile=prof,
                      exposure="features") as cam:
        for _ in range(20):
            cam.grab()
        for _ in range(args.frames):
            raw = cam._cap.read()[1]
            if raw is not None:
                frames.append(raw)
    if not frames:
        print("no frames captured", file=sys.stderr)
        return 1

    gains = calibrate.measure_colour_gains(frames, prof.swap_eyes)
    before = calibrate.measure_colour_gains(frames[:1], prof.swap_eyes)
    calib["colour_gains_right"] = gains
    calibrate.save(calib, args.calibration)
    names = ("B", "G", "R")
    print(f"measured over {len(frames)} frames, right eye onto left:")
    for n, g in zip(names, gains):
        print(f"  {n}  x{g:.4f}   ({(1/g - 1) * 100:+.1f}% before correction)")
    rg_before = (1 / gains[2]) / (1 / gains[1])
    print(f"\nR/G mismatch {abs(rg_before - 1) * 100:.1f}% -> corrected")
    print(f"Wrote colour_gains_right to {args.calibration}")
    return 0


def cmd_export(args):
    left, right = export.export_calibration(args.calibration, args.out,
                                            args.name)
    print(f"Wrote {left}\n      {right}")
    print("\nIn RTAB-Map: Preferences > Source > Stereo > 'Video Side-by-Side',"
          "\npoint it at the device and load these two files.")
    print("\nAlso set Vis/MinDepth = 0.2 and Vis/MaxDepth = 4.0 — leaving "
          "MaxDepth unbounded lets far, noisy points dominate and is the most "
          "likely cause of drift with a 52 mm baseline (spec 9.4).")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="stereokit", description=__doc__)
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

    s = sub.add_parser("validate",
                       help="measure depth against the board's true plane")
    s.add_argument("calibration", nargs="?", default="calibration.json")
    s.add_argument("--target", default="near", choices=("near", "far"),
                   help="'near' sweeps the ChArUco board (0.2-1.0 m); "
                        "'far' uses the 85 mm ArUco grid (1-4 m)")
    s.add_argument("--save", default=None, metavar="DIR",
                   help="bank rectified pairs plus the board's true pose, so "
                        "parameters can be swept offline against ground truth")
    s.add_argument("--quiet", action="store_true",
                   help="no audio feedback (on by default: the screen is "
                        "unreadable from several metres away)")
    s.set_defaults(func=cmd_validate)

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

    s = sub.add_parser("colour")
    s.add_argument("calibration")
    s.add_argument("--frames", type=int, default=15)
    s.set_defaults(func=cmd_colour)

    s = sub.add_parser("export")
    s.add_argument("calibration")
    s.add_argument("--out", default="rtabmap")
    s.add_argument("--name", default="ar0144")
    s.set_defaults(func=cmd_export)

    for sub_p in (s for s in sub.choices.values()):
        sub_p.add_argument("--camera", default=profiles.DEFAULT_PROFILE.name,
                           choices=sorted(profiles.PROFILES),
                           help="camera profile (default: %(default)s)")

    args = p.parse_args(argv)
    return args.func(args)
