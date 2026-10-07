"""Decide whether a camera has a rolling or global shutter.

Shaking the camera and looking for skew is a weak test: hand motion is often
too slow to reveal the row delay. This uses a flashing screen instead, which
does not depend on how fast anything moves.

  ROLLING shutter: rows are exposed at different instants, so a screen that
  toggles mid-frame puts some rows in the bright state and others in the dark
  one -- a hard horizontal BAND inside a single frame.

  GLOBAL shutter: every row is exposed together, so each frame is uniformly
  bright or dark. No band, but large frame-to-frame swings.

The discriminator is therefore banding WITHIN a frame versus variation
BETWEEN frames. Exposure is forced short so the sensor cannot average the
flashes away.

Point the camera at the flashing window, filling as much of the frame as
possible, then press q.

Run with:  uv run shutter_test.py [--camera skl2mp220]
"""
import argparse

import cv2
import numpy as np

from stereokit import profiles as P
from stereokit.camera import StereoCamera

EXPOSURE = 50          # 5 ms, well under any plausible frame period
MIN_FRAMES = 60


def row_profile(gray):
    """Brightness per row, with any smooth vignetting gradient removed."""
    prof = gray.mean(axis=1).astype(np.float64)
    smooth = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 101), 0).ravel()
    return prof - smooth


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", default="skl2mp220", choices=sorted(P.PROFILES))
    ap.add_argument("--mode", default=None)
    args = ap.parse_args()

    cv2.namedWindow("flash", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("flash", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    white = np.full((200, 200), 255, np.uint8)
    black = np.zeros((200, 200), np.uint8)

    banding, frame_means, samples = [], [], []
    state = True

    with StereoCamera(profile=args.camera, mode=args.mode,
                      lock_exposure=True, exposure=EXPOSURE) as cam:
        print("Point the camera at the flashing window. q to finish.\n")
        while True:
            # Toggle every grab: the screen changes state mid-readout, which
            # is exactly what a rolling shutter turns into a visible band.
            cv2.imshow("flash", white if state else black)
            state = not state
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            left, _, _ = cam.grab()
            g = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            prof = row_profile(g)
            banding.append(prof.std())
            frame_means.append(g.mean())
            if len(samples) < 4 and prof.std() > 8:
                samples.append(g.copy())

    cv2.destroyAllWindows()

    if len(banding) < MIN_FRAMES:
        print(f"Only {len(banding)} frames; need {MIN_FRAMES}. Run longer.")
        return

    within = float(np.mean(banding))          # banding inside a frame
    between = float(np.std(frame_means))      # variation frame to frame
    ratio = within / max(between, 1e-6)

    print(f"frames analysed            {len(banding)}")
    print(f"banding WITHIN a frame     {within:6.2f} DN")
    print(f"variation BETWEEN frames   {between:6.2f} DN")
    print(f"ratio within/between       {ratio:6.2f}")
    print()
    if within > 10 and ratio > 1.0:
        print("=> ROLLING shutter: rows are exposed at different instants, so")
        print("   the screen's mid-frame toggle lands as a band. This skews")
        print("   the image during motion and corrupts triangulation.")
    elif between > 10 and ratio < 0.5:
        print("=> GLOBAL shutter: whole frames flip together with no internal")
        print("   band. Good for SLAM while moving.")
    else:
        print("=> INCONCLUSIVE. Both signals are weak, which usually means the")
        print("   screen did not fill enough of the frame or the room light")
        print("   swamped it. Fill the frame with the window and dim the room.")

    for i, s in enumerate(samples):
        cv2.imwrite(f"shutter_sample_{i}.png", s)
    if samples:
        print(f"\nwrote {len(samples)} banded frame(s) as shutter_sample_*.png")


if __name__ == "__main__":
    main()
