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
