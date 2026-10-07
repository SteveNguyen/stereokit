"""Dense stereo: rectified pair in, metric depth out.

For SLAM and for collision checking, OUTLIER RATE matters more than density. A
sparse but correct map is useful; a dense one with a few percent of gross
outliers corrupts odometry and drags the occupancy grid with it. Every filter
here is tuned toward rejection (spec 7).
"""
import cv2
import numpy as np


def normalise_pair(gray_left, gray_right):
    """Equalise mean and variance between the eyes before matching.

    The two sensors are not photometrically identical — roughly 6% colour
    difference under auto white balance, 22% with it locked (spec 2.3) — and
    SGBM's cost function assumes they agree. Normalising to a common mean and
    standard deviation removes both the gain and the colour mismatch in one
    step, and costs almost nothing.
    """
    out = []
    target_mean, target_std = 128.0, 50.0
    for g in (gray_left, gray_right):
        g = g.astype(np.float32)
        std = float(g.std())
        if std < 1e-6:
            out.append(np.full(g.shape, target_mean, np.uint8))
            continue
        scaled = (g - float(g.mean())) / std * target_std + target_mean
        out.append(np.clip(scaled, 0, 255).astype(np.uint8))
    return out[0], out[1]


class DepthEstimator:
    def __init__(self, fx, baseline_m, num_disparities=192, block_size=5,
                 min_depth=None, max_depth=4.0, uniqueness=10,
                 speckle_window=100, speckle_range=2, disp12_max_diff=1):
        if num_disparities % 16:
            raise ValueError("num_disparities must be a multiple of 16")
        self.fx = float(fx)
        self.baseline_m = float(baseline_m)
        self.fb = self.fx * self.baseline_m
        self.num_disparities = int(num_disparities)
        # Derive from the search range unless overridden: the nearest depth
        # SGBM can represent is fb/num_disparities, and hardcoding a floor
        # would silently discard the near band a larger range buys.
        self.min_depth = (float(min_depth) if min_depth is not None
                          else self.fb / self.num_disparities)
        self.max_depth = float(max_depth)

        channels = 1          # matching runs on normalised grayscale
        self._matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=block_size,
            P1=8 * channels * block_size ** 2,
            P2=32 * channels * block_size ** 2,
            disp12MaxDiff=disp12_max_diff,      # left-right consistency
            uniquenessRatio=uniqueness,
            speckleWindowSize=speckle_window,
            speckleRange=speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    @property
    def min_disparity_for_range(self):
        """Disparity corresponding to max_depth."""
        return self.fb / self.max_depth

    def _gray(self, img):
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def disparity(self, left_rect, right_rect):
        gl, gr = normalise_pair(self._gray(left_rect), self._gray(right_rect))
        raw = self._matcher.compute(gl, gr)       # fixed point, x16
        disp = raw.astype(np.float32) / 16.0
        disp[disp <= 0] = np.nan                  # 0 and negatives are invalid
        return disp

    def depth_from_disparity(self, disp):
        disp = np.asarray(disp, np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = self.fb / disp
        depth[~np.isfinite(depth)] = np.nan
        depth[disp <= 0] = np.nan
        depth[(depth < self.min_depth) | (depth > self.max_depth)] = np.nan
        return depth.astype(np.float32)

    def compute(self, left_rect, right_rect):
        disp = self.disparity(left_rect, right_rect)
        return disp, self.depth_from_disparity(disp)
