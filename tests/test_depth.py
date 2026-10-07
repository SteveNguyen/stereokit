"""Depth maths is tested exactly; SGBM behaviour is tested by property.

The exact test matters because a wrong f*B is a silent scale error: every
depth is wrong by a constant factor and nothing looks broken.
"""
import numpy as np
import pytest

from stereokit import depth as D

FX = 669.7
B = 0.052
FB = FX * B          # 34.83


@pytest.fixture
def est():
    return D.DepthEstimator(fx=FX, baseline_m=B)


def test_depth_from_constant_disparity_is_exact(est):
    disp = np.full((10, 10), 34.83, np.float32)
    depth = est.depth_from_disparity(disp)
    assert np.allclose(depth, FB / 34.83, rtol=1e-5)


def test_known_depths_match_the_spec_table(est):
    """Spec 3: Z = f*B/d, so 1 m needs ~34.8 px at this focal length."""
    for z in (0.5, 1.0, 2.0, 3.0):
        d = FB / z
        depth = est.depth_from_disparity(np.full((4, 4), d, np.float32))
        assert np.allclose(depth, z, rtol=1e-4)


def test_depth_disparity_round_trip(est):
    disp = np.linspace(10.0, 180.0, 64).astype(np.float32).reshape(8, 8)
    depth = est.depth_from_disparity(disp)
    back = FB / depth
    assert np.allclose(back, disp, rtol=1e-5)


def test_invalid_disparity_becomes_nan(est):
    disp = np.array([[0.0, -1.0, 34.83]], np.float32)
    depth = est.depth_from_disparity(disp)
    assert np.isnan(depth[0, 0])
    assert np.isnan(depth[0, 1])
    assert np.isfinite(depth[0, 2])


def test_depth_outside_the_clamp_is_rejected(est):
    """Beyond 4 m the error budget says it is not worth keeping (spec 3)."""
    too_far = FB / 10.0            # 10 m
    too_near = FB / 0.05           # 5 cm
    depth = est.depth_from_disparity(
        np.array([[too_far, too_near]], np.float32))
    assert np.isnan(depth).all()


def test_num_disparities_sets_the_near_limit(est):
    """192 disparities => 0.18 m minimum (spec 3.2)."""
    assert abs(FB / 192 - 0.1814) < 1e-3
    assert est.min_disparity_for_range == pytest.approx(FB / 4.0, rel=1e-3)


def test_min_depth_derives_from_num_disparities_when_not_given():
    """A larger search range must actually deliver its near band, not be
    silently capped by a hardcoded floor."""
    est256 = D.DepthEstimator(FX, B, num_disparities=256)
    assert est256.min_depth == pytest.approx(FB / 256, rel=1e-6)


def test_min_depth_explicit_override_is_still_honoured():
    est = D.DepthEstimator(FX, B, num_disparities=256, min_depth=0.3)
    assert est.min_depth == pytest.approx(0.3, rel=1e-6)


def test_normalise_removes_a_gain_difference():
    """The eyes are NOT photometrically identical (spec 2.3); SGBM's cost
    assumes they are, so normalise before matching."""
    rng = np.random.default_rng(0)
    base = rng.integers(40, 200, (64, 64)).astype(np.uint8)
    dim = (base * 0.75).astype(np.uint8)
    nl, nr = D.normalise_pair(base, dim)
    assert abs(float(nl.mean()) - float(nr.mean())) < 2.0
    assert abs(float(nl.std()) - float(nr.std())) < 2.0


def test_normalise_preserves_structure():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (64, 64)).astype(np.uint8)
    out, _ = D.normalise_pair(img, img)
    # Monotonic remap: ordering of pixel values must survive.
    assert np.corrcoef(img.ravel(), out.ravel())[0, 1] > 0.99


def test_synthetic_shifted_pair_yields_the_planted_disparity(est):
    """A right image that is the left shifted by k px must measure ~k."""
    rng = np.random.default_rng(2)
    shift = 40
    w, h = 480, 240
    wide = rng.integers(0, 255, (h, w + shift)).astype(np.uint8)
    wide = np.dstack([wide] * 3)
    left = wide[:, :-shift]
    right = wide[:, shift:]
    disp = est.disparity(left, right)
    mid = disp[h // 4:3 * h // 4, w // 2 - 60:w // 2 + 60]
    valid = mid[np.isfinite(mid)]
    assert valid.size > 0.5 * mid.size
    assert abs(float(np.median(valid)) - shift) < 1.5
