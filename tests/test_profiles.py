"""Profile selection, and the collision it exists to prevent.

The AR0144 and the SKL both advertise 1280x480, 2560x720 and 1600x600. Picking
a device by "first node that accepts this size" would therefore bind whichever
enumerates first when both are plugged in — silently, because both really do
deliver the frame. Identity has to be checked before the mode is requested.
"""
import pytest

from stereokit import camera as C
from stereokit import profiles as P


def test_both_profiles_registered():
    assert set(P.PROFILES) == {"ar0144", "skl2mp220"}
    assert P.DEFAULT_PROFILE is P.AR0144


def test_unknown_profile_rejected():
    with pytest.raises(KeyError):
        P.get("nosuchcamera")


def test_the_two_cameras_really_do_collide_on_mode_names():
    """If this ever stops being true the identity matching is less critical —
    but it is true today, which is why the matching exists."""
    shared = set(P.AR0144.modes) & set(P.SKL2MP220.modes)
    assert shared, "expected overlapping mode names"
    assert "1280x480" in shared


def test_colliding_modes_are_not_interchangeable():
    """Same name, same combined size, different frame rate — so a calibration
    or a capture from one is not valid for the other."""
    a = P.AR0144.modes["1280x480"]
    s = P.SKL2MP220.modes["1280x480"]
    assert (a.combined_w, a.combined_h) == (s.combined_w, s.combined_h)
    assert a.fps != s.fps


def test_every_mode_is_double_width_in_both_profiles():
    for prof in P.PROFILES.values():
        for name, m in prof.modes.items():
            assert m.eye_w * 2 == m.combined_w, f"{prof.name}/{name}"
            assert m.eye_h == m.combined_h, f"{prof.name}/{name}"


def test_default_mode_exists_in_each_profile():
    for prof in P.PROFILES.values():
        assert prof.default_mode in prof.modes


def test_ar0144_keeps_its_focal_priors_and_skl_has_none():
    """The SKL has no usable vendor FOV figure, so the f-vs-prior acceptance
    gate must have nothing to compare against rather than an invented number."""
    assert all(m.f_nominal is not None for m in P.AR0144.modes.values())
    assert all(m.f_nominal is None for m in P.SKL2MP220.modes.values())


def test_candidate_indices_picks_only_matching_cards(monkeypatch):
    monkeypatch.setattr(P, "video_nodes", lambda: [
        (0, "Integrated Camera: Integrated C"),
        (2, "Integrated Camera: Integrated I"),
        (4, "SPCA2100 PC Camera: PC Camera"),
        (6, "CCB Camera: CCB Camera"),
    ])
    assert P.candidate_indices(P.SKL2MP220) == [4]
    assert P.candidate_indices(P.AR0144) == [6]


def test_candidate_indices_ignores_a_lower_indexed_impostor(monkeypatch):
    """The collision case: the SKL sits at a LOWER index than the AR0144 and
    accepts the same mode. Identity matching must still choose the AR0144."""
    monkeypatch.setattr(P, "video_nodes", lambda: [
        (0, "SPCA2100 PC Camera: PC Camera"),
        (4, "CCB Camera: CCB Camera"),
    ])
    assert P.candidate_indices(P.AR0144) == [4]


def test_candidate_indices_is_empty_when_the_camera_is_absent(monkeypatch):
    """The regression this file exists for. Falling back to "all nodes" when
    nothing matches is how the AR0144 profile bound the SKL with only the SKL
    attached: both accept 1280x480, so the frame-shape guard saw nothing wrong.
    An absent camera must yield no candidates at all."""
    monkeypatch.setattr(P, "video_nodes", lambda: [
        (0, "Integrated Camera: Integrated C"),
        (4, "SPCA2100 PC Camera: PC Camera"),
    ])
    assert P.candidate_indices(P.AR0144) == []
    assert P.candidate_indices(P.SKL2MP220) == [4]


def test_candidate_indices_falls_back_when_sysfs_is_unreadable(monkeypatch):
    """Without sysfs the frame-shape guard in camera.py is the backstop, so
    every node must still be offered rather than none."""
    monkeypatch.setattr(P, "video_nodes", lambda: [])
    assert P.candidate_indices(P.AR0144) == list(range(10))


def test_camera_module_still_exposes_the_old_names():
    """Existing callers and tests reach for camera.MODES / camera.Mode."""
    assert C.MODES is P.AR0144.modes
    assert C.DEFAULT_MODE == P.AR0144.default_mode
    assert C.Mode is P.Mode


def test_stereo_camera_accepts_a_profile_by_name():
    cam = C.StereoCamera(profile="skl2mp220")
    assert cam.profile is P.SKL2MP220
    assert cam.mode.name == "1280x480"
    assert cam.mode.fps == 30            # the SKL's rate, not the AR0144's 60


def test_stereo_camera_rejects_a_mode_from_the_other_profile():
    with pytest.raises(KeyError):
        C.StereoCamera(mode="1280x360", profile="skl2mp220")
    with pytest.raises(KeyError):
        C.StereoCamera(mode="3040x1080", profile="ar0144")


def test_provisional_K_is_numeric_for_every_mode_of_every_profile():
    """The crash this guards: f_nominal became optional, a None went into
    np.array(), and the resulting dtype=object matrix was rejected by
    solvePnP the moment the board first appeared during capture."""
    import numpy as np
    for prof in P.PROFILES.values():
        for name, m in prof.modes.items():
            K = P.provisional_K(m)
            assert K.dtype == np.float64, f"{prof.name}/{name}"
            assert np.isfinite(K).all(), f"{prof.name}/{name}"
            assert K[0, 0] > 0 and K[2, 2] == 1.0


def test_provisional_f_uses_the_vendor_figure_when_there_is_one():
    assert P.provisional_f(P.AR0144.modes["1280x480"]) == 669.7


def test_provisional_f_is_plausible_when_there_is_no_vendor_figure():
    f = P.provisional_f(P.SKL2MP220.modes["1280x480"])
    assert P.SKL2MP220.modes["1280x480"].f_nominal is None
    assert 300 < f < 900          # a sane focal length for a 640px-wide eye


def test_baseline_gate_only_fires_when_a_nominal_is_known():
    """The AR0144's 52 mm was hardcoded into the gate, so the SKL's perfectly
    good 16.98 mm calibration was rejected as "67% off, suspect print
    scaling". A camera whose baseline is not yet known must skip the check,
    not fail it.

    Uses a throwaway profile rather than a real one: whether any particular
    camera's baseline is known changes as it gets calibrated, and this is a
    test of the GATE, not of a camera's current state.
    """
    from stereokit import calibrate as CAL
    unknown = P.SKL2MP220._replace(name="unknown-baseline",
                                   baseline_nominal_m=None)
    calib = {"baseline_m": 0.01698, "mode": "1280x480",
             "metrics": {"rms_left": 0.22, "rms_right": 0.22,
                         "rms_stereo": 0.22, "epipolar_px": 0.05,
                         "f_recovered": 605.1, "f_nominal": None}}
    assert CAL.check_acceptance(calib, profile=unknown) == []
    # The same numbers against a camera whose baseline IS known and different
    # must be rejected.
    failures = CAL.check_acceptance(calib, profile=P.AR0144)
    assert any("baseline" in f for f in failures)


def test_skl_baseline_now_known_from_two_agreeing_calibrations():
    """16.9837 mm at 1280x480 and 16.9925 mm at 1600x600, independent
    sessions and modes, 0.05% apart. One run alone would not have earned
    this, since gating a calibration against a figure derived from that same
    run proves nothing."""
    assert P.SKL2MP220.baseline_nominal_m is not None
    assert abs(P.SKL2MP220.baseline_nominal_m - 0.016988) < 1e-5


def test_ar0144_baseline_gate_still_rejects_a_bad_baseline():
    from stereokit import calibrate as CAL
    calib = {"baseline_m": 0.060, "mode": "1280x480",
             "metrics": {"rms_left": 0.2, "rms_right": 0.2, "rms_stereo": 0.2,
                         "epipolar_px": 0.1, "f_recovered": 669.0,
                         "f_nominal": 669.7}}
    assert any("baseline" in f
               for f in CAL.check_acceptance(calib, profile=P.AR0144))
    calib["baseline_m"] = 0.0519          # within tolerance
    assert CAL.check_acceptance(calib, profile=P.AR0144) == []
