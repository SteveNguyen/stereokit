"""The validate subcommand's wiring, and the missing-calibration guard.

Nothing here opens a camera. What is worth testing is that the subcommand
exists with the arguments the README documents, that it reaches the module
that does the work, and that a wrong path produces a usable message rather
than a traceback -- the guard exists precisely because the repo holds five
calibrations and the default name is right for only one of them.
"""
import json

import pytest

from stereokit import cli


def test_validate_subcommand_parses_the_documented_arguments():
    captured = {}

    def fake_main(args):
        captured["target"] = args.target
        captured["save"] = args.save
        captured["calibration"] = args.calibration

    from stereokit import validate
    original = validate.main
    validate.main = fake_main
    try:
        rc = cli.main(["validate", "calibration.json",
                       "--target", "far", "--save", "out"])
    finally:
        validate.main = original

    assert rc == 0
    assert captured == {"target": "far", "save": "out",
                        "calibration": "calibration.json"}


def test_validate_refuses_a_missing_calibration_without_opening_a_camera():
    # Reaching StereoRGBD would try to open hardware; the guard must fire
    # first and return a non-zero status.
    assert cli.main(["validate", "definitely-not-here.json"]) == 2


def test_view_refuses_a_missing_calibration():
    assert cli.main(["view", "--calibration", "definitely-not-here.json"]) == 2


def test_missing_calibration_message_lists_what_is_actually_present(
        tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "calibration_demo.json").write_text(
        json.dumps({"camera": "ar0144", "mode": "2560x720"}))

    assert cli.require_calibration("absent.json") == 1
    err = capsys.readouterr().err
    assert "calibration_demo.json" in err
    assert "ar0144 2560x720" in err


def test_missing_calibration_message_says_how_to_make_one_when_none_exist(
        tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.require_calibration("absent.json") == 1
    assert "stereokit capture" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["board", "capture", "solve", "validate",
                                  "view", "record", "colour", "export"])
def test_every_documented_subcommand_is_registered(name):
    # The module docstring is what a reader follows; it must not drift from
    # the parser.
    assert f"stereokit {name}" in cli.__doc__
