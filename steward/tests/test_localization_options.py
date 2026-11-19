"""The web and direct Ego launchers opt in to experimental localization."""

from pathlib import Path

import pytest

from steward import cli


@pytest.mark.parametrize(
    ("command", "flags", "drift", "apply", "config"),
    [
        ("up", [], False, False, None),
        ("up", ["--odom-drift"], True, False, None),
        ("world", ["--odom-drift"], True, False, None),
        ("up", ["--apply-localization"], False, True, cli.LOCALIZATION_CONFIG),
        ("ego", ["--apply-localization"], False, True, cli.LOCALIZATION_CONFIG),
        ("up", ["--localization-experiment"], True, True, cli.LOCALIZATION_CONFIG),
        ("up", ["--localization-config", "custom.toml"], False, False, Path("custom.toml")),
        ("up", ["--localization-experiment", "--localization-config", "custom.toml"], True, True, Path("custom.toml")),
    ],
)
def test_startup_options_reach_the_command(monkeypatch, command, flags, drift, apply, config):
    def capture(args):
        assert getattr(args, "odom_drift", False) is drift
        assert getattr(args, "apply_localization", False) is apply
        actual = getattr(args, "localization_config", None)
        assert (Path(actual) if actual else None) == config
        return 0

    monkeypatch.setattr(cli, f"cmd_{command}", capture)
    assert cli.main([command, *flags]) == 0
