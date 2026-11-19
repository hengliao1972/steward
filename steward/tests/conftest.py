from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from steward.authoring.service import AuthoringService, build_default_kitchen


def pytest_addoption(parser):
    group = parser.getgroup("Steward localization")
    group.addoption("--localization-experiment", action="store_true", help="enable odometry drift and localization feedback")
    group.addoption("--odom-drift", action="store_true", help="inject cumulative odometry drift into test worlds")
    group.addoption("--apply-localization", action="store_true", help="apply localization feedback in test Ego runtimes")
    group.addoption("--localization-config", help="TOML used by opted-in localization feedback")
    group.addoption("--localization-mask-source", choices=("simulation", "sam"), default="simulation",
                    help="semantic input for opted-in localization feedback (default: simulation)")


@pytest.fixture(autouse=True)
def localization_options(request, monkeypatch):
    options = request.config.option
    drift = options.localization_experiment or options.odom_drift
    fused = options.localization_experiment or options.apply_localization
    if not drift and not fused:
        return

    from steward.cli import LOCALIZATION_CONFIG

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from motion_matrix_plugin import install_localization_options

    install_localization_options(
        monkeypatch, drift=True if drift else None,
        localization_config=(options.localization_config or str(LOCALIZATION_CONFIG)) if fused else None,
    )
    if fused and options.localization_mask_source == "simulation":
        from motion_matrix_simulation_masks import install

        install(monkeypatch, lambda fn: fn(), matrix=False)


@pytest.fixture(scope="session")
def kitchen_root(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("steward")
    build_default_kitchen(AuthoringService(root))
    return str(root)


CAMERA_UP = np.array([-0.5, 0.0, math.sqrt(3.0) / 2.0])
"""The world's up direction in the frame of a camera pitched 30 degrees down (x forward, z up)."""


def _box_depth(shape: tuple[int, int], fovy_deg: float, boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
               ) -> tuple[np.ndarray, np.ndarray]:
    """Depth image of axis-aligned boxes seen by a camera 1.5 m above the origin, looking along
    world +x and pitched 30 degrees down.

    :param boxes: ``(low corner, high corner)`` per box, world metres.
    :returns: ``(depth, owner)``: metres along the camera's forward axis (0 where a ray hits
        nothing) and the index of the box each pixel shows (-1 for none)."""
    h, w = shape
    f = (h / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    vs, us = np.mgrid[0:h, 0:w].astype(np.float64)
    rays = np.stack([np.ones((h, w)), -(us - (w - 1) / 2.0) / f, -(vs - (h - 1) / 2.0) / f], axis=-1)
    forward, left = np.array([math.sqrt(3.0) / 2.0, 0.0, -0.5]), np.array([0.0, 1.0, 0.0])
    world = rays[..., :1] * forward + rays[..., 1:2] * left + rays[..., 2:] * np.cross(forward, left)
    origin = np.array([0.0, 0.0, 1.5])
    depth, owner = np.full((h, w), np.inf), np.full((h, w), -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        for n, (low, high) in enumerate(boxes):
            a = (np.asarray(low) - origin) / world
            b = (np.asarray(high) - origin) / world
            enter, leave = np.minimum(a, b).max(axis=-1), np.maximum(a, b).min(axis=-1)
            hit = (leave >= enter) & (enter > 0) & (enter < depth)
            depth[hit], owner[hit] = enter[hit], n
    return np.where(owner >= 0, depth, 0.0).astype(np.float32), owner


@pytest.fixture
def box_depth():
    """:func:`_box_depth`, with :data:`CAMERA_UP` as the camera's up direction for it."""
    return _box_depth


@pytest.fixture
def kitchen_scene(kitchen_root) -> dict:
    return AuthoringService(kitchen_root).load_scenario("phase0-kitchen")["sceneState"]
