"""Measure stationary localization against physical truth over five simulated minutes."""

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from steward.assets.generators import steward_bot
from steward.authoring.service import AuthoringService
from steward.ego.distance_localization import LocalizationConfig, SegmentedLocalizer
from steward.math3d import yaw_from_quat
from steward.world.sim import PerceptionConfig, WorldSim

DURATION_SECONDS = 300
BODY_HZ = 10
POSITION_BIAS_M_PER_S = 0.0002
YAW_BIAS_RAD_PER_S = 0.0002


def _errors(rows, prefix):
    elapsed = np.array([row["t"] for row in rows])
    actual = np.array([row["truth"] for row in rows])
    estimate = np.array([row[prefix] for row in rows])
    position = estimate[:, :2] - actual[:, :2]
    metres = np.linalg.norm(position, axis=1)
    yaw = np.degrees((estimate[:, 2] - actual[:, 2] + math.pi) % math.tau - math.pi)
    early, late = elapsed < 30, elapsed > DURATION_SECONDS - 30
    return {
        "positionRmseM": float(np.sqrt(np.mean(metres**2))),
        "positionMaxM": float(metres.max()),
        "positionWindowShiftM": float(np.linalg.norm(position[late].mean(axis=0) - position[early].mean(axis=0))),
        "yawRmseDeg": float(np.sqrt(np.mean(yaw**2))),
        "yawMaxDeg": float(np.abs(yaw).max()),
        "yawWindowShiftDeg": float(yaw[late].mean() - yaw[early].mean()),
    }


@pytest.mark.skipif(os.environ.get("STEWARD_LOCALIZATION_EXPERIMENT") != "1", reason="opt in to simulated mask experiment")
def test_stationary_localization_with_and_without_time_bias(kitchen_root, tmp_path):
    from localization_experiment import capture_simulated

    cfg = LocalizationConfig.load(Path(os.environ["STEWARD_MATRIX_LOCALIZATION_CONFIG"]))
    model = steward_bot().to_dict()
    trackers = {name: SegmentedLocalizer(model, cfg) for name in ("native", "timeBias")}
    rows = {name: [] for name in trackers}
    accepted = {name: [] for name in trackers}
    sim = WorldSim(AuthoringService(kitchen_root).load_scene("phase0-kitchen"), seed=0,
                   perception=PerceptionConfig(mode="rgbd", masks=True))
    try:
        sim.step_seconds(2)
        for tick in range(DURATION_SECONDS * BODY_HZ + 1):
            elapsed = tick / BODY_HZ
            frame = capture_simulated(sim, trackers["native"]) if tick % BODY_HZ == 0 else None
            header = frame["header"] if frame is not None else sim.sensors("w:robot-1")
            estimates = {}
            for name, tracker in trackers.items():
                sample = header
                if name == "timeBias":
                    odom = header["odometry"]
                    sample = {**header, "odometry": {**odom, "x": odom["x"] + POSITION_BIAS_M_PER_S * elapsed,
                                                    "yaw": odom["yaw"] + YAW_BIAS_RAD_PER_S * elapsed}}
                estimate = tracker.observe({**frame, "header": sample}) if frame is not None else tracker.predict(sample)
                estimates[name] = estimate
                if estimate["accepted"]:
                    accepted[name].append(elapsed)
            truth = sim._entity_pose("w:robot-1")
            actual = [*truth["p"][:2], yaw_from_quat(truth["q"])]
            for name, estimate in estimates.items():
                rows[name].append({"t": elapsed, "truth": actual, "raw": estimate["rawPose"], "fused": estimate["pose"],
                                   "state": estimate["state"], "reason": estimate["reason"]})
            if tick < DURATION_SECONDS * BODY_HZ:
                sim.step_seconds(1 / BODY_HZ)
    finally:
        sim.stop()

    truth = np.array([row["truth"] for row in rows["native"]])
    displacement = float(np.linalg.norm(truth[:, :2] - truth[0, :2], axis=1).max())
    yaw_change = float(np.degrees(np.abs(np.unwrap(truth[:, 2]) - truth[0, 2])).max())
    report = {"seed": 0, "simulationSeconds": DURATION_SECONDS, "bodyHz": BODY_HZ, "visualHz": 1,
              "trueMaxDisplacementM": displacement, "trueMaxYawChangeDeg": yaw_change,
              "injectedPositionBiasMPerS": POSITION_BIAS_M_PER_S, "injectedYawBiasRadPerS": YAW_BIAS_RAD_PER_S,
              "modes": {name: {"raw": _errors(samples, "raw"), "fused": _errors(samples, "fused"),
                               "acceptedCorrections": len(accepted[name]),
                               "longestGapS": max(np.diff([0., *accepted[name], DURATION_SECONDS]))}
                        for name, samples in rows.items()}}
    (tmp_path / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    columns = ["mode", "t", "gtX", "gtY", "gtYaw", "rawX", "rawY", "rawYaw",
               "fusedX", "fusedY", "fusedYaw", "state", "reason"]
    with (tmp_path / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for name, samples in rows.items():
            writer.writerows([name, row["t"], *row["truth"], *row["raw"], *row["fused"], row["state"], row["reason"]]
                             for row in samples)
    assert displacement < .005 and yaw_change < .25, "the stationary robot moved physically"
    assert accepted["native"], "the camera produced no successful corrections"
    assert report["modes"]["timeBias"]["raw"]["positionWindowShiftM"] > .05
