"""Room-loop localization comparison with unchanged instantaneous measurement noise.

The scripted route exercises the physical base actuators, body sensor channel and Ego self
pose. It does not run perception, SLAM, navigation planning or a localization correction.
Ground truth is used for scoring and checking waypoint completion, never to update Ego pose.
"""

import csv
import json
import math
import os
import hashlib
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.authoring.service import AuthoringService
from steward.ego.estimator import EgoEstimator
from steward.math3d import yaw_from_quat
from steward.world.sim import PerceptionConfig, WorldSim


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("mode", ["baseline", "drift", pytest.param("segmented", marks=pytest.mark.skipif(
    os.environ.get("STEWARD_SAM_LOCALIZATION_E2E") != "1" and os.environ.get("STEWARD_LOCALIZATION_EXPERIMENT") != "1",
    reason="opt in to segmented room-loop requests"))])
def test_room_loop_localization(kitchen_root, tmp_path, seed, mode):
    case_started = time.perf_counter()
    drift = mode != "baseline"
    scene = AuthoringService(kitchen_root).load_scene("phase0-kitchen")
    simulated = mode == "segmented" and os.environ.get("STEWARD_LOCALIZATION_EXPERIMENT") == "1"
    config = PerceptionConfig(masks=simulated or mode != "segmented", odom_scale_error=0.01 if drift else 0.0,
                              odom_yaw_drift_per_m=0.005 if drift else 0.0)
    sim = WorldSim(scene, seed=seed, perception=config)
    estimator = EgoEstimator(robot_model=steward_bot().to_dict())
    robot = "w:robot-1"
    noise_before = asdict(sim.perception)
    assert config.odom_pos_sigma == 0.003 and config.odom_yaw_sigma == 0.003
    rows = []
    localization_rows = []
    localizer = segmenter = None
    next_capture = 0.0
    if mode == "segmented":
        try:
            from steward.ego.distance_localization import LocalizationConfig, SegmentedLocalizer
            from steward.ego.segmenter import Segmenter
            localizer = SegmentedLocalizer(steward_bot().to_dict(), LocalizationConfig.load(
                Path(os.environ["STEWARD_MATRIX_LOCALIZATION_CONFIG"]) if simulated else
                Path(__file__).resolve().parents[1] / "config/localization.toml"))
            if not simulated:
                segmenter = Segmenter.from_files(cache_dir=tmp_path / "sam-cache")
        except BaseException:
            sim.stop()
            raise

    def sample():
        nonlocal next_capture
        header = sim.sensors(robot)
        estimator.update_body(header)
        ego = estimator.to_scene().to_dict()
        me = next(e for e in ego["entities"] if e["id"] == ego["self"])["pose"]["value"]
        truth = next(e for e in sim.world_state()["entities"] if e["id"] == robot)["pose"]["value"]
        yaw_error = (yaw_from_quat(me["q"]) - yaw_from_quat(truth["q"]) + math.pi) % (2 * math.pi) - math.pi
        row = {"t": sim.time, "gtX": truth["p"][0], "gtY": truth["p"][1],
               "egoX": me["p"][0], "egoY": me["p"][1],
               "positionErrorM": math.dist(me["p"][:2], truth["p"][:2]),
               "yawErrorDeg": abs(math.degrees(yaw_error))}
        rows.append(row)
        if localizer is not None:
            if localizer.tracker.stamp is None or header["stamp"] > localizer.tracker.stamp:
                if row["t"] + 1e-9 >= next_capture:
                    from steward.ego.perception import decode_rgbd
                    capture_started = time.perf_counter()
                    if simulated:
                        from localization_experiment import capture_simulated
                        rgbd = capture_simulated(sim, localizer)
                    else:
                        rgbd = decode_rgbd(sim.rgbd(robot, localizer.selector.config.camera))
                    capture_wall = time.perf_counter() - capture_started
                    sam_started = time.perf_counter()
                    if not simulated:
                        assert rgbd["mask"] is None
                        segmentation = segmenter.segment(rgbd["rgb"])
                        rgbd["mask"], rgbd["labels"] = segmentation.mask, segmentation.labels
                    sam_wall = time.perf_counter() - sam_started
                    result = localizer.observe(rgbd)
                    result.update(captureWallSeconds=capture_wall, segmentationWallSeconds=sam_wall)
                    localization_rows.append(result)
                    with (tmp_path / "localization.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(result) + "\n")
                    next_capture = row["t"] + 1.0
                    if len(localization_rows) % 10 == 0:
                        print(f"segmented seed={seed} frames={len(localization_rows)} t={row['t']:.2f}", flush=True)
                else:
                    result = localizer.predict(header)
                sample.latest = result
            result = sample.latest
            corrected_yaw = result["pose"][2]
            yaw_difference = (corrected_yaw - yaw_from_quat(truth["q"]) + math.pi) % math.tau - math.pi
            row.update(correctedX=result["pose"][0], correctedY=result["pose"][1],
                       correctedYaw=corrected_yaw, correctedYawErrorDeg=math.degrees(yaw_difference),
                       localizationState=result["state"])
        return header["odometry"], row

    def move(target, speed):
        odom, _ = sample()
        start = np.array([odom["x"], odom["y"], odom["yaw"]])
        goal = np.array(target)
        duration = max(np.linalg.norm(goal[:2] - start[:2]) / speed,
                       abs(goal[2] - start[2]) / 0.6, 0.1)
        ticks = math.ceil(duration / 0.05)
        for tick in range(1, ticks + 1):
            sim.command(robot, {"base": (start + (goal - start) * tick / ticks).tolist()})
            sim.step(25)
            sample()
        for _ in range(20):
            sim.step(25)
            sample()
        assert math.dist([rows[-1]["gtX"], rows[-1]["gtY"]], target[:2]) < 0.05, "route blocked or waypoint not reached"

    try:
        sim.step(250)
        _, first = sample()
        origin = [first["gtX"], first["gtY"]]
        # This loop follows the kitchen's free aisle around the furniture, with an approach
        # from the spawn and a return along that same approach. No pose is teleported.
        route = [(-2.0, origin[1]), (-2.0, -2.0), (2.0, -2.0), (2.0, 1.35),
                 (-2.0, 1.35), (-2.0, origin[1]), tuple(origin)]
        for x, y in route:
            odom, _ = sample()
            heading = math.atan2(y - odom["y"], x - odom["x"])
            yaw = odom["yaw"] + (heading - odom["yaw"] + math.pi) % (2 * math.pi) - math.pi
            move([odom["x"], odom["y"], yaw], 0.4)
            move([x, y, yaw], 0.4)
        odom, _ = sample()
        yaw = odom["yaw"] + (-odom["yaw"] + math.pi) % (2 * math.pi) - math.pi
        move([*origin, yaw], 0.4)
        gt = np.array([[r["gtX"], r["gtY"]] for r in rows])
        es = np.array([[r["egoX"], r["egoY"]] for r in rows])
        errors = np.linalg.norm(es - gt, axis=1)
        report = {"seed": seed, "mode": mode, "drift": drift, "samples": len(rows), "simulationSeconds": rows[-1]["t"] - first["t"],
                  "distanceM": float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum()),
                  "positionRmseM": float(np.sqrt(np.mean(errors**2))),
                  "positionP95M": float(np.percentile(errors, 95)), "positionMaxM": float(errors.max()),
                  "finalPositionErrorM": float(errors[-1]),
                  "trueReturnDistanceM": float(np.linalg.norm(gt[-1] - gt[0])),
                  "estimatedReturnDistanceM": float(np.linalg.norm(es[-1] - es[0])),
                  "relativeClosureErrorM": float(np.linalg.norm((es[-1] - es[0]) - (gt[-1] - gt[0]))),
                  "yawRmseDeg": float(np.sqrt(np.mean([r["yawErrorDeg"]**2 for r in rows]))),
                  "noise": {k: v for k, v in noise_before.items() if k.startswith("odom_")}}
        raw_rows = [{k: v for k, v in r.items() if k not in
                     ("correctedX", "correctedY", "correctedYaw", "correctedYawErrorDeg", "localizationState")} for r in rows]
        report["rawTrajectorySha256"] = hashlib.sha256(json.dumps(raw_rows, sort_keys=True).encode()).hexdigest()
        if localizer is not None:
            output_errors = np.array([math.dist([r["correctedX"], r["correctedY"]], [r["gtX"], r["gtY"]]) for r in rows])
            accepted_stamps = [r["stamp"] for r in localization_rows if r["accepted"]]
            start_time, end_time = rows[0]["t"], rows[-1]["t"]
            gaps = np.diff([start_time, *accepted_stamps, end_time])
            report["segmented"] = {
                "observations": len(localization_rows), "acceptedCorrections": len(accepted_stamps),
                "acceptedObservationShare": len(accepted_stamps) / len(localization_rows),
                "continuousPositionRmseM": float(np.sqrt(np.mean(output_errors**2))),
                "continuousPositionP95M": float(np.percentile(output_errors, 95)),
                "continuousYawRmseDeg": float(np.sqrt(np.mean([r["correctedYawErrorDeg"]**2 for r in rows]))),
                "finalOutputErrorM": float(output_errors[-1]),
                "longestTimeWithoutAcceptedCorrectionS": float(max(gaps)),
                "finalCorrectionAgeS": None if not accepted_stamps else end_time - accepted_stamps[-1],
                "captureWallSeconds": sum(r["captureWallSeconds"] for r in localization_rows),
                "segmentationWallSeconds": sum(r["segmentationWallSeconds"] for r in localization_rows),
                "selectionWallSeconds": sum(r["selectionWallSeconds"] for r in localization_rows),
                "localizationWallSeconds": sum(r["localizationWallSeconds"] for r in localization_rows)}
        report["caseWallSeconds"] = time.perf_counter() - case_started
        assert asdict(sim.perception) == noise_before
        assert report["distanceM"] > 14
        assert report["trueReturnDistanceM"] < 0.05
        if drift:
            assert report["finalPositionErrorM"] > 0.05
        else:
            assert report["positionRmseM"] < 0.02
        (tmp_path / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with (tmp_path / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(report), flush=True)
        print(f"localization artifacts: {tmp_path}", flush=True)
    finally:
        sim.stop()
