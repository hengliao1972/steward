"""Large alternating physical yaw sweeps with continuous segmented-depth localization.

The unwrapped command schedule deliberately takes the long 400-degree path at reversals.
GT scores and validates motion after execution; neither pose prediction nor matching reads it.
"""

from collections import Counter
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pytest

pytest.importorskip("scipy")

from steward.assets.generators import steward_bot
from steward.authoring.service import AuthoringService
from steward.ego.distance_localization import LocalizationConfig, SegmentedLocalizer
from steward.ego.perception import decode_rgbd
from steward.ego.segmenter import Segmentation, Segmenter
from steward.math3d import yaw_from_quat
from steward.world.sim import PerceptionConfig, WorldSim

ROOT = Path(__file__).resolve().parents[1]
TARGETS_DEG = (200, -200, 200, -200, 0)


def _angle_error_deg(estimate, truth):
    return math.degrees((estimate - truth + math.pi) % (2 * math.pi) - math.pi)


def _run_rotation(kitchen_root, output: Path, seed: int, segmenter, *, real_sam: bool, simulated: bool = False):
    cfg = LocalizationConfig.load(Path(os.environ["STEWARD_MATRIX_LOCALIZATION_CONFIG"]) if simulated else
                                  ROOT / "config/localization.toml")
    localizer = SegmentedLocalizer(steward_bot().to_dict(), cfg)
    perception = PerceptionConfig(mode="rgbd", masks=simulated, odom_scale_error=0.01, odom_yaw_drift_per_m=0.005)
    noise_before = asdict(perception)
    assert perception.odom_pos_sigma == 0.003 and perception.odom_yaw_sigma == 0.003
    rows, observations, legs = [], [], []
    inputs = output / "inputs"
    if real_sam:
        inputs.mkdir()
    sim = WorldSim(AuthoringService(kitchen_root).load_scene("phase0-kitchen"), seed=seed, perception=perception)
    started = time.perf_counter()
    next_capture = 0.0

    def sample(stream):
        nonlocal next_capture
        header = sim.sensors("w:robot-1")
        if header["stamp"] + 1e-9 >= next_capture:
            capture_started = time.perf_counter()
            if simulated:
                from localization_experiment import capture_simulated
                frame = capture_simulated(sim, localizer)
            else:
                frame = decode_rgbd(sim.rgbd("w:robot-1", cfg.camera))
            capture_wall = time.perf_counter() - capture_started
            assert frame["header"]["stamp"] == header["stamp"]
            sam_started = time.perf_counter()
            cached = False
            if not simulated:
                assert frame["mask"] is None
                segmentation = segmenter.segment(frame["rgb"])
                frame["mask"], frame["labels"] = segmentation.mask, segmentation.labels
                cached = segmentation.cached
            sam_wall = time.perf_counter() - sam_started
            result = localizer.observe(frame)
            result.update(captureWallSeconds=capture_wall, segmentationWallSeconds=sam_wall,
                          segmentationCached=cached)
            observations.append(result)
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            if real_sam:
                np.savez_compressed(inputs / f"frame-{len(observations)-1:03d}.npz", rgb=frame["rgb"],
                                    depth=frame["depth"], mask=frame["mask"], metadata=json.dumps(
                                        {k: frame[k] for k in ("header", "camera", "fovy", "labels")}))
            next_capture = header["stamp"] + 1.0
            if len(observations) % 10 == 0:
                print(f"rotation seed={seed} observations={len(observations)} t={header['stamp']:.2f}", flush=True)
        else:
            result = localizer.predict(header)
        truth = next(e for e in sim.world_state()["entities"] if e["id"] == "w:robot-1")["pose"]["value"]
        gt_yaw = yaw_from_quat(truth["q"])
        raw, corrected = result["rawPose"], result["pose"]
        rows.append({"t": header["stamp"], "gtX": truth["p"][0], "gtY": truth["p"][1], "gtYaw": gt_yaw,
                     "rawX": raw[0], "rawY": raw[1], "rawYaw": raw[2],
                     "outputX": corrected[0], "outputY": corrected[1], "outputYaw": corrected[2],
                     "rawPositionErrorM": math.dist(raw[:2], truth["p"][:2]),
                     "outputPositionErrorM": math.dist(corrected[:2], truth["p"][:2]),
                     "rawYawErrorDeg": _angle_error_deg(raw[2], gt_yaw),
                     "outputYawErrorDeg": _angle_error_deg(corrected[2], gt_yaw),
                     "state": result["state"], "secondsSinceCorrection": result["secondsSinceCorrection"],
                     "stale": result["stale"]})
        return header

    try:
        sim.step(250)
        with (output / "localization.jsonl").open("w", encoding="utf-8") as stream:
            first = sample(stream)
            anchor = first["odometry"]
            previous = 0.0
            for target_deg in TARGETS_DEG:
                start_index = len(rows) - 1
                target = math.radians(target_deg)
                ticks = math.ceil(abs(target - previous) / 0.6 / 0.05)
                for tick in range(1, ticks + 1):
                    yaw = anchor["yaw"] + previous + (target - previous) * tick / ticks
                    sim.command("w:robot-1", {"base": [anchor["x"], anchor["y"], yaw]})
                    sim.step(25)
                    sample(stream)
                for _ in range(20):
                    sim.step(25)
                    sample(stream)
                measured = np.unwrap([r["gtYaw"] for r in rows[start_index:]])
                measured_deg = math.degrees(measured[-1] - measured[0])
                expected_deg = math.degrees(target - previous)
                assert abs(measured_deg - expected_deg) < 5, "rotation did not execute the commanded long path"
                legs.append({"targetDeg": target_deg, "expectedSignedTravelDeg": expected_deg,
                             "measuredSignedTravelDeg": measured_deg,
                             "endRouteSeconds": rows[-1]["t"] - rows[0]["t"]})
                previous = target
        assert asdict(sim.perception) == noise_before
    finally:
        sim.stop()
    wall = time.perf_counter() - started
    position = np.array([[r["gtX"], r["gtY"]] for r in rows])
    max_displacement = float(np.linalg.norm(position - position[0], axis=1).max())
    assert max_displacement < 0.03, "base translated instead of rotating in place"
    yaw = np.unwrap([r["gtYaw"] for r in rows])
    travel = float(np.abs(np.diff(yaw)).sum() * 180 / math.pi)
    assert abs(travel - 1600) < 20
    wrap_crossings = int(np.count_nonzero(np.abs(np.diff([r["gtYaw"] for r in rows])) > math.pi))
    assert wrap_crossings >= 4, "motion must cross the wrapped yaw discontinuity repeatedly"
    accepted = [r for r in observations if r["accepted"]]
    times = [rows[0]["t"], *[r["stamp"] for r in accepted], rows[-1]["t"]]
    gap_index = int(np.argmax(np.diff(times)))
    origin = rows[0]["t"]
    report = {"seed": seed, "source": "simulation" if simulated else "sam3" if real_sam else "empty_semantic_stand_in",
              "targetsDeg": list(TARGETS_DEG), "legs": legs, "samples": len(rows),
              "simulationSeconds": rows[-1]["t"] - origin, "caseWallSeconds": wall,
              "trueTotalYawTravelDeg": travel, "trueYawWrapCrossings": wrap_crossings,
              "trueMaxDisplacementM": max_displacement,
              "trueFinalYawOffsetDeg": _angle_error_deg(rows[-1]["gtYaw"], rows[0]["gtYaw"]),
              "observations": len(observations), "acceptedCorrections": len(accepted),
              "acceptedObservationShare": len(accepted) / len(observations),
              "longestTimeWithoutAcceptedCorrectionS": max(np.diff(times)),
              "longestGapRouteSeconds": [times[gap_index] - origin, times[gap_index+1] - origin],
              "finalCorrectionAgeS": rows[-1]["secondsSinceCorrection"],
              "reasons": dict(Counter(r["reason"] for r in observations)), "noise": noise_before}
    for prefix in ("raw", "output"):
        for quantity in ("PositionErrorM", "YawErrorDeg"):
            values = np.array([r[prefix + quantity] for r in rows])
            report[prefix + quantity + "Rmse"] = float(np.sqrt(np.mean(values**2)))
            report[prefix + quantity + "P95"] = float(np.percentile(np.abs(values), 95))
            report[prefix + quantity + "Final"] = abs(float(values[-1]))
    for key in ("captureWallSeconds", "segmentationWallSeconds", "selectionWallSeconds", "localizationWallSeconds"):
        report[key] = sum(r[key] for r in observations)
    latency = [r["selectionWallSeconds"] + r["localizationWallSeconds"] for r in observations]
    report.update(selectionAndLocalizationMeanMs=float(np.mean(latency) * 1000),
                  selectionAndLocalizationP95Ms=float(np.percentile(latency, 95) * 1000))
    with (output / "trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    print(f"rotation artifacts: {output}", flush=True)
    return report


def test_large_reverse_turns_keep_prediction_when_no_targets(kitchen_root, tmp_path):
    class NoTargets:
        def segment(self, rgb):
            return Segmentation(np.zeros(rgb.shape[:2], np.uint8), {})

    result = _run_rotation(kitchen_root, tmp_path, 0, NoTargets(), real_sam=False)
    assert result["acceptedCorrections"] == 0
    assert result["outputPositionErrorMRmse"] == pytest.approx(result["rawPositionErrorMRmse"])
    assert result["outputYawErrorDegRmse"] == pytest.approx(result["rawYawErrorDegRmse"])
    assert result["longestTimeWithoutAcceptedCorrectionS"] == pytest.approx(result["simulationSeconds"])


@pytest.mark.skipif(os.environ.get("STEWARD_LOCALIZATION_EXPERIMENT") != "1", reason="opt in to simulation mask experiment")
def test_large_reverse_turns_with_simulated_masks(kitchen_root, tmp_path):
    result = _run_rotation(kitchen_root, tmp_path, 0, None, real_sam=False, simulated=True)
    assert result["acceptedCorrections"] > 0


@pytest.mark.skipif(os.environ.get("STEWARD_SAM_LOCALIZATION_E2E") != "1", reason="opt in to real SAM rotation requests")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_large_reverse_turns_with_real_sam(kitchen_root, tmp_path, seed):
    segmenter = Segmenter.from_files(cache_dir=tmp_path / "sam-cache")
    _run_rotation(kitchen_root, tmp_path, seed, segmenter, real_sam=True)
