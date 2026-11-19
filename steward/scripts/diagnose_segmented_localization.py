"""Private simulation smoke with real SAM masks; GT scores outputs only, never selects points."""

import argparse
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np

from steward.assets.generators import steward_bot
from steward.authoring.service import AuthoringService, build_default_kitchen
from steward.ego.distance_localization import LocalizationConfig, SegmentedLocalizer
from steward.ego.perception import decode_rgbd
from steward.ego.segmenter import Segmenter
from steward.math3d import yaw_from_quat
from steward.world.sim import PerceptionConfig, WorldSim

ROOT = Path(__file__).resolve().parents[1]


def replay(source: Path, output: Path, localizer: SegmentedLocalizer, count: int):
    """Replay saved semantic RGB-D inputs without rendering, network or ground-truth input."""
    results = []
    paths = sorted(source.glob("frame-*.npz"))[:count]
    if not paths:
        raise ValueError("replay directory contains no frame-*.npz inputs")
    with (output / "frames.jsonl").open("w", encoding="utf-8") as stream:
        for path in paths:
            with np.load(path, allow_pickle=False) as saved:
                metadata = json.loads(str(saved["metadata"]))
                metadata["labels"] = {int(k): v for k, v in metadata["labels"].items()}
                frame = metadata | {k: np.array(saved[k]) for k in ("rgb", "depth", "mask")}
            result = localizer.observe(frame)
            stream.write(json.dumps(result) + "\n")
            results.append(result)
    times = [r["selectionWallSeconds"] + r["localizationWallSeconds"] for r in results]
    report = {"scope": "fixed semantic RGB-D replay; no rendering or SAM request", "frames": len(results),
              "accepted": sum(r["accepted"] for r in results),
              "selectionAndLocalizationMeanSeconds": float(np.mean(times)),
              "selectionAndLocalizationP95Seconds": float(np.percentile(times, 95))}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


def main():
    """Render a short physical turn, segment with the configured service, and archive every input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/localization.toml")
    parser.add_argument("--segmentation-config", type=Path, default=ROOT / "config/segmentation.toml")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--replay", type=Path, help="saved semantic RGB-D directory; no simulation or SAM calls")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least two")
    cfg = LocalizationConfig.load(args.config)
    localizer = SegmentedLocalizer(steward_bot().to_dict(), cfg)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.replay is not None:
        replay(args.replay, args.output, localizer, args.frames)
        return
    segmenter = Segmenter.from_files(args.segmentation_config, cache_dir=args.output / "sam-cache")
    rows = []
    with tempfile.TemporaryDirectory(prefix="steward-localization-") as root:
        service = AuthoringService(root)
        build_default_kitchen(service)
        sim = WorldSim(service.load_scene("phase0-kitchen"), seed=args.seed,
                       perception=PerceptionConfig(mode="rgbd", masks=False,
                                                   odom_scale_error=0.01, odom_yaw_drift_per_m=0.005))
        try:
            sim.step(250)
            start = sim.sensors("w:robot-1")["odometry"]
            with (args.output / "frames.jsonl").open("w", encoding="utf-8") as stream:
                for index in range(args.frames):
                    sim.command("w:robot-1", {"base": [start["x"], start["y"], start["yaw"] + index * 0.04]})
                    sim.step(100)
                    capture_started = time.perf_counter()
                    frame = decode_rgbd(sim.rgbd("w:robot-1", cfg.camera))
                    capture_wall = time.perf_counter() - capture_started
                    assert frame["mask"] is None, "this smoke must not use ideal simulator masks"
                    sam_started = time.perf_counter()
                    segmentation = segmenter.segment(frame["rgb"])
                    sam_wall = time.perf_counter() - sam_started
                    frame["mask"], frame["labels"] = segmentation.mask, segmentation.labels
                    result = localizer.observe(frame)
                    truth = next(e for e in sim.world_state()["entities"] if e["id"] == "w:robot-1")["pose"]["value"]
                    result.update(captureWallSeconds=capture_wall, segmentationWallSeconds=sam_wall,
                                  segmentationCached=segmentation.cached, source="sam3",
                                  gtPose=[*truth["p"][:2], yaw_from_quat(truth["q"])])
                    for key, pose in (("rawErrorM", result["rawPose"]), ("outputErrorM", result["pose"])):
                        result[key] = math.dist(pose[:2], truth["p"][:2])
                    np.savez_compressed(args.output / f"frame-{index:03d}.npz", rgb=frame["rgb"],
                                        depth=frame["depth"], mask=frame["mask"],
                                        metadata=json.dumps({k: frame[k] for k in ("header", "camera", "fovy", "labels")}))
                    rows.append(result)
                    stream.write(json.dumps(result) + "\n")
                    stream.flush()
                    print(f"frame={index} state={result['state']} reason={result['reason']} points={result['points']}", flush=True)
        finally:
            sim.stop()
    timings = np.asarray([r["selectionWallSeconds"] + r["localizationWallSeconds"] for r in rows])
    report = {"scope": "short physical turn with real SAM; not a full room-loop accuracy benchmark",
              "seed": args.seed, "config": str(args.config), "frames": len(rows),
              "accepted": sum(r["accepted"] for r in rows),
              "rawPositionRmseM": float(np.sqrt(np.mean([r["rawErrorM"]**2 for r in rows]))),
              "continuousOutputRmseM": float(np.sqrt(np.mean([r["outputErrorM"]**2 for r in rows]))),
              "selectionAndLocalizationMeanSeconds": float(timings.mean()),
              "selectionAndLocalizationP95Seconds": float(np.percentile(timings, 95)),
              "captureTotalSeconds": sum(r["captureWallSeconds"] for r in rows),
              "segmentationTotalSeconds": sum(r["segmentationWallSeconds"] for r in rows)}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
