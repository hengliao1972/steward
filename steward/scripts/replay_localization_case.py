"""Replay a recorded rotation case at its original body cadence, reusing exact SAM predictions."""

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np

from steward.assets.generators import steward_bot
from steward.ego.distance_localization import LocalizationConfig, SegmentedLocalizer

ROOT = Path(__file__).resolve().parents[1]


def replay_case(source: Path, output: Path, config: LocalizationConfig, *, verify_legacy=False):
    """Feed raw odometry and matched semantic frames; read recorded GT only after each update."""
    with (source / "trajectory.csv").open(encoding="utf-8", newline="") as stream:
        recorded = list(csv.DictReader(stream))
    paths = sorted((source / "inputs").glob("frame-*.npz"))
    if not paths or not recorded:
        raise ValueError("source needs trajectory.csv and inputs/frame-*.npz")
    if verify_legacy and config.ekf.enabled:
        raise ValueError("legacy verification requires ekf.enabled=false")
    output.mkdir(parents=True, exist_ok=False)
    tracker = SegmentedLocalizer(steward_bot().to_dict(), config)
    rows, observations = [], []
    iterator = iter(paths)
    def read_frame():
        path = next(iterator, None)
        if path is None:
            return None
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            metadata["labels"] = {int(k): v for k, v in metadata["labels"].items()}
            return metadata | {k: np.array(data[k]) for k in ("rgb", "depth", "mask")}
    frame = read_frame()
    episode = frame["header"]["episode"]
    started = time.perf_counter()
    with (output / "localization.jsonl").open("w", encoding="utf-8") as stream:
        for row in recorded:
            stamp = float(row["t"])
            raw = np.array([float(row[k]) for k in ("rawX", "rawY", "rawYaw")])
            if frame is not None and abs(frame["header"]["stamp"] - stamp) < 1e-8:
                observed_raw = frame["header"]["odometry"]
                np.testing.assert_allclose(raw, [observed_raw[k] for k in ("x", "y", "yaw")], rtol=0, atol=1e-12)
                result = tracker.observe(frame)
                observations.append(result)
                stream.write(json.dumps(result) + "\n")
                frame = read_frame()
            else:
                if frame is not None and frame["header"]["stamp"] < stamp:
                    raise ValueError("semantic frame has no matching body timestamp")
                result = tracker.predict({"stamp": stamp, "episode": episode,
                                          "odometry": dict(zip(("x", "y", "yaw"), raw))})
            pose = result["pose"]
            if verify_legacy:
                previous = [float(row[k]) for k in ("outputX", "outputY", "outputYaw")]
                np.testing.assert_allclose(pose, previous, rtol=0, atol=1e-10)
            # Ground truth is accessed only after the estimator has produced this sample.
            truth = [float(row[k]) for k in ("gtX", "gtY", "gtYaw")]
            row = dict(row)
            row.update(outputX=pose[0], outputY=pose[1], outputYaw=pose[2], state=result["state"],
                       outputPositionErrorM=math.dist(pose[:2], truth[:2]),
                       outputYawErrorDeg=math.degrees((pose[2]-truth[2]+math.pi) % (2*math.pi)-math.pi),
                       secondsSinceCorrection=result["secondsSinceCorrection"], stale=result["stale"])
            rows.append(row)
    if frame is not None or len(observations) != len(paths):
        raise ValueError("some semantic frames were not replayed")
    accepted = [r for r in observations if r["accepted"]]
    times = [float(rows[0]["t"]), *[r["stamp"] for r in accepted], float(rows[-1]["t"])]
    report = {"source": str(source), "mode": "ekf" if config.ekf.enabled else "legacy",
              "scope": "fixed SAM frames with original body prediction cadence; no new SAM or rendering",
              "samples": len(rows), "observations": len(observations), "acceptedCorrections": len(accepted),
              "longestTimeWithoutAcceptedCorrectionS": float(max(np.diff(times))),
              "replayWallSeconds": time.perf_counter()-started, "legacyMatchesRecording": verify_legacy}
    for prefix in ("raw", "output"):
        for name in ("PositionErrorM", "YawErrorDeg"):
            values = np.array([float(r[prefix+name]) for r in rows])
            report[prefix+name+"Rmse"] = float(np.sqrt(np.mean(values**2)))
            report[prefix+name+"Max"] = float(np.abs(values).max())
    report["reasons"] = {reason: sum(r["reason"] == reason for r in observations)
                         for reason in sorted({r["reason"] for r in observations})}
    with (output / "trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    return report


def main():
    """Run a fresh replay directory, optionally proving compatibility with legacy recorded outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    if args.config is None and args.legacy:
        cfg = LocalizationConfig()
    else:
        cfg = LocalizationConfig.load(args.config or ROOT/"config/localization.toml")
    if args.legacy:
        cfg = replace(cfg, ekf=replace(cfg.ekf, enabled=False))
    replay_case(args.source, args.output, cfg, verify_legacy=args.legacy)


if __name__ == "__main__":
    main()
