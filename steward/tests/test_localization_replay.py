"""Recorded-input replay preserves body cadence and keeps scoring GT outside the estimator."""

import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import runpy

import numpy as np
import pytest

pytest.importorskip("scipy")

from steward.ego.distance_localization import LocalizationConfig


def _source(directory, truth_offset=0):
    directory.mkdir()
    inputs = directory / "inputs"
    inputs.mkdir()
    rows = []
    for index, stamp in enumerate((0, 0.1, 1.0)):
        angle = math.pi - 0.01 + index*0.02
        wrapped = (angle+math.pi) % (2*math.pi)-math.pi
        raw = {"x": index*0.1, "y": 0, "yaw": angle}
        rows.append({"t": stamp, "rawX": raw["x"], "rawY": 0, "rawYaw": angle,
                     "outputX": raw["x"], "outputY": 0, "outputYaw": wrapped,
                     "gtX": truth_offset, "gtY": truth_offset, "gtYaw": truth_offset,
                     "rawPositionErrorM": math.dist([raw["x"], 0], [truth_offset]*2),
                     "rawYawErrorDeg": math.degrees((angle-truth_offset+math.pi) % (2*math.pi)-math.pi)})
        if index != 1:
            metadata = {"camera": "wide_camera", "fovy": 100, "labels": {},
                        "header": {"episode": 1, "stamp": stamp, "odometry": raw}}
            np.savez_compressed(inputs/f"frame-{index:03d}.npz", rgb=np.zeros((8, 8, 3), np.uint8),
                                depth=np.zeros((8, 8)), mask=np.zeros((8, 8), np.uint8), metadata=json.dumps(metadata))
    with (directory/"trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_replay_predicts_between_frames_and_gt_only_changes_scores(tmp_path):
    replay = runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/replay_localization_case.py"))["replay_case"]
    config = LocalizationConfig()
    _source(tmp_path/"normal")
    _source(tmp_path/"different-truth", 100)
    first = replay(tmp_path/"normal", tmp_path/"output", config)
    second = replay(tmp_path/"different-truth", tmp_path/"output-other", config)
    assert first["samples"] == 3 and first["observations"] == 2
    assert first["acceptedCorrections"] == 0 and first["mode"] == "ekf"
    assert first["outputPositionErrorMRmse"] != second["outputPositionErrorMRmse"]
    for directory in ("output", "output-other"):
        with (tmp_path/directory/"trajectory.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        np.testing.assert_allclose([float(r["outputX"]) for r in rows], [0, 0.1, 0.2], atol=1e-12)
        assert all(abs(float(r["outputYaw"])) <= math.pi for r in rows)
    legacy = replay(tmp_path/"normal", tmp_path/"legacy", replace(config, ekf=replace(config.ekf, enabled=False)),
                    verify_legacy=True)
    assert legacy["legacyMatchesRecording"]
