"""Simulation semantics follow anonymous pixels and never disclose entity identities or poses."""

from pathlib import Path
import io
import json
import zipfile

import numpy as np
import pytest


def test_metadata_append_preserves_compressed_sensor_members(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from motion_matrix_simulation_masks import append_labels
    stream = io.BytesIO()
    mask = np.arange(16, dtype=np.uint8).reshape(4, 4)
    np.savez_compressed(stream, mask=mask, depth=mask.astype(np.uint16)*100)
    before = stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(before)) as archive:
        sensor_end = archive.start_dir
    after = append_labels(before, {7: {"wall": 1.0}})
    assert after[:sensor_end] == before[:sensor_end], "sensor members must not be recompressed"
    with np.load(io.BytesIO(after)) as archive:
        np.testing.assert_array_equal(archive["mask"], mask)
        np.testing.assert_array_equal(archive["depth"], mask.astype(np.uint16)*100)
        assert json.loads(str(archive["matrix_semantic_labels"])) == {"7": {"wall": 1.0}}


def test_simulation_labels_follow_shuffled_mask_ids_and_reject_mixed_parts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from motion_matrix_simulation_masks import semantic_labels
    parts = np.array([[0, 0, -1], [1, 1, -1]])
    mask = np.array([[211, 211, 0], [7, 7, 0]], dtype=np.uint8)
    assert semantic_labels(parts, mask, ["wall", "cabinet"]) == {211: {"wall": 1.0}, 7: {"cabinet": 1.0}}
    mixed = np.array([[7, 7, 0], [7, 7, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="exactly one"):
        semantic_labels(parts, mixed, ["wall", "cabinet"])


def test_simulated_rgbd_semantics_are_localization_only_and_need_no_sam(monkeypatch, kitchen_scene):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from motion_matrix_simulation_masks import install
    from steward.world.sim import WorldSim
    from steward.ego import runtime as runtime_module
    from steward.ego.perception import decimated
    from steward.assets.generators import steward_bot
    install(monkeypatch, lambda operation: operation())
    sim = WorldSim(kitchen_scene)
    try:
        sim.step(250)
        frame = runtime_module.decode_rgbd(sim._rgbd_now("w:robot-1", "wide_camera"))
        assert "labels" not in frame, "main object perception must retain its original semantics"
        assert frame["_matrix_labels"] and "w:" not in str(frame["_matrix_labels"])
        categories = {category for scores in frame["_matrix_labels"].values() for category in scores}
        assert "wall" in categories
        config = runtime_module.RuntimeConfig(localization_config=str(Path(__file__).resolve().parents[1]/"config/localization.toml"), localization_apply=True)
        runtime = runtime_module.EgoRuntime(object(), steward_bot().to_dict(), config)
        localized = runtime._localization_frames(frame["header"], [decimated(frame, 120)])
        assert localized[0]["labels"] == frame["_matrix_labels"]
        runtime._advance_localization(frame["header"], localized)
        assert runtime._fusion_stats["simulatedMaskFrames"] == 1
        assert runtime._fusion_stats["independentSamRequests"] == 0
        assert runtime.localization_status()["applied"]
    finally:
        sim.stop()


def test_report_names_follow_test_identity_after_filtering(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from motion_matrix_report import case_name
    assert case_name("tests/test_chores.py::test_a_chair_is_carried_and_put_down_on_the_floor", True) == "椅子搬到地面"
    assert case_name("tests/test_puzzle.py::test_the_robot_takes_off_the_lid_and_pours_the_pieces_out_onto_the_table[puzzle-dog-4-lid_at1-pour_at1-12]", False) == "Dog puzzle pouring"


def test_analysis_cohort_rejects_failed_cases_and_selective_removal_of_passes(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from summarize_motion_control_matrix import analysis_manifest
    raw = {"nodes": ["a", "b", "c"], "maskSource": "simulation"}
    baseline = [{"nodeid": "a", "status": "passed"}, {"nodeid": "b", "status": "failed"}, {"nodeid": "c", "status": "passed"}]
    selected = analysis_manifest(raw, {"retainedNodes": ["a", "c"], "effectiveManifest": {}}, baseline)
    assert selected["nodes"] == ["a", "c"] and raw["nodes"] == ["a", "b", "c"]
    for invalid in (["b"], ["a"], ["a", "c", "c"], ["c", "a"], ["missing"]):
        with pytest.raises(ValueError):
            analysis_manifest(raw, {"retainedNodes": invalid, "effectiveManifest": {}}, baseline)


def test_report_handles_noncontiguous_raw_case_numbers(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from motion_matrix_report import write_reports
    stages = ["baseline", "drift", "drift-fused", "clean-fused"]
    cases = [{"stage": stage, "case": raw_id, "nodeid": node, "status": "passed", "directory": f"{stage}/{raw_id}", "wallSeconds": 1.0}
             for stage in stages for raw_id, node in [(0, "tests/test_e2e.py::test_cup_to_table_end_to_end"), (2, "tests/test_chores.py::test_a_chair_is_carried_and_put_down_on_the_floor")]]
    totals = [{"stage": stage, "passed": 2, "failed": 0, "timeout": 0, "skipped": 0, "passRate": 1.0, "wallSeconds": 2.0} for stage in stages]
    write_reports({"stages": totals, "cases": cases, "manifest": {"maskSource": "simulation", "caseTimeoutSeconds": 600}}, tmp_path)
    report = (tmp_path/"report.md").read_text(encoding="utf-8")
    assert "Chair to floor" in report and "(clean-fused/2/pytest.log)" in report
