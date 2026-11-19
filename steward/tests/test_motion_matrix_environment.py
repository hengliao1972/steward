"""Simulation runs remove inherited segmenter configuration from the child environment."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("inherited", ["", "http://localhost:9999"])
def test_simulation_child_removes_segmenter_url_without_changing_parent(tmp_path, monkeypatch, inherited):
    scripts = Path(__file__).resolve().parents[1]/"scripts"
    spec = importlib.util.spec_from_file_location("matrix_environment_test", scripts/"motion_control_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path/"source"/"steward"
    for name in ("runtime.py", "distance_localization.py", "pose_ekf.py"):
        path = source/"src"/"steward"/"ego"/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    world = source/"src"/"steward"/"world"/"sim.py"
    world.parent.mkdir(parents=True)
    world.write_text("", encoding="utf-8")
    (source/"tests").mkdir()
    manifest = tmp_path/"input.json"
    manifest.write_text(json.dumps({"selectors": ["tests/test_case.py::test_case"]}), encoding="utf-8")
    children = []
    def run(command, **kwargs):
        if "--collect-only" in command:
            return subprocess.CompletedProcess(command, 0, "tests/test_case.py::test_case\n", "")
        children.append(kwargs["env"])
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setenv("STEWARD_SEGMENTER_URL", inherited)
    monkeypatch.setattr(sys, "argv", ["matrix", "--source-root", str(source.parent),
        "--output", str(tmp_path/"result"), "--manifest", str(manifest), "--stages", "drift-fused",
        "--mask-source", "simulation", "--skill-diagnostics", "--capture-final"])
    module.main()
    assert len(children) == 1
    assert "STEWARD_SEGMENTER_URL" not in children[0]
    assert children[0]["STEWARD_MATRIX_SKILL_DIAGNOSTICS"] == "1"
    assert children[0]["STEWARD_MATRIX_CAPTURE_FINAL"] == "1"
    assert module.os.environ["STEWARD_SEGMENTER_URL"] == inherited
