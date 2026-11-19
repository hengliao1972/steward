"""Run physical localization experiments and save comparable pose evidence."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "room": "tests/test_localization_loop.py::test_room_loop_localization[segmented-0]",
    "rotation": "tests/test_localization_rotation.py::test_large_reverse_turns_with_simulated_masks",
    "stationary": "tests/test_localization_stationary.py::test_stationary_localization_with_and_without_time_bias",
}


def capture_simulated(sim, localizer):
    """Read simulated labelled depth via the matrix's private test adapter."""
    from motion_matrix_plugin import _main
    from steward.ego import runtime
    camera = localizer.selector.config.camera
    frame = runtime.decode_rgbd(_main(lambda: sim.rgbd("w:robot-1", camera)))
    return {**frame, "labels": frame["_matrix_labels"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config/localization.toml")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=["room", "rotation"])
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True)
    config = output / "localization.toml"
    config.write_bytes(args.config.read_bytes())
    (output / "protocol.json").write_text(json.dumps({"observationIntervalS": 1.0, "cases": args.cases},
                                                     indent=2) + "\n", encoding="utf-8")
    env = os.environ | {"PYTHONPATH": os.pathsep.join(map(str, (ROOT / "src", ROOT / "scripts"))),
                        "STEWARD_MATRIX_STAGE": "drift-fused", "STEWARD_MATRIX_MASK_SOURCE": "simulation",
                        "STEWARD_MATRIX_LOCALIZATION_CONFIG": str(config), "STEWARD_LOCALIZATION_EXPERIMENT": "1"}
    env.pop("STEWARD_SEGMENTER_URL", None)
    summary = {}
    for case in args.cases:
        directory = output / case
        directory.mkdir()
        started = time.perf_counter()
        with (directory / "pytest.log").open("wb") as log:
            cmd = [sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider", "-p", "motion_matrix_plugin",
                   "--basetemp", str(directory / "tmp"), CASES[case]]
            try:
                code = subprocess.run(cmd, cwd=ROOT, env=env | {"STEWARD_MATRIX_CASE_OUTPUT": str(directory)},
                                      stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code = None
        report = next((directory / "tmp").glob("test_*/report.json"), None)
        result = directory / "pytest-result.json"
        phases = json.loads(result.read_text(encoding="utf-8"))["phases"] if result.is_file() else []
        status = "timeout" if code is None else "failed" if code else "skipped" if any(
            phase["outcome"] == "skipped" for phase in phases) else "passed"
        summary[case] = {"status": status,
                         "exitCode": code, "wallSeconds": time.perf_counter() - started,
                         "pose": json.loads(report.read_text(encoding="utf-8")) if report and report.is_file() else None}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(case, summary[case]["status"], flush=True)
    return int(any(case["status"] != "passed" or case["pose"] is None for case in summary.values()))


if __name__ == "__main__":
    raise SystemExit(main())
