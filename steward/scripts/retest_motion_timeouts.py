"""Rerun only identified deadline failures with longer waits, preserving the original matrix."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def timeout_reason(result, directory):
    """Distinguish the external ceiling from known internal deadline failures."""
    if result["status"] == "timeout":
        return "external_600_second_ceiling"
    report = directory/"pytest-result.json"
    if result["status"] != "failed" or not report.exists():
        return None
    phases = json.loads(report.read_text(encoding="utf-8"))["phases"]
    errors = "\n".join(line for p in phases if p["outcome"] == "failed"
                       for line in p["details"].splitlines() if line.startswith("E "))
    if "AssertionError: timed out" in errors:
        return "internal_exploration_120_second_deadline"
    if "KeyError: 'box'" in errors and "test_the_robot_takes_off_the_lid" in result["nodeid"]:
        return "internal_puzzle_300_second_deadline_missing_result"
    if "KeyError: 'w:piece-" in errors and "test_pieces_dropped_behind_and_under" in result["nodeid"]:
        return "internal_floor_240_second_deadline_missing_result"
    if "is still running" in errors and result["nodeid"].startswith("tests/test_puzzle.py::"):
        return "internal_job_wait_120_second_deadline"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--multiplier", type=float, default=4)
    parser.add_argument("--case-timeout", type=float, default=2400)
    parser.add_argument("--only-case", action="append", default=[], help="limit to an eligible stage:index; repeatable")
    args = parser.parse_args()
    if args.multiplier <= 1 or args.case_timeout <= 600:
        parser.error("extended reruns require longer internal and external budgets")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected = []
    cohort_path = args.pre/"cohort-selection.json"
    shared_retained = set(json.loads(cohort_path.read_text(encoding="utf-8"))["retainedNodes"]) if cohort_path.exists() else None
    for parent, stages in ((args.pre, ("baseline", "drift")), (args.post, ("drift-fused", "clean-fused"))):
        manifest = json.loads((parent/"manifest.json").read_text(encoding="utf-8"))
        retained = shared_retained if shared_retained is not None else set(manifest["nodes"])
        if not retained <= set(manifest["nodes"]):
            raise ValueError("shared analysis cohort is absent from a raw manifest")
        for stage in stages:
            results = json.loads((parent/stage/"results.json").read_text(encoding="utf-8"))
            if len(results) != len(manifest["nodes"]):
                raise ValueError(f"original stage {stage} is incomplete")
            for index, result in enumerate(results):
                if result["nodeid"] not in retained:
                    continue
                directory = parent/stage/f"case-{index:02d}"
                reason = timeout_reason(result, directory)
                if reason is not None:
                    selected.append({"stage": stage, "case": index, "nodeid": result["nodeid"], "reason": reason,
                                     "maskSource": manifest.get("maskSource", "sam"),
                                     "originalStatus": result["status"], "originalWallSeconds": result["wallSeconds"]})
    if args.only_case:
        selected = [c for c in selected if f"{c['stage']}:{c['case']}" in args.only_case]
        if len(selected) != len(set(args.only_case)):
            raise ValueError("an explicit rerun selection is not an identified timeout case")
    frozen = {"multiplier": args.multiplier, "externalCeilingSeconds": args.case_timeout, "selected": selected,
              "successAssertions": "verified AST-identical before execution", "controlThresholds": "unchanged"}
    (output/"manifest.json").write_text(json.dumps(frozen, indent=2)+"\n", encoding="utf-8")
    outcomes = []
    (output/"results.json").write_text("[]\n", encoding="utf-8")
    for case in selected:
        directory = output/case["stage"]/f"case-{case['case']:02d}"
        directory.mkdir(parents=True)
        env = os.environ | {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                            "PYTHONPATH": os.pathsep.join([str(ROOT/"src"), str(ROOT/"scripts")]),
                            "STEWARD_MATRIX_STAGE": case["stage"], "STEWARD_MATRIX_CASE_OUTPUT": str(directory),
                            "STEWARD_MATRIX_LOCALIZATION_CONFIG": str(ROOT/"config/localization.toml"),
                            "STEWARD_MATRIX_TIMEOUT_MULTIPLIER": str(args.multiplier)}
        env["STEWARD_MATRIX_MASK_SOURCE"] = case["maskSource"]
        if case["maskSource"] == "simulation":
            env.pop("STEWARD_SEGMENTER_URL", None)
            env["STEWARD_MATRIX_CAPTURE_FINAL"] = str(int(case["stage"] == "clean-fused" and
                "test_the_robot_takes_off_the_lid" in case["nodeid"] and "puzzle-dog" in case["nodeid"]))
        started = time.perf_counter()
        with (directory/"pytest.log").open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run([sys.executable, "-m", "pytest", "-p", "motion_matrix_plugin", "-p",
                                         "motion_matrix_extended_plugin", "-q", "-s", case["nodeid"],
                                         "--basetemp", str(directory/"tmp")], cwd=ROOT, env=env,
                                        stdout=log, stderr=subprocess.STDOUT, timeout=args.case_timeout)
                status, code = ("passed" if result.returncode == 0 else "failed"), result.returncode
            except subprocess.TimeoutExpired:
                status, code = "timeout", None
        report_path = directory/"pytest-result.json"
        if report_path.exists():
            phases = json.loads(report_path.read_text(encoding="utf-8"))["phases"]
            if any(p["outcome"] == "skipped" for p in phases):
                status = "skipped"
        outcome = {**case, "status": status, "exitcode": code, "wallSeconds": time.perf_counter()-started,
                   "directory": str(directory)}
        outcomes.append(outcome)
        (output/"results.json").write_text(json.dumps(outcomes, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(outcome), flush=True)


if __name__ == "__main__":
    main()
