"""Freeze a motion-test manifest and run each matrix case in its own process with identical pumping."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="repository root of the implementation under test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=("baseline", "drift", "drift-fused", "clean-fused"), required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT/"config/motion-control-cases.json")
    parser.add_argument("--case-timeout", type=float, default=600)
    parser.add_argument("--mask-source", choices=("sam", "simulation"), default="sam")
    parser.add_argument("--skill-diagnostics", action="store_true", help="save complete skill responses and job logs")
    parser.add_argument("--capture-final", action="store_true", help="save the final simulation scene for every selected case")
    args = parser.parse_args()
    if args.case_timeout <= 0:
        parser.error("--case-timeout must be positive")
    source = args.source_root.resolve()/"steward"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/"input-manifest.json").write_bytes(args.manifest.read_bytes())
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    environment = os.environ | {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                                "PYTHONPATH": os.pathsep.join([str(source/"src"), str(ROOT/"scripts")])}
    collected = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", *manifest["selectors"]],
                               cwd=source, env=environment, capture_output=True, text=True, encoding="utf-8")
    (output/"collection.log").write_text(collected.stdout+collected.stderr, encoding="utf-8")
    if collected.returncode:
        raise RuntimeError("test collection failed; inspect collection.log")
    nodes = [line.strip() for line in collected.stdout.splitlines() if line.startswith("tests/") and "::" in line]
    if not nodes:
        raise ValueError("no selected cases collected")
    frozen = manifest | {"nodes": nodes, "sourceRoot": str(source), "caseTimeoutSeconds": args.case_timeout,
                         "maskSource": args.mask_source,
                         "skillDiagnostics": args.skill_diagnostics, "captureFinal": args.capture_final,
                         "manifestSha256": hashlib.sha256(json.dumps(nodes).encode()).hexdigest()}
    frozen["sourceFingerprints"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        source/"src/steward/ego/runtime.py", source/"src/steward/ego/distance_localization.py",
        source/"src/steward/ego/pose_ekf.py", source/"src/steward/world/sim.py",
        ROOT/"scripts/motion_matrix_plugin.py", ROOT/"scripts/motion_matrix_simulation_masks.py",
        ROOT/"config/localization.toml", args.manifest.resolve())}
    (output/"manifest.json").write_text(json.dumps(frozen, indent=2)+"\n", encoding="utf-8")
    inventory = []
    for path in sorted((source/"tests").glob("test_*.py")):
        text = path.read_text(encoding="utf-8-sig")
        for function in ast.parse(text).body:
            if not isinstance(function, ast.FunctionDef) or not function.name.startswith("test_"):
                continue
            calls = sorted({ast.unparse(n.func) for n in ast.walk(function) if isinstance(n, ast.Call)})
            if any(any(token in call.lower() for token in ("motion", "skill", "egoruntime", "worldsim", "stewardhost", "navigate", "pick", "place", "grasp", "move")) for call in [*calls, function.name]):
                selector = f"tests/{path.name}::{function.name}"
                selected = any(node == selector or node.startswith(selector+"[") for node in manifest["selectors"])
                category = ("independent_cobot" if path.name.startswith("test_cobot") else
                            "localization_benchmarks" if "localization" in path.name or "ekf" in path.name or "odometry" in path.name else
                            "direct_physics" if "WorldSim" in calls and "EgoRuntime" not in calls else
                            "mock_or_offline" if "StewardHost" not in calls else "sensor_or_host_only")
                inventory.append({"selector": selector, "line": function.lineno, "selected": selected,
                                  "exclusionReason": None if selected else category, "calls": calls})
    (output/"screening.json").write_text(json.dumps(inventory, indent=2)+"\n", encoding="utf-8")
    for stage in args.stages:
        stage_dir = output/stage
        stage_dir.mkdir()
        results = []
        for index, node in enumerate(nodes):
            directory = stage_dir/f"case-{index:02d}"
            directory.mkdir()
            env = environment | {"STEWARD_MATRIX_STAGE": stage, "STEWARD_MATRIX_CASE_OUTPUT": str(directory),
                                 "STEWARD_MATRIX_MASK_SOURCE": args.mask_source,
                                 "STEWARD_MATRIX_SKILL_DIAGNOSTICS": str(int(args.skill_diagnostics)),
                                 "STEWARD_MATRIX_LOCALIZATION_CONFIG": str(ROOT/"config/localization.toml")}
            if args.mask_source == "simulation":
                env.pop("STEWARD_SEGMENTER_URL", None)
                env["STEWARD_MATRIX_CAPTURE_FINAL"] = str(int(args.capture_final or (stage == "clean-fused" and
                    "test_the_robot_takes_off_the_lid" in node and "puzzle-dog" in node)))
            started = time.perf_counter()
            timeout = False
            with (directory/"pytest.log").open("w", encoding="utf-8") as log:
                try:
                    run = subprocess.run([sys.executable, "-m", "pytest", "-p", "motion_matrix_plugin", "-q", "-s", node,
                                          "--basetemp", str(directory/"tmp")], cwd=source, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, timeout=args.case_timeout)
                    code = run.returncode
                except subprocess.TimeoutExpired:
                    code, timeout = None, True
            status = "timeout" if timeout else "passed" if code == 0 else "failed"
            recorded = directory/"pytest-result.json"
            phases = json.loads(recorded.read_text()) if recorded.exists() else None
            if phases and any(p["outcome"] == "skipped" for p in phases["phases"]):
                status = "skipped"
            result = {"nodeid": node, "status": status, "exitcode": code,
                      "wallSeconds": time.perf_counter()-started, "directory": str(directory)}
            results.append(result)
            (stage_dir/"results.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
            print(json.dumps({"stage": stage, "case": index, **result}), flush=True)
        print(json.dumps({"stage": stage, "selected": len(nodes), "passed": sum(r["status"] == "passed" for r in results)}), flush=True)


if __name__ == "__main__":
    main()
