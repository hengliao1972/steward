"""Archive the completed four-stage matrix, including failures and actual feedback coverage."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from retest_motion_timeouts import timeout_reason
from motion_matrix_report import write_reports


def archive_log(source: Path, directory: Path) -> str:
    """Preserve exact log bytes in gzip and provide an LF copy without trailing whitespace."""
    raw = source.read_bytes()
    (directory/"pytest.log.gz").write_bytes(gzip.compress(raw, mtime=0))
    readable = "\n".join(line.rstrip() for line in raw.decode("utf-8").splitlines()).rstrip()+"\n"
    (directory/"pytest.log").write_bytes(readable.encode("utf-8"))
    return hashlib.sha256(raw).hexdigest()


def analysis_manifest(raw: dict, cohort: dict | None, baseline: list[dict]) -> dict:
    """Resolve a baseline-only cohort, retaining every passing raw case in its collected order."""
    effective = dict(raw)
    if cohort is None:
        return effective
    retained = cohort["retainedNodes"]
    if not retained or len(set(retained)) != len(retained) or not set(retained) <= set(raw["nodes"]):
        raise ValueError("analysis cohort must be a nonempty unique subset of the raw cases")
    if retained != [r["nodeid"] for r in baseline if r["status"] == "passed"]:
        raise ValueError("cohort must retain every passing baseline case in raw order")
    effective.update(cohort["effectiveManifest"])
    effective["nodes"] = retained
    effective["manifestSha256"] = hashlib.sha256(json.dumps(retained).encode()).hexdigest()
    return effective


def summarize(pre: Path, post: Path, output: Path, extended: list[Path] | None = None):
    """Require identical complete manifests; preserve per-case logs and sensor/feedback evidence."""
    manifests = [json.loads((path/"manifest.json").read_text(encoding="utf-8")) for path in (pre, post)]
    if manifests[0]["nodes"] != manifests[1]["nodes"]:
        raise ValueError("the two runs selected different cases")
    if manifests[0].get("maskSource", "sam") != manifests[1].get("maskSource", "sam"):
        raise ValueError("the two runs used different segmentation sources")
    raw_count = len(manifests[0]["nodes"])
    effective = dict(manifests[0])
    cohort_path = pre/"cohort-selection.json"
    cohort = None
    if cohort_path.exists():
        cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
        baseline = json.loads((pre/"baseline/results.json").read_text(encoding="utf-8"))
        effective = analysis_manifest(manifests[0], cohort, baseline)
    retained = set(effective["nodes"])
    count = len(retained)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pre/"manifest.json", output/"raw-manifest.json")
    (output/"manifest.json").write_text(json.dumps(effective, indent=2)+"\n", encoding="utf-8")
    for name in ("screening.json", "collection.log", "input-manifest.json", "cohort-selection.json"):
        if not (pre/name).exists():
            continue
        shutil.copyfile(pre/name, output/name)
    if not (output/"protocol.json").exists():
        (output/"protocol.json").write_text(json.dumps({"caseTimeoutSeconds": manifests[0]["caseTimeoutSeconds"],
            "maskSource": manifests[0].get("maskSource", "sam"), "cumulativeDistanceScale": 0.01,
            "headingDriftRadiansPerMetre": 0.005, "instantaneousOdometrySigma": 0.003,
            "removedBaselineFailures": effective.get("removedBaselineFailures", {}),
            "localizationPoseInput": "raw odometry; no GT robot poses", "assertions": "unchanged"}, indent=2)+"\n", encoding="utf-8")
    rows, raw_rows, totals = [], [], []
    for parent, stages in ((pre, ("baseline", "drift")), (post, ("drift-fused", "clean-fused"))):
        for stage in stages:
            results = json.loads((parent/stage/"results.json").read_text(encoding="utf-8"))
            if len(results) != raw_count or [r["nodeid"] for r in results] != manifests[0]["nodes"]:
                raise ValueError(f"{stage} is incomplete or reordered")
            for index, result in enumerate(results):
                source = parent/stage/f"case-{index:02d}"
                target = output/stage/f"case-{index:02d}"
                target.mkdir(parents=True, exist_ok=True)
                log_hash = archive_log(source/"pytest.log", target)
                for name in ("pytest-result.json", "evidence.json", "final-world.png", "final-render.json"):
                    if (source/name).exists():
                        shutil.copyfile(source/name, target/name)
                evidence = json.loads((source/"evidence.json").read_text(encoding="utf-8"))
                phases_path = source/"pytest-result.json"
                phases = json.loads(phases_path.read_text(encoding="utf-8"))["phases"] if phases_path.exists() else []
                failures = [p for p in phases if p["outcome"] == "failed"]
                stats = evidence["runtimes"]
                row = {"stage": stage, "case": index, **result, "directory": target.relative_to(output).as_posix(),
                       "rawLogSha256": log_hash,
                       "failedPhases": [p["phase"] for p in failures],
                       "failures": [p["details"] for p in failures], "evidence": evidence,
                       "mappedHeaders": sum(r["integration"].get("mappedHeaders", 0) for r in stats),
                       "visualFrames": sum(r["integration"].get("visualFrames", 0) for r in stats),
                       "acceptedVisual": sum(r["integration"].get("acceptedVisual", 0) for r in stats),
                       "visualErrors": sum(r["integration"].get("visualErrors", 0) for r in stats),
                       "independentSamRequests": sum(r["integration"].get("independentSamRequests", 0) for r in stats)}
                row["simulatedMaskFrames"] = sum(r["integration"].get("simulatedMaskFrames", 0) for r in stats)
                if (target/"final-world.png").exists():
                    row["finalRender"] = (target/"final-world.png").relative_to(output).as_posix()
                raw_rows.append(row)
                if row["nodeid"] in retained:
                    rows.append(row)
            stage_rows = [r for r in rows if r["stage"] == stage]
            totals.append({"stage": stage, "selected": count,
                           **{status: sum(r["status"] == status for r in stage_rows)
                              for status in ("passed", "failed", "timeout", "skipped")},
                           "passRate": sum(r["status"] == "passed" for r in stage_rows)/count,
                           "wallSeconds": sum(r["wallSeconds"] for r in stage_rows),
                           **{key: sum(r[key] for r in stage_rows) for key in
                              ("mappedHeaders", "visualFrames", "acceptedVisual", "visualErrors", "independentSamRequests", "simulatedMaskFrames")}})
    (output/"raw-summary.json").write_text(json.dumps({"manifest": manifests[0], "cases": raw_rows}, indent=2)+"\n", encoding="utf-8")
    payload = {"manifestSha256": effective["manifestSha256"], "manifest": effective, "stages": totals, "cases": rows,
               "excludedCases": [r for r in raw_rows if r["nodeid"] not in retained], "cohortSelection": cohort}
    if extended is not None:
        records, manifests_extended, outcomes, seen = [], [], [], set()
        for parent in extended:
            manifest = json.loads((parent/"manifest.json").read_text(encoding="utf-8"))
            results = json.loads((parent/"results.json").read_text(encoding="utf-8"))
            if len(results) != len(manifest["selected"]):
                raise ValueError("extended reruns are incomplete")
            manifests_extended.append(manifest)
            outcomes.extend(results)
        for result in outcomes:
            if result["nodeid"] not in retained:
                raise ValueError("an extended rerun belongs to an excluded case")
            key = result["stage"], result["case"]
            if key in seen:
                raise ValueError(f"duplicate extended result for {key}")
            seen.add(key)
            directory = Path(result["directory"])
            target = output/"extended"/result["stage"]/f"case-{result['case']:02d}"
            target.mkdir(parents=True, exist_ok=True)
            log_hash = archive_log(directory/"pytest.log", target)
            for name in ("pytest-result.json", "evidence.json", "deadline-overrides.json", "final-world.png", "final-render.json"):
                if (directory/name).exists():
                    shutil.copyfile(directory/name, target/name)
            report_path = directory/"pytest-result.json"
            phases = json.loads(report_path.read_text(encoding="utf-8"))["phases"] if report_path.exists() else []
            failures = [p for p in phases if p["outcome"] == "failed"]
            records.append({**result, "directory": target.relative_to(output).as_posix(),
                            "rawLogSha256": log_hash,
                            "failedPhases": [p["phase"] for p in failures],
                            "failures": [p["details"] for p in failures],
                            "evidence": json.loads((directory/"evidence.json").read_text(encoding="utf-8")),
                            "internalDeadlineReached": result["status"] == "failed" and timeout_reason(result, directory) is not None})
            if (directory/"final-world.png").exists():
                records[-1]["finalRender"] = (target/"final-world.png").relative_to(output).as_posix()
        protocol = {"rounds": manifests_extended}
        payload["extended"] = {"protocol": protocol, "cases": records}
        (output/"extended-manifest.json").write_text(json.dumps(protocol, indent=2)+"\n", encoding="utf-8")
        replaced = {(r["stage"], r["case"]): r for r in records}
        composite = []
        for stage in totals:
            effective = [replaced.get((r["stage"], r["case"]), r) for r in rows if r["stage"] == stage["stage"]]
            passed = sum(r["status"] == "passed" for r in effective)
            external = sum(r["status"] == "timeout" for r in effective)
            internal = sum(r.get("internalDeadlineReached", False) for r in effective)
            composite.append({"stage": stage["stage"], "selected": count, "passed": passed,
                              "externalTimeout": external, "internalDeadline": internal,
                              "otherFailures": count-passed-external-internal, "passRate": passed/count})
        payload["compositeStages"] = composite
    (output/"summary.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    keys = ["stage", "case", "nodeid", "status", "wallSeconds", "mappedHeaders", "visualFrames", "acceptedVisual", "visualErrors", "simulatedMaskFrames"]
    with (output/"comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    source_root = Path(__file__).resolve().parents[1]
    hashes = {}
    for relative in ("src/steward/ego/runtime.py", "src/steward/ego/localization_feedback.py",
                     "src/steward/ego/pose_ekf.py", "src/steward/ego/distance_localization.py",
                     "config/localization.toml", "scripts/motion_matrix_plugin.py", "scripts/motion_control_matrix.py",
                     "scripts/motion_matrix_extended_plugin.py", "scripts/retest_motion_timeouts.py",
                     "scripts/motion_matrix_simulation_masks.py"):
        hashes[relative] = hashlib.sha256((source_root/relative).read_bytes()).hexdigest()
    (output/"source-sha256.json").write_text(json.dumps(hashes, indent=2)+"\n", encoding="utf-8")
    write_reports(payload, output)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extended", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps(summarize(args.pre, args.post, args.output, args.extended)["stages"], indent=2))


if __name__ == "__main__":
    main()
