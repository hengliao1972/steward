"""Serve the physical dog-puzzle localization demonstration on a local console."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    """Launch a private simulated-mask demo, or signal an existing demo to close."""
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("output", type=Path)
    args.add_argument("--stop", action="store_true")
    options = args.parse_args()
    output = options.output.resolve()
    if options.stop:
        (output / "stop").touch()
        return 0
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    env = os.environ | {
        "PYTHONPATH": os.pathsep.join(map(str, (root / "src", root / "scripts"))),
        "STEWARD_MATRIX_STAGE": "drift-fused",
        "STEWARD_MATRIX_MASK_SOURCE": "simulation",
        "STEWARD_MATRIX_CASE_OUTPUT": str(output),
        "STEWARD_MATRIX_LOCALIZATION_CONFIG": str(root / "config/localization.toml"),
        "STEWARD_LOCALIZATION_LIVE_DEMO": str(output),
    }
    env.pop("STEWARD_SEGMENTER_URL", None)
    node = ("tests/test_puzzle.py::test_the_robot_takes_off_the_lid_and_pours_the_pieces_out_onto_the_table"
            "[puzzle-dog-4-lid_at1-pour_at1-12]")
    return subprocess.call([sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
                            "-p", "motion_matrix_plugin", "--basetemp", str(output / "tmp"), node], cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
