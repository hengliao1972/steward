"""Full physical route still produces odometry predictions when segmentation finds no targets."""

import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from steward.ego.segmenter import Segmentation, Segmenter
from test_localization_loop import test_room_loop_localization as run_room_loop


def test_empty_semantic_observations_do_not_interrupt_continuous_room_loop(kitchen_root, tmp_path, monkeypatch):
    class NoTargets:
        def segment(self, rgb):
            return Segmentation(np.zeros(rgb.shape[:2], np.uint8), {})

    monkeypatch.setattr(Segmenter, "from_files", classmethod(lambda cls, **kwargs: NoTargets()))
    run_room_loop(kitchen_root, tmp_path, 0, "segmented")
    report = json.loads((tmp_path / "report.json").read_text())
    output = report["segmented"]
    assert output["observations"] > 50
    assert output["acceptedCorrections"] == 0
    assert output["continuousPositionRmseM"] == pytest.approx(report["positionRmseM"])
    assert output["longestTimeWithoutAcceptedCorrectionS"] == pytest.approx(report["simulationSeconds"])
    assert output["finalCorrectionAgeS"] is None
