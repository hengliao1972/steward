"""SAM preview pixels, camera isolation and read-only HTTP delivery."""

import base64
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image

from steward.ego.segmentation_preview import SegmentationPreview
from steward.ego.segmenter import Segmentation
from steward.ego.server import ego_api
from steward.net import call
from steward.world.server import StewardHost


def test_overlay_keeps_background_and_uses_the_source_frame():
    preview = SegmentationPreview()
    rgb = np.full((360, 480, 3), 100, np.uint8)
    mask = np.zeros((360, 480), np.uint8)
    mask[100:240, 100:240] = 1
    preview.publish("head_camera", 12.5, rgb, Segmentation(mask, {1: {"cup": 0.9}}))
    frame = preview.latest("head_camera")
    pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(frame["image"].split(",", 1)[1]))))
    assert np.max(np.abs(pixels[20, 20].astype(int) - 100)) <= 2
    assert np.max(np.abs(pixels[150, 150].astype(int) - 100)) > 20
    assert frame["stamp"] == 12.5
    assert frame["labels"][0]["category"] == "cup"
    assert preview.latest("wide_camera") is None
    assert "image" not in preview.latest("head_camera", frame["id"])
    preview.clear()
    assert preview.latest("head_camera") is None


def test_console_relay_is_unrecorded_and_reports_failure_and_phase_changes():
    preview = SegmentationPreview()
    runtime = SimpleNamespace(segmentation_preview=preview, perception_mode="rgbd-segmenter",
                              segmentation={"lastError": None})
    # An accidental recorded request would call runtime.apply, which this runtime deliberately lacks.
    api = ego_api(runtime).start()
    host = StewardHost.__new__(StewardHost)
    host.ego_apis = {"robot": api.url}
    try:
        query = {"robot": "robot", "camera": "head_camera"}
        assert host._relay_segmentation(query) == {"active": True, "error": None, "frame": None}
        preview.publish("head_camera", 3.0, np.zeros((8, 8, 3), np.uint8),
                        Segmentation(np.zeros((8, 8), np.uint8), {}))
        frame = host._relay_segmentation(query)["frame"]
        assert frame["labels"] == []
        assert frame["image"].startswith("data:image/jpeg;base64,")
        runtime.segmentation["lastError"] = "sam3 unreachable"
        result = host._relay_segmentation(query | {"after": frame["id"]})
        assert result["error"] == "sam3 unreachable"
        assert "image" not in result["frame"]
        runtime.perception_mode = "rgbd"
        assert host._relay_segmentation(query)["frame"] is None
        assert call("GET", api.url + "/ego/segmentation")["active"] is False
    finally:
        api.stop()
