"""Test-only simulated semantic masks; never installed by the application runtime."""

import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np


def semantic_labels(parts, mask, categories):
    """Associate shuffled mask ids with categories, exporting no entity ids or poses."""
    labels = {}
    for instance in np.unique(mask):
        if instance == 0:
            continue
        origins = np.unique(parts[mask == instance])
        if len(origins) != 1 or origins[0] < 0:
            raise ValueError("a simulated instance mask must cover exactly one perception part")
        labels[int(instance)] = {categories[int(origins[0])]: 1.0}
    return labels


def append_labels(blob, labels):
    """Append a small NPY entry while preserving all existing compressed sensor members."""
    metadata = io.BytesIO()
    np.save(metadata, np.array(json.dumps(labels)), allow_pickle=False)
    stream = io.BytesIO(blob)
    with zipfile.ZipFile(stream, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("matrix_semantic_labels.npy", metadata.getvalue())
    return stream.getvalue()


def install(monkeypatch, main, *, matrix=True):
    """Install private packet metadata and localization-only semantics for one pytest case."""
    from steward.world.sim import WorldSim
    from steward.world.server import StewardHost
    from steward.ego import runtime as runtime_module
    from steward.ego.segmenter import Segmenter

    original_rgbd = WorldSim._rgbd_now
    original_masks = WorldSim._anonymous_masks
    original_decode = runtime_module.decode_rgbd
    original_frames = runtime_module.EgoRuntime._localization_frames

    def masks(sim, parts, rng):
        mask = original_masks(sim, parts, rng)
        categories = [category for entity in sim.template.entities if entity.id != sim._matrix_camera_robot
                      for _part, _links, category in sim.perception_parts(entity)]
        sim._matrix_mask_labels = semantic_labels(parts, mask, categories)
        return mask

    def rgbd(sim, robot_id, sensor_id=None):
        sim._matrix_camera_robot = robot_id
        sim._matrix_mask_labels = None
        blob = original_rgbd(sim, robot_id, sensor_id)
        if sim._matrix_mask_labels is None:
            if not matrix:
                return blob
            raise ValueError("simulation-mask experiments require world instance masks")
        return append_labels(blob, sim._matrix_mask_labels)

    def decode(blob):
        frame = original_decode(blob)
        with np.load(io.BytesIO(blob)) as packet:
            if "matrix_semantic_labels" not in packet.files:
                if not matrix:
                    return frame
                raise ValueError("missing simulation-mask experiment metadata")
            frame["_matrix_labels"] = {int(key): value for key, value in
                                       json.loads(str(packet["matrix_semantic_labels"])).items()}
        return frame

    def frames(runtime, header, available):
        if not runtime.config.localization_apply:
            return original_frames(runtime, header, available)
        if header.get("episode") != runtime._localization_epoch:
            runtime._localization_epoch, runtime._next_localization = header.get("episode"), 0.0
        if header["stamp"] + 1e-9 < runtime._next_localization:
            return []
        camera = runtime.localizer.selector.config.camera
        selected = next((frame for frame in available if frame["camera"] == camera), None)
        if selected is None:
            selected = decode(runtime.link.rgbd(camera))
        labels = selected.get("_matrix_labels")
        if labels is None:
            raise ValueError("simulation-mask localization requires world instance masks; use --localization-mask-source=sam")
        runtime._next_localization = selected["header"]["stamp"] + runtime.config.localization_interval_s
        runtime._fusion_stats["simulatedMaskFrames"] = runtime._fusion_stats.get("simulatedMaskFrames", 0)+1
        return [{**selected, "labels": labels}]

    class NoSam:
        def segment(self, _rgb):
            raise AssertionError("simulation-mask experiments must never request SAM")

    monkeypatch.setattr(WorldSim, "_anonymous_masks", masks)
    monkeypatch.setattr(WorldSim, "_rgbd_now", rgbd)
    monkeypatch.setattr(runtime_module, "decode_rgbd", decode)
    monkeypatch.setattr(runtime_module.EgoRuntime, "_localization_frames", frames)
    if matrix:
        monkeypatch.setattr(Segmenter, "from_files", classmethod(lambda cls, **kwargs: NoSam()))

    if matrix and os.environ.get("STEWARD_MATRIX_CAPTURE_FINAL") == "1":
        original_stop = StewardHost.stop
        def stop(host):
            def capture():
                camera = {"lookat": [0.3, -0.6, 0.65], "distance": 3.4, "azimuth": 130, "elevation": -35}
                directory = Path(os.environ["STEWARD_MATRIX_CASE_OUTPUT"])
                with host.sim.lock:
                    png = host.sim.render(camera, width=1280, height=960)
                    stamp = host.sim.time
                (directory/"final-world.png").write_bytes(png)
                (directory/"final-render.json").write_text(json.dumps({"camera": camera, "stamp": stamp,
                    "source": "MuJoCo final scene before host.stop", "width": 1280, "height": 960}, indent=2)+"\n", encoding="utf-8")
            main(capture)
            return original_stop(host)
        monkeypatch.setattr(StewardHost, "stop", stop)
