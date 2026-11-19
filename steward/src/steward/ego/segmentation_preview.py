"""Bounded, read-only previews of completed SAM frames for the operator console."""

from __future__ import annotations

import base64
import colorsys
import io
import threading
import uuid

import numpy as np
from PIL import Image

from steward.ego.segmenter import Segmentation


class SegmentationPreview:
    """Keep one immutable encoded preview per camera; reads never invoke perception."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frames: dict[str, dict] = {}

    def publish(self, camera: str, stamp: float, rgb: np.ndarray, result: Segmentation) -> None:
        """Overlay SAM instances on their source RGB frame and retain a JPEG up to 480 px wide.

        :param camera: Robot sensor id.
        :param stamp: Source frame's simulation time.
        :param rgb: Source RGB pixels, with the same dimensions as the instance mask.
        :param result: SAM instances before geometric filling and peeling.
        """
        image = Image.fromarray(rgb).convert("RGB")
        image.thumbnail((480, 360))
        mask = np.asarray(Image.fromarray(result.mask).resize(image.size, Image.Resampling.NEAREST))
        pixels = np.array(image)
        labels = []
        for instance in np.unique(result.mask):
            if instance == 0:
                continue
            color = tuple(round(c * 255) for c in colorsys.hsv_to_rgb((int(instance) * 0.618034) % 1, 0.75, 1))
            region = mask == instance
            pixels[region] = (pixels[region] * 0.55 + np.array(color) * 0.45).astype(np.uint8)
            scores = result.labels.get(int(instance), {})
            category = max(scores, key=scores.get) if scores else "unknown"
            labels.append({"id": int(instance), "category": category, "score": scores.get(category),
                           "color": "#" + "".join(f"{v:02x}" for v in color)})
        encoded = io.BytesIO()
        Image.fromarray(pixels).save(encoded, format="JPEG", quality=85)
        frame = {"id": uuid.uuid4().hex, "camera": camera, "stamp": stamp, "labels": labels,
                 "image": "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")}
        with self._lock:
            self._frames[camera] = frame

    def latest(self, camera: str, after: str | None = None) -> dict | None:
        """Return preview metadata, omitting unchanged image pixels when ``after`` matches its id.

        :param camera: Robot sensor id.
        :param after: Last displayed preview id.
        :returns: A copy of the preview, or None before the first completed SAM result.
        """
        with self._lock:
            frame = self._frames.get(camera)
            if frame is None:
                return None
            return {k: v for k, v in frame.items() if k != "image" or after != frame["id"]}

    def clear(self) -> None:
        """Discard previews when the robot enters a new episode."""
        with self._lock:
            self._frames.clear()
