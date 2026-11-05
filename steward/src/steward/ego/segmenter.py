"""Robot-side open-vocabulary instance segmentation with SAM 3.1 (the ``phase1-sam`` level).

At this level the world sends RGB and depth only; the Ego Runtime segments chosen frames itself.
One frame is segmented in two passes: the whole image with every prompt of the vocabulary, then
the regions of support surfaces (tables, counters) cropped, enlarged and segmented one by one with
the small-object prompts (tiled into one image they are found far less often), because objects of a few dozen pixels are missed at the
camera's resolution. Detections of one object under several prompts are merged into one instance
that carries every label with its score; overlaps are resolved so that the smaller instance (a bowl)
keeps the pixels it shares with a larger one (the table under it). Regions SAM did not find are
then filled from depth (:mod:`steward.ego.providers.geometric_masks`) as unnamed instances, which
the shape priors alone classify.

Every result is cached on disk under a key of the image and the settings, so an episode replays
without the service and the same image is never sent twice (steward.md §13.7.4).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from steward import perception_services
from steward.ego.providers.geometric_masks import GeometricMaskConfig, masks_from_depth

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "segmentation.toml"


class SegmenterError(Exception):
    """A frame could not be segmented (service unreachable, rejected request, malformed reply)."""


@dataclass(frozen=True)
class SegmenterConfig:
    service: str = "sam3"
    threshold: float = 0.4
    merge_iou: float = 0.6
    min_pixels: int = 25
    crop_min_pixels: int = 8
    """Smallest mask kept from the enlarged crop. A cup three metres away is only a few dozen
    pixels after it is mapped back, below ``min_pixels``, and would otherwise be discarded."""
    crop_categories: tuple[str, ...] = ("table", "counter", "sideboard", "cabinet")
    crop_target_px: int = 480
    crop_max_scale: float = 4.0
    crop_up: float = 0.3
    crop_depth: float = 0.5
    max_crops: int = 2
    crop_prompts: tuple[str, ...] = ()
    shape_weight: float = 0.5
    """Exponent of the shape-prior probability in :func:`steward.ego.classify.fuse_semantics`."""
    label_floor: float = 0.03
    """Semantic likelihood of a category SAM did not name."""
    vocabulary: dict[str, str] = field(default_factory=dict)
    """Prompt text → Ego-State category."""
    geometric_fill: bool = False
    """Add depth clusters SAM left unlabelled as unnamed instances (:meth:`Segmenter.fill`).
    The shipped ``segmentation.toml`` turns this on; a constructed config leaves it off."""
    fill_free_share: float = 0.7
    """Share of a depth cluster's pixels SAM must have left unlabelled for it to be added."""
    fill_supports_only: bool = True
    """When the fill is on, add only clusters standing on a support. Residual walls and furniture
    stay out: depth clustering otherwise boxes a corner into a piece of furniture."""

    def key(self) -> str:
        """Digest of every setting that changes a SAM result, part of each cache key; the fill
        settings are left out because the fill is computed after the cached result."""
        sam = {k: v for k, v in asdict(self).items()
               if k not in ("geometric_fill", "fill_free_share", "fill_supports_only")}
        return hashlib.sha256(json.dumps(sam, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class Segmentation:
    """One segmented frame: ``mask`` (H, W) uint8 instance ids (0 = none) and, per id, the score of
    each category it was detected as."""

    mask: np.ndarray
    labels: dict[int, dict[str, float]]
    cached: bool = False
    seconds: float = 0.0


def load_config(path: Path = DEFAULT_PATH) -> SegmenterConfig:
    """The segmenter settings in ``path``; raises ``ValueError`` naming the offending key."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: {exc}") from None
    raw, vocab = data.get("segmenter"), data.get("vocabulary")
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: missing [segmenter] table")
    if not isinstance(vocab, dict) or not vocab or not all(isinstance(v, str) and v for v in vocab.values()):
        raise ValueError(f"{path}: [vocabulary] must map prompt texts to category names")
    defaults = SegmenterConfig()
    values = {}
    for key, default in asdict(defaults).items():
        if key == "vocabulary" or key not in raw:
            continue
        value = raw[key]
        if isinstance(default, tuple):
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError(f"{path}: segmenter.{key} must be a list of strings")
            value = tuple(value)
        elif isinstance(default, (int, float)) and not isinstance(default, bool):
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{path}: segmenter.{key} must be a non-negative number")
            value = type(default)(value)
        elif not isinstance(value, type(default)):
            raise ValueError(f"{path}: segmenter.{key} must be a {type(default).__name__}")
        values[key] = value
    unknown = set(raw) - set(asdict(defaults))
    if unknown:
        raise ValueError(f"{path}: unknown segmenter keys {sorted(unknown)}")
    config = SegmenterConfig(**values, vocabulary=dict(vocab))
    missing = [p for p in config.crop_prompts if p not in config.vocabulary]
    if missing:
        raise ValueError(f"{path}: segmenter.crop_prompts not in [vocabulary]: {missing}")
    return config


class Sam3Client:
    """``POST /api/segment`` of a SAM 3.1 service: one image, several text prompts."""

    def __init__(self, service: perception_services.Service):
        self.service = service
        self._base: str = service.urls[0]

    def segment(self, png: bytes, prompts: list[str], threshold: float) -> list[tuple[str, float, np.ndarray]]:
        """Every detection as ``(prompt, score, mask)``, the mask a boolean array of the image's size.

        A connection failure (the blocked domain resets it or answers with a block page) moves on to
        the service's next base URL, starting from the one that answered last; an error status from
        the service itself is not retried."""
        boundary = uuid.uuid4().hex
        options = json.dumps({"prompts": [{"text": p} for p in prompts], "threshold": threshold})
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"frame.png\"\r\n"
                f"Content-Type: image/png\r\n\r\n").encode() + png + \
               (f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"options\"\r\n\r\n{options}\r\n"
                f"--{boundary}--\r\n").encode()
        urls = self.service.urls
        order = [self._base, *(u for u in urls if u != self._base)]
        failures = []
        for base in order:
            method, url = self.service.url(base, "segment")
            request = urllib.request.Request(url, data=body, method=method,
                                             headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            try:
                with urllib.request.urlopen(request, timeout=self.service.timeout_s) as resp:
                    raw, kind = resp.read(), resp.headers.get("Content-Type", "")
            except urllib.error.HTTPError as exc:
                if exc.headers.get("Content-Type", "").startswith("application/json"):
                    raise SegmenterError(f"{self.service.name}: HTTP {exc.code}: {exc.read()[:300]!r}") from None
                failures.append(f"{base}: HTTP {exc.code} (not the service)")
                continue
            except (urllib.error.URLError, OSError) as exc:
                failures.append(f"{base}: {getattr(exc, 'reason', exc)}")
                continue
            if not kind.startswith("application/json"):
                failures.append(f"{base}: {kind or 'no content type'} reply (not the service)")
                continue
            self._base = base
            break
        else:
            raise SegmenterError(f"{self.service.name}: no reachable URL ({'; '.join(failures)})")
        try:
            reply = json.loads(raw)
        except ValueError as exc:
            raise SegmenterError(f"{self.service.name}: malformed reply ({exc})") from None
        try:
            return [(r["prompt"]["text"], float(score), _decode_mask(png_b64))
                    for r in reply["results"] for score, png_b64 in zip(r["scores"], r["masks"], strict=True)]
        except (KeyError, TypeError, ValueError) as exc:
            raise SegmenterError(f"{self.service.name}: malformed reply ({type(exc).__name__}: {exc})") from None


def _decode_mask(data: str) -> np.ndarray:
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data))).convert("L")) > 127


def _png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _merge_boxes(boxes: list[tuple[int, int, int, int]], inside: float = 0.7) -> list[tuple[int, int, int, int]]:
    """Union ``(y0, y1, x0, x1)`` boxes of which one lies mostly (``inside`` of its area) in the
    other, largest area first. Merely overlapping neighbours stay apart: a union would be enlarged
    less and its small objects missed."""
    merged = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                common = max(0, min(a[1], b[1]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[2], b[2]))
                smaller = min((a[1] - a[0]) * (a[3] - a[2]), (b[1] - b[0]) * (b[3] - b[2]))
                if smaller and common / smaller >= inside:
                    merged[i] = [min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])]
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return sorted((tuple(b) for b in merged), key=lambda b: -(b[1] - b[0]) * (b[3] - b[2]))


def _crop_background(mask: np.ndarray) -> bool:
    """Whether a crop mask paints most of its window: background mixed in, not an object on the surface."""
    return mask.size > 0 and int(mask.sum()) >= 0.6 * mask.size


def _furniture_slice(category: str, mask: np.ndarray, parents: list[tuple[str, np.ndarray]]) -> bool:
    """Whether a ``box`` detection is the upper body of a cabinet, sideboard or counter.

    Most of its pixels lie on that furniture and it covers at least a quarter of the furniture.
    A box standing on the top is much smaller than the body, so it stays.

    :param category: the detection's category name.
    :param mask: the detection in the full image.
    :param parents: support surfaces from the full-image pass, each ``(category, mask)``.
    """
    if category != "box":
        return False
    area = int(mask.sum())
    if area < 1:
        return False
    for parent_cat, parent in parents:
        if parent_cat not in ("cabinet", "sideboard", "counter"):
            continue
        parent_area = int(parent.sum())
        if parent_area < 1:
            continue
        inside = int(np.logical_and(mask, parent).sum())
        if inside >= 0.7 * area and area >= 0.25 * parent_area:
            return True
    return False


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


class Segmenter:
    """Segments RGB frames with a SAM 3.1 client, caching every result under ``cache_dir``."""

    def __init__(self, config: SegmenterConfig, client, cache_dir: Path | None = None):
        self.config = config
        self.client = client
        self.cache_dir = cache_dir
        self._key = config.key()

    @classmethod
    def from_files(cls, config_path: Path = DEFAULT_PATH, services_path: Path = perception_services.DEFAULT_PATH,
                   cache_dir: Path | None = None) -> Segmenter:
        """The segmenter configured in ``config_path`` with its service from ``services_path``."""
        config = load_config(config_path)
        services = perception_services.load(services_path)
        if config.service not in services:
            raise ValueError(f"{config_path}: segmenter.service {config.service!r} is not in {services_path}")
        return cls(config, Sam3Client(services[config.service]), cache_dir)

    def segment(self, rgb: np.ndarray) -> Segmentation:
        """Instances of one RGB image (H, W, 3) uint8. Raises :class:`SegmenterError` when the
        service fails; nothing is cached then."""
        digest = hashlib.sha256(rgb.tobytes() + str(rgb.shape).encode() + self._key.encode()).hexdigest()
        path = None if self.cache_dir is None else self.cache_dir / digest[:2] / f"{digest}.npz"
        if path is not None and path.exists():
            with np.load(path) as z:
                labels = {int(k): v for k, v in json.loads(str(z["labels"])).items()}
                return Segmentation(z["mask"], labels, cached=True)
        started = time.monotonic()
        detections = self._full_pass(rgb)
        detections += self._crop_pass(rgb, detections)
        mask, labels = self._instances(detections, rgb.shape[:2])
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, mask=mask, labels=np.array(json.dumps(labels)))
            tmp.replace(path)
        return Segmentation(mask, labels, seconds=time.monotonic() - started)

    def fill(self, seg: Segmentation, depth: np.ndarray, fovy_deg: float, up_cam: np.ndarray) -> Segmentation:
        """``seg`` with every depth cluster of which at least ``fill_free_share`` of the pixels SAM
        left unlabelled added, on those pixels, as an instance without labels.

        :param depth: (H, W) metres of the segmented image.
        :param up_cam: the world's up direction in the camera frame (x forward, z up).
        :returns: ``seg`` itself when the fill is off."""
        cfg = self.config
        if not cfg.geometric_fill:
            return seg
        geo = masks_from_depth(depth, fovy_deg, GeometricMaskConfig(min_pixels=cfg.min_pixels), up_cam=up_cam,
                               supports_only=cfg.fill_supports_only)
        mask, labels = seg.mask.copy(), dict(seg.labels)
        free = mask == 0
        next_id = int(mask.max()) + 1
        for instance in np.unique(geo):
            if instance == 0 or next_id > 255:
                continue
            region = geo == instance
            own = region & free
            if int(own.sum()) < cfg.min_pixels or own.sum() < cfg.fill_free_share * region.sum():
                continue
            mask[own] = next_id
            next_id += 1
        return Segmentation(mask, labels, cached=seg.cached, seconds=seg.seconds)

    # ------------------------------------------------------------------ passes
    def _detections(self, rgb: np.ndarray, prompts: list[str]) -> list[tuple[str, float, np.ndarray]]:
        cfg = self.config
        out = []
        for prompt, score, mask in self.client.segment(_png(rgb), prompts, cfg.threshold):
            if score >= cfg.threshold and prompt in cfg.vocabulary and mask.shape == rgb.shape[:2]:
                out.append((cfg.vocabulary[prompt], score, mask, False))
        return out

    def _full_pass(self, rgb: np.ndarray) -> list[tuple[str, float, np.ndarray]]:
        return self._detections(rgb, list(self.config.vocabulary))

    def _crop_pass(self, rgb: np.ndarray, found: list[tuple[str, float, np.ndarray]]) -> list[tuple[str, float, np.ndarray]]:
        """Segment the enlarged regions of support surfaces, one request per merged region."""
        cfg = self.config
        if not cfg.crop_prompts or cfg.crop_target_px <= 0:
            return []
        h, w = rgb.shape[:2]
        boxes = []
        for category, _score, mask, _from_crop in found:
            if category not in cfg.crop_categories:
                continue
            ys, xs = np.nonzero(mask)
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            height = y1 - y0
            boxes.append((max(0, y0 - int(cfg.crop_up * height)), min(h, y0 + max(1, int(cfg.crop_depth * height))),
                          max(0, x0 - 4), min(w, x1 + 4)))
        out = []
        for y0, y1, x0, x1 in _merge_boxes(boxes)[:cfg.max_crops]:
            scale = min(cfg.crop_max_scale, cfg.crop_target_px / max(y1 - y0, x1 - x0))
            if scale <= 1.0:
                continue  # already large in the image: the full pass saw it at full resolution
            size = (max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale)))
            tile = np.asarray(Image.fromarray(rgb[y0:y1, x0:x1]).resize(size, Image.BICUBIC))
            parents = [(category, mask) for category, _score, mask, _from_crop in found]
            for category, score, mask, _from_crop in self._detections(tile, list(cfg.crop_prompts)):
                small = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((x1 - x0, y1 - y0), Image.NEAREST)) > 127
                if _crop_background(small):
                    continue  # the enlarged window's background came back as part of the mask
                full = np.zeros((h, w), dtype=bool)
                full[y0:y1, x0:x1] = small
                if _furniture_slice(category, full, parents):
                    continue  # the upper body of the cabinet, not a box standing on it
                if full.any():
                    out.append((category, score, full, True))
        return out

    def _instances(self, detections, shape) -> tuple[np.ndarray, dict[int, dict[str, float]]]:
        """Merge detections of one object and paint instances largest first, so smaller ones keep
        the pixels they share with what they stand on.

        A detection found only in an enlarged crop is kept down to ``crop_min_pixels``. The full
        image still drops anything under ``min_pixels``."""
        cfg = self.config
        groups: list[tuple[np.ndarray, dict[str, float], bool]] = []
        for category, score, mask, from_crop in sorted(detections, key=lambda d: -d[1]):
            for i, (gmask, glabels, crop_only) in enumerate(groups):
                if _iou(gmask, mask) >= cfg.merge_iou:
                    glabels[category] = max(glabels.get(category, 0.0), score)
                    groups[i] = (gmask | mask, glabels, crop_only and from_crop)
                    break
            else:
                groups.append((mask.copy(), {category: score}, from_crop))
        groups.sort(key=lambda g: -int(g[0].sum()))
        canvas = np.zeros(shape, dtype=np.int32)
        for i, (gmask, _, _) in enumerate(groups, start=1):
            canvas[gmask] = i
        mask = np.zeros(shape, dtype=np.uint8)
        labels: dict[int, dict[str, float]] = {}
        for i, (_, glabels, crop_only) in enumerate(groups, start=1):
            visible = canvas == i
            floor = cfg.crop_min_pixels if crop_only else cfg.min_pixels
            if int(visible.sum()) < floor or len(labels) >= 255:
                continue
            new_id = len(labels) + 1
            mask[visible] = new_id
            labels[new_id] = {k: round(v, 4) for k, v in glabels.items()}
        return mask, labels
