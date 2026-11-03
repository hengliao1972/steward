"""Thin client for LinxLab SAM3 / SAM3.1 instance segmentation (PCS prompts).

Default deployment: ``https://linxlab-research.com/sam3``. Opt-in only — Phase 1 still
defaults to sim oracle masks unless ``mask_mode`` / ``mask_source`` selects ``sam3`` or
``geo_sam3``.
"""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


# Promptable Concept Segmentation prompts used in local Phase-1 ablations.
# LinxLab /api/segment requires each prompt to be an object (e.g. {"text": "cup"}).
DEFAULT_PCS_TEXTS: list[str] = [
    "cup", "mug", "bowl", "plate", "dish",
    "table", "counter", "cabinet", "drawer", "chair",
    "object", "container",
]
DEFAULT_PCS_PROMPTS: list[dict] = [{"text": t} for t in DEFAULT_PCS_TEXTS]


def normalize_prompts(prompts: list | None) -> list[dict]:
    """Accept ``["cup"]`` or ``[{"text": "cup"}]``; always emit API objects."""
    if not prompts:
        return list(DEFAULT_PCS_PROMPTS)
    out: list[dict] = []
    for p in prompts:
        if isinstance(p, str):
            out.append({"text": p})
        elif isinstance(p, dict):
            out.append(p)
        else:
            raise TypeError(f"prompt must be str or dict, got {type(p).__name__}")
    return out


@dataclass
class Sam3Config:
    base_url: str = "https://linxlab-research.com/sam3"
    threshold: float = 0.28
    min_pixels: int = 25
    timeout_s: float = 180.0
    prompts: list[dict] = field(default_factory=lambda: list(DEFAULT_PCS_PROMPTS))


class Sam3Client:
    def __init__(self, config: Sam3Config | None = None):
        if requests is None:
            raise RuntimeError("requests is required for Sam3Client (pip install requests)")
        self.config = config or Sam3Config()

    def ready(self) -> bool:
        try:
            r = requests.get(f"{self.config.base_url.rstrip('/')}/health/ready", timeout=15)
            if r.ok:
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                return body.get("status", "ready") in ("ready", "ok", True) or r.ok
            # Prefer /api/health when /health/ready is absent.
            r = requests.get(f"{self.config.base_url.rstrip('/')}/api/health", timeout=15)
            return r.ok and (r.json().get("status") == "ready" if r.headers.get("content-type", "").startswith("application/json") else r.ok)
        except Exception:
            return False

    def segment(self, rgb: np.ndarray, *, prompts: list | None = None,
                threshold: float | None = None) -> dict:
        """POST RGB → JSON with per-prompt mask lists (base64 PNG)."""
        cfg = self.config
        buf = io.BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, format="PNG")
        r = requests.post(
            f"{cfg.base_url.rstrip('/')}/api/segment",
            files={"file": ("frame.png", buf.getvalue(), "image/png")},
            data={"options": json.dumps({
                "prompts": normalize_prompts(prompts if prompts is not None else cfg.prompts),
                "threshold": float(cfg.threshold if threshold is None else threshold),
            })},
            timeout=cfg.timeout_s,
        )
        r.raise_for_status()
        return r.json()

    def masks_from_rgb(self, rgb: np.ndarray, *, prompts: list | None = None,
                       threshold: float | None = None) -> np.ndarray:
        """Anonymous uint8 instance mask (0 = background), same contract as geometric masks."""
        return results_to_mask(self.segment(rgb, prompts=prompts, threshold=threshold),
                               rgb.shape[:2], min_pixels=self.config.min_pixels)


def decode_mask_b64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64.split(",")[-1] if "," in b64 else b64)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("L")) > 127


def results_to_mask(resp: dict, hw: tuple[int, int], min_pixels: int = 25) -> np.ndarray:
    h, w = hw
    canvas = np.zeros((h, w), np.uint8)
    nid = 1
    for entry in resp.get("results") or []:
        for b64 in entry.get("masks") or []:
            m = decode_mask_b64(b64)
            if m.shape != (h, w):
                m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
            sel = m & (canvas == 0)
            if int(sel.sum()) < min_pixels or nid > 254:
                continue
            canvas[sel] = nid
            nid += 1
    return canvas


def merge_masks(geometric: np.ndarray, sam: np.ndarray) -> np.ndarray:
    """Union: keep geometric ids, append SAM instances with remapped ids (SAM wins on overlap)."""
    out = np.asarray(geometric, dtype=np.uint8).copy()
    sam = np.asarray(sam, dtype=np.uint8)
    if sam.shape != out.shape:
        raise ValueError(f"mask shape mismatch geo={out.shape} sam={sam.shape}")
    base = int(out.max())
    for iid in [i for i in np.unique(sam) if i > 0]:
        out[sam == iid] = base + int(iid)
    return out
