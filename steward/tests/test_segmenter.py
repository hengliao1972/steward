"""Robot-side SAM 3.1 segmentation (the ``phase1-sam`` level): instance merging, crops, cache,
semantic fusion and keyframe gating, against a stand-in for the service."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from conftest import CAMERA_UP
from steward.assets.generators import steward_bot
from steward.ego.classify import fuse_semantics
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.segmenter import Segmentation, SegmenterConfig, SegmenterError, Segmenter, load_config
from steward.world.server import StewardHost

VOCAB = {"table": "table", "bowl": "bowl", "plate": "plate", "wall": "wall"}


def _box(shape, y0, y1, x0, x1) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


class FakeClient:
    """Answers each request from ``reply(image, prompts)`` and records what was asked."""

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[tuple[tuple[int, ...], list[str]]] = []

    def segment(self, png, prompts, threshold):
        from io import BytesIO

        from PIL import Image

        image = np.asarray(Image.open(BytesIO(png)))
        self.calls.append((image.shape, list(prompts)))
        return self.reply(image, prompts)


def test_detections_of_one_object_merge_and_a_small_object_keeps_the_pixels_it_shares():
    shape = (60, 80)
    table, bowl = _box(shape, 30, 60, 0, 80), _box(shape, 25, 35, 30, 40)

    def reply(image, prompts):
        return [("table", 0.9, table), ("bowl", 0.7, bowl), ("plate", 0.5, bowl), ("wall", 0.2, ~table)]

    seg = Segmenter(SegmenterConfig(vocabulary=VOCAB, crop_prompts=()), FakeClient(reply)).segment(
        np.zeros((*shape, 3), np.uint8))
    assert sorted(map(sorted, seg.labels.values())) == [["bowl", "plate"], ["table"]], "the 0.2 wall is below threshold"
    bowl_id = next(i for i, l in seg.labels.items() if "bowl" in l)
    assert seg.labels[bowl_id] == {"bowl": 0.7, "plate": 0.5}
    assert (seg.mask == bowl_id).sum() == bowl.sum(), "the bowl is painted over the table"


def test_support_surfaces_are_cropped_enlarged_and_their_small_objects_mapped_back():
    shape = (120, 160)
    table = _box(shape, 80, 120, 20, 140)

    def reply(image, prompts):
        if image.shape[:2] == shape:
            return [("table", 0.9, table)]
        h, w = image.shape[:2]  # the crop: a bowl in its middle
        return [("bowl", 0.8, _box((h, w), h // 3, 2 * h // 3, w // 3, w // 2))]

    client = FakeClient(reply)
    cfg = SegmenterConfig(vocabulary=VOCAB, crop_prompts=("bowl",), crop_target_px=480, crop_max_scale=4.0)
    seg = Segmenter(cfg, client).segment(np.zeros((*shape, 3), np.uint8))
    assert len(client.calls) == 2 and client.calls[1][1] == ["bowl"]
    assert client.calls[1][0][1] > 160 * 2, "the crop is enlarged"
    bowl_id = next(i for i, l in seg.labels.items() if "bowl" in l)
    ys, _ = np.nonzero(seg.mask == bowl_id)
    assert 68 <= ys.min() and ys.max() <= 100, "the bowl lands on the table's upper edge in the full frame"


SPECK_SCENE = [((-1.0, -4.0, -0.1), (8.0, 4.0, 0.0)), ((1.6, -0.6, 0.0), (2.4, 0.6, 0.9)),
               ((1.96, -0.04, 0.9), (2.04, 0.04, 1.0))]
"""A floor, a counter and a cup-sized box on it, for :func:`conftest._box_depth`: at 120 rows the
box is some two dozen pixels."""


def _bowl_in_any_crop(rows: int):
    def reply(image, prompts):
        if image.shape[0] == rows:
            return []
        hh, ww = image.shape[:2]
        return [("bowl", 0.85, _box((hh, ww), hh // 4, 3 * hh // 4, ww // 4, 3 * ww // 4))]
    return reply


def test_a_speck_above_a_support_is_enlarged_until_it_can_be_named(box_depth):
    """A cluster on a counter under the fill floor is asked as its own enlarged crop, and the
    bowl that crop returns is kept in the full frame."""
    h, w = 120, 160
    depth, owner = box_depth((h, w), 60.0, SPECK_SCENE)
    assert 8 <= int((owner == 2).sum()) < 60
    client = FakeClient(_bowl_in_any_crop(h))
    cfg = SegmenterConfig(vocabulary={"bowl": "bowl"}, crop_prompts=("bowl",), crop_target_px=480,
                          crop_max_scale=4.0, min_pixels=60, crop_min_pixels=8)
    blank = Segmentation(np.zeros((h, w), np.uint8), {})
    out = Segmenter(cfg, client).zoom(blank, np.zeros((h, w, 3), np.uint8), depth, 60.0, CAMERA_UP)
    assert client.calls and max(client.calls[0][0][:2]) < h, "the request is the speck's window, not the frame"
    assert any("bowl" in labels for labels in out.labels.values())


def test_a_speck_hidden_by_the_counter_mask_is_still_asked_and_replaces_it(box_depth):
    """The counter's mask covers the bowl. The tight crop is still asked, and the bowl takes those
    pixels back from the counter."""
    h, w = 120, 160
    depth, owner = box_depth((h, w), 60.0, SPECK_SCENE)
    mask = (owner >= 1).astype(np.uint8)
    cfg = SegmenterConfig(vocabulary={"bowl": "bowl", "counter": "counter"}, crop_prompts=("bowl",),
                          crop_target_px=480, crop_max_scale=4.0, min_pixels=60, crop_min_pixels=8)
    covered = Segmentation(mask, {1: {"counter": 0.9}})
    out = Segmenter(cfg, FakeClient(_bowl_in_any_crop(h))).zoom(
        covered, np.zeros((h, w, 3), np.uint8), depth, 60.0, CAMERA_UP)
    bowl = next(i for i, labels in out.labels.items() if "bowl" in labels)
    assert int(((out.mask == bowl) & (mask == 1)).sum()) >= 8
    assert int((out.mask == 1).sum()) > 0, "the counter keeps the pixels the bowl did not take"


def test_the_specks_of_two_cameras_are_asked_together(box_depth):
    """The enlarged crops of the images of one instant are one batch of requests: with two workers
    both are in flight at once, and each image gets the bowl its own crop returned."""
    import threading

    h, w = 120, 160
    depth, _owner = box_depth((h, w), 60.0, SPECK_SCENE)
    together = threading.Barrier(2, timeout=10.0)
    bowl = _bowl_in_any_crop(h)

    def reply(image, prompts):
        together.wait()  # broken (the test fails) unless the other image's request is waiting too
        return bowl(image, prompts)

    client = FakeClient(reply)
    cfg = SegmenterConfig(vocabulary={"bowl": "bowl"}, crop_prompts=("bowl",), crop_target_px=480,
                          crop_max_scale=4.0, min_pixels=60, crop_min_pixels=8, concurrency=2)
    blank = Segmentation(np.zeros((h, w), np.uint8), {})
    frames = [(blank, np.full((h, w, 3), shade, np.uint8), depth, 60.0, CAMERA_UP, None) for shade in (0, 40)]
    out = Segmenter(cfg, client).zoom_frames(frames)
    assert len(client.calls) == 2
    assert all(any("bowl" in labels for labels in seg.labels.values()) for seg in out)
    untouched = Segmenter(cfg, client).zoom_frames([(blank, frames[0][1], np.zeros((h, w), np.float32), 60.0, CAMERA_UP, None)])
    assert untouched[0] is blank and len(client.calls) == 2, "an image without a speck is not asked about"


def test_pistol_and_perfume_are_asked_only_inside_a_drawer_crop():
    shape = (120, 160)
    drawer = _box(shape, 70, 110, 40, 120)
    table = _box(shape, 10, 40, 20, 140)

    def reply(image, prompts):
        if image.shape[:2] == shape:
            return [("drawer", 0.9, drawer), ("table", 0.8, table)]
        return []

    cfg = SegmenterConfig(vocabulary={**VOCAB, "drawer": "drawer", "pistol": "pistol", "perfume": "perfume_bottle"},
                          crop_prompts=("bowl",), drawer_prompts=("pistol", "perfume"),
                          crop_categories=("drawer", "table"), crop_target_px=480, crop_max_scale=4.0)
    client = FakeClient(reply)
    Segmenter(cfg, client).segment(np.full((*shape, 3), 40, np.uint8))
    full = client.calls[0][1]
    assert "pistol" not in full and "perfume" not in full
    crops = [prompts for _shape, prompts in client.calls[1:]]
    assert any("pistol" in prompts and "perfume" in prompts for prompts in crops)
    assert any("pistol" not in prompts for prompts in crops)


def test_a_drawer_that_already_fills_the_frame_is_still_asked_about_its_contents():
    """Enlargement stops once the window is the target size. An open drawer is often already
    that large, and skipping it is how the pistol and the perfume stay unasked."""
    shape = (500, 640)
    drawer = _box(shape, 5, 495, 5, 635)

    def reply(image, prompts):
        if image.shape[:2] == shape:
            return [("drawer", 0.9, drawer)]
        return []

    cfg = SegmenterConfig(vocabulary={**VOCAB, "drawer": "drawer", "pistol": "pistol", "perfume": "perfume_bottle"},
                          crop_prompts=("bowl",), drawer_prompts=("pistol", "perfume"),
                          crop_categories=("drawer",), crop_target_px=480, crop_max_scale=4.0, max_crops=2)
    client = FakeClient(reply)
    Segmenter(cfg, client).segment(np.full((*shape, 3), 40, np.uint8))
    crops = [prompts for _shape, prompts in client.calls[1:]]
    assert crops and all("pistol" in prompts and "perfume" in prompts for prompts in crops)


def test_a_drawer_crop_includes_the_floor_and_a_table_crop_stays_at_the_top():
    """The perfume stands on the drawer floor. The tabletop crop cuts that floor off, so a drawer
    window keeps the opening and a table window still leaves the legs out."""
    shape = (120, 160)
    rgb = np.full((*shape, 3), 40, np.uint8)
    drawer = _box(shape, 40, 100, 40, 120)
    table = _box(shape, 10, 50, 10, 80)
    rgb[94:100, 70:90] = (255, 0, 0)
    rgb[44:50, 30:50] = (0, 0, 255)
    rgb[10:16, 30:50] = (0, 255, 0)
    seen: list[tuple[np.ndarray, list[str]]] = []

    def reply(image, prompts):
        seen.append((image.copy(), list(prompts)))
        if image.shape[:2] == shape:
            return [("drawer", 0.9, drawer), ("table", 0.8, table)]
        return []

    cfg = SegmenterConfig(vocabulary={**VOCAB, "drawer": "drawer", "pistol": "pistol", "perfume": "perfume_bottle"},
                          crop_prompts=("bowl",), drawer_prompts=("pistol", "perfume"),
                          crop_categories=("drawer", "table"), crop_target_px=480, crop_max_scale=4.0,
                          crop_depth=0.5, drawer_crop_depth=1.0, max_crops=2)
    Segmenter(cfg, FakeClient(reply)).segment(rgb)
    drawer_crops = [image for image, prompts in seen if "perfume" in prompts]
    table_crops = [image for image, prompts in seen if image.shape[:2] != shape and "perfume" not in prompts]
    assert drawer_crops and any(int(image[:, :, 0].max()) > 150 for image in drawer_crops)
    assert table_crops
    assert all(int(image[:, :, 2].max()) < 80 for image in table_crops)
    assert any(int(image[:, :, 1].max()) > 150 for image in table_crops)


def test_a_sideboard_crop_is_asked_about_the_pistol():
    """The pistol is asked on a sideboard window. The face is often named sideboard, not drawer,
    and the pistol is then never in the prompt list."""
    shape = (120, 160)
    sideboard = _box(shape, 40, 110, 20, 140)

    def reply(image, prompts):
        if image.shape[:2] == shape:
            return [("sideboard", 0.9, sideboard)]
        return []

    cfg = SegmenterConfig(vocabulary={**VOCAB, "sideboard": "sideboard", "pistol": "pistol",
                                      "perfume": "perfume_bottle"},
                          crop_prompts=("bowl",), drawer_prompts=("pistol", "perfume"),
                          crop_categories=("sideboard",), crop_target_px=480, crop_max_scale=4.0)
    client = FakeClient(reply)
    Segmenter(cfg, client).segment(np.full((*shape, 3), 40, np.uint8))
    crops = [prompts for _shape, prompts in client.calls[1:]]
    assert crops and all("pistol" in prompts and "perfume" in prompts for prompts in crops)


def test_a_cup_of_a_few_dozen_pixels_found_in_a_crop_is_kept():
    """Mapped back from the enlarged crop, a distant cup is about twenty pixels. The full-image
    floor of 25 pixels drops it; the crop floor keeps it. A speck of a few pixels is still dropped."""
    shape = (40, 50)
    cup = _box(shape, 10, 15, 10, 14)  # 5 * 4 = 20 pixels
    speck = _box(shape, 0, 2, 0, 3)  # 6 pixels
    cfg = SegmenterConfig(vocabulary={"cup": "cup"}, min_pixels=25, crop_min_pixels=8)
    segmenter = Segmenter(cfg, FakeClient(lambda image, prompts: []))
    kept, labels = segmenter._instances([("cup", 0.8, cup, True)], shape)
    assert any("cup" in v for v in labels.values()) and int((kept > 0).sum()) == 20
    _, dropped = segmenter._instances([("cup", 0.8, cup, False)], shape)
    assert dropped == {}, "the same mask from the full image stays under min_pixels"
    _, noise = segmenter._instances([("cup", 0.8, speck, True)], shape)
    assert noise == {}


def test_a_crop_that_names_the_cabinet_body_box_or_paints_the_whole_window_is_dropped():
    shape = (120, 160)
    cabinet = _box(shape, 40, 110, 30, 90)

    def reply(image, prompts):
        if image.shape[:2] == shape:
            return [("cabinet", 0.9, cabinet)]
        h, w = image.shape[:2]
        upper = _box((h, w), int(h * 0.45), int(h * 0.95), int(w * 0.05), int(w * 0.95))
        bowl = _box((h, w), int(h * 0.15), int(h * 0.35), int(w * 0.4), int(w * 0.55))
        filled = np.ones((h, w), dtype=bool)
        return [("box", 0.8, upper), ("bowl", 0.7, bowl), ("plate", 0.6, filled)]

    cfg = SegmenterConfig(vocabulary={**VOCAB, "cabinet": "cabinet", "box": "box"},
                          crop_prompts=("box", "bowl", "plate"), crop_categories=("cabinet",),
                          crop_target_px=480, crop_max_scale=4.0)
    seg = Segmenter(cfg, FakeClient(reply)).segment(np.zeros((*shape, 3), np.uint8))
    names = [sorted(v) for v in seg.labels.values()]
    assert ["bowl"] in names and ["cabinet"] in names
    assert not any("box" in v or "plate" in v for v in names)


def test_results_are_cached_by_image_and_settings(tmp_path):
    shape = (20, 20)
    client = FakeClient(lambda image, prompts: [("table", 0.9, _box(shape, 0, 20, 0, 20))])
    cfg = SegmenterConfig(vocabulary=VOCAB, crop_prompts=())
    rgb = np.zeros((*shape, 3), np.uint8)
    first = Segmenter(cfg, client, tmp_path).segment(rgb)
    again = Segmenter(cfg, client, tmp_path).segment(rgb)
    assert not first.cached and again.cached and len(client.calls) == 1
    assert again.labels == first.labels and np.array_equal(again.mask, first.mask)
    Segmenter(SegmenterConfig(vocabulary=VOCAB, crop_prompts=(), threshold=0.5), client, tmp_path).segment(rgb)
    assert len(client.calls) == 2, "other settings are another cache entry"


def test_the_prompts_of_a_pass_are_divided_over_the_service_workers_and_answered_the_same():
    shape = (60, 80)
    masks = {"table": _box(shape, 30, 60, 0, 80), "bowl": _box(shape, 25, 35, 30, 40),
             "plate": _box(shape, 25, 35, 30, 40), "wall": _box(shape, 0, 20, 0, 80)}
    scores = {"table": 0.9, "bowl": 0.7, "plate": 0.5, "wall": 0.8}

    def reply(image, prompts):
        return [(p, scores[p], masks[p]) for p in prompts]

    rgb = np.zeros((*shape, 3), np.uint8)
    one, four = FakeClient(reply), FakeClient(reply)
    whole = Segmenter(SegmenterConfig(vocabulary=VOCAB, crop_prompts=()), one).segment(rgb)
    split = Segmenter(SegmenterConfig(vocabulary=VOCAB, crop_prompts=(), concurrency=4), four).segment(rgb)
    assert len(one.calls) == 1 and sorted(len(prompts) for _shape, prompts in four.calls) == [1, 1, 1, 1]
    assert sorted(p for _shape, prompts in four.calls for p in prompts) == sorted(VOCAB)
    assert split.labels == whole.labels and np.array_equal(split.mask, whole.mask)

    both = FakeClient(reply)
    pair = Segmenter(SegmenterConfig(vocabulary=VOCAB, crop_prompts=(), concurrency=4), both).segment_frames(
        [(rgb, None), (np.ones((*shape, 3), np.uint8), None)])
    assert sorted(len(prompts) for _shape, prompts in both.calls) == [2, 2, 2, 2], "two images share the four workers"
    assert [seg.labels for seg in pair] == [whole.labels, whole.labels]


def test_an_enlarged_crop_takes_its_pixels_from_the_frame_the_image_was_reduced_from():
    shape = (60, 80)
    table = _box(shape, 40, 60, 10, 70)
    detail = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    tiles = []

    def reply(image, prompts):
        if image.shape[:2] == shape:
            return [("table", 0.9, table)]
        tiles.append(image)
        return []

    cfg = SegmenterConfig(vocabulary=VOCAB, crop_prompts=("bowl",), crop_target_px=4000, crop_max_scale=4.0)
    Segmenter(cfg, FakeClient(reply)).segment(np.zeros((*shape, 3), np.uint8), detail)
    assert len(tiles) == 1
    # The window is rows 34..50 and columns 6..74 of the image; enlarged four times, it is that
    # window of the frame four times its size, pixel for pixel.
    assert np.array_equal(tiles[0], detail[34 * 4:50 * 4, 6 * 4:74 * 4])


def test_a_service_failure_is_not_cached(tmp_path):
    def reply(image, prompts):
        raise SegmenterError("sam3: no reachable URL")

    with pytest.raises(SegmenterError):
        Segmenter(SegmenterConfig(vocabulary=VOCAB), FakeClient(reply), tmp_path).segment(np.zeros((8, 8, 3), np.uint8))
    assert not any(tmp_path.rglob("*.npz"))


def test_depth_clusters_sam_missed_become_unnamed_instances_and_found_ones_are_not_repeated():
    shape = (120, 160)
    depth = np.full(shape, 2.0, np.float32)
    depth[30:50, 40:60] = 1.2  # found by SAM
    depth[30:50, 100:120] = 1.3  # missed by SAM
    sam = np.zeros(shape, np.uint8)
    sam[30:50, 40:60] = 1
    seg = Segmentation(sam, {1: {"bowl": 0.8}})
    up = np.array([0.0, 0.0, 1.0])
    filled = Segmenter(SegmenterConfig(vocabulary=VOCAB, min_pixels=20, geometric_fill=True,
                                       fill_supports_only=False), None).fill(seg, depth, 58.0, up)
    assert filled.labels == seg.labels, "filled instances carry no names"
    assert (filled.mask[30:50, 40:60] == 1).all(), "what SAM found keeps its instance"
    missed = filled.mask[30:50, 100:120]
    assert missed.min() > 1 and len(np.unique(missed)) == 1
    off = Segmenter(SegmenterConfig(vocabulary=VOCAB), None).fill(seg, depth, 58.0, up)
    assert off is seg


def test_a_bowl_sam_did_not_name_is_filled_from_the_table_mask_that_covers_it(box_depth):
    """SAM's table mask covers the bowl standing on the table. The bowl's depth stands clear of the
    top, so it becomes an unnamed instance; the bare top, a stool on the floor and the wall do not."""
    floor, wall = ((-1.0, -4.0, -0.1), (8.0, 4.0, 0.0)), ((4.0, -4.0, 0.0), (4.1, 4.0, 2.6))
    table, stool = ((1.6, -0.6, 0.0), (2.4, 0.6, 0.75)), ((1.7, 1.2, 0.0), (2.1, 1.6, 0.45))
    bowl, mat = ((1.9, -0.1, 0.75), (2.1, 0.1, 0.83)), ((1.8, 0.25, 0.75), (2.1, 0.5, 0.76))
    depth, owner = box_depth((240, 320), 60.0, [floor, wall, table, stool, bowl, mat])
    sam = np.where(np.isin(owner, (2, 4, 5)), 1, 0).astype(np.uint8)
    seg = Segmentation(sam, {1: {"table": 0.9}})
    filled = Segmenter(SegmenterConfig(vocabulary=VOCAB, geometric_fill=True), None).fill(seg, depth, 60.0, CAMERA_UP)
    assert filled.labels == seg.labels
    added = [int(i) for i in np.unique(filled.mask) if i > 1]
    assert len(added) == 1, "only the bowl: the stool and the wall stand on no raised surface"
    # The bowl's lowest two centimetres are within the tolerance of the table's plane and stay table.
    assert (filled.mask[owner == 4] == added[0]).mean() > 0.7
    assert (filled.mask[owner == 5] == 1).all(), "a centimetre of table mat is still the table"
    bare, bare_owner = box_depth((240, 320), 60.0, [floor, wall, table, stool])
    seg = Segmentation(np.where(bare_owner == 2, 1, 0).astype(np.uint8), {1: {"table": 0.9}})
    assert not (Segmenter(SegmenterConfig(vocabulary=VOCAB, geometric_fill=True), None).fill(seg, bare, 60.0, CAMERA_UP).mask > 1).any()


def test_semantic_scores_lead_and_shape_only_breaks_ties():
    shape = {"bowl": 0.6, "mug": 0.3, "unknown": 0.1}
    fused = fuse_semantics({"mug": 0.8}, shape, weight=0.5, floor=0.03)
    assert max(fused, key=fused.get) == "mug"
    tie = fuse_semantics({"bowl": 0.6, "mug": 0.6}, shape, weight=0.5, floor=0.03)
    assert tie["bowl"] > tie["mug"]
    novel = fuse_semantics({"kettle": 0.7}, shape, weight=0.5, floor=0.03)
    assert max(novel, key=novel.get) == "kettle", "a category without a shape template is named by SAM"
    misfit = fuse_semantics({"teddy_bear": 0.7}, {**shape, "teddy_bear": 0.002}, weight=0.5, floor=0.03)
    assert max(misfit, key=misfit.get) == "teddy_bear", "a confident name outweighs a template that fits badly"
    vetoed = fuse_semantics({"bowl": 0.5}, {"bowl": 1e-5, "counter": 0.6, "unknown": 0.1}, weight=0.5, floor=0.03)
    assert max(vetoed, key=vetoed.get) == "counter", "a template that cannot fit (a two-metre bowl) overrules the name"


def test_the_shipped_settings_load_and_bad_settings_fail_loud(tmp_path):
    cfg = load_config()
    assert cfg.service == "sam3" and set(cfg.crop_prompts) <= set(cfg.vocabulary)
    bad = tmp_path / "s.toml"
    for text, message in [("[vocabulary]\nbowl='bowl'\n", "missing [segmenter]"),
                          ("[segmenter]\nthreshold='x'\n[vocabulary]\nbowl='bowl'\n", "segmenter.threshold"),
                          ("[segmenter]\ntresh=0.4\n[vocabulary]\nbowl='bowl'\n", "unknown segmenter keys"),
                          ("[segmenter]\ncrop_prompts=['cup']\n[vocabulary]\nbowl='bowl'\n", "not in [vocabulary]")]:
        bad.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=message.replace("[", r"\[")):
            load_config(bad)


class StubSegmenter:
    """Names the largest regions of one flat colour a "thing" (a category without a shape
    template); ``fail`` makes it raise instead."""

    def __init__(self):
        self.config = SegmenterConfig(vocabulary=VOCAB)
        self.calls = 0
        self.fail = False

    def segment_frames(self, frames):
        return [self.segment(rgb) for rgb, _detail in frames]

    def segment(self, rgb):
        self.calls += 1
        if self.fail:
            raise SegmenterError("sam3: no reachable URL")
        q = rgb.astype(np.int32) // 16
        key = (q[..., 0] << 8) | (q[..., 1] << 4) | q[..., 2]
        values, counts = np.unique(key, return_counts=True)
        mask = np.zeros(rgb.shape[:2], np.uint8)
        for i, value in enumerate(values[np.argsort(-counts)][:8], start=1):
            mask[key == value] = i
        return Segmentation(mask, {i: {"thing": 0.9} for i in range(1, int(mask.max()) + 1)}, seconds=0.01)

    def clusters(self, depth, fovy_deg, up_cam):
        return None

    def fill(self, seg, depth, fovy_deg, up_cam, clusters=None):
        return seg

    def zoom_frames(self, frames, clusters=None):
        return [frame[0] for frame in frames]

    def peel(self, seg, rgb, depth, fovy_deg=None):
        return seg


def _pump_until(host, done, timeout=120.0):
    deadline = time.time() + timeout
    while not done() and time.time() < deadline:
        host.pump(0.05)
    assert done(), "timed out"


def test_without_world_masks_the_robot_segments_keyframes_and_alerts_on_failure(kitchen_root, tmp_path: Path):
    host = StewardHost(kitchen_root, scenario="phase0-chores", speed=4.0, seed=3, run_dir=tmp_path).start()
    stub = StubSegmenter()
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict(), segmenter=stub)
    runtime.start()
    try:
        host.set_phase("phase1-sam")
        assert "mask" not in host.sim.sensors("w:robot-1") and host.sim.sensors("w:robot-1")["masks"] is False
        # A keyframe batch is segmented camera by camera before it is fused, so wait for the fusion.
        _pump_until(host, lambda: runtime.perception_mode == "rgbd-segmenter" and stub.calls >= 2
                    and any(t.semantic_views for t in list(runtime.estimator.tracks.values())))
        assert runtime.segmentation["frames"] == stub.calls
        assert "segmentation" in runtime.summary()
        preview = runtime.segmentation_preview.latest("head_camera")
        assert preview is not None and preview["image"].startswith("data:image/jpeg;base64,")
        stub.fail = True
        _pump_until(host, lambda: runtime.segmentation["failures"] >= 2)
        assert sum("分割失败" in a["message"] for a in runtime.alerts) == 1, "one alert per run of failures"
    finally:
        runtime.stop()
        host.stop()


def test_a_pink_bottle_farther_than_the_drawer_front_becomes_its_own_instance():
    """SAM names the open drawer and the bottle in it as one drawer. The bottle is farther than
    the front panel and a different colour, so those pixels leave the drawer. The handle, on the
    panel, stays."""
    shape = (80, 100)
    rgb = np.full((*shape, 3), 180, np.uint8)
    rgb[:, :, 1] = 150
    rgb[:, :, 2] = 110
    depth = np.full(shape, 1.0, np.float32)
    drawer = _box(shape, 30, 70, 20, 80)
    bottle = _box(shape, 34, 50, 40, 48)
    handle = _box(shape, 58, 64, 44, 56)
    rgb[bottle] = (230, 150, 190)
    rgb[handle] = (140, 140, 145)
    depth[bottle] = 1.15
    mask = np.zeros(shape, dtype=np.uint8)
    mask[drawer] = 1
    seg = Segmentation(mask, {1: {"drawer": 0.9}})
    peeled = Segmenter(SegmenterConfig(), None).peel(seg, rgb, depth)
    assert set(np.unique(peeled.mask[bottle])) == {2}
    assert int(peeled.mask[handle].max()) == 1
    assert 2 not in peeled.labels
