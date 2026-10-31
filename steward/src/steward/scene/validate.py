"""Validators for the JSON form of SceneState.

``validate_world`` enforces the World State rules (exact values, ``w:`` ids, complete entities);
``validate_ego`` enforces the Ego-State rules (``e:`` ids, no asset references, no World State
identifiers anywhere). Both return a list of human-readable errors; an empty list means valid.
A ``w:`` string inside an Ego-State is reported as a leak because it can only originate from
the World State.
"""

from __future__ import annotations

import base64
import binascii
import math
from typing import Any

from steward.scene.schema import (
    EGO_ID_PREFIX,
    GEOMETRY_KINDS,
    GEOMETRY_ROLES,
    JOINT_TYPES,
    RELATION_TYPES,
    ROLES,
    ROOT_TYPES,
    SHAPES,
    SOURCES,
    UNCERTAINTY_KINDS,
    WORLD_ID_PREFIX,
)

_SHAPE_SIZE_LEN = {"box": 3, "sphere": 1, "capsule": 2, "cylinder": 2, "plane": 2, "ellipsoid": 3}
_ESTIMATE_META = ("unc", "t", "src")


class SceneValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors[:10]) + (f" (+{len(errors) - 10} more)" if len(errors) > 10 else ""))
        self.errors = errors


def validate_world(scene: dict) -> list[str]:
    errors = _validate_common(scene, "world")
    if scene.get("self") is not None:
        errors.append("world state must not set self")
    for path, est in _walk_estimates(scene):
        for key in _ESTIMATE_META:
            if key in est:
                errors.append(f"{path}: world state estimate must not carry {key!r}")
    for i, ent in enumerate(scene.get("entities", [])):
        if ent.get("existence", 1.0) != 1.0:
            errors.append(f"entities[{i}]: world state existence must be 1")
        for j, link in enumerate(ent.get("body", {}).get("links", [])):
            if link.get("geometry", {}).get("observed"):
                errors.append(f"entities[{i}].links[{j}]: world state must not carry observed geometry")
    return errors


def validate_ego(scene: dict) -> list[str]:
    errors = _validate_common(scene, "ego")
    ids = {e.get("id") for e in scene.get("entities", [])}
    if scene.get("self") is not None and scene["self"] not in ids:
        errors.append(f"self {scene['self']!r} does not name an entity")
    for i, ent in enumerate(scene.get("entities", [])):
        if "asset" in ent:
            errors.append(f"entities[{i}]: ego state must not reference assets")
        ex = ent.get("existence", 1.0)
        if not (0.0 <= ex <= 1.0):
            errors.append(f"entities[{i}]: existence must be within [0, 1]")
    for path, value in _walk_strings(scene):
        if value.startswith(WORLD_ID_PREFIX):
            errors.append(f"{path}: world identifier {value!r} leaked into ego state")
    for path, est in _walk_estimates(scene):
        unc = est.get("unc")
        if unc is not None and unc.get("kind") not in UNCERTAINTY_KINDS:
            errors.append(f"{path}.unc: unknown uncertainty kind {unc.get('kind')!r}")
        src = est.get("src")
        if src is not None and src not in SOURCES:
            errors.append(f"{path}.src: unknown source {src!r}")
    return errors


def require_valid(scene: dict) -> None:
    errors = validate_world(scene) if scene.get("mode") == "world" else validate_ego(scene)
    if errors:
        raise SceneValidationError(errors)


def _validate_common(scene: dict, mode: str) -> list[str]:
    errors: list[str] = []
    if scene.get("mode") != mode:
        errors.append(f"mode must be {mode!r}")
    expected_frame = "world" if mode == "world" else "map"
    if scene.get("frame", expected_frame) != expected_frame:
        errors.append(f"frame must be {expected_frame!r}")
    prefix = WORLD_ID_PREFIX if mode == "world" else EGO_ID_PREFIX
    seen: set[str] = set()
    for i, ent in enumerate(scene.get("entities", [])):
        where = f"entities[{i}]"
        eid = ent.get("id")
        if not isinstance(eid, str) or not eid.startswith(prefix):
            errors.append(f"{where}: id must be a string starting with {prefix!r}")
        elif eid in seen:
            errors.append(f"{where}: duplicate id {eid!r}")
        else:
            seen.add(eid)
        if ent.get("role") not in ROLES:
            errors.append(f"{where}: unknown role {ent.get('role')!r}")
        errors += _check_est(ent.get("category"), f"{where}.category", lambda v: isinstance(v, str))
        errors += _check_est(ent.get("pose"), f"{where}.pose", _is_pose)
        errors += _validate_body(ent.get("body"), where)
        for key, var in ent.get("stateVars", {}).items():
            errors += _check_est(var, f"{where}.stateVars.{key}", lambda v: True)
    for i, rel in enumerate(scene.get("relations", [])):
        where = f"relations[{i}]"
        if rel.get("type") not in RELATION_TYPES:
            errors.append(f"{where}: unknown relation type {rel.get('type')!r}")
        for side in ("a", "b"):
            anchor = rel.get(side) or {}
            if anchor.get("entity") not in seen:
                errors.append(f"{where}.{side}: unknown entity {anchor.get('entity')!r}")
        if "derived" not in rel:
            errors.append(f"{where}: derived is required")
    return errors


def _validate_body(body: Any, where: str) -> list[str]:
    if not isinstance(body, dict):
        return [f"{where}.body: missing"]
    if body.get("kind") == "deformable":
        errors = []
        if body.get("model") not in ("cloth", "volume", "cable"):
            errors.append(f"{where}.body: unknown deformable model {body.get('model')!r}")
        errors += _check_est(body.get("keypoints"), f"{where}.body.keypoints", lambda v: isinstance(v, list))
        errors += _check_est(body.get("material"), f"{where}.body.material", lambda v: isinstance(v, dict))
        return errors
    if body.get("kind") != "rigid":
        return [f"{where}.body: unknown kind {body.get('kind')!r}"]
    errors = _check_est(body.get("root"), f"{where}.body.root", lambda v: v in ROOT_TYPES)
    links = body.get("links") or []
    if not links:
        return errors + [f"{where}.body: a rigid tree needs at least one link"]
    link_ids: list[str] = []
    for j, link in enumerate(links):
        lw = f"{where}.links[{j}]"
        if not isinstance(link.get("id"), str):
            errors.append(f"{lw}: id must be a string")
            continue
        if link["id"] in link_ids:
            errors.append(f"{lw}: duplicate link id {link['id']!r}")
        link_ids.append(link["id"])
        for role, geoms in link.get("geometry", {}).items():
            if role not in GEOMETRY_ROLES:
                errors.append(f"{lw}.geometry: unknown role {role!r}")
                continue
            for k, g in enumerate(geoms):
                errors += _check_geometry(g, f"{lw}.geometry.{role}[{k}]")
    parent_of: dict[str, str] = {}
    for j, joint in enumerate(body.get("joints", [])):
        jw = f"{where}.joints[{j}]"
        errors += _check_est(joint.get("type"), f"{jw}.type", lambda v: v in JOINT_TYPES)
        errors += _check_est(joint.get("origin"), f"{jw}.origin", _is_pose)
        errors += _check_est(joint.get("q"), f"{jw}.q", lambda v: isinstance(v, list))
        errors += _check_est(joint.get("qd"), f"{jw}.qd", lambda v: isinstance(v, list))
        if joint.get("parent") not in link_ids:
            errors.append(f"{jw}: unknown parent link {joint.get('parent')!r}")
        child = joint.get("child")
        if child not in link_ids:
            errors.append(f"{jw}: unknown child link {child!r}")
        elif child == link_ids[0]:
            errors.append(f"{jw}: the root link cannot have a parent joint")
        elif child in parent_of:
            errors.append(f"{jw}: link {child!r} already has a parent joint")
        else:
            parent_of[child] = joint.get("parent")
        limits = (joint.get("limits") or {}).get("value")
        if limits is not None and not (limits.get("lo", 0) <= limits.get("hi", 0)):
            errors.append(f"{jw}.limits: lo must not exceed hi")
    for link_id in link_ids[1:]:
        if link_id not in parent_of:
            errors.append(f"{where}: link {link_id!r} is not connected by a joint")
    for link_id in parent_of:
        seen_chain = {link_id}
        cur = link_id
        while cur in parent_of:
            cur = parent_of[cur]
            if cur in seen_chain:
                errors.append(f"{where}: joint cycle through {link_id!r}")
                break
            seen_chain.add(cur)
    return errors


def _check_geometry(g: Any, where: str) -> list[str]:
    if not isinstance(g, dict):
        return [f"{where}: geometry must be an object"]
    kind = g.get("kind")
    if kind not in GEOMETRY_KINDS:
        return [f"{where}: unknown geometry kind {kind!r}"]
    errors = []
    if "pose" in g and not _is_pose(g["pose"]):
        errors.append(f"{where}.pose: invalid pose")
    if kind == "primitive":
        shape = g.get("shape")
        if shape not in SHAPES:
            errors.append(f"{where}: unknown shape {shape!r}")
        else:
            size = g.get("size")
            if not isinstance(size, list) or len(size) != _SHAPE_SIZE_LEN[shape] or any(
                not isinstance(s, (int, float)) or s <= 0 for s in size
            ):
                errors.append(f"{where}: {shape} needs {_SHAPE_SIZE_LEN[shape]} positive size values")
        if "texture" in g and (shape != "box" or not _is_png_b64(g["texture"])):
            errors.append(f"{where}: a texture is a base64 PNG on a box")
    elif not isinstance(g.get("ref"), str):
        errors.append(f"{where}: {kind} geometry needs a ref")
    return errors


def _is_png_b64(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    try:
        return base64.b64decode(v[:24], validate=True).startswith(b"\x89PNG")
    except (binascii.Error, ValueError):  # not base64: rejected as an invalid texture
        return False


def _check_est(est: Any, where: str, value_ok) -> list[str]:
    if not isinstance(est, dict) or "value" not in est:
        return [f"{where}: missing estimate value"]
    if not value_ok(est["value"]):
        return [f"{where}: invalid value {est['value']!r}"]
    return []


def _is_pose(v: Any) -> bool:
    if not isinstance(v, dict):
        return False
    p, q = v.get("p"), v.get("q")
    if not (isinstance(p, list) and len(p) == 3 and isinstance(q, list) and len(q) == 4):
        return False
    norm = math.sqrt(sum(float(x) * float(x) for x in q))
    return abs(norm - 1.0) < 1e-3


def _walk_estimates(node: Any, path: str = "$"):
    if isinstance(node, dict):
        if "value" in node and set(node) <= {"value", *_ESTIMATE_META}:
            yield path, node
            return
        for k, v in node.items():
            yield from _walk_estimates(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_estimates(v, f"{path}[{i}]")


def _walk_strings(node: Any, path: str = "$"):
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _walk_strings(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_strings(v, f"{path}[{i}]")
