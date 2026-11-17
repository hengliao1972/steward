"""Import OpenUSD through MuJoCo's optional native decoder into Steward assets.

Import requires the ``usd`` extra. The resulting template and binary meshes also load in the
regular MuJoCo runtime without OpenUSD. Unsupported dynamics fail before the output directory
is published. Source files are never edited; normalization uses a flattened temporary stage.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
from collections import defaultdict, deque

import mujoco
import numpy as np

from steward.scene.schema import Entity, Joint, Link, RigidTree, SceneState, exact, pose, primitive
from steward.scene.mjcf import build_mjcf
from steward.scene.extent import entity_obb
from steward.scene.validate import require_valid


class UsdImportError(ValueError):
    """An input or decoder failure; no completed asset directory is published."""


def import_usd(source: Path, output: Path, *, meters_per_unit: float | None = None,
               orient_joints: bool = False, fixed_base: bool = False,
               bound_mass: float = 0.0, bound_inertia: float = 0.0,
               inertia_from_bounds: bool = False, drive_frequency: float | None = None) -> dict:
    """Publish a template, meshes and report in a new directory and return the report.

    Unit metadata and joint directions are preserved unless explicitly overridden. ``fixed_base``
    removes top-level free joints. Positive mass/inertia bounds permit incomplete source physics;
    every applied bound is reported. ``inertia_from_bounds`` estimates missing inertias from uniform
    boxes; ``drive_frequency`` replaces gains with critically damped servos at the rest pose.
    Existing output directories are refused. Mesh references use the absolute output directory.
    """
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
        from importlib.metadata import version
        decoder_version = version("mujoco-usd")
        if decoder_version != mujoco.__version__:
            raise UsdImportError(f"mujoco-usd {decoder_version} does not match MuJoCo {mujoco.__version__}; install the usd extra in a clean environment")
    except ImportError as exc:
        raise UsdImportError("USD import requires the usd extra: pip install -e './steward[usd]'") from exc
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
        raise UsdImportError(f"USD source must be an existing .usd/.usda/.usdc/.usdz file: {source}")
    for key, value in (("meters_per_unit", meters_per_unit), ("bound_mass", bound_mass), ("bound_inertia", bound_inertia), ("drive_frequency", drive_frequency)):
        if value is not None and (not math.isfinite(value) or value < 0 or (key in ("meters_per_unit", "drive_frequency") and value == 0)):
            raise UsdImportError(f"{key} must be finite and {'positive' if key in ('meters_per_unit', 'drive_frequency') else 'nonnegative'}")
    if output.exists():
        raise UsdImportError(f"output already exists: {output}; choose a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"source": str(source), "sha256": digest,
              "mujoco": mujoco.__version__, "adjustments": [], "limitations": [
                  "Imported bodies require their own controller; the built-in Cobot controller is not assigned.",
                  "Materials retain decoder diffuse colors; USD shader networks, textures and cameras are not transferred."]}
    try:
        stage = Usd.Stage.Open(str(source))
        if not stage:
            raise UsdImportError(f"cannot open USD stage: {source}")
        # Flatten resolves USD composition; unresolved dependencies cannot yield a complete body.
        from pxr import UsdUtils
        _, _, unresolved = UsdUtils.ComputeAllDependencies(str(source))
        missing_geometry = [p for p in unresolved if Path(p).suffix.lower() != ".mdl"]
        if missing_geometry:
            raise UsdImportError(f"unresolved USD dependencies: {missing_geometry}")
        if unresolved:
            report["limitations"].append("Unavailable MDL shaders omitted: " + ", ".join(unresolved))
        stage.SetEditTarget(stage.GetSessionLayer())
        report["sourceMetersPerUnit"] = UsdGeom.GetStageMetersPerUnit(stage)
        if meters_per_unit is not None:
            UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
            report["adjustments"].append(f"metersPerUnit = {meters_per_unit}")
        if orient_joints:
            report["adjustments"] += _orient_joints(stage)
        with tempfile.TemporaryDirectory(prefix=".usd-import-", dir=output.parent) as scratch:
            work = Path(scratch)
            normalized = work / "source.usdc"
            stage.Flatten().Export(str(normalized))
            spec = mujoco.MjSpec.from_file(str(normalized))
            for body in list(spec.worldbody.bodies):
                prim = stage.GetPrimAtPath(body.name)
                if prim and not prim.HasAPI(UsdPhysics.RigidBodyAPI) and not list(body.bodies) and not list(body.geoms):
                    report["adjustments"].append(f"removed empty nonphysical container: {body.name}")
                    spec.delete(body)
            if fixed_base:
                for body in spec.worldbody.bodies:
                    for joint in list(body.joints):
                        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                            spec.delete(joint)
                            report["adjustments"].append(f"fixed root: {body.name}")
            _convert_units(spec, UsdGeom.GetStageMetersPerUnit(stage), UsdPhysics.GetStageKilogramsPerUnit(stage))
            spec.compiler.boundmass = bound_mass
            spec.compiler.boundinertia = bound_inertia
            if bound_mass or bound_inertia:
                report["adjustments"].append(f"compiler bounds: mass={bound_mass} kg, inertia={bound_inertia} kg m^2")
            model = spec.compile()
            template = _project(model, stage, work, report, inertia_from_bounds)
            require_valid(SceneState(mode="world", entities=[template]).to_dict())
            build = build_mjcf(SceneState(mode="world", entities=[template]))
            imported = build.compile()
            if drive_frequency is not None:
                for joint in template.rigid.joints:
                    if joint.actuator is None:
                        continue
                    jid = imported.joint(build.names[template.id].joints[joint.id]).id
                    effective_mass = float(imported.dof_M0[imported.jnt_dofadr[jid]])
                    omega = 2 * math.pi * drive_frequency
                    joint.actuator["gains"] = {"kp": effective_mass * omega ** 2, "kv": 2 * effective_mass * omega}
                report["adjustments"].append(f"critically damped position drives: {drive_frequency} Hz at rest pose")
                build = build_mjcf(SceneState(mode="world", entities=[template]))
                imported = build.compile()
            data = mujoco.MjData(imported)
            build.apply_initial(imported, data)
            bounds = entity_obb(imported, data, build.names[template.id])
            ground_shift = -(bounds.center[2] - bounds.size[2] / 2)
            template.pose.value["p"][2] += ground_shift
            report["adjustments"].append(f"floor placement: vertical shift {ground_shift:.9g} m")
            report.update(links=len(template.rigid.links), joints=len(template.rigid.joints),
                          meshes=model.nmesh, actuators=imported.nu, couplings=imported.neq, sizeMeters=bounds.size,
                          template=str(output / "template.json"))
            for link in template.rigid.links:
                for geoms in link.geometry.values():
                    for geom in geoms:
                        if geom["kind"] == "mesh":
                            geom["ref"] = str(output / Path(geom["ref"]).name)
            (work / "template.json").write_text(json.dumps(template.to_dict(), indent=2) + "\n")
            (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            normalized.unlink()
            work.rename(output)
        return report
    except UsdImportError:
        raise
    except Exception as exc:  # The USD/native compiler exception is retained as the import failure's cause.
        raise UsdImportError(f"USD import failed for {source}: {exc}") from exc


def _orient_joints(stage) -> list[str]:
    """Orient a rooted acyclic USD joint graph; reversed coordinates negate limits and drives."""
    from pxr import UsdPhysics
    graph = defaultdict(list)
    roots = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(a) > 1 or len(b) > 1:
            raise UsdImportError(f"joint must reference one body on each side: {prim.GetPath()}")
        if not a:
            roots.extend(b)
        elif not b:
            raise UsdImportError(f"world anchor must be body0: {prim.GetPath()}")
        else:
            graph[a[0]].append((b[0], joint))
            graph[b[0]].append((a[0], joint))
    if not roots:
        raise UsdImportError("--orient-joints requires a world-anchored joint tree")
    visited, edges, todo, changes = set(roots), set(), deque(roots), []
    while todo:
        parent = todo.popleft()
        for child, joint in graph[parent]:
            path = joint.GetPath()
            if path in edges:
                continue
            edges.add(path)
            if child in visited:
                raise UsdImportError(f"closed joint loop is unsupported: {path}")
            visited.add(child)
            todo.append(child)
            if joint.GetBody0Rel().GetTargets() == [parent]:
                continue
            if any("physxMimicJoint" in rel.GetName() and (prim == joint.GetPrim() or path in rel.GetTargets())
                   for prim in stage.Traverse() for rel in prim.GetRelationships()):
                raise UsdImportError(f"cannot reverse a joint participating in a mimic constraint: {path}")
            joint.GetBody0Rel().SetTargets([parent])
            joint.GetBody1Rel().SetTargets([child])
            for kind in ("Pos", "Rot"):
                a, b = getattr(joint, f"GetLocal{kind}0Attr")(), getattr(joint, f"GetLocal{kind}1Attr")()
                av, bv = a.Get(), b.Get()
                a.Set(bv)
                b.Set(av)
            prim = joint.GetPrim()
            lo, hi = prim.GetAttribute("physics:lowerLimit"), prim.GetAttribute("physics:upperLimit")
            if lo and hi:
                lv, hv = lo.Get(), hi.Get()
                lo.Set(-hv)
                hi.Set(-lv)
            for attr in prim.GetAttributes():
                if attr.GetName().endswith((":targetPosition", ":targetVelocity", ":position", ":velocity")) and attr.HasAuthoredValueOpinion():
                    attr.Set(-attr.Get())
            changes.append(f"reversed joint coordinate: {path}")
    if set(graph) - visited:
        raise UsdImportError("joint graph contains an unanchored component")
    return changes


def _project(model, stage, output: Path, report: dict, inertia_from_bounds: bool) -> Entity:
    """Project the native decoder's rigid tree; reject features the scene schema cannot retain."""
    if model.ntendon or model.neq or model.nflex or model.nu:
        raise UsdImportError("USD model contains tendons, equalities, flexes or native actuators not supported by this importer")
    if sum(model.body_parentid[1:] == 0) != 1:
        raise UsdImportError("USD must contain exactly one robot root")
    root_type = "fixed"
    if model.body_jntnum[1]:
        if model.body_jntnum[1] != 1 or model.jnt_type[model.body_jntadr[1]] != mujoco.mjtJoint.mjJNT_FREE:
            raise UsdImportError("robot root supports only a fixed or free joint")
        root_type = "free"
    meshes = {}
    for mid in range(model.nmesh):
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        payload = (struct.pack("<4i", int(vn), 0, 0, int(fn))
                   + model.mesh_vert[va:va + vn].astype("<f4").tobytes()
                   + model.mesh_face[fa:fa + fn].astype("<i4").tobytes())
        path = output / (hashlib.sha256(payload).hexdigest() + ".msh")
        path.write_bytes(payload)
        meshes[mid] = str(path)
    ids = {bid: model.body(bid).name for bid in range(1, model.nbody)}
    links, joints = [], []
    for bid, name in ids.items():
        inertia = {"mass": float(model.body_mass[bid]), "diagonal": model.body_inertia[bid].tolist(),
                   "pose": pose(model.body_ipos[bid], model.body_iquat[bid].tolist())}
        link = Link(name, name.rsplit("/", 1)[-1], inertial=exact(inertia) if inertia["mass"] else None)
        for gid in range(model.body_geomadr[bid], model.body_geomadr[bid] + model.body_geomnum[bid]):
            geom = _geometry(model, gid, meshes)
            role = "collision" if model.geom_contype[gid] or model.geom_conaffinity[gid] else "visual"
            link.geometry.setdefault(role, []).append(geom)
        if inertia_from_bounds and link.geometry:
            from pxr import UsdPhysics
            prim = stage.GetPrimAtPath(name)
            source_inertia = UsdPhysics.MassAPI(prim).GetDiagonalInertiaAttr().Get() if prim else None
            if source_inertia is None or not all(float(v) > 0 for v in source_inertia):
                _estimate_inertia(model, bid, link)
                report["adjustments"].append(f"uniform bounding-box inertia estimate: {name}")
        links.append(link)
        if bid == 1:
            continue
        if model.body_jntnum[bid] > 1:
            raise UsdImportError(f"multiple joints on one link are unsupported: {name}")
        parent = ids[int(model.body_parentid[bid])]
        origin = exact(pose(model.body_pos[bid], model.body_quat[bid].tolist()))
        if not model.body_jntnum[bid]:
            joints.append(Joint(f"fixed:{name}", parent, name, exact("fixed"), origin, exact([]), exact([])))
            continue
        jid = int(model.body_jntadr[bid])
        kind = {int(mujoco.mjtJoint.mjJNT_HINGE): "revolute", int(mujoco.mjtJoint.mjJNT_SLIDE): "prismatic"}.get(int(model.jnt_type[jid]))
        if kind is None:
            raise UsdImportError(f"only revolute and prismatic internal joints are supported: {name}")
        jname = model.joint(jid).name
        limits = exact(dict(zip(("lo", "hi"), model.jnt_range[jid].tolist()))) if model.jnt_limited[jid] else None
        q0 = float(model.qpos0[model.jnt_qposadr[jid]])
        if q0 != 0:
            raise UsdImportError(f"nonzero joint reference is unsupported: {jname}")
        joints.append(Joint(jname, parent, name, exact(kind), origin, exact([q0]), exact([0.0]),
                            axis=exact(model.jnt_axis[jid].tolist()), limits=limits,
                            anchor=exact(model.jnt_pos[jid].tolist()),
                            actuator=_drive(stage, jname, kind),
                            dynamics=exact({"damping": float(model.dof_damping[model.jnt_dofadr[jid]])})))
    couplings = _couplings(stage, joints)
    return Entity(id="w:template", role="robot", category=exact("usd-robot"),
                  pose=exact(pose(model.body_pos[1], model.body_quat[1].tolist())),
                  body=RigidTree(exact(root_type), links, joints, couplings=couplings))


def _drive(stage, joint_name: str, kind: str) -> dict | None:
    """Map authored force drives to position servos; hold the imported rest pose at startup."""
    from pxr import UsdPhysics, UsdGeom
    prim = stage.GetPrimAtPath(joint_name)
    if not prim:
        return None
    drive = UsdPhysics.DriveAPI.Get(prim, "angular" if kind == "revolute" else "linear")
    if not drive:
        return None
    if str(drive.GetTypeAttr().Get()) != "force":
        raise UsdImportError(f"acceleration drive is unsupported: {joint_name}")
    kp, kv = float(drive.GetStiffnessAttr().Get()), float(drive.GetDampingAttr().Get())
    if kp <= 0:
        raise UsdImportError(f"drive requires positive stiffness: {joint_name}")
    mass = UsdPhysics.GetStageKilogramsPerUnit(stage)
    length = UsdGeom.GetStageMetersPerUnit(stage)
    gain_scale = mass * length ** 2 * 180 / math.pi if kind == "revolute" else mass
    kp, kv = kp * gain_scale, kv * gain_scale
    force_scale = mass * length ** 2 if kind == "revolute" else mass * length
    force = float(drive.GetMaxForceAttr().Get()) * force_scale
    return {"kind": "position", "gains": {"kp": kp, "kv": kv},
            "limits": {"effort": force} if math.isfinite(force) else {}}


def _geometry(model, gid: int, meshes: dict[int, str]) -> dict:
    at = pose(model.geom_pos[gid], model.geom_quat[gid].tolist())
    mid = int(model.geom_matid[gid])
    rgba = (model.mat_rgba[mid] if mid >= 0 else model.geom_rgba[gid]).tolist()
    kind, size = int(model.geom_type[gid]), model.geom_size[gid]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        return {"kind": "mesh", "ref": meshes[int(model.geom_dataid[gid])], "pose": at, "rgba": rgba}
    shapes = {int(mujoco.mjtGeom.mjGEOM_BOX): ("box", (size * 2).tolist()),
              int(mujoco.mjtGeom.mjGEOM_SPHERE): ("sphere", [float(size[0])]),
              int(mujoco.mjtGeom.mjGEOM_CAPSULE): ("capsule", [float(size[0]), float(size[1] * 2)]),
              int(mujoco.mjtGeom.mjGEOM_CYLINDER): ("cylinder", [float(size[0]), float(size[1] * 2)]),
              int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): ("ellipsoid", (size * 2).tolist())}
    if kind not in shapes:
        raise UsdImportError(f"unsupported USD geometry type {kind}: {model.geom(gid).name}")
    shape, dimensions = shapes[kind]
    return primitive(shape, dimensions, at, rgba)


def _couplings(stage, joints: list[Joint]) -> list[dict]:
    """Translate PhysX single-axis mimic equations into MuJoCo joint equalities."""
    from pxr import UsdGeom
    by_id = {j.id: j for j in joints}
    out = []
    for prim in stage.Traverse():
        for rel in prim.GetRelationships():
            if not (rel.GetName().startswith("physxMimicJoint:") and rel.GetName().endswith(":referenceJoint")):
                continue
            targets = rel.GetTargets()
            first = str(prim.GetPath())
            if len(targets) != 1 or first not in by_id or str(targets[0]) not in by_id:
                raise UsdImportError(f"mimic relationship must reference two imported joints: {rel.GetPath()}")
            second = str(targets[0])
            prefix = rel.GetName().rsplit(":", 1)[0]
            gearing, offset = prim.GetAttribute(prefix + ":gearing"), prim.GetAttribute(prefix + ":offset")
            # PhysxMimicJointAPI schema defaults: q1 + gearing*q2 + offset = 0.
            ratio = float(gearing.Get()) if gearing and gearing.HasAuthoredValueOpinion() else 1.0
            shift = float(offset.Get()) if offset and offset.HasAuthoredValueOpinion() else 0.0
            unit = lambda j: math.pi / 180 if j.type.value == "revolute" else UsdGeom.GetStageMetersPerUnit(stage)
            if not math.isfinite(ratio) or not math.isfinite(shift):
                raise UsdImportError(f"mimic coefficients must be finite: {rel.GetPath()}")
            out.append({"joint1": first, "joint2": second, "offset": -shift * unit(by_id[first]),
                        "multiplier": -ratio * unit(by_id[first]) / unit(by_id[second])})
            by_id[first].actuator = None
    return out


def _estimate_inertia(model, bid: int, link: Link) -> None:
    """Estimate a uniform box's COM and diagonal inertia for a body with missing inertia."""
    from steward.math3d import matrix_from_quat
    points = []
    for gid in range(model.body_geomadr[bid], model.body_geomadr[bid] + model.body_geomnum[bid]):
        center, half = model.geom_aabb[gid, :3], model.geom_aabb[gid, 3:]
        rotation = matrix_from_quat(model.geom_quat[gid])
        for x in (-1, 1):
            for y in (-1, 1):
                for z in (-1, 1):
                    points.append(model.geom_pos[gid] + rotation @ (center + half * [x, y, z]))
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    size = high - low
    mass = float(model.body_mass[bid])
    diagonal = mass / 12 * np.array([size[1]**2 + size[2]**2, size[0]**2 + size[2]**2, size[0]**2 + size[1]**2])
    if mass <= 0 or not np.all(diagonal > 0):
        raise UsdImportError(f"cannot estimate nonzero mass/inertia for {link.id}")
    link.inertial = exact({"mass": mass, "diagonal": diagonal.tolist(), "pose": pose((low + high) / 2)})


def _convert_units(spec, length: float, mass: float) -> None:
    """Convert the 4.0 decoder's stage-valued lengths and masses to SI before compilation."""
    if not math.isfinite(length) or length <= 0 or not math.isfinite(mass) or mass <= 0:
        raise UsdImportError("USD metersPerUnit and kilogramsPerUnit must be finite and positive")
    for body in spec.bodies:
        body.pos[:] *= length
        body.ipos[:] *= length
        body.mass *= mass
        body.inertia[:] *= mass * length ** 2
        body.fullinertia[:] *= mass * length ** 2
    for mesh in spec.meshes:
        mesh.scale[:] *= length
    for geom in spec.geoms:
        geom.pos[:] *= length
        if geom.type != mujoco.mjtGeom.mjGEOM_MESH:
            geom.size[:] *= length
        if math.isfinite(geom.mass):
            geom.mass *= mass
    for joint in spec.joints:
        joint.pos[:] *= length
        angular = joint.type == mujoco.mjtJoint.mjJNT_HINGE
        factor = mass * length ** 2 if angular else mass
        joint.damping *= factor
        joint.stiffness *= factor
        joint.armature *= factor
        joint.frictionloss *= mass * (length ** 2 if angular else length)
        if joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
            joint.range[:] *= length
            joint.springref *= length
