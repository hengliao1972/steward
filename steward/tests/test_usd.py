"""USD imports compile, preserve joint motion and fail without publishing incomplete assets."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import struct

import mujoco
import numpy as np
import pytest

from steward.assets.usd import UsdImportError, import_usd
from steward.scene.extent import entity_obb
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import Entity, Joint, Link, RigidTree, SceneState, exact, pose, primitive
from steward.scene.validate import validate_world


def _mesh(path):
    vertices = np.array([[x, y, z] for x in (0, 0.2) for y in (0, 0.3) for z in (0, 0.4)], dtype='<f4')
    faces = np.array([[0, 1, 2], [1, 3, 2], [4, 6, 5], [5, 6, 7], [0, 4, 1], [1, 4, 5],
                      [2, 3, 6], [3, 7, 6], [0, 2, 4], [2, 6, 4], [1, 5, 3], [3, 5, 7]], dtype='<i4')
    path.write_bytes(struct.pack('<4i', 8, 0, 0, 12) + vertices.tobytes() + faces.tobytes())
    return {"kind": "mesh", "ref": str(path), "pose": pose()}


def test_mesh_geometry_bounds_anchor_and_inertia_round_trip(tmp_path):
    geometry = _mesh(tmp_path / 'box.msh')
    root = Link('root', 'root', geometry={'visual': [primitive('sphere', [0.02])]})
    arm = Link('arm', 'arm', geometry={'collision': [geometry], 'visual': [geometry]},
               inertial=exact({'mass': 2.0, 'diagonal': [0.1, 0.2, 0.25], 'pose': pose([0.1, 0.15, 0.2])}))
    joint = Joint('hinge', 'root', 'arm', exact('revolute'), exact(pose()), exact([math.pi / 2]),
                  exact([0.0]), axis=exact([0, 0, 1]), anchor=exact([0.1, 0.0, 0.0]))
    ent = Entity('w:robot', 'robot', exact('mesh-robot'), exact(pose()), RigidTree(exact('fixed'), [root, arm], [joint]))
    scene = SceneState(mode='world', entities=[ent])
    assert validate_world(scene.to_dict()) == []
    assert SceneState.from_dict(scene.to_dict()).to_dict() == scene.to_dict()
    build = build_mjcf(scene)
    model = build.compile()
    data = mujoco.MjData(model)
    build.apply_initial(model, data)
    assert model.nmesh == 1
    assert len(build.assets) == 1
    assert model.body_mass[2] == pytest.approx(2)
    assert model.body_inertia[2] == pytest.approx([0.1, 0.2, 0.25])
    assert model.jnt_pos[0] == pytest.approx([0.1, 0, 0])
    bounds = entity_obb(model, data, build.names[ent.id], ['arm'])
    assert bounds.size == pytest.approx([0.3, 0.2, 0.4], abs=1e-6)
    assert bounds.center == pytest.approx([-0.05, 0, 0.2], abs=1e-6)
    invalid = scene.to_dict()
    invalid['entities'][0]['body']['joints'][0]['anchor']['value'] = [float('nan'), 0, 0]
    assert any('anchor' in e for e in validate_world(invalid))
    invalid = scene.to_dict()
    invalid['entities'][0]['body']['links'][1]['inertial']['value']['diagonal'] = [0, 1, 1]
    assert any('inertia' in e for e in validate_world(invalid))
    invalid = scene.to_dict()
    del invalid['entities'][0]['body']['links'][1]['inertial']
    assert any('mesh collisions' in e for e in validate_world(invalid))


@pytest.fixture
def usd_stage(tmp_path):
    pytest.importorskip('pxr.Usd')
    try:
        importlib.metadata.version('mujoco-usd')
    except importlib.metadata.PackageNotFoundError:
        pytest.skip('install the usd extra to exercise native import')
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    path = tmp_path / 'robot.usda'
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, '/Scene')
    for name, x in [('/Scene/Base', 0), ('/Scene/Arm', 0.5)]:
        prim = UsdGeom.Xform.Define(stage, name).GetPrim()
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(x, 0, 0))
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(1)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.01, .01, .01))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
        cube = UsdGeom.Cube.Define(stage, name + '/geometry')
        cube.CreateSizeAttr(.2)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.SetDefaultPrim(stage.GetPrimAtPath('/Scene'))
    anchor = UsdPhysics.FixedJoint.Define(stage, '/Scene/Anchor')
    anchor.CreateBody1Rel().SetTargets(['/Scene/Base'])
    joint = UsdPhysics.RevoluteJoint.Define(stage, '/Scene/hinge')
    joint.CreateBody0Rel().SetTargets(['/Scene/Base'])
    joint.CreateBody1Rel().SetTargets(['/Scene/Arm'])
    joint.CreateLocalPos0Attr(Gf.Vec3f(.5, 0, 0))
    joint.CreateAxisAttr('Z')
    joint.CreateLowerLimitAttr(-30)
    joint.CreateUpperLimitAttr(90)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
    drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(2)
    drive.CreateDampingAttr(1)
    drive.CreateMaxForceAttr(10)
    stage.GetRootLayer().Save()
    return stage, path


def test_native_import_creates_independent_model_and_actuated_joint(usd_stage, tmp_path):
    stage, source = usd_stage
    before = source.read_bytes()
    output = tmp_path / 'import'
    report = import_usd(source, output, fixed_base=True)
    assert source.read_bytes() == before
    assert report['sha256'] == hashlib.sha256(before).hexdigest()
    snapshot = {k: report[k] for k in ('links', 'joints', 'meshes', 'actuators', 'couplings')}
    snapshot['sizeMeters'] = [round(x, 6) for x in report['sizeMeters']]
    assert snapshot == json.loads((Path(__file__).parent / 'expected/usd-import.json').read_text())
    ent = Entity.from_dict(json.loads((output / 'template.json').read_text()))
    joints = [j for j in ent.rigid.joints if j.type.value != 'fixed']
    assert joints[0].limits.value == pytest.approx({'lo': -math.pi / 6, 'hi': math.pi / 2})
    assert joints[0].actuator['gains']['kp'] == pytest.approx(2 * 180 / math.pi)
    build = build_mjcf(SceneState(mode='world', entities=[ent]))
    model = build.compile()
    data = mujoco.MjData(model)
    build.apply_initial(model, data)
    data.ctrl[:] = .2
    for _ in range(500):
        mujoco.mj_step(model, data)
    assert not np.any(data.warning.number)
    assert data.qpos[0] == pytest.approx(.2, abs=.03)
    with pytest.raises(UsdImportError, match='already exists'):
        import_usd(source, output)
    assert source.read_bytes() == before


def test_invalid_source_and_options_leave_no_output(usd_stage, tmp_path):
    _, source = usd_stage
    for value in [0, -1, float('nan')]:
        output = tmp_path / 'bad'
        with pytest.raises(UsdImportError, match='meters_per_unit'):
            import_usd(source, output, meters_per_unit=value)
        assert not output.exists()
    invalid = tmp_path / 'invalid.usda'
    invalid.write_text('not a USD file')
    with pytest.raises(UsdImportError, match='USD import failed'):
        import_usd(invalid, tmp_path / 'invalid-output')
    assert not (tmp_path / 'invalid-output').exists()


def test_reversed_joint_normalization_preserves_limits_and_source(usd_stage, tmp_path):
    from pxr import UsdPhysics
    stage, source = usd_stage
    joint = UsdPhysics.Joint(stage.GetPrimAtPath('/Scene/hinge'))
    joint.GetBody0Rel().SetTargets(['/Scene/Arm'])
    joint.GetBody1Rel().SetTargets(['/Scene/Base'])
    stage.GetRootLayer().Save()
    before = source.read_bytes()
    report = import_usd(source, tmp_path / 'oriented', orient_joints=True, fixed_base=True,
                        bound_mass=.001, bound_inertia=.000001, drive_frequency=5)
    assert any('reversed joint' in item for item in report['adjustments'])
    assert source.read_bytes() == before
    ent = Entity.from_dict(json.loads(Path(report['template']).read_text()))
    moving = next(j for j in ent.rigid.joints if j.type.value == 'revolute')
    assert moving.limits.value == pytest.approx({'lo': -math.pi / 2, 'hi': math.pi / 6})


def test_mimic_finger_is_driven_through_imported_equality(usd_stage, tmp_path):
    from pxr import Gf, Sdf, UsdPhysics
    stage, source = usd_stage
    Sdf.CopySpec(stage.GetRootLayer(), '/Scene/Arm', stage.GetRootLayer(), '/Scene/OtherArm')
    stage.GetPrimAtPath('/Scene/OtherArm').GetAttribute('xformOp:translate').Set(Gf.Vec3d(-.5, 0, 0))
    joint = UsdPhysics.RevoluteJoint.Define(stage, '/Scene/other_hinge')
    joint.CreateBody0Rel().SetTargets(['/Scene/Base'])
    joint.CreateBody1Rel().SetTargets(['/Scene/OtherArm'])
    joint.CreateLocalPos0Attr(Gf.Vec3f(-.5, 0, 0))
    joint.CreateAxisAttr('Z')
    joint.CreateLowerLimitAttr(-90)
    joint.CreateUpperLimitAttr(30)
    joint.GetPrim().CreateRelationship('physxMimicJoint:rotX:referenceJoint').SetTargets(['/Scene/hinge'])
    stage.GetRootLayer().Save()
    report = import_usd(source, tmp_path / 'coupled', fixed_base=True, drive_frequency=5)
    ent = Entity.from_dict(json.loads(Path(report['template']).read_text()))
    assert report['couplings'] == 1
    assert ent.rigid.couplings == [{'joint1': '/Scene/other_hinge', 'joint2': '/Scene/hinge',
                                   'offset': 0.0, 'multiplier': -1.0}]
    build = build_mjcf(SceneState(mode='world', entities=[ent]))
    model = build.compile()
    data = mujoco.MjData(model)
    build.apply_initial(model, data)
    data.ctrl[:] = .2
    for _ in range(1000):
        mujoco.mj_step(model, data)
    values = {name: data.qpos[model.jnt_qposadr[model.joint(actual).id]]
              for name, actual in build.names[ent.id].joints.items()}
    assert values['/Scene/hinge'] == pytest.approx(.2, abs=.01)
    assert values['/Scene/other_hinge'] == pytest.approx(-.2, abs=.01)
    assert not np.any(data.warning.number)
    bad = ent.to_dict()
    bad['body']['couplings'][0]['joint2'] = 'missing'
    assert any('couplings' in e for e in validate_world({'mode': 'world', 'entities': [bad]}))


def test_native_failure_does_not_publish_a_partial_asset(usd_stage, tmp_path):
    from pxr import UsdPhysics
    stage, source = usd_stage
    joint = UsdPhysics.Joint(stage.GetPrimAtPath('/Scene/hinge'))
    joint.GetBody1Rel().SetTargets(['/Scene/Base'])
    stage.GetRootLayer().Save()
    output = tmp_path / 'failed'
    with pytest.raises(UsdImportError):
        import_usd(source, output, orient_joints=True)
    assert not output.exists()
    assert not list(tmp_path.glob('.usd-import-*'))


def test_usd_mesh_asset_survives_source_removal(usd_stage, tmp_path):
    from pxr import UsdGeom, UsdPhysics
    stage, source = usd_stage
    stage.RemovePrim('/Scene/Arm/geometry')
    mesh = UsdGeom.Mesh.Define(stage, '/Scene/Arm/geometry')
    mesh.CreatePointsAttr([(-.1, -.1, -.1), (.1, -.1, -.1), (0, .1, -.1), (0, 0, .1)])
    mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    report = import_usd(source, tmp_path / 'mesh-import', fixed_base=True)
    assert report['meshes'] == 1
    source.unlink()
    template = Entity.from_dict(json.loads(Path(report['template']).read_text()))
    model = build_mjcf(SceneState(mode='world', entities=[template])).compile()
    assert model.nmesh == 1
    assert np.all(np.isfinite(model.mesh_vert))


def test_stage_units_and_override_are_explicit(usd_stage, tmp_path):
    from pxr import UsdGeom
    stage, source = usd_stage
    UsdGeom.SetStageMetersPerUnit(stage, .01)
    stage.GetRootLayer().Save()
    small = import_usd(source, tmp_path / 'centimeters', fixed_base=True)
    large = import_usd(source, tmp_path / 'meters', fixed_base=True, meters_per_unit=1)
    assert small['sourceMetersPerUnit'] == .01
    assert small['sizeMeters'] == pytest.approx(np.array(large['sizeMeters']) / 100, abs=1e-6)
