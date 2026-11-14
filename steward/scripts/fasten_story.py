#!/usr/bin/env python3
"""Illustrate the upstream airplane manual; no robot controller or GL at import.

Run on a machine with its GL environment already selected. --no-render never constructs
Renderer. Direct placements and an invisible support during belly-up assembly are
story fixtures, not robot capabilities or measurements of grasp/flip control.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


GEAR = 'w:plane-gear'
LABELS = ('initial-table', 'wing-aligned', 'wing-fastened', 'belly-up',
          'gear-aligned', 'gear-fastened', 'upright-stable')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('fasten-story'))
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('--mode', choices=('driver', 'seat'), help='override STEWARD_FASTEN_MODE')
    parser.add_argument('--appearance', choices=('flat', 'textured'), default='textured')
    args = parser.parse_args(argv)
    # Lazy imports: importing this script or asking for --help needs no MuJoCo/GL.
    from dataclasses import replace
    import numpy as np
    import mujoco
    import fasten_demo as demo
    from steward.authoring.service import toy_assembly_goal
    from steward.math3d import matrix_from_quat, pose_compose, pose_inverse, quat_rotate
    from steward.scene.fasteners import FastenConfig, STATE_KEY
    from steward.scene.schema import pose
    from steward.world.goals import evaluate
    from steward.world.sim import WorldSim

    cfg = FastenConfig.from_env()
    if args.mode:
        cfg = replace(cfg, mode=args.mode)
    if cfg.mode == 'off':
        parser.error('the assembly story requires fastening: use --mode driver or --mode seat')
    args.out.mkdir(parents=True, exist_ok=True)
    sim = WorldSim(demo.toy_scene(), fasten=cfg, appearance=args.appearance)
    support = None
    frames = []
    parts = (demo.BASE, demo.WING, GEAR)
    tool_home = sim._entity_pose(demo.TOOL)

    def step(seconds, tool=None):
        for _ in range(math.ceil(seconds / sim.timestep)):
            if support is not None:
                demo.set_pose(sim, demo.BASE, support)
            if tool is not None:
                demo.hold(sim, demo.TOOL, demo.tool_pose(sim, tool))
            sim.step()

    def bounds(eid):
        # Exact vertical primitive extents for the boxes/cylinders in this kit.
        low, high = [], []
        for gid in range(sim.model.ngeom):
            if sim._body_entity.get(int(sim.model.geom_bodyid[gid])) != eid:
                continue
            size = sim.model.geom_size[gid]
            z = sim.data.geom_xmat[gid].reshape(3, 3)[2]
            kind = sim.model.geom_type[gid]
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                half = float(np.abs(z) @ size)
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                half = float(size[0] * np.linalg.norm(z[:2]) + size[1] * abs(z[2]))
            else:
                raise ValueError(f'unexpected kit geometry {kind}')
            center = float(sim.data.geom_xpos[gid, 2])
            low.append(center - half)
            high.append(center + half)
        return min(low), max(high)

    def capture(number):
        label = LABELS[number]
        print(f'STEP {number}: {label}; sim time={sim.time:.3f}s', flush=True)
        frames.append({'step': number, 'label': label, 'time': sim.time,
                       'poses': {eid: sim._entity_pose(eid) for eid in parts}})
        if args.no_render:
            return
        # WorldSim.render uses mujoco.Renderer offscreen, the same shading,
        # cutaway walls, materials and JPEG encoder as /wca/render.
        if number == 0:
            center = sim._entity_pose('w:table-plane')['p'][:2] + [.80]
            distance = 1.7
        else:
            p = sim._entity_pose(demo.BASE)['p']
            lo, hi = bounds(demo.BASE)
            center = [p[0], p[1], (lo + hi) / 2]
            distance = .60
        forward = matrix_from_quat(sim._entity_pose(demo.BASE)['q'])[:, 0]
        heading = math.degrees(math.atan2(forward[1], forward[0]))
        for name, azimuth, elevation in (('oblique', heading + 140, -28),
                                          ('nose', heading + 180, 0), ('side', heading + 90, 0)):
            camera = {'lookat': center, 'distance': distance, 'azimuth': azimuth, 'elevation': elevation}
            data = sim.render(camera, width=960, height=720, fmt='jpeg')
            (args.out / f'{number:02d}_{label}_{name}.jpg').write_bytes(data)

    def move_component(target):
        for eid, at in sim._fastening.targets(demo.BASE, target).items():
            demo.set_pose(sim, eid, at)

    def fasten_pair(part, screws):
        for i, screw in enumerate(screws):
            hole = demo.feature_world(sim, 'screw-hole', part, f'mount_{i}')
            demo.set_pose(sim, screw, pose(hole['entry'], sim._entity_pose(part)['q']))
            # Give contacts time to settle; driver mode still needs the held tool.
            for _ in range(math.ceil(4.0 / sim.timestep)):
                step(sim.timestep, screw if cfg.mode == 'driver' else None)
                if STATE_KEY in sim.template.entity(screw).state_vars:
                    break
            else:
                raise RuntimeError(f'{cfg.mode}: failed to fasten {screw} on {part}')
            demo.release(sim)
            demo.set_pose(sim, demo.TOOL, tool_home)
        step(.3)

    try:
        step(.5)
        capture(0)
        base = sim._entity_pose(demo.BASE)
        demo.set_pose(sim, demo.WING, pose_compose(base, pose([.02, 0, .041])))
        step(.5)
        capture(1)
        fasten_pair(demo.WING, demo.SCREWS)
        capture(2)

        # Roll about the fuselage X axis: keeps heading +X throughout the story.
        # Elevated support leaves the wing/tail clear of the table during assembly.
        base = sim._entity_pose(demo.BASE)
        table_top = bounds('w:table-plane')[1]
        upright_q = base['q']
        support = pose([base['p'][0], base['p'][1], table_top + .18],
                       pose_compose(pose(q=upright_q), pose(q=[0, 1, 0, 0]))['q'])
        move_component(support)
        step(.5)
        capture(3)

        # The gear's crossbar lies against the belly; legs point up now, down
        # after the final roll. This is inverted relative to the fuselage.
        lower = demo.feature_world(sim, 'screw-hole', demo.BASE, 'gear_1')
        gear_feature = next(f for f in sim._fastening.inventory['screw-hole']
                            if f.entity == GEAR and f.spec['id'] == 'mount_0')
        q = upright_q
        rot = matrix_from_quat(q)
        upper_entry = lower['entry'] + lower['axis'] * gear_feature.spec['depth']
        demo.set_pose(sim, GEAR, pose(upper_entry - rot @ np.asarray(gear_feature.spec['entry']), q))
        step(.5)
        capture(4)
        fasten_pair(GEAR, ['w:plane-screw-2', 'w:plane-screw-3'])
        capture(5)

        target = pose(support['p'], upright_q)
        move_component(target)
        members = sim._fastening.component(demo.BASE)
        bottom = min(bounds(eid)[0] for eid in members)
        target['p'][2] += table_top + .001 - bottom
        move_component(target)
        support = None  # all parts freely dynamic for the full five seconds
        first = {eid: sim._entity_pose(eid) for eid in parts}
        late = None
        drift, angular = 0., 0.
        relative_start = {eid: pose_compose(pose_inverse(first[demo.BASE]), first[eid]) for eid in parts}
        for n in range(math.ceil(5 / sim.timestep)):
            sim.step()
            if n * sim.timestep >= 4:
                if late is None:
                    late = {eid: sim._entity_pose(eid) for eid in parts}
                for eid in parts:
                    current = sim._entity_pose(eid)
                    drift = max(drift, float(np.linalg.norm(np.asarray(current['p']) - late[eid]['p'])))
                    angular = max(angular, math.degrees(2 * math.acos(min(1., abs(float(np.dot(current['q'], late[eid]['q'])))))))
        capture(6)
        # Evaluate the unmodified FULL upstream goal, then print its airplane rows.
        goal = toy_assembly_goal()
        result = evaluate(goal, sim._entity_pose)
        up = {eid: bool(quat_rotate(sim._entity_pose(eid)['q'], [0, 0, 1])[2] > math.cos(math.radians(20)))
              for eid in parts}
        rows = [dict(row, uprightSatisfied=up[row['a']] and up[row['b']])
                for row in result['errors'] if row['a'] in parts and row['b'] in parts]
        heights = [{'entity': eid, 'rootZ': sim._entity_pose(eid)['p'][2],
                    'minZ': bounds(eid)[0], 'maxZ': bounds(eid)[1], 'centerZ': sum(bounds(eid)) / 2}
                   for eid in parts]
        heights.sort(key=lambda row: row['centerZ'])
        relative_drift = {}
        for eid in parts[1:]:
            rel = pose_compose(pose_inverse(sim._entity_pose(demo.BASE)), sim._entity_pose(eid))
            relative_drift[eid] = float(np.linalg.norm(np.asarray(rel['p']) - relative_start[eid]['p']))
        table_gap = min(bounds(eid)[0] for eid in members) - table_top
        stable = (drift < .005 and angular < 2 and max(relative_drift.values()) < .005
                  and abs(table_gap) < .005)
        report = {'mode': cfg.mode, 'appearance': args.appearance, 'steps': frames,
                  'airplaneRelations': rows, 'upright': up, 'heightOrderBottomToTop': heights,
                  'stability': {'seconds': 5, 'lastSecondMaxDriftM': drift,
                                'lastSecondMaxRotationDeg': angular, 'relativeDriftM': relative_drift,
                                'tableContactGapM': table_gap,
                                'satisfied': stable},
                  'fullGoal': result}
        (args.out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k not in ('steps', 'fullGoal')},
                         indent=2, ensure_ascii=False), flush=True)
        print('Expected manual/goal conflict: final gear local +Z faces down; goals upright requires +Z up.')
        print('STABILITY ' + ('PASS' if stable else 'FAIL'))
        # A goal mismatch is reported, never repaired. Physics instability is an execution failure.
        return 0 if stable else 1
    finally:
        sim.stop()


if __name__ == '__main__':
    raise SystemExit(main())
