"""Ordinary placement and a nominal driver stroke, using Ego estimates only."""
from itertools import product
import math
import numpy as np

from steward.ego.esim import PredictionError
from steward.ego.fasten_plan import tool_estimate, support_anchor
from steward.ego.planner import entity_box
from steward.math3d import matrix_from_quat, quat_from_yaw


def offset_on(entity, xy):
    box, _, _ = entity_box(entity)
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    dx, dy = np.asarray(xy)-[box.cx, box.cy]
    return [float(c*dx+s*dy), float(-s*dx+c*dy)]


def landing_error(entity, target, stamp, after):
    """Do not replace the measured screw with its intended release location."""
    feature = tool_estimate(entity, stamp)
    tip = feature.get('tip') if feature else None
    centre = list(entity.pose.value['p']) if entity else None
    fresh = bool(feature and feature.get('source') == 'rgbd_endpoint_width'
                 and feature['stamp'] > after)
    return dict(screw=entity.id if entity else None, screwPositionM=centre,
        screwTipM=tip, targetHoleM=list(target),
        centreMinusHoleM=(np.asarray(centre)-target).tolist() if centre is not None else None,
        tipMinusHoleM=(np.asarray(tip)-target).tolist() if tip is not None else None,
        source=feature.get('source') if feature else 'unavailable',
        stamp=feature.get('stamp') if feature else None, freshEndpointObserved=fresh,
        axis=feature.get('axis') if feature else None)


def driver_stroke(planner, es, params):
    """Orient by grasp FK, dock, lower, dwell and withdraw. No contact corridor/identity gates."""
    held = planner._held(es)
    if held is None:
        raise PredictionError('driver stroke needs a held object')
    feature = tool_estimate(held, es.stamp)
    if feature is None:
        # A changed perception label is not a tool-identity veto.
        from steward.ego.fasten_geometry import nominal_tool
        box, lo, hi = entity_box(held)
        feature = nominal_tool('electric_screwdriver', held.pose.value['p'],
            [2*box.hx, 2*box.hy, hi-lo], matrix_from_quat(held.pose.value['q']), es.stamp)
    current = {'wrist_yaw':0., 'wrist_roll':0., 'wrist_pitch':0., **planner._arm_joints(es)}
    grip = np.asarray(feature.get('grip', held.pose.value['p']))
    base_yaw = planner._robot_xy_yaw(es)[2]
    r0 = matrix_from_quat(quat_from_yaw(base_yaw)) @ planner.arm.tool_rotation(
        current['wrist_yaw'], current['wrist_roll'], current['wrist_pitch'])
    axis_local = r0.T @ feature['axis']
    tip_local = r0.T @ (np.asarray(feature['tip'])-grip)
    g = planner.arm.geometry
    options = []
    for pitch, roll in product(np.linspace(0, g.wrist_pitch_max, 19),
                               np.linspace(-g.wrist_roll_max, g.wrist_roll_max, 73)):
        r = planner.arm.tool_rotation(0., roll, pitch)
        error = math.acos(float(np.clip(-(r @ axis_local)[2], -1, 1)))
        options.append((error, abs(pitch-current['wrist_pitch'])+abs(roll-current['wrist_roll']), pitch, roll))
    options.sort()
    # This is a request for a different grasp, not an attachment admission test.
    # A wrist rolling around the shaft cannot change its inclination.
    if options[0][0] > math.radians(12):
        raise PredictionError('simple driver needs transverse regrasp: wrist cannot point this grasp down')
    body = es.entity(params['body'])
    warnings = []
    anchor = support_anchor(planner, es, body, warnings)
    goal = np.asarray(params['head'], float)
    safe = max(grip[2], goal[2]+feature['lengthM']+.04)
    if safe > planner.lift_max:
        raise PredictionError('driver stroke lift outside arm range')
    failures = []
    # Best orientation first; ordinary docking retains navigation and arm reach checks.
    for error, _, pitch, roll in options[:8]:
        rotation = planner.arm.tool_rotation(0., roll, pitch)
        for heading in (base_yaw, base_yaw+math.pi/2, base_yaw-math.pi/2, base_yaw+math.pi):
            world_r = matrix_from_quat(quat_from_yaw(heading)) @ rotation
            endpoint = goal-world_r @ tip_local
            def wrist(yaw):
                return (heading-yaw+math.pi) % (2*math.pi)-math.pi, float(roll)
            def path(yaw, reach, near):
                wy, wr = wrist(yaw)
                at = dict(current); targets = [at]
                changes = [dict(lift=safe), dict(reach=near), dict(wrist_yaw=wy),
                           dict(wrist_roll=wr), dict(wrist_pitch=float(pitch)),
                           dict(reach=reach), dict(lift=float(endpoint[2])),
                           dict(lift=safe), dict(reach=near),
                           dict(wrist_yaw=current['wrist_yaw']), dict(wrist_roll=current['wrist_roll']),
                           dict(wrist_pitch=current['wrist_pitch'])]
                for change in changes:
                    end = {**at, **change}
                    n = max(1, *(math.ceil(abs(end[k]-at[k])/(.01 if k in ('lift','reach') else .1)) for k in change))
                    targets.extend({k:at[k]+(end[k]-at[k])*i/n for k in at} for i in range(1,n+1))
                    at = end
                return planner.arm.reachable_path(targets,
                    start_joints={j.id:j.q.value[0] for j in planner._self(es).rigid.joints if j.q.value})
            try:
                approach = planner._approach(es, anchor, endpoint[:2], float(endpoint[2]), {held.id},
                    pitches=[float(pitch)], lifts=[safe, float(endpoint[2])], wrist=wrist, approach_path=path)
            except PredictionError as exc:
                failures.append(str(exc)); continue
            wy, wr = wrist(approach['yaw'])
            plan = dict(approach, object=held.id, support=anchor.id,
                gripSpot=endpoint[:2].tolist(), lift=float(endpoint[2]), safeLift=safe,
                wristYaw=wy, wristRoll=wr, wristPitch=float(pitch), goal=goal.tolist(),
                estimatedAxisErrorDeg=math.degrees(error), dwellS=.5,
                parkWrist={k:current[k] for k in ('wrist_yaw','wrist_roll','wrist_pitch')})
            receipt = planner._receipt(es, '_simple_driver', params, .03, [], plan)
            receipt['warnings'].extend(warnings)
            return receipt
    raise PredictionError('simple driver has no reachable stroke: '+'; '.join(failures[-1:]))
