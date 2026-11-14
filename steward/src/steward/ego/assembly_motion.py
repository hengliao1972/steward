"""Motion targets from robot-owned registrations, actions and RGB-D only."""
from copy import deepcopy
from itertools import product
import math
import numpy as np

from steward.ego.planner import entity_box
from steward.math3d import matrix_from_quat, quat_from_yaw


def half_turn_roll(current, limit):
    candidates = [current + math.pi, current - math.pi]
    candidates = [r for r in candidates if abs(r) <= limit + 1e-8]
    if not candidates:
        raise ValueError('relative half turn is outside wrist limits')
    return min(candidates, key=abs)


def placement_check(tracks, source, expected, after, stamp):
    """Compare independent RGB-D at both ends of a transport, including changed IDs."""
    rows = []
    for t in tracks:
        v = t.manipulation_view or t.assembly_observation
        if (t.held or t.category != source['category'] or not v or
                v.get('source', 'rgbd_segment') != 'rgbd_segment' or
                v['stamp'] <= after or not 0 <= stamp-v['stamp'] <= 20 or
                v.get('truncated', True) or v.get('invalidDepthFraction', 1) > .05):
            continue
        p = np.asarray(v['centreM'])
        shape = np.asarray(v['visibleSizeM'])
        from steward.ego.priors import SCREW_HOLE_FRAMES
        catalog = SCREW_HOLE_FRAMES.get(source['category'],{}).get('sizeXYM',expected['size'][:2])
        if not (np.allclose(np.sort(shape[:2]), np.sort(expected['size'][:2]), atol=.015) or
                np.allclose(np.sort(shape[:2]), np.sort(catalog), atol=.015)):
            continue
        goal = np.asarray(expected['position'])
        rows.append(dict(entity=t.id, observedAt=v['stamp'], position=p.tolist(),
            sourceDistanceM=float(np.linalg.norm(p-np.asarray(source['position']))),
            deltaXYM=float(np.linalg.norm(p[:2]-goal[:2])), deltaZM=float(p[2]-goal[2])))
        from steward.ego.assembly_quick import mounting_face_up
        rows[-1]['mountingFaceUp'] = mounting_face_up(v)
    placed = [r for r in rows if r['deltaXYM'] <= .03 and abs(r['deltaZM']) <= .02]
    stayed = [r for r in rows if r['sourceDistanceM'] <= .04 and
              (r['deltaXYM'] > .06 or abs(r['deltaZM']) > .03)]
    missed = [r for r in rows if r['entity'] == expected['entity'] and
              (r['deltaXYM'] > .06 or abs(r['deltaZM']) > .03)]
    # A falling component may acquire a new segmentation id. Accept one nearby
    # separate shape below the target; distant same-category parts stay unrelated.
    fallen = [r for r in rows if r['deltaXYM'] < .18 and r['deltaZM'] < -.03]
    if not missed and len(fallen) == 1 and not placed:
        missed = fallen
    passed = None if placed and stayed else True if len(placed) == 1 else False if stayed or missed else None
    return dict(passed=passed, status='ok' if passed is True else 'fail' if passed is False else 'unknown',
        reason='source_still_occupied' if stayed and not placed else 'independent_position_and_height',
        observations=rows, retrySource=(stayed or missed or [None])[0])


def flip_check(before, after, member_before, member_after, since, stamp):
    """A face signature or a reversed body/member height order is evidence; FK is not."""
    from steward.ego.fasten_geometry import vertical_signature
    def valid(v, cutoff):
        return bool(v and v.get('source') == 'rgbd_segment' and v['stamp'] > cutoff
            and 0 <= stamp-v['stamp'] <= 20 and not v.get('truncated', True)
            and v.get('invalidDepthFraction', 1) <= .05)
    if not valid(after, since):
        return dict(passed=None, status='unknown', reason='no_fresh_separate_body')
    a, b = vertical_signature(before), vertical_signature(after)
    if a and b and before['stamp'] <= since:
        return dict(passed=a != b, status='ok' if a != b else 'fail',
                    reason='rgbd_face_signature', before=a, after=b)
    for eid, old in member_before.items():
        new = member_after.get(eid)
        if not before or not old or not valid(new, since):
            continue
        # Independent nearby member and body masks, never an encoder-carried pose.
        if (abs(old['stamp']-before['stamp']) > 5 or
                np.linalg.norm(np.asarray(new['centreM'])[:2]-after['centreM'][:2]) > .15):
            continue
        dz0 = old['centreM'][2]-before['centreM'][2]
        dz1 = new['centreM'][2]-after['centreM'][2]
        if min(abs(dz0), abs(dz1)) > .015:
            passed = dz0*dz1 < 0
            return dict(passed=passed, status='ok' if passed else 'fail',
                        reason='independent_member_height_order', member=eid, beforeDZ=dz0, afterDZ=dz1)
    return dict(passed=None, status='unknown', reason='face_and_height_order_unresolved')


def local_bounds(record):
    ref = record['reference']
    origin = np.asarray(ref['position'])
    rotation = matrix_from_quat(quat_from_yaw(ref['yaw']))
    points = []
    for item in [ref, *record['parts']]:
        r = matrix_from_quat(quat_from_yaw(item['yaw']))
        for signs in product((-1, 1), repeat=3):
            p = np.asarray(item['position']) + r @ (np.asarray(item['size'])*signs/2)
            points.append(rotation.T @ (p-origin))
    return np.min(points, axis=0), np.max(points, axis=0)


def restore_track(est, record):
    """Restore an execution hypothesis, retaining its original observation age."""
    from steward.ego.estimator import Track
    ref = record['reference']
    track = est.tracks.get(ref['entity'])
    if track is None:
        entity = record['entity']
        cov = (entity.pose.unc or {}).get('cov')
        variance = max(.02**2, (cov[0]+cov[7])/2) if cov else .02**2
        track = Track(id=ref['entity'], category_counts={ref['category']:1.},
            pos=np.array(ref['position'], dtype=float), size=np.array(ref['size'], dtype=float),
            yaw=ref['yaw'], var=np.full(3, variance),
            yaw_var=.1, observations=0, first_seen=ref['registeredAt'], last_seen=ref['registeredAt'],
            existence=entity.existence)
        est.tracks[track.id] = track
    return track


def front_spots(planner, scene, table, half):
    """Reach annulus intersected with the inset table, centre of front sector first.

    Every candidate still goes through ordinary placement collision/risk checks.
    """
    box, _, _ = entity_box(table)
    x, y, yaw = planner._robot_xy_yaw(scene)
    g = planner.arm.geometry
    centre = (g.reach_min+g.reach_max)/2 + planner.tool_offset
    rotation = np.array([[math.cos(box.yaw), -math.sin(box.yaw)],
                         [math.sin(box.yaw), math.cos(box.yaw)]])
    inset = np.array([box.hx, box.hy])-half-.04
    if np.any(inset <= 0):
        return []
    result = []
    for radius in (centre, centre*.85, centre*1.15):
        for angle in (0., .25, -.25, .5, -.5):
            p = np.array([x,y])+radius*np.array([math.cos(yaw+angle),math.sin(yaw+angle)])
            local = np.clip(rotation.T @ (p-[box.cx,box.cy]), -inset, inset)
            world = np.array([box.cx,box.cy])+rotation @ local
            if np.dot(world-[x,y], [math.cos(yaw),math.sin(yaw)]) <= 0:
                continue
            if not any(np.linalg.norm(local-other) < .005 for other in result):
                result.append(local)
    return [p.tolist() for p in result]


def released_record(record, spot, table, delta, stamp, view=None):
    """Rebase the commanded frame; retain independent views as separate evidence."""
    new = deepcopy(record)
    old = record['reference']; ref = new['reference']
    r0 = matrix_from_quat(quat_from_yaw(old['yaw']))
    low, high = local_bounds(record)
    corners = [delta @ r0 @ (np.where(np.asarray(signs)>0, high, low))
               for signs in product((-1,1), repeat=3)]
    z = entity_box(table)[2] - min(p[2] for p in corners)
    position = np.array([*spot, z])
    # A post-release mask can be a screw, an old track, or the combined outline.
    # Preserve the commanded body frame; observations belong in the postcheck.
    heading = delta @ r0[:,0]
    yaw = math.atan2(heading[1],heading[0])
    rotation = delta @ np.asarray(old.get('rotationM', r0))
    # Optional catalogue bearing slab distinguishes a mounting face from fins or
    # handles in the full collision envelope. The latter still sizes the sweep.
    from steward.ego.classify import SHAPES
    bearing = SHAPES.get(old['category'], {}).get('bearing_height')
    below = [p for p in record['parts'] if
             (delta @ (np.asarray(p['position'])-old['position']))[2] < 0]
    support_depth = max((p['size'][2] for p in below), default=0.)
    base_z = entity_box(table)[2] + support_depth
    if bearing is not None:
        top_z = base_z + bearing
        position[2] = top_z-old['size'][2]/2 if rotation[2,2] < 0 else base_z+old['size'][2]/2
    ref.update(position=position.tolist(), yaw=yaw, support=table.id, registeredAt=stamp,
               source='commanded_release',
               ownReleaseAt=stamp)
    ref['rotationM'] = rotation.tolist()
    ref['commandedUpZ'] = float(np.asarray(ref['rotationM'])[2, 2])
    ref['landingTopM'] = float(position[2] + old['size'][2]/2)
    if bearing is not None:
        ref['landingTopM'] = float(top_z)
        ref['graspHeightM'] = float(base_z+bearing/2)
        ref['heightSource'] = 'commanded_bearing_stack_catalogue'
    if view:
        new['releaseObservation'] = {k:deepcopy(v) for k,v in view.items() if k != 'points'}
    r1 = matrix_from_quat(quat_from_yaw(yaw))
    for h in ref['holes']:
        points = h.get('positionsByRegistration', [h['p']])
        moved = []
        for point in points:
            q = r1.T @ delta @ r0 @ np.array([*point[:2],0.])
            moved.append([*q[:2].tolist(), point[2]])
        h['p'] = moved[0]
        if 'positionsByRegistration' in h:
            h['positionsByRegistration'] = moved
        h['transportSource'] = ref['source']
    from steward.scene.schema import Est
    ent = new['entity']
    ent.pose.value = dict(p=ref['position'], q=quat_from_yaw(yaw))
    ent.pose.t, ent.pose.src = stamp, 'inferred'
    ent.rigid.links[0].geometry['collision'][0]['size'] = ref['size']
    ent.rigid.links[0].affordances = [a for a in ent.rigid.links[0].affordances
        if not (isinstance(a.value,dict) and 'screw-hole' in a.value)] + [Est(value={'screw-hole':h},src='prior') for h in ref['holes']]
    new.update(status='frozen', hardMotion=False, reason=None, support=deepcopy(table),
               motionSamples={}, warnings=[], aliases=[ref['entity']],
               registrationEvidence=dict(source=ref['source'], confidence='low', views=0,
                   observedAt=old['registeredAt'], propagatedAt=stamp,
                   tier=1, upright=None, directedYawObserved=False))
    for p in new['parts']:
        p['position'] = (position+delta @ (np.asarray(p['position'])-old['position'])).tolist()
        if bearing is not None:
            p['position'][2] = (entity_box(table)[2]+p['size'][2]/2 if
                any(b['entity'] == p['entity'] for b in below) else top_z+p['size'][2]/2)
        axis = delta @ matrix_from_quat(quat_from_yaw(p['yaw']))[:,0]
        p['yaw'] = math.atan2(axis[1],axis[0])
        p['status'] = 'nominal_carry_after_release'
    return new
