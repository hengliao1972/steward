"""Grasp/release interface frames. Encoder predictions are never observations."""
from copy import deepcopy
import math
import numpy as np

from steward.math3d import matrix_from_quat, quat_from_matrix


def grasp_frame(entity, grip, rotation):
    from steward.ego.classify import SHAPES
    from steward.ego.planner import entity_box
    holes = [deepcopy(a.value['screw-hole']) for a in entity.rigid.links[0].affordances
             if isinstance(a.value, dict) and 'screw-hole' in a.value
             and a.value['screw-hole'].get('frame') == 'entity']
    if not holes:
        return None
    box, low, high = entity_box(entity)
    size = [2*box.hx, 2*box.hy, high-low]
    height = SHAPES.get(entity.category.value, {}).get('dims', size)[2]
    # Outline registration only centres XY. Catalogue Z is measured from its base.
    for h in holes:
        if h.get('source') != 'observed':
            h['p'][2] -= height/2
    r = matrix_from_quat(entity.pose.value['q'])
    return dict(entity=entity.id, category=entity.category.value, size=size, holes=holes,
        offset=(rotation.T @ (np.asarray(entity.pose.value['p'])-grip)).tolist(),
        rotation=(rotation.T @ r).tolist(), observedAt=entity.pose.t,
        source='grasp_geometry_plus_release_fk')


def release_frame(grasp, grip, rotation, target, stamp):
    """Propagate the acquired component, not the command spot or a post-release mask."""
    if not grasp or not (target or {}).get('mateReference'):
        return None
    p = np.asarray(grip)+rotation @ grasp['offset']
    r = rotation @ np.asarray(grasp['rotation'])
    release_p = p.copy()
    # Gravity settling is an explicit Z-only hypothesis. XY and orientation come
    # from FK; the postcheck must still reject a fallen or tilted component.
    settled = target.get('releasedFrame', {}).get('settledCentreZM')
    if settled is not None:
        p[2] = settled
    return dict(entity=grasp['entity'], body=target['mateReference']['entity'],
        category=grasp['category'], size=deepcopy(grasp['size']), position=p.tolist(),
        rotation=r.tolist(), releasePosition=release_p.tolist(),
        releaseGrip=list(map(float, grip)), releaseGripRotation=rotation.tolist(),
        holes=[dict(h, point=(p+r @ h['p']).tolist()) for h in grasp['holes']],
        holePairs=deepcopy(target.get('holePairs', [])), releasedAt=float(stamp),
        observedAt=grasp['observedAt'], source=grasp['source'],
        heightSource='assumed_settled_on_support' if settled is not None else 'release_fk',
        observed=False)


def placement_pairs(planner, held, mate, spot, yaw, params):
    """Keep the correspondence of the actual placement fit, including its half turn."""
    ph = planner._select_holes(planner._screw_holes(held), params.get('heldHole'), held.id)
    bh = planner._select_holes(planner._screw_holes(mate), params.get('hole'), mate.id)
    fits = planner._hole_fit_candidates(held, ph, mate, bh,
        use_catalog=params.get('hole') is None and params.get('heldHole') is None)
    r = matrix_from_quat(mate.pose.value['q'])
    def error(fit):
        offset, angle, _ = fit
        xy = np.asarray(mate.pose.value['p'])[:2]+r[:2,:2] @ offset
        return np.linalg.norm(xy-spot)+abs((angle-yaw+math.pi) % (2*math.pi)-math.pi)
    return [dict(partHole=a, bodyHole=b) for a, b in min(fits, key=error)[2]] if fits else []


def released_interface(es, part, body):
    """The body also carries the snapshot when segmentation has lost the part id."""
    from steward.ego.fasten_plan import value
    frames = [f for e in es.entities for f in value(e, 'releasedInterfaces', [])
              if f['entity'] == part and f['body'] == body]
    return max(frames, key=lambda f:f['releasedAt']) if frames else None


def component_support(es, frame):
    """A planning proxy from acquisition geometry; never a new measured entity."""
    from steward.scene.schema import Est
    ent = deepcopy(es.find(frame['entity']) or es.entity(frame['body']))
    ent.id = frame['entity']
    ent.category = Est(value=frame['category'], src='inferred')
    ent.pose.value = dict(p=frame['position'], q=quat_from_matrix(frame['rotation']))
    ent.rigid.links[0].geometry = {'collision':[dict(type='box', size=frame['size'])]}
    ent.rigid.links[0].affordances = [Est(value={'screw-hole':h}, src='inferred') for h in frame['holes']]
    ent.state_vars = {}
    return ent


def loose_components(es, body):
    from steward.ego.fasten_plan import value
    ent = es.entity(body)
    attempts = value(ent, 'fasteningEvidence', [])
    info = value(ent, 'orientation', {})
    ledger = info.get('stationaryAssembly', {})
    parts = {p['entity']:deepcopy(p) for p in ledger.get('parts', [])}
    parts.update({f['entity']:deepcopy(f) for f in value(ent, 'releasedInterfaces', [])
                  if f['body'] == body})
    return [p for eid, p in parts.items() if not any(
        a['part'] == eid and a['body'] == body and a.get('seatConfirmed')
        for a in attempts)]


def body_grip_points(reference, parts):
    """Body-only pinches, excluding the volumes of loose components."""
    from steward.math3d import quat_from_yaw
    p = np.asarray(reference['position'], float).copy()
    p[2] = reference.get('graspHeightM', p[2])
    r = matrix_from_quat(quat_from_yaw(reference['yaw']))
    size = np.asarray(reference['size'])
    axis = int(np.argmax(size[:2]))
    result = []
    for fraction in (.25, -.25, .4, -.4, 0.):
        local = np.zeros(3); local[axis] = size[axis]*fraction
        point = p+r @ local
        overlaps = False
        for part in parts:
            pr = np.asarray(part.get('rotation', matrix_from_quat(quat_from_yaw(part.get('yaw', 0.)))))
            delta = pr.T @ (point-np.asarray(part['position']))
            if np.all(abs(delta) <= np.asarray(part['size'])/2+.006):
                overlaps = True
                break
        if not overlaps:
            result.append((point[:2].tolist(), (point-p)[:2].tolist()))
    return result


def seat_check(entity, frame, entry, after, stamp):
    """Cheap independent RGB-D posterior, including a shaft-hidden head observation."""
    from steward.ego.fasten_plan import value
    from steward.ego.fasten_geometry import screw_head
    feature = value(entity, 'toolFeature')
    view = value(entity, 'contactView')
    valid = lambda t: t > after and 0 <= stamp-t <= 20
    endpoint = bool(feature and feature.get('source') == 'rgbd_endpoint_width'
                    and valid(feature.get('stamp', -1)))
    head = (dict(head=feature['head'], stamp=feature['stamp'], source=feature['source'])
            if endpoint else screw_head(view, dict(stamp=after)) if view and valid(view['stamp']) else None)
    out = dict(status='unknown', passed=None, reason='no_fresh_separate_screw',
               observedAt=None, freshEndpointObserved=endpoint)
    if head is None:
        return out
    e, h = np.asarray(entry), np.asarray(head['head'])
    xy = float(np.linalg.norm(h[:2]-e[:2])); dz = float(h[2]-e[2])
    inside = None
    if frame:
        q = np.asarray(frame['rotation']).T @ (h-np.asarray(frame['position']))
        inside = bool(np.all(abs(q[:2]) <= np.asarray(frame['size'])[:2]/2+.006))
    if endpoint:
        tip = np.asarray(feature['tip'])
        length = feature.get('lengthM', float(np.linalg.norm(h-tip)))
        aligned = (np.linalg.norm(tip[:2]-e[:2]) <= .012 and abs(tip[2]-e[2]) <= .012
                   and feature['axis'][2] >= math.cos(math.radians(20)))
        height_ok = -.004 <= dz <= length+.006
    else:
        aligned = True
        height_ok = -.004 <= dz <= .012
    passed = bool(xy <= .012 and height_ok and aligned and inside is not False)
    return dict(status='confirmed' if passed else 'failed', passed=passed,
        reason='independent_screw_geometry', observedAt=head['stamp'], source=head['source'],
        headM=h.tolist(), headMinusEntryZM=dz, headDistanceXYM=xy,
        insideComponent=inside, freshEndpointObserved=endpoint)
