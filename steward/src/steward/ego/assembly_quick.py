"""Short transport targets and passive postchecks, using Ego geometry only."""
import math
import numpy as np
from steward.ego.planner import entity_box
from steward.ego.fasten_plan import hole_point


def occupied_at(entity, xy, margin):
    """Use separate registered members, not empty corners of their union box."""
    from steward.ego.fasten_plan import value
    assembly=value(entity,'orientation',{}).get('assembly',{})
    box,_,_=entity_box(entity)
    items=([assembly['reference'],*assembly.get('parts',[])] if assembly.get('reference') else
           [dict(position=[box.cx,box.cy,0],size=[2*box.hx,2*box.hy,0],yaw=box.yaw)])
    for item in items:
        c,s=math.cos(item['yaw']),math.sin(item['yaw'])
        q=np.array([[c,s],[-s,c]]) @ (np.asarray(xy)-item['position'][:2])
        if np.all(abs(q)<np.array(item['size'][:2])/2+margin):
            return True
    return False


def staging_spots(planner, scene, body, part, table, holes):
    """Within 10 cm of every selected hole, outside the body/future part."""
    from steward.ego.priors import SCREW_HOLE_FRAMES
    points = np.array([hole_point(planner, body, h['id']) for h in holes])
    centre = points.mean(axis=0)
    box, _, _ = entity_box(table)
    b, _, _ = entity_box(body)
    c, s = math.cos(b.yaw), math.sin(b.yaw)
    rot = np.array([[c,-s],[s,c]])
    # Catalogue frames use interface axes, unlike an undirected sorted OBB.
    footprint = SCREW_HOLE_FRAMES.get(part.category.value, {}).get('sizeXYM')
    if footprint is None:
        pbox, _, _ = entity_box(part)
        footprint = [2*pbox.hx,2*pbox.hy]
    robot = np.array(planner._robot_xy_yaw(scene)[:2])
    spots = []
    for radius in (.07,.085,.095):
        for angle in np.linspace(-math.pi,math.pi,32,endpoint=False):
            xy = centre + radius*np.array([math.cos(angle),math.sin(angle)])
            if max(np.linalg.norm(points-xy,axis=1)) > .10:
                continue
            local = rot.T @ (xy-centre)
            if np.all(abs(local) < np.array(footprint)/2+.014):
                continue
            if occupied_at(body,xy,.018):
                continue
            tc,ts=math.cos(box.yaw),math.sin(box.yaw)
            q=np.array([[tc,ts],[-ts,tc]]) @ (xy-[box.cx,box.cy])
            if np.any(abs(q)+.04 > [box.hx,box.hy]):
                continue
            spots.append(xy)
    spots.sort(key=lambda xy:np.linalg.norm(xy-robot))
    return [p.tolist() for p in spots]


def mounting_face_up(view):
    """Broad crossbar below two narrow supports => legs up; converse => flip.

    Never infer this from fused Z or commanded orientation. Occlusion is unknown.
    """
    if (not view or view.get('source') != 'rgbd_segment' or view.get('truncated',True)
            or view.get('invalidDepthFraction',1) > .05):
        return None
    p=np.asarray(view.get('points',[]),float)
    if p.ndim != 2 or len(p)<80:
        return None
    lo,hi=np.quantile(p[:,2],[.02,.98]); height=hi-lo
    if height < .015:
        return None
    bands=[p[p[:,2]<lo+.3*height],p[p[:,2]>hi-.3*height]]
    # The bar fills the centre between legs, so inspect central occupancy rather
    # than bounding area (two leg tips also span the full outer width).
    xy=p[:,:2]-np.median(p[:,:2],axis=0)
    _,_,v=np.linalg.svd(xy,full_matrices=False)
    width=np.ptp(xy @ v[0])
    if width<.035 or min(map(len,bands))<15:
        return None
    ratios=[float(np.mean(abs((b[:,:2]-np.median(p[:,:2],axis=0)) @ v[0])<.18*width)) for b in bands]
    if ratios[0]>.25 and ratios[1]<.10:
        return True
    if ratios[1]>.25 and ratios[0]<.10:
        return False
    return None


def level_assessment(record):
    """Catalogue support calculation; no unsupported table-edge motion."""
    from steward.ego.classify import SHAPES
    shape=SHAPES.get(record['reference']['category'],{})
    slab=shape.get('bearing_height')
    if slab is None or not record['parts']:
        return dict(applied=False,reason='no_asymmetric_support_model')
    protrusion=max(0.,shape['dims'][2]-slab)
    wing=max(SHAPES.get(p.get('category'),{}).get('dims',p['size'])[2] for p in record['parts'])
    span=shape['dims'][0]/2
    tilt=math.degrees(math.atan2(max(0.,protrusion-wing),span))
    return dict(applied=False,reason='yaw_cannot_level_unequal_supports_edge_pose_unresolved',
                estimatedTiltDeg=tilt, onTableYawImprovementDeg=0.,
                idealCantileverTiltDeg=0.,
                limitation='edge placement needs directed protrusion pose and supported centre of mass; outline registration supplies neither')
