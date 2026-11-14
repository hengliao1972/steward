"""Robot-side contact geometry from segmented RGB-D, never world fastener metadata."""
from copy import deepcopy
import math
import numpy as np


def directed_tool(points, category, stamp):
    """Resolve a shaft's end polarity by measured endpoint width; refuse ambiguous masks.

    The wide end is the screw head / driver handle. These are shape hypotheses,
    not calibrated metrology. Full-resolution points are required (not 1 cm voxels).
    """
    if category not in ('toy_screw', 'electric_screwdriver'):
        return None
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 30 or not np.isfinite(p).all():
        return None
    centre = np.median(p, axis=0)
    _, values, axes = np.linalg.svd(p-centre, full_matrices=False)
    if values[0] < 1.25*values[1]:
        return None
    axis = axes[0]
    along = (p-centre) @ axis
    lo, hi = np.quantile(along, [.01, .99])
    length = hi-lo
    if not (.012 <= length <= .035 if category == 'toy_screw' else .10 <= length <= .23):
        return None
    radial = p-centre-along[:, None]*axis
    ends = [radial[along <= lo+.22*length], radial[along >= hi-.22*length]]
    if min(map(len, ends)) < 6:
        return None
    widths = [float(np.quantile(np.linalg.norm(e-np.median(e, axis=0), axis=1), .9)) for e in ends]
    if max(widths) < 1.35*max(min(widths), .001) or abs(widths[1]-widths[0]) < .0015:
        return None
    endpoints = [centre+lo*axis+np.median(ends[0], axis=0),
                 centre+hi*axis+np.median(ends[1], axis=0)]
    wide = int(widths[1] > widths[0])
    head, tip = endpoints[wide], endpoints[1-wide]
    direction = (head-tip)/np.linalg.norm(head-tip)
    return {'tip': tip.tolist(), 'head': head.tolist(),
            'axis': (direction if category == 'toy_screw' else -direction).tolist(),
            'lengthM': float(np.linalg.norm(head-tip)), 'radiusM': max(widths),
            'stamp': float(stamp), 'source': 'rgbd_endpoint_width', 'polarityObserved': True}


def nominal_tool(category, centre, size, rotation, stamp):
    """Catalogue endpoints in the estimated object frame, never visual evidence.

    Long box axis estimates a lying shaft; a tall box estimates a standing shaft.
    Polarity is a nominal positive-axis hypothesis until RGB-D resolves the ends.
    Capture this frame ONCE at grasp, then transport it with robot FK.
    """
    from steward.ego.classify import SHAPES
    if category not in ('toy_screw', 'electric_screwdriver'):
        return None
    dims = np.asarray(SHAPES[category]['dims'])
    length = float(max(dims))
    local = np.eye(3)[int(np.argmax(size))]
    axis = np.asarray(rotation) @ local
    centre = np.asarray(centre)
    direction = axis if category == 'toy_screw' else -axis
    return dict(tip=(centre-direction*length/2).tolist(),
                head=(centre+direction*length/2).tolist(), axis=axis.tolist(),
                lengthM=length, radiusM=float(sorted(dims)[1]/2), stamp=float(stamp),
                source='catalogue_pose_estimate', polarityObserved=False,
                warnings=[dict(reason='nominal_tool_endpoints', category=category,
                               lengthM=length, polarityObserved=False)])


def transform_feature(feature, rotation, translation):
    out = deepcopy(feature)
    for key in ('tip', 'head'):
        out[key] = (rotation @ np.asarray(feature[key])+translation).tolist()
    out['axis'] = (rotation @ np.asarray(feature['axis'])).tolist()
    return out


def surface_patch(points, xy, *, max_tilt_deg=8):
    """Estimate the upward entry surface at a registered XY, including crossbars.

    A local fit, rather than the whole object's top (which can be a leg or tail).
    Returns None for a side face, occlusion, mixed heights, or insufficient support.
    """
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        return None
    near = p[np.linalg.norm(p[:, :2]-xy, axis=1) <= .009]
    if len(near) < 12 or np.ptp(near[:, 0]) < .004 or np.ptp(near[:, 1]) < .004:
        return None
    # Raised marker and board can coexist. Fit the dominant upper horizontal band.
    z = np.quantile(near[:, 2], .8)
    near = near[abs(near[:, 2]-z) <= .002]
    if len(near) < 8:
        return None
    design = np.c_[near[:, :2]-xy, np.ones(len(near))]
    if np.linalg.matrix_rank(design) < 3:
        return None
    fit = np.linalg.lstsq(design, near[:, 2], rcond=None)[0]
    residual = float(np.quantile(abs(design @ fit-near[:, 2]), .95))
    normal = np.r_[-fit[:2], 1.]; normal /= np.linalg.norm(normal)
    if normal[2] < math.cos(math.radians(max_tilt_deg)) or residual > .0015:
        return None
    return {'point': [float(xy[0]), float(xy[1]), float(fit[2])],
            'normal': normal.tolist(), 'residualM': residual, 'points': len(near)}


def head_sank(before, after, entry):
    """Visible axial head motion with stable lateral location; commands cannot satisfy it."""
    if not before or not after or after['stamp'] <= before['stamp']:
        return False
    if before.get('source') != 'rgbd_endpoint_width' or after.get('source') not in ('rgbd_endpoint_width', 'rgbd_screw_head'):
        return False
    a, b, e = map(np.asarray, (before['head'], after['head'], entry))
    return bool(.006 <= a[2]-b[2] <= .030 and np.linalg.norm(a[:2]-b[:2]) <= .008
                and np.linalg.norm(b[:2]-e[:2]) <= .012 and -.003 <= b[2]-e[2] <= .012)


def follows(before, samples, minimum_motion=.035):
    """Two independent post-lift RGB-D pairs, with excitation and stable relative pose.

    Predicted carried poses and merged envelopes are deliberately ineligible.
    """
    if len(samples) < 2:
        return False
    last = max(o['stamp'] for o in before)
    base_rel = np.asarray(before[1]['centreM'])-before[0]['centreM']
    for pair in samples:
        if len(pair) != 2 or min(o['stamp'] for o in pair) <= last:
            return False
        if any(o.get('source') != 'rgbd_segment' or o.get('truncated', True)
               or o.get('invalidDepthFraction', 1) > .05 for o in (*before, *pair)):
            return False
        if abs(pair[0]['stamp']-pair[1]['stamp']) > .5:
            return False
        for old, new in zip(before, pair):
            if not np.allclose(sorted(old['visibleSizeM']), sorted(new['visibleSizeM']), atol=.012):
                return False
            if np.linalg.norm(np.asarray(new['centreM'])-old['centreM']) < minimum_motion:
                return False
        rel = np.asarray(pair[1]['centreM'])-pair[0]['centreM']
        yaw = (pair[0]['boxYawRad']-before[0]['boxYawRad']+math.pi/2) % math.pi-math.pi/2
        c,s=math.cos(yaw),math.sin(yaw)
        expected = np.array([c*base_rel[0]-s*base_rel[1], s*base_rel[0]+c*base_rel[1],base_rel[2]])
        member_yaw = (pair[1]['boxYawRad']-before[1]['boxYawRad']-yaw+math.pi/2) % math.pi-math.pi/2
        if abs(member_yaw) > math.radians(8) or np.linalg.norm(rel-expected) > .010:
            return False
        last = max(o['stamp'] for o in pair)
    return True


def screw_head(view, before):
    """After insertion the shaft can be hidden; require a separately segmented head."""
    if (not view or view.get('source') != 'rgbd_segment' or view.get('truncated', True)
            or view.get('invalidDepthFraction', 1) > .05 or view['stamp'] <= before['stamp']):
        return None
    p = np.asarray(view['points'], float)
    if len(p) < 20 or not np.isfinite(p).all():
        return None
    size = np.quantile(p, .98, axis=0)-np.quantile(p, .02, axis=0)
    if min(size[:2]) < .007 or max(size[:2]) > .025 or size[2] > .025:
        return None
    top = np.quantile(p[:,2], .95)
    cap = p[p[:,2] >= top-.002]
    if len(cap) < 8:
        return None
    head = [*np.median(cap[:,:2], axis=0), top]
    return {'head':list(map(float,head)), 'stamp':view['stamp'], 'source':'rgbd_screw_head'}


def vertical_signature(view):
    """Which end of a full RGB-D shape has a narrow protrusion (e.g. a fin).

    No category, desired pose or catalogue coordinates. Symmetric/occluded objects
    remain unknown; merely seeing a flat box cannot determine its exposed face.
    """
    if (not view or view.get('source') != 'rgbd_segment' or view.get('truncated',True)
            or view.get('invalidDepthFraction',1) > .05):
        return None
    p=np.asarray(view['points'],float)
    if len(p)<80 or not np.isfinite(p).all():
        return None
    lo,hi=np.quantile(p[:,2],[.01,.99]);height=hi-lo
    if height<.025:
        return None
    full=np.prod(np.ptp(p[:,:2],axis=0))
    if full<=0:
        return None
    ends=[p[p[:,2]<=lo+.18*height],p[p[:,2]>=hi-.18*height]]
    if min(map(len,ends))<12:
        return None
    areas=[float(np.prod(np.ptp(e[:,:2],axis=0))/full) for e in ends]
    if min(areas)<.25 and max(areas)>.5 and max(areas)>3*max(min(areas),.01):
        return 'detail_above' if areas[1]<areas[0] else 'detail_below'
    return None


def settled_flip_rotation(release_delta, before, after):
    """Project a tilted release to a flat flip only after observed face reversal.

    Upstream can release past vertical and let gravity settle the remaining angle.
    Two fresh stable, separate full-shape views must show the opposite protrusion;
    action history alone cannot select the other face.
    """
    delta=np.asarray(release_delta,float)
    if (delta @ [0,0,1])[2]>=-.2 or len(after)<2:
        return None
    sign=vertical_signature(before)
    if sign is None:
        return None
    last=before['stamp'];centre=None
    for v in after:
        current=vertical_signature(v)
        if current is None or current==sign or v['stamp']<=last:
            return None
        if not np.allclose(sorted(v['visibleSizeM']), sorted(before['visibleSizeM']),atol=.015):
            return None
        if centre is not None and np.linalg.norm(np.asarray(v['centreM'])-centre)>.008:
            return None
        centre=np.asarray(v['centreM']);last=v['stamp']
    u,_,vt=np.linalg.svd(delta[:2,:2]);xy=u @ vt
    if np.linalg.det(xy)>0:
        return None
    flat=np.zeros((3,3));flat[:2,:2]=xy;flat[2,2]=-1
    return flat
