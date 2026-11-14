"""Planar template-to-box registration using shape geometry, never category names.

The outline fixes an axis, not its sign. Both rigid transforms survive in each
feature's positionsByRegistration; a shared index denotes one whole-object frame.
No scaling of hole spacing is inferred from noisy outline dimensions.
"""
from __future__ import annotations

import math

import numpy as np


def assembly_posture(observed_size, template_size, *, slack_m=0.0, extent_ratio=1.4):
    """A shape-consistency alarm, never an upright measurement from a yaw-only box.

    XY axes may swap; vertical extent may not. Incompatibility can mean tipping,
    occlusion, a merged mask or wrong identity. Even a compatible box cannot
    distinguish upright from upside down, or certify a particular tilt angle.
    """
    result = {"marker": "[RSI_ASSEMBLY_POSTURE]", "status": "unknown", "upright": None,
              "source": "perceived_extents_and_catalogue", "dimensionsCompatible": None}
    try:
        observed, expected = np.asarray(observed_size, float), np.asarray(template_size, float)
        if (observed.shape != (3,) or expected.shape != (3,)
                or not np.isfinite(observed).all() or not np.isfinite(expected).all()
                or min(*observed, *expected) <= 0):
            raise ValueError("invalid extents")
    except (TypeError, ValueError):
        return {**result, "reason": "missing or invalid shape evidence; inspect support and top/side features"}
    observed = np.r_[np.sort(observed[:2]), observed[2]]
    expected = np.r_[np.sort(expected[:2]), expected[2]]
    ratio = np.maximum(observed / expected, expected / observed)
    mismatch = (ratio > extent_ratio) & (abs(observed - expected) > slack_m)
    compatible = not bool(np.any(mismatch))
    # A depth mask often sees only the main body, not a thin raised feature.
    # Height loss alone is insufficient to reject an otherwise matching footprint.
    height_only_partial = bool(mismatch[2] and not np.any(mismatch[:2]) and observed[2] < expected[2])
    return {**result, "status": "unknown" if compatible or height_only_partial else "inconsistent",
            "dimensionsCompatible": compatible,
            "reason": ("partial vertical extent alone cannot establish tipping; inspect top/side features"
                       if height_only_partial else
                       "compatible extents do not measure roll/pitch or distinguish upside down; inspect top/side features"
                       if compatible else "shape inconsistent with upright catalogue: possible tipping, partial/merged mask or wrong identity; stop mating and reobserve")}


def register_hole_template(holes: list[dict], template: dict | None, observed_size,
                           *, extent_ratio: float = 1.4, square_ratio: float = .9,
                           extent_slack_m: float = 0.0, best_effort: bool = False
                           ) -> tuple[list[dict], dict]:
    """Register centred template XY axes to the observed box's local XY axes.

    Templates declare sizeXYM in the same frame as their holes and centerXYM
    (the outline centre in that frame). Z remains template data: this routine
    establishes planar mating geometry only, not a 3D seating transform.
    """
    info = {"marker": "[RSI_HOLE_REGISTER]", "status": "unregistered"}

    def reject(reason):
        return ([{**h, "frame": "unregistered"} for h in holes], {**info, "reason": reason})

    if template is None:
        return reject("template outline frame is missing")
    try:
        ts = np.asarray(template["sizeXYM"], float)
        centre = np.asarray(template["centerXYM"], float)
        obs = np.asarray(observed_size, float)[:2]
        points = np.asarray([h["p"] for h in holes], float)
        if (ts.shape != (2,) or centre.shape != (2,) or obs.shape != (2,)
                or points.shape != (len(holes), 3) or not holes
                or not all(np.isfinite(a).all() for a in (ts, centre, obs, points))
                or min(*ts, *obs) <= 0):
            return reject("invalid template or observed geometry")
    except (KeyError, TypeError, ValueError):
        return reject("invalid template or observed geometry")
    warnings = []
    # Features on the outline centre do not depend on the in-plane rotation, so they need no axis.
    centred = bool(np.allclose(points[:, :2], centre, atol=1e-6))
    if not centred and (min(ts) / max(ts) >= square_ratio or min(obs) / max(obs) >= square_ratio):
        if not best_effort:
            return reject("near-square outline has no distinguishable principal axis")
        warnings.append(dict(reason="ambiguous_outline_axis", axisRatio=float(min(obs)/max(obs))))
    ratios = np.sort(obs) / np.sort(ts)
    # A small part's outline carries an absolute measurement error (mask bleed, depth noise) that a
    # pure ratio would reject; a dimension is compatible within the ratio or within the slack.
    off_ratio = (ratios > extent_ratio) | (ratios < 1 / extent_ratio)
    if np.any(off_ratio & (np.abs(np.sort(obs) - np.sort(ts)) > extent_slack_m)):
        if not best_effort:
            return reject("observed outline dimensions are incompatible with the template")
        warnings.append(dict(reason="template_extent_mismatch", ratios=ratios.tolist()))
    # A box's long axis can be either local axis; do not assume a particular producer.
    angle = (int(np.argmax(obs)) - int(np.argmax(ts))) * math.pi / 2
    positions = []
    for turn in (angle, angle + math.pi):
        c, s = math.cos(turn), math.sin(turn)
        xy = (points[:, :2] - centre) @ np.array([[c, s], [-s, c]])
        positions.append(np.column_stack((xy, points[:, 2])).tolist())
    registered = [{**h, "p": positions[0][i], "frame": "entity",
                   "registration": "outline_axes",
                   "positionsByRegistration": [ps[i] for ps in positions]}
                  for i, h in enumerate(holes)]
    return registered, {**info, "status": "axis_registered", "directedYawObserved": False,
                        "templateToEntityYawRad": [angle, angle + math.pi], "warnings": warnings,
                        "lowConfidence": bool(warnings),
                        "reason": "outline-compatible prior; both half-turn frames remain possible"}


def coherent_interface_box(rows, template, holes):
    """Estimate centre, extents AND axial yaw from the same close, uncut views.

    A translated union of narrow masks can have a perpendicular principal axis.
    Combining that axis with median per-view extents creates a fictitious box.
    Axial consensus is modulo pi; it never resolves the directed half-turn.
    Rows: centre, size, range, yaw, valid_depth_and_uncut.
    """
    rows = [r for r in rows if len(r) >= 5 and r[4]
            and np.isfinite(np.r_[r[0], r[1], r[2], r[3]]).all()]
    if rows:
        near = min(r[2] for r in rows)
        rows = [r for r in rows if r[2] <= near + .4]
    valid = []
    for r in rows:
        _, registration = register_hole_template(holes, template, r[1])
        if registration['status'] == 'axis_registered' and min(r[1][:2])/max(r[1][:2]) < .9:
            valid.append(r)
    if len(valid) < 3:
        return None
    axes = np.array([r[3] + (math.pi/2 if r[1][1] > r[1][0] else 0) for r in valid])
    deltas = (axes[:, None]-axes[None, :]+math.pi/2) % math.pi-math.pi/2
    medoid = int(np.argmin(np.median(abs(deltas), axis=1)))
    keep = abs(deltas[medoid]) <= math.radians(15)
    if keep.sum() < max(3, math.ceil(.7*len(valid))):
        return None
    selected = [r for r, k in zip(valid, keep) if k]
    yaw = float(axes[medoid] + np.median((axes[keep]-axes[medoid]+math.pi/2) % math.pi-math.pi/2))
    size = np.median([[max(r[1][:2]), min(r[1][:2]), r[1][2]] for r in selected], axis=0)
    centre = np.median([r[0] for r in selected], axis=0)
    return centre, size, yaw


def multiview_interface_box(views, template, holes, *, after=-math.inf,
                            allow_reduced=False, diagnostics=None):
    """Refit a stationary outline from balanced, close RGB-D point sets.

    Repeated images from one pose are NOT independent views. Use three camera
    origins spanning at least 20 degrees and 20 cm. Fit the observed union, not
    a mean of minimum rectangles of one-sided masks; retain half-turn ambiguity.
    allow_reduced retains geometry/depth checks when camera diversity is missing
    and reports low confidence; a split-cloud check is not a second viewpoint.
    """
    from steward.ego.perception import box_of
    diag = diagnostics if diagnostics is not None else {}
    diag.update(inputSamples=len(views), rejected={}, selectedViews=0)
    def reject(reason):
        diag['reason'] = reason
        return None
    def skip(reason):
        diag['rejected'][reason] = diag['rejected'].get(reason, 0) + 1
    selected = []
    for v in sorted(views, key=lambda v: v['stamp'], reverse=True):
        reason = ('old' if v['stamp'] < after else 'far' if v['range'] > 2.0 else
                  'truncated' if v['truncated'] else 'invalid_depth' if v['invalidDepthFraction'] > .05 else None)
        if reason:
            skip(reason); continue
        if any(np.linalg.norm(np.asarray(v['camera'])[:2]-np.asarray(w['camera'])[:2]) < .12 for w in selected):
            skip('repeated_camera'); continue
        points = np.asarray(v['points'], float)
        if len(points) < 30 or not np.isfinite(points).all():
            skip('invalid_points'); continue
        centre, size, yaw = box_of(points, trim=.01)
        _, info = register_hole_template(holes, template, size, extent_slack_m=.015)
        if info['status'] != 'axis_registered':
            skip('template_mismatch'); continue
        selected.append({**v, 'points': points, 'centre': centre, 'size': size, 'yaw': yaw})
        if len(selected) == 4:
            break
    diag['selectedViews'] = len(selected)
    if not selected or (len(selected) < 3 and not allow_reduced):
        return reject('insufficient_independent_views')
    centres = np.array([v['centre'] for v in selected])
    centre = np.median(centres, axis=0)
    # Reject moving bodies or registrations composed of different neighbours.
    if np.max(np.linalg.norm(centres-centre, axis=1)) > .035:
        return reject('centre_inconsistent')
    cameras = np.array([v['camera'] for v in selected])
    bearings = np.arctan2(cameras[:,1]-centre[1], cameras[:,0]-centre[0])
    span = float(np.max(abs((bearings[:,None]-bearings[None,:]+math.pi) % (2*math.pi)-math.pi)))
    baseline = float(np.max(np.linalg.norm(cameras[:,None,:2]-cameras[None,:,:2], axis=2)))
    diag.update(baselineM=baseline, viewSpanDeg=math.degrees(span))
    reduced = len(selected) < 3 or span < math.radians(20) or baseline < .20
    if reduced and not allow_reduced:
        return reject('insufficient_baseline_or_span')
    # Equal point counts prevent the nearest/densest side from dominating trimming.
    n = min(512, min(len(v['points']) for v in selected))
    cloud = np.vstack([v['points'][np.linspace(0, len(v['points'])-1, n, dtype=int)] for v in selected])
    centre, size, yaw = box_of(cloud, trim=.01)
    registered, info = register_hole_template(holes, template, size, extent_slack_m=.015)
    if info['status'] != 'axis_registered':
        return reject('union_template_mismatch')
    # Check leave-one-view-out fits: a single slanted edge must not set the axis.
    axes = []
    clouds = ([v['points'] for v in selected] if reduced else
              [np.vstack([v['points'] for i,v in enumerate(selected) if i != k])
               for k in range(len(selected))])
    # One view still needs two spatially interleaved point subsets to agree.
    # This checks fit conditioning, not independent camera evidence.
    if len(selected) == 1:
        pts = selected[0]['points']
        order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
        clouds = [pts[order[::2]], pts[order[1::2]]]
    for points in clouds:
        _, subsize, axis = box_of(points, trim=.01)
        _, subinfo = register_hole_template(holes, template, subsize, extent_slack_m=.015)
        if subinfo['status'] != 'axis_registered':
            return reject('subset_template_mismatch')
        axes.append(abs((axis-yaw+math.pi/2) % math.pi-math.pi/2))
    spread = max(axes)
    diag['axisSpreadDeg'] = math.degrees(spread)
    if spread > math.radians(8):
        return reject('axis_inconsistent')
    evidence = {'source': 'checked_reduced_rgbd_outline' if reduced else 'balanced_multiview_rgbd_outline',
                'views': len(selected),
                'observedAt': max(v['stamp'] for v in selected), 'baselineM': baseline,
                'viewSpanDeg': math.degrees(span), 'leaveOneOutAxisSpreadDeg': None if reduced else math.degrees(spread),
                'axisCheckSpreadDeg': math.degrees(spread),
                'confidence': 'low' if reduced else 'multiview',
                'mode': ('single_view' if len(selected) == 1 else 'reduced_baseline') if reduced else 'multiview',
                'checks': ['template', 'point_subsets' if len(selected) == 1 else 'cross_view_axis', 'centre_consistency'],
                'directedYawObserved': False, 'upright': None}
    diag['reason'] = 'accepted'
    return centre, size, yaw, registered, info, evidence


def best_interface_box(views, template, holes, fallback, stamp):
    """Best available estimate, with quality checks retained as numeric diagnostics.

    Prefer a recent complete near view, then the latest complete near view,
    then any finite raw view, finally the fused estimate. Never refresh its age.
    """
    from steward.ego.perception import box_of
    candidates = []
    for view in views:
        points = np.asarray(view.get('points', []), float)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3 or not np.isfinite(points).all():
            continue
        complete = (view['range'] <= 2.0 and not view['truncated']
                    and view['invalidDepthFraction'] <= .05 and len(points) >= 30)
        tier = 0 if complete and 0 <= stamp-view['stamp'] <= 5 else 1 if complete else 2
        candidates.append((tier, -view['stamp'], view))
    if candidates:
        tier, _, view = min(candidates, key=lambda x: x[:2])
        centre, size, yaw = box_of(view['points'], trim=.01)
        observed = view['stamp']
        quality = dict(rangeM=view['range'], truncated=view['truncated'],
                       invalidDepthFraction=view['invalidDepthFraction'], points=len(view['points']))
    else:
        tier = 3
        centre, size, yaw, observed = fallback
        quality = dict(rangeM=None, points=0)
    centre, size = np.asarray(centre), np.asarray(size)
    registered, registration = register_hole_template(holes, template, size, best_effort=True)
    evidence = dict(source='best_available_rgbd' if tier < 3 else 'fused_estimate', views=int(tier < 3),
        observedAt=observed, ageS=stamp-observed, confidence='low', lowConfidence=True,
        tier=tier, directedYawObserved=False, upright=None, **quality)
    evidence['warnings'] = [dict(reason='best_available_estimate', **{k:v for k,v in evidence.items()
        if k in ('tier','ageS','rangeM','truncated','invalidDepthFraction','points')})] + registration.get('warnings', [])
    return centre, size, yaw, registered, registration, evidence
