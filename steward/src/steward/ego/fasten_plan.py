"""Receipt planning for staged fastening and precise, measured feature contact."""
from itertools import product
import math
import numpy as np

from steward.ego.esim import PredictionError
from steward.ego.fasten_geometry import surface_patch, nominal_tool
from steward.ego.planner import entity_box
from steward.math3d import matrix_from_quat, quat_from_yaw

SKILLS = ('regrasp_upright', 'seat_screw', 'drive_screw', 'fasten_onto', 'flip_assembly')


def value(entity, key, default=None):
    item = entity.state_vars.get(key) if entity else None
    return item.value if item else default


def tool_estimate(entity, stamp):
    feature = value(entity, 'toolFeature')
    if feature or entity is None:
        return feature
    box, low, high = entity_box(entity)
    return nominal_tool(entity.category.value, entity.pose.value['p'],
                        [2*box.hx, 2*box.hy, high-low],
                        matrix_from_quat(entity.pose.value['q']), stamp)


def interface_holes(planner, entity):
    from steward.ego.hole_registration import register_hole_template
    from steward.ego.priors import SCREW_HOLE_FRAMES, screw_holes
    holes = planner._screw_holes(entity)
    if holes and all(h.get('frame') == 'entity' for h in holes):
        return holes
    box, low, high = entity_box(entity)
    category = entity.category.value
    holes, _ = register_hole_template(screw_holes(category), SCREW_HOLE_FRAMES.get(category),
                                      [2*box.hx, 2*box.hy, high-low], best_effort=True)
    return holes


def mating_holes(planner, es, plan):
    """Select the component's compatible interface, never all unrelated body holes."""
    body, part = es.entity(plan['body']), es.entity(plan['part'])
    bh = planner._select_holes(interface_holes(planner, body), plan.get('selectedHoleIds', plan.get('hole')), body.id)
    if plan.get('selectedHoleIds'):
        return bh
    ph = planner._select_holes(interface_holes(planner, part), plan.get('heldHole'), part.id)
    pairs, grouped = planner._hole_group_pairs(ph, bh,
        plan.get('hole') is None or plan.get('heldHole') is None)
    if grouped and pairs:
        # Same geometric choice as placement, including its minimum-turn tie break.
        fits = planner._hole_fit_candidates(part, ph, body, bh)
        _, _, matched = planner._choose_hole_fit(fits, 'offset',
            preferred_yaw=entity_box(part)[0].yaw)
        plan['selectedHeldHoleIds'] = [a for a, _ in matched]
        ids = {b for _, b in matched}
        return [h for h in bh if h['id'] in ids]
    if grouped:
        raise PredictionError('no compatible component interface')
    return bh


def support_anchor(planner, es, entity, warnings):
    anchor = planner._anchor(es, entity)
    if anchor is not entity and planner._affords(anchor, 'support-surface'):
        return anchor
    candidates = [e for e in es.entities if e.id != entity.id and planner._affords(e, 'support-surface')]
    if not candidates:
        raise PredictionError('no support geometry available for checked contact approach')
    anchor = min(candidates, key=lambda e: np.linalg.norm(
        np.asarray(e.pose.value['p'])[:2]-entity.pose.value['p'][:2]))
    warnings.append(dict(reason='support_relation_estimated', entity=entity.id, support=anchor.id))
    return anchor


def fresh_view(entity, stamp):
    view = value(entity, 'contactView')
    if not view:
        return None
    warnings = []
    if stamp-view['stamp'] > 20 or view['truncated'] or view['invalidDepthFraction'] > .05:
        warnings.append(dict(reason='contact_view_quality', ageS=stamp-view['stamp'],
            truncated=view['truncated'], invalidDepthFraction=view['invalidDepthFraction']))
    return {**view, 'warnings':warnings}


def hole_point(planner, entity, selector):
    holes = planner._select_holes(interface_holes(planner, entity), selector, entity.id)
    if len(holes) != 1 or holes[0].get('frame') != 'entity':
        raise PredictionError('seat_screw needs one registered body hole id, not a group or unregistered prior')
    h = holes[0]
    # XY only: outline registration does not calibrate template root Z.
    r = matrix_from_quat(entity.pose.value['q'])
    point = np.asarray(entity.pose.value['p']) + r @ np.asarray(h['p'])
    return point[:2]


def seat_target(planner, es, params):
    from steward.ego.fasten_release import released_interface
    frame = released_interface(es, params.get('part'), params.get('body'))
    if frame:
        pair = next((p for p in frame['holePairs'] if p['bodyHole'] == params.get('hole')), None)
        h = next((h for h in frame['holes'] if pair and h['id'] == pair['partHole']), None)
        if h is None:
            raise PredictionError('release frame has no correspondence for this interface hole')
        return dict(point=list(h['point']), normal=[0.,0.,1.], partHole=h['id'],
                    observedAt=frame['observedAt'], releasedAt=frame['releasedAt'],
                    targetSource=frame['source'], componentFrame=frame, warnings=[
                        dict(reason='release_fk_interface_estimate', observed=False,
                             heightSource=frame['heightSource'])])
    part, body = es.find(str(params.get('part',''))), es.find(str(params.get('body','')))
    if part is None or body is None or part.id == body.id:
        raise PredictionError('seat_screw needs distinct current part and body ids')
    xy = hole_point(planner, body, params.get('hole'))
    part_holes = [h for h in interface_holes(planner, part) if h.get('frame') == 'entity']
    if not part_holes:
        raise PredictionError('released component has no registered interface')
    r = matrix_from_quat(part.pose.value['q']); p = np.asarray(part.pose.value['p'])
    candidates = [(np.linalg.norm((p+r @ h['p'])[:2]-xy), h, (p+r @ h['p'])[:2]) for h in part_holes]
    distance, h, upper_xy = min(candidates, key=lambda x: x[0])
    warnings = [dict(reason='hole_pair_error', distanceM=float(distance), toleranceM=.020)] if distance > .020 else []
    view = fresh_view(part, es.stamp)
    patch = surface_patch(view['points'], upper_xy) if view else None
    if patch is None:
        # Catalogue interface height is relative to the template base, not box centre.
        # In particular, crossbar height must not become the top of the legs.
        from steward.ego.classify import SHAPES
        height = SHAPES.get(part.category.value, {}).get('dims', [0,0,entity_box(part)[2]-entity_box(part)[1]])[2]
        z = p[2]-height/2+float(h['p'][2])
        patch = dict(point=[*upper_xy.tolist(), float(z)], normal=[0.,0.,1.])
        warnings.append(dict(reason='entry_surface_estimated', heightM=patch['point'][2], observedPoints=len(view['points']) if view else 0))
    warnings.append(dict(reason='catalogue_interface_hypothesis', part=part.id, body=body.id))
    return {**patch, 'partHole': h['id'], 'observedAt': view['stamp'] if view else part.pose.t,
            'warnings':warnings+(view.get('warnings', []) if view else [])}


def drive_context(planner, es, params):
    """Prefer the action ledger; otherwise associate the nearest catalogue interface.

    This supplies an execution target, never a claim that a release occurred.
    """
    attempts = {}
    for ent in es.entities:
        for row in value(ent, 'fasteningEvidence', []):
            if (params.get('screw') and row['screw'] == params['screw'] or
                    not params.get('screw') and row['body'] == params.get('body') and row['hole'] == params.get('hole')):
                attempts[(row['body'], row['hole'])] = row
    if attempts:
        return next(iter(attempts.values()))
    screws = [e for e in es.entities if e.category.value == 'toy_screw'
              and (not params.get('screw') or e.id == params['screw'])]
    choices = []
    for body in es.entities:
        if params.get('body') and body.id != params['body']:
            continue
        for hole in interface_holes(planner, body):
            if params.get('hole') and hole['id'] != params['hole']:
                continue
            for part in es.entities:
                if part.id == body.id or params.get('part') and part.id != params['part']:
                    continue
                peers = interface_holes(planner, part)
                if not any(h.get('group') in hole.get('mates', ()) for h in peers):
                    continue
                target = seat_target(planner, es, dict(part=part.id, body=body.id, hole=hole['id']))
                for screw in screws:
                    feature = tool_estimate(screw, es.stamp)
                    error = np.linalg.norm(np.asarray(feature['tip'])-target['point'])
                    # Also prefer a component whose interface is already near this body.
                    xy = hole_point(planner, body, hole['id'])
                    error += np.linalg.norm(np.asarray(target['point'])[:2]-xy)
                    choices.append((error, dict(body=body.id, part=part.id, hole=hole['id'],
                        partHole=target['partHole'], screw=screw.id, entry=target['point'],
                        headBeforeRelease=feature['head'], headBeforeSource=feature['source'],
                        seatedAt=feature['stamp'], verified=False, status='seating_record_estimated')))
    if not choices:
        raise PredictionError('drive has no screw/interface target in current estimates')
    return min(choices, key=lambda pair:pair[0])[1]


def predict(planner, es, skill, params):
    if params.get('method', 'simple') not in ('simple', 'contact'):
        raise PredictionError('screw method must be simple or contact')
    if skill == 'fasten_onto':
        result = planner._predict_assemble_onto(es, params)
        result.update(skill=skill, params=dict(params))
        result['plan'].update(method=params.get('method', 'simple'), correctOnce=bool(params.get('correctOnce', True)), drive=bool(params.get('drive', True)))
        for key in ('stageScrews', 'recoverDropped', 'closeRelease', 'wingRetry'):
            result['plan'][key] = bool(params.get(key, False))
        result['plan']['fastenSkill'] = skill
        result['plan']['selectedHoleIds'] = [h['id'] for h in mating_holes(planner, es, result['plan'])]
        result['plan']['stages'] += ['seat_each_screw', 'drive_each_screw', 'verify_attachment']
        return result
    if skill == 'regrasp_upright':
        target = es.find(str(params.get('target', '')))
        from steward.ego.slender_tool import axial_feature
        if target is None or axial_feature(target, es.stamp) is None:
            raise PredictionError('regrasp_upright needs an estimated elongated target')
        held = planner._held(es)
        if held is not None and held.id != target.id:
            raise PredictionError('regrasp target differs from the held load')
        support = es.find(str(params.get('on', ''))) if params.get('on') else support_anchor(planner, es, target, [])
        if support is None:
            raise PredictionError('regrasp support unavailable')
        return planner._receipt(es, skill, params, .02, [],
            dict(params, support=support.id, fastenSkill=skill, stagedPrediction=True))
    admission_warnings = []
    if planner._held(es) is not None:
        raise PredictionError(f'{skill} starts with an empty hand; park the current load using place')
    body = es.find(str(params.get('body', '')))
    if skill == 'drive_screw':
        context = drive_context(planner, es, params)
        params = {**params, 'screw':context['screw'], 'driveContext':context}
        admission_warnings.append(dict(reason='drive_target_best_estimate', screw=params['screw'], status=context['status']))
    else:
        if body is None:
            raise PredictionError(f'{skill}: body id is absent from current Ego state')
        if skill == 'seat_screw':
            admission_warnings.extend(seat_target(planner, es, params).get('warnings', []))
        else:
            attempts = value(body, 'fasteningEvidence', [])
            admission_warnings.append(dict(reason='attachment_confidence', verifiedAttempts=sum(bool(a.get('verified')) for a in attempts)))
            from steward.ego.fasten_release import loose_components
            loose = loose_components(es, body.id)
            if loose:
                admission_warnings.append(dict(reason='loose_components_before_flip',
                    parts=[p['entity'] for p in loose], confirmedSeatedScrews=0,
                    advisory=True))
        # Attachment confidence is reported by execution, not an admission gate.
    receipt = planner._receipt(es, skill, params, .02,
        ['Staged skill; every motion uses a new receipt. Occluded evidence remains unknown; later stages can fail.'],
        {**params, 'fastenSkill': skill, 'stagedPrediction': True})
    receipt['warnings'].extend(admission_warnings)
    return receipt


def contact(planner, es, params):
    """Plan a measured tool endpoint onto a measured surface, without arbitrary goal coordinates."""
    held = planner._held(es)
    if held is None:
        raise PredictionError('contact needs the requested tool held')
    operation = params.get('operation')
    standing = operation == 'stand'
    expected = 'toy_screw' if operation == 'seat' else 'electric_screwdriver'
    if not standing and (operation not in ('seat', 'drive') or held.category.value != expected):
        raise PredictionError('contact tool identity does not match the operation')
    from steward.ego.slender_tool import axial_feature
    feature = axial_feature(held, es.stamp) if standing else tool_estimate(held, es.stamp)
    if not feature:
        raise PredictionError('held object has no supported tool geometry')
    if 'grip' not in feature:
        # Recovery for a restored held state without the grasp snapshot.
        feature = {**feature, 'grip':list(held.pose.value['p']),
                   'warnings':[*feature.get('warnings', []), dict(reason='grasp_centre_estimated')]}
    body = es.find(str(params.get('body', '')))
    if standing:
        body = es.find(str(params.get('stageOn', '')))
    if body is None:
        raise PredictionError('contact body disappeared')
    staging = bool(params.get('stageOn'))
    if staging:
        anchor = es.entity(params['stageOn'])
        if not planner._affords(anchor, 'support-surface'):
            raise PredictionError('screw staging requires a perceived table surface')
        target = dict(point=[*params['stageXY'], entity_box(anchor)[2]], normal=[0,0,1], warnings=[])
    else:
        target = seat_target(planner, es, params)
    goal = np.asarray(target['point'])
    if operation == 'drive':
        screw = es.find(str(params.get('screw', '')))
        observed = value(screw, 'toolFeature')
        attempts = value(body, 'fasteningEvidence', [])
        attempt = next((a for a in attempts if a['screw'] == params.get('screw')), {})
        if not observed or observed.get('stamp', 0) <= attempt.get('seatedAt', -1):
            from steward.ego.classify import SHAPES
            head = attempt.get('headBeforeRelease', (goal+[0,0,max(SHAPES['toy_screw']['dims'])]).tolist())
            observed = dict(head=head, tip=goal.tolist(), axis=[0,0,1], stamp=es.stamp)
            target.setdefault('warnings', []).append(dict(reason='seated_head_nominal', screw=params.get('screw')))
        target.setdefault('warnings', []).append(dict(reason='screw_head_estimate',
            ageS=es.stamp-observed['stamp'], axisZ=float(observed['axis'][2]),
            entryErrorM=float(np.linalg.norm(np.asarray(observed['tip'])-goal))))
        goal = np.asarray(observed['head']) + [0, 0, .002]
    else:
        goal = goal + [0, 0, .001]  # controlled release, not the ordinary 20 mm place drop
    anchor = anchor if staging else support_anchor(planner, es, body, target.setdefault('warnings', []))
    current = {'wrist_pitch':0., 'wrist_roll':0., 'wrist_yaw':0., **planner._arm_joints(es)}
    _, _, base_yaw = planner._robot_xy_yaw(es)
    # A robot-side actual pinch point is carried in the inferred feature, not world pose.
    grip = np.asarray(feature['grip'])
    pitch = float(current.get('wrist_pitch', 0))
    r0 = matrix_from_quat(quat_from_yaw(base_yaw)) @ planner.arm.tool_rotation(
        current.get('wrist_yaw', 0), current.get('wrist_roll', 0), pitch)
    tip_local = r0.T @ (np.asarray(feature['tip'])-grip)
    axis_local = r0.T @ np.asarray(feature['axis'])
    want = np.array([0., 0., 1. if operation == 'seat' else -1.])
    solutions = []
    # Some wrists roll about the tool axis, so roll alone cannot stand a shaft.
    # Search only the body's existing pitch envelope; never expand joint/risk limits.
    pitches = sorted(set([pitch, *np.linspace(0, planner.arm.geometry.wrist_pitch_max, 13)]))
    for trial_pitch, roll in product(pitches, np.linspace(-planner.arm.geometry.wrist_roll_max,
                                                        planner.arm.geometry.wrist_roll_max, 49)):
        base_rotation = planner.arm.tool_rotation(0., roll, trial_pitch)
        error = math.acos(float(np.clip((base_rotation @ axis_local) @ want, -1, 1)))
        if error > math.radians(6):
            continue
        for alpha in np.linspace(-math.pi, math.pi, 25):
            r = matrix_from_quat(quat_from_yaw(alpha)) @ base_rotation
            endpoint = goal-r @ tip_local
            turn = abs(roll-current.get('wrist_roll', 0))+abs(trial_pitch-pitch)
            solutions.append((error, turn, alpha, roll, trial_pitch, endpoint))
    if not solutions:
        raise PredictionError('cannot stand the estimated shaft upright with this grasp; park and regrasp')
    radius = max(feature['lengthM'], float(np.linalg.norm(tip_local))) + .025
    top = max(entity_box(e)[2] for e in (anchor, body, held) if e is not held)
    safe = max(grip[2], top + radius + .04)
    if safe > planner.lift_max:
        raise PredictionError('insufficient lift clearance for the held tool rotation')
    failures = []
    for _, _, alpha, roll, trial_pitch, endpoint in sorted(solutions, key=lambda x: x[:2])[:24]:
        def wrist(heading):
            yaw = (alpha-heading+math.pi) % (2*math.pi)-math.pi
            return yaw, float(roll)
        def heading_issue(heading):
            return 'wrist yaw limit' if abs(wrist(heading)[0]) > planner.arm.geometry.wrist_yaw_max else None
        def path(heading, reach, near):
            yaw, rr = wrist(heading)
            at = dict(current)
            stages = [{'lift': safe}, {'reach': near}, {'wrist_yaw': yaw}, {'wrist_roll': rr}, {'wrist_pitch':float(trial_pitch)},
                      {'reach': reach}, {'lift': float(endpoint[2])}, {'lift': safe}, {'reach': near}]
            if operation in ('seat', 'stand'):
                stages += [{'wrist_yaw':0.}, {'wrist_roll':0.}, {'wrist_pitch':0.}]
            targets = [at]
            for change in stages:
                end = {**at, **change}
                n = max(1, *(math.ceil(abs(end[k]-at[k])/(.01 if k in ('lift','reach') else .1)) for k in change))
                targets.extend({k: at[k]+(end[k]-at[k])*i/n for k in at} for i in range(1, n+1))
                at = end
            return planner.arm.reachable_path(targets,
                start_joints={j.id:j.q.value[0] for j in planner._self(es).rigid.joints if j.q.value})
        try:
            approach = planner._approach(es, anchor, endpoint[:2], float(endpoint[2]), {held.id},
                pitches=[float(trial_pitch)], lifts=[safe, float(endpoint[2])], wrist=wrist,
                heading_issue=heading_issue, approach_path=path)
        except PredictionError as exc:
            failures.append(str(exc)); continue
        # Do not silently displace a contact goal to a nearby free place.
        sbox, _, _ = entity_box(anchor)
        dx,dy=goal[:2]-[sbox.cx,sbox.cy]; c,s=math.cos(sbox.yaw),math.sin(sbox.yaw)
        if abs(c*dx+s*dy)+radius > sbox.hx or abs(-s*dx+c*dy)+radius > sbox.hy:
            raise PredictionError('tool sweep too near support edge')
        for other in es.entities:
            if other.id in (held.id, None if staging else body.id, params.get('part'), params.get('screw'), anchor.id, es.self_id):
                continue
            box, low, high = entity_box(other)
            dx,dy=goal[:2]-[box.cx,box.cy]; c,s=math.cos(box.yaw),math.sin(box.yaw)
            intersects = (abs(c*dx+s*dy)<box.hx+feature['radiusM']+.01 and
                          abs(-s*dx+c*dy)<box.hy+feature['radiusM']+.01) if staging else (
                              math.hypot(dx,dy)<max(box.hx,box.hy)+feature['radiusM']+.01)
            if staging:
                from steward.ego.assembly_quick import occupied_at
                intersects = occupied_at(other,goal[:2],feature['radiusM']+.01)
            if high > goal[2]+.008 and low < safe and intersects:
                raise PredictionError(f'contact corridor obstructed by {other.id}')
        yaw, rr = wrist(approach['yaw'])
        plan = {**approach, 'object':held.id, 'support':anchor.id, 'gripSpot':endpoint[:2].tolist(),
            'lift':float(endpoint[2]), 'safeLift':safe, 'wristYaw':yaw, 'wristRoll':rr, 'wristPitch':float(trial_pitch),
            'goal':goal.tolist(), 'desiredAxis':want.tolist(), 'operation':operation,
            'targetEvidence':target, 'contactParams':dict(params),
            'toolLengthM':feature['lengthM'], 'toolRadiusM':feature['radiusM'],
            'polarityObserved':feature.get('polarityObserved', False)}
        receipt = planner._receipt(es, '_upright_tool' if standing else ('_stage_screw' if staging else '_fastener_contact'), params, .03,
            ['Endpoint pose uses best available shape plus grasp/encoders; verify independently after contact.'], plan)
        receipt['warnings'].extend(feature.get('warnings', []))
        return receipt
    raise PredictionError('no continuous contact approach: ' + '; '.join(failures[-1:]))


def reset_tool(planner, es, params):
    """Restore grasp-time roll at the current clear, retracted carrying pose."""
    held = planner._held(es); feature = tool_estimate(held, es.stamp)
    if feature and 'grip' not in feature:
        feature = {**feature, 'grip':held.pose.value['p']}
    if not held or held.category.value != 'electric_screwdriver' or not feature or 'grip' not in feature:
        raise PredictionError('driver roll reset needs a held driver target')
    grip = np.asarray(feature['grip'])
    radius = max(np.linalg.norm(np.asarray(feature[k])-grip) for k in ('tip','head'))+feature['radiusM']+.02
    for other in es.entities:
        if other.id in (held.id,es.self_id):
            continue
        box,lo,hi = entity_box(other)
        if hi < grip[2]-radius or lo > grip[2]+radius:
            continue
        c,s=math.cos(box.yaw),math.sin(box.yaw);dx,dy=grip[:2]-[box.cx,box.cy]
        if abs(c*dx+s*dy) < box.hx+radius and abs(-s*dx+c*dy) < box.hy+radius:
            raise PredictionError(f'tool reset sweep obstructed by {other.id}')
    current={'wrist_pitch':0., 'wrist_roll':0., 'wrist_yaw':0., **planner._arm_joints(es)}
    grasp=feature.get('graspWrist', {'wrist_roll':0.,'wrist_pitch':0.})
    target={k:float(grasp.get(k,0.)) for k in ('wrist_roll','wrist_pitch')}
    target['wrist_yaw']=0.
    path=[current]
    for key in ('wrist_yaw','wrist_roll','wrist_pitch'):
        begin=path[-1];steps=max(1,math.ceil(abs(begin.get(key,0)-target[key])/.1))
        path.extend({**begin,key:begin.get(key,0)+(target[key]-begin.get(key,0))*i/steps} for i in range(1,steps+1))
    if not planner.arm.reachable_path(path,
            start_joints={j.id:j.q.value[0] for j in planner._self(es).rigid.joints if j.q.value}):
        raise PredictionError('no continuous arm path to park the driver horizontally')
    receipt = planner._receipt(es, '_fastener_reset', params, .02, [],
        {'wristYaw':0.,'wristRoll':target['wrist_roll'],'wristPitch':target['wrist_pitch']})
    receipt['warnings'].extend(feature.get('warnings', []))
    if 'graspWrist' not in feature:
        receipt['warnings'].append(dict(reason='nominal_grasp_wrist_reset'))
    return receipt


def driver_grasp_heading(planner, es, target, pitch):
    """Choose a docking side from estimated shaft direction before gripping it.

    A poor grasp around a tool-axis wrist may leave no reachable tip-down pose,
    even though ordinary pick succeeds. Do not discover that only after lifting.
    """
    from steward.ego.slender_tool import axial_feature
    feature=axial_feature(target, es.stamp)
    if not feature:
        raise PredictionError('driver grasp requires a driver target')
    axis=np.asarray(feature['axis'])
    future=[planner.arm.tool_rotation(0.,rr,pp)[2] for pp,rr in product(
        np.linspace(0,planner.arm.geometry.wrist_pitch_max,13),np.linspace(-math.pi,math.pi,25))]
    future=np.asarray(future)
    def issue(heading):
        grasp=matrix_from_quat(quat_from_yaw(heading)) @ planner.arm.tool_rotation(0.,0.,pitch)
        local=grasp.T @ axis
        return None if float(np.min(future @ local))<=-math.cos(math.radians(6)) else 'this grasp leaves no reachable driver-tip-down orientation'
    return issue
