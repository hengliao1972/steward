"""Cancellable fastening over robot perception, encoders and ordinary bounded-risk receipts.

No world connection graph, fastener events, judge, assets or scenario access.
Pre-action perceptual confidence is advisory; post-contact claims need observations.
"""
from copy import deepcopy
import math
import time
import numpy as np

from steward.ego.assemble import AssembleOnto
from steward.ego.fasten_geometry import directed_tool, follows, head_sank, screw_head, settled_flip_rotation
from steward.ego.fasten_plan import value, tool_estimate, interface_holes, drive_context, support_anchor, mating_holes
from steward.ego.planner import entity_box
from steward.ego.skills import SkillFailed
from steward.math3d import matrix_from_quat, quat_from_yaw
from steward.scene.schema import Est


class Fasten(AssembleOnto):
    def step(self, job, skill, params, *, receipt=None):
        self._observation_job = job
        self.check()
        started = time.monotonic()
        job.log.append('predict '+skill)
        receipt = receipt or self.rt.predict(skill, params)
        predict_s = time.monotonic()-started
        root = job.postcondition
        for warning in receipt.get('warnings', []):
            self.warn(**warning)
        row = {'skill':skill, 'wallStart':time.time()-predict_s, 'predictWallS':predict_s, 'receipt':deepcopy(receipt), 'result':None}
        root.setdefault('stages', []).append(row)
        if not receipt.get('withinRiskLimit'):
            row['failure'] = '; '.join(receipt.get('issues', []))
            self.progress(job, skill, False, row['failure'])
            raise SkillFailed(f'{skill}: {row["failure"]}')
        try:
            plan = self.rt._consume_receipt(receipt)
            self.ex._begin(job, skill, receipt)
            self.check()
            if skill == 'pick':
                scene = self.rt.ego_state(); target = scene.find(plan['target'])
                origins = self.rt.estimator.fastener_origins
                position = target.pose.value['p'] if target else plan['targetP']
                origin = dict(support=plan['support'], xy=list(position[:2]))
                origins[plan['target']] = origin
                self.rt.estimator.fastener_origins = origins
                self.ex._pick(job, plan)
                if self.rt.estimator.held_id:
                    origins[self.rt.estimator.held_id] = origin
                    getattr(self, '_tool_sources', {}).pop(plan['target'], None)
            elif skill == 'place':
                self.ex._place(job, plan)
                with self.rt._lock:
                    t = self.rt.estimator.tracks.get(plan['object'])
                    if t is not None and hasattr(self, '_tool_sources'):
                        self._tool_sources[t.id] = deepcopy(t)
            elif skill == 'navigate':
                self.ex._drive(job, plan['path'], plan['yaw'])
            elif skill == '_fastener_reset':
                self.ex._set_joint('wrist_yaw', plan['wristYaw'], tol=.02)
                self.ex._set_joint('wrist_roll', plan['wristRoll'], tol=.02)
                if 'wrist_pitch' in self.ex._state()['joints']:
                    self.ex._set_joint('wrist_pitch', plan['wristPitch'], tol=.02)
            elif skill == '_simple_driver':
                self.simple_stroke(job, plan)
            elif skill == '_fastener_contact':
                self.contact(job, plan)
            elif skill in ('_stage_screw', '_upright_tool'):
                self.stage_release(job, plan)
            elif skill == 'assemble_onto':
                AssembleOnto(self.rt).run(job, plan)
            else:
                child = Fasten(self.rt)
                child._tool_sources = getattr(self, '_tool_sources', {})
                child._seat_retry_budget = getattr(self, '_seat_retry_budget', [2])
                child.run(job, plan)
            row['result'] = deepcopy(job.postcondition) if job.postcondition is not root else {'executed':True}
            self.progress(job, skill, True, 'executed', result=root)
            return plan
        except Exception as exc:
            row['failure'] = str(exc)
            self.progress(job, skill, False, str(exc), result=root)
            if job.postcondition is not root:
                row['result'] = deepcopy(job.postcondition)
            raise
        finally:
            row['wallS'] = time.monotonic()-started
            job.postcondition = root

    def reference(self, body_id):
        with self.rt._lock:
            records = self.rt.estimator.stationary_assemblies.records
            record = records.get(body_id)
            if record and record.get('status') != 'frozen' and not record.get('hardMotion'):
                self.warn('stationary_reference_estimated', entity=body_id, status=record.get('status'))
                records.pop(body_id)
        return super().reference(body_id)

    def sample(self, entity):
        with self.rt._lock:
            track = self.rt.estimator.tracks.get(entity)
            return deepcopy(track.manipulation_view) if track else None

    def wait_view(self, entity, after, *, timeout=2):
        start = self.ex._state()['stamp']; deadline = time.monotonic()+300
        while self.ex._state()['stamp']-start < timeout and time.monotonic() < deadline:
            self.check()
            view = self.sample(entity)
            if (view and view['stamp'] > after and not view['truncated']
                    and view['invalidDepthFraction'] <= .05):
                return view
            try:
                self.ex._wait(.25)
            except SkillFailed as exc:
                self.check()
                self.warn('view_wait_timeout_continue', entity=entity, detail=str(exc))
                break
        view = self.sample(entity)
        self.warn('no_fresh_tool_view', entity=entity, after=after, ageS=self.ex._state()['stamp']-view['stamp'] if view else None)
        return view

    def feature(self, entity):
        ent = self.rt.ego_state().find(entity)
        feature = tool_estimate(ent, self.rt.estimator.stamp)
        attempt = next((a for a in self.rt.estimator.fastening_attempts.values() if a['screw'] == entity), None)
        if attempt and (not feature or feature.get('source') != 'rgbd_endpoint_width'
                        or feature['stamp'] <= attempt['seatedAt']):
            from steward.ego.classify import SHAPES
            feature = dict(tip=attempt['entry'], head=attempt['headBeforeRelease'], axis=[0,0,1],
                stamp=attempt['seatedAt'], source='seating_action_estimate', polarityObserved=False,
                lengthM=max(SHAPES['toy_screw']['dims']), radiusM=min(SHAPES['toy_screw']['dims'])/2)
        if not feature:
            raise SkillFailed(f'{entity}: no tool target or seating action to locate it')
        if not feature.get('polarityObserved'):
            self.warn('nominal_tool_endpoints', entity=entity, source=feature['source'])
        return feature

    def choose(self, category, support, *, free_screw=False):
        with self.rt._lock:
            for eid, track in getattr(self, '_tool_sources', {}).items():
                associated = any(t.category == track.category and not t.held
                    and t.last_seen >= track.last_seen and np.linalg.norm(t.pos-track.pos) < .04
                    for t in self.rt.estimator.tracks.values())
                if eid not in self.rt.estimator.tracks and not track.held and not associated:
                    self.rt.estimator.tracks[eid] = deepcopy(track)
                    self.rt.estimator.version += 1
                    self.warn('tool_last_source_restored', entity=eid, observedAt=track.last_seen)
        scene = self.rt.ego_state(); used = self.rt.estimator.fastening_attempts.values()
        used_ids = {a['screw'] for a in used}
        candidates = []
        for ent in scene.entities:
            if ent.category.value != category:
                continue
            feature = tool_estimate(ent, scene.stamp)
            if not feature:
                continue
            if free_screw and (ent.id in used_ids or ent.id in getattr(self, '_retired_screws', set())
                    or ent.id in getattr(self, '_missed_screws', set())
                    or ent.id in getattr(self, '_reserved_screws', set())):
                continue
            if free_screw and any(np.linalg.norm(np.asarray(ent.pose.value['p'])[:2]-a['entry'][:2]) < .025
                                  for a in self.rt.estimator.fastening_attempts.values()):
                continue  # includes a new segmentation ID for an already released screw
            if free_screw and any(np.linalg.norm(np.asarray(ent.pose.value['p'])[:2]-xy) < .025
                                  for xy in getattr(self,'_reserved_positions',[])):
                continue
            anchor = self.rt.esim._anchor(scene, ent)
            if anchor.id != support:
                self.warn('tool_support_estimated', entity=ent.id, expected=support, estimated=anchor.id)
            self.warn('tool_estimate', entity=ent.id, ageS=scene.stamp-feature['stamp'], source=feature['source'])
            candidates.append(ent)
        if not candidates:
            raise SkillFailed(f'no available {category} target on support')
        o = self.ex._state()['odometry']
        return min(candidates, key=lambda e: (self.rt.esim._anchor(scene, e).id != support,
            math.hypot(e.pose.value['p'][0]-o['x'], e.pose.value['p'][1]-o['y']), e.id))

    def stage_release(self, job, plan):
        """Same checked endpoint path as contact; release upright on the table."""
        self.ex._dock(job, plan, retract=False)
        self.ex._set_joint('lift', plan['safeLift'])
        self.ex._set_joint('reach', plan['retractReach'])
        for key, field in (('wrist_yaw','wristYaw'),('wrist_roll','wristRoll'),('wrist_pitch','wristPitch')):
            if key in self.ex._state()['joints']:
                self.ex._set_joint(key, plan[field], tol=.02)
        self.ex._align_lateral(plan['gripSpot'], tol=.002)
        self.ex._set_joint('reach', self.ex._reach_to(plan['gripSpot']))
        self.ex._set_joint('lift', plan['lift'])
        self.ex._settle_for_release()
        self.check()
        self.ex._command({'gripper':'open'})
        self.ex._released(plan['object'], plan['goal'][:2])
        job.postcondition['stagingReleaseAt'] = self.ex._state()['stamp']
        self.ex._wait(.15)
        self.ex._set_joint('lift', plan['safeLift'])
        self.ex._withdraw(plan)
        for key in ('wrist_yaw','wrist_roll','wrist_pitch'):
            if key in self.ex._state()['joints']:
                self.ex._set_joint(key, 0., tol=.02)

    def stage_screws(self, job, plan, holes):
        from steward.ego.assembly_quick import staging_spots
        from steward.ego.classify import SHAPES
        self.body_id = plan['body']
        self.reference(self.body_id)
        scene = self.rt.ego_state()
        body, part = scene.entity(plan['body']), scene.entity(plan['part'])
        table = support_anchor(self.rt.esim, scene, body, [])
        spots = staging_spots(self.rt.esim, scene, body, part, table, holes)
        staged=[]; self._reserved_screws=set(); self._reserved_positions=[]
        for hole in holes:
            candidates=[xy for xy in spots if all(np.linalg.norm(np.asarray(xy)-r['xy'])>.035 for r in staged)]
            if not candidates:
                self.warn('staging_no_near_table_slot', hole=hole['id']); break
            self._stance_retry_used=False; self._missed_screws=set()
            try:
                for pick_try in range(2):
                    screw=self.acquire_tool(job,'toy_screw',table.id,free_screw=True)
                    # RGB-D endpoints win; otherwise the catalogue's table-rest
                    # pose is a better hypothesis than a padded tiny OBB axis.
                    from steward.ego.fasten_geometry import nominal_tool
                    with self.rt._lock:
                        t=self.rt.estimator.tracks[screw.id]
                        if not t.tool_feature or t.tool_feature.get('source') != 'rgbd_endpoint_width':
                            dims=SHAPES['toy_screw']['dims']
                            t.tool_feature=nominal_tool('toy_screw',t.pos,dims,np.eye(3),t.last_seen)
                    try:
                        self.reach_retry(job,'pick',dict(target=screw.id),table.id)
                        break
                    except SkillFailed as exc:
                        self.check()
                        if pick_try or 'grasp missed' not in str(exc) or self.rt.estimator.held_id:
                            raise
                        self._missed_screws.add(screw.id)
                eid=self.rt.estimator.held_id or screw.id
                snapshot=deepcopy(self.rt.estimator.tracks[eid])
                last=None
                for xy in candidates:
                    params=dict(body=body.id,stageOn=table.id,stageXY=xy,operation='seat')
                    receipt=self.rt.predict('_stage_screw',params)
                    if not receipt.get('withinRiskLimit'):
                        last=receipt.get('issues'); continue
                    self.step(job,'_stage_screw',params,receipt=receipt)
                    break
                else:
                    raise SkillFailed(f'no reachable nearby upright staging slot: {last}')
                row=dict(entity=eid,xy=list(xy),hole=hole['id'],support=table.id,
                         position=[*xy,entity_box(table)[2]+max(SHAPES['toy_screw']['dims'])/2],
                         stagedAt=self.ex._state()['stamp'])
                # An action target, explicitly not a new sensory sample.
                snapshot.pos=np.array(row['position']); snapshot.held=False
                snapshot.size=np.array(SHAPES['toy_screw']['dims']); snapshot.yaw=0.
                snapshot.tool_feature=snapshot.held_tool_feature=None
                snapshot.manipulation_view=None
                row['track']=snapshot
                staged.append(row); self._reserved_screws.update((screw.id,eid))
                self._reserved_positions.append(np.asarray(xy))
            except SkillFailed as exc:
                self.check(); self.warn('staging_failed_continue',hole=hole['id'],detail=str(exc))
                if self.rt.estimator.held_id and not self.park(job,table.id):
                    raise
        job.postcondition['stagedScrews']=[{k:v for k,v in r.items() if k!='track'} for r in staged]
        return staged

    def acquire_tool(self, job, category, support, *, free_screw=False):
        scene = self.rt.ego_state(); table = scene.entity(support)
        box,_,_=entity_box(table); o=self.ex._state()['odometry']
        origin=math.atan2(o['y']-box.cy,o['x']-box.cx)
        last = None
        for turn in (None, 0, .65, -.65):
            try:
                if turn is not None:
                    arrived=self.view(job,table,origin,turn)
                    self.look(self.ex._state()['stamp'] if arrived is None else arrived)
                return self.choose(category,support,free_screw=free_screw)
            except SkillFailed as exc:
                last = exc
        raise SkillFailed(str(last))

    def park(self, job, support):
        """Origin, any estimated table space, then nearby floor; failure is a result."""
        from steward.ego.fasten_simple import offset_on
        held = self.rt.estimator.held_id
        if not held:
            return True
        scene = self.rt.ego_state()
        origin = getattr(self.rt.estimator, 'fastener_origins', {}).get(held)
        options = []
        if origin and scene.find(origin['support']):
            options.append(dict(on=origin['support'], offset=offset_on(scene.entity(origin['support']), origin['xy'])))
        tables = [e for e in scene.entities if self.rt.esim._affords(e, 'support-surface')]
        tables.sort(key=lambda e:(e.id != support, np.linalg.norm(np.asarray(e.pose.value['p'])[:2]-self.rt.esim._robot_xy_yaw(scene)[:2])))
        options.extend(dict(on=e.id) for e in tables)
        x,y,yaw = self.rt.esim._robot_xy_yaw(scene)
        options.extend(dict(at=[x+r*math.cos(yaw+a), y+r*math.sin(yaw+a)])
                       for r in (.45,.65) for a in (0, math.pi/2, -math.pi/2, math.pi))
        for params in options:
            try:
                self.step(job, 'place', dict(params, alignHoles=False))
                if not self.rt.estimator.held_id:
                    return True
            except SkillFailed as exc:
                self.check()
                self.warn('parking_candidate_failed', params=params, detail=str(exc))
        self.warn('parking_exhausted', held=self.rt.estimator.held_id)
        return False

    def simple_seat(self, job, plan):
        from steward.ego.fasten_simple import offset_on
        from steward.ego.fasten_plan import seat_target
        self.body_id = plan['body']
        key = (plan['body'], plan['hole'])
        previous = self.rt.estimator.fastening_attempts.get(key)
        if previous and previous.get('seatConfirmed'):
            job.postcondition.update(deepcopy(previous), executed=True, repeated=True, confirmedSeatedScrews=1)
            return
        scene = self.rt.ego_state()
        support = support_anchor(self.rt.esim, scene, scene.entity(self.body_id), []).id
        if previous:
            attempt = previous
        else:
            self._stance_retry_used = False
            self._missed_screws = set()
            self._reserved_screws = set(plan.get('excludedScrews', []))
            self._reserved_positions = [np.asarray(xy) for xy in plan.get('excludedPositions', [])]
            for pick_try in range(2):
                staged_id = plan.get('stagedScrew') if pick_try == 0 else None
                screw = self.rt.ego_state().find(staged_id) if staged_id else None
                screw = screw or self.acquire_tool(job, 'toy_screw', support, free_screw=True)
                try:
                    self.reach_retry(job, 'pick', dict(target=screw.id,
                        **({'stayNear':True} if plan.get('deferVerify') else {})), support)
                    break
                except SkillFailed as exc:
                    self.check()
                    if pick_try or 'grasp missed' not in str(exc) or self.rt.estimator.held_id:
                        raise
                    self._missed_screws.add(screw.id)
                    self.warn('screw_pick_retry_other', missed=screw.id, detail=str(exc))
            screw_id = self.rt.estimator.held_id or screw.id
            scene = self.rt.ego_state()
            target = seat_target(self.rt.esim, scene, plan)
            from steward.ego.fasten_release import component_support
            part = (component_support(scene, target['componentFrame']) if target.get('componentFrame')
                    else scene.entity(plan['part']))
            point = np.asarray(target['point'])
            params = dict(on=part.id, offset=offset_on(part, point[:2]), alignHoles=False, clearance=0.,
                          seatBody=plan['body'], seatHole=plan['hole'])
            if plan.get('closeRelease'):
                params['closeRelease'] = True
            if plan.get('deferVerify'):
                params['stayNear'] = True
            place = self.reach_retry(job, 'place', params, support)
            # Preserve the target actually consumed by the placement receipt.
            target = place.get('seatTarget', target)
            point = np.asarray(target['point'])
            after = place.get('releasedAt', self.ex._state()['stamp'])
            from steward.ego.classify import SHAPES
            height = max(SHAPES['toy_screw']['dims'])
            nominal_tip = [*place['spot'], point[2]]
            attempt = dict(body=plan['body'], part=part.id, hole=plan['hole'], partHole=target['partHole'],
                screw=screw_id, entry=point.tolist(), headBeforeRelease=(np.asarray(nominal_tip)+[0,0,height]).tolist(),
                headBeforeSource='ordinary_place_settled_estimate', seatedAt=after,
                status='released_unverified', verified=False, method='simple', releaseExecuted=True,
                commandedSpotM=place['spot'], estimatedSettledTipM=nominal_tip, landingErrors=[],
                targetSource=target.get('targetSource', 'current_component_estimate'),
                componentFrame=deepcopy(target.get('componentFrame')))
            with self.rt._lock:
                self.rt.estimator.fastening_attempts[key] = attempt
        if plan.get('deferVerify'):
            job.postcondition.update(deepcopy(attempt), executed=True, verificationDeferred=True)
            return
        self.confirm_seat(job, attempt, plan)

    def seat_posterior(self, attempt):
        from steward.ego.fasten_release import seat_check
        from steward.ego.fasten_simple import landing_error
        scene = self.rt.ego_state()
        screw = scene.find(attempt['screw'])
        check = seat_check(screw, attempt.get('componentFrame'), attempt['entry'],
                           attempt['seatedAt'], scene.stamp)
        frame = attempt.get('componentFrame')
        view = value(scene.find(attempt['part']), 'contactView')
        if (frame and view and view.get('source') == 'rgbd_segment' and
                frame['releasedAt'] < view['stamp'] <= scene.stamp and scene.stamp-view['stamp'] <= 20
                and not view.get('truncated', True) and view.get('invalidDepthFraction', 1) <= .05):
            size = np.asarray(view.get('visibleSizeM', [0,0,0]))
            expected = np.asarray(frame['size'])
            # Only a separate, plausible component view can contradict FK.
            # A union envelope cannot shift the target or verify placement.
            separate = np.all(size > expected*.4) and np.all(size < expected*1.6+.01)
            if separate and np.linalg.norm(np.asarray(view['centreM'])-frame['position']) > .035:
                check.update(status='failed', passed=False, reason='component_displaced_after_release',
                             componentObservedAt=view['stamp'])
        if check['passed'] and any(a is not attempt and a.get('seatConfirmed') and
                a['screw'] == attempt['screw'] for a in self.rt.estimator.fastening_attempts.values()):
            check.update(status='failed', passed=False, reason='screw_already_confirmed_in_another_hole')
        attempt.setdefault('landingErrors', []).append(
            landing_error(screw, attempt['entry'], scene.stamp, attempt['seatedAt']))
        attempt.update(seatCheck=check, seatConfirmed=check['passed'] is True,
                       seatStatus=check['status'])
        if not (attempt.get('verified') and check['passed']):
            attempt['status'] = 'seat_'+check['status']
        return check

    def confirm_seat(self, job, attempt, plan, *, observe=True):
        """Two replays per job, after action and independent perception only."""
        from steward.ego.fasten_plan import seat_target
        budget = getattr(self, '_seat_retry_budget', None)
        if budget is None:
            budget = self._seat_retry_budget = [2]
        if observe:
            self.wait_view(attempt['screw'], attempt['seatedAt'])
        check = self.seat_posterior(attempt)
        if observe and check['status'] == 'unknown':
            self.close_seat_view(job, [attempt])
            check = self.seat_posterior(attempt)
        while check['passed'] is not True and budget[0] > 0:
            self.check()
            budget[0] -= 1
            attempt['seatRetryCount'] = attempt.get('seatRetryCount', 0)+1
            attempt.setdefault('seatHistory', []).append(dict(
                screw=attempt['screw'], seatedAt=attempt['seatedAt'], check=deepcopy(check)))
            scene = self.rt.ego_state()
            support = support_anchor(self.rt.esim, scene, scene.entity(attempt['body']), []).id
            try:
                # Only regrasp the old screw when an independent view locates it.
                # A merged mask/predicted release point is not a safe screw identity.
                screw = scene.find(attempt['screw']) if check.get('observedAt') is not None else None
                screw = screw or self.acquire_tool(job, 'toy_screw', support, free_screw=True)
                self.step(job, 'pick', dict(target=screw.id))
                held = self.rt.estimator.held_id or screw.id
                if plan.get('method', 'simple') == 'contact':
                    self.step(job, '_fastener_contact', dict(plan, screw=held, operation='seat'))
                    replacement = self.rt.estimator.fastening_attempts[(attempt['body'], attempt['hole'])]
                    history = {k:attempt[k] for k in ('seatRetryCount', 'seatHistory')}
                    attempt.update(replacement, **history)
                    self.rt.estimator.fastening_attempts[(attempt['body'], attempt['hole'])] = attempt
                else:
                    target = seat_target(self.rt.esim, self.rt.ego_state(), attempt)
                    placed = self.step(job, 'place', dict(on=attempt['part'], alignHoles=False,
                        clearance=0., seatBody=attempt['body'], seatHole=attempt['hole'],
                        **({'closeRelease':True} if plan.get('closeRelease') else {})))
                    target = placed.get('seatTarget', target)
                    attempt.update(screw=held, entry=list(target['point']),
                        seatedAt=placed.get('releasedAt', self.ex._state()['stamp']),
                        commandedSpotM=placed['spot'], componentFrame=deepcopy(target.get('componentFrame')),
                        verified=False, seatConfirmed=False)
                self.wait_view(attempt['screw'], attempt['seatedAt'])
                check = self.seat_posterior(attempt)
            except SkillFailed as exc:
                self.check()
                self.warn('seat_replay_failed', detail=str(exc), hole=attempt['hole'])
                if self.rt.estimator.held_id:
                    self.park(job, support)
        job.postcondition.update(deepcopy(attempt), executed=True,
            confirmedSeatedScrews=int(attempt.get('seatConfirmed', False)))
        if not attempt.get('seatConfirmed'):
            raise SkillFailed('screw landing '+check['status']+' after posterior replay budget')

    def close_seat_view(self, job, attempts):
        """One short camera view aimed at the actual interface, after releases."""
        scene = self.rt.ego_state()
        table = support_anchor(self.rt.esim, scene, scene.entity(attempts[0]['body']), [])
        box,_,_ = entity_box(table); o = self.ex._state()['odometry']
        old_close = getattr(self, '_close_view', False)
        old_target = getattr(self, 'observation_target', None)
        self._close_view = True
        self.observation_target = np.mean([a['entry'] for a in attempts], axis=0).tolist()
        try:
            since = self.view(job, table, math.atan2(o['y']-box.cy,o['x']-box.cx), 0.)
            self.look(self.ex._state()['stamp'] if since is None else since, cycles=1)
        except SkillFailed as exc:
            self.check(); self.warn('seat_posterior_view_unavailable', detail=str(exc))
        finally:
            self._close_view = old_close
            if old_target is None:
                del self.observation_target
            else:
                self.observation_target = old_target
        for a in attempts:
            self.wait_view(a['screw'], a['seatedAt'], timeout=0)

    def reach_retry(self, job, skill, params, support):
        """One changed base station after an unreachable screw pick or placement."""
        try:
            return self.step(job, skill, params)
        except SkillFailed as exc:
            self.check()
            if (getattr(self, '_stance_retry_used', False) or
                    not any(word in str(exc) for word in ('out of range', 'no feasible approach', 'unreachable')) or
                    skill == 'place' and not self.rt.estimator.held_id):
                raise
            self._stance_retry_used = True
            self.warn('screw_reach_retry_station', skill=skill, detail=str(exc))
            table = self.rt.ego_state().entity(support)
            box, _, _ = entity_box(table)
            o = self.ex._state()['odometry']
            origin = math.atan2(o['y']-box.cy, o['x']-box.cx)
            self.view(job, table, origin, .65)
            return self.step(job, skill, params)

    def regrasp_upright(self, job, plan):
        """Place an elongated load vertically, then take it with a top-down grasp.

        All placement rotations use the model's FK and continuous IK path. A
        quarter roll is one solution; a tool-axis wrist can require pitch too.
        """
        from steward.ego.slender_tool import axial_feature
        support = plan['support']
        target = plan.get('target') or self.rt.estimator.held_id
        if not self.rt.estimator.held_id:
            scene = self.rt.ego_state()
            with self.rt._lock:
                track = self.rt.estimator.tracks[target]
                track.tool_feature = track.tool_feature or axial_feature(scene.entity(target), scene.stamp)
            self.step(job, 'pick', dict(target=target))
        for attempt in range(2):
            scene = self.rt.ego_state()
            held = scene.entity(self.rt.estimator.held_id)
            feature = axial_feature(held, scene.stamp)
            if feature is None:
                raise SkillFailed('no elongated geometry for stand-and-regrasp')
            # Ordinary placement supplies a free support-relative location.
            slot = self.rt.predict('place', dict(on=support, alignHoles=False))
            if not slot.get('withinRiskLimit'):
                raise SkillFailed('no reachable stand-and-regrasp placement: '+str(slot.get('issues')))
            params = dict(stageOn=support, stageXY=slot['plan']['spot'], operation='stand')
            upright = self.rt.predict('_upright_tool', params)
            if upright.get('withinRiskLimit'):
                break
            if attempt:
                raise SkillFailed('upright regrasp: '+str(upright.get('issues')))
            self.warn('upright_change_grasp', detail=upright.get('issues'))
            old = held.id
            self.step(job, 'place', dict(on=support, alignHoles=False), receipt=slot)
            self.step(job, 'pick', dict(target=old, prepareUpright=True))
        eid = self.rt.estimator.held_id
        with self.rt._lock:
            snapshot = deepcopy(self.rt.estimator.tracks[eid])
        placed = self.step(job, '_upright_tool', params, receipt=upright)
        # Cache executed geometry without turning it into a fresh camera sample.
        tip = np.asarray(placed['goal'])
        head = tip + [0., 0., placed['toolLengthM']]
        with self.rt._lock:
            track = self.rt.estimator.tracks.get(eid, snapshot)
            observed = track.tool_feature if (track.tool_feature and
                track.tool_feature.get('source') == 'rgbd_endpoint_width' and
                track.tool_feature.get('stamp', 0) > job.postcondition.get('stagingReleaseAt', math.inf)) else None
            track.pos = (tip+head)/2 if observed is None else (np.asarray(observed['tip'])+observed['head'])/2
            if observed is None:
                track.size = np.array([2*placed['toolRadiusM'], 2*placed['toolRadiusM'], placed['toolLengthM']])
                track.yaw = 0.
            track.tool_feature = observed or dict(tip=tip.tolist(), head=head.tolist(), axis=[0.,0.,-1.],
                lengthM=placed['toolLengthM'], radiusM=placed['toolRadiusM'],
                stamp=feature['stamp'], source='upright_release_estimate', polarityObserved=False)
            track.held_tool_feature = None
            self.rt.estimator.tracks[eid] = track
            self.rt.estimator.version += 1
        self.step(job, 'pick', dict(target=eid, topGrasp=True))
        job.postcondition['uprightRegrasp'] = dict(entity=self.rt.estimator.held_id,
            executed=True, verified=False, source='executed_release_and_encoder_grasp')

    def simple_drive(self, job, plan):
        pairs = [a for a in self.rt.estimator.fastening_attempts.values() if a['screw'] == plan['screw']]
        attempt = pairs[0] if pairs else deepcopy(plan.get('driveContext') or drive_context(self.rt.esim, self.rt.ego_state(), plan))
        if not pairs:
            self.warn('seating_record_estimated', screw=plan['screw'])
            self.rt.estimator.fastening_attempts[(attempt['body'],attempt['hole'])] = attempt
        self.body_id = attempt['body']
        scene = self.rt.ego_state()
        support = support_anchor(self.rt.esim, scene, scene.entity(self.body_id), []).id
        driver = self.acquire_tool(job, 'electric_screwdriver', support)
        before = deepcopy(self.feature(attempt['screw']))
        self.step(job, 'pick', dict(target=driver.id))
        params = dict(body=self.body_id, head=before['head'])
        receipt = self.rt.predict('_simple_driver', params)
        if not receipt.get('feasible', False) and any('transverse regrasp' in s for s in receipt.get('issues', [])):
            self.warn('driver_transverse_regrasp', detail=receipt['issues'])
            self.regrasp_upright(job, dict(target=self.rt.estimator.held_id, support=support))
            receipt = self.rt.predict('_simple_driver', params)
        try:
            self.step(job, '_simple_driver', params, receipt=receipt)
            after = self.ex._state()['stamp']
            attempt.update(drivenAt=after, status='driven_unverified', method='simple')
        finally:
            self.check()
            self.park(job, support)
        view = self.wait_view(attempt['screw'], after)
        observed = screw_head(view, before)
        verified = head_sank(before, observed, attempt['entry'])
        attempt.update(verified=verified, status='head_sink_observed' if verified else 'attachment_unknown',
                       evidence=dict(kind='head_sink', before=before, after=observed))
        job.postcondition.update(deepcopy(attempt), executed=True)

    def simple_stroke(self, job, plan):
        self.ex._dock(job, plan, retract=False)
        self.check()
        self.ex._set_joint('lift', plan['safeLift'])
        self.ex._set_joint('reach', plan['retractReach'])
        self.ex._set_joint('wrist_yaw', plan['wristYaw'], tol=.02)
        self.ex._set_joint('wrist_roll', plan['wristRoll'], tol=.02)
        if 'wrist_pitch' in self.ex._state()['joints']:
            self.ex._set_joint('wrist_pitch', plan['wristPitch'], tol=.02)
        self.ex._align_lateral(plan['gripSpot'], tol=.002)
        self.ex._set_joint('reach', self.ex._reach_to(plan['gripSpot']))
        self.ex._set_joint('lift', plan['lift'])
        self.ex._wait(.5)
        self.check()
        self.ex._set_joint('lift', plan['safeLift'])
        self.ex._set_joint('reach', plan['retractReach'])
        for joint, value in plan['parkWrist'].items():
            if joint in self.ex._state()['joints']:
                self.ex._set_joint(joint, value, tol=.02)
        self.ex._backoff(plan)

    def contact_error(self, plan):
        feature = self.feature(plan['object'])
        position = float(np.linalg.norm(np.asarray(feature['tip'])-plan['goal']))
        angle = math.acos(float(np.clip(np.asarray(feature['axis']) @ plan['desiredAxis'], -1, 1)))
        return position, angle

    def contact(self, job, plan):
        """Execute the exact checked approach; no unplanned IK fallback or target shifting."""
        contact_started = self.ex._state()['stamp']
        self.ex._dock(job, plan, retract=False)
        self.check()
        self.ex._set_joint('lift', plan['safeLift'])
        self.ex._set_joint('reach', plan['retractReach'])
        self.ex._set_joint('wrist_yaw', plan['wristYaw'], tol=.02)
        self.ex._set_joint('wrist_roll', plan['wristRoll'], tol=.02)
        if 'wrist_pitch' in self.ex._state()['joints']:
            self.ex._set_joint('wrist_pitch', plan['wristPitch'], tol=.02)
        self.ex._align_lateral(plan['gripSpot'], tol=.002)
        self.ex._set_joint('reach', self.ex._reach_to(plan['gripSpot']))
        from steward.ego.fasten_plan import seat_target
        from steward.ego.esim import PredictionError
        try:
            target = seat_target(self.rt.esim, self.rt.ego_state(), plan['contactParams'])
        except PredictionError as exc:
            self.warn('entry_recheck_unavailable', detail=str(exc))
            target = plan['targetEvidence']
        if np.linalg.norm(np.asarray(target['point'])-plan['targetEvidence']['point']) > .003:
            self.warn('entry_surface_shift', distanceM=float(np.linalg.norm(np.asarray(target['point'])-plan['targetEvidence']['point'])), toleranceM=.003)
        self.ex._set_joint('lift', plan['lift'])
        self.ex._set_joint('reach', self.ex._reach_to(plan['gripSpot']), ramp=False, tol=.0015)
        self.ex._set_joint('lift', plan['lift'], ramp=False, tol=.0015)
        self.ex._settle_for_release()
        position, angle = self.contact_error(plan)
        tolerance = .0025 if plan['operation'] == 'seat' else .008
        if position > tolerance or angle > math.radians(8):
            if self.feature(plan['object']).get('polarityObserved', True):
                raise SkillFailed(f'contact missed: endpoint={position:.4f} m axis={math.degrees(angle):.1f} deg; keeping tool')
            self.warn('contact_estimate_error', distanceM=position, angleDeg=math.degrees(angle), toleranceM=tolerance)
        # Direct measured body motion is a collision risk; missing registration
        # remains advisory and does not make predicted contact into evidence.
        with self.rt._lock:
            memory = self.rt.estimator.stationary_assemblies
            memory.update(self.rt.estimator.tracks, self.rt.estimator.stamp)
            for key in ('body', 'part'):
                record = memory.records.get(plan['contactParams'][key])
                if record and record.get('hardMotion'):
                    raise SkillFailed('contact registration invalidated by direct body motion')
                if record is None or record['status'] != 'frozen':
                    self.warn('contact_reference_missing', entity=plan['contactParams'][key], stamp=self.rt.estimator.stamp)
        if plan['operation'] == 'seat':
            feature = self.feature(plan['object'])
            raw = self.sample(plan['object'])
            before_release = None
            if (raw and raw['stamp'] >= contact_started and raw.get('source') == 'rgbd_segment'
                    and not raw.get('truncated', True) and raw.get('invalidDepthFraction', 1) <= .05):
                observed = directed_tool(raw['points'], 'toy_screw', raw['stamp'])
                if observed and np.linalg.norm(np.asarray(observed['tip'])-plan['goal']) <= .012:
                    before_release = observed
            self.ex._command({'gripper':'open'})
            # Save the action immediately, even if retreat or observation then fails.
            with self.rt._lock:
                p = plan['contactParams']; key = (p['body'], p['hole'])
                self.rt.estimator.fastening_attempts[key] = {
                    'body':p['body'], 'part':p['part'], 'hole':p['hole'], 'partHole':plan['targetEvidence']['partHole'],
                    'screw':plan['object'], 'entry':plan['targetEvidence']['point'],
                    'seatedAt':self.ex._state()['stamp'], 'status':'released_unverified',
                    'verified':False, 'headBeforeRelease':feature['head'],
                    'headBeforeSource':feature.get('source', 'nominal'),
                    'headObservationBeforeRelease':before_release,
                    'componentFrame':deepcopy(plan['targetEvidence'].get('componentFrame'))}
            # Use the current sensed-grasp endpoint frame, not a segmentation-old
            # fused centre left at the carrying pose.
            xy = ((np.asarray(feature['tip'])+feature['head'])/2)[:2].tolist()
            self.ex._released(plan['object'], xy)
            job.postcondition['releaseExecuted'] = True
            self.ex._wait(.4)
        else:
            # The preceding descent brings the bit just above the measured head.
            # Hold without adding an unplanned displacement or force command.
            began = self.ex._state()['stamp']
            deadline = time.monotonic()+30
            while self.ex._state()['stamp']-began < .4:
                if time.monotonic() >= deadline:
                    raise SkillFailed('contact hold timed out')
                self.check()
                distance, angle = self.contact_error(plan)
                if distance > tolerance or angle > math.radians(10):
                    if self.feature(plan['object']).get('polarityObserved', True):
                        raise SkillFailed('driver contact was not maintained; no fastening claim')
                    self.warn('driver_contact_estimate_error', distanceM=distance, angleDeg=math.degrees(angle))
                self.ex._wait(.04)
            job.postcondition['contactHeldS'] = self.ex._state()['stamp']-began
        self.ex._set_joint('lift', plan['safeLift'])
        self.ex._withdraw(plan)
        if plan['operation'] == 'seat':
            for key in ('wrist_yaw','wrist_roll','wrist_pitch'):
                if key in self.ex._state()['joints']:
                    self.ex._set_joint(key, 0., tol=.02)
        self.ex._backoff(plan)

    def seat(self, job, plan):
        if plan.get('method', 'simple') == 'simple':
            return self.simple_seat(job, plan)
        self.body_id = plan['body']
        self.reference(self.body_id)
        key = (self.body_id, plan['hole'])
        with self.rt._lock:
            previous = self.rt.estimator.fastening_attempts.get(key)
        if previous:
            self.warn('reuse_seating_attempt', body=self.body_id, hole=plan['hole'])
            self.confirm_seat(job, previous, plan)
            return
        scene = self.rt.ego_state()
        support = support_anchor(self.rt.esim, scene, scene.entity(self.body_id), job.postcondition.setdefault('warnings', [])).id
        screw = self.acquire_tool(job, 'toy_screw', support, free_screw=True)
        self.step(job, 'pick', {'target':screw.id})
        held = self.rt.estimator.held_id
        if held != screw.id:
            self.warn('screw_alias_changed', requested=screw.id, held=held, stamp=self.rt.estimator.stamp)
        self.reference(self.body_id)
        self.step(job, '_fastener_contact', {**plan, 'screw':held, 'operation':'seat'})
        attempt = self.rt.estimator.fastening_attempts[key]
        job.postcondition.update(deepcopy(attempt), releaseExecuted=True)
        view = self.wait_view(screw.id, attempt['seatedAt'])
        from steward.ego.fasten_simple import landing_error
        scene = self.rt.ego_state()
        attempt['landingErrors'] = [landing_error(scene.find(screw.id), attempt['entry'], scene.stamp, attempt['seatedAt'])]
        job.postcondition['landingErrors'] = deepcopy(attempt['landingErrors'])
        feature = self.feature(screw.id)
        seated = bool(view and feature.get('source') == 'rgbd_endpoint_width'
                      and feature['stamp'] >= view['stamp'] > attempt['seatedAt']
                      and np.linalg.norm(np.asarray(feature['tip'])-attempt['entry']) <= .012
                      and feature['axis'][2] >= math.cos(math.radians(10)))
        # Seat mode may already have buried the shaft before a complete tool view.
        # Head motion is independent evidence, regardless of whether drive ran.
        before = attempt.get('headObservationBeforeRelease') or dict(
            head=attempt['headBeforeRelease'], stamp=attempt['seatedAt'],
            source=attempt.get('headBeforeSource', 'nominal'))
        observed = screw_head(view, before)
        attached = head_sank(before, observed, attempt['entry'])
        with self.rt._lock:
            attempt.update(status='head_sink_observed' if attached else 'seated_observed' if seated else 'released_unverified',
                           verified=attached, seatedObserved=seated, beforeDrive=deepcopy(feature))
        if not seated and not attached:
            self.warn('seating_unverified', screw=screw.id)
        self.confirm_seat(job, attempt, plan, observe=False)

    def drive(self, job, plan):
        if plan.get('method', 'simple') == 'simple':
            return self.simple_drive(job, plan)
        with self.rt._lock:
            attempts = self.rt.estimator.fastening_attempts
            pairs = [(k,a) for k,a in attempts.items() if a['screw'] == plan['screw']]
            if not pairs:
                context = plan.get('driveContext') or drive_context(self.rt.esim, self.rt.ego_state(), plan)
                key = (context['body'], context['hole'])
                attempts[key] = deepcopy(context)
                pairs = [(key, attempts[key])]
                self.warn('seating_record_estimated', screw=plan['screw'])
            key, attempt = pairs[0]
            if attempt.get('verified'):
                job.postcondition.update(deepcopy(attempt)); return
            params = {k:attempt[k] for k in ('body','part','hole','screw')}
        self.body_id = params['body']; self.reference(self.body_id)
        scene = self.rt.ego_state()
        support = support_anchor(self.rt.esim, scene, scene.entity(self.body_id), job.postcondition.setdefault('warnings', [])).id
        driver = self.acquire_tool(job, 'electric_screwdriver', support)
        before = deepcopy(self.feature(params['screw']))
        self.step(job, 'pick', {'target':driver.id, 'flip':True, 'prepareDriver':True})
        if self.rt.estimator.held_id != driver.id:
            self.warn('driver_alias_changed', requested=driver.id, held=self.rt.estimator.held_id, stamp=self.rt.estimator.stamp)
        self.step(job, '_fastener_contact', {**params, 'operation':'drive'})
        after = self.ex._state()['stamp']
        with self.rt._lock:
            attempt.update(status='driven_unverified', drivenAt=after)
        self.step(job, '_fastener_reset', {})
        self.park(job, support)
        view = self.wait_view(params['screw'], after)
        observed = screw_head(view, before)
        verified = head_sank(before, observed, attempt['entry'])
        with self.rt._lock:
            attempt.update(verified=verified, status='head_sink_observed' if verified else 'attachment_unknown',
                           evidence={'kind':'head_sink', 'before':before, 'after':observed})
        job.postcondition.update(deepcopy(attempt))
        job.postcondition['executed'] = True
        if not verified:
            self.warn('attachment_unverified_after_drive', screw=params['screw'])

    def recover_component(self, job, plan, part, source):
        with self.rt._lock:
            records=self.rt.estimator.stationary_assemblies.records
            records.pop(part,None)
            records.pop(source['entity'],None)
            records[plan['body']]['parts'][:]=[p for p in records[plan['body']]['parts'] if p['entity']!=part]
            track=self.rt.estimator.tracks.get(source['entity'])
            if track is not None:
                track.pos=np.asarray(source['position'],dtype=float)
        if plan.get('recoverDropped') and source.get('mountingFaceUp') is False:
            from steward.ego.fasten_simple import offset_on
            scene=self.rt.ego_state(); table=support_anchor(self.rt.esim,scene,scene.entity(plan['body']),[])
            self.warn('restore_component_face_on_table',entity=source['entity'])
            self.step(job,'pick',dict(target=source['entity'],flip=True))
            self.step(job,'place',dict(on=table.id,offset=offset_on(table,source['position'][:2]),
                                       flip=True,alignHoles=False,closeRelease=True))

    def fasten(self, job, plan):
        self.body_id = plan['body']
        if not hasattr(self, '_seat_retry_budget'):
            self._seat_retry_budget = [2]
        job.postcondition['confirmedSeatedScrews'] = 0
        scene = self.rt.ego_state()
        holes = mating_holes(self.rt.esim, scene, plan)
        selected_ids = {h['id'] for h in holes}
        job.postcondition['selectedHoleIds'] = sorted(selected_ids)
        job.postcondition['requestedScrews'] = len(selected_ids)
        # Staging precedes placing the component, including all long source trips.
        with self.rt._lock:
            self._tool_sources = {t.id:deepcopy(t) for t in self.rt.estimator.tracks.values()
                if not t.held and tool_estimate(scene.find(t.id), scene.stamp)}
        source_track = deepcopy(self.rt.estimator.tracks.get(plan['part']))
        staged = self.stage_screws(job, plan, holes) if plan.get('stageScrews') else []
        if source_track is not None and (plan['part'] not in self.rt.estimator.tracks or
                self.rt.estimator.tracks[plan['part']].category != source_track.category):
            self.rt.estimator.tracks[plan['part']] = source_track
            self.warn('component_last_source_restored', entity=plan['part'])
        quick = plan.get('method', 'simple') == 'simple'
        params = {k:plan[k] for k in ('part','body','hole','heldHole') if k in plan}
        params['hole'] = sorted(selected_ids)
        if plan.get('selectedHeldHoleIds'):
            params['heldHole'] = plan['selectedHeldHoleIds']
        params.update(deferVerify=quick, closeRelease=bool(plan.get('closeRelease')))
        placement_retries = max(0, 2-plan.get('_dropRetries',0)) if plan.get('recoverDropped') else 1
        pickup_retry = False
        for retry in range(placement_retries+1):
            try:
                self.step(job, 'assemble_onto', params)
            except SkillFailed as exc:
                self.check()
                assembly = job.postcondition['stages'][-1].get('result') or {}
                if not assembly.get('releaseExecuted'):
                    source = plan.get('partSource', {})
                    with self.rt._lock:
                        fresh = [t for t in self.rt.estimator.tracks.values()
                            if source and self.matches(t, source) and t.id != params['part']
                            and t.last_seen >= (source_track.last_seen if source_track else 0.)]
                    if (not pickup_retry and not self.rt.estimator.held_id and fresh
                            and 'grasp missed' in str(exc)):
                        replacement = max(fresh, key=lambda t:t.last_seen)
                        pickup_retry = True
                        params['part'] = replacement.id
                        self.warn('component_pick_retry_fresh_source', entity=replacement.id)
                        self.step(job, 'assemble_onto', params)
                    elif (plan.get('wingRetry') and not pickup_retry and not self.rt.estimator.held_id
                            and any(x in str(exc) for x in ('grasp missed','out of range','no feasible approach'))):
                        pickup_retry = True
                        self.warn('component_pick_retry_station', detail=str(exc))
                        table=self.rt.ego_state().entity(plan['support'])
                        box,_,_=entity_box(table); o=self.ex._state()['odometry']
                        self.view(job,table,math.atan2(o['y']-box.cy,o['x']-box.cx),.65)
                        self.step(job,'assemble_onto',params)
                    else:
                        raise
                else:
                    self.warn('assembly_postcheck_failed_continue', detail=str(exc))
            assembly = job.postcondition['stages'][-1]['result']
            check = assembly.get('placementCheck', {})
            if check.get('passed') is not False or not check.get('retrySource') or retry == placement_retries:
                break
            source = check['retrySource']
            if not plan.get('recoverDropped') and source.get('sourceDistanceM',0) > .04:
                break
            self.warn('placement_retry_from_observed_source', attempt=retry+1, evidence=deepcopy(check))
            job.postcondition.setdefault('previousAssemblies', []).append(deepcopy(assembly))
            self.recover_component(job, plan, assembly['part'], source)
            params['part']=source['entity']
        job.postcondition['placementCheck'] = deepcopy(check)
        job.postcondition.update(releaseExecuted=assembly.get('releaseExecuted',False), assembly=assembly)
        if check.get('passed') is False:
            count = 'one retry' if placement_retries == 1 else f'{placement_retries} retries'
            raise SkillFailed(f'component placement contradicted by fresh RGB-D after {count}')
        part = assembly['part']; body = plan['body']
        released_at = assembly.get('releasedAt')
        if released_at is not None:
            with self.rt._lock:
                old = self.rt.estimator.fastening_attempts
                stale = [k for k,a in old.items() if a['body'] == body and
                         a['hole'] in selected_ids and a['seatedAt'] < released_at]
                for k in stale:
                    job.postcondition.setdefault('supersededAttempts', []).append(deepcopy(old.pop(k)))
        scene = self.rt.ego_state(); base = scene.find(body)
        # Retain the pre-placement interface selection even if the body's mask
        # changes category or loses its hole catalogue after adding the part.
        failures = []
        for hole_index, hole in enumerate(holes):
            key = (body, hole['id'])
            for skill in ('seat_screw', 'drive_screw'):
                attempt = self.rt.estimator.fastening_attempts.get(key)
                if skill == 'drive_screw' and (not plan.get('drive', True) or not attempt or attempt.get('verified')):
                    continue
                params = {'part':part, 'body':body, 'hole':hole['id']} if skill == 'seat_screw' else {'screw':attempt['screw']}
                try:
                    params.update(method=plan.get('method', 'simple'), correctOnce=bool(plan.get('correctOnce', True)))
                    if skill == 'seat_screw':
                        params.update(deferVerify=quick, closeRelease=bool(plan.get('closeRelease')))
                        if hole_index < len(staged):
                            row=staged[hole_index]
                            with self.rt._lock:
                                nearby=[t for t in self.rt.estimator.tracks.values() if t.category=='toy_screw'
                                    and not t.held and t.last_seen>row['stagedAt']
                                    and np.linalg.norm(t.pos[:2]-row['xy'])<.025]
                                t=min(nearby,key=lambda t:np.linalg.norm(t.pos[:2]-row['xy'])) if nearby else deepcopy(row['track'])
                                self.rt.estimator.tracks[t.id]=t
                            params['stagedScrew']=t.id
                        params['excludedScrews'] = [r['entity'] for r in staged if r['entity'] != params.get('stagedScrew')]
                        params['excludedScrews'] += list(getattr(self,'_retired_screws',set()))
                        params['excludedPositions'] = [r['xy'] for i,r in enumerate(staged) if i != hole_index]
                    self.step(job, skill, params)
                except SkillFailed as exc:
                    self.check()  # cancellation/session reset is never a recoverable screw failure
                    failures.append(dict(hole=hole['id'], skill=skill, failure=str(exc)))
                    self.warn('fastener_attempt_failed', **failures[-1])
                    # A failed contact can leave a screw/driver in hand. Free the
                    # hand through its own checked receipt before the next attempt.
                    if self.rt.estimator.held_id:
                        scene = self.rt.ego_state()
                        support = support_anchor(self.rt.esim, scene, scene.entity(body), job.postcondition.setdefault('warnings', [])).id
                        if plan.get('method', 'simple') == 'contact' and self.rt.estimator.tracks[self.rt.estimator.held_id].category == 'electric_screwdriver':
                            try:
                                self.step(job, '_fastener_reset', {})
                            except SkillFailed as reset_error:
                                self.check()
                                self.warn('driver_reset_failed', detail=str(reset_error))
                        self.park(job, support)
        attempts = [a for a in self.rt.estimator.fastening_attempts.values() if a['body'] == body and a['part'] == part]
        if quick:
            # Both holes were attempted before any optional wait or correction.
            from steward.ego.fasten_simple import landing_error
            for a in attempts:
                scene=self.rt.ego_state()
                a['landingErrors']=[landing_error(scene.find(a['screw']),a['entry'],scene.stamp,a['seatedAt'])]
            released_at=assembly.get('releasedAt')
            if released_at is not None and attempts:
                job.postcondition['releaseToFirstScrewS']=min(a['seatedAt'] for a in attempts)-released_at
            # Now take the deferred view, and report the same independent position check.
            if attempts:
                self.close_seat_view(job, attempts)
            scene = self.rt.ego_state()
            for a in attempts:
                a['landingErrors'].append(landing_error(scene.find(a['screw']),a['entry'],scene.stamp,a['seatedAt']))
            from steward.ego.assembly_motion import placement_check
            record=self.rt.estimator.stationary_assemblies.records.get(body,{})
            expected=next((p for p in record.get('parts',[]) if p['entity']==part),None)
            if expected and released_at is not None:
                check=placement_check(self.rt.estimator.tracks.values(),plan['partSource'],expected,
                                      released_at,self.rt.estimator.stamp)
                job.postcondition['placementCheck']=check
        if quick and check.get('passed') is False:
            used=plan.get('_dropRetries',0)+retry
            source=check.get('retrySource')
            limit = 2 if plan.get('recoverDropped') else 1
            retry_source = source and (plan.get('recoverDropped') or source.get('sourceDistanceM', 0) <= .04)
            if retry_source and used < limit:
                job.postcondition.setdefault('discardedAttempts',[]).extend(deepcopy(attempts))
                self._retired_screws=getattr(self,'_retired_screws',set()) | {a['screw'] for a in attempts}
                with self.rt._lock:
                    for a in attempts:
                        self.rt.estimator.fastening_attempts.pop((body,a['hole']),None)
                self.recover_component(job,plan,part,source)
                next_plan={**plan,'part':source['entity'],'_dropRetries':used+1,
                           'partSource':dict(id=source['entity'],category=plan['partSource']['category'],position=source['position'])}
                job.postcondition['dropRecoveryCount']=used+1
                return self.fasten(job,next_plan)
            raise SkillFailed('component placement contradicted after one retry; recovery budget exhausted or source unknown')
        # The first pass still batches the holes. A shared posterior view then
        # triggers at most two replays in this same job, independent of g8 flags.
        if quick:
            for a in attempts:
                root = job.postcondition
                job.postcondition = {}
                try:
                    self.confirm_seat(job, a, dict(plan, body=body, part=part, hole=a['hole']), observe=False)
                except SkillFailed as exc:
                    self.check()
                    failures.append(dict(hole=a['hole'], skill='seat_screw', failure=str(exc)))
                finally:
                    root.setdefault('seatPostchecks', []).append(deepcopy(job.postcondition))
                    job.postcondition = root
                for stage in job.postcondition['stages']:
                    if stage['skill'] == 'seat_screw' and stage.get('receipt', {}).get('params', {}).get('hole') == a['hole']:
                        stage['result'] = deepcopy(a)
                        if not a.get('seatConfirmed'):
                            stage['failure'] = 'landing '+a.get('seatCheck', {}).get('status', 'unknown')
        confirmed = sum(bool(a.get('seatConfirmed')) for a in attempts)
        job.postcondition.update(verified=any(a.get('verified') for a in attempts), executed=True,
            part=part, body=body, attempts=deepcopy(attempts), partialFailures=failures,
            confirmedSeatedScrews=confirmed, requestedScrews=len(selected_ids),
            seatRetriesUsed=2-self._seat_retry_budget[0],
            reason='landing count uses independent screw geometry; attachment remains separate')
        if confirmed != len(selected_ids):
            raise SkillFailed(f'confirmed screw landings {confirmed}/{len(selected_ids)}; failed or unknown posterior')

    def flip(self, job, plan):
        self.body_id = plan['body']
        record = self.reference(self.body_id)
        if plan.get('levelBelly'):
            from steward.ego.assembly_quick import level_assessment
            job.postcondition['levelBelly'] = level_assessment(record)
            self.warn('level_belly_assessment', assessment=job.postcondition['levelBelly'])
        from steward.ego.assembly_motion import restore_track, released_record, front_spots, local_bounds
        with self.rt._lock:
            body_track = self.rt.estimator.tracks.get(self.body_id)
            if body_track is None:
                body_track = restore_track(self.rt.estimator, record)
            body_track.pos = np.array(record['reference']['position'])
            body_track.size = np.array(record['reference']['size'])
            body_track.yaw = record['reference'].get('yaw', body_track.yaw)
        support_track = deepcopy(self.rt.estimator.tracks.get(record['reference'].get('support')))
        members = record['parts']
        turn_over = plan.get('turnOver', True)
        attempts = self.rt.estimator.fastening_attempts
        from steward.ego.fasten_release import loose_components
        loose = loose_components(self.rt.ego_state(), self.body_id)
        if loose:
            self.warn('loose_components_before_flip', parts=[p['entity'] for p in loose],
                      confirmedSeatedScrews=0, advisory=True)
        if not members or any(not any(a['part'] == p['entity'] and a['body'] == self.body_id and a.get('verified')
                                      for a in attempts.values()) for p in members):
            self.warn('attachment_unverified_before_flip', members=len(members), verifiedAttempts=sum(bool(a.get('verified')) for a in attempts.values()))
        # Conservative circumsphere includes every known released component through
        # rotation, also for pick/navigation. It does not create a connection belief.
        root = np.asarray(record['reference']['position'])
        sphere = max([np.linalg.norm(np.asarray(record['reference']['size'])/2)] +
            [np.linalg.norm(np.asarray(p['position'])-root)+np.linalg.norm(np.asarray(p['size'])/2) for p in members])
        before = {p['entity']:self.sample(p['entity']) for p in members}
        body_before = self.sample(self.body_id)
        now = self.ex._state()['stamp']
        if (not body_before or not all(before.values()) or
                any(now-v['stamp'] > 20 for v in [body_before, *before.values()] if v)):
            self.warn('flip_previews_incomplete', bodyAvailable=bool(body_before), availableMembers=sum(bool(v) for v in before.values()), stamp=now)
        def independent(v):
            return v if (v and 0 <= now-v['stamp'] <= 20 and not v.get('truncated', True)
                         and v.get('invalidDepthFraction', 1) <= .05) else None
        body_before = independent(body_before)
        before = {eid:independent(v) for eid,v in before.items()}
        with self.rt._lock:
            self.rt.estimator.manipulation_envelopes[self.body_id] = float(2*sphere)
        try:
            self.step(job, 'pick', {'target':self.body_id, 'flip':turn_over, 'assemblyGrip':True})
            if self.rt.estimator.held_id != self.body_id:
                self.warn('body_alias_changed_at_grasp', requested=self.body_id, held=self.rt.estimator.held_id, stamp=self.rt.estimator.stamp)
            after = self.ex._state()['stamp']; observations = []
            for _ in range(2):
                body_view = self.wait_view(self.body_id, after)
                row = {}
                for p in members:
                    row[p['entity']] = self.wait_view(p['entity'], after)
                observations.append((body_view,row))
                after = max([after] + [v['stamp'] for v in [body_view, *row.values()] if v])
            follow = {p['entity']:bool(body_before and before[p['entity']] and all(b and rows[p['entity']] for b,rows in observations)) and follows([body_before, before[p['entity']]],
                [[b,rows[p['entity']]] for b,rows in observations]) for p in members}
            job.postcondition['followEvidence'] = follow
            job.postcondition['followSamples'] = [
                [{k:v for k,v in (b or {}).items() if k != 'points'},
                 {eid:{k:v for k,v in (row or {}).items() if k != 'points'} for eid,row in rows.items()}]
                for b,rows in observations]
            if not all(follow.values()):
                self.warn('following_unverified', observedMembers=sum(follow.values()), totalMembers=len(members))
            with self.rt._lock:
                est = self.rt.estimator
                gr = est.gripper_rotation(est.held_gripper)
                gp = est.gripper_position(est.held_gripper)
                # Keep component identities and encoder-predicted poses while carried.
                # Independent raw RGB-D samples remain untouched for contradiction checks.
                for eid, raw in observations[-1][1].items():
                    if not raw or not follow[eid]:
                        continue
                    est.rigid_carry[eid] = {'offset':gr.T @ (np.asarray(raw['centreM'])-gp),
                        'rotation':gr.T @ matrix_from_quat(quat_from_yaw(raw['boxYawRad'])),
                        'observedAt':raw['stamp']}
            support = record['reference']['support']
            # The original body frame is now invalid. Keep it only as a pre-motion
            # hypothesis and re-register from new views after the executed flip.
            with self.rt._lock:
                est = self.rt.estimator
                rot0 = est.gripper_rotation(est.held_gripper).copy()
                est.flip_capture = {'body':self.body_id, 'rotation':rot0, 'record':deepcopy(record)}
            scene = self.rt.ego_state()
            table = scene.find(support) or record['support']
            low, high = local_bounds(record)
            half = float(max(abs(low[:2]).max(), abs(high[:2]).max()))
            last = None
            from steward.ego.fasten_simple import offset_on
            tables = [table] + [e for e in scene.entities if e.id != table.id and
                               self.rt.esim._affords(e, 'support-surface')]
            candidates = []
            for surface in tables:
                offsets = front_spots(self.rt.esim, scene, surface, half)
                if surface.id == support:
                    offsets.insert(0, offset_on(surface, root[:2]))
                for offset in [*offsets, None]:
                    candidates.append(dict(on=surface.id, flip=turn_over, alignHoles=False,
                        assemblyBody=self.body_id, **({'offset':offset} if offset is not None else {})))
            for candidate in candidates:
                receipt = self.rt.predict('place', candidate)
                if not receipt.get('withinRiskLimit'):
                    last = '; '.join(receipt.get('issues', []))
                    self.warn('assembly_place_candidate_failed', on=candidate['on'], offset=candidate.get('offset'), detail=last)
                    continue
                place_plan = self.step(job, 'place', candidate, receipt=receipt)
                support = candidate['on']
                table = next(t for t in tables if t.id == support)
                break
            else:
                raise SkillFailed('assembly placement has no reachable table candidate: '+str(last))
            with self.rt._lock:
                captured = self.rt.estimator.flip_capture
                self.rt.estimator.flip_capture = None
            if not captured or 'releaseRotation' not in captured:
                self.warn('flip_release_frame_estimated')
                axis = matrix_from_quat(quat_from_yaw(record['reference']['yaw']))
                delta = axis @ np.diag([1.,-1.,-1.]) @ axis.T if turn_over else np.eye(3)
                after = self.ex._state()['stamp']
            else:
                delta = captured['releaseRotation'] @ rot0.T
                after = captured['releasedAt']
            fresh = self.wait_view(self.body_id, after)
            orientation_observed = False
            if turn_over and (delta @ np.array([0.,0.,1.]))[2] > -math.cos(math.radians(10)):
                second = self.wait_view(self.body_id, fresh['stamp'] if fresh else after)
                settled = settled_flip_rotation(delta, body_before, [fresh,second]) if body_before and fresh and second else None
                if settled is None:
                    self.warn('flip_face_unverified', releaseUpZ=float(delta[2,2]))
                    # Nominal completed half turn: preserve the observed horizontal
                    # axis as best possible. This transports priors, not evidence.
                    u, _, vt = np.linalg.svd(delta[:2,:2])
                    xy = u @ vt
                    if np.linalg.det(xy) > 0:
                        u[:,-1] *= -1
                        xy = u @ vt
                    delta = np.zeros((3,3)); delta[:2,:2] = xy; delta[2,2] = -1
                else:
                    delta = settled; fresh = second; orientation_observed = True
                    job.postcondition['faceReversalObserved'] = True
            # Rebase directly from the completed release. Old pre-flip tracks and
            # registrations cannot supply a stationary reference for the next face.
            scene = self.rt.ego_state()
            table = scene.find(support) or table
            new = released_record(record, place_plan['spot'], table, delta, after, fresh)
            new_ref = new['reference']
            # One cheap look from the release retreat. Neither a half-turn command
            # nor propagated tracks count as a visually completed flip.
            self.observation_target = new_ref['position']
            self.look(after, cycles=1)
            with self.rt._lock:
                fresh_bodies = []
                for t in self.rt.estimator.tracks.values():
                    v = t.manipulation_view
                    if (not v or t.held or t.category not in (record['reference']['category'], record['reference'].get('family'))
                            or v['stamp'] <= after or self.rt.estimator.stamp-v['stamp'] > 20
                            or v.get('truncated', True) or v.get('invalidDepthFraction', 1) > .05
                            or not np.allclose(np.sort(v['visibleSizeM'][:2]),
                                               np.sort(record['reference']['size'][:2]), atol=.025)):
                        continue
                    d = min(np.linalg.norm(np.asarray(v['centreM'])[:2]-root[:2]),
                            np.linalg.norm(np.asarray(v['centreM'])[:2]-new_ref['position'][:2]))
                    if d <= .10:
                        fresh_bodies.append(deepcopy(v))
            fresh = fresh_bodies[0] if len(fresh_bodies) == 1 else None
            from steward.ego.assembly_motion import flip_check
            check = flip_check(body_before, fresh, before,
                {p['entity']:self.sample(p['entity']) for p in members}, after, self.rt.estimator.stamp)
            if (fresh and np.linalg.norm(root[:2]-new_ref['position'][:2]) > .10 and
                    np.linalg.norm(np.asarray(fresh['centreM'])[:2]-root[:2]) < .04):
                check = dict(passed=False, status='fail', reason='body_still_at_pickup',
                             observedPositionM=fresh['centreM'], expectedPositionM=new_ref['position'])
            job.postcondition['orientationCheck'] = check
            orientation_observed = check['passed'] is True
            if check['passed'] is False and fresh:
                # A measured unchanged face invalidates the commanded turn
                # hypothesis. Keep the observed location and the previous face.
                landing_table = record['support'] if check['reason'] == 'body_still_at_pickup' else table
                new = released_record(record, fresh['centreM'][:2], landing_table, np.eye(3), after, fresh)
                new_ref = new['reference']
            with self.rt._lock:
                est = self.rt.estimator
                for eid in {self.body_id, *(p['entity'] for p in members)}:
                    est.stationary_assemblies.records.pop(eid, None)
                    est.assembly_memory.invalidate(eid)
                if self.body_id not in est.tracks:
                    self.warn('body_track_restored_after_flip', entity=self.body_id)
                t = restore_track(est, new)
                t.held = False
                t.pos = np.asarray(new_ref['position'])
                t.yaw = new_ref['yaw']
                t.size = np.asarray(new_ref['size'])
                t.interface_views.clear(); t.interface_best_near = None
                est.stationary_assemblies.records[self.body_id] = new
                if support == record['reference']['support'] and support not in est.tracks and support_track is not None:
                    est.tracks[support] = support_track
                for a in attempts.values():
                    if a['body'] == self.body_id:
                        a['entry'] = (np.asarray(new_ref['position'])+delta @ (np.asarray(a['entry'])-root)).tolist()
                        a['transportedAt'] = after
                job.postcondition.update(verified=(orientation_observed if turn_over else bool(members) and all(follow.values())),
                    executed=True, releaseExecuted=True, body=self.body_id, reference=deepcopy(new_ref),
                    flipped=check['passed'] if turn_over else False, commandedTurnOver=turn_over,
                    orientationSource='commanded_relative_half_turn' if check['passed'] is not False else 'observed_unchanged_face', upright=None,
                    passed=check['passed'] if turn_over else None)
                # Independently observed following also confirms seat-mode attachment.
                for a in attempts.values():
                    if a['body'] == self.body_id and follow.get(a['part']):
                        a.update(verified=True, status='following_observed')
        finally:
            with self.rt._lock:
                # While still held, retain the full load envelope for safe recovery.
                if self.rt.estimator.held_id != self.body_id:
                    self.rt.estimator.manipulation_envelopes.pop(self.body_id, None)
                self.rt.estimator.flip_capture = None

    def run(self, job, plan):
        skill = plan['fastenSkill']
        self._observation_job = job
        job.postcondition = {'skill':skill, 'verified':False, 'stages':[]}
        try:
            {'seat_screw':self.seat, 'drive_screw':self.drive,
             'fasten_onto':self.fasten, 'flip_assembly':self.flip, 'regrasp_upright':self.regrasp_upright}[skill](job,plan)
        except Exception as exc:
            job.postcondition['failure'] = str(exc)
            with self.rt._lock:
                attempts = [a for a in self.rt.estimator.fastening_attempts.values()
                            if not plan.get('body') or a['body'] == plan['body']]
                job.postcondition['attempts'] = deepcopy(attempts)
                if skill == 'seat_screw':
                    selected = set(job.postcondition.get('selectedHoleIds', [plan.get('hole')]))
                    job.postcondition['confirmedSeatedScrews'] = sum(
                        bool(a.get('seatConfirmed')) for a in attempts if a['hole'] in selected)
            raise
        job.message = f'{skill}: see sensory evidence and unresolved limits in postcondition'
