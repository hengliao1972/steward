"""Deterministic observe/register/pick/fit/place/verify controller over Ego only.

Runs inside ONE cancellable SkillExecutor job. It never starts a nested job or
reads task definitions. Each actuator stage consumes a fresh bounded-risk receipt.
"""
from copy import deepcopy
import math
import json
import time

import numpy as np

from steward.ego.hole_registration import multiview_interface_box, assembly_posture, best_interface_box
from steward.ego.classify import SHAPES
from steward.ego.priors import screw_holes, SCREW_HOLE_FRAMES
from steward.ego.planner import entity_box
from steward.ego.skills import SkillFailed, SkillStopped
from steward.math3d import quat_from_yaw, matrix_from_quat
from steward.scene.schema import Est


class AssembleOnto:
    def __init__(self, runtime):
        self.rt = runtime
        self.ex = runtime.executor
        self.generation = runtime._generation

    def warn(self, reason, **numbers):
        row = dict(reason=reason, **numbers)
        job = getattr(self, '_observation_job', None)
        if job is not None:
            if job.postcondition is None:
                job.postcondition = {}
            warnings = job.postcondition.setdefault('warnings', [])
            if row not in warnings:
                warnings.append(row)
        return row

    def progress(self, job, stage, ok, reason, *, result=None):
        job._progress_sequence = getattr(job, '_progress_sequence', 0)+1
        row = dict(id=job._progress_sequence, stage=stage, ok=bool(ok), reason=str(reason),
                   wallTime=time.time())
        result = job.postcondition if result is None else result
        result.setdefault('progress', []).append(row)
        job.log.append('stage ' + json.dumps(row, sort_keys=True))

    def check(self):
        if self.ex._stop.is_set():
            raise SkillStopped()
        if self.generation != self.rt._generation:
            raise SkillFailed('sensor session changed during assembly')

    def step(self, job, skill, params, *, receipt=None):
        started = time.monotonic()
        root = job.postcondition
        try:
            return self._step(job, skill, params, receipt=receipt)
        except Exception as exc:
            self.progress(job, skill, False, str(exc), result=root)
            raise
        finally:
            if root.get('motionPlans'):
                root['motionPlans'][-1]['wallS'] = time.monotonic()-started

    def _step(self, job, skill, params, *, receipt=None):
        root = job.postcondition
        self._observation_job = job
        self.check()
        started = time.monotonic()
        job.log.append('predict '+skill)
        receipt = receipt or self.rt.predict(skill, params)
        root.setdefault('motionPlans', []).append(dict(skill=skill, params=deepcopy(params),
            wallStart=time.time()-(time.monotonic()-started), predictWallS=time.monotonic()-started,
            plan=deepcopy(receipt.get('plan', {})), withinRiskLimit=receipt.get('withinRiskLimit')))
        for warning in receipt.get('warnings', []):
            self.warn(**warning)
        if not receipt.get('withinRiskLimit'):
            raise SkillFailed(f'{skill}: ' + '; '.join(receipt['issues']))
        try:
            plan = self.rt._consume_receipt(receipt)
        except PermissionError as exc:
            raise SkillFailed(str(exc)) from exc
        job.log.append(f"assembly stage {skill}: {receipt['id']}")
        self.ex._begin(job, skill, receipt)
        self.check()
        if skill == 'navigate':
            self.ex._drive(job, plan['path'], plan['yaw'])
        elif skill == 'pick':
            self.ex._pick(job, plan)
        elif skill == 'place':
            # Check again immediately before the irreversible release sequence.
            self.reference(self.body_id)
            self.ex._place(job, plan)
        self.progress(job, skill, True, 'executed with bounded-risk receipt', result=root)
        return plan

    def reference(self, body_id):
        self.check()
        with self.rt._lock:
            memory = self.rt.estimator.stationary_assemblies
            memory.update(self.rt.estimator.tracks, self.rt.estimator.stamp)
            r = memory.records.get(body_id)
            if r is None and body_id in self.rt.estimator.tracks:
                track = deepcopy(self.rt.estimator.tracks[body_id])
                scene = self.rt.ego_state()
                support = self.rt.esim._anchor(scene, scene.entity(body_id))
                job = getattr(self, '_observation_job', None)
                if job is not None:
                    plan = dict(body=body_id, bodySource=dict(id=body_id, category=track.category, position=track.pos.tolist()))
                    self.warn('stationary_reference_rebuilt', entity=body_id, ageS=self.rt.estimator.stamp-track.last_seen)
                    category = 'airplane_fuselage' if track.category == 'toy_airplane' else track.category
                    self._store_registration(job, plan, support, category, track, self.best_estimate(track))
                    r = memory.records[body_id]
            if r is not None and r['status'] != 'frozen':
                self.warn('stationary_reference_invalidated_continue', entity=body_id, detail=r.get('reason'))
                r.update(status='frozen', hardMotion=False, motionSamples={})
            if r is None or r['status'] != 'frozen':
                raise SkillFailed('stationary reference unavailable or invalidated; ' + json.dumps(
                    {'body': body_id, 'reason': r.get('reason') if r else 'missing',
                     'aliases': r.get('aliases') if r else [], 'stamp': self.rt.estimator.stamp}))
            for warning in r.get('warnings', []):
                self.warn(**warning)
            return deepcopy(r)

    def observation_diagnostics(self, source):
        """Sensor numbers only; absence of a sample is not proof of occlusion."""
        with self.rt._lock:
            est = self.rt.estimator
            candidates = [t for t in est.tracks.values()
                          if source and (t.id == source['id'] or
                              np.linalg.norm(t.pos-np.asarray(source['position'])) <= .15)]
            samples = [v for t in candidates if t.id == source['id'] or self.matches(t, source)
                       for v in t.interface_views]
            newest = max(samples, key=lambda v: v['stamp'], default=None)
            return dict(stamp=est.stamp, requestedTrackId=source['id'] if source else None,
                matchedTrackIds=[t.id for t in candidates if self.matches(t, source)],
                trackIdChanges=[t.id for t in candidates if t.id != source['id'] and self.matches(t, source)],
                nearbyTracks={t.id: dict(category=t.category, held=t.held, existence=t.existence,
                    distanceToSourceM=float(np.linalg.norm(t.pos-np.asarray(source['position']))),
                    matches=bool(self.matches(t, source))) for t in candidates},
                latestSampleAgeS=est.stamp-newest['stamp'] if newest else None,
                latestSampleRangeM=newest['range'] if newest else None,
                sampler={t.id: deepcopy(t.interface_sample_status) for t in candidates})

    def look(self, since, source=None, *, cycles=3):
        """Wait for N completed imaged instants after arrival, not N body polls.

        Wall time is only a no-progress watchdog, reset by every imaged cycle.
        Slow but progressing perception gets its full frame budget.
        """
        with self.rt._lock:
            start_cycle = self.rt.estimator.perception_cycles
        seen, statuses, rejects, track_ids = set(), set(), {}, set()
        started = progressed = time.monotonic()
        last_cycle = start_cycle
        result = dict(since=since, requestedCycles=cycles, waitedCycles=0)
        job = getattr(self, '_observation_job', None)
        if job is not None:
            job.postcondition.setdefault('observationWaits', []).append(result)
        while True:
            self.check()
            with self.rt._lock:
                est = self.rt.estimator
                for seq, stamp in est.perception_stamps:
                    if seq >= start_cycle and stamp > since:
                        seen.add(seq)
                if est.perception_cycles != last_cycle:
                    progressed = time.monotonic()
                    last_cycle = est.perception_cycles
                tracks = [t for t in est.tracks.values() if source is None or self.matches(t, source)]
                fresh = any((t.assembly_observation and t.assembly_observation['stamp'] > since)
                            if source is None else any(v['stamp'] > since for v in t.interface_views)
                            for t in tracks)
                for t in tracks:
                    status = t.interface_sample_status
                    key = (t.id, status.get('stamp'))
                    if status.get('stamp', -math.inf) > since and key not in statuses:
                        statuses.add(key)
                        for reason, count in status.get('rejected', {}).items():
                            rejects[reason] = rejects.get(reason, 0)+count
                        outcome = status.get('outcome', 'unknown')
                        rejects[outcome] = rejects.get(outcome, 0)+1
                result.update(self.observation_diagnostics(source), waitedCycles=len(seen),
                              samplerOutcomeCounts=rejects, wallS=time.monotonic()-started)
                track_ids.update(result['matchedTrackIds'])
                result['seenTrackIds'] = sorted(track_ids)
                result['trackIdChanges'] = sorted(tid for tid in track_ids if source and tid != source['id'])
            if fresh:
                result['reason'] = 'fresh_sample'
                return result
            if len(seen) >= cycles or time.monotonic()-progressed >= 180 or time.monotonic()-started >= 300:
                result['reason'] = 'frame_budget_exhausted' if len(seen) >= cycles else 'observation_timeout_continue'
                if source is not None:
                    self.warn('no_fresh_view', diagnostics=deepcopy(result), ageS=result.get('latestSampleAgeS'), rangeM=result.get('latestSampleRangeM'))
                return result
            self.ex._wait(.25)

    @staticmethod
    def matches(track, source):
        cats = {source['category']}
        if source['category'] in ('airplane_fuselage', 'toy_airplane'):
            cats |= {'airplane_fuselage', 'toy_airplane'}
        return (not track.held and track.category in cats and track.existence >= .5
                and np.linalg.norm(track.pos-np.asarray(source['position'])) <= .10)

    def locate(self, source):
        with self.rt._lock:
            est = self.rt.estimator
            tracks = [t for t in est.tracks.values() if self.matches(t, source)]
            if not tracks:
                tracks = [t for t in est.tracks.values() if not t.held and
                          (t.id == source['id'] or t.category == source['category'])]
            if not tracks:
                cached = getattr(self, '_source_tracks', {}).get(source['id'])
                if cached is None:
                    raise SkillFailed(f"no finite source estimate: {source['id']}")
                tracks = [cached]
            track = min(tracks, key=lambda t: (np.linalg.norm(t.pos-np.asarray(source['position'])) > .10,
                self.rt.estimator.stamp-t.last_seen > 20, -t.last_seen,
                np.linalg.norm(t.pos-np.asarray(source['position']))))
            self.warn('source_estimate', entity=track.id, requested=source['id'],
                      ageS=est.stamp-track.last_seen, candidates=len(tracks),
                      distanceM=float(np.linalg.norm(track.pos-np.asarray(source['position']))))
            return deepcopy(track)

    def camera_geometry(self, target, odometry=None):
        with self.rt._lock:
            est = self.rt.estimator
            cameras = []
            for camera in est.cameras:
                pose = est.camera_pose(odometry, camera)
                intr = est.camera_intrinsics(camera)
                offset = np.asarray(target)-pose['p']
                local = matrix_from_quat(pose['q']).T @ offset
                vertical = math.degrees(math.atan2(local[2], local[0]))
                horizontal = math.degrees(math.atan2(local[1], local[0]))
                half_v = intr['fovy']/2
                half_h = math.degrees(math.atan(math.tan(math.radians(half_v))*intr['width']/intr['height']))
                cameras.append(dict(camera=camera, rangeM=float(np.linalg.norm(offset)),
                    elevationDeg=math.degrees(math.atan2(offset[2], np.linalg.norm(offset[:2]))),
                    horizontalOffsetDeg=horizontal, verticalOffsetDeg=vertical,
                    centreInFov=bool(local[0] > 0 and abs(vertical) < half_v and abs(horizontal) < half_h),
                    occlusion='unknown', cameraPositionM=pose['p']))
            return cameras

    def view(self, job, support, origin, turn):
        self._observation_job = job
        if job.postcondition is None:
            job.postcondition = {}
        box, _, top = entity_box(support)
        target = getattr(self, 'observation_target', [box.cx, box.cy, top])
        close = getattr(self, '_close_view', False)
        # The retry approaches the support edge facing the target, not a wider orbit.
        if close:
            offset = np.asarray(target[:2])-[box.cx, box.cy]
            if np.linalg.norm(offset) > .02:
                origin, turn = math.atan2(offset[1], offset[0]), 0
        candidates = []
        for delta in (0, .25, -.25, .5, -.5):
            angle = origin+turn+delta
            edge = min(box.hx/max(abs(math.cos(angle-box.yaw)), 1e-9),
                       box.hy/max(abs(math.sin(angle-box.yaw)), 1e-9))
            radius = max(.60 if close else .75, edge+(.30 if close else .45))
            x,y = box.cx+radius*math.cos(angle), box.cy+radius*math.sin(angle)
            params = {'x':x, 'y':y, 'yaw':math.atan2(target[1]-y,target[0]-x)}
            geometry = self.camera_geometry(target, params)
            visible = [c for c in geometry if c['centreInFov']]
            score = min((c['rangeM']+.005*abs(c['verticalOffsetDeg']) for c in visible), default=math.inf)
            candidates.append((score, params, geometry))
        for _, params, geometry in sorted(candidates, key=lambda c: c[0]):
            receipt = self.rt.predict('navigate', params)
            if receipt.get('withinRiskLimit'):
                self.step(job, 'navigate', params, receipt=receipt)
                arrived = self.ex._state()
                job.postcondition.setdefault('observationPoses', []).append(dict(
                    retry=close, commanded=params, arrivedAt=arrived['stamp'],
                    odometry=arrived['odometry'], cameras=self.camera_geometry(target, arrived['odometry']),
                    pointing='base_yaw; camera mounts from robot FK'))
                return arrived['stamp']
        raise SkillFailed('no reachable observation pose around the perceived support')

    def stationary_since(self, source, support, track, result):
        # The only motion veto is the common body-only, post-grasp raw-view test.
        with self.rt._lock:
            memory = self.rt.estimator.stationary_assemblies
            observed = result[-1]['observedAt']
            for event in memory.motion_events:
                if event['entity'] in (source['id'], track.id, support.id) and event['stamp'] >= observed:
                    self.warn('prior_robot_motion', entity=event['entity'], stamp=event['stamp'], sampleAt=observed)
            if track.interface_motion_at > observed:
                self.warn('interface_motion_flag_ignored', entity=track.id, stamp=track.interface_motion_at, sampleAt=observed)
            memory.update(self.rt.estimator.tracks, self.rt.estimator.stamp)
            record = memory.records.get(source['id'], {})
            if record.get('hardMotion'):
                self.warn('body_motion_estimate_continue', detail=record['reason'])
        return True, 'no_direct_body_motion; stationary_assumption'

    def freeze(self, job, plan, support, origin):
        self._observation_job = job
        self.check()
        with self.rt._lock:
            old = self.rt.estimator.stationary_assemblies.records.get(plan['body'])
        if old is not None:
            self.reference(plan['body'])
            self.progress(job, 'register', True, 'reuse stationary estimate')
            return
        source = plan['bodySource']
        self.observation_target = source['position']
        category = 'airplane_fuselage' if source['category'] == 'toy_airplane' else source['category']
        frame, holes = SCREW_HOLE_FRAMES.get(category), screw_holes(category)
        track = self.locate(source)
        diag = {}
        multiview_interface_box(list(track.interface_views), frame, holes,
                                after=self.rt.estimator.stamp-20, diagnostics=diag)
        job.postcondition.setdefault('registrationAttempts', []).append(diag)
        self.warn('registration_quality', diagnostics=deepcopy(diag))
        result = self.best_estimate(track)
        self._store_registration(job, plan, support, category, track, result)
        self.progress(job, 'register', True, 'best available estimate; quality is advisory')

    def best_estimate(self, track):
        category = 'airplane_fuselage' if track.category == 'toy_airplane' else track.category
        views = list(track.interface_views) + ([track.interface_best_near] if track.interface_best_near else [])
        result = best_interface_box(views, SCREW_HOLE_FRAMES.get(category),
            screw_holes(category), (track.pos, track.size, track.yaw, track.last_seen), self.rt.estimator.stamp)
        for warning in result[-1]['warnings']:
            self.warn(**warning, entity=track.id)
        return result

    def prepare_part(self, source):
        """Install the selected measured estimate before predicting the next motion.

        A far/fused estimate remains correctable by the first complete near RGB-D
        sample while docking; pinning must not prevent that one correction.
        """
        track = self.locate(source)
        result = self.best_estimate(track)
        with self.rt._lock:
            est = self.rt.estimator
            current = est.tracks.get(track.id)
            if current is None:
                current = deepcopy(track)
                est.tracks[current.id] = current
            current.pos, current.size, current.yaw = result[0].copy(), result[1].copy(), result[2]
            current.grasp_estimate_correction = None
            current.grasp_estimate_pending = result[-1]['tier'] >= 2
            current.interface_estimate_at = result[-1]['observedAt']
            est.version += 1
            return deepcopy(current)

    def _store_registration(self, job, plan, support, category, track, result):
        centre,size,yaw,registered,registration,evidence = result
        evidence.setdefault('lowConfidence', evidence['confidence'] == 'low')
        evidence.setdefault('ageS', self.rt.estimator.stamp-evidence['observedAt'])
        posture = assembly_posture(size, SHAPES.get(category, {}).get('dims'), slack_m=.015)
        if posture['status'] == 'inconsistent':
            self.warn('body_posture', posture=posture, sizeM=size.tolist())
        with self.rt._lock:
            est = self.rt.estimator
            if self.generation != self.rt._generation:
                raise SkillFailed('sensor session changed while registering')
            safe, reason = self.stationary_since(plan['bodySource'], support, track, result)
            if not safe:
                raise SkillFailed('body registration invalidated before freeze: '+reason)
            # An isolated entity snapshot: raw perception continues to update independently.
            ent = est._track_entity(track)
            ent.id = plan['body']
            ent.pose.value = {'p':centre.tolist(), 'q':quat_from_yaw(yaw)}
            ent.pose.t, ent.pose.src = evidence['observedAt'], 'fused'
            ent.rigid.links[0].geometry['collision'][0]['size'] = size.tolist()
            ent.rigid.links[0].affordances = [a for a in ent.rigid.links[0].affordances
                                            if not (isinstance(a.value,dict) and 'screw-hole' in a.value)]
            ent.rigid.links[0].affordances += [Est(value={'screw-hole':h}, src='prior') for h in registered]
            ent.state_vars['orientation'] = Est(value={'holeRegistration':registration,
                'posture':posture, 'directedYawObserved':False, 'interfaceAxisConsistent':None}, src='inferred')
            ref = {'entity':ent.id, 'category':category, 'family':'toy_airplane' if category == 'airplane_fuselage' else category,
                   'position':centre.tolist(), 'yaw':yaw, 'size':size.tolist(),
                   'holes':registered, 'holeSource':'prior', 'registration':registration,
                   'support':support.id, 'registeredAt':evidence['observedAt']}
            record = est.stationary_assemblies.freeze(ent, support, ref, evidence)
            if track.id not in record['aliases']:
                record['aliases'].append(track.id)
            est.version += 1
        job.postcondition['registrationEvidence'] = evidence
        job.log.append(f"froze {ent.id}: {evidence['views']} views, confidence={evidence['confidence']}")
        return

    def verify(self, record, after):
        ref = record['reference']
        relations = []
        with self.rt._lock:
            tracks = list(self.rt.estimator.tracks.values())
            stamp = self.rt.estimator.stamp
            for p in record['parts']:
                candidates = []
                for t in tracks:
                    obs = t.assembly_observation
                    if (t.held or t.id == ref['entity'] or t.category != p['category'] or not obs
                            or obs['stamp'] <= max(after,p['releasedAt']) or stamp-obs['stamp'] > 5
                            or obs['truncated'] or obs['invalidDepthFraction'] > .05):
                        continue
                    if np.linalg.norm(np.asarray(obs['centreM'])[:2]-np.asarray(p['position'])[:2]) > .12:
                        continue
                    if not np.allclose(np.sort(obs['visibleSizeM'][:2]), np.sort(p['size'][:2]), atol=.025):
                        continue  # merged envelope is not a separately observed part
                    candidates.append(obs)
                row = {'entity':p['entity'], 'body':ref['entity'], 'releasedAt':p['releasedAt'], 'relativePose':None,
                       'planarPassed':None, 'passed':None, 'upright':None, 'directedYawObserved':False}
                if len(candidates)==1:
                    obs=candidates[0]
                    delta=np.asarray(obs['centreM'])-ref['position']; c,s=math.cos(ref['yaw']),math.sin(ref['yaw'])
                    error=float(np.linalg.norm(np.asarray(obs['centreM'])[:2]-np.asarray(p['position'])[:2]))
                    axis=abs((obs['boxYawRad']-p['yaw']+math.pi/2)%math.pi-math.pi/2)
                    row.update(observedAt=obs['stamp'], relativePose={'positionM':[c*delta[0]+s*delta[1],
                        -s*delta[0]+c*delta[1],delta[2]], 'boxYawRad':obs['boxYawRad']-ref['yaw'],
                        'bodySource':'frozen_registration', 'partSource':'fresh_rgbd_segment'},
                        deltaXYM=error, axisErrorDeg=math.degrees(axis), planarPassed=error<=.03 and axis<=math.radians(10))
                    if not row['planarPassed']: row['passed']=False
                relations.append(row)
                current = self.rt.estimator.stationary_assemblies.records.get(ref['entity'])
                if current is not None:
                    entry = next((x for x in current['parts'] if x['entity'] == p['entity']), None)
                    if entry is not None:
                        entry['verification'] = deepcopy(row)
        return {'stamp':stamp, 'relations':relations, 'bodyUpright':None,
                'referenceStatus':record['status'], 'passed':False if any(r['passed'] is False for r in relations) else None}

    def verify_chain(self, body_id, after):
        """Reobserve earlier placements too when the supporting body has changed."""
        seen, samples = set(), []
        while body_id not in seen:
            seen.add(body_id)
            record = self.reference(body_id)
            samples.append(self.verify(record, after))
            body_id = record['reference']['support']
            with self.rt._lock:
                if body_id not in self.rt.estimator.stationary_assemblies.records:
                    break
        return {**samples[0], 'relations': [r for s in samples for r in s['relations']],
                'passed': False if any(s['passed'] is False for s in samples) else None}

    def run(self, job, plan):
        self._active_stage = 'register'
        try:
            self._run(job, plan)
        except Exception as exc:
            if not (job.postcondition or {}).get('progress', []) or job.postcondition['progress'][-1]['ok']:
                self.progress(job, self._active_stage, False, str(exc))
            raise

    def _run(self, job, plan):
        self.body_id = plan['body']
        self._observation_job = job
        with self.rt._lock:
            self._source_tracks = deepcopy(self.rt.estimator.tracks)
        job.postcondition = {'releaseExecuted':False, 'passed':None, 'samples':[]}
        scene = self.rt.ego_state()
        support = scene.find(plan['support'])
        if support is None:
            raise SkillFailed('perceived support disappeared')
        o=self.ex._state()['odometry']; box,_,_=entity_box(support)
        origin=math.atan2(o['y']-box.cy,o['x']-box.cx)
        self.freeze(job, plan, support, origin)
        registration = self.reference(self.body_id)['registrationEvidence']
        job.postcondition['registrationEvidence'] = registration
        # Find the loose component afresh; never look up the body through segmentation again.
        source=plan['partSource']
        self.observation_target = source['position']
        self._active_stage = 'pick'
        part = self.prepare_part(source)
        receipt = self.rt.predict('pick', {'target':part.id})
        if not receipt['plan'].get('protectGraspFrame'):
            self.warn('grasp_frame_unprotected', entity=part.id,
                      ageS=self.rt.estimator.stamp-part.last_seen)
        if not receipt.get('withinRiskLimit'):
            self.progress(job, 'pick', False, '; '.join(receipt.get('issues', [])))
            raise SkillFailed('loose part has no feasible pick within risk limit: ' + '; '.join(receipt.get('issues', [])))
        self.reference(self.body_id)
        self.step(job,'pick',{'target':part.id},receipt=receipt)
        with self.rt._lock:
            held=self.rt.estimator.tracks.get(self.rt.estimator.held_id)
            assoc=held.grasp_association if held else None
            if assoc and assoc.get('arrivalCorrection'):
                self.warn('arrival_estimate_correction', correction=deepcopy(assoc['arrivalCorrection']))
            if (not assoc or assoc['requested'] != part.id or assoc['category'] != part.category
                    or np.linalg.norm(np.asarray(assoc['positionAtGrasp'])-part.pos) > .08
                    or assoc['distanceToGripperM'] > .08):
                self.warn('grasp_association_uncertain', requested=part.id, held=held.id if held else None, association=deepcopy(assoc))
            if held is None:
                raise SkillFailed('pick did not acquire a load')
        self.reference(self.body_id)
        self._active_stage = 'fit_and_place'
        params={'on':self.body_id,'alignHoles':True, **{k:plan[k] for k in ('hole','heldHole') if k in plan}}
        if plan.get('closeRelease'):
            params['closeRelease'] = True
        if plan.get('deferVerify'):
            params['stayNear'] = True
        place=self.rt.predict('place',params)
        assembly_report = job.postcondition
        try:
            self.step(job,'place',params,receipt=place)
        finally:
            # Release may have succeeded before retreat failed; never retry that component blindly.
            with self.rt._lock:
                record=deepcopy(self.rt.estimator.stationary_assemblies.records.get(self.body_id, {'parts':[], 'reference':{}}))
            released=any(p['entity']==held.id for p in record['parts'])
            placement_evidence = job.postcondition
            job.postcondition = assembly_report
            if placement_evidence is not assembly_report:
                job.postcondition['placementEvidence'] = placement_evidence
            job.postcondition.update({'releaseExecuted':released,'part':held.id,'body':self.body_id,
                               'passed':None,'samples':[], 'reference':record['reference']})
            with self.rt._lock:
                child = self.rt.estimator.stationary_assemblies.records.get(held.id)
                if child is not None:
                    job.postcondition['releasedReference'] = deepcopy(child['reference'])
        if not released:
            raise SkillFailed('placement did not record a release')
        self._active_stage = 'retreat_verify'
        after=max(p['releasedAt'] for p in record['parts'])
        job.postcondition['releasedAt'] = after
        if plan.get('deferVerify'):
            # Read whatever independent image is already available. No camera
            # wait, base retreat or two-view tour before the first fastener.
            from steward.ego.assembly_motion import placement_check
            expected = next(p for p in record['parts'] if p['entity'] == held.id)
            with self.rt._lock:
                check = placement_check(self.rt.estimator.tracks.values(), source, expected,
                                        after, self.rt.estimator.stamp)
            job.postcondition.update(placementCheck=check, passed=check['passed'],
                                     verificationDeferred=True)
            return
        samples=[]
        self.observation_target = record['reference']['position']
        for turn in (0,.65):
            try:
                arrived=self.view(job,support,origin,turn)
                self.look(arrived)
                sample=self.verify_chain(self.body_id, max(after,arrived))
                samples.append(sample)
            except SkillFailed as exc:
                job.postcondition['verificationError']=str(exc)
                break
        stable = len(samples) == 2 and samples[1]['stamp'] > samples[0]['stamp']
        if stable:
            for before, current in zip(samples[0]['relations'], samples[1]['relations']):
                a,b = before['relativePose'],current['relativePose']
                stable &= bool(a and b and current.get('observedAt',0) > before.get('observedAt',0)
                    and np.linalg.norm(np.asarray(a['positionM'])-b['positionM']) <= .02
                    and abs((a['boxYawRad']-b['boxYawRad']+math.pi/2)%math.pi-math.pi/2) <= math.radians(10))
        with self.rt._lock:
            final_status = self.rt.estimator.stationary_assemblies.records.get(self.body_id, {}).get('status')
        job.postcondition.update(samples=samples, stableObserved=stable, referenceStatus=final_status,
            passed=False if final_status != 'frozen' or any(s['passed'] is False for s in samples) else None,
            reason='fresh component poses relative to stationary registration; directed yaw/uprightness may remain unknown')
        from steward.ego.assembly_motion import placement_check
        expected = next(p for p in record['parts'] if p['entity'] == held.id)
        with self.rt._lock:
            check = placement_check(self.rt.estimator.tracks.values(), source, expected,
                                    after, self.rt.estimator.stamp)
        job.postcondition['placementCheck'] = check
        if check['passed'] is not None:
            job.postcondition['passed'] = check['passed']
        self.progress(job, 'retreat_verify', job.postcondition['passed'] is True,
                      'measured mismatch' if job.postcondition['passed'] is False else 'see measured relations; unobserved axes remain unknown')
        if job.postcondition['passed'] is False:
            raise SkillFailed('post-release position/orientation mismatch; release will not be retried')
        job.message='component released; inspect postcondition for measured relations and unresolved verification'
