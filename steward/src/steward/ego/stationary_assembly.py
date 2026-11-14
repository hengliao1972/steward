"""Robot-owned stationary registration, independent of segmentation track lifetime.

Persistence is a motion assumption, not fresh perception or evidence of attachment.
Raw tracks remain available for collision geometry and independent verification.
"""
from copy import deepcopy
import math

import numpy as np

from steward.ego.assembly_memory import bounds
from steward.math3d import yaw_from_quat, quat_from_yaw
from steward.scene.schema import Relation, Est


# Conservative calibration from qg3/qc8 sensor logs; see ANALYSIS_g3.md.
# Max observed stationary association/position excursion < .08 m; .25 > 3*.08.
BODY_DISPLACEMENT_M = .25
NEAR_VIEW_M = 2.0
FRESH_VIEW_S = 5.0


class StationaryAssemblies:
    def __init__(self):
        self.records = {}
        self.first_grasp_at = None
        self.motion_events = []  # episode-local robot actions, also before any freeze

    def freeze(self, entity, support, reference, evidence):
        old = self.records.get(entity.id)
        if old is not None:
            if old['status'] == 'invalidated':
                raise ValueError('stationary reference invalidated; recover and register a new body track')
            return old
        record = dict(entity=deepcopy(entity), support=deepcopy(support),
                      reference=deepcopy(reference), registrationEvidence=deepcopy(evidence),
                      status='frozen', parts=[], motionSamples={}, reason=None, aliases=[entity.id], warnings=[])
        self.records[entity.id] = record
        return record

    def invalidate(self, entity, track=None, stamp=None):
        if stamp is not None:
            self.motion_events.append(dict(entity=entity, stamp=float(stamp),
                category=track.category if track is not None else None,
                position=track.pos.tolist() if track is not None else None))
        for r in self.records.values():
            ref = r['reference']
            spatial = track is not None and track.category in (ref['category'], ref.get('family')) and np.linalg.norm(track.pos-np.asarray(ref['position'])) < .10
            if entity in (*r['aliases'], ref['support']) or spatial:
                self.warn(r, 'robot_action_on_reference', entity=entity, stamp=stamp)
            for p in r['parts']:
                if p['entity'] == entity:
                    p['status'] = 'invalidated'

    @staticmethod
    def warn(record, reason, **numbers):
        warning = dict(reason=reason, **numbers)
        rows = record.setdefault('warnings', [])
        if warning not in rows:
            rows.append(warning)
            del rows[:-32]

    def released(self, entity, position, target, stamp):
        ref = (target or {}).get('mateReference', {})
        r = self.records.get(ref.get('entity'))
        if r is not None and r['status'] == 'frozen':
            # The release TCP includes clearance; the next flip needs the settled
            # part envelope, otherwise every turn accumulates that drop height.
            position = list(position)
            frame = target.get('releasedFrame', {})
            if 'settledCentreZM' in frame:
                position[2] = frame['settledCentreZM']
            r['parts'][:] = [p for p in r['parts'] if p['entity'] != entity]
            r['parts'].append(dict(entity=entity, position=list(position), size=list(target['sizeM']),
                                   yaw=target['yawRad'], category=target.get('partCategory'),
                                   releasedAt=stamp, status='released_unverified'))

    def promote_release(self, entity, target, stamp):
        """Registered grasp geometry + executed release yields a LOW confidence body.

        Height assumes settling on the perceived supporting top. This is inferred,
        not fresh RGB-D, seating, or attachment evidence.
        """
        frame = (target or {}).get('releasedFrame')
        parent_id = (target or {}).get('mateReference', {}).get('entity')
        parent = self.records.get(parent_id)
        if not frame or not parent or parent['status'] != 'frozen' or entity.id in self.records:
            return
        part = next((p for p in parent['parts'] if p['entity'] == entity.id), None)
        if part is None or part['status'] != 'released_unverified':
            return
        position = [*part['position'][:2], frame['settledCentreZM']]
        entity = deepcopy(entity)
        entity.pose.value = {'p': position, 'q': quat_from_yaw(part['yaw'])}
        entity.pose.src, entity.pose.t = 'inferred', stamp
        entity.rigid.links[0].geometry['collision'][0]['size'] = list(part['size'])
        entity.rigid.links[0].affordances = [a for a in entity.rigid.links[0].affordances
                                            if not (isinstance(a.value, dict) and 'screw-hole' in a.value)]
        entity.rigid.links[0].affordances += [Est(value={'screw-hole': h}, src=h.get('source', 'prior')) for h in frame['holes']]
        entity.state_vars = {'orientation': Est(value={
            'holeRegistration': frame['registration'], 'interfaceAxisConsistent': True,
            'directedYawObserved': False, 'posture': {'status': 'unknown', 'upright': None}}, src='inferred')}
        ref = {'entity': entity.id, 'category': part['category'],
               'family': 'toy_airplane' if part['category'] == 'airplane_fuselage' else part['category'],
               'position': position, 'size': part['size'], 'yaw': part['yaw'],
               'holes': frame['holes'], 'holeSource': 'prior', 'registration': frame['registration'],
               'support': parent_id, 'registeredAt': stamp, 'source': 'executed_release_propagation'}
        evidence = {'source': 'registered_grasp_and_executed_release', 'confidence': 'low',
                    'views': 0, 'observedAt': frame['sourceObservedAt'], 'propagatedAt': stamp,
                    'upright': None, 'directedYawObserved': False,
                    'seatingObserved': False, 'heightSource': 'assumed_settled_on_perceived_top'}
        self.freeze(entity, parent['entity'], ref, evidence)

    def update(self, tracks, stamp):
        for r in self.records.values():
            if r['status'] != 'frozen':
                continue
            ref = r['reference']
            origin = np.asarray(ref['position'])
            support = tracks.get(ref['support'])
            obs = (support.manipulation_view or support.assembly_observation) if support else None
            if obs:
                distance = float(np.linalg.norm(np.asarray(obs['centreM'])-r['support'].pose.value['p']))
                if distance > .04:
                    self.warn(r, 'support_motion_ignored', entity=support.id, distanceM=distance, ageS=stamp-obs['stamp'])
            candidates = []
            for t in tracks.values():
                # Spatial association with a comparable BODY outline, never the
                # support, released members, a historical id or a merged envelope.
                if (t.id == ref['support'] or t.id in {p['entity'] for p in r['parts']}
                        or t.category not in (ref['category'], ref.get('family'))):
                    continue
                obs = t.manipulation_view or t.assembly_observation
                if not obs:
                    continue
                distance = float(np.linalg.norm(np.asarray(obs['centreM'])-origin))
                age = stamp-obs['stamp']
                range_m = obs.get('rangeM', float('inf'))
                complete = (not obs['truncated'] and obs['invalidDepthFraction'] <= .05
                            and np.allclose(np.sort(obs['visibleSizeM'][:2]), np.sort(ref['size'][:2]), atol=.025))
                if distance <= max(.5, 2*BODY_DISPLACEMENT_M) and complete:
                    candidates.append((distance, t, obs, age, range_m))
            candidates.sort(key=lambda x:x[0])
            # A nearer body at the reference, or two plausible neighbours, defeats
            # a displacement claim about the more distant one.
            if not candidates:
                continue
            distance, t, obs, age, range_m = candidates[0]
            if (r['registrationEvidence'].get('tier', 0) >= 2 and not r['parts']
                    and (self.first_grasp_at is None or distance <= BODY_DISPLACEMENT_M)
                    and 0 <= age <= FRESH_VIEW_S and range_m <= NEAR_VIEW_M):
                # A first close view corrects an explicitly provisional far/fused
                # frame. It is an estimate update, never a movement verdict.
                from steward.ego.hole_registration import register_hole_template
                from steward.ego.priors import screw_holes, SCREW_HOLE_FRAMES
                holes, registration = register_hole_template(screw_holes(ref['category']),
                    SCREW_HOLE_FRAMES.get(ref['category']), obs['visibleSizeM'], best_effort=True)
                ref.update(position=list(obs['centreM']), size=list(obs['visibleSizeM']),
                           yaw=obs['boxYawRad'], registeredAt=obs['stamp'], holes=holes, registration=registration)
                r['entity'].pose.value = dict(p=list(ref['position']), q=quat_from_yaw(ref['yaw']))
                r['entity'].pose.t = obs['stamp']
                r['entity'].rigid.links[0].affordances = [a for a in r['entity'].rigid.links[0].affordances
                    if not (isinstance(a.value, dict) and 'screw-hole' in a.value)] + [Est(value={'screw-hole':h},src='prior') for h in holes]
                r['registrationEvidence'].update(tier=0, observedAt=obs['stamp'], ageS=age, rangeM=range_m,
                                                 source='first_near_correction')
                self.warn(r, 'first_near_body_correction', distanceM=distance, ageS=age, rangeM=range_m)
                distance = 0.
            if t.id not in r['aliases']:
                r['aliases'].append(t.id)
                self.warn(r, 'spatial_alias', entity=t.id, distanceM=distance)
            r['rawBodyObservations'] = {t.id: {k:deepcopy(v) for k,v in obs.items() if k != 'points'}}
            qualified = (self.first_grasp_at is not None and obs['stamp'] > max(ref['registeredAt'], self.first_grasp_at)
                         and 0 <= age <= FRESH_VIEW_S and range_m <= NEAR_VIEW_M
                         and (len(candidates) == 1 or candidates[1][0]-distance > .10))
            moved = distance > BODY_DISPLACEMENT_M
            if distance > .04 or not qualified:
                self.warn(r, 'body_motion_diagnostic', entity=t.id, distanceM=distance,
                          thresholdM=BODY_DISPLACEMENT_M, ageS=age,
                          rangeM=range_m if math.isfinite(range_m) else None,
                          firstGraspAt=self.first_grasp_at, qualified=bool(qualified))
            last, count = r['motionSamples'].get('body', (-1, 0))
            if obs['stamp'] > last:
                count = count+1 if qualified and moved else 0
                r['motionSamples']['body'] = (obs['stamp'], count)
            if count >= 2 and 'ownReleaseAt' in ref:
                self.warn(r, 'post_release_body_displacement', **dict(entity=t.id, distanceM=distance, stamp=obs['stamp']))
            elif count >= 2:
                r.update(status='invalidated', hardMotion=True,
                         reason='direct near fresh body displacement',
                         displacementEvidence=dict(entity=t.id, distanceM=distance,
                            thresholdM=BODY_DISPLACEMENT_M, ageS=age, rangeM=range_m, stamp=obs['stamp']))

    def overlay(self, entities, relations, tracks, stamp):
        self.update(tracks, stamp)
        for body_id, r in self.records.items():
            ref = r['reference']
            if not any(e.id == ref['support'] for e in entities):
                support = deepcopy(r['support'])
                support.pose.src = 'inferred'
                entities.append(support)
            live = next((e for e in entities if e.id == body_id), None)
            ent = deepcopy(live if live is not None and (r['status'] == 'invalidated' or getattr(tracks.get(body_id), 'held', False)) else r['entity'])
            # A moved/held body must retain its current pose for recovery.
            if 'orientation' not in ent.state_vars:
                ent.state_vars['orientation'] = deepcopy(r['entity'].state_vars['orientation'])
            info = ent.state_vars['orientation'].value
            info['stationaryAssembly'] = {k: deepcopy(v) for k, v in r.items()
                                          if k not in ('entity', 'support', 'motionSamples')}
            info['assembly'] = {'status': r['status'], 'reference': deepcopy(ref),
                                'parts': deepcopy(r['parts']), 'upright': None,
                                'directedYawVerified': False}
            info['posture'] = {**info.get('posture', {}), 'status': 'unknown', 'upright': None,
                               'source': 'frozen_registration', 'observedAt': ref['registeredAt']}
            ent.pose.src = 'inferred'  # retain the ORIGINAL time, never make a fresh detection
            boxes = [bounds(ref['position'], ref['size'], ref['yaw'], ref)]
            boxes += [bounds(p['position'], p['size'], p['yaw'], ref) for p in r['parts']]
            if live is not None:
                boxes.append(bounds(live.pose.value['p'], live.rigid.links[0].geometry['collision'][0]['size'],
                                    yaw_from_quat(live.pose.value['q']), ref))
            lo, hi = np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
            if r['status'] == 'frozen':
                ent.rigid.links[0].geometry['collision'][0]['size'] = (2*np.maximum(abs(lo), abs(hi))).tolist()
            if r['status'] == 'invalidated':
                info['posture'].update(status='inconsistent', reason=r['reason'])
            entities[:] = [e for e in entities if e.id != body_id] + [ent]
            if r['status'] == 'frozen':
                relations[:] = [x for x in relations if not (x.type == 'supported-by' and x.a['entity'] == body_id)]
                relations.append(Relation(type='supported-by', a={'entity': body_id}, b={'entity': ref['support']},
                                          derived=True, confidence=r['entity'].existence))
