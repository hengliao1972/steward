"""Stationary mating references, corroborated by subsequent RGB-D envelopes.

An envelope check is deliberately weaker than verifying holes, directed yaw or
uprightness. Commands alone never count as observations or mechanical attachment.
"""
from copy import deepcopy
from itertools import product, combinations
import math

import numpy as np


def bounds(position, size, yaw, reference):
    corners = np.array(list(product((-.5, .5), repeat=3))) * size
    c, s = math.cos(yaw), math.sin(yaw)
    corners[:, :2] = corners[:, :2] @ np.array([[c, s], [-s, c]])
    corners += np.asarray(position) - reference['position']
    c, s = math.cos(reference['yaw']), math.sin(reference['yaw'])
    corners[:, :2] = corners[:, :2] @ np.array([[c, -s], [s, c]])
    return corners.min(0), corners.max(0)


class AssemblyMemory:
    def __init__(self):
        self.records = []

    def invalidate(self, entity):
        for record in self.records:
            if entity in (record['entity'], record['reference']['entity'], record['support']) or any(
                    p['entity'] == entity for p in record['parts']):
                record['status'] = 'invalidated'

    def released(self, entity, position, target, stamp):
        ref = target.get('mateReference') if target else None
        if not ref:
            return
        record = next((r for r in self.records if ref['entity'] in (r['entity'], r['reference']['entity'])
                       and r['status'] != 'invalidated'), None)
        if record is None:
            record = {'entity': ref['entity'], 'reference': deepcopy(ref), 'parts': [],
                      'support': ref['support'], 'status': 'pending'}
            self.records.append(record)
        record['parts'].append({'entity': entity, 'position': list(position),
                                'size': list(target['sizeM']), 'yaw': target['yawRad'],
                                'category': target.get('partCategory'),
                                'releasedAt': stamp, 'status': 'released_unverified'})

    def views(self, tracks, stamp):
        result = {}
        for r in self.records:
            if r['status'] == 'invalidated':
                if r['entity'] in tracks and not tracks[r['entity']].held:
                    result[r['entity']] = deepcopy(r)
                continue
            ref = r['reference']
            base = bounds(ref['position'], ref['size'], ref['yaw'], ref)
            candidates = []
            for t in tracks.values():
                # No arbitrary nearest-neighbour identity transfer. Family alias is
                # the same catalogue hypothesis used by the normal registrar.
                known_categories = {ref['category'], ref.get('family')} | {p.get('category') for p in r['parts']}
                if t.held or t.category not in known_categories:
                    continue
                obs = t.assembly_observation
                if not obs or obs['truncated'] or obs['invalidDepthFraction'] or stamp-obs['stamp'] > 2.0:
                    continue
                if obs['stamp'] <= min(p['releasedAt'] for p in r['parts']):
                    continue
                if np.linalg.norm(np.asarray(obs['centreM'])-ref['position']) > .25:
                    continue
                actual = bounds(obs['centreM'], obs['visibleSizeM'], obs['boxYawRad'], ref)
                slack = min(.03, .01 + 2*t.sigma)
                eligible = [p for p in r['parts'] if obs['stamp'] > p['releasedAt']]
                # Either a separately segmented body, or a union of known releases.
                # Unknown neighbours cannot enlarge the expected footprint.
                matches = []
                for n in range(len(eligible)+1):
                    for subset in combinations(eligible, n):
                        boxes = [base] + [bounds(p['position'], p['size'], p['yaw'], ref) for p in subset]
                        lo, hi = np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
                        error = max(np.max(abs(actual[0]-lo)), np.max(abs(actual[1]-hi)))
                        if error <= slack:
                            matches.append((error, subset, lo, hi))
                if matches:
                    candidates.append((t, obs, min(matches, key=lambda x: x[0])))
            if len(candidates) != 1:
                r['status'] = 'unresolved'
                if r['entity'] in tracks:
                    result[r['entity']] = deepcopy(r)
                continue
            t, obs, (_, subset, lo, hi) = candidates[0]
            # A moved original body cannot be hidden by a coincident replacement.
            original = tracks.get(r['entity'])
            if original is not None and original is not t:
                r['status'] = 'unresolved'
                if r['entity'] in tracks:
                    result[r['entity']] = deepcopy(r)
                continue
            r['entity'], r['status'] = t.id, 'stationary_envelope_observed'
            for p in subset:
                p['status'] = 'envelope_observed'
                p['observedAt'] = obs['stamp']
            # Independently visible parts provide stronger positional evidence.
            for p in r['parts']:
                part = tracks.get(p['entity'])
                po = part.assembly_observation if part is not None and part is not t and not part.held else None
                if po and po['stamp'] > p['releasedAt'] and stamp-po['stamp'] <= 2 and not po['truncated'] and not po['invalidDepthFraction']:
                    pb = bounds(po['centreM'], po['visibleSizeM'], po['boxYawRad'], ref)
                    expected = bounds(p['position'], p['size'], p['yaw'], ref)
                    if max(np.max(abs(pb[0]-expected[0])), np.max(abs(pb[1]-expected[1]))) <= .02:
                        p['status'], p['observedAt'] = 'geometry_observed', po['stamp']
                    else:
                        p['status'] = 'invalidated'
            if any(p['status'] == 'invalidated' for p in r['parts']):
                r['status'] = 'invalidated'
                result[t.id] = deepcopy(r)
                continue
            # Collision extent includes all released pieces, even before verification.
            boxes = [base] + [bounds(p['position'], p['size'], p['yaw'], ref) for p in r['parts']]
            full_lo, full_hi = np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
            result[t.id] = {**deepcopy(r), 'observedAt': obs['stamp'], 'observation': deepcopy(obs),
                            'collisionSize': (2*np.maximum(abs(full_lo), abs(full_hi))).tolist(),
                            'expectedEnvelope': {'min': lo.tolist(), 'max': hi.tolist()},
                            'upright': None, 'directedYawVerified': False}
        return result
