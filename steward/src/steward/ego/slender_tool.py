"""Axis estimates for stand-and-regrasp, independent of object names and tasks."""
import numpy as np
from steward.ego.planner import entity_box
from steward.math3d import matrix_from_quat


def axial_feature(entity, stamp):
    saved = entity.state_vars.get('toolFeature')
    if saved and saved.value:
        feature = saved.value
        direction = np.asarray(feature['tip'])-feature['head']
        length = np.linalg.norm(direction)
        if length <= 0 or not np.isfinite(length):
            return None
        return {**feature, 'axis':(direction/length).tolist()}
    box, low, high = entity_box(entity)
    size = np.array([2*box.hx, 2*box.hy, high-low])
    ordered = sorted(size)
    if ordered[-1] < 2*ordered[-2] or ordered[-2] <= 0:
        return None
    axis = matrix_from_quat(entity.pose.value['q']) @ np.eye(3)[np.argmax(size)]
    centre = np.asarray(entity.pose.value['p'])
    length = float(max(size))
    return dict(axis=axis.tolist(), tip=(centre+axis*length/2).tolist(),
        head=(centre-axis*length/2).tolist(), lengthM=length, radiusM=float(ordered[-2]/2),
        stamp=entity.pose.t if entity.pose.t is not None else stamp,
        source='estimated_long_axis', polarityObserved=False)
