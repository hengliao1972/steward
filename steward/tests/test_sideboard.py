"""Sideboards compile both with and without a sliding drawer."""
import pytest

from steward.assets.generators import sideboard
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import SceneState


@pytest.mark.parametrize('drawer_width', [0., .5])
def test_sideboard_contains_the_requested_drawer(drawer_width):
    entity = sideboard(drawer_width=drawer_width)
    build = build_mjcf(SceneState(mode='world', entities=[entity]))
    model = build.compile()
    joints = {joint.id: joint for joint in entity.rigid.joints}
    if drawer_width:
        assert joints['drawer_slide'].limits.value['hi'] > 0
        assert model.jnt_range[model.joint(build.names[entity.id].joints['drawer_slide']).id, 1] > 0
    else:
        assert not joints and model.njnt == 0
