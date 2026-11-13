"""Toy-assembly workshop: airplane and car parts settle, with screw holes on the mating faces."""

from __future__ import annotations

from steward.assets.generators import (airplane_fuselage, airplane_landing_gear, airplane_wing, car_body, car_wheel,
                                       electric_screwdriver, toy_screw)
from steward.authoring.service import AuthoringService, build_toy_assembly, toy_assembly_goal


def _hole_ids(entity) -> set[str]:
    out = set()
    for link in entity.rigid.links:
        for aff in link.affordances:
            value = aff.value
            if isinstance(value, dict) and "screw-hole" in value:
                out.add(value["screw-hole"]["id"])
    return out


def test_airplane_and_car_parts_carry_the_screw_holes_the_task_describes():
    assert _hole_ids(airplane_fuselage()) == {"wing_0", "wing_1", "gear_0", "gear_1"}
    assert _hole_ids(airplane_wing()) == {"mount_0", "mount_1"}
    assert _hole_ids(airplane_landing_gear()) == {"mount_0", "mount_1"}
    assert len(_hole_ids(car_body())) == 4
    assert _hole_ids(car_wheel()) == {"hub"}
    assert any(a.value == "screwdriver" for a in electric_screwdriver().rigid.links[0].affordances)
    assert toy_screw().category.value == "toy_screw"


def test_build_toy_assembly_publishes_a_settled_scene_with_both_kits(tmp_path):
    out = build_toy_assembly(AuthoringService(tmp_path))
    assert out["scene"]["report"]["ok"]
    assert out["scene"]["entities"] == 25
    scenario = AuthoringService(tmp_path).load_scenario("toy-assembly")
    assert scenario["task"].startswith("两张桌子上分别有一套拆开的玩具")
    assert "螺旋桨" not in scenario["task"]
    assert "一次只装一个零件" in scenario["task"]
    assert "刀头朝下" in scenario["task"] and "不要横着" in scenario["task"]
    assert "侧面立起" in scenario["task"] and "螺丝孔朝上" in scenario["task"]
    assert "ego_screw_holes" in scenario["task"] and "facing=up" in scenario["task"]
    assert "插入该轮子" in scenario["task"] and "带螺丝的轮子" in scenario["task"]
    assert "竖直朝下" in scenario["task"] and "尖端向下" in scenario["task"]
    assert "通孔" in scenario["task"]
    assert scenario["task"].index("组装汽车玩具") < scenario["task"].index("组装玩具飞机")
    assert "组装说明书" in scenario["task"] and "参考图" in scenario["task"]
    assert scenario["goal"] == toy_assembly_goal()
    ids = {e["id"] for e in scenario["sceneState"]["entities"]}
    assert {"w:plane-fuselage", "w:plane-wing", "w:plane-gear", "w:car-body", "w:screwdriver",
            "w:plane-manual", "w:plane-picture", "w:car-manual", "w:car-picture",
            "w:plane-screw-0", "w:plane-screw-3", "w:car-screw-3", "w:table-plane", "w:table-car"} <= ids
    assert "w:plane-prop-l" not in ids and "w:plane-prop-r" not in ids
    by_id = {e["id"]: e for e in scenario["sceneState"]["entities"]}
    assert by_id["w:plane-manual"]["category"]["value"] == "manual"
    assert by_id["w:car-manual"]["category"]["value"] == "manual"
    assert by_id["w:plane-picture"]["category"]["value"] == "picture"
    assert by_id["w:car-picture"]["category"]["value"] == "picture"
    for eid in ("w:plane-manual", "w:plane-picture", "w:car-manual", "w:car-picture"):
        assert any(g.get("texture") for g in by_id[eid]["body"]["links"][0]["geometry"]["collision"])
