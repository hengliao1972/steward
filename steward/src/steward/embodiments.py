"""Embodiments: which robot body a scenario puts in the world, registered in one place.

A scene's robot entity carries the identity of its body in ``asset.id`` and ``category`` (including
``steward-bot``, ``cobot-magic`` and the imported ``agilex-usd``). This registry maps that id to what the
rest of the system needs to know about a body without opening its model: a label for the console,
the ``steward`` subcommand that runs its Ego Runtime, and the five-layer profile document under
``steward/embodiments/``. Like :mod:`steward.phases`, one entry per implementation; a body the
registry does not know is shown by its asset id and gets no Ego Runtime.

Switching bodies is never a live switch: the body is compiled into the world model, so another
body means a new episode and another Ego Runtime process. A scenario names the body it was built
with, but any registered body can take it (:func:`with_body`).
"""

from __future__ import annotations

import copy

EMBODIMENTS: dict[str, dict] = {
    "steward-bot": {
        "label": "Steward 移动厨房机器人",
        "description": "平面移动底盘 + 升降柱 + 直线伸出臂 + 平行爪；1.5 m 固定头部 RGB-D 相机（俯视 30°）",
        "profile": "steward-bot.yaml",
        "ego": "ego",
        # Its missions are clients of the Ego API (`steward chores --api`, the Ego Harness).
        "runs_mission": False,
        "skills": True,
    },
    "agilex-usd": {
        "label": "AgileX USD 双 Piper",
        "description": "robot.usd 原始网格与双臂关节；平面移动底盘、升降、双爪与仿真 RGB-D 相机",
        "profile": "agilex-usd.yaml", "ego": "ego", "runs_mission": False, "skills": True,
    },
    "cobot-magic": {
        "label": "AgileX Cobot Magic 双 Piper",
        "description": "TRACER 差速底盘 + 升降柱 + 双 Piper 6-DoF 臂 + 平行爪；桅杆头相机（巡视+短焦）+ 左右腕部 RGB-D（link6 eye-in-hand）",
        "profile": "cobot-magic.yaml",
        # The same Ego Runtime as the kitchen robot (`steward ego --body cobot-magic`): one Piper arm
        # and the torso act as its planar arm (steward.ego.cobot_arm), so ESim, the skills and the
        # Ego Harness drive it unchanged. The scripted missions stay with `steward cobot`.
        "ego": "ego",
        "runs_mission": False,
        "skills": True,
    },
}


def embodiment_of(entity: dict) -> str | None:
    """The registered embodiment id of a robot entity (a scene-state dict), else ``None``."""
    asset = entity.get("asset")
    candidates = [asset.get("id") if isinstance(asset, dict) else asset,
                  (entity.get("category") or {}).get("value")]
    return next((c for c in candidates if c in EMBODIMENTS), None)


def body_id_of(entity: dict) -> str | None:
    """What the entity calls its body, registered or not: the asset id, else the category."""
    asset = entity.get("asset")
    if isinstance(asset, dict):
        asset = asset.get("id")
    return asset or (entity.get("category") or {}).get("value")


def describe(embodiment_id: str | None, body_id: str | None = None) -> dict:
    """``{id, label, description, registered, skills}`` for the console; an unregistered body is
    named by its asset id so the operator sees what is there rather than a blank."""
    if embodiment_id in EMBODIMENTS:
        entry = EMBODIMENTS[embodiment_id]
        return {"id": embodiment_id, "label": entry["label"], "description": entry["description"],
                "registered": True, "skills": entry["skills"]}
    shown = body_id or embodiment_id or "unknown"
    return {"id": shown, "label": f"未注册本体 · {shown}", "description": "这个机器人资产没有登记在本体注册表里，"
            "控制台能显示它，但没有 Ego Runtime 会去驱动它", "registered": False, "skills": False}


def scenario_embodiments(scenario: dict) -> dict[str, dict]:
    """Robot id -> :func:`describe` entry for every robot a loaded scenario lists."""
    entities = {e["id"]: e for e in scenario["sceneState"]["entities"]}
    out = {}
    for rid in scenario["robots"]:
        ent = entities.get(rid, {})
        out[rid] = describe(embodiment_of(ent), body_id_of(ent))
    return out


def embodiment_catalog() -> list[dict]:
    """Every registered body with its label and description."""
    return [describe(k) for k in EMBODIMENTS]


def robot_entity(embodiment_id: str) -> dict:
    """A fresh scene-state entity of a registered body, as its generator builds it."""
    if embodiment_id == "steward-bot":
        from steward.assets.generators import steward_bot

        return steward_bot().to_dict()
    if embodiment_id == "agilex-usd":
        from steward.assets.agilex import agilex_usd

        return agilex_usd()
    if embodiment_id == "cobot-magic":
        from steward.assets.cobot import cobot_magic

        return cobot_magic().to_dict()
    raise ValueError(f"unknown body {embodiment_id!r}; registered: {', '.join(EMBODIMENTS)}")


def with_body(scenario: dict, embodiment_id: str) -> dict:
    """``scenario`` with every robot it lists given the body ``embodiment_id``: same id, same start
    pose, the body's own model. Any body works in any scenario: what a scenario fixes is the room,
    the objects and the task, not which robot does it."""
    if embodiment_id not in EMBODIMENTS:
        raise ValueError(f"unknown body {embodiment_id!r}; registered: {', '.join(EMBODIMENTS)}")
    out = copy.deepcopy(scenario)
    robots = set(out["robots"])
    entities = out["sceneState"]["entities"]
    for i, ent in enumerate(entities):
        if ent["id"] not in robots or embodiment_of(ent) == embodiment_id:
            continue
        body = robot_entity(embodiment_id)
        body.update({"id": ent["id"], "pose": copy.deepcopy(ent["pose"]), "asset": {"id": embodiment_id}})
        entities[i] = body
    return out
