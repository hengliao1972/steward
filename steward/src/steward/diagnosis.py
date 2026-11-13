"""Self-diagnosis of a stuck mission from its own failure log and the Ego-State.

A mission that stops making progress explains why before it raises an alert: each recent failure
message is sorted into a cause, the dominant cause is named with the share of failures it explains,
and supporting findings come from the robot's own records only (its poses at each failure and the
objects in its Ego-State), never from the World State.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

TABLEWARE = frozenset({"bowl", "cup", "mug", "plate", "wine_glass"})
COLOR_ZH = {"red": "红色", "green": "绿色", "blue": "蓝色", "black": "黑色", "white": "白色", "gray": "灰色",
            "clear": "透明", "brown": "棕色", "beige": "米色", "yellow": "黄色"}
CATEGORY_ZH = {"table": "餐桌", "counter": "操作台", "cabinet": "柜子", "sideboard": "餐边柜", "cup": "杯子", "bowl": "碗",
               "plate": "盘子", "drawer": "抽屉", "wine_glass": "高脚杯", "mug": "咖啡杯", "teddy_bear": "玩具熊",
               "scissors": "剪刀", "chair": "椅子", "wall": "墙", "unknown": "未知物体", "toy_airplane": "玩具飞机",
               "airplane_fuselage": "飞机身体", "airplane_wing": "飞机翅膀", "airplane_propeller": "螺旋桨",
               "airplane_landing_gear": "起落架", "car_body": "汽车身体", "car_wheel": "车轮",
               "toy_screw": "螺丝", "screw_board": "螺丝板", "electric_screwdriver": "电动螺丝刀",
               "manual": "说明书",
               "picture": "参考图", "pistol": "手枪", "pill_bottle": "药瓶", "perfume_bottle": "香水瓶",
               "plant": "绿植"}
SKILL_ZH = {"pick": "抓取", "place": "放置", "navigate": "导航", "open": "打开", "close": "关上",
            "drive_screw": "拧螺丝", "insert_screw_into_hole": "插入螺丝", "pour": "倾倒"}


def describe(obj: dict) -> str:
    """A Chinese name of an Ego-State object for operator text: colour (tableware only), role (or
    category), id.

    :param obj: an object of the Ego API summary.
    :returns: for example ``白色碗 e:16``.
    """
    kind = obj.get("role") or obj["category"]
    color = COLOR_ZH.get(obj.get("color") or "", "") if kind in TABLEWARE else ""
    return f"{color}{CATEGORY_ZH.get(kind, kind)} {obj['id']}"


def explain(message: str, summary: dict | None = None) -> str:
    """A failure message in short Chinese, with the objects it mentions named.

    :param message: a failure message of a prediction, a skill or the mission.
    :param summary: the Ego API summary used to name objects, if available.
    :returns: the explanation.
    """
    names = {o["id"]: describe(o) for o in (summary or {}).get("objects", [])}
    name = lambda oid: names.get(oid, oid)  # noqa: E731 - local shorthand
    parts = []
    approach = re.search(r"no feasible approach: (.*?)(?:'\]|$)", message)
    if approach:
        sides = approach.group(1).split("; ")
        blockers = Counter(m for side in sides for m in re.findall(r"collides with (e:\d+)", side))
        blocked = sum(1 for side in sides if "collides with" in side)
        if blocked:
            parts.append(f"{blocked} 个方向的停靠位被 " + "、".join(name(b) for b in blockers) + " 挡住")
        for reach in re.findall(r"reach ([\d.]+) m out of range", approach.group(1)):
            parts.append(f"1 个方向伸手 {float(reach):.2f} 米超出范围")
    rules = [
        (r"height ([\d.]+) m is outside the lift range \[([\d.]+), ([\d.]+)\]",
         lambda m: f"高度 {float(m[1]):.2f} 米超出升降范围 {m[2]}–{m[3]} 米" + ("（物体在地面上）" if float(m[1]) < 0.1 else "")),
        (r"existence probability is only ([\d.]+)", lambda m: f"目标的存在概率只有 {m[1]}"),
        (r"missed the handle of (e:\d+)", lambda m: f"没抓到 {name(m[1])} 的把手"),
        (r"missed", lambda m: "抓取没有对准"),
        (r"base is blocked", lambda m: "底盘受阻，原地没有进展"),
        (r"turn did not converge", lambda m: "原地转身没有到位"),
        (r"joint (\w+) did not reach ([\d.]+) \(at ([\d.]+)\)", lambda m: f"{m[1]} 没能到 {float(m[2]):.2f}（停在 {float(m[3]):.2f}，被挡住）"),
        (r"no support surface is known under (e:\d+)", lambda m: f"不知道 {name(m[1])} 放在什么上面"),
        (r"no ([\w ]+?) in the ego state", lambda m: f"Ego State 里没有 {CATEGORY_ZH.get(m[1], m[1])}"),
        (r"(e:\d+) \((\w+)\) is in the way and there is no free spot", lambda m: f"{name(m[1])} 挡路，且没有空位可以挪开它"),
    ]
    for pattern, render in rules:
        found = re.search(pattern, message)
        if found and not approach:
            parts.append(render(found))
            break
    return "；".join(parts) if parts else re.sub(r"e:\d+", lambda m: name(m[0]), message)[:160]
STATIONARY_M = 0.25
TIGHT_M = 0.45
LOW_CONFIDENCE = 0.55


@dataclass(frozen=True)
class Cause:
    """One kind of failure: how failure messages show it and what usually resolves it."""

    key: str
    label: str
    patterns: tuple[str, ...]
    advice: str


CAUSES = (
    Cause("support-unknown", "感知不完整：不知道物体放在什么上面", (r"no support surface is known",),
          "靠近观察物体和它下面的台面，让支撑关系建立起来"),
    Cause("perception-missing", "感知缺失：需要的物体不在 Ego State 里",
          (r"no [\w ]+ in the ego state", r"no longer in the ego state", r"not seen", r"no drawer seen", r"pieces of tableware are known"),
          "到能看清目标的位置环顾；如果物体确实不在了，请操作员确认任务对象"),
    Cause("out-of-reach", "目标够不着：位置超出手臂或升降范围",
          (r"outside the lift range", r"out of range", r"too far"),
          "确认目标确实是要操作的物体（地面上的扁平物体容易被误认成餐具），或换一个能够到的位置"),
    Cause("low-confidence", "感知不确定：目标的存在概率或位置不确定度过高",
          (r"existence probability", r"uncertainty", r"risk limit"), "靠近、换角度多看几帧再预测"),
    Cause("grasp-miss", "抓取未对准：物体或把手的位置估计有误差", (r"missed", r"not closing"),
          "靠近重新观察目标，修正位置估计后再抓"),
    Cause("base-blocked", "底盘受阻：贴住障碍物或空间不足，无法前进或转身",
          (r"base is blocked", r"turn did not converge", r"no progress"), "后撤离开障碍，检查是否被家具或墙夹住"),
    Cause("arm-blocked", "手臂受阻：手臂或手中物体碰到了家具", (r"joint \w+ did not reach",),
          "检查放置点上方是否被台面或抽屉边缘挡住，换一个放置点"),
    Cause("no-space", "没有可用空间：停放点或放置点都被占用", (r"no free spot", r"too little", r"does not fit"),
          "清理目标区域，或请操作员腾出空间"),
    Cause("path-blocked", "路径或停靠位被挡", (r"in the way", r"collides", r"no free path", r"unreachable", r"not reachable", r"blocked"),
          "搬开可移动的障碍物，或换一个停靠方向"),
)
UNKNOWN = Cause("unknown", "未归类的失败", (), "查看最近失败的原文，必要时请操作员检查现场")


def classify_failure(message: str) -> Cause:
    """The first cause whose pattern appears in ``message`` (:data:`UNKNOWN` when none does).

    :param message: a failure message from a skill, a prediction or the mission itself.
    :returns: the matching cause.
    """
    text = message.lower()
    return next((c for c in CAUSES if any(re.search(p, text) for p in c.patterns)), UNKNOWN)


@dataclass(frozen=True)
class Diagnosis:
    """The dominant cause of recent failures, the findings behind it and what to try next."""

    cause: Cause
    explained: int
    total: int
    findings: tuple[str, ...]

    @property
    def text(self) -> str:
        lines = [f"主要原因是「{self.cause.label}」（最近 {self.total} 次失败中 {self.explained} 次）。"]
        if self.findings:
            lines.append("依据：" + "；".join(f"{i + 1}) {f}" for i, f in enumerate(self.findings)) + "。")
        lines.append(f"建议：{self.cause.advice}。")
        return "\n".join(lines)


def diagnose(failures: list[dict], summary: dict | None, needs: dict[str, int]) -> Diagnosis:
    """Explain a run of failures.

    :param failures: recent failures, oldest first, each with ``message`` and optionally ``pose``
        (the robot's ``[x, y]`` when it failed).
    :param summary: the Ego API summary at diagnosis time, ``None`` when it could not be read.
    :param needs: how many tracks of each category the failed work relies on.
    :returns: the diagnosis.
    """
    causes = [classify_failure(f["message"]) for f in failures]
    counts = Counter(c.key for c in causes)
    by_key = {c.key: c for c in causes}
    top_key, explained = counts.most_common(1)[0] if counts else (UNKNOWN.key, 0)
    findings = []
    if len(counts) > 1:
        findings.append("失败原因分布：" + "、".join(f"{by_key[k].label.split('：')[0]} ×{n}" for k, n in counts.most_common()))
    findings += _motion_findings(failures, summary)
    findings = _failed_target_findings(failures, summary) + findings
    if summary is not None:
        findings += _blocker_findings(failures, summary)
        findings += _target_findings(failures, summary)
        findings += _ego_state_findings(summary, needs)
    return Diagnosis(by_key.get(top_key, UNKNOWN), explained, len(failures), tuple(findings))


def _motion_findings(failures: list[dict], summary: dict | None) -> list[str]:
    poses = [f["pose"] for f in failures if f.get("pose")]
    out = []
    if len(poses) >= 2:
        cx, cy = sum(p[0] for p in poses) / len(poses), sum(p[1] for p in poses) / len(poses)
        if max(math.dist(p, (cx, cy)) for p in poses) <= STATIONARY_M:
            out.append(f"这 {len(poses)} 次失败时机器人基本没有移动（都在 ({cx:.2f}, {cy:.2f}) 附近 {STATIONARY_M:.2f} 米内）")
    if summary is not None:
        me = summary["self"]
        near = [(math.dist((me["x"], me["y"]), _closest_point(o, (me["x"], me["y"]))), o) for o in summary["objects"]
                if o["existence"] >= 0.5 and o["id"] != me.get("holding") and o["position"][2] < 1.2]
        close = sorted((d, o) for d, o in near if d < TIGHT_M)
        if close:
            out.append("机器人现在离 " + "、".join(f"{describe(o)} {d:.2f} 米" for d, o in close[:3])
                       + "，周围空间很窄")
    return out


def _failed_target_findings(failures: list[dict], summary: dict | None) -> list[str]:
    """Each object the failed actions worked on, named, with what was tried and why it failed."""
    names = {o["id"]: describe(o) for o in (summary or {}).get("objects", [])}
    grouped: dict[str, list[dict]] = {}
    for f in failures:
        target = next(iter(re.findall(r"e:\d+", f.get("step") or "")), None) or next(iter(re.findall(r"e:\d+", f["message"])), "—")
        grouped.setdefault(target, []).append(f)
    out = []
    for target, items in grouped.items():
        skill = re.match(r"(pick|place|navigate|open|close|drive_screw|insert_screw_into_hole|pour)\b",
                         items[-1]["message"])
        action = SKILL_ZH[skill[1]] if skill else "处理"
        reasons = list(dict.fromkeys(explain(f["message"], summary) for f in items))
        what = names.get(target, target) if target != "—" else "（未指明对象）"
        out.append(f"{action} {what} 失败 {len(items)} 次：" + "；".join(reasons[:2]))
    return out


def _blocker_findings(failures: list[dict], summary: dict) -> list[str]:
    """The entity that most often blocked a dock pose or a path."""
    hits = Counter(oid for f in failures for oid in
                   re.findall(r"(?:collides with|blocked by) (e:\d+)|(e:\d+) \(\w+\) is in the way", f["message"])
                   for oid in oid if oid)
    if not hits:
        return []
    oid, n = hits.most_common(1)[0]
    obj = next((o for o in summary["objects"] if o["id"] == oid), None)
    name = describe(obj) if obj else f"{oid}（已不在 Ego State 里）"
    return [f"停靠位或路径 {n} 次被 {name} 挡住"]


def _target_findings(failures: list[dict], summary: dict) -> list[str]:
    """Objects the failed actions worked on that look implausible for their category."""
    ids = dict.fromkeys(oid for f in failures for oid in re.findall(r"e:\d+", f.get("step") or ""))
    out = []
    for obj in (o for o in summary["objects"] if o["id"] in ids):
        e = obj.get("extentM") or obj["sizeM"]
        bottom = obj["position"][2] - e[2] / 2
        if obj["category"] in TABLEWARE and bottom < 0.05:
            out.append(f"操作目标 {describe(obj)} 在地面上（高 {e[2]:.2f} 米、底部离地 {max(bottom, 0.0):.2f} 米），"
                       "可能是把地面上的扁平物体误认成了餐具")
    return out


def _closest_point(obj: dict, p: tuple[float, float]) -> tuple[float, float]:
    """The point of the object's footprint closest to ``p``."""
    e = obj.get("extentM") or obj["sizeM"]
    yaw = math.radians(obj["yawDeg"])
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = p[0] - obj["position"][0], p[1] - obj["position"][1]
    lx = max(-e[0] / 2, min(e[0] / 2, c * dx + s * dy))
    ly = max(-e[1] / 2, min(e[1] / 2, -s * dx + c * dy))
    return obj["position"][0] + c * lx - s * ly, obj["position"][1] + s * lx + c * ly


def _ego_state_findings(summary: dict, needs: dict[str, int]) -> list[str]:
    """The objects the failed work needs, by role: missing (with the likeliest candidates),
    duplicated, or held with little confidence (with the rival categories)."""
    out = []
    objects = [o for o in summary["objects"] if o["existence"] >= 0.5]
    for category, expected in needs.items():
        found = [o for o in objects if o.get("role", o["category"]) == category]
        zh = CATEGORY_ZH.get(category, category)
        if not found:
            candidates = sorted((o for o in objects if o["categories"].get(category, 0.0) >= 0.1),
                                key=lambda o: -o["categories"][category])[:2]
            hint = ("；最可能的候选是 " + "、".join(f"{describe(o)}（是{zh}的概率 {o['categories'][category]:.2f}）" for o in candidates)
                    if candidates else "，也没有像它的物体")
            out.append(f"Ego State 里没有{zh}{hint}，需要它的步骤都无法规划")
        elif len(found) > expected:
            out.append(f"Ego State 里有 {len(found)} 个{zh}（任务只需要 {expected} 个），可能有重复轨迹或把别的物体认成了它："
                       + "、".join(f"{o['id']} 在 ({o['position'][0]:.2f}, {o['position'][1]:.2f})" for o in found))
        for o in found:
            confidence = o.get("roleConfidence", o["categoryConfidence"])
            if confidence < LOW_CONFIDENCE:
                rivals = sorted(((p, c) for c, p in o["categories"].items() if c != category), reverse=True)[:2]
                out.append(f"{o['id']} 被认作{zh}的把握只有 {confidence:.2f}（其次 "
                           + "、".join(f"{CATEGORY_ZH.get(c, c)} {p:.2f}" for p, c in rivals) + "）")
    return out
