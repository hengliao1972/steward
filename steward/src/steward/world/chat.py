"""Operator chat with the World Harness and the robots' Ego Harnesses.

Each console message goes to one harness (``world`` or ``robot``) and runs one
``dsh --profile <profile> --json`` process in that harness's own DSH home. The first turn opens a
Session; later turns pass ``--session-id`` so each harness keeps its own conversation. A
``--patch`` overlay points the harness's provider at this host's World Control API or at the
robot's Ego API. Progress is read from the NDJSON run events and kept as one transcript that the
console polls.

A robot turn takes exclusive control of the robot through the Ego API first (stopping the running
skill and pausing other clients such as the scripted shuttle) and releases it when the turn ends.
The Ego Harness keeps its robot-only tool set and empty workspace; the operator's words are its
only extra input.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from steward.net import HttpError, call

WORLD_BRIEF = (
    "你是 Steward 的 World Harness 操作员助手。一个世界仿真正在运行（机器人正在执行任务），"
    "操作员会用自然语言要求对 World State 做运行中的增量修改：移动、添加或删除物体与家具。\n"
    "规则：先用 world_scene 查看当前实体 id 与位置，再用 world_edit 执行修改；"
    "需要新物体时用 asset_search 选资产；不要启动、停止或重载 episode，不要读写文件或运行 shell 命令。"
    "world_edit 被拒绝时（例如穿模、物体正被机器人拿着），说明原因并给出可行的替代方案。"
    "完成后用一两句中文说明做了什么。\n\n操作员请求："
)

# Per-body identity and hand-use rules, inserted after the shared opening. Keys are embodiment ids.
EMBODIMENT_BRIEFS: dict[str, str] = {
    "steward-bot": (
        "本体：你是 Steward 单手移动厨房机器人——平面底盘、升降柱、直线伸出臂，只有一个平行爪。"
        "规划动作和写 run_code 时必须充分利用这一只手：能用手完成的拿、放、拧、翻、对准、固定都要主动做，"
        "不要只导航观察而不动手；同一只手串行处理的多个步骤写进同一个程序循环里做完。\n"
    ),
    "cobot-magic": (
        "本体：你是 AgileX Cobot Magic 双手机器人——差速底盘、升降柱、左右两只 Piper 6-DoF 臂，各带一个平行爪（臂标识 L / R）。"
        "规划动作、写 run_code 和调用技能时必须充分利用双手：能双手并行的就双手一起做（一手持物一手操作、"
        "一手固定一手拧螺丝、双手同时搬运或整理），不要只用单臂而让另一只手闲着；需要指定手臂时用 arm: \"L\" 或 \"R\"。\n"
    ),
}

ROBOT_BRIEF_HEAD = (
    "你是家用服务机器人 robot-1。操作员通过控制台给你下达新任务；在这一轮对话期间你独占控制机器人，"
    "原来的后台任务会暂停，等你这一轮结束后再继续。\n"
)

ROBOT_BRIEF_BODY = (
    "做法：先用 ego_state 看看你已经知道哪些物体；找不到或不确定时用 ego_explore 环顾，或导航靠近观察。"
    "按物体的 role 理解操作员说的「餐边柜」「柜子」「咖啡杯」等指称：role 是结合整个场景推断出的用途标签，"
    "category 只是单个视角下最像的类别，局部视角下家具和相似餐具容易混淆。"
    "每个动作先用 ego_predict 得到收据，风险超限就先靠近观察再预测，然后用 skill_run 执行；"
    "组装带螺丝孔的零件时，先看 ego_state / ego_observe 的 screwHoles（相机看到的为 observed，否则是类别先验）；"
    "摆放或翻侧零件后，必须用 ego_screw_holes 读出世界系孔位；只有 tipSource=observed 且 holesFaceUp 才可信"
    "（place 的 wristRoll 不能当朝上证据，相机会刷新 tip）。holesFaceUp 为 false 时先 place roll=±π/2 再查，"
    "不要去拿下一个零件。place 要对准螺丝孔：把零件放到对应本体上（on=本体），"
    "或放到桌面上并设 alignTo=本体；多组孔用 hole 指定族（如 wing / gear / wheel_0）；不要靠手估 offset。"
    "装到螺丝孔朝侧面的本体上时，ego_predict 会拒绝，直到本体孔朝上。"
    "组装顺序：一次只装一个零件——摆准并立刻用螺丝和螺丝刀拧紧固定，确认后再装下一个；"
    "不要先堆齐所有零件再统一拧螺丝。"
    "把螺丝插入孔用 insert_screw_into_hole（手持 toy_screw，on=带孔零件，可选 hole / maxAttempts）："
    "pick 之后必须先 ego_observe / ego_screw_holes 到 tipSource=observed，分清尖头与螺丝帽；"
    "未观测 tip 时 ego_predict 会拒绝（否则会把未知朝向当成已竖直）。"
    "技能在手里按 tip 把螺丝立成尖端朝下再对准孔放下，松手后稍停并先抬起再回撤，避免碰倒；"
    "再感知确认是否竖直入孔、螺丝帽朝上；若倒了或未入孔会重新拿起再放，直到成功或次数用尽。"
    "拧紧已入孔的螺丝用 drive_screw（target=该螺丝）：须先 insert_screw_into_hole（或等价的尖端朝下 place）把螺丝插好，"
    "再 ego_observe / ego_screw_holes 确认 tipSource=observed 且螺丝竖直入孔；然后 pick 电动螺丝刀"
    "（刀平放且刀身尽量平行靠近机器人的桌边，从侧面抓让刀身横过夹爪；不要端对抓，也不要先 place roll 立刀），"
    "直接 drive_screw 让技能腕滚转刀头朝下按下。"
    "若 drive_screw 报 shaft along the gripper reach，放下并把刀身改成平行近边后再侧面 pick。"
    "侧躺的螺丝也必须让螺杆横过夹爪再 insert；若预测报 shaft along the gripper reach，先放下再直立抓取，"
    "不要反复 place roll=±π/2（端对抓时腕滚转只会自旋，放下后仍横躺）。"
    "技能会把刀身转到侧向再 roll 立起（必须刀头朝下；单靠 roll 立不起来），对准螺丝帽按下拧紧，刀留在手里。"
    "螺丝未竖直入孔或 tip 未观测确认时 ego_predict 会拒绝。不要用普通 place 代替 insert_screw_into_hole，也不要用 place 代替拧紧。"
    "装配汽车时把车身放到桌上：place 会强制/自动 roll=±π/2 使侧孔朝上；摆好后必须用 ego_screw_holes 确认 "
    "holesFaceUp。在车身孔朝上之前，ego_predict 会拒绝 pick 车轮。车轮中央是通孔（through）：平放轴线竖直即可"
    "从上插入，facing 读到 down 也不必 flip 轮子。装每个轮：先 insert_screw_into_hole 把螺丝尖端朝下插入轮子中央孔"
    "（on=轮，勿 flip/横放）；ego_predict 会拒绝非竖直插入。"
    "再把带螺丝的轮子对准车身孔 place（alignTo/hole），然后用立起的螺丝刀拧紧；一侧装完再翻到另一侧同样操作。"
    "执行后用 ego_state 自查结果。宣布完成之前，如果任务要求结果和某个参照图或参照物一致（例如按盒盖图案拼图），"
    "必须用独立证据核对最终状态：把摆好的排列合成图或逐块 ego_match_face 得分与参照比较，相关系数要明显高于随机排列，"
    "达不到就继续修正或调用 reflect，不要只数格位是否占满、正反面是否朝上。完成或无法完成时，用一两句中文向操作员汇报。\n"
    "新视角生成通过 rgbd-video-gen 或 novelanything skill 及其工具调用；明确指定模型时不得替换。"
    "模型输入输出保存在机器人工作空间 inputs/outputs；使用工具返回的真实路径和 local_manifest_path，勿猜文件名。"
    "运动过程取证先用 ego_rgb_recording_start 按要求fps录制 head_camera RGB，运动结束后用 ego_rgb_recording_stop；"
    "录制按仿真时钟采样，不用重复快照模拟帧率。成功后用 rgbd_video_gen_prepare_recording 或"
    " novelanything_prepare_gripper_view 准备输入，位姿来自机器人里程计、编码器和标定。"
    "机器人现有 head_camera 与 wide_camera 的布局和感知分工保持原样；生成的 RGBD 是预测，不是真实传感器观测。\n"
    "不要反复试错：ego_predict 的 issues 里给出建议时（例如「原地放下」的参数）先照做；同一个目标连续 3 次预测都不可行，"
    "或同一个技能失败 2 次，就停止尝试并调用 report_problem，然后向操作员说明。遇到工具报错或异常且无法绕过时，同样调用 report_problem。"
    "报告前先回看你自己这一轮的工具调用记录做诊断：哪个技能或预测失败、报错原文、重复了几次、失败时你在哪里、"
    "和 blockers、不确定度、Ego State 里缺失或重复的物体有什么关系；报告里写清你要达成的目标、卡在哪一步、"
    "诊断出的原因和依据、已经尝试过什么、需要操作员做什么。\n"
    "感知给物体起的名字不一定是操作员的叫法：分割网络按外观命名（印着图案的盒子可能叫 picture），分类器只认识它有形状先验的类别。"
    "找不到指令里的物体、或一步失败时，先对照指令里的描述（大小、颜色、位置、和其他物体的关系）想一想是不是认错了物体，"
    "用 ego_candidates 传入指令里的名字和大小找出可能的对象，核实后用 ego_annotate 改正。"
    "感知也可能留下根本不存在的物体：ego_state 里标着 unsupported 的物体悬在空中、下面没有任何东西托着，或者嵌在别的物体里，"
    "这在物理上不可能。遇到这种物体，先别拿它做计划，走到能看清它所在位置的地方，用 ego_observe 再看几次（必要时换个角度）；"
    "重新看到了，就以新的观测为准；它所在的位置已经清楚可见、却一直感知不到（lastSeenAgoS 一直在增长），"
    "就用 ego_forget 把它从 Ego State 删除，reason 里写明它为什么不合理、你是怎样复查的。\n"
    "机器人工具或程序失败后，你会收到一条【反思】消息，里面有原始指令、之前的失败、失败归类和相关物体的证据；"
    "同类失败再次出现或多次反思仍受阻时，收到的是【深度反思】，还附有场景摘要；连续多个程序没有反思时，会收到【阶段回顾】。"
    "【反思】和【阶段回顾】逐条回答，每条一句话；【深度反思】先在思考中充分推敲（先宏观：任务、场景和指令的措辞；再微观：物理原因；"
    "最后是方法和效率），再分层逐条作答。回答完再决定下一步。没有工具报错、但结果和目标不符或一直没有进展时，"
    "在程序里调用 reflect 描述困难并打印返回的问题；认为做不到、或想改用低于指令要求的做法（例如不按图案对位）之前，"
    "也先调用 reflect 深度反思，核实这个结论的依据没有被你自己的动作改变（例如参照物被压住或翻面）。\n"
    "先想清楚为什么不可行：预测结果的 blockers 列出挡路的物体；如果是可搬动的（movable，例如椅子），"
    "先把它搬到 parkAt 给出的不碍事位置，再重新预测原来的动作；不可搬动的障碍才需要换做法或报告。\n"
    "放下任何东西之前（放置、暂放、倾倒、放回腾空的容器、搬开挡路的物体），先想它对后面的动作有什么影响："
    "列出剩下的步骤各要用哪块台面或地面、要多大地方（主要的工作区、倒出的东西摊开的范围、之后还要放下的容器和盖子），"
    "按 ego_state 里的尺寸和位置先给它们分好地方，再决定这一次放在哪里。暂时不用的东西靠边、靠角、彼此挨紧，占地越少越好，"
    "把最大的一块连续空白留给主要工作；倾倒时让倒出的范围偏向一侧，给之后要放下的容器留出位置。"
    "台面确实放不下时可以放到地面：放在墙角或贴墙，避开家具前面的停靠位和机器人要走的路线（预测给出的 parkAt 就是这样的位置）。"
    "放下后用 ego_state 核实剩下的地方还够后面的步骤用；不够就先挪开刚放的东西，不要把后面的东西硬塞进去。"
    "这条对任何任务都适用：选位置时既看这一步能不能做成，也看会不会妨碍之后的动作。\n"
    "工具都在 run_code 程序里调用。每个程序结束后你都要停下来思考，这段时间机器人原地不动，思考越多任务越慢，所以：\n"
    "一个程序完成一整段工作：用循环处理一组物体（例如逐个拿起、检查、放好），每一轮从 ego_state 或上一步的结果里取得这一轮的参数"
    "（下一个物体、目标位置、备选的做法），每一步检查预测和技能结果；程序能判断的情况（换备选位置、跳过已完成的物体、重试一次）"
    "在程序里处理，只有需要你重新判断时才打印原因并结束程序。不要每个动作单独调用一次 run_code，也不要为同一段工作反复生成相似的程序；"
    "长的循环可以把 timeoutMs 设到上限。\n"
    "程序里声明的函数和变量会留在内存里，同一轮对话的后续程序可以直接用：第一次把常用操作写成函数（导航、预测并执行、坐标换算、比对图案），"
    "之后直接调用，不要重写。函数用 function 声明或赋给 var 变量；一次性的变量放进 { ... } 块里，避免重复声明 const 或 let 报错。"
    "操作员发来新消息后内存会清空，用 typeof 检查，不存在再定义。\n"
    "数值计算、颜色比对和坐标换算都写进程序里算，只打印结论；不要把颜色网格或大量坐标打印出来，再在思考里逐个比较或心算。"
    "思考和回复都要简短：只写决定和关键依据，不复述工具结果、之前的计划或指令；程序的 description 用一句话。\n"
    "任务没有现成技能能直接完成时（例如需要比对图案、按图摆放许多物体），先用 capability_list 查你以前保存的能力："
    "有合适的 recipe 就直接用 capability_get 取出来运行（code 是以 tools 和 args 为参数的 async 函数体："
    "var AsyncFunction = (async () => {}).constructor; var run = new AsyncFunction('tools', 'args', cap.code); "
    "await run(tools, {...})，不要重写它）；没有就自己写程序，把 ego_inspect（看清物体顶面）、"
    "ego_match_face（找出物体的图案在参考图上的位置和转向；机器人会记住看到过的物体顶面，先走到能看清所有物体的地方，再一次传入全部物体比对；小物体的图案要走近才看得清："
    "ego_inspect 结果的 sourcePx 是顶面在相机图像里的像素宽，带 detailHint 时说明还不够清楚并给出要走到多近，"
    "先按提示走近再看、再比对，不要用看不清的图案硬配；图案比对用 ego_match_face 时把 px 提到 24–32（上限 32），"
    "让相机分辨率真正用到匹配上；卡片、拼图块、照片这类扁平物体可能背面朝上，背面通常是均匀的单一颜色，"
    "判断正反面用 ego_inspect 的 plain（true）或 interiorColors（约 20 以下即背面硬纸板）；"
    "不要用 spread / interiorSpread——桌面光照会让单色背面的 RGB 标准差达到 25–35，看起来像有图案；"
    "拼图块挨着时边缘格还会混进邻块颜色；"
    "判断前先把互相压着的摊开、再走近看清，判定背面朝上的用 place 的 flip 翻过来再比对，侧放用 place 的 roll=π/2）、ego_annotate（标注你认出的物体类别与可供性）"
    "和 ego_predict / skill_run 组合起来，并尽量复用已有的 skill。做成之后用 capability_save 保存："
    "把每个能用在别的物体上的操作存成 skill（例如把一堆物体摊开、把扁平物体翻成正面、按网格摆放），"
    "再存一个只负责组合这些 skill 的 recipe；物体 id、位置和尺寸都放进参数，做不到预期结果时让代码抛出错误。"
    "任务结束前整理一次能力库：删掉被更通用的版本取代的、重复的和标为 deprecated 的能力。\n\n"
)

ROBOT_FOLLOW_UP = "操作员任务："


def robot_brief(embodiment_id: str | None = None) -> str:
    """The Ego Harness system brief for a robot body: shared rules plus that body's hand-use rules.

    :param embodiment_id: a registered id (``steward-bot``, ``cobot-magic``); unknown or None uses
        the single-hand Steward body so a missing registry never omits hand-use guidance.
    :returns: the full first-turn prefix ending in :data:`ROBOT_FOLLOW_UP`.
    """
    body = EMBODIMENT_BRIEFS.get(embodiment_id or "steward-bot", EMBODIMENT_BRIEFS["steward-bot"])
    return ROBOT_BRIEF_HEAD + body + ROBOT_BRIEF_BODY + ROBOT_FOLLOW_UP


# Default brief (single-hand Steward); the host replaces it when a scenario's body is known.
ROBOT_BRIEF = robot_brief("steward-bot")

ROBOT_OWNER = "operator"
TURN_BUDGET_S = 24 * 24 * 3600.0


def capabilities_digest(home: Path | None, *, max_description: int = 120) -> str:
    """A short list of saved capabilities to prepend to the first robot turn, so the model sees
    what it can reuse without calling ``capability_list`` first.

    :param home: the robot's DSH home (``DSH_HOME``); ``None`` or a missing library yields "".
    :param max_description: longest description kept per capability.
    :returns: a Chinese paragraph ending with a blank line, or "" when there is nothing to list."""
    if home is None:
        return ""
    root = home / "steward-capabilities"
    if not root.is_dir():
        return ""
    lines: list[str] = []
    for path in sorted(root.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        name = record.get("name")
        if not isinstance(name, str) or not name:
            continue
        layer = record.get("layer", "?")
        status = record.get("status", "?")
        description = record.get("description", "")
        if not isinstance(description, str):
            description = ""
        if len(description) > max_description:
            description = description[: max_description - 1] + "…"
        lines.append(f"- {name}（{layer}，{status}）：{description}")
    if not lines:
        return ""
    return ("你已保存的能力（本轮开始时的清单；有合适的先用 capability_get 取出来运行，不要重写）：\n"
            + "\n".join(lines) + "\n\n")


def _terminate(proc: subprocess.Popen) -> None:
    """End a harness run and any process it started (its whole process tree)."""
    try:
        if os.name == "nt":  # os.killpg is POSIX-only; taskkill /T takes the launcher cmd and node down together
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=10)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except OSError:  # the run already exited
        pass


def load_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines of a dotenv file; missing files yield no variables."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip().removeprefix("export ").strip()] = value.strip().strip("'\"")
    return out


@dataclass
class ChatConfig:
    """Where the harnesses live. ``robot_home`` / ``ego_api`` may be unset to disable robot chat."""

    dsh: Path
    cwd: Path
    env_file: Path
    world_home: Path | None = None
    robot_home: Path | None = None
    ego_api: str | None = None
    timeout_s: float = TURN_BUDGET_S  # wall seconds a turn may run before it is ended
    stop_wait_s: float = 15.0  # how long a stopped turn may take to end before a new task gives up on it


@dataclass
class _Agent:
    key: str
    label: str
    profile: str
    home: Path | None
    cwd: Path
    brief: str
    follow_up: str
    patch: Callable[[], str]
    before_turn: Callable[[], None] | None = None
    after_turn: Callable[[], None] | None = None
    on_failure: Callable[[str], None] | None = None
    extra_reason: Callable[[], str | None] | None = None
    session_id: str | None = None
    busy: bool = False
    proc: subprocess.Popen | None = None
    cancelled: bool = False
    # What the harness is waiting on during a busy turn: ``llm`` while the model answers,
    # ``run_code`` / ``tool`` while a tool call is open. Cleared when the turn ends.
    wait_kind: str | None = None
    wait_since: float | None = None
    wait_tool: str | None = None
    wait_detail: str | None = None


@dataclass
class _Entry:
    id: int
    agent: str
    role: str  # user | assistant | tool | error | system
    text: str
    t: float = field(default_factory=time.time)
    tool: str | None = None
    input: object = None
    status: str | None = None
    call_id: str | None = None

    def to_dict(self) -> dict:
        d = {"id": self.id, "agent": self.agent, "role": self.role, "text": self.text, "t": self.t}
        if self.tool is not None:
            d |= {"tool": self.tool, "input": self.input, "status": self.status}
        return d


class HarnessChat:
    def __init__(self, config: ChatConfig | None, wca_url: Callable[[], str]):
        self.config = config
        self._lock = threading.Lock()
        self._entries: list[_Entry] = []
        self._next_id = 1
        self.agents: dict[str, _Agent] = {}
        self._seen_alerts: set[int] = set()
        if config is None:
            return
        self.agents["world"] = _Agent(
            key="world", label="World Harness", profile="world-task", home=config.world_home, cwd=config.cwd,
            brief=WORLD_BRIEF, follow_up="操作员请求：",
            patch=lambda: f"- id: steward-world-api\n  config:\n    endpoint: '{wca_url()}'\n")
        robot_home = config.robot_home
        self.agents["robot"] = _Agent(
            key="robot", label="robot-1", profile="ego", home=robot_home,
            cwd=(robot_home / "workspace") if robot_home else config.cwd,
            brief=ROBOT_BRIEF, follow_up=ROBOT_FOLLOW_UP,
            patch=lambda: (f"- id: steward-ego-runtime\n  config:\n    endpoint: '{config.ego_api}'\n"
                           f"    owner: {ROBOT_OWNER}\n"),
            before_turn=lambda: self._robot_control("acquire"),
            after_turn=lambda: self._robot_control("release"),
            on_failure=lambda message: self._robot_alert(message),
            extra_reason=lambda: None if config.ego_api else "the robot's Ego API is not known to this host")

    def set_robot_embodiment(self, embodiment_id: str | None) -> None:
        """Refresh the robot agent's first-turn brief for the body now in the scene.

        :param embodiment_id: registered body id, or None to keep the Steward single-hand brief.
        """
        if "robot" not in self.agents:
            return
        self.agents["robot"].brief = robot_brief(embodiment_id)

    # ------------------------------------------------------------ public API
    def unavailable_reason(self, key: str) -> str | None:
        agent = self.agents.get(key)
        if agent is None or self.config is None:
            return "operator chat is disabled"
        if not self.config.dsh.exists():
            return f"DSH launcher not found at {self.config.dsh}; run steward/scripts/setup-dsh-profiles.sh"
        if agent.home is None or not (agent.home / "profiles" / agent.profile).exists():
            return f"profile {agent.profile!r} missing in {agent.home}; run steward/scripts/setup-dsh-profiles.sh"
        return agent.extra_reason() if agent.extra_reason else None

    def state(self, since: int = 0, activity_since: int = 0) -> dict:
        robot = {}
        if "robot" in self.agents and self.config and self.config.ego_api:
            robot = {"control": self._robot_owner(), "alerts": self._robot_alerts(),
                     "plan": self._robot_plan(activity_since)}
        with self._lock:
            agents = {key: {"label": a.label, "available": self.unavailable_reason(key) is None,
                            "reason": self.unavailable_reason(key), "busy": a.busy, "sessionId": a.session_id,
                            "wait": self._wait_of(a)}
                      for key, a in self.agents.items()}
            messages = [e.to_dict() for e in self._entries if e.id > since]
        if robot:
            agents["robot"].update(robot)
        return {"agents": agents, "messages": messages}

    def send(self, key: str, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            raise HttpError(400, "message is empty")
        if key not in self.agents:
            raise HttpError(400, f"unknown chat target {key!r}; use one of {sorted(self.agents)}")
        reason = self.unavailable_reason(key)
        if reason:
            raise HttpError(503, reason)
        agent = self.agents[key]
        with self._lock:
            if agent.busy:
                raise HttpError(409, f"{agent.label} is still working on the previous message")
            agent.busy = True
            self._set_wait(agent, "llm")
            entry = self._add(key, "user", text)
            first = agent.session_id is None
        threading.Thread(target=self._run, args=(agent, text, first), name=f"chat-{key}", daemon=True).start()
        return {"accepted": True, "id": entry.id}

    def reset(self, key: str) -> dict:
        agent = self.agents.get(key)
        if agent is None:
            raise HttpError(400, f"unknown chat target {key!r}")
        with self._lock:
            if agent.busy:
                raise HttpError(409, "cannot start a new conversation while a message is running")
            agent.session_id = None
            self._add(key, "system", f"已开始新的 {agent.label} 会话")
        return {"reset": True}

    def clear(self) -> None:
        """Empty the transcript of every harness and start each idle one on a new conversation (an
        episode reset). A harness still running keeps its conversation for the rest of its turn."""
        with self._lock:
            self._entries.clear()
            for agent in self.agents.values():
                if not agent.busy:
                    agent.session_id = None

    def cancel(self, key: str) -> dict:
        """Stop the running turn of one harness; robot control is released when the turn ends."""
        agent = self.agents.get(key)
        if agent is None:
            raise HttpError(400, f"unknown chat target {key!r}")
        proc = agent.proc
        if not agent.busy or proc is None or proc.poll() is not None:
            return {"stopped": False}
        agent.cancelled = True
        _terminate(proc)
        with self._lock:
            self._add(key, "system", f"已停止 {agent.label} 的本轮任务")
        if key == "robot" and self.config and self.config.ego_api:
            try:
                call("POST", f"{self.config.ego_api}/skill/stop", {})
            except (HttpError, OSError) as exc:
                self._report(agent, f"停止机器人动作失败：{exc}")
        return {"stopped": True}

    def stop_turn(self, key: str) -> None:
        """Stop the running turn of ``key``, if any, and return once it has ended; raises
        ``HttpError(504)`` when it is still running after ``stop_wait_s``."""
        agent = self.agents.get(key)
        if agent is None:
            raise HttpError(400, f"unknown chat target {key!r}")
        deadline = time.monotonic() + self.config.stop_wait_s
        stopped = None
        while agent.busy:
            proc = agent.proc
            if proc is not None and proc is not stopped:
                self.cancel(key)
                stopped = proc
            if time.monotonic() > deadline:
                raise HttpError(504, f"{agent.label} did not end its running turn within {self.config.stop_wait_s:g} s")
            time.sleep(0.05)

    def assign(self, key: str, text: str) -> dict:
        """Give ``key`` the task ``text`` in a new conversation, its running turn stopped first.

        Returns:
            What :meth:`send` returns for the task message.
        """
        self.stop_turn(key)
        self.reset(key)
        return self.send(key, text)

    def stop(self) -> None:
        for agent in self.agents.values():
            proc = agent.proc
            if proc is not None and proc.poll() is None:
                _terminate(proc)

    # -------------------------------------------------------------- internals
    def _robot_control(self, action: str) -> None:
        call("POST", f"{self.config.ego_api}/ego/control", {"action": action, "owner": ROBOT_OWNER})

    def _robot_alerts(self) -> list[dict]:
        """Unacknowledged robot alerts; each new one is also added to the transcript as an error."""
        try:
            active = call("GET", f"{self.config.ego_api}/ego/alerts", timeout=0.5)["active"]
        except (HttpError, OSError, KeyError, TypeError):  # no API, or an answer we do not understand
            # A robot-side fault is not the operator console's fault: the console polls this
            # endpoint continuously, and an exception here takes the whole chat down with it.
            return []
        with self._lock:
            for alert in active:
                if alert["id"] in self._seen_alerts:
                    continue
                self._seen_alerts.add(alert["id"])
                source = {"agent": "机器人智能体", "skill": "技能执行", "runtime": "感知", "api": "Ego API",
                          "harness": "Ego Harness", "mission": "家务任务"}.get(alert["source"], alert["source"])
                context = "".join(f"\n{label}：{alert[key]}" for key, label in
                                  (("goal", "目标"), ("step", "当前步骤"), ("diagnosis", "自我诊断")) if alert.get(key))
                self._add("robot", "error", f"⚠ 机器人遇到无法自行处理的情况（{source}）：{alert['message']}{context}")
        return active

    def _robot_plan(self, activity_since: int = 0) -> dict | None:
        """The plan the robot last published, the skill it is running and its activity entries
        newer than ``activity_since``; ``None`` when unknown."""
        try:
            return call("GET", f"{self.config.ego_api}/ego/plan?since={int(activity_since)}", timeout=0.5)
        except (HttpError, OSError):  # the robot's Ego API is down or predates plans; nothing to show
            return None

    def acknowledge_alerts(self) -> dict:
        if not (self.config and self.config.ego_api):
            raise HttpError(503, "the robot's Ego API is not known to this host")
        return call("POST", f"{self.config.ego_api}/ego/alerts/ack", {})

    def _robot_alert(self, message: str) -> None:
        try:
            call("POST", f"{self.config.ego_api}/ego/alert", {"message": message, "source": "harness", "level": "error"})
        except (HttpError, OSError):  # the alert stays visible in the transcript as an error entry
            pass

    def _robot_owner(self) -> str | None:
        try:
            return call("GET", f"{self.config.ego_api}/ego/control", timeout=0.5)["owner"]
        except (HttpError, OSError):  # the robot's Ego API is down; the console shows no owner
            return None

    def _add(self, agent: str, role: str, text: str, **extra) -> _Entry:
        entry = _Entry(self._next_id, agent, role, text, **extra)
        self._next_id += 1
        self._entries.append(entry)
        del self._entries[:-400]
        return entry

    def post_error(self, agent: str, text: str) -> None:
        """Add an error line to the transcript under ``agent`` (for example a perception fault the
        world found), so the operator reads every alert in the chat."""
        with self._lock:
            self._add(agent, "error", text)

    def post_note(self, agent: str, text: str) -> None:
        """Add a system line to the transcript under ``agent``: something the operator should know
        that is not a fault."""
        with self._lock:
            self._add(agent, "system", text)

    def _report(self, agent: _Agent, text: str) -> None:
        with self._lock:
            self._add(agent.key, "error", text)

    def _run(self, agent: _Agent, text: str, first: bool) -> None:
        cfg = self.config
        fd, name = tempfile.mkstemp(prefix=f"steward-{agent.key}-", suffix=".yml")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(agent.patch())
        patch = Path(name)
        args = [str(cfg.dsh), "--profile", agent.profile, "--patch", str(patch), "--json"]
        if agent.session_id:
            args += ["--session-id", agent.session_id]
        args.append("-")
        env = os.environ | load_env_file(cfg.env_file) | {"DSH_HOME": str(agent.home)}
        stderr_tail: list[str] = []
        controlled = False
        try:
            if agent.before_turn is not None:
                agent.before_turn()
                controlled = True
            agent.cwd.mkdir(parents=True, exist_ok=True)
            proc = agent.proc = subprocess.Popen(args, cwd=agent.cwd, env=env, stdin=subprocess.PIPE,
                                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                                 encoding="utf-8", errors="replace",  # dsh speaks UTF-8; the Windows
                                                 start_new_session=True)  # locale (GBK) would kill the reader
            prefix = agent.brief if first else agent.follow_up
            if first and agent.key == "robot":
                # Reflection quotes the text after the last follow_up marker as the instruction.
                prefix = prefix.removesuffix(agent.follow_up) + capabilities_digest(agent.home) + agent.follow_up
            proc.stdin.write(prefix + text)
            proc.stdin.close()
            err_reader = threading.Thread(target=lambda: stderr_tail.extend(proc.stderr), daemon=True)
            err_reader.start()
            agent.cancelled = False
            timed_out = False

            def _on_timeout() -> None:
                nonlocal timed_out
                timed_out = True
                _terminate(proc)

            timer = threading.Timer(cfg.timeout_s, _on_timeout)
            timer.start()
            last_text: str | None = None
            for line in proc.stdout:
                last_text = self._ingest(agent, line, last_text)
            code = proc.wait()
            timer.cancel()
            err_reader.join(2)
            if timed_out:
                with self._lock:
                    self._add(agent.key, "system",
                              f"{agent.label} 回合预算（{cfg.timeout_s:.0f} 秒）到时，本轮已按设计结束，非异常")
            elif code != 0 and code != -15 and not agent.cancelled:
                detail = "".join(stderr_tail[-6:]).strip() or f"exit code {code}"
                self._report(agent, f"{agent.label} 运行失败：{detail}")
                if agent.on_failure is not None:
                    agent.on_failure(f"{agent.label} 运行失败：{detail[-400:]}")
        except HttpError as exc:
            self._report(agent, f"无法接管机器人：{exc}")
        except OSError as exc:
            self._report(agent, f"无法启动 {agent.label}：{exc}")
        finally:
            patch.unlink(missing_ok=True)
            if controlled and agent.after_turn is not None:
                try:
                    agent.after_turn()
                except (HttpError, OSError) as exc:
                    self._report(agent, f"交还机器人控制失败：{exc}")
            with self._lock:
                agent.busy = False
                agent.proc = None
                self._clear_wait(agent)

    def _wait_of(self, agent: _Agent) -> dict | None:
        """Live wait state for the console progress bar, or ``None`` when the agent is idle."""
        if not agent.busy or agent.wait_kind is None or agent.wait_since is None:
            return None
        out: dict = {"kind": agent.wait_kind, "since": agent.wait_since}
        if agent.wait_tool:
            out["tool"] = agent.wait_tool
        if agent.wait_detail:
            out["detail"] = agent.wait_detail
        return out

    @staticmethod
    def _set_wait(agent: _Agent, kind: str, tool: str | None = None, detail: str | None = None) -> None:
        agent.wait_kind = kind
        agent.wait_since = time.time()
        agent.wait_tool = tool
        agent.wait_detail = detail

    @staticmethod
    def _clear_wait(agent: _Agent) -> None:
        agent.wait_kind = None
        agent.wait_since = None
        agent.wait_tool = None
        agent.wait_detail = None

    @staticmethod
    def _tool_wait_detail(inp: object) -> str | None:
        """A short label for an open tool call, taken from its common description fields."""
        if not isinstance(inp, dict):
            return None
        for key in ("description", "goal", "summary", "title"):
            value = inp.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:80]
        return None

    def _ingest(self, agent: _Agent, line: str, last_text: str | None) -> str | None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return last_text
        kind = event.get("type")
        with self._lock:
            if kind == "session":
                agent.session_id = event.get("sessionId") or agent.session_id
            elif kind == "text" and event.get("text", "").strip():
                last_text = event["text"].strip()
                self._add(agent.key, "assistant", last_text)
            elif kind == "tool_call":
                tool = event.get("tool")
                self._add(agent.key, "tool", "", tool=tool, input=event.get("input"), status="running",
                          call_id=event.get("callId"))
                wait_kind = "run_code" if tool == "run_code" else "tool"
                self._set_wait(agent, wait_kind, tool=tool, detail=self._tool_wait_detail(event.get("input")))
            elif kind == "tool_result":
                for entry in reversed(self._entries):
                    if entry.role == "tool" and entry.call_id == event.get("callId"):
                        entry.status = event.get("status")
                        entry.text = str(event.get("result", ""))[:1200]
                        break
                self._set_wait(agent, "llm")
            elif kind == "final":
                agent.session_id = event.get("sessionId") or agent.session_id
                final = (event.get("text") or "").strip()
                if final and final != last_text:
                    self._add(agent.key, "assistant", final)
                    last_text = final
        return last_text
