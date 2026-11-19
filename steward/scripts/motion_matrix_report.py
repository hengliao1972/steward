"""Paired reports for motion matrices with explicitly recorded segmentation and test scope."""

import os
from pathlib import Path


def case_name(node, chinese):
    """Name cases by function and parameter, independent of their filtered row index."""
    function = node.split("::")[-1].split("[")[0]
    names = {
        "test_cup_to_table_end_to_end": ("Cup to table", "杯子放桌"),
        "test_shuttle_moves_objects_around_the_cycle": ("Shuttle", "往返搬运"),
        "test_a_chair_is_carried_and_put_down_on_the_floor": ("Chair to floor", "椅子搬到地面"),
        "test_picking_a_chair_by_the_wall_leaves_room_to_move_and_it_can_be_put_down_in_place": ("Chair beside wall", "靠墙椅子原地放回"),
        "test_a_chair_blocking_the_sideboard_is_moved_out_of_the_way_before_the_drawer_swap": ("Clear chair and swap drawers", "移开挡路椅子并交换抽屉"),
        "test_one_chores_round_is_confirmed_by_world_state": ("Full chores round", "完整家务轮次"),
        "test_pieces_dropped_behind_and_under_the_table_are_picked_off_the_floor_and_put_back": ("Floor piece recovery", "地面拼图回收"),
        "test_phase1_exploration_builds_an_accurate_ego_state_from_rgbd": ("RGB-D exploration", "RGB-D 探索建图"),
    }
    if function == "test_the_robot_takes_off_the_lid_and_pours_the_pieces_out_onto_the_table":
        return ("狗形拼图倾倒" if chinese else "Dog puzzle pouring") if "puzzle-dog" in node else ("猫形拼图倾倒" if chinese else "Cat puzzle pouring")
    return names.get(function, (function, function))[int(chinese)]


def table(headers, rows):
    """Render rows with a fixed column count as a CommonMark table."""
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def write_reports(payload, output):
    """Render results and an actual terminal scene, distinguishing simulation semantics from SAM."""
    stages, cases = payload["stages"], payload["cases"]
    manifest = payload["manifest"]
    simulation = manifest.get("maskSource", "sam") == "simulation"
    baseline = [row for row in cases if row["stage"] == "baseline"]
    for chinese in (False, True):
        def t(en, zh):
            return zh if chinese else en
        labels = [t("Baseline", "基线"), t("Drift", "累计漂移"), t("Drift + feedback", "累计漂移＋融合"), t("No drift + feedback", "无累计漂移＋融合")]
        text = [t("# Motion control matrix", "# 运动控制四组实验"),
                t("English | [中文](report.zh.md)", "[English](report.md) | 中文"), t("## Conditions", "## 实验条件")]
        text.append(t(f"Segmentation source: **{'simulation semantic masks' if simulation else 'SAM'}**. The frozen set contains {len(baseline)} parametrized cases; each cell runs once in a separate process.",
                      f"分割来源：**{'仿真语义掩膜' if simulation else 'SAM'}**。固定清单包含 {len(baseline)} 个参数化用例，每格在独立进程运行一次。"))
        text.append(t(f"Cumulative drift is 1% distance scale and 0.005 rad/m. Instantaneous odometry position/yaw sigmas remain 0.003. Original scenarios, seeds, assertions and internal deadlines are retained; the common external ceiling is {manifest['caseTimeoutSeconds']:g} seconds. [Protocol](protocol.json), [manifest](manifest.json), [screening](screening.json) and [source fingerprints](source-sha256.json) record the experiment.",
                      f"累计漂移为距离比例 1% 和每米 0.005 rad 航向漂移，瞬时里程计位置／航向标准差仍为 0.003。场景、种子、断言和内部截止时间沿用原测试，每例统一外部上限 {manifest['caseTimeoutSeconds']:g} 秒。[协议](protocol.json)、[清单](manifest.json)、[筛选记录](screening.json)和[源码指纹](source-sha256.json)记录实验设置。"))
        if simulation:
            text.append(t("This is a test-only ideal-semantics iteration: shuffled simulation instance masks retain their existing boundary degradation and small-fragment filtering; depth noise is unchanged. Category scores are 1.0 from simulator metadata. Only localization receives those labels; main object perception is unchanged. No robot GT poses, entity identifiers or GT map are supplied to the localizer. SAM calls are forbidden by an assertion. This is not a real-SAM accuracy measurement.",
                          "这是仅用于测试的理想语义快速迭代：仿真实例掩膜保留原有随机编号、边缘退化和小碎片过滤，深度噪声不变；类别标签来自仿真元数据，置信度为 1.0。仅定位器使用这些标签，主对象感知保持原样。定位器不接收机器人 GT 位姿、实体标识或 GT 地图，任何 SAM 调用都会触发断言。这不是实际 SAM 分割精度测量。"))
        removed = manifest.get("removedBaselineFailures", {})
        if removed:
            root = Path(__file__).resolve().parents[2]
            evidence = Path(os.path.relpath(root/manifest["baselineEvidence"], output)).as_posix()
            text += [t("## Frozen exclusions", "## 固定剔除项"),
                     t(f"Baseline failures were excluded according to the [selection evidence]({evidence}). Selection uses only baseline outcomes, never drift or feedback outcomes. A selected baseline's 100% rate is by construction, not an independent reliability estimate. Raw outcomes are retained separately.",
                       f"依据[筛选记录]({evidence})剔除基线失败项。筛选只使用基线结果，不使用漂移或融合组表现；筛选后基线的 100% 是清单构造结果，不是独立可靠性估计。原始结果另行保留。"),
                     table([t("Task", "任务"), "Test node"], [(case_name(node, chinese), f"`{node}`") for node in removed])]
        text += [t("## Results", "## 结果汇总"), table([t("Group", "组别"), t("Pass", "通过"), t("Fail", "失败"), t("Timeout", "超时"), t("Skip", "跳过"), t("Rate", "通过率"), t("Seconds", "总耗时（秒）")],
                 [[label, s["passed"], s["failed"], s["timeout"], s["skipped"], f"{s['passRate']:.1%}", f"{s['wallSeconds']:.2f}"] for label, s in zip(labels, stages)])]
        rows = []
        for index, original in enumerate(baseline):
            row = [f"{index}: {case_name(original['nodeid'], chinese)}"]
            for stage in stages:
                result = next(r for r in cases if r["stage"] == stage["stage"] and r["nodeid"] == original["nodeid"])
                status = {"passed": "通过", "failed": "失败", "timeout": "超时", "skipped": "跳过"}[result["status"]] if chinese else result["status"]
                row.append(f"[{status}]({result['directory']}/pytest.log) {result['wallSeconds']:.1f}s")
            rows.append(row)
        text += [t("## Case results", "## 逐例结果"), table([t("Task", "任务"), *labels], rows)]
        text += [t("## Feedback coverage", "## 融合链路覆盖"), table([t("Group", "组别"), t("Header reads", "投影头读取"), t("Visual frames", "视觉帧"), t("Accepted updates", "接受校正"), t("Simulation frames", "仿真掩膜帧"), "SAM", t("Segmentation errors", "分割错误")],
                 [[label, *(s.get(key, 0) for key in ("mappedHeaders", "visualFrames", "acceptedVisual", "simulatedMaskFrames", "independentSamRequests", "visualErrors"))] for label, s in zip(labels, stages)])]
        if "extended" in payload:
            rows = []
            for r in payload["extended"]["cases"]:
                label = labels[[s["stage"] for s in stages].index(r["stage"])]
                suffix = t(" (internal deadline)", "（仍达内部时限）") if r["internalDeadlineReached"] else ""
                rows.append([label, case_name(r["nodeid"], chinese), f"[{r['status']}{suffix}]({r['directory']}/pytest.log)", f"{r['wallSeconds']:.2f}"])
            text += [t("## Extended reruns", "## 延时重测"), t("Only identified deadline failures are rerun. Actual override values are saved in each deadline-overrides.json. Assertions and motion thresholds are unchanged; these are fresh simulations, not exact input replays.", "只重测确认的时限失败；实际覆盖值保存在每例 deadline-overrides.json 中，断言和运动阈值不变。重测是新的仿真，并非固定输入回放。"), table([t("Group", "组别"), t("Task", "任务"), t("Outcome", "结果"), t("Seconds", "耗时（秒）")], rows),
                     t("The composite substitutes only these reruns, retaining all other original cells; it is not a second full matrix.", "综合结果仅替换这些重测项，其余沿用原结果，不是第二轮完整矩阵。"),
                     table([t("Group", "组别"), t("Composite rate", "综合通过率"), t("Still internal deadline", "仍达内部时限")], [[label, f"{s['passRate']:.1%}", s["internalDeadline"]] for label, s in zip(labels, payload["compositeStages"])])]
        render_cases = cases + payload.get("extended", {}).get("cases", [])
        render = next((r for r in reversed(render_cases) if r.get("finalRender")), None)
        if render is not None:
            text += [t("## Terminal scene render", "## 实验结束场景渲染"),
                     t(f"Actual MuJoCo scene immediately before host shutdown. Case: `{render['nodeid']}`; group: `{render['stage']}`; result: `{render['status']}`. The view is not synthesized or reconstructed from an initial scene.",
                       f"仿真停止前的实际 MuJoCo 场景。用例：`{render['nodeid']}`；组别：`{render['stage']}`；结果：`{render['status']}`。图片不是合成示意图，也不是重建的初始场景。"),
                     f"![{t('Final experimental scene', '实验结束场景')}]({render['finalRender']})"]
        text += [t("## Evidence and limits", "## 证据与限制"),
                 t("Pass rates include complete test assertions: action states, world support relations and Observer metrics. They are not manipulation completion alone or a multi-seed reliability estimate. Original tests use GT for scene setup, physical scoring and puzzle target association. The optional feedback adapter leaves the default webpage configuration unchanged.",
                   "通过率包含动作状态、世界支撑关系和 Observer 指标等完整断言，不是仅判断搬运动作完成，也不是多种子可靠性估计。原测试使用 GT 进行场景设置、物理评分和拼图目标关联。可选融合适配保持默认网页配置不变。"),
                 t("Cases run serially on a shared host. Timings include startup, rendering, processing, task execution and cleanup; early failures may look faster. Header-read counts include repeated reads, not independent visual corrections. Readable pytest.log copies normalize trailing whitespace and LF endings; pytest.log.gz preserves exact raw bytes, verified by rawLogSha256. [Structured results](summary.json) and [CSV](comparison.csv) retain all outcomes.",
                   "用例在共享宿主机串行运行。耗时包含启动、渲染、处理、任务与清理，提前失败可能显得更快。投影头读取次数包含重复读取，不等于独立视觉校正次数。可读 pytest.log 仅整理行尾空白和 LF 换行，pytest.log.gz 保存原始字节，可按 rawLogSha256 校验。[结构化结果](summary.json)和[CSV](comparison.csv)保留全部结果。")]
        (output/("report.zh.md" if chinese else "report.md")).write_text("\n\n".join(text)+"\n", encoding="utf-8")
