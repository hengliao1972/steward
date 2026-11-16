# Steward

[English](README.md) | 中文

`steward-bot` 的 `place` 在设置 `flip: true` 时，会先翻转持物，再释放到支撑面或地面。默认腕部转角上限允许半圈翻转，并与机器人模型声明的关节限位取交集；显式配置的更小上限仍然有效。地面旋转会从实测关节状态检查伸臂、转腕及下降路径；不可达时在执行前拒绝，不会忽略 `flip` 后仍返回放置成功。

机器人 Ego Harness 可选用 `rgbd_video_gen` 和 `NovelAnything` 两个 skill 调用远程新视角生成服务。`scripts/setup-dsh-profiles.sh`（Linux/macOS）和 `scripts/setup-dsh-profiles.ps1`（Windows）会在机器人 profile 中安装两者；Windows 上也可用 `scripts/setup-rgbd-video-gen.ps1`、`scripts/setup-novelanything.ps1` 加到已有 profile。使用方法见 [rgbd_video_gen](dsh/packages/dsh-bundle-rgbd-video-gen/README.zh.md) 和 [NovelAnything](dsh/packages/dsh-bundle-novelanything/README.zh.md)。

模型证据沿用原有 `head_camera`；连续录制按仿真时钟采样并保存机器人里程计和编码器，手爪目标由机器人自身运动学计算。快照工具使用现有头部／广角相机各自的标定，不改变两者的布局、视场或感知分工。此次集成不包含 SLARM 或新增三目相机。

可直接使用的机器人提示词、相机约定和包职责见[感知基模skill说明](docs/novel-view-generation/README.zh.md)。

Steward 是基于 DeepSeek Harness 的具身智能机器人系统。设计见仓库根目录的 [steward.md](../steward.md)，当前实现进度与运行命令见其中的「13. Implementation Progress」；Phase 1 感知如何从 RGB-D 得到几何与类别、以及泛化路线见 [13.7](../steward.md#137-phase-1-感知方法与泛化)。

## 快速开始

图案匹配在一个已记录的世界时刻复制机器人观测，再释放输入调度锁和世界时钟暂停，随后计算相关性。计算期间状态查询可以继续；关闭 HTTP 连接会在拼片或四分之一转之间取消计算，观测快照的获取仍会完成。结果保留快照时刻及当时的参考物姿态，物体移动后应重新观察。`/ego/inputs` 记录快照获取，包括之后被取消的请求，不是计算完成日志。

诊断运行中 Ego API 的拼图匹配：先摊开拼图片，让机器人观察拼图片和盒盖，停止其他模型和工具请求，然后在仓库根目录运行 `python3 steward/scripts/diagnose_match.py --api http://127.0.0.1:7420 --output /tmp/match-diagnosis.json`。脚本只用 Python 标准库，自动发现 `puzzle_piece` 和唯一的 `lid`，测量三次状态查询基线，再分别测试 15 秒匹配超时和 2 秒后主动断开。`--px 32` 提高匹配分辨率；`--reference ID` 和 `--entities ID ...` 可覆盖自动发现。脚本不移动机器人。JSON 保存响应、并发及断开后的状态查询耗时和输入应用计数；恢复失败、断开后无法确认计算已停止或匹配返回无有效候选时退出码为 2，并停止后续匹配。未观察到问题不代表已修复，HTTP 证据也不能证明 CPU 持续计算。API 地址必须是没有路径前缀的直接 HTTP 地址；每个恢复查询最多等待 `--recovery-timeout` 秒（默认 120），脚本总耗时可能超过该值。

```sh
uv sync
.venv/bin/steward demo          # 构建厨房并运行「把杯子放到桌上」，按 World State 判定成败
.venv/bin/steward up --scenario phase0-shuttle --no-dispatch-task   # 常驻运行世界与机器人，搬运交给下面的 shuttle
open http://127.0.0.1:7400/console   # 浏览器控制台：斜视主视图、相机、地图、指标、控制
.venv/bin/steward shuttle --api http://127.0.0.1:7420 --steps 0   # 在餐桌、柜子、餐边柜之间循环搬运
./scripts/check.sh              # Python 测试、DSH 插件编译与测试、组合配置快照比对
.venv/bin/steward build-kitchen # 机器人模型改动后（如头部加装的广角相机）重建默认场景
```

点击任一预览面板或用 Tab 聚焦：WASD 水平移动共用视角，Shift 加速，拖动原地转头，滚轮缩放。松开按键、焦点离开预览或按 Escape 停止移动，预设视角可回到场景中心。这些操作不移动机器人或其传感器相机。

控制台的“当前思考”为每个分块标明模型步骤，用分隔线区分正文和后续工具调用，并突出最近一步。突出显示表示最近记录的步骤，不代表模型仍在生成。每段不超过2000字符时原样显示，超出时保留末尾2000字符、标明已截断，并提供展开／收起入口。全文通过 `GET /console/thinking?robot=<id>&entry=<id>` 按需读取；同一记录的展开状态在轮询刷新时保留，离开可见记录窗口后清除。读取失败保留预览并提供重试；场景重置后，旧会话也不能通过全文接口读取。

机器人头部有两台 RGB-D 相机：`head_camera` 看前方，`wide_camera` 是短焦广角相机，看身前台面与夹爪；两台都进入感知，见 [steward.md 13.8](../steward.md#138-机器人本体的双相机)。

在 `phase1-sam` 中，Cobot 任一夹爪的开合、持物或持把手状态变化都可以触发近景相机关键帧。两次分割帧之间，机器人自身状态更新保留双夹爪反馈，持物随夹持它的机械臂移动。

`cobot-magic` 左右臂 joint1～joint6 初始角度均为 0 弧度，夹爪初始张开。`cobot-magic` 夹爪使用简化的掌体、导轨、滑块与手指几何。手指沿工具轴延伸，对称张开时内侧开口最大为 100 mm；抓取点位于两侧指垫之间、指尖后方 5 mm。这些显示几何不代表厂家 CAD，也不提供接触式抓取。

`cobot-magic` 双臂和升降机构的几何采用厂家 `robot.usd` 数据；[本体配置说明](embodiments/cobot-magic.yaml)记录坐标映射、单位假设和保留的仿真近似。运行时资产由 `src/steward/assets/cobot.py` 生成，说明性 YAML 不直接驱动资产。已有场景及原始升降指令的调整见[迁移说明](../docs/upgrade-guide/v0.2.2-alpha.2/cobot-lift-coordinates/guide.zh.md)。

AgileX 在停靠成功前保持机械臂姿态。放置规划检查目标腕姿及从实测关节状态出发的 IK 连续性，但不代替完整碰撞或动力学规划。操作成功且双手空闲时，双臂各六个关节归零后任务才完成，升降轴和夹爪开度保持不变。任一手持物或抓着把手时不归位；失败和停止也不启动归位。归位失败与已完成的动作分别报告。

传入 `--misnames-config ''` 会禁用名称映射。缺少映射时切换到 `phase0-misnamed` 会被拒绝，当前世界和后续 Episode 保留原感知级别。

## 与 DeepSeek Harness 一起运行

```sh
echo 'DEEPSEEK_API_KEY=sk-...' > .env     # 已被 Git 忽略，运行脚本自动加载
.venv/bin/steward up                      # WCA :7400，机器人端口 :7410，Ego API :7420；场景任务自动交给 Ego Harness
.venv/bin/steward up --phase phase1-sam   # 机器人用 SAM 3.1 自行分割 RGB-D（设置在 config/segmentation.toml）
./scripts/setup-dsh-profiles.sh           # 构建插件，创建 ego、world、world-task profile
./scripts/run-ego.sh "把杯子放到桌上"       # 机器人智能体
./scripts/run-world.sh "按描述构建一间厨房……"  # World Harness 一次性任务
./scripts/run-world.sh --port 3091        # World Harness 浏览器界面
./scripts/install-desktop-app.sh          # 在桌面安装 Steward.app：选场景后启动世界、Ego Harness 并打开控制台
.venv/bin/steward services                # 检查远程感知服务（SAM 3.1、NovelAnything、NVS、SLARM）是否可达
```

## Windows

Windows 上使用同样的命令：`uv sync` 把 `steward` 可执行文件装在 `.venv\Scripts` 下，每个 `.sh` 脚本都有同名、同参数的 PowerShell 版本。如果 PowerShell 拒绝运行未签名脚本，先执行一次 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。

```powershell
uv sync
.venv\Scripts\steward up --phase phase1-sam
.\scripts\setup-dsh-profiles.ps1          # 构建插件，创建 ego、world、world-task profile
.\scripts\run-ego.ps1 "把杯子放到桌上"      # 机器人智能体
.\scripts\run-world.ps1 --port 3091       # World Harness 浏览器界面
.\scripts\check.ps1                       # Python 测试、DSH 插件编译与测试、组合配置快照比对
.\scripts\install-desktop-app.ps1         # 在桌面创建 Steward 快捷方式：选场景后启动世界、Ego Harness 并打开控制台
```

桌面启动器的日志写在 `%LOCALAPPDATA%\Steward\Logs`；macOS 上写在 `~/Library/Logs/Steward`。

`ego` profile 是机器人的 Ego Harness，只能经 Ego API 感知与行动，shell、文件、网页等通用工具已禁用；`world` 与 `world-task` profile 是 World Harness，经世界控制接口构建资产、组合场景、控制 Episode。机器人与世界使用不同的 `DSH_HOME`。

## 目录

| 路径 | 内容 |
|---|---|
| `src/steward/scene` | WS 与 ES 共用的 SceneState、校验器、MJCF 转换 |
| `src/steward/assets`、`authoring` | 资产生成、校验、预览、资产库；场景与 Scenario |
| `src/steward/world` | WSim 与 Steward 主机（WCA、机器人端口） |
| `src/steward/ego` | Ego Runtime：估计器、ESim、技能、Ego API |
| `src/steward/observer` | ES 与 World State 的对比评估 |
| `dsh/` | DeepSeek Harness 插件与 bundle（锁定 `@deepseek-ai/dsh@0.2.1-rc.3`） |
| `scripts/` | profile 创建、检查、多种子回归 |
| `config/perception-services.toml` | 远程感知服务清单（地址、健康检查、接口），见 steward.md 13.7.6 |
| `config/segmentation.toml` | `phase1-sam` 的机器人侧分割设置：词表、阈值、裁剪与语义融合，见 steward.md 13.7.7 |
| `compat/snapshots` | 组合配置快照 |

运行数据写入 `var/`，不纳入版本管理。
