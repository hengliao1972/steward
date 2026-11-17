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

`cobot-magic` 左右臂 joint1～joint6 初始角度均为 0 弧度，夹爪初始张开。`cobot-magic` 夹爪使用简化的掌体、导轨、滑块与手指几何。手指沿工具轴延伸，对称张开时内侧开口最大为 100 mm；抓取点位于两侧指垫之间、指尖后方 5 mm。这些显示几何不代表厂家 CAD，也不提供接触式抓取。

`cobot-magic` 双臂和升降机构的几何采用厂家 `robot.usd` 数据；[本体配置说明](embodiments/cobot-magic.yaml)记录坐标映射、单位假设和保留的仿真近似。运行时资产由 `src/steward/assets/cobot.py` 生成，说明性 YAML 不直接驱动资产。已有场景及原始升降指令的调整见[迁移说明](../docs/upgrade-guide/v0.2.2-alpha.2/cobot-lift-coordinates/guide.zh.md)。

AgileX 在停靠成功前保持机械臂姿态。放置规划检查目标腕姿及从实测关节状态出发的 IK 连续性，但不代替完整碰撞或动力学规划。操作成功且双手空闲时，双臂各六个关节归零后任务才完成，升降轴和夹爪开度保持不变。任一手持物或抓着把手时不归位；失败和停止也不启动归位。归位失败与已完成的动作分别报告。

传入 `--misnames-config ''` 会禁用名称映射。缺少映射时切换到 `phase0-misnamed` 会被拒绝，当前世界和后续 Episode 保留原感知级别。

## 导入 USD 机器人

随仓库提供的 [AgileX 资产](assets/agilex/README.zh.md) 无需转换即可运行；下面的步骤用于重新转换并检查源模型。可选的 `usd` extra 使用 MuJoCo 4.0 的原生 OpenUSD 解码器。在独立的 Python 3.12 环境中安装它，以保留常规仿真环境。从仓库根目录运行：

```sh
uv venv --python 3.12 steward/var/usd-env
uv pip install --python steward/var/usd-env/bin/python -e './steward[usd]'
steward/var/usd-env/bin/steward import-usd steward/assets/agilex/robot.usd --name agilex-usd \
  --meters-per-unit 1 --orient-joints --fixed-base \
  --bound-mass 0.001 --bound-inertia 0.000001 --inertia-from-bounds --drive-frequency 5
steward/.venv/bin/steward world --scenario agilex-usd --phase phase0 \
  --wca-port 7450 --robot-port 7460 --ego-api '' --world-harness '' --robot-harness ''
```

打开 `http://127.0.0.1:7450/console`。导入会创建带地面的检查场景，并将 `template.json`、二进制网格和 `report.json` 写入 `var/steward/imports/<name>/`。已有名称会被拒绝。网格引用使用绝对路径，因此应保留该目录的位置。查看结果也支持 MuJoCo 3.17，不再读取源 USD。源文件始终不会被修改。

示例选项用于提供的 AgileX 资产：文件将米制几何数值标记为厘米，一个底盘关节方向相反，且缺少惯量。质量和惯量下限、均匀盒体惯量估算以及 5 Hz 临界阻尼驱动均为显式仿真近似，会记入报告。正确编写的资产可省略这些选项；依赖动力学结果前应检查报告。驱动在启动时保持导入的静止姿态，不执行 USD 中的驱动目标。

导入器支持一棵刚体机器人树，包括固定、旋转和平移关节、网格与基本几何体、力式位置驱动以及标量 PhysX mimic 联动约束。转换保留关节限位、枢轴和显式惯量。网格碰撞使用 MuJoCo 凸包。着色网络、贴图和相机不会转移；缺失的 MDL 着色器会记入报告，缺失的几何依赖则会中止导入。闭环和不支持的动力学会被拒绝。通用导入不会自动分配任务控制器；可通过机器人端口的 `/command` 接口按名称驱动关节。提供的双 Piper 模型另有已注册的适配器 `agilex-usd`，使用方式如下。


在任务场景中启动随仓库提供的 `agilex-usd` 本体：

```sh
steward/.venv/bin/steward up --scenario cobot-table --body agilex-usd --phase phase0 \
  --speed 1 --no-dispatch-task
```

打开 `http://127.0.0.1:7400/console`，在本体选择器中选择 **AgileX USD 双 Piper**。也可用它替换厨房场景中的机器人。`up` 会启动其 Ego Runtime；导航、抓取和放置使用与其他本体相同的预测回执和技能 API。原始导入检查场景仍独立保留，不注册为可执行任务的本体。

[USD 独立配置](config/agilex-usd.toml) 管理导入机器人的伺服增益和原始 `lift_home`、平面 `[base]` 驱动、`[cameras.head]`/`[cameras.wide]` 内参与安装位置、`[arm]` 展开与 IK 参数、`[motion]` 执行参数、`[trajectory]` 接近与恢复路点及重试上限，以及 `[planning]` 回执参数。可用 `STEWARD_AGILEX_USD_CONFIG` 指定本地 TOML，逐项覆盖；省略的字段使用仓库附带的 USD 默认值。模板路径相对声明它的文件解析。已有扁平配置仍受支持；提供 `arm_effort` 而省略 `finger_effort` 时，夹爪沿用手臂的力限。未知字段、类型错误、非有限数值、无效范围以及超过驱动限速的底盘速度会在启动前被拒绝；模板缺失或不兼容也会明确报错。

`lift_home` 和原始 `/command` 关节目标使用 USD 坐标（约 `[-0.6, 0]` 米）；共享 Ego 机械臂的虚拟 `lift` 表示 TCP 离地高度。默认原始初始位置仍为 `-0.45` 米。两臂启动和归位时的关节角度均为零；`arm.deploy_joints` 是中间工作姿态和 IK 初始解，不是启动姿态。关节限位、TCP 偏移和安装变换来自导入模型。依赖工作空间的工作高度与退出高度由该模型的 IK 计算，不从简化机器人复制。

Piper 的 TCP 反馈来自实测编码器，包含横向偏移和伺服滞后，不以目标位置替代测量。向支撑面放置时，仅在 `motion.release_settle_s` 内确认实测高度和伸距进入 `motion.release_joint_tol`（Piper 配置为 8 毫米）后才打开夹爪。反馈缺失或等待到期会使任务失败并保持夹持。释放成功本身不证明物体最终由目标表面支撑，任务测试还会核对最终世界状态。

普通表面放置后，机械臂按运动策略撤回，底盘退离后才降低到收拢高度，以保持撤离过程中的桌面间隙。两种 Piper 本体使用 `PiperMotionPolicy`：接近把手时，若下降伸距会将手指置于柜体上方，底盘先退开并留出 `trajectory.handle_lower_clearance` 再下降。释放把手后，机械臂先升到由自身模型计算出的工作高度再缩臂，随后底盘退离，机械臂降低到收拢高度，以避开简化 Cobot 在收拢高度的短伸距 IK 不可达区。滑台机器人使用 `PlanarMotionPolicy`，不执行关节臂的下降前退让或绕行工作高度。

对于柜体顶板下方的抽屉内容物，AgileX 通过 `planning.drawer_entry_pitch`、`drawer_grasp_raise`、`drawer_grasp_backoff` 和 `drawer_entry_backoff` 配置倾斜的前侧抓取点及下降间隙。规划使用本体的工作高度，并从抽屉开口正面接近。闭合高度必须同时满足 IK 工作空间和物体顶部的抓取范围。机械臂先稳定倾斜姿态，从抽屉露出部分的中部下降，再斜向接近抓取点；下降间隙以柜体前缘为基准。抓住后先退出顶板覆盖区再抬升。已露出顶板的抽屉内容物沿用普通接近路径。

Phase 0 从盒沿抓取可倾倒容器；空心中心不能与盒内拼图竞争抓取候选。倒料规划根据容器相对工具的姿态，检查其偏离世界竖直方向的倾角是否达到 `planning.pour_roll_min`，不能仅检查腕部滚转角。Piper 默认请求腕部滚转 2.3 弧度：以 1.05 弧度工具俯仰抓住竖直容器时，容器倾角约为 1.83 弧度；腕部滚转 1.9 弧度时则只有约 1.57 弧度。每条候选轨迹仍须通过连续可达性检查。回执提供 `containerTilt`，物理任务测试仍要求内容物实际离开盒子并落到桌面。

适配器保留 USD 网格、惯量和手指联动，添加抓取点，并将移动底盘映射为轮子关节固定的平面位置控制。两个仿真 RGB-D 视角安装在 `top_camera_link` 上并随升降运动，其标定不来自 USD。编辑此配置不会改变 `cobot-magic` 或其旧控制器。两种本体都使用共享 Ego Runtime；Phase 0 使用理想附着，不依赖夹爪接触抓取。该配置支持仿真任务，不提供真机控制或经过验证的轮式动力学。其增益和任务容差是仿真参数，不保证抓放成功。

运行装配位于 [`ego/agilex.py`](src/steward/ego/agilex.py)；`prepare_body` 从同一份配置快照生成模型、控制器和策略。两种 Piper 本体共享 [`piper_arm.py`](src/steward/ego/piper_arm.py)、[`piper_ik.py`](src/steward/ego/piper_ik.py)、ESim 和 SkillExecutor。可注入的 [`MotionPolicy`](src/steward/ego/motion_policy.py) 负责停靠后的机械臂准备、接近、把手接近、搬运抬升前的撤离、放置和释放把手后的退离、撤臂及失败恢复；执行器负责技能顺序、计时、取消和任务结果。接近把手时的下降失败会先记入当前任务日志，再尝试恢复。IK 自碰撞连杆分组可通过 [`SelfCollisionPolicy`](src/steward/ego/collision_policy.py) 注入；USD 装配从自身模型选择障碍连杆。原有 `cobot_arm` 与 `ik.CobotIK` 导入路径仍可用。机器人端口通过 `GET /identity` 提供本体身份，CLI 在启动 Ego 前核对。旧 `CobotEgoRuntime`、`CobotSkills` 和 Cobot 脚本任务只接受 `cobot-magic`；导入模型应使用 `steward up --body agilex-usd`。

本地资产集成测试命令为 `STEWARD_TEST_AGILEX_USD=1 steward/.venv/bin/python -m pytest -q steward/tests/test_agilex.py`。它通过 Ego API 导航、抓起碗并放到桌面另一处，最后核对世界坐标。运动学测试夹具包含源关节数据，以小盒体替代网格；集成测试使用随仓库提供的网格，因完整物理任务耗时较长，需显式启用。

同一资产开关还会启用 [`test_agilex_physics.py`](tests/test_agilex_physics.py)，验证原生肩高、原始升降行程、相机外参、手指联动、多个姿态下的网格遮罩覆盖，以及从记录的释放姿态撤离后物体仍由桌面支撑、从记录的停靠姿态下降时避开柜体并实际拉开抽屉，以及存在小幅感知误差时仍抓住正确的抽屉物体。[`test_cobot_ego.py`](tests/test_cobot_ego.py) 经各本体的正式运行装配执行共享任务断言；未启用开关时会跳过真实 USD 参数。[`test_body_retreat.py`](tests/test_body_retreat.py) 为三种本体验证表面支撑、抽屉打开与关闭、最终收拢姿态和观察位置，以及释放保护。轻量 IK 对照使用相同物理肩高，原生初态检查则保留 USD 配置自身的初始升降位置。

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
