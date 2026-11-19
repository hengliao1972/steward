# 分割辅助连续定位

[English](README.md) | 中文

## 范围

这项实验性 CPU 实现通过里程计预测平面位姿，并与同步语义掩码选出的深度表面配准以校正位姿。诊断模式使用现有感知标签。融合模式可对当前帧独立请求 SAM。

## 运行

在内层 `steward` 目录执行 `python -m pip install -e ".[localization]"` 安装可选依赖。SciPy 提供距离变换和鲁棒最小二乘求解，不需要 ROS、RTAB-Map 或原生编译。`uv.lock` 也已记录该可选依赖。

如果还没有生成拼图场景，先在内层 `steward` 目录运行 `python -m steward.cli build-kitchen`。然后运行 `python -m steward.cli up --scenario puzzle-dog --phase phase1-sam --localization-experiment --no-dispatch-task`，打开 `http://127.0.0.1:7400`。该命令使用配置的 SAM 服务和共享 Ego Runtime。打开**定位建图**，可以对照估计轨迹、仿真真实轨迹、墙面和零距离地图格点。

普通运行保留瞬时里程计噪声，累计漂移和融合定位默认关闭。实验开关会同时启用 `--odom-drift`（1% 平移尺度偏差和每米 0.005 rad 的航向漂移）以及 `--apply-localization`（用融合位姿更新 Ego-State 和共享运行时控制）。两项也可以在 `up` 单独启用。分开运行时，给 `world` 传 `--odom-drift`，给 `ego` 传 `--apply-localization`。融合模式自动使用默认 TOML。单独传 `--localization-config config/localization.toml` 可采集诊断数据，与融合开关合用则可以指定自定义 TOML。

对支持的进程内动作测试，在常用 pytest 命令后添加 `--localization-experiment`。pytest 也接受单独的 `--odom-drift` 或 `--apply-localization`，并允许用 `--localization-config` 指定融合配置。融合测试默认给仿真实例掩膜添加仿真语义标签。场景没有仿真掩膜时，使用 `--localization-mask-source=sam` 并配置分割服务。绕房、大角度转动、长时静止和狗狗拼图可用下文的实验脚本或动作测试矩阵测量。

从该实例的 Ego API 读取 `GET /ego/localization`。未启用时返回 `enabled: false`；启用后包含 `rawPose`、校正／预测 `pose`、`mapFromOdometry`、时间戳、Episode、状态、拒绝原因、地图版本、距上次校正的时间及指标。运行时还会在观测目录追加 `localization.jsonl`。控制台和 Observer 评估 Ego-State；开启融合后，本体与新融合的观测使用校正后的地图坐标，诊断模式保持原样。

## 选点与配准

随附配置使用 `wide_camera`。它用同一采集时刻的语义标签选取墙、柜体、操作台、餐边柜和桌子。最高分标签达到阈值才提供定位点。缺失或匿名掩码不提供定位点。

掩码腐蚀去掉边界混合像素。通过标定反投影与本体运动学去除机器人自身，并选择近似垂直表面。空间采样和各区域的点数上限防止较大的区域占满两路相机共享的点数预算。

第一组足够的观测初始化局部表面地图。后续观测先与历史地图配准，再插入。三个高度层分别维护间距为 5 cm 的二维无符号距离场，通过双线性插值为变换后的点打分。优化只调整平面平移和航向。里程计先验由 EKF 处理，优化期间参配点集合固定。

接受条件检查重叠率、内点比例、表面距离均方根、优化状态、校正幅度及不含里程计先验的几何信息。退化比例计算时角度信息列按点的力臂缩放，视觉协方差则使用未缩放雅可比，单位为米／弧度。信息特征值比例太低时拒绝整个校正，尚未实现只校正可观测方向。之后由 EKF 检查是否接受视觉候选。拒绝帧不能更新地图或校正变换；接受帧也只有在平移或转角足够后才用融合位姿插图。

两次观测之间，将原始里程计与最近的地图到里程计校正变换组合。`initializing` 表示首帧建图，`corrected` 表示接受配准，`predicted` 表示仅里程计更新或拒绝配准。`accepted` 只标记本次成功匹配。`stale` 表示尚无成功校正或其年龄超过 `stale_after_s`，不是经过标定的协方差或导航安全判断。跟踪器拒绝重复／非递增时间戳，运行时对已处理帧直接跳过；Episode 变化重置地图与校正。

## EKF 融合

`[ekf]` 表在这个显式启用的实验中默认开启三状态 `(x, y, yaw)` 滤波器。根据底盘相对里程计增量及 SE(2) 雅可比传播位姿。过程方差随时间和测得的运动增长；运动量按配置的较粗时间间隔及死区估计，避免把高频绝对位姿抖动算成数米独立运动。协方差保留里程计采样噪声下限。这些是可配置的近似噪声模型，不是经过标定的置信保证。

视觉协方差根据内点几何雅可比与残差尺度估计，包含栅格化下限、独立的位置／航向及地图误差下限、有效点数上限和协方差膨胀。滤波器对航向创新量做角度环绕，拒绝过大的视觉不确定度与归一化创新平方（NIS），并要求两次接受观测至少间隔 `min_visual_interval_s`。默认 NIS 门槛为 11.345。接受后使用卡尔曼增益与 Joseph 形式更新后验协方差；非法协方差明确报错，拒绝观测不会改变预测位姿／协方差或地图。

输出包含 `filterCovariance`；候选指标包含 `candidatePose`、`visualCovariance`、`innovation`、`innovationNis`、`predictedCovariance`，融合后另有 `kalmanGain` 和 `fusedDelta`。`delta` 仍表示优化器提出的改变量，`fusedDelta` 才是实际应用的滤波改变量。成功更新返回 `ekf_fused`，滤波拒绝原因包括 `innovation_outlier`、`visual_uncertainty`、`correlated_observation`。拒绝视觉观测后仍继续预测并传播协方差，只有接受更新才重置校正年龄。

设置 `[ekf].enabled = false` 可复现此前直接视觉校正模式及其弱优化先验（`prior_position_sigma_m`、`prior_yaw_sigma_rad`）。启用模式从视觉目标函数移除这些先验，避免重复应用同一个运动先验。自建地图、重复视角和基于里程计的初值仍导致观测相关；协方差下限与观测间隔只是保守近似，不是严格的地图／轨迹联合滤波。实现不估计速度或标定偏置，不撤销历史地图错误，也不保证全局建图一致性；应用到 Ego-State 需要额外开启融合参数。

## 配置与限制

定位 TOML 配置相机、采样、地图分辨率和接受门槛。融合模式每模拟秒从 `wide_camera` 采样一次，使用与采集时间一致的标签。帧缺少标签时请求 SAM，两次观测之间继续里程计预测。SAM 失败会出现在定位状态和操作告警中。 [配置](../../config/localization.toml)

分层地图的初始水平范围为 5 m × 5 m。有效插图时，各层一起扩格以容纳观测到的表面和距离查询余量。已有格点的地图坐标保持固定。默认需要三次有效插图确认某处可见为空，才清除历史占用。深度射线须穿过该格点记录的高度附近，再到达更远的表面。同格的新命中会重置空闲计数。`mapClearedCells` 显示累计清除数。任一轴扩建后若超过 1,025 个格点，定位器会报告 `map_grid_limit` 并保留原网格。

可在定位 TOML 中设置 `signed_walls = true` 试验带符号距离场。每个格点的距离大小取决于最近的历史占用格点。最近点来自墙面时，朝向观测相机的法线定义可见侧，另一侧为负。非墙面或朝向不明确时保留无符号距离。配准使用带符号残差，重叠和内点检查使用残差绝对值。

## 融合坐标

在控制台打开 **定位建图**，可以对照仿真墙面、零距离格点和两条轨迹。高度选择器可单看一层，或让各层的半透明颜色叠加显示。切换 Episode 会重置清除计数。

融合发布不可变的地图到里程计变换快照。感知头和实时本体读数在更新 Ego-State 或规划技能前复制到地图坐标，定位原始输入保持原样。已投影的数据不重复校正，Episode 切换重置快照，航向保留未环绕的累计转数。地图锚定初始化时刻：已有静态轨迹保持固定，新观测进入相同坐标系；不会回溯优化历史地图畸变。

仿真器的底盘位置伺服目标使用启动时标定的执行器参考系。命令沿用该约定，不将目标重新解释为漂移的传感器坐标。其他实体机器人需要匹配自身执行器参考系的适配。SAM 在 Ego-State 锁外运行，但其延迟仍会推迟感知；仅预测区间和原任务截止时间仍影响结果。

## 运动任务矩阵

实验脚本 `scripts/localization_experiment.py` 接收一个新输出目录，以 1 Hz 和仿真语义掩码运行绕房与交替大角度旋转。脚本保存所用 TOML 和原始、融合位姿误差。传入 `--config` 可在相同路线对比带符号和无符号距离场。

传入 `--cases stationary` 可测试五个模拟分钟内不发送移动指令的情况。两组共享同一批仿真 RGB-D 帧，一组保留原本的噪声，另一组给观测的 x 和航向叠加线性偏置。里程计采样为 10 Hz，视觉处理为 1 Hz。实验用仿真真值检查实际移动并保存两组轨迹，通过比较首尾 30 秒的平均位姿误差区分长期偏移和逐帧抖动。

快速仿真掩膜迭代可从仓库根目录执行 `steward/.venv/Scripts/python.exe steward/scripts/motion_control_matrix.py --source-root . --output steward/var/motion-matrix/my-sim-mask-run --manifest steward/config/motion-control-sim-masks.json --mask-source simulation --stages baseline drift drift-fused clean-fused`。该清单保留六个参数化用例：剔除原基线的三项失败，以及本轮基线新增失败的猫形拼图。剔除只依据基线，不依据漂移或融合组结果；筛选后基线的 100% 因此不是独立可靠性估计。旧的十用例 SAM 清单和已有结果继续保留。

若已收集的运行在基线结束后进一步筛选，`cohort-selection.json` 会固定保留的用例标识和有效清单。汇总与延时重测工具将它一致应用于所有组，在 `raw-summary.json` 保留全部原始结果，并从比较分母中排除剔除项；原收集清单保存在 `raw-manifest.json`，后续运行则直接使用精简的六用例清单。

仿真适配仅存在于 pytest 实验进程中，将原有随机编号的实例掩膜关联到仿真类别标签（置信度 1.0），保留掩膜退化、深度和里程计噪声。只有定位器接收这些标签，主对象感知保持原样。适配不提供机器人 GT 位姿、实体标识或 GT 地图，任何 SAM 调用都会触发断言。这是理想语义下的迭代测量，而不是 SAM 精度测试。无累计漂移＋融合的狗形拼图用例会在关闭前将真实 MuJoCo 结束场景保存为 `final-world.png`，并记录相机和时间；测试失败也保存失败现场，报告会在图旁标明用例及结果。

使用仿真掩膜的子进程会移除 `STEWARD_SEGMENTER_URL`，抓取前的旧版近距离重新观察也会跳过分割。添加 `--skill-diagnostics` 后，子进程将完整的 predict/run/wait 响应写入 `skill-events.jsonl`，将任务历史和环境变量存在性分别写入 `jobs.json` 与 `environment.json`。父进程环境保持原样。使用 `--capture-final` 可保存每个所选仿真用例的结束场景，包括失败用例。单例重测可用只含该参数化用例的 `selectors` 清单，并指定所需阶段。诊断增加文件写入，常规耗时对比时关闭。

四组使用同一运行目录时，将该完整目录同时传给 `scripts/summarize_motion_control_matrix.py` 的 `--pre` 和 `--post`。使用 `--extended <重测目录>` 加入独立记录的延时重测，其分割来源继承原清单。

`scripts/motion_control_matrix.py` 固定 `config/motion-control-cases.json` 中的选择器、记录测试清单，并在独立进程运行每个参数化用例。从仓库根目录执行 `steward/.venv/Scripts/python.exe steward/scripts/motion_control_matrix.py --source-root . --output steward/var/motion-matrix/my-run --stages baseline drift drift-fused clean-fused`，使用全新输出目录和可用 SAM 服务。四组使用相同主线程渲染、原断言和内部截止时间，每例外部上限 600 秒。累计漂移为距离比例 1% 和每米 0.005 弧度，瞬时噪声保持不变；跳过与超时仍计入所选用例分母。[固定协议](results/motion-control-matrix-20261117/protocol.json) 记录实验范围。

`scripts/summarize_motion_control_matrix.py --pre <基线及漂移输出> --post <融合输出> --output <报告目录>` 归档完整矩阵并生成双语报告。`scripts/retest_motion_timeouts.py` 接受相同路径，只为已确认的超时用例创建全新重测目录。默认重测将内部等待延长为原来的四倍，外部上限改为 2400 秒。独立 pytest 扩展修改测试截止时间与任务等待，验证成功断言的 AST 完全相同，并记录每项覆盖值。睡眠、控制阈值、定位设置及噪声均不变。重测结果补充原始通过率，不覆盖它们。

## 验证与首次测量

执行 `python -m pytest tests/test_pose_ekf.py tests/test_distance_localization.py tests/test_segmenter.py -q`，验证加权融合、环绕创新量、大跳变拒绝、协方差稳定性、仅用融合位姿插图、确定性的位姿校正、预测连续性、退化拒绝、时间戳／Episode 隔离、语义选点、复用同帧且不额外调用模型、HTTP 输出和现有分割回归。测试使用独立临时目录和动态分配的 HTTP 端口，不设置依赖机器性能的速度断言。

真实 SAM 冒烟可用 `python scripts/diagnose_segmented_localization.py --output var/localization/my-smoke --frames 8`，输出目录必须尚不存在。脚本建立私有厨房，通过执行器转动机器人，取得无掩码 RGB-D，再请求已配置 SAM 服务，保存带标签 RGB-D 输入和逐帧输出。GT 只在定位之后读取用于评分。这个独立诊断自行请求 SAM。错误向上传递，私有仿真会关闭。

首次启用 EKF 前的种子 0 冒烟共八帧：一帧初始化地图，七帧接受校正。选点加定位平均 85.94 ms、P95 103.74 ms；采集总计 0.284 秒，SAM 总计 37.752 秒。原始位置 RMSE 为 0.481 cm，连续输出 RMSE 为 1.874 cm。这次以旋转为主的短程运行使位置精度变差，仅证明真实掩码链路执行成功，不证明漂移改善、整圈覆盖率或端到端实时性。它与此前使用头部相机的 RTAB-Map 测试不可直接比较。可用上一段命令生成回放输入。

使用 `python scripts/diagnose_segmented_localization.py --replay var/localization/my-smoke --output var/localization/my-replay --frames 8`，可不经渲染或 SAM 回放这批固定预测结果。保存的掩码和标签与各帧传感器时间戳一致。回放只报告选点和定位耗时，没有 GT 派生输入或新的分割推理。

原有绕房测试也增加了 `segmented` 组，每秒模拟时间请求一次真实 SAM，每次身体采样继续推进位姿。除非设置 `STEWARD_SAM_LOCALIZATION_E2E=1`，该组默认跳过；启用后 `python -m pytest tests/test_localization_loop.py -k segmented -q -s` 执行三个种子。报告保存原始及连续预测／校正误差、观测接受比例、最长无可信校正间隔、原始轨迹哈希和采集／SAM／选点／定位耗时，其中用例计时不含仿真关闭。[完整串行实验](results/segmented-room-loop-20261117/summary.json)已完成全部九组：三个种子的位置 RMSE 均降低，但校正间隔和 SAM 延迟仍较显著。常规集成测试另用替代分割器，在语义结果为空时完成整条物理路线，验证的是连续预测，不是 SAM 精度。

大角度反转样例为 `python -m pytest tests/test_localization_rotation.py -q -s`。以 0.6 rad/s 执行 0° → +200° → −200° → +200° → −200° → 0°，每个目标保持一秒，每秒模拟时间进行一次语义观测。中间指令保留 400° 长路径反转，不选择角度环绕后的最短转向。GT 检查每段有符号转角、累计约 1600°、反复跨越 ±π，以及底盘真实位移小于 3 cm；定位仍不接收 GT。

未设置启用环境变量时，该样例使用空语义结果验证预测；`STEWARD_SAM_LOCALIZATION_E2E=1` 另对种子 0、1、2 运行真实 SAM。报告包含全程位置与环绕后的航向误差、观测接受比例、最长校正间隔及采集／SAM／选点／定位耗时。真实 SAM 用例在各测试目录的 `inputs` 保存同步的带标签 RGB-D，可通过诊断脚本的 `--replay` 回放。用例计时从仿真构建后开始，至仿真关闭后结束，包含观测日志和输入快照写入，不含最终报告和轨迹 CSV 写入。噪声保持不变，现有累计航向偏差依赖平移距离，因此主要检验视角变化、角度环绕与配准稳定性，不额外注入按旋转量增长的漂移；测试不强制断言精度改善。

[启用 EKF 前的反复转动实测](results/segmented-rotation-20261117/summary.json)已完成全部四项。三个真实 SAM 种子的位置与航向 RMSE 均增大，其中两次在真实位移仅毫米量级时接受了较大的错误平移校正。较小的终点误差掩盖了明显的中途误差。这些结果暴露了直接视觉校正的不足，绕房收益不代表旋转稳定性；该轮未修改定位参数。

`scripts/replay_localization_case.py` 按原始身体采样频率回放旋转记录，包含语义帧之间的预测。`--source` 指向含 `trajectory.csv` 和 `inputs/frame-*.npz` 的目录，`--output` 指向尚不存在的新目录。默认配置评估 EKF；`--legacy` 禁用 EKF，并逐帧核对位姿与此前记录一致。GT 仅在每次估计输出后读取用于评分。这不同于短程诊断中的仅观测帧回放，不会重新请求 SAM。

[固定输入的 EKF 旋转对比](results/segmented-ekf-20261117/rotation-comparison.csv)保留原始身体采样频率，三个种子的旧模式均逐帧复现原记录。EKF 使位置误差保持接近原始里程计，没有引入此前较大的偏移。[重新请求 SAM 的 EKF 绕房对比](results/segmented-ekf-20261117/room-loop-comparison.csv)与旧轮的原始轨迹相同，但分割重新推理；两个种子改善，一个基本相当。协方差是近似模型，校正间隔仍存在，这些结果都不证明全局或实时可靠性。
