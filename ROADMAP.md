# yunqian_vision 短期路线图（Sim2Real）

> 状态：P1 同机闭环已落地（共享内存，不走 ROS2）。最后更新：2026-11-15。
> 部分关键决策已确认（见 §5），其余待确认；确认后再动代码。
> 边缘算力平台不是一种。本仓库适配 **NUC**；**Orin NX** 适配在 `27-wheel-leg-wbc` 的 `origin/vision`。三号步兵（计划内第一台）用 Orin NX。详见 `AGENTS.md` §2.3。

## 1. 总体目标

按优先级依次完成：

1. **先把 `yunqian_vision` 在 `bevy_robomaster_simulator` 里跑通闭环**。
2. **让仿真具备“重建现实机器人”的能力**，形成实机与仿真的并排对比，服务 sim2real。

## 2. 目标形态

```
┌──────────────────────────────┐        网络         ┌──────────────────────────────────┐
│ 实机边缘算力平台              │  ──图像/数据──>    │ 个人电脑（本地）                  │
│ - 运行 yunqian_vision        │                    │ ┌──────────────┬───────────────┐ │
│ - 相机、云台、IMU、下位机     │  <──控制/请求──    │ │ 左：实机处理图 │ 右：数据重建   │ │
│ - 处理多张调试图              │                    │ │（可切换）     │ 的仿真机器人   │ │
└──────────────────────────────┘                    │ └──────────────┴───────────────┘ │
                                                     │ PC 运行同一套视觉代码，对比 sim/real │
                                                     └──────────────────────────────────┘
```

- 用户通过 SSH 登录边缘算力平台调试实机。边缘平台按机器人区分，不要默认都是本仓库这套 NUC 构建：
  - **NUC**（x86_64，OpenVINO）：本仓库直接对应的平台。
  - **Orin NX**（aarch64，TensorRT/CUDA，OpenVINO 仅回退）：适配在 `/home/scj/桌面/rm2027/27-wheel-leg-wbc` 的 `origin/vision`，部署记录见该分支的 `JETSON_DEPLOY.md`。
  - **三号步兵是计划内第一台调试机器人，使用 Orin NX。** 给它开分支或做实机联调时，推理/部署以 Orin NX 适配为准，不要拿本仓库的 NUC 构建直接上板。
- **PC 上运行同一套 `yunqian_vision` 视觉代码**（已确认的关键决策）。PC 侧对比用的是业务逻辑同一套；实机推理后端仍按该机器人的算力平台选择。
- 边缘平台把**选中的调试图 + 必要状态**转发回个人电脑；PC 上通过菜单选择“传哪张图”。
- 个人电脑看到左右两个画面：
  - 左：实机相机**处理后的画面**，可从 raw / binary / detector / reprojection 等中间图里切换。
  - 右：用**视觉估计结果**重建的仿真（不要求真实 ground truth）：有车架 + 对应装甲板即可，素材来自 `bevy_robomaster_simulator`。
- 每帧数据都带**真实时间戳**；目标是现实与仿真尽量完全同步，做不到时靠时间戳对齐供 AI 对比。

## 3. 阶段计划

### P0：仿真接口对齐（只读调研）

目标：不改代码，先把仿真的真实接口、坐标、单位、时基搞清楚。

> 注意：用户尚未最终确定仿真接口。下表是参考工程**当前代码的实际实现**，只作为事实参考，不代表最终协议。

仿真的**代码实际接口**（以 `bevy_robomaster_simulator/src/ros2/topic.rs` 为准）：

| 方向 | 话题 | 消息类型 | 说明 |
|---|---|---|---|
| 仿真发布 | `/image_raw` | `sensor_msgs/Image` | `rgb8`，默认 1440×1080 |
| 仿真发布 | `/image_compressed` | `sensor_msgs/CompressedImage` | JPEG；与 `/image_raw` 二选一发布 |
| 仿真发布 | `/camera_info` | `sensor_msgs/CameraInfo` | 由 FOV 计算内参，默认 FOV 45° |
| 仿真发布 | `/tf` | `tf2_msgs/TFMessage` | 场景/机器人/装甲等 TF 树 |
| 仿真发布 | `/gimbal_pose` | `geometry_msgs/PoseStamped` | 云台位姿 |
| 仿真发布 | `/odom_pose` | `geometry_msgs/PoseStamped` | 底盘/里程计位姿 |
| 仿真发布 | `/muzzle_pose` | `geometry_msgs/PoseStamped` | 枪口位姿 |
| 仿真发布 | `/camera_pose` | `geometry_msgs/PoseStamped` | 相机位姿 |
| 仿真发布 | `/simulator/marker` | `visualization_msgs/Marker` | 前哨站等标记 |
| 仿真发布 | `/simulator/tech_core/state` | `std_msgs/String` | 能量机关等状态 JSON |
| 仿真发布 | `/livox/lidar` | `sensor_msgs/PointCloud2` | 雷达点云 |
| 仿真订阅 | `/rm_gimbal/cmd` | `rm_interfaces/msg/GimbalCmd` | QoS: sensor_data |

`GimbalCmd` 字段：

```
std_msgs/Header header
float64 pitch
float64 yaw
float64 yaw_diff
float64 pitch_diff
float64 distance
bool fire_advice
```

需要确认/记录：

- 仿真 README 里写的 `/armor_solver/cmd_gimbal` 与代码实际 `/rm_gimbal/cmd` 不一致，**以代码为准**。
- 图像是 `rgb8`，本项目 OpenCV 默认 BGR，需要转换。
- `/image_raw` 与 `/image_compressed` 默认只发一个（`publish_compressed: false`），联调时选一个。
- 位姿/TF 的 frame 命名、坐标轴方向、单位（m/rad）、时间戳来源。
- 云台姿态如何映射到 `Solver::set_R_gimbal2world(q)`。
- `rm_interfaces` 的安装与编译方式。

验收：形成一份“仿真接口事实表”，确认无遗漏。

### P1：自瞄代码在仿真跑通（最小闭环）

目标：让视觉代码吃仿真图像、出云台指令，在仿真里稳定跟踪/瞄准。

**已实现（2026-11-15）**：同机闭环走仿真自带的 Talos 共享内存，不走 ROS2。默认 DDS 传 1440×1080 的 raw 图会有数毫秒到十几毫秒延迟并丢帧，换 CycloneDDS 也只是补丁；仿真和视觉在同一台机器上时直接映射 `/tmp/talos_ipc_meta` 与 `/tmp/talos_ipc_image_pool`。

- `io/sim/talos_shm.*`：与 `bevy_robomaster_simulator/crates/talos-ipc` 逐字节对齐的消费者。编译期用 `static_assert` 锁死字段大小和 `camera_info` 偏移（1728）。
- `io/sim/sim_io.*`：`SimCamera` 读 BGR 图、云台四元数、仿真时钟；`SimGimbal` 把 `io::Command` 的弧度换成仿真要的度数写回。
- `src/sim_standard.cpp`：循环与 `src/standard.cpp` 相同，复用 `YOLO` / `Solver` / `Tracker` / `Aimer` / `Shooter`。`control == false` 时写 `distance = -1`，仿真据此停止云台 PID。
- `configs/sim.yaml`：仿真相机内参（1440×1080、垂直视场角 45°）、单位外参、弹速 23。

坐标约定：共享内存通道在消费前对 pitch 取反（`src/talos/plugin.rs`），再交给 `from_solver_degrees`，后者做 `(pitch_deg - 90°) × π/180` 且抬头为负。两步合起来，本侧写 `pitch_deg = -(90 + pitch_rad × 180/π)`。ROS2 通道不取反，换算不同，不要混用。

验证（2026-11-15）：`sim_standard` 已完整编译。`tests/talos_loopback` 按仿真布局持续发帧，`sim_standard` 在 6 秒内回写 489 帧云台指令，无解时 `distance = -1`，无掉线误判、无日志刷屏。还差一件事要在真实仿真里确认：画面里有装甲板时 yaw/pitch 的方向是否指对目标。

运行：先启动带 `talos` 特性的仿真，再跑 `./build/sim_standard configs/sim.yaml`。该入口不链接海康/迈德威视 SDK，也不依赖 ROS2，x86_64 与 aarch64 编的是同一份代码。推理后端仍由 `YOLOBase` 按编译开关选择。

计划步骤：

1. 新增一个仿真入口（后续实现，例如 `src/sim_standard.cpp`）：
   - 订阅 `/image_raw`（或 `/image_compressed`）、`/camera_info`、云台姿态（`/gimbal_pose` 或 `/tf`）。
   - 复用 `Detector` / `YOLO` / `Solver` / `Tracker` / `Aimer` / `Shooter`。
   - 发布 `/rm_gimbal/cmd`。
2. 处理图像格式、时间戳、相机内参。
3. 对齐云台姿态/坐标，校验 yaw/pitch 方向。
4. 跑通后记录延迟、帧率、跟踪状态。

验收：

- 假人移动时能稳定 `tracking`。
- 云台指令方向正确，`fire_advice` 合理。
- 长时间运行不崩溃、不发散。

### P2：调试图总线与切换

目标：把代码里的多张中间图变成可远程选择/切换的流，供左屏显示。

当前已知的调试图来源（需继续梳理）：

- 原始 BGR 图
- 二值化图 `binary_img`（`tasks/auto_aim/detector.cpp`）
- 检测结果叠加图（`Detector::show_result`）
- 重投影图（`src/uav_debug.cpp`、`src/sentry_debug.cpp` 的 `imshow("reprojection")`）
- YOLO 输入/输出、plotter 数据（后续确认）

计划：

1. 设计“调试图总线”：每条流有 `name`、`stamp`、`encoding`、`image`。
2. 边缘是 headless 环境，**去掉/绕过 `cv::imshow`**。
3. 用 ROS2 图像话题或单话题 + 选择指令实现切换。
4. 左屏显示选中的流，并显示流名、帧率、时间戳。

验收：PC 上能切换至少 raw / binary / detector / reprojection 四路。

### P3：边缘 → PC 遥测转发

目标：把实机运行所需的数据低延迟传回 PC。

需要转发的数据分两类：

- **图像类**：选中的调试图流（优先压缩，如 JPEG / H.264）。
- **状态类**：时间戳、云台四元数/角度、底盘位姿、模式、目标 EKF 状态、装甲板列表、瞄准点、开火建议、弹速、延迟指标。

已确认的网络条件：

- 可能是**网线直连**，也可能是**无线网卡 / 共享热点**。
- 因此传输层必须能适应带宽波动，不能假设稳定千兆局域网。

可选传输方案（仍待定）：

- ROS2 DDS 跨机
- MQTT（protobuf / JSON）
- 自定义 UDP/TCP（protobuf / JSON）
- WebRTC（适合视频 + 低延迟）

必须解决：

- 带宽：1440×1080 raw 太大，必须压缩或降分辨率/选流；无线热点下要能自适应。
- 时间同步：方案未定，但**每一帧都必须携带真实时间戳**；PC 侧按时间戳对齐，而不是按到达时间。
- 断线重连、丢包、延迟统计。
- PC 端要能运行同一套视觉代码，因此转发数据必须足够驱动：图像 + 云台姿态 + 弹速 + 模式等输入。

验收：PC 能稳定收到图像 + 状态，能跑通同一套视觉代码，且每帧有时间戳。

### P4：PC 侧数字孪生重建

目标：右屏用**视觉估计结果**重建仿真机器人，与左屏实机画面对比。

已确认的重建语义：

- **视觉信念重建**：用视觉输出的装甲板位姿、目标 EKF 状态、云台姿态重建“机器人认为的世界”。
- 不要求真实 ground truth，也不要求底盘物理仿真。
- 精度要求：**有车架 + 对应装甲板即可**；素材直接复用 `bevy_robomaster_simulator` 里已有的模型/场景。

计划：

1. 给 Bevy 增加“外部遥测驱动”模式：不跑本地物理，按外部时间戳更新实体位姿。
2. 右侧场景：车架 + 云台 + 目标装甲 + 瞄准点（视觉估计）。
3. 左侧：实机处理图 + 可切换流。
4. 左右共用同一时间轴，支持暂停、逐帧查看。

验收：左右画面能按时间戳对齐，装甲板/云台姿态无明显错位。

### P5：Sim2Real 对比与调优

目标：量化并缩小仿真与实机差异。

计划：

- 同一时间戳并排显示：左实机、右仿真。
- 指标：重投影误差、目标状态差、瞄准点差、云台响应延迟、开火时机差。
- 用实机数据反向校准仿真参数（FOV、延迟、噪声、摩擦等）。

已确认的范围：

- **重建机器人功能不需要**注入噪声/延迟/丢帧。
- **纯 PC 仿真模式**直接按参考工程 `bevy_robomaster_simulator` 的现有实现跑，不额外魔改。
- 离线录制/回放暂不做，列为未来想法。

验收：能定位并解释至少一类 sim2real 差异。

## 4. 数据接口草案（待确认）

调试图流元数据：

```
DebugStream {
  name: string          // raw / binary / detector / reprojection / ...
  stamp: time
  encoding: string
  width, height: int
  data: bytes
}
```

遥测状态草案：

```
Telemetry {
  stamp
  gimbal_quaternion / gimbal_ypr
  chassis_pose / odom
  mode
  bullet_speed
  target: { state_x[11], converged, name, armor_num }
  armors: [{ id, name, type, xyz, yaw, priority }]
  aim_point: { xyz, armor_id }
  fire_advice: bool
  latency_ms
}
```

## 5. 已确认决策（2026-11-15）

| # | 问题 | 决策 |
|---|---|---|
| 1 | “本地代码”指什么 | **PC 上运行同一套 `yunqian_vision` 视觉代码** |
| 2 | 右屏重建语义 | 用**视觉估计结果**重建；有车架 + 对应装甲板即可；素材用 `bevy_robomaster_simulator` |
| 3 | 边缘→PC 链路 | 可能网线直连，也可能无线网卡/共享热点；传输要能适应带宽波动 |
| 4 | 左屏图片选择 | PC 上点开菜单，选择**传哪张图**（控制边缘选流） |
| 5 | 仿真接口 | **暂未定**，后续设计；当前代码里的 `/rm_gimbal/cmd` 只作事实参考 |
| 6 | 时间同步 | 方案未定；期望现实与仿真尽量完全同步，**至少每帧带真实时间戳**供 AI 对比 |
| 7 | 噪声/延迟注入 | 重建机器人不需要；纯 PC 仿真模式按参考工程现有实现 |
| 8 | 离线录制/回放 | 暂不需要，列为未来想法 |
| 9 | 算力平台 | 本仓库适配 **NUC**（OpenVINO）；`27-wheel-leg-wbc` 的 `origin/vision` 适配 **Orin NX**（TensorRT/CUDA）。不同机器人用不同平台 |
| 10 | 首台调试机器人 | **三号步兵，Orin NX**。开该机器人分支时移植 vision 分支的平台适配，不直接使用本仓库 NUC 构建 |
| 11 | 本地开发是否依赖 NUC | **不依赖**。NUC 只提供 x86_64 + Intel iGPU + 四个外设；个人电脑占前两项，缺的四项外设用替身替换后即可完整验证业务逻辑。详见 [`docs/local_dev_platform.md`](docs/local_dev_platform.md) |
| 12 | 平台差异如何隔离 | 平台相关代码锁在 `auto_aim::YOLOBase` 与 `io::CameraBase` 之后；`SP_WITH_TENSORRT` / `SP_WITH_CUDA_PREPROCESS` 只允许出现在后端实现与 CMake，禁止进入 `solver` / `target` / `tracker` / `aimer` / `shooter` / `detector` 后处理 |

### 5.1 仍需确认

1. 仿真接口的具体定义：话题、消息、坐标系、单位、时间戳语义。
2. 调试图清单与优先级：raw / binary / detector / reprojection / YOLO / plotter 等。
3. 边缘→PC 的传输方案：ROS2 DDS / MQTT / 自定义 UDP/TCP / WebRTC。
4. 时间同步的具体实现：PTP / NTP / 软同步。
5. “纯 PC 仿真模式”的启动方式与配置入口。

## 6. 风险

- README 与实际话题/消息不一致，按错接口实现。
- 坐标轴、单位、左右手系、图像 RGB/BGR 弄错。
- 边缘无显示器，`imshow` 类代码会阻塞或崩溃。
- 把 NUC 构建直接拿到 Orin NX（或反过来）会在架构、OpenVINO 设备、TensorRT engine 上失败。三号步兵必须走 Orin NX 路径。
- 原始图像带宽过大，跨机传输不稳定。
- 时间戳/延迟不同步，导致 sim2real 对比失真。
- `rm_interfaces` 依赖安装/编译问题。
- 重建语义不清，做出“看起来对但没意义”的画面。

## 7. 下一步（仍不改代码）

1. 基于 §5 已确认决策，产出 **P1 仿真入口设计草案**（仿真接口未定，先给候选方案）。
2. 只读核对仿真的 TF 树、坐标轴、时间戳、`GimbalCmd` 语义。
3. 设计 **调试图菜单 + 控制通道**（PC 选择边缘传哪张图）。
4. 设计**每帧时间戳 + 遥测格式**（保证 AI 可对齐）。
5. 以上确认后再开始写代码。

## 8. 自瞄算法与评估改进计划（借鉴 `sp_vision_26`，2026-11-20）

基于《sp_vision_26区域赛存档》（`BV1wYLR6PEGh`）中展示的高射频小陀螺追瞄、自身小陀螺近距射击、大小惯量云台对比及 `sim2aim` + Rerun 闭环评估，将以下四项纳入迭代计划（均未改代码，待按优先级逐项确认后实施）：

| 编号 | 改进项 | 目标与要点 | 涉及模块 | 状态 |
|---|---|---|---|---|
| **A1** | 修复 `Aimer` 小陀螺分支 Bug + 仿真接入 MPC `Planner` | 修复 `aimer.cpp:163` 下标误用（KI-003：`ekf_x()[8]` → `ekf_x()[7]`）；在 `sim_standard` 中支持 MPC `Planner` 路径（输出平滑锯齿轨迹 + 速度/加速度前馈），与 `Aimer + Shooter` 路径可对比 | `tasks/auto_aim/aimer.cpp`、`src/sim_standard.cpp`、`configs/sim.yaml` | 已实现（支持 `--mode=aimer\|planner` 切换） |
| **A2** | 改进 MPC `Planner` 提前量、火控与动力学约束 | ① `Planner::plan()` 飞行时间改为迭代收敛，消除平移+旋转及近距大角速度下的提前量误差；② 开火条件升级为 3D 弹道微段与所有正面装甲板直接求交（`check_ballistic_hit`：含 3D 入射角 `max_fire_angle=55°`、机身圆柱遮挡、17mm 弹丸半径与 `fire_margin=0.85`），抑制切板边缘误射并拉长有效开火占空比；③ 按各车云台惯量标定 `max_yaw_acc` 与延迟参数（三号步兵 Yaw `J=0.0146 kg·m²` → 实车 `120–180 rad/s²`） | `tasks/auto_aim/planner/planner.*`、`configs/*.yaml` | ①②③ 已实现并完成仿真闭环验证（小陀螺 `recon_hit=99.2%` / `sim_hit=96.2%` / `fire_duty=73.7%+` / `yaw_rms=1.33°`） |
| **A3** | 弹道解算加入空气阻力模型与 3D 跑打 | `tools/trajectory.*` 升级为含二次空气阻力的迭代解算（分子有理化消除近距数值不稳定），并新增 `Trajectory3D` / `ballistic_point_3d` 叠加自身底盘速度 `v_self` | `tools/trajectory.*`、`aimer.*`、`planner.*`、`sim_metrics.*` | 已实现并单测验证（`shoot_on_move_test` 5 项全 100% 命中） |
| **A4** | 仿真闭环定量评估与真实云台动力学复现（借鉴 `sim2aim`） | 在仿真闭环/可视化侧补齐 `plan_yaw` vs `gimbal_yaw` 时间轴对比曲线与**重建命中 / 仿真实际**双命中率 KPI/曲线及清除弹丸统计按钮；新增 `io/sim/sim_gimbal_model.*` 复现三号步兵 500 Hz LQI + 50 Hz 陷波器 + 摩擦前馈 + 6 ms 力矩环滞后 + DM4310 转速-力矩包络；导出电控离线调参包 `records/ec_gimbal_tuning_pack.zip` | `tools/sim_metrics.*`、`io/sim/{sim_gimbal_model,talos_shm,sim_io}.*`、`src/sim_standard.cpp`、`tools/web_viewer.*`、`web/index.html` | 已实现并完成 100 Hz 实时闭环验证（`daedalus` 240 Hz 物理步进、RTX 4060 TRT 加速、双命中率对齐） |
| **A5** | 视觉弹道闭环（识别飞行弹丸轨迹修正弹道偏差） | 裁判系统弹速仅 10 Hz 且有延迟；参考 SJTU `rm.cv.fans`（`BV1vX4y1W7U7`）及 `27-wheel-leg-wbc` `origin/vision` 移植 `tasks/auto_aim/aim_corrector.*`：发射事件按实测 IMU 姿态生成虚拟弹丸（对接本仓库二维耦合 RK4 空气阻力 `ballistic_point` 与 `muzzle_z_offset`）→ 局部 ROI 单应帧差 ∩ 荧光绿 HSV 检出真实弹丸 → 偏差中位数同步写入 `Aimer` 与 `Planner` 的 `yaw_offset / pitch_offset` | `tasks/auto_aim/aim_corrector.*`、`solver.*`、`aimer.*`、`planner.*`、`tools/trajectory.*`、`tools/runtime_params.hpp`、`src/sim_standard.cpp` | 已移植并打通 `Aimer` / `Planner` 与 `sim_standard --corrector` 预览；`aim_corrector_test` 单元测试通过（合成误差恢复精度 < 0.01°）；待实车接入发射反馈 |
| **A6** | 跑打（底盘速度 + `roll` 倾角融合） | `Target::predict` / `focus_roi` 与 `Tracker` 支持 `self_vel` 解耦预测；`Planner` 新增 `SelfMotion` 融合 `roll` 引起的 3D 炮口平移与零偏旋转；仿真通过 `talos_shm` 读取 `chassis_observation`，实机 `io/gimbal` 补充 IMU `roll` 提取与底盘速度预留字段 | `tools/trajectory.*`、`tasks/auto_aim/{target,tracker}.*`、`planner.*`、`io/sim/talos_shm.*`、`io/gimbal/gimbal.*`、`src/{sim_standard,standard_mpc}.cpp`、`tests/shoot_on_move_test.cpp` | 已实现，`shoot_on_move_test`（直行/横移/坡面大 roll/跑打+小陀螺/高频换板）全部 100% 命中、`fire_duty` 81.6%–87.3%；实机待串口下发底盘速度 |
