# yunqian_vision — AI 协作说明

> 本文件写给在本仓库工作的 AI 和新队员。开始任何改动前先读完。
> 最后整理：2026-11-15。标注“待确认 / 讨论中”的内容不要擅自实现。

## 0. 速查

| 项 | 内容 |
|---|---|
| 代码基线 | `sp_vision_25` @ `bd9f5e7`（已完整复制，业务代码未改） |
| 本仓库算力平台 | **NUC**（x86_64，OpenVINO）。不是 Orin NX |
| Orin NX 适配 | `27-wheel-leg-wbc` 的 `origin/vision`（同一套 sp25 基线 + TensorRT/CUDA） |
| 首台调试机器人 | **三号步兵**，算力平台 **Orin NX**（已开分支 `standard3-orin`，准备移植推理/部署层并实机部署） |
| 仿真参考 | `bevy_robomaster_simulator` @ `07af246` |
| 已确认规则变更 | 2027 规则 §5.2 自适应装甲（已录入，未实现） |
| 关键待确认 | 加固信息接口、未知模式行为、σ 统计口径、ROI 用法；其余机器人的平台分配 |
| 硬约束 | 只改本仓库；规则未确认不实现；不擅自 `commit` / `push`；开机器人分支前先确认算力平台 |

> 短期实施计划（Sim2Real、仿真跑通、边缘→PC 数字孪生）见 `ROADMAP.md`。
> 算力平台与机器人分支规则见 §2.3。开某个机器人的分支之前必须先读。
> 本地开发（手上没有 NUC）如何搭建与验证见 `docs/local_dev_platform.md`。
> 实机与仿真 100 Hz 调试、云台辨识与 MPC 跟随调参流程见 `docs/real_robot_tuning_guide.md`。
> 常用编译、仿真闭环、实机启动、单测与 Orin NX 部署脚本速查见 `scripts/README.md`。

## 1. 项目定位

`yunqian_vision` 是面向 **RoboMaster 2027 赛季** 的视觉/自瞄代码库。

- 初始代码基线：直接复制自 2026 赛季已验证的 `sp_vision_25`。
- 2027 规则发生变化，需要在稳定基线上做针对性变更。
- 目标不是重写，而是：**按 2027 规则做最小、可控、可验证的改动。**

## 2. 参考仓库与规则文件

### 2.1 权威参考（只读，不在其中做修改）

| 用途 | 路径 | 基线 commit |
|---|---|---|
| 代码基线 / 上游参考 | `/home/scj/桌面/rm2027/opensource/sp_vision_25` | `bd9f5e7` |
| Orin NX 视觉适配（只读参考） | `/home/scj/桌面/rm2027/27-wheel-leg-wbc` 的 `origin/vision` | `93cb1b1`（源码提交 `50288b4`） |
| 仿真、ROS2 接口、闭环验证参考 | `/home/scj/桌面/rm2027/opensource/bevy_robomaster_simulator` | `07af246` |
| 自瞄算法参考（只读参考） | `/home/scj/桌面/rm2027/opensource/awakening`（`WUST-RM/awakening`） | `7a0c4ba` |

这两个开源项目在 2026 年取得了显著成果，是**权威参考**：

- `sp_vision_25`：识别 → 解算 → 整车 EKF 估计 → 瞄准/开火决策 → 下位机通信的完整实现。
- `bevy_robomaster_simulator`：Bevy 物理/渲染仿真，发布 `/image_raw`、`/camera_info`、`/tf`，订阅 `/armor_solver/cmd_gimbal`，可用于自瞄闭环验证。

### 2.2 规则 / 协议 PDF

以下为**绝对路径，后续查阅规则、补全代码时必须使用**：

```
/home/scj/桌面/rm2027/rule/
├── UTF-8__RoboMaster 2027 机甲大师超级对抗赛比赛规则手册V1.0.0（20261015）.pdf   # §5.2 自适应装甲
├── UTF-8__RoboMaster 2027 机甲大师高校系列赛通信协议 V1.0.0（20261029）.pdf      # 加固字段 0x0A05 / 0x020E / 0x0121 / 0x0206 / 0x0301
├── UTF-8__RoboMaster 2027 机甲大师高校联盟赛比赛规则手册 V1.0.0（20261015）.pdf
└── UTF-8__RoboMaster 2027 机甲大师高校系列赛机器人制作规范手册V1.0.0（20261015）.pdf
```

当前已用到的两个关键文件：

- `/home/scj/桌面/rm2027/rule/UTF-8__RoboMaster 2027 机甲大师超级对抗赛比赛规则手册V1.0.0（20261015）.pdf`
- `/home/scj/桌面/rm2027/rule/UTF-8__RoboMaster 2027 机甲大师高校系列赛通信协议 V1.0.0（20261029）.pdf`

> PDF 有加密但允许复制/提取文本，可用 `pdftotext -layout` 导出后搜索关键词（如“加固”）。

**只修改 `yunqian_vision`，绝不修改上述参考仓库。** `27-wheel-leg-wbc` 同样只读；需要 Orin NX 适配时，在本仓库开分支并移植，不要改那个仓库。

### 2.3 算力平台与机器人分支（已确认，2026-11-15）

同一套 `sp_vision_25` 基线已经按两种边缘算力平台分叉。**不同种类的机器人使用不同算力平台**，开某个机器人的分支时必须先确认它跑在哪一块板上，不能把 NUC 代码直接拿到 Orin NX 上编，也不能反过来。

| 算力平台 | 代码在哪 | 架构 / 推理 | 适用 |
|---|---|---|---|
| **NUC**（上游写明 NUC12WSKI7，i7-1260P） | **本仓库** `yunqian_vision` | x86_64，OpenVINO（`/opt/intel/openvino_2024.6.0`） | 使用 NUC 的机器人 |
| **NVIDIA Orin NX**（部署记录为 Orin NX 16G） | `27-wheel-leg-wbc` 的远程分支 `origin/vision` | aarch64，JetPack 6.2.2，TensorRT 10.3 + CUDA（`sm_87`），OpenVINO 仅作 CPU 回退 | 使用 Orin NX 的机器人 |

`origin/vision` 不是电控 `wbc` 分支的子目录，本地工作区默认检出的是 `wbc`。查看方式：

```bash
git -C /home/scj/桌面/rm2027/27-wheel-leg-wbc show origin/vision:JETSON_DEPLOY.md
```

相对本仓库，Orin NX 适配已经改过的部分（移植时以 `JETSON_DEPLOY.md` 为准，不要凭记忆重写）：

- `configs/*.yaml`：`device: GPU` 改为 `CPU`（Jetson 没有 Intel GPU）。
- `configs/{demo,standard3,standard4}.yaml`：增加 `trt_engine_path`，空则回退 OpenVINO。
- `tasks/auto_aim/CMakeLists.txt`：探测 TensorRT/CUDA，定义 `SP_WITH_TENSORRT`；aarch64 上编译 CUDA 预处理，`CMAKE_CUDA_ARCHITECTURES=87`。
- 新增 `tasks/auto_aim/yolos/trt_engine.*`、`cuda_preprocess.*`；`yolo11` / `yolov5` 在配置了 engine 时走 TensorRT。
- `io/CMakeLists.txt`：aarch64 且存在 `/opt/MVS/lib/aarch64` 时优先用官方 MVS。
- engine 与本机绑定，不能把 Orin 上编出来的 `.engine` 拿到 NUC 上用。

**已确认的机器人分配：**

| 机器人 | 算力平台 | 分支状态 |
|---|---|---|
| 三号步兵（计划内第一台调试） | **Orin NX** | 已开分支 `standard3-orin`（以本仓库为业务基线，推理/部署层按 `JETSON_DEPLOY.md` 移植 `origin/vision`；目标机 `yq11@192.168.55.1`） |
| 其余兵种 / 机器人 | 未确认 | 开分支前必须先问用户该机器人用 NUC 还是 Orin NX |

开机器人分支时的规则：

1. 先确认兵种和算力平台，再起分支。平台未确认就不要开始改构建、推理或部署脚本。
2. 三号步兵分支的目标平台是 Orin NX。业务改动（2027 规则、自瞄逻辑）仍以本仓库为准；平台差异只限推理后端、CUDA 预处理、相机 SDK、部署脚本和对应 YAML。
3. 使用 NUC 的机器人分支保持本仓库的 OpenVINO 路径，不要引入 TensorRT/CUDA 作为必需依赖。
4. 共享的识别 / 解算 / EKF / 决策逻辑尽量保持可对比；平台专属代码用现有的 `SP_WITH_TENSORRT` / `SP_WITH_CUDA_PREPROCESS` 这类编译开关隔开，避免复制出两套业务代码。
5. 不把 `27-wheel-leg-wbc` 的电控分支（`wbc` / `gimbal` / `chassis`）混进视觉分支。

**本地开发（手上没有 NUC）如何做验证**：见 [`docs/local_dev_platform.md`](docs/local_dev_platform.md)。要点：NUC 不是本地开发的必要条件；业务逻辑与推理后端解耦，缺的四项外设用替身替换；平台相关代码必须锁在 `YOLOBase` / `CameraBase` 之后。

## 3. 变更工作流

1. 用户逐条给出 2027 规则变更需求。
2. 收到需求后，先在 §12 变更记录中登记（日期、规则依据、影响模块、状态）。
3. 只实现已明确确认的变更；需求模糊时先提问，不要猜。
4. 每次改动后尽量编译验证；无法编译时明确说明原因。
5. 改动尽量局部化，保持与上游 `sp_vision_25` 的可对比性。
6. 未经用户明确要求，不执行 `git commit` / `push`。

## 4. 仓库结构与核心数据流

### 4.1 目录结构

```
yunqian_vision/
├── CMakeLists.txt          # C++17，主工程入口
├── configs/                # 各兵种/相机 YAML 配置
├── io/                     # 硬件与通信输入输出
│   ├── camera.cpp/.hpp     # 相机抽象
│   ├── cboard.cpp/.hpp     # 下位机通信
│   ├── gimbal/             # 云台
│   ├── dm_imu/             # IMU
│   ├── hikrobot/           # 海康相机
│   ├── mindvision/         # 迈德威视相机
│   ├── usbcamera/          # USB 相机
│   ├── serial/             # 串口
│   └── ros2/               # ROS2 通信（哨兵/导航预留）
├── tasks/
│   ├── auto_aim/           # 自瞄核心
│   │   ├── detector.*      # 装甲板检测（传统/YOLO）
│   │   ├── yolo*           # 神经网络推理
│   │   ├── classifier.*    # 装甲板分类
│   │   ├── armor.*         # 装甲板数据结构
│   │   ├── solver.*        # solvePnP 位姿解算 + yaw 优化
│   │   ├── target.*        # 整车 EKF / 目标状态估计
│   │   ├── tracker.*       # 目标跟踪状态机
│   │   ├── aimer.*         # 瞄准位置决策
│   │   ├── shooter.*       # 开火决策
│   │   ├── voter.*         # 目标投票/筛选
│   │   └── planner/        # MPC 轨迹规划（TinyMPC）
│   ├── auto_buff/          # 能量机关
│   └── omniperception/     # 全向感知（多相机）
├── tools/                  # 数学、EKF、日志、PID、轨迹等通用工具
├── src/                    # 各兵种入口（standard/sentry/uav/...）
├── tests/                  # 各模块独立测试可执行程序
└── calibration/            # 相机/手眼标定程序
```

### 4.2 核心数据流

```
相机图像 ──> detector/yolo ──> 装甲板关键点 + 类别
                                   │
                                   v
                            solver (solvePnP)
                                   │
                     xyz_in_world / ypr_in_world / ypd_in_world
                                   │
                                   v
                     tracker ──(选目标/状态机/装甲板匹配)
                                   │
                                   v
                     target (EKF predict + update)
                                   │
                                   v
                     aimer / shooter / planner
                                   │
                                   v
                     云台指令 ──> cboard / ROS2 / 仿真
```

## 5. 目标观测与 EKF 状态（改 EKF 必读）

状态向量 `x`（13 维，见 `tasks/auto_aim/target.cpp`）：

| 下标 | 符号 | 含义 |
|---|---|---|
| 0 | x | 旋转中心 world x |
| 1 | vx | x 方向速度 |
| 2 | y | 旋转中心 world y |
| 3 | vy | y 方向速度 |
| 4 | z | 旋转中心 world z |
| 5 | vz | z 方向速度 |
| 6 | a | 参考装甲板 yaw 角 |
| 7 | w | yaw 角速度 |
| 8 | r | 装甲板半径 r1 |
| 9 | l | r2 - r1（半径差） |
| 10 | h | z2 - z1（高度差） |
| 11 | roll | 装甲环绕世界 x 轴的倾角 |
| 12 | pitch | 装甲环绕世界 y 轴的倾角 |

当前观测量 `z`（6 维，`Target::update_ypda`）：

```
z = [yaw, pitch, distance, armor_yaw, armor_pitch, armor_roll]
```

- 前三维：装甲板中心世界系球坐标
- 后三维：当前板在世界系的 yaw/pitch/roll

EKF 是通用实现（`tools/extended_kalman_filter.*`），`update(z, H, R, h, z_subtract)` 接受任意观测维度和非线性观测函数。若要把观测系改为 **uv 3dof（图像平面 u、v + 角度）**，只需替换 `Target` 中的观测定义 `h` 与雅可比 `H`，状态量无需改变。

发散/收敛判断：

- `Target::diverged()`：半径是否合理
- `Target::convergened()`：更新次数 + 半径
- EKF NIS 卡方检验：连续失败判失锁

## 6. 构建与验证

依赖（与上游一致）：OpenCV、Eigen3、fmt、spdlog、yaml-cpp、nlohmann_json、OpenVINO、Ceres（能量机关）。详见 `CMakeLists.txt` 与 `readme.md`。

```bash
# 推荐：使用快捷脚本（自动识别本机 .deps/toolchain.cmake 或板端系统依赖，并配置动态库路径）
./scripts/build.sh sim        # 编译仿真闭环 + 4 项离线单测
./scripts/build.sh real       # 编译实机程序（standard_mpc / camera_web_test / real_gimbal_sysid）
./scripts/run_sim.sh          # 一键启动 daedalus 仿真器 + sim_standard（WebViewer :8090）
./scripts/run_tests.sh        # 运行 4 项离线单测（shoot_on_move / uvl_tracker / aim_corrector / sim_camera_model）
./scripts/run_standard.sh     # 实机安全调试启动（默认 --no-fire --web；实弹加 --live）
./scripts/deploy_orin.sh --build  # 同步代码到三号步兵 Orin NX 并远程编译

# 手动 CMake 构建（NUC / 板端有系统级依赖时）
cmake -B build
make -C build -j$(nproc)
```

更多参数与脚本说明见 [`scripts/README.md`](scripts/README.md)。本项目没有统一的 `ctest` 流程；测试以独立可执行程序为主，改动 C++ 后至少确认相关 target 能编译通过并跑通 `./scripts/run_tests.sh`。

## 7. 2027 规则：自适应装甲（§5.2）

### 7.1 规则要点

来源：规则手册 §5.2（P72–73）。

- 比赛开始时所有装甲板为“默认”状态。
- 双方每经过 1min 获得 1 次可累积的**加固机会**。
- 一方当局累计命中弹丸数 ≥ 10 后，该方**雷达**可随时向裁判系统服务器发送指令，消耗 1 次机会选择/改变 **1 个中心点**。
- 在中心点下次改变前，该方**所有机器人、前哨站、基地**的装甲模块上，以该中心点为圆心、面积为 **A** 的圆形区域为“已加固”，其余为“未加固”。
- 伤害倍率：
  - 命中未加固区域：**×1.5**
  - 命中已加固区域：**×0.5**
  - 命中默认状态区域：×1
- 该倍率作用于弹丸原始伤害，与攻击/防御增益为乘算。
- 每发弹丸的击中位置 `(x, y, 区域类别)`：
  - 通过裁判系统常规链路发送给**受击方**；
  - 同时在**雷达无线链路**发送给**攻击方**。
- 装甲模块坐标系：以装甲模块检测面**左下角为原点 O**，沿检测面**水平向右为 x 正方向、竖直向上为 y 正方向**。
- 面积计算：
  - 命中率 `η = 对方被命中的所有弹丸数 / 己方发射的所有弹丸数`
  - 散布圆半径 `σ`：所有命中点 `(x,y)` 取横、纵坐标各自的中位数 `(x0,y0)`，σ = 各命中点到 `(x0,y0)` 距离的中位数
  - 准确率 `Λ = η / σ`
  - 相对准确率 `ρ = Λ_己方 / Λ_对方`
  - 可加固面积 `A = 装甲板面积 × ρ² / (2 + ρ²)`（ρ 越大，A 越接近整块装甲板面积）

### 7.2 通信协议字段

来源：通信协议（搜索“加固”）。

| 命令码 | 名称 / 链路 | 加固相关字段 |
|---|---|---|
| `0x020E` | 雷达自主决策信息同步，服务器→**己方雷达**，1Hz，8B | offset1 bit0：是否存在生效加固区；bit1-3：剩余加固机会；offset2-3 圆心 x(mm)；4-5 圆心 y(mm)；6-7 半径(mm) |
| `0x0121` | 雷达自主决策指令，雷达→服务器，子内容 ID `0x0121`，13B | offset8：加固请求计数器（单调递增）；offset9-10 圆心 x(mm)；11-12 圆心 y(mm) |
| `0x0A05` | 对方各机器人增益效果，信号发射源→**雷达**，雷达无线链路，10Hz，47B | offset41-42：对方加固圆心 x(mm)；43-44：圆心 y(mm)；45-46：半径(mm，0 为暂无加固情报) |
| `0x0206` | 伤害状态数据，主控→对应机器人 | offset0：装甲模块 ID + 扣血原因；offset1-2：命中事件序号；offset3-4：命中 x(mm)；5-6：命中 y(mm) |
| `0x0301` | 机器人交互数据，仅限己方，≤30Hz，内容≤112B | 子内容 ID `0x0200~0x02FF` 为机器人之间通信，可用于**雷达→自瞄机器人**中继加固信息 |
| `0x0311` | 自定义终端→机器人，图传链路，≤75Hz，60B | 另一条可选的自定义数据下发通道 |
| `0x0208` | 服务器→己方重装/步兵/哨兵，10Hz，8B | 己方 17mm/42mm 允许发弹量、剩余金币、堡垒储备弹量（ROI 输入） |

加固圆心/半径坐标系：**装甲模块检测面左下角为原点 O，x 向右、y 向上，单位 mm**。

关键结论：

1. 敌方加固圆只到**己方雷达**（`0x0A05`），自瞄机器人不能直接从裁判系统拿到。
2. 自瞄机器人要获得敌方加固信息，必须由雷达通过 `0x0301`（子内容 `0x0200~0x02FF`）或经 `0x0311` 中继下发。
3. 半径已由协议直接给出，机器人侧不需要自己按 5.2 公式计算 A/ρ。
4. `0x0206` 是**受击方**拿到自己的命中位置，不是攻击方拿到命中反馈；它更适合用于我方雷达选择自己的加固中心。
5. 当前 `io/cboard.*` 只转发 IMU 四元数与弹速，**没有解析裁判系统加固数据**；接入点应放在 MCU 转发（CAN）或 ROS2/自定义通信层。

### 7.3 架构：未知 / 已知敌方加固区两套逻辑

已确认的运行架构：

- **自瞄（视觉）跑在上位机**（本仓库），**电控跑在下位机（MCU）**。
- 下位机解析裁判系统协议，并把视觉需要的加固信息转发给上位机。
- 视觉侧区分两种模式。

#### 模式 A：未知敌方加固区（默认 / 无情报）

- 触发条件：未收到加固情报，或半径 `= 0`。
- 行为：保持现有自瞄逻辑，瞄准装甲板几何中心（当前 `aimer` 默认行为）。
- 不在未知状态下做激进偏移，避免无依据打偏。

#### 模式 B：已知敌方加固区

- 输入：`{center_x_mm, center_y_mm, radius_mm}`。
- 行为：在装甲板有效区域内，尽量选择**加固圆之外**的点（未加固 ×1.5，已加固 ×0.5）。
- 若加固圆覆盖整个装甲面：
  - 该装甲没有 ×1.5 区域，考虑改打其他装甲/目标；
  - 同一圆心+半径作用于所有装甲模块，大装甲面积更大、未加固比例更高，可优先打大装甲。
- 最终瞄准点仍需满足弹道可达性、云台可达性和命中概率约束，不能只按几何选最远点。

#### 期望数据流

```
裁判系统 ──0x0A05──> 己方雷达（拿到敌方加固圆心+半径）
                        │
                        └─0x0301 子内容 0x0200~0x02FF（或自定义链路）
                                   │
                                   v
                          下位机 MCU 解析/转发
                                   │
                                   v
                        上位机 yunqian_vision（新增输入接口）
                                   │
                                   v
                         aimer：未知模式 / 已知模式
```

注意：机器人常规链路本身拿不到敌方加固圆（`0x0A05` 只发给雷达），因此“MCU 解析裁判协议”必须是指 MCU 收到**雷达中继的数据**，而不是直接解析常规链路的敌方加固字段。

### 7.4 瞄准点到装甲坐标的映射

当前 PnP 装甲局部系：`x` 法向、`y` 水平、`z` 竖直，中心在原点。

- 协议 `(u_mm, v_mm)` → 装甲局部：
  - `y = u_mm - 装甲宽度 / 2`
  - `z = v_mm - 装甲高度 / 2`
  - `x = 0`
- 宽度：小装甲 `135mm`、大装甲 `230mm`；高度：灯条长 `56mm`（见 `solver.cpp` 的 `SMALL_ARMOR_POINTS` / `BIG_ARMOR_POINTS`）。
- 再经装甲位姿（yaw/pitch、世界系）转成世界坐标，交给 `aimer` / `planner`。
- 左右方向与装甲局部 `y` 轴的符号需结合装甲朝向确认。

### 7.5 未决问题

- 加固信息接口：走 CAN 还是 ROS2？字段、单位、`has_info` 如何定义？
- 未知模式是否只瞄中心，是否需要保守偏移？
- σ 是否区分弹种、是否跨大/小装甲统一计算？
- 瞄准点偏移如何在 `aimer` / `planner` 中落地（几何点 → 可达轨迹）？

## 8. 策略与经济性分析（讨论中，未定稿）

自适应装甲把“打哪、打多准、打多快”变成了策略/博弈问题，而不是纯视觉问题。

### 8.1 散布、命中率与自适应装甲

- **散布是双向的**：
  - 低散布 → `σ↓`、`Λ↑` → 我方可加固面积 A 更大、敌方 A 更小（防御收益）。
  - 但低散布意味着命中点集中，敌方一个固定圆更容易覆盖我方大部分命中（进攻被针对）。
  - 工程控制代价：低散布要求更严苛的火控阈值，**同一自瞄时长内“可开火时间窗口”更短**，实际射频/输出下降。
  - 因此存在“最优散布”，不是越低越好。
- **η 与 σ 分开作用**：`Λ = η/σ`。脱靶只直接降低 η（σ 只统计命中点）；散布增大 σ。两者都会降低 Λ，从而放大敌方 A、缩小我方 A。所以“为了打未加固区而接受很低命中率”要谨慎。
- **已知 = exploit，未知 = explore**：
  - 已知圆心/半径时，打圆外收益最高；但固定打同一点会被敌方下次改圆心针对，应对候选未加固点做随机化/混合策略。
  - 未知时打中心可能正撞加固区（×0.5）；需要“高命中率 + 多候选点轮换”，降低被单个圆覆盖的比例。
  - 本质是 minimax：敌方选使其减伤最大的圆心，我方选使“任意半径 r 的圆最大覆盖比例”最小的命中分布。
- **待确认**：σ 的统计口径（见 §7.5）。

### 8.2 经济与经验基础数据

规则依据：规则手册 §5.4（P75-77）、§5.5.1（P89-90）；通信协议 `0x0208`。

**金币与弹丸**

- 初始金币 800；`05:59 / 04:59 / 03:59 / 02:59 / 01:59` 各 +100；`00:59` +300；装配能量单元可额外获得。
- 17mm：远程 150 金币/100 发（1.5 金币/发）；非远程 10 金币/10 发（1 金币/发）；金币兑换上限 1000 发/队。
- 42mm：远程 150 金币/10 发（15 金币/发）；非远程 10 金币/1 发（10 金币/发）；上限 100 发/队。**仅重装机器人可发射 42mm**。
- 17mm 由步兵/哨兵发射；步兵开局允许发弹量为 **0**，必须先兑换；每发射 1 发消耗 1 发允许发弹量。
- 金币是**队伍共享池**，还可用于远程回血、立即复活。每发弹的机会成本 = 弹药价 + 挤占回血/复活/队友弹药的机会成本。

**伤害与护甲**

- 原始伤害（表 5-2）：机器人/前哨站/基地，42mm = 200，17mm = 20，即 42mm = 10 × 17mm。
- 42mm 单发价格也约为 17mm 的 10 倍 → **单位伤害金币成本基本一致**：非远程约 `0.05 金币/点伤害`，远程约 `0.075 金币/点伤害`。
- 重装血量：400 / 450 / 500 / 600（表 5-16）；步兵约 150-400（表 5-17）。
- 42mm 屏蔽：重装战亡/离线、无弹或超发、模块通信丢包率 ≥ 5% 等情况下，对方全体装甲屏蔽 42mm 伤害，直到重装存活且允许发弹量 > 0；17mm 不受影响。
- 检测间隔：42mm 最小 200ms，17mm 最小 50ms。

**经验（仅步兵适用；哨兵在击杀结算中视为 1 级）**

- 发射弹丸：每发 +1 经验。
- 造成伤害：对机器人 +4/点；对前哨站装甲 +2/点；对基地装甲 +1/2 点（奇数向上取整）。
- 击杀：若被击毁者等级 `L_v ≥` 击毁者等级 `L_k`，经验 `= 50 * L_v * (1 + 0.2 * (L_v - L_k))`；否则 `= 50 * L_v`。
- 等级门槛：L2 = 550，L10 = 5000（表 5-15）；升级提升上限血量、底盘功率、射击热量/冷却等。
- 经验按“点伤害”结算；17mm/42mm 单位伤害金币成本基本一致，所以伤害经验/金币也基本一致；差异主要来自溢出、命中率和 42mm 屏蔽。
- 伤害倍率影响实际伤害，因此打未加固区（×1.5）很可能同时带来更高经验（按裁判系统实际结算伤害计，口径待确认）。

### 8.3 ROI 决策

单次开火 ROI ≈

```
[p_hit × (实际伤害 × 伤害价值 + 伤害经验 × 经验价值) + p_kill × (击杀经验 + 战术价值)]
/ [弹药金币成本 + 热量/暴露风险 + 对 η/σ 的元游戏成本]
```

- **平台绑定**：42mm 只有重装能打，17mm 属于步兵/哨兵，两者是不同平台的收益模型，不是同一台机器人的选择。
- **经济方差**：42mm 是一发 = 10 发 17mm 钱的“一次大额下注”，未命中整份金币浪费；17mm 把同一笔钱拆成 10 次独立机会，期望相近但相对波动约按 `1/√n` 下降。
- **相关风险**：42mm 屏蔽触发时全部 42mm 同时失效，无法靠分散降低；17mm 不受影响。
- **时间/热量**：42mm 单发爆发快，适合限时击杀/高血量目标；17mm 需要更多射击时间与热量换稳定性。
- **溢出**：42mm 单发 200，打残血目标容易过量；对 400-600 血的重装反而更不溢出（2-3 发）。
- **选择**：`p_hit` 不确定、目标小/远、需要稳定削血 → 17mm；`p_hit` 高、目标血厚、需要短时间爆发/确保击杀 → 42mm。
- **激励错位**：金币队伍共享，经验个人所有，存在“个人刷经验、队伍买单”；决策应以队伍总目标为准。
- **可观测输入**：`0x0208` 提供己方 17mm/42mm 允许发弹量、剩余金币、堡垒储备弹量。

注意区分两种“方差”：

- **经济方差**：一次大注 vs 十次小注，与空间散布无关。
- **空间散布 σ**：规则里的命中点散布。42mm 空间散布是否更大目前**没有依据**，不要和经济方差混为一谈；若后续实测更大，才需要考虑它对全队 σ 的影响。

### 8.4 目标选择

- **敌方重装 = 高价值 + 高血量**：
  - 击杀它移除高 DPS，并触发对敌方 42mm 的屏蔽；但 TTK 长、单次交战 `p_kill` 低，零散伤害可能被回血抵消。
  - 能确保击杀/集火时优先；否则打低血量高价值目标或持续压制可能更划算。
  - 大装甲面积更大，同一加固圆半径下未加固面积更多；重装既是肉盾，也是更容易找到 ×1.5 区域的目标。
- **经验口径错位**：重装不适用步兵经验体系，等级与装配相关；击杀重装的个人经验可能低于高等级步兵，但队伍战略价值更高，需分开计价。
- **我方重装**：高血量 + 高 DPS，是队伍核心，应优先保护/回血，并保持 42mm 输出。
- **17mm 平台**：单发成本低、粒度细，适合压制/补刀/维持命中；避免“为刷经验/解锁 ≥10 命中”浪费队伍金币、拉低 η。
- **综合因素**：敌方等级、剩余血量、能否击杀、加固区倍率、弹丸成本、比赛剩余时间（回血/复活成本随时间变化）。

## 9. 编码约定

- 语言：C++17；构建：CMake。
- 代码风格遵循仓库根目录 `.clang-format`（基于 LLVM 风格，2 空格缩进）。
- 命名空间与现有代码保持一致：`auto_aim`、`tools`、`io`、`omniperception` 等。
- 新增代码沿用现有文件组织与命名习惯；不要无故引入新第三方库。
- 非必要不重构；规则相关的改动保持小而聚焦。
- 注释：跟随现有代码风格（中文说明可接受），不要为显而易见的行为堆砌注释。

## 10. 与仿真联调（可选）

参考 `bevy_robomaster_simulator` 时注意接口差异：

- 仿真发布：`/image_raw`、`/camera_info`、`/tf`
- 仿真订阅：`/armor_solver/cmd_gimbal`
- 本项目 `io/ros2` 当前使用 `sp_msgs::EnemyStatusMsg` / `AutoaimTargetMsg`，发布 `auto_aim_target_pos`

两者话题与消息类型并不直接匹配，联调时需要适配层，不要假设“即插即用”。

## 11. 给 AI 的硬性约束

1. 只改 `yunqian_vision`，参考仓库只读。
2. 规则变更未明确前，不实现、不删除、不“顺手优化”相关逻辑。
3. 改 `Target` / EKF 前，先说明状态量与观测量是否变化、雅可比是否需要同步修改。
4. 不提交密钥、标定私密数据、日志、`build/`。
5. 未经用户明确要求，不执行 `git commit` / `push`。
6. 每次给出改动时，说明：改了哪些文件、为什么、如何验证。
7. 单开某个机器人分支前，先确认该机器人的算力平台（见 §2.3）。三号步兵是 Orin NX；未确认的机器人不要默认沿用本仓库的 NUC / OpenVINO 路径，也不要默认搬 Orin NX 的 TensorRT 路径。

## 12. 变更记录

| 日期 | 规则依据 | 变更内容 | 影响模块 | 状态 |
|---|---|---|---|---|
| 2026-11-15 | 2027 规则手册 §5.2 | 自适应装甲机制（未加固 ×1.5、已加固 ×0.5；需区分未知/已知加固区） | `auto_aim/aimer`、`shooter`、`solver`、`target`、`io`、决策层 | 已录入，未实现 |
| 2026-11-15 | 算力平台分工（用户确认） | 本仓库 = NUC / OpenVINO；Orin NX 适配在 `27-wheel-leg-wbc` 的 `origin/vision`。三号步兵用 Orin NX。开机器人分支前必须先确认平台 | 分支策略、推理后端、部署 | 已录入，未开分支 |
| 2026-11-15 | 本地开发平台（用户确认） | 新增 `docs/local_dev_platform.md`：NUC 不作为本地开发前提；四项外设用替身替换；平台差异锁在 `YOLOBase` / `CameraBase` 之后；给定本机依赖安装与最小可跑闭环 | 文档、`io`、`tasks/auto_aim` 的构建路径 | 已录入，未改代码 |
| 2026-11-15 | 仿真同机闭环（用户确认走共享内存） | 新增 `io/sim/` 与 `src/sim_standard.cpp`：用 Talos 共享内存收仿真图像与位姿、回写云台指令，复用现有自瞄链路。不走 ROS2，NX 与 NUC 编同一份 | `io/sim`、`src/sim_standard.cpp`、`configs/sim.yaml` | `sim_standard` 已编译，共享内存回环 489 帧通过；待真实仿真确认瞄准方向 |
| 2026-11-15 | 用户报告（仿真验收） | 新增已知问题 KI-001：仿真“斜面上旋转的英雄”无法稳定重建。11 维 EKF 假设水平装甲环、两个半径、两个高度，换板时 `r/l/h` 振荡触发 `diverged()`，相位重置。硬约束：**四块板各自半径/高度不得预知或硬编码**；装甲板外形、90° 分布、15° 安装角按制作规范（S120/S121/S124/S125）作先验；不得读 GT，必须在线估计 | `auto_aim/target`、`tracker`、`sim_standard` / viewer | 已记录（见 [`docs/known_issues.md`](docs/known_issues.md)） |
| 2026-11-15 | KI-001 P1 | `Target` 状态扩到 13 维，加装甲环 roll/pitch；观测扩到 6 维，加入板面 pitch/roll；雅可比改数值差分。1 号英雄保持四块大装甲。锁住当前板、去掉板面 yaw 更新会破坏平地步兵，已回退 | `auto_aim/target`、`tracker`、`solver`、`tools/extended_kalman_filter` | 已实现；平地步兵旋转正常，斜面英雄仍因板面 yaw 双解失锁 |
| 2026-11-17 | KI-001 复测 | `Solver::resolve_yaw()` 在进 EKF 前用当前倾角重投影，从低代价局部极小里取离预测最近的 yaw。状态/观测/雅可比不变 | `auto_aim/solver`、`target`、`tracker` | 斜面英雄静止、匀速转不再持续失锁；匀速转相位每圈少记一块。平地步兵换板正常 |
| 2026-11-17 | KI-002（用户确认） | 前哨站三块板高度不同，`h_armor_xyz()` 的三块板分支却共用一个 `z`。一个 `h` 不够表示三档高度 | `auto_aim/target` | 已记录，未改模型 |
| 2026-11-17 | 制作规范 S120/S121/S124（用户确认） | 一台地面机器人两个半径够用：对角两块共用一个半径，横向偏移 ≤ 50 mm、下边缘高度差 ≤ 100 mm 由 `l`、`h` 表示。P3 的四块各自半径不做 | `auto_aim/target` | 已记录，不扩状态 |
| 2026-11-17 | KI-001 复测 | 转完停下后当前板贴住，另外三块预测板仍偏。第二块板进画面时 `w` 从 0 跳到约 −7.5 rad/s。`resolve_yaw()` 增加 0.5 rad 门限，静止段 `w` 回到 0 | `auto_aim/solver` | 静止段已止住；匀速转未复测 |
| 2026-11-20 | `sp_vision_26` 区域赛存档视频（`BV1wYLR6PEGh`）分析 | 纳入 4 项算法与评估改进计划：① 修复 `Aimer` 小陀螺分支下标 Bug（KI-003）并让仿真接入 MPC `Planner`；② 改进 `Planner` 迭代飞行时间、几何化开火窗口与云台实际跟踪误差约束、按惯量标定动力学约束；③ 弹道升级为含空气阻力迭代模型；④ 仿真闭环增加 `plan_yaw`/`gimbal_yaw` 曲线与虚拟发弹 `hit_rate` 统计（详见 `ROADMAP.md` §8） | `auto_aim/aimer`、`planner`、`tools/trajectory`、`src/sim_standard`、viewer | 已纳入计划，未实现 |
| 2026-11-20 | 视觉弹道闭环（用户确认移植 `origin/vision`） | 移植 `AimCorrector` 与 `RuntimeParams`，补充 `tools::ballistic_point`（二维耦合 RK4 阻力）与 `Solver` 像素射线/单点投影接口，打通 `Aimer` / `Planner` 的 `yaw_offset`/`pitch_offset` 运行期原子同步，并在 `sim_standard` 新增 `--corrector` 弹道闭环预览 | `tools/runtime_params.hpp`、`tools/trajectory.*`、`auto_aim/solver.*`、`aimer.*`、`planner.*`、`aim_corrector.*`、`src/sim_standard.cpp`、`tests/aim_corrector_test.cpp` | 已实现，`sim_standard` 与 `aim_corrector_test` 编译通过且单元测试通过 |
| 2026-11-20 | 三号步兵 Orin NX 实机迁移（用户确认） | 创建并切换至三号步兵分支 `standard3-orin`，移植 `origin/vision` 的 Orin 推理/相机适配层（MVS aarch64、HikRobot `frame_rate`/运行期调参、TensorRT + CUDA `sm_87` 预处理 + EfficientNMS），并同步部署至 `yq11@192.168.55.1:~/yunqian_vision` 完成编译与实测（USB3 相机 200 fps、TRT+CUDA Graph 就绪、`aim_corrector_test` 通过） | `io/CMakeLists.txt`、`io/camera.*`、`io/hikrobot/*`、`tasks/auto_aim/CMakeLists.txt`、`tasks/auto_aim/yolos/*`、`tasks/auto_buff/CMakeLists.txt`、`CMakeLists.txt`、`configs/standard3.yaml` | 已移植、部署并完成 Orin NX 编译与核心链路验证 |
| 2026-11-20 | 参考工程补充（用户确认） | 将 `WUST-RM/awakening`（`https://github.com/WUST-RM/awakening`）纳入权威只读参考工程列表；同步完成三号步兵串口 pitch 抬头正（`+30° ~ -20°`）边界换算、相机在线 ChArUco 内参标定与 EKF 竖直过程噪声/观测方差调优 | `AGENTS.md`、`io/gimbal/*`、`configs/standard3.yaml`、`tasks/auto_aim/target.*`、`tracker.cpp`、`src/standard_mpc.cpp` | 已录入并实机验证通过 |
| 2026-11-20 | 借鉴 `WUST-RM/awakening` 识别与预测（用户确认） | 状态量保持 13 维不变，观测量新增图像平面 `UVL`（左右灯条归一化重投影）+ 单板 IPPE 深度差 `DiffMeasure` 多观测联合更新与图像空间四点代价匹配，自适应多维 NIS 卡方阈值；新增整车状态驱动的动态聚焦 ROI（`Tracker::focus_roi` + `YOLO::detect(img, roi)`，丢帧自动扩回全图），并接入 `standard_mpc`、`camera_web_test`、`sim_standard` | `tools/extended_kalman_filter.cpp`、`tasks/auto_aim/solver.*`、`target.*`、`tracker.*`、`yolo.*`、`src/standard_mpc.cpp`、`src/sim_standard.cpp`、`tests/camera_web_test.cpp`、`tests/uvl_tracker_test.cpp`、`configs/{standard3,sim}.yaml` | 已实现，本机编译及 `uvl_tracker_test`/`aim_corrector_test` 全部通过（平地大高低差与斜面旋转 `med|w_err|≤0.05 rad/s`、`med_px_err≤0.5 px`） |
| 2026-11-21 | 三号步兵 Orin NX 实机联调与系统辨识（用户确认） | ① 同步最新代码至 Orin NX 并编过 `standard_mpc`、`camera_web_test`、`uvl_tracker_test`、`aim_corrector_test`（两项单测通过）；② 核对串口 `/dev/ttyACM0`（1000 Hz 零误码，电控已统一为抬头为正、左正右负）；③ 在 Yaw `[-30°, +30°]` 限位下完成 `real_gimbal_sysid` 实车辨识（Yaw `wn=35.54 rad/s, zeta=0.517, R²=0.9475`；Pitch `wn=47.36 rad/s, zeta=0.909, R²=0.9911`；延迟 `0.025 s`）；④ 实机 A/B 对比确认武科观测（`use_uvl_ekf`）失败主因为**标定参数不对**导致频繁发散（记入 KI-004，期望后续找到**自动标定方法**后再启用），三号步兵暂切回 `YPDA`（60 s 零发散）；⑤ 调优传递函数 MPC 权重 `Q_yaw_plant=[40.0, 0.08], R_yaw_plant=[2.5], max_yaw_cmd_offset=0.14`，小陀螺锯齿波 Yaw 跟踪真值 `med=0.92°`、`RMSE=2.04°`、换板收敛 `39 ms`，Pitch `med=0.26°` | `configs/standard3.yaml`、`AGENTS.md`、`docs/known_issues.md` | 已部署并完成实机 100 fps 闭环跟踪验证（待实弹测试与自动标定方案） |
| 2026-11-21 | 三号步兵武科 UVL 观测与动态聚焦 ROI 实机调通（用户确认） | ① 定位并解决 KI-004：发现下位机 IMU 存在 `+90°` 滚转安装角、零漂及与编码器脱节问题，在 `io/gimbal` 新增 `R_gimbal2imu`、`use_euler_q`、`camera_latency_ms`（`8.0 ms`）、`zero_vel_ff`；② 修复 UVL 首帧倾角初值、`DiffMeasure` 15° 先验消歧、灯条长度/角度噪声及平动速度 `P0/v1` 抑制云台耦合；③ 完成 4 种控制组合 5 s 实机对比（平滑 MPC 关前馈 `delay=0.060 s` 最稳：关 ROI Yaw `1.01°`/重投影 `4.70 px`、开 ROI Yaw `1.32°`/重投影 `5.33 px`；开速度/加速度前馈 `delay=0.020 s` Yaw `1.38°`；叠加传递函数 MPC 因 Yaw 欠阻尼 `zeta=0.517` 双重前馈过冲至 `17.0°`），当前保留 `gimbal_*_wn: 0.0` + `zero_vel_ff: false` + `delay=0.020 s`；④ 视觉专项诊断定位出 3 项待续测项：1~4 帧短脉冲丢检（`13.8%`，待测降置信度/关传统精修及修正 ROI 下 `ArmorSmoother` 坐标）、灯条长度比偏短 ~10%（`0.907`）、相机外参安装角（EKF 收敛到 `roll=-9.04°, pitch=-4.01°`） | `io/gimbal/gimbal.*`、`tasks/auto_aim/{target,tracker,solver}.cpp`、`src/standard_mpc.cpp`、`configs/standard3.yaml`、`docs/known_issues.md`、`AGENTS.md` | 已同步至 Orin NX 编译并完成实机 100 fps 闭环验证；小电脑已关机，视觉端 3 项优化待下次上电续测（未 commit） |
| 2026-11-22 | 仿真高帧率、三号步兵真实云台动力学模型与开火窗口修复（用户确认） | ① 编译 release 版 `daedalus`（`.rust/target/release/daedalus`）+ P 核 `taskset` 隔离 + 原生 Wayland `immediate` 模式（仿真帧率由 19–31 FPS 提至 80–110 FPS）；新增 `tools/opt_sim_pid.py` 离线代价函数优化器；② 拉取 `27-wheel-leg-wbc-gimbal` 至 `7991a95`，在本仓库新增 `io/sim/sim_gimbal_model.*` 复现三号步兵 500 Hz LQI + 50 Hz 陷波器 + 库仑/粘性摩擦前馈 + 6 ms 力矩环滞后 + DM4310 转速-力矩包络（Yaw `J=0.0146, K=[24,0.96,0,120], tau_lim=6.0 Nm`；Pitch `J=0.0115, K=[20,0.82,0,85], tau_lim=4.5 Nm`），接入 `SimGimbal` 与 `configs/sim.yaml`；③ 导出电控离线调参包 `records/ec_gimbal_tuning_pack.zip`（含坐标/协议 `README.md` 与 100 Hz 目标曲线 CSV）；④ 修复 `Planner` 开火判据：将受击装甲回退 `active_delay_time_` 至 `now + fly_time` 评估真实入射角与几何窗口、扣除 17 mm 弹丸半径 `8.5 mm`、补齐 Pitch 跟踪误差判据、`plan_ready` 对齐至 `now_idx + shoot_offset_`，`fire_margin` 收紧至 `0.70` | `io/sim/sim_gimbal_model.*`、`io/sim/sim_io.*`、`src/sim_standard.cpp`、`tasks/auto_aim/planner/planner.cpp`、`tools/opt_sim_pid.py`、`CMakeLists.txt`、`configs/sim.yaml`、`records/ec_gimbal_tuning_pack.zip`、`AGENTS.md` | 已编译并通过仿真闭环验证（小陀螺 1.78 m 处 `hit_rate` 由 `93.5%` 提至 `98.6%` (`71/72`)，约 11 发/s；未 commit） |
| 2026-11-23 | 仿真 100 Hz 实时性优化与本机 RTX 4060 TensorRT 加速（用户确认） | ① `SimCamera` 拆出独立后台抓帧 + `SimCameraModel` 退化流水线，与主线程 YOLO/EKF 并行；共享内存轮询缩至 `150 μs`，`ready_slot_` 零拷贝 `cv::swap`，绝对周期网格 + 末段短自旋稳锁 `100.0 Hz`；优化 `apply_motion_blur` 连续字节段与暗角移位（1440×1080 全退化单测 `3.96 ms`）；② 本机 x86_64 接入 TensorRT 11 + CUDA 12.8（`trt_engine.hpp` 改用 `cuda_runtime_api.h`），为 RTX 4060 编译 `assets/yolov5_rtx4060.engine`（`.gitignore` 忽略 `*.engine`），`yolov5` 推理 + CUDA Graph 由 CPU `8.0 ms` 降至 `3.8–4.0 ms`；③ `sim_standard` 新增自动 P/E 核三级隔离（`TRT-GPU` 模式分配 2 P 核给主线程、其余 P 核给 `SimCameraModel`、独占 1 核给 100 Hz MPC 规划）+ 混合定时（`sleep_until - 500μs` + 短自旋）+ `--profile` 分项耗时探针；④ `.sim/run/config.toml` 开启极限实时（`preview.enabled = false`、`substep_count = 4`，画面由 `WebViewer` 查看） | `io/sim/sim_io.*`、`io/sim/sim_camera_model.cpp`、`src/sim_standard.cpp`、`tasks/auto_aim/yolos/trt_engine.hpp`、`configs/sim.yaml`、`.sim/run/config.toml`、`.gitignore`、`AGENTS.md` | 已编译并通过全闭环实测：`daedalus` 375–480 FPS，视觉稳锁 100 FPS（`vis_dt=10.00ms, sd=0.72–0.86ms`），控制器 100 Hz（`plan_dt=10.00ms, sd=0.00–0.02ms`）；`sim_camera_model_test` 通过（未 commit） |
| 2026-11-23 | 跑打（底盘速度 + roll 融合）、3D 开火窗口重构与 WebViewer 双命中率评估（用户确认） | ① `tools/trajectory.*` 新增 3D 跑打弹道 `Trajectory3D` / `ballistic_point_3d`（叠加 `v_self` + 二次空气阻力，分子有理化消除近距数值不稳定）；`Target::predict` / `focus_roi` 与 `Tracker` 支持 `self_vel` 解耦预测；② `Planner` 新增 `SelfMotion`（`vel/roll/pitch/muzzle_in_gimbal`）融合 `roll` 引起的 3D 炮口平移与零偏旋转，并将开火判据重构为 `check_ballistic_hit`（3D 弹道微段与所有正面装甲板求交 + 机身圆柱遮挡 + 3D 入射角，修复 `get_trajectory(Target&)` 错移 `hit_now` `+0.5 s` Bug）；③ `io/sim/talos_shm.*` 映射 `chassis_observation`（offset `1856`）、`ground_truth`（offset `1984`，仅评估用）与 `runtime_state.following`（offset `3648`），修复 `peek_gimbal_quat` / `peek_odom_pos` 跳过 `write_idx` 且按单调 `frame_seq` 读取以彻底消除无锁脏读与偶发倒退尖峰；`io/gimbal` 补充 `roll` 与底盘速度预留字段；④ WebViewer 新增**重建命中**与**仿真实际**双命中率 KPI/曲线及**清除弹丸统计**按钮（`/api/metrics/reset`）；调优 `.sim/run/config.toml`（`kp=45`、`fixed_hz=240, substep_count=2`）与 `configs/sim.yaml`（`Q_yaw=[2e7,0]`、`max_yaw_acc=180`、`delay=0.045s`、`fire_margin=0.85`、`max_fire_angle=55`）兼顾平滑、高开火占空比与零相位 | `tools/trajectory.*`、`tools/sim_metrics.*`、`tools/web_viewer.*`、`tasks/auto_aim/{target,tracker}.*`、`tasks/auto_aim/planner/planner.*`、`io/sim/{talos_shm,sim_io}.*`、`io/gimbal/gimbal.*`、`src/{sim_standard,standard_mpc}.cpp`、`web/index.html`、`configs/{sim,standard3}.yaml`、`.sim/run/config.toml`、`tests/shoot_on_move_test.cpp`、`CMakeLists.txt`、`docs/{real_robot_tuning_guide,gimbal_serial_protocol}.md`、`ROADMAP.md`、`AGENTS.md` | 已编译并通过全套单测（`shoot_on_move_test` 5 项全 100% 命中 / `fire_duty` 81.6%–87.3%、`aim_corrector_test`、`uvl_tracker_test`）；仿真小陀螺实测 `recon_hit=99.2%` / `sim_hit=96.2%` / `fire_duty=73.7%+` / `yaw_rms=1.33°`（未 commit） |
| 2026-11-24 | 常用启动与部署脚本（用户请求） | 在 `scripts/` 新增一套通用启动/编译/测试/部署脚本：`scripts/_env.sh`（自动配置 `.deps/`、OpenVINO/TBB、MVS、Rust 动态库路径）、`run_sim.sh`（一键启动 `daedalus` + `sim_standard` 并统一清理退出）、`run_standard.sh`（默认 `--no-fire --web` 安全调试，`--live` 实弹）、`run_camera_web.sh`（相机预览与抓拍）、`run_tests.sh`（4 项离线单测汇总）、`build.sh`（自动识别本地 toolchain 或系统依赖）、`deploy_orin.sh`（`rsync` 同步至 Orin NX 并可选远程编译）及 `scripts/README.md`；同步更新文档入口 | `scripts/*`、`AGENTS.md`、`docs/real_robot_tuning_guide.md` | 已通过 `bash -n`、`--help` 与 `./scripts/run_tests.sh`（4/4 全绿）验证（未 commit） |
