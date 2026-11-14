# yunqian_vision — AI 协作说明

> 本文件写给在本仓库工作的 AI（以及新加入的队员）。开始任何改动前先读完本文件。

## 1. 项目定位

`yunqian_vision` 是面向 **RoboMaster 2027 赛季** 的视觉/自瞄代码库。

- 初始代码基线：直接复制自 2026 赛季已验证的 `sp_vision_25`。
- 2027 赛季比赛规则发生变化，因此需要在基线上做针对性变更。
- **规则变更内容尚未全部录入本文件。用户会在后续对话中逐条给出，禁止自行臆测或提前实现未确认的规则。**

本项目的目标不是重写，而是：**在稳定基线之上，按 2027 规则做最小、可控、可验证的改动。**

## 2. 权威参考（只读，不在其中做修改）

| 用途 | 路径 | 基线 commit |
|---|---|---|
| 代码基线 / 上游参考 | `/home/scj/桌面/rm2027/opensource/sp_vision_25` | `bd9f5e7` |
| 仿真、ROS2 接口、闭环验证参考 | `/home/scj/桌面/rm2027/opensource/bevy_robomaster_simulator` | `07af246` |

这两个开源项目在 2026 年取得了显著成果，是**权威参考**：

- `sp_vision_25`：识别 → 解算 → 整车 EKF 估计 → 瞄准/开火决策 → 下位机通信的完整实现。
- `bevy_robomaster_simulator`：Bevy 物理/渲染仿真，发布 `/image_raw`、`/camera_info`、`/tf`，订阅 `/armor_solver/cmd_gimbal`，可用于自瞄闭环验证。

规则文档位于：

```
/home/scj/桌面/rm2027/rule/
├── UTF-8__RoboMaster 2027 机甲大师超级对抗赛比赛规则手册V1.0.0（20261015）.pdf
├── UTF-8__RoboMaster 2027 机甲大师高校联盟赛比赛规则手册 V1.0.0（20261015）.pdf
└── UTF-8__RoboMaster 2027 机甲大师高校系列赛机器人制作规范手册V1.0.0（20261015）.pdf
```

**只修改 `yunqian_vision`。绝不修改上述两个参考仓库。**

## 3. 变更工作流（重要）

1. 用户会逐条给出 2027 规则变更需求。
2. 收到需求后，先在 `## 8. 2027 规则变更记录` 中登记（日期、规则依据、影响模块、状态）。
3. 只实现已明确确认的变更；需求模糊时先提问，不要猜。
4. 每次改动后尽量编译验证；无法编译时明确说明原因。
5. 改动尽量局部化，保持与上游 `sp_vision_25` 的可对比性。

## 4. 目录结构与模块职责

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

## 5. 核心数据流

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

## 6. 目标观测与状态变量（改 EKF 必读）

状态向量 `x`（11 维，见 `tasks/auto_aim/target.cpp`）：

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

当前观测量 `z`（4 维，`Target::update_ypda`）：

```
z = [yaw, pitch, distance, armor_yaw]
```

- 前三维：装甲板中心世界系球坐标
- 第四维：装甲板自身 yaw

EKF 是通用实现（`tools/extended_kalman_filter.*`），`update(z, H, R, h, z_subtract)` 接受任意观测维度和非线性观测函数。若要把观测系改为 **uv 3dof（图像平面 u、v + 角度）**，只需替换 `Target` 中的观测定义 `h` 与雅可比 `H`，状态量无需改变。

发散/收敛判断：

- `Target::diverged()`：半径是否合理
- `Target::convergened()`：更新次数 + 半径
- EKF NIS 卡方检验：连续失败判失锁

## 7. 构建与验证

依赖（与上游一致）：OpenCV、Eigen3、fmt、spdlog、yaml-cpp、nlohmann_json、OpenVINO、Ceres（能量机关）。详见 `CMakeLists.txt` 与 `readme.md`。

```bash
cmake -B build
make -C build -j$(nproc)
```

常用验证程序（编译后位于 `build/`）：

```bash
./build/auto_aim_test
./build/auto_buff_test
./build/detector_video_test
./build/minimum_vision_system
```

本项目没有统一的 `ctest` 流程；测试以独立可执行程序为主。改动 C++ 后至少确认相关 target 能编译通过。

## 8. 2027 规则变更记录

> 由用户逐条提供后填写；未确认的条目保持“待确认”。

| 日期 | 规则依据 | 变更内容 | 影响模块 | 状态 |
|---|---|---|---|---|
| — | — | 待用户提供 | — | 待确认 |

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
