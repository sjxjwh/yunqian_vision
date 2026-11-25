# yunqian_vision — AI 协作说明

> 面向 **RoboMaster 2027 赛季** 的视觉/自瞄代码库（基于 `sp_vision_25` @ `bd9f5e7`）。
> 目标：**按 2027 规则做最小、可控、可验证的改动**，保持高信噪比与上游可对比性。
> 最后整理：2026-11-25。标注“待确认”的内容不要擅自实现。

## 1. 速查与文档导航

| 项 | 内容 |
|---|---|
| 算力平台分工 | 本仓库 = **NUC**（x86_64，OpenVINO）；**Orin NX**（aarch64，TensorRT + CUDA `sm_87`）参考 `27-wheel-leg-wbc` 的 `origin/vision` |
| 首台调试机器人 | **三号步兵 = Orin NX**（分支 `standard3-orin`，目标机 `yq11@192.168.55.1`）；其余兵种开分支前必须先确认平台 |
| 视觉/电控边界 | 视觉仅输出平滑可达参考轨迹 `(pos, vel, acc)`；云台阻尼与动力学前馈下沉电控 1 kHz 闭环，不在视觉叠传递函数逆模型 |
| 硬约束 | 只改本仓库；规则未确认不实现；改 EKF 前先说明状态/观测/雅可比影响；不擅自 `git commit` / `push` |

**延伸文档（按需查阅，不要把细节堆回本文件）：**
- 完整历史变更流水账：[`docs/CHANGELOG.md`](docs/CHANGELOG.md)（后续详细变更统一记在这里）
- 2027 规则 §5.2、通信协议与经济/火控推导：[`docs/rules_2027_notes.md`](docs/rules_2027_notes.md)
- 已知问题与演进设计（KI-001~KI-004）：[`docs/known_issues.md`](docs/known_issues.md)
- 短期实施路线图：[`ROADMAP.md`](ROADMAP.md)
- 串口协议与扩展草案：[`docs/gimbal_serial_protocol.md`](docs/gimbal_serial_protocol.md)
- 本地开发 / 实机调参 / 脚本速查：[`docs/local_dev_platform.md`](docs/local_dev_platform.md)、[`docs/real_robot_tuning_guide.md`](docs/real_robot_tuning_guide.md)、[`scripts/README.md`](scripts/README.md)

## 2. 只读参考仓库与规则 PDF（绝不修改）

| 用途 | 路径 | 基线 |
|---|---|---|
| 上游视觉基线 | `/home/scj/桌面/rm2027/opensource/sp_vision_25` | `bd9f5e7` |
| Orin NX 适配参考 | `/home/scj/桌面/rm2027/27-wheel-leg-wbc` (`origin/vision`) | `93cb1b1` |
| 仿真参考 | `/home/scj/桌面/rm2027/opensource/bevy_robomaster_simulator` | `07af246` |
| 自瞄算法参考 | `/home/scj/桌面/rm2027/opensource/awakening` (`WUST-RM/awakening`) | `7a0c4ba` |

**2027 规则与协议 PDF**（可用 `pdftotext -layout` 检索）：
- `/home/scj/桌面/rm2027/rule/UTF-8__RoboMaster 2027 机甲大师超级对抗赛比赛规则手册V1.0.0（20261015）.pdf`
- `/home/scj/桌面/rm2027/rule/UTF-8__RoboMaster 2027 机甲大师高校系列赛通信协议 V1.0.0（20261029）.pdf`
- `/home/scj/桌面/rm2027/rule/UTF-8__RoboMaster 2027 机甲大师高校系列赛机器人制作规范手册V1.0.0（20261015）.pdf`

## 3. 算力平台与分支规则

1. **开机器人分支前先确认算力平台**（NUC 还是 Orin NX）。
2. 业务代码（识别/解算/EKF/规划/火控）保持共享，平台差异锁在 `YOLOBase` / `CameraBase` 与 `SP_WITH_TENSORRT` / `SP_WITH_CUDA_PREPROCESS` 编译开关之后。
3. NUC 分支不引入 TensorRT/CUDA 必需依赖；Orin NX 的 `.engine` 与本机绑定不可跨机复用；不把 `27-wheel-leg-wbc` 电控分支混入视觉分支。

## 4. 核心数据流与 13 维 EKF

```
相机 / 仿真 / SyncPlayer ──> YOLO (动态聚焦 ROI) ──> Solver (PnP + 重投影)
  ──> Tracker & Target (13 维 EKF，支持 UVL+Diff 图像平面观测 / YPDA)
  ──> Planner (TinyMPC 轨迹 + 3D 跑打弹道 + 三层概率火控 + 防吞弹射频)
  ──> io/gimbal (串口) / io/sim (Talos 共享内存)
```

**13 维 EKF 状态 `x`（`tasks/auto_aim/target.cpp`）**：
- `0..5`: 旋转中心 `[x, vx, y, vy, z, vz]`；`6..7`: 参考板 `[a, w]`；`8`: 半径 `r1`；`11..12`: 装甲环倾角 `[roll, pitch]`。
- `9..10`: 四板/二板为 `[l = r2 - r1, h = z2 - z1]`；**三板前哨站（`armor_num_ == 3`）复用为板 1、板 2 相对板 0 的在线高度差 `[h1, h2]`**。
- 观测雅可比基于 `h_armor_xyz()` 数值差分，修改观测或状态前务必同步核对 `diverged()`、`P0` 与 `Q`。

## 5. 2027 规则关键事实与火控架构

- **装甲板形态变更**：2027 取消灯条、更换装甲板图案；不再投入旧灯条传统精修或旧灯条四点模型，待新装甲板定型后统一走纯神经网络关键点 + 同步落地自动外参/延迟标定（方案 1A）。
- **§5.2 自适应装甲**：未加固 ×1.5、已加固 ×0.5、默认 ×1；敌方加固圆仅发给己方雷达（`0x0A05`），需雷达经 `0x0301`（`0x0200~0x02FF`）→ 下位机转发上位机；视觉分未知模式（瞄中心）与已知模式（圆外选点，待实现）。详细协议与经济/经验推导见 [`docs/rules_2027_notes.md`](docs/rules_2027_notes.md)。
- **三层概率火控（已实现，纯 CPU `< 1 μs`）**：Layer 1 3D 弹道求交 + 机身圆柱遮挡 → Layer 2 物理不确定度 `P_phys`（EKF 协方差 + `AimCorrector` 实测散布 EMA 同步）→ Layer 3 轻量残差 MLP（`14→32→16→1`）→ 单旋钮动态阈值（近距爆发 `0.70` → 远距 Peek `0.82` + 等级/爆发键/加固区/高热量偏置）+ 单板防吞弹建议射频 `plan.fire_rate_hz`。

## 6. 常用构建与验证命令

```bash
./scripts/build.sh sim             # 编译 sim_standard + replay_standard + 全部离线单测
./scripts/build.sh real            # 编译 standard_mpc / replay_standard / camera_web_test / real_gimbal_sysid
./scripts/run_tests.sh --build     # 编译并运行全部 5 项离线单测（改动 C++ 后必跑）
./scripts/run_sim.sh               # 启动 daedalus 仿真 + sim_standard（WebViewer :8090）
./scripts/run_standard.sh          # 实机安全调试（默认 --no-fire --web；--sync-record=auto 可同步录包）
./scripts/deploy_orin.sh --build   # 同步代码到三号步兵 Orin NX 并远程编译
```

## 7. 编码约定与硬性约束

- **风格**：C++17、CMake、`.clang-format`（LLVM 2 空格）；沿用现有命名空间与目录结构，不无故引入新第三方库，非必要不重构。
- **硬性约束**：
  1. 只改 `yunqian_vision`，参考仓库只读。
  2. 规则变更未明确前，不实现、不删除、不“顺手优化”。
  3. 改 `Target` / EKF 前，先说明状态量与观测量是否变化、雅可比是否需同步修改。
  4. 不提交密钥、标定私密数据、日志、`build/`、`*.engine`。
  5. 未经用户明确要求，不执行 `git commit` / `push`。
  6. 每次改动后说明：改了哪些文件、为什么、如何验证；详细流水账记入 [`docs/CHANGELOG.md`](docs/CHANGELOG.md)，**不要往本文件追加长篇日志**。
