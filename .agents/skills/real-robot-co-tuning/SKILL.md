---
name: real-robot-co-tuning
description: 单人视觉-电控实车自瞄联合自动调试工作流。适用于开发机同时连接云台 MCU 下载器（通过 EmberProbe MCP）并可 SSH 访问视觉小电脑（Orin NX / NUC）的场景，涵盖伺服电机强安全红线、观测与控制解耦排障（IDLE vs AUTO_AIM、IMU vs 编码器、camera_latency_ms 寻优）以及平滑跟随曲线精调。
---

# 视觉-电控单人实车联调 Skill（Real-Robot Co-Tuning）

本 Skill 用于指导 AI 在**单人联调**环境下，同时操作**视觉仓库 + 视觉小电脑（SSH）**与**指定的电控仓库 + 云台 MCU 下载器（EmberProbe MCP）**，完成从链路核验、观测排障到平滑跟随曲线精调的自动化闭环工作流。

---

## 1. 强安全红线（伺服电机防甩飞协议 — 最高优先级）

> **核心风险**：RoboMaster 云台常用伺服电机（如 DM 达妙、GM6020 等）在**单片机突然停止、复位、崩溃或被 SWD 烧录/断点暂停（halt）**时，若单片机停止前最后一帧仍下发着非零力矩/电流/速度，电机会在通信超时保护生效前**保持上一刻下发的值继续发力**，极易造成云台猛甩、撞碎限位、扯断线束或伤人！

执行任何可能暂停/复位 MCU 的操作（`flash_program`、`debug` 断点/暂停、改控制代码重启）前，**必须严格按顺序执行以下安全门控**：

1. **停视觉控制输出**：先在小电脑停掉自瞄进程（`pkill -x standard_mpc 2>/dev/null || true`），切断下行指令源。
2. **确认电机已无力/零力矩**：
   - 提示操作员用遥控器切出 `AUTO_AIM`，切到 **无力/失能（`ZERO_FORCE` / 拨杆关机）** 状态；
   - **强制校验**：在调用 `flash_program` 或 `debug` halt 前，必须用 `emberprobe.variables_read` 读取云台模式与输出力矩（如 `gimbal_cmd_recv.gimbal_mode`、`gimbal_lqr_output.YawTorqueNm`、`gimbal_lqr_output.PitchTorqueNm`、`yaw_speed_radps`），确认 **力矩为 `0` 且角速度接近 `0`** 后，方可烧录或暂停。
   - **绝对禁止**在 `AUTO_AIM` 跟踪运动中、或力矩非零时直接热烧录/断点暂停 MCU！
3. **架台单人联调软保护**：
   - 线束未整理或架台测试时，确认电控启用临时机械软限位（如 `GIMBAL_YAW_SOFT_LIMIT_ENABLE = 1` 限制开机原点 `±30°`、Pitch `[-20°, +30°]`）与底盘零力锁死（`CMD_FORCE_CHASSIS_ZERO_FORCE = 1`）；
   - 上位机始终以安全模式运行：`--no-fire --web`（禁止自动发弹）。
4. **高频振动/异响急停**：
   - 一旦操作员报告异响/抖动，或采样发现 `>30 Hz` 高频满力矩翻转（如 50 Hz 扭振自激），**立即停视觉 + 让操作员切无力**，回退最近一次高频增益改动（尤其慎提高下位机 `K_vel`）。
5. **烧录后重连注意**：
   - 烧录复位会重新锁存上电零位，并导致 USB-CDC `/dev/ttyACM0` 短暂重枚举；等待 1.5–2 s 确认 `/dev/ttyACM0` 重新出现后再启动上位机。

---

## 2. 硬件与 AI 环境要求（本机双端架构）

单人自动联调要求**同一台开发机**同时具备对视觉端和电控端的控制权：

```text
                ┌── USB/Type-C (SSH 192.168.55.1) ──▶ 小电脑 (Orin NX / NUC, 视觉程序 + WebViewer :8090)
开发机 (AI) ────┤                                            ▲
                │                                            │ USB-CDC (/dev/ttyACM0, 1000 Hz)
                └── SWD 下载器 (CMSIS-DAP / ST-Link) ──▶ 云台 MCU (STM32F407, 500 Hz LQR)
                     ▲ 经 VS Code EmberProbe Agent Bridge / MCP
```

### 2.1 AI 访问电控下载器的环境配置（EmberProbe）
1. **VS Code 侧**：安装 **EmberProbe** 扩展，在 VS Code 中打开**指定的电控代码仓库**（如 `/home/scj/桌面/rm2027/27-wheel-leg-wbc-gimbal`），确认侧边栏已识别探头（probe、serial）、芯片 target（如 `stm32f4x.cfg`）与编译产物 ELF（如 `build/basic_framework.elf`），并生成 `.agents/skills/_emberprobe/agent-bridge.json`。
2. **AI 会话目录路由**：
   - EmberProbe MCP 按当前会话工作目录匹配电控仓库的 Agent Bridge；
   - 若 AI 从视觉仓库启动，可先用 `opencode.session.move` 将会话目录切到**指定的电控仓库**以激活 EmberProbe MCP，同时用绝对路径读写视觉仓库（如 `/home/scj/桌面/rm2027/yunqian_vision`）。
3. **EmberProbe 使用约束**：
   - 读写变量、采样波形、烧录固件一律通过 `tools.emberprobe.*`（`variables_read`、`variables_sample`、`flash_program` 等），**禁止绕过 EmberProbe 自行调用裸 `openocd` / `gdb`**；
   - 探头单所有者原则：烧录前必须停止一切采样或调试会话；
   - 绝不猜测 `elf`、`target`、`probe` 或 `probeSerial`，缺失时向用户确认或先查询 `config_get`。

### 2.2 小电脑 SSH 与编译注意
- 免密 SSH 访问小电脑（默认如 `yq11@192.168.55.1:~/yunqian_vision`）；
- **时钟偏移陷阱**：若小电脑离线无 NTP（系统时间落后于开发机），`scp` 过去的新文件会触发 `make` 的 `Clock skew detected` 并可能跳过重新链接——远程重编可执行文件时先 `rm -f build/<target>` 再 `make -C build -j8 <target>/fast`。
- 采集实时波形优先通过 SSH 在小电脑本机连 `ws://127.0.0.1:8090/ws`（避免开发机侧防火墙或 USB 网卡抖动超时），且小电脑通常自带 `numpy`/`scipy`。

---

## 3. 联调开始前的信息确认

每次启动本工作流时，先向用户核对或从环境读取以下信息（已知则直接复用，不重复追问）：
1. **电控代码仓库路径**（如 `/home/scj/桌面/rm2027/27-wheel-leg-wbc-gimbal`）与探头信息；
2. **视觉配置文件与小电脑地址**（如 `configs/standard3.yaml`、`yq11@192.168.55.1`）；
3. **当前机械约束**：是否有临时软限位要求、底盘是否需锁死零力、IMU 安装朝向是否变动过；
4. **靶标情况**：敌方颜色（红/蓝）、静止还是旋转小陀螺。

---

## 4. 自动化单人联调五阶段工作流

### 阶段 1：基础链路与安全边界核验
1. 核对电控临时安全宏（软限位 + 底盘零力）与上位机 `--no-fire --web`。
2. 核验小电脑相机与推理帧率稳定（如 `100 fps`）、能检出目标装甲板。
3. 核验串口上行（1000 Hz、0 CRC 错）与下行模式切换（`IDLE` ↔ `AUTO_AIM`）。
4. 若为新结构/新云台，先在安全幅值内运行端到端系统辨识（`real_gimbal_sysid` + `fit_gimbal_tf.py`），记录 `wn / zeta / tau_d / a_max`。
   - **原则**：若下位机已含速度/加速度前馈，上位机 `gimbal_yaw_wn` / `gimbal_pitch_wn` 保持 `0.0`，避免植物逆 MPC 与下位机前馈叠加放大过冲。

### 阶段 2：观测层解耦排障（先静止，再运动）

> **黄金法则**：世界系目标角在理论上应与云台运动完全无关。**先确认云台静止时观测完美，再查动云台时的耦合源**，绝不在观测发歪的情况下硬调控制参数。

1. **开启 `IDLE` 识别预测预览**：确保 `standard_mpc` 在 `IDLE` 模式下持续跑检测、EKF 与规划（仅在 `gimbal.send` 拦截为 `control=false`），使操作员能在云台不动时直接看 WebViewer 锯齿曲线。
2. **`IDLE` 静止基准采集（4 s）**：采集 WebViewer `/ws`，验收纯视觉观测基准：
   - 单板 `ref_yaw` 锯齿线性度 RMS `< 0.25°`；
   - 丢检率 `det=0` `< 10%`（偏高时适当降低 `exposure_ms` 减运动模糊，如 `4.0 → 3.0`，并适当降低传统精修 `threshold`，如 `150 → 110`）；
   - EKF `nis < 2`、`r` 稳定、零发散。
3. **可选：无标定板自动慢摆标定（排除内参/手眼外参）**：
   - 停转靶标，控制云台做 10 s 低速正弦扫动（Yaw `±9° @ 0.25 Hz`、Pitch `±5° @ 0.35 Hz`），拟合像素角与云台角线性斜率：
   - 正常应满足 `d(cam_yaw)/d(gimbal_yaw) ≈ -1.000 ± 0.01`、`d(cam_pit)/d(gimbal_pit) ≈ -1.000 ± 0.01`、`cam_roll < 1°`。
4. **`AUTO_AIM` 运动对照（同步采双端 4 s）**：
   - 并行执行 `emberprobe.variables_sample`（采 MCU 编码器角、IMU 世界角、力矩）与小电脑 `/ws` 采集。
   - **检查 A — 世界系一致性（编码器 vs IMU）**：
     - 对比 `IMU_Yaw - Encoder_Yaw`：若换板加减速时两者差值随力矩跳变（例如底盘零力反作用晃动导致 `RMS > 0.3°`），则视觉必须使用 **IMU 世界系 Yaw** 而非电机编码器！
     - 切回 IMU 前，**必测静止 `INS.Accel[0..2]`** 确认哪一轴朝上（`≈ +9.8 m/s²`），避免 IMU 侧装/旋转导致角速度轴选错（例如 `Accel[1] ≈ +9.8` 时，世界 Z-up 航向角速度是 `INS.Gyro[1]` 而非 `Gyro[2]`）。
     - 机械软限位仍保留电机编码器，与视觉世界系解耦。
   - **检查 B — 相机时间戳对齐（`camera_latency_ms` 寻优）**：
     - 扫描 `L = 0..15 ms`，计算重建世界装甲角 `gimbal_yaw(t - L) + cam_du(t)` 的二阶差分 RMS，取最小值对应的 `L` 填入 `camera_latency_ms`（实测海康 3 ms 曝光 + USB-CDC 典型值约 **`4.0 ~ 4.5 ms`**）。
     - 警告：`camera_latency_ms` 偏差超过 `2 ms` 会在高速换板时直接拉爆 EKF `nis` 触发 `Target diverged`。

### 阶段 3：定时与执行器内环精调
1. **消除规划掉拍台阶**：确认上位机规划线程使用 `std::this_thread::sleep_until(next_tick)` 严格 100 Hz 周期，验收 WebViewer 中连续重复指令帧 `dup_cmd_frames == 0%`。
2. **压制换板后低频振铃（5–7 Hz）**：
   - 观察 10 ms 逐帧 `gimbal - cmd` 波形：若换板后在指令附近来回弹 2–3 次，优先**降低下位机 LQR 位置积分增益 `K_i`**（如 `120 → 35`）以释放 5–6 Hz 相位裕度；
   - **慎动高频速度增益 `K_vel`**：提高 `K_vel` 易激发机械柔性扭振（50–60 Hz 异响），若需修改必须小步试探并实时监测 10 ms 力矩波形。

### 阶段 4：平滑跟随曲线精调（对齐同济 S 曲线）
参考优秀实车跟随曲线的特征——**加速度呈圆润钟形 S 曲线（不顶死限幅），云台速度/位置 1:1 贴死规划轨迹，线性段零滞后紧贴目标角**：
1. **压柔规划加速度与权重**：在视觉 `configs/*.yaml` 中适当收敛 `max_yaw_acc`（如 `45.0 rad/s²`）与 `Q_yaw`（如 `[2.0e6, 0]`），消除换板尖峰与回弹；
2. **线性段零相位对齐**：根据实测 `gimbal` 相对 `ref` 的互相关滞后，微调 `high_speed_delay_time / low_speed_delay_time`（例如执行器滞后已压到 `<10 ms` 时，回调至 `0.002 ~ 0.005 s`），使端到端相位滞后收敛至 `0.0 ms`。

### 阶段 5：定量验收与收尾
每次调参后必须用同一套 4 s `/ws` 采集脚本做定量对比，达到以下参考线即可收工：

| 指标 | 合格参考线 | 优秀（同济级）实测基准 |
|---|---|---|
| 单板 `ref_yaw` 锯齿线性度 RMS | `< 0.35°` | `0.22°` |
| 执行器跟踪误差 `RMS(gimbal - cmd)` | `< 1.20°` | `0.69°`（`r > 0.97`） |
| 线性段（单板中段 60%）`RMS(gimbal - ref)` | `< 0.80°` | `0.50°`（零偏 `< 0.2°`） |
| 全周期中位误差 `median\|gimbal - ref\|` | `< 0.90°` | `0.54°` |
| 端到端相位滞后（`gimbal` vs `ref`） | `±10 ms` 内 | `0.0 ms` |
| 换板后 0~120 ms 最大偏差 `max\|gimbal - cmd\|` | `< 2.0°` | `1.33°` |
| Pitch 跟踪误差 `RMS(gimbal - ref)` | `< 0.60°` | `0.42°` |
| EKF 稳定性 | 0 次 `Target diverged` | `r` 稳定、`nis < 2` |

**收尾提醒**：
- 汇总电控与视觉两端的最终参数改动清单（不擅自 `git commit` / `push`）；
- 明确提醒用户：上线/整车跑动前，记得把临时软限位宏与底盘零力锁死宏改回 `0`。
