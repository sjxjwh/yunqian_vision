# 实机与仿真自瞄调试手册（100 Hz 实时闭环 → 辨识 → 跟随调参）

> 基于三号步兵（Orin NX）实机联调与 100 Hz 仿真闭环（`SimCameraModel` + `SimGimbalModel` 500 Hz LQI）的完整调试经验整理。
> 适用于新机器人上电首调、换电控参数后重调，以及 Sim2Real 参数迁移。

---

## 0. 调试总顺序（务必按序执行）

```
① 稳帧率与实时性 (100 FPS / 100 Hz)
        │
        v
② 硬件、坐标系、时戳与标定排雷
        │
        v
③ 端到端云台系统辨识 (SysID)
        │
        v
④ MPC 一般跟随调参（零滞后 + 零过冲）
        │
        v
⑤ 开火窗口与实弹弹道闭环 (AimCorrector)
```

> **核心原则**：帧率和延迟不稳时，绝对不要硬调 `delay_time` 或 EKF 噪声。帧率一变、延迟一飘，之前调好的参数会全部失效。

### 0.1 实机上电安全与启动检查

1. **物理安全**：首次上电、改完符号/限位或执行系统辨识前，**关闭摩擦轮与拨弹轮电源**，清空云台旋转干涉物，确保随时可切回手动档或 `Ctrl+C` 急停。
2. **安全启动参数**：实机联调跟随阶段推荐直接运行 `./scripts/run_standard.sh`（默认带 `--no-fire --web` 启动 `standard_mpc`，强制 `fire=false` 并开启 WebViewer 观察曲线与 3D 重建；实弹测试再显式加 `--live`）；仅在排查下位机模式位时才临时加 `--force-aim`，确认后立即去掉，避免上电直接甩头。相机单独预览与无损标定抓拍可直接运行 `./scripts/run_camera_web.sh`。
3. **算力板性能模式**：Orin NX 调试前确认已开启满血功耗模式（`nvpmodel` / `jetson_clocks`）且散热正常，避免运行几分钟后降频导致推理延迟尖峰。

---

## 1. 第一步：把视觉与控制器稳在 100 Hz

### 1.1 目标指标

| 环节 | 目标均值 | 目标抖动 / 上限 | 说明 |
|---|---|---|---|
| 相机采集 | `100 FPS` (`10.0 ms`) | `sd < 1.0 ms` | 实机在 YAML 设 `frame_rate: 100`，与 MPC 同频 |
| 神经网络推理 + 精修 | `< 6.0 ms` | `p95 < 7.0 ms` | 留足 `3–4 ms` 余量，防止单帧超 `10 ms` 掉帧 |
| EKF 跟踪 (`track`) | `< 1.0 ms` | `p95 < 1.5 ms` | UVL + Diff 正常约 `0.4–0.8 ms` |
| 100 Hz MPC 规划 (`plan_dt`) | `10.00 ms` | **`sd < 0.2 ms`** | 独立线程 + 核隔离 + 混合定时（`sleep_until - 500μs` + 短自旋） |

### 1.2 平台推理与核隔离要点

- **NVIDIA Orin NX（如三号步兵）**：
  - 必须编译并启用 TensorRT FP16 + CUDA 预处理 + CUDA Graph（配置 `trt_engine_path`），**引擎与本机 GPU 绑定，需在板端现场用 `trtexec` 生成**。
  - `device` 保持 `CPU`（仅作回退；Orin 没有 Intel GPU）。
- **Intel NUC / x86_64**：
  - 有独显时优先走 TensorRT；纯 CPU 走 OpenVINO 时开启 `PCORE_ONLY` 并限制 `ov_num_threads: 8`（实测 8 线程延迟最低，满核反而抢锁）。
  - 采集线程与推理线程分离，只保留最新一帧不积压队列；100 Hz 控制线程独占 1 个核。
- **验证方法**：
  - 仿真用 `./scripts/run_sim.sh --profile`（或 `./build/sim_standard --profile configs/sim.yaml`）观察 `vis_dt / detect / track / plan_dt`；
  - 实机用 `./scripts/run_standard.sh` 启动并观察 `standard_mpc` 日志中的视觉 FPS、规划 Hz 与推理耗时，确保连续 60 s 无掉帧毛刺。

---

## 2. 第二步：硬件、坐标系、时戳与标定排雷

在开始辨识和调参前，逐项核对以下 5 件事（均为三号步兵实机踩过的真坑）：

1. **串口 / CAN 通信与符号约定**
   - 确认下位机回传频率稳定（如 `/dev/ttyACM0` 1000 Hz 零误码）。
   - 核对 Pitch 方向与限位：视觉内部右手系为**低头为正、抬头为负**；若电控协议为**抬头为正**，在 YAML 设 `pitch_up_positive: true` 并核对 `pitch_min_deg / pitch_max_deg`（如 `-20° ~ +30°`）。
   - 核对 Yaw 方向：左正右负。
2. **IMU 安装角与零漂（KI-004 根因）**
   - 静止平放云台，检查 `gimbal_ypr` 的 roll/pitch/yaw：若出现 `±90°` 滚转，说明 IMU 相对云台存在安装旋转，需配置 `R_gimbal2imu` 修正。
   - 若下位机 IMU 存在明显零漂或与编码器脱节，可启用 `use_euler_q: true` 直接由编码器 yaw/pitch 构造四元数。
3. **相机-云台时间对齐（`camera_latency_ms`）**
   - 实机相机曝光 + 传输存在固有延迟（三号步兵实测 `camera_latency_ms: 8.0`）。
   - 若 `camera_latency_ms` 偏小（如 `0 ms`），转动云台时视觉观测与云台姿态错拍，极易引发**云台-EKF 耦合发散**。
4. **相机内参标定**
   - 用无损 PNG 拍 6+ 姿态 ChArUco 标定（`CALIB_FIX_K3`，重投影 `RMSE < 0.7 px`），更新 `camera_matrix` / `distort_coeffs`。
   - 核对 `R_camera2gimbal`、`t_camera2gimbal`（特别是相机前向偏移，近距 1.5 m 时 6 cm 偏移就是 4% 距离误差）与 `muzzle_z_offset`。
5. **观测模式选择（`use_uvl_ekf`）**
   - 标定准确、安装角无误时，开启 `use_uvl_ekf: true`（UVL + Diff 图像重投影）和 `use_focus_roi: true`。
   - 若实机现场出现频繁 EKF 发散，先临时切回 `use_uvl_ekf: false`（传统 YPDA）验证通信与云台链路，再回头排查标定与 `R_gimbal2imu` / `camera_latency_ms`。
6. **曝光、灯条与机械安装角自检**
   - 优先压低 `exposure_ms`（建议 `2.0–4.0 ms`）再补 `gain`，防止小陀螺高速旋转时运动模糊拉长灯条、导致 PnP/UVL 距离估计漂移。
   - 静止对准水平摆放的目标，若 EKF 收敛后的装甲环 `roll / pitch` 长期稳定偏离 `0°` 超过 `3°–5°`，说明相机机械安装存在倾角，需修正 `R_camera2gimbal`。
7. **静止状态与 WebViewer 曲线核对**
   - **实机（`standard_mpc --web`）**：网页绿色“炮口角”直接来自下位机串口 `gs.yaw`。若云台机械静止但曲线仍在漂移或跳动，说明下位机回传的 `yaw` 存在 IMU 零漂、编码器/陀螺仪源混用或串口解包异常，需先排查下位机与 `use_euler_q`。
   - **仿真（`sim_standard`）**：当仿真开启自瞄跟随（按 `F5`，`sim_following=true`）时，网页绿色“炮口角”直接显示 `daedalus` 共享内存回传的真实炮口姿态，并同步统计**重建命中 (`recon_hit`)** 与 **仿真实际 (`sim_hit`)** 双命中率（可点“清除弹丸统计”重置）；未开 `F5` 且 `sim_gimbal_model.enabled: true` 时回退显示 `SimGimbalModel` 500 Hz 虚拟云台预览（青线同步对齐 `target_yaw` 消除相位差）。

### 2.1 Sim2Real 参数迁移禁忌（切勿直接照抄 `configs/sim.yaml`）

| 参数 | 仿真典型值 (`sim.yaml`) | 实机注意事项 (`standard3.yaml`) |
|---|---|---|
| `trt_engine_path` | 本机独显生成的 `.engine` | **严禁跨机拷贝**：必须在 Orin NX 板端现场用 `trtexec` 重新生成 |
| `enemy_color` | `"blue"` | 按实际比赛/测试己方阵营修改 |
| `threshold` | `60`（仿真纯色发光） | 实机保持 `~150`，过低会引入场地反光误检 |
| `air_k` | `0.0`（仿真默认关阻力） | 实机 17 mm 弹丸需填 `~0.02043` |
| `high/low_speed_delay_time` | `0.045 s` / `0.045 s` | 实机需按 1000 Hz 串口闭环互相关实测重扫（常见 `0.020–0.060 s`） |
| `R_camera2gimbal` / `t_camera2gimbal` / `muzzle_z_offset` | 仿真 URDF + `pose_correction` | 必须替换为实车手眼标定/实车结构尺寸，实机不读取 `pose_correction` |

---

## 3. 第三步：端到端云台系统辨识（SysID）

目的：定量测出当前电控参数下的**纯延迟 `tau_d`、带宽 `wn`、阻尼比 `zeta`、有效峰值加速度**，避免盲猜。

### 3.1 实机一键辨识

```bash
# 1. 确保已编译 real_gimbal_sysid
cmake --build build --target real_gimbal_sysid -j$(nproc)

# 2. 安全前提：关闭摩擦轮、清空云台四周障碍物、随时准备拨出 AUTO_AIM 或 Ctrl+C 急停
# 3. 按当前机器人机械限位执行（以三号步兵为例）
./tools/run_real_sysid.sh configs/standard3.yaml \
  --bus gimbal \
  --yaw-min -30 --yaw-max 30 \
  --pitch-min -15 --pitch-max 15 \
  --yaw-amp 12 --pitch-amp 6 \
  --rate 100
```

> 仿真端对应工具：`./build/sim_gimbal_sysid -o=/tmp/opencode/gimbal_sysid.csv configs/sim.yaml` + `python3 tools/fit_gimbal_tf.py /tmp/opencode/gimbal_sysid.csv`。

### 3.2 如何解读辨识结果

| 指标 | 健康参考值 | 异常信号与处理 |
|---|---|---|
| 拟合 `R²` | `> 0.94` | `< 0.90` 说明存在限位碰撞、机械卡滞、掉帧或符号接反，需先排查再调参 |
| `zeta`（阻尼比） | `0.65 ~ 1.0` | `< 0.55` 说明电控位置环欠阻尼，切板容易振铃，应让电控加阻尼或视觉禁用逆模型前馈 |
| `tau_d`（纯延迟） | 实机 `15–30 ms`；仿真 `40–60 ms` | 过大说明通信/采集链路积压缓冲 |
| 有效峰值加速度 | 供 `max_yaw_acc / max_pitch_acc` 取 `70%–85%` 安全裕度 | 不要超过电机额定力矩/惯量上限，防止电流饱和 |

### 3.3 实测基准对照（参考）

| 平台 | Yaw `(tau_d, wn, zeta, R²)` | Pitch `(tau_d, wn, zeta, R²)` |
|---|---|---|
| 三号步兵实机（2026-11-21） | `25 ms, 35.54 rad/s, 0.517, 0.948` | `25 ms, 47.36 rad/s, 0.909, 0.991` |
| 仿真 100 Hz + 500 Hz LQI 模型（2026-11-23） | `40 ms, 29.25 rad/s, 0.632, 0.986` | `60 ms, 62.10 rad/s, 1.121, 0.984` |

---

## 4. 第四步：MPC 一般跟随调参（核心结论）

### 4.1 控制架构选择（最重要防坑点）

实机与仿真均已反复验证：**绝对不要同时开启“下位机速度/加速度前馈”与“上位机传递函数逆模型 MPC (`gimbal_*_wn > 0`)”**，两者叠加会形成双重前馈，导致小陀螺换板瞬间剧烈过冲（实机曾过冲至 `17°`，仿真摆幅过冲 `38%`、误差翻倍）。

| 方案 | 适用场景 | YAML 关键配置 | 特点 |
|---|---|---|---|
| **方案 A：平滑 MPC + 下位机前馈（首选）** | 电控支持接收 `yaw_vel / yaw_acc` 前馈（如三号步兵最新 `gimbal.c` 与仿真 `SimGimbalModel`） | `gimbal_yaw_wn: 0.0`<br>`gimbal_pitch_wn: 0.0`<br>`zero_vel_ff: false` | **零过冲、跟踪最稳**（仿真 `med=0.66°, rms=1.71°, lag=0ms`；实机 `med≈1.0°`） |
| **方案 B：传递函数 MPC（纯位置环备选）** | 电控**只收位置角**、不接速度/加速度前馈 | `gimbal_yaw_wn: <辨识值>`<br>`gimbal_yaw_zeta: <辨识值>`<br>`Q_yaw_plant: [40.0, 0.08]`<br>`R_yaw_plant: [2.5]`<br>`max_yaw_cmd_offset: 0.14`<br>`zero_vel_ff: true` | 用 MPC 逆模型补位置环滞后，必须配速度阻尼和 `max_yaw_cmd_offset` 限幅防甩头 |

### 4.2 调参三步法（以首选方案 A 为例）

1. **先定 `max_yaw_acc / max_pitch_acc`**
   - 从电机额定力矩/惯量或辨识峰值加速度的 `70%–85%` 起步（如三号步兵 Yaw `120–160 rad/s²`，Pitch `85–100 rad/s²`）。
   - 太小：小陀螺换板跟不上、锯齿幅值衰减；太大：电机饱和或机械抖动。
2. **再扫 `high_speed_delay_time / low_speed_delay_time`（相位零滞后对齐）**
   - 固定目标做旋转（小陀螺）或匀速平移，录 `5–10 s` 闭环日志（含 `ref_yaw` 与真实 `muzzle_yaw`）。
   - 对 `muzzle_yaw` 与 `ref_yaw` 做互相关平移扫描：
     - 若 `muzzle_yaw` 滞后 `ref_yaw`（`lag > 0`），**调大** `delay_time`；
     - 若 `muzzle_yaw` 超前 `ref_yaw`（`lag < 0`），**调小** `delay_time`。
   - 典型区间（100 Hz 稳态）：
     - 实机（1000 Hz 串口 + 开前馈）：`0.020 ~ 0.060 s`；
     - 仿真（含渲染、100 Hz 采集链与 `daedalus` 240 Hz `kp=45` PID）：`0.045 s`。
   - 小步精调（步长 `0.005 s`），取 **`lag = 0 ms` 且 `yaw_rms` 最小** 的点。
3. **核对一般跟随验收标准**
   - 互相关相位滞后：`lag = 0 ms`（容忍 `±10 ms`）；
   - 小陀螺跟踪误差：Yaw 中位误差 `< 1.0°`，Yaw RMS `< 1.5°`（仿真 `Q_yaw=[2e7,0], max_yaw_acc=180` 实测 `1.33°`），Pitch RMS `< 0.3°`；
   - 无过冲：实测云台峰峰值 `pp(muzzle) ≤ pp(ref)`，换板无二次振铃。

---

## 5. 第五步：开火窗口、跑打融合与实弹弹道闭环

跟随调稳后再动开火与弹道参数，避免把跟随误差当成弹道零偏去补：

1. **弹道物理参数**：
   - `bullet_speed`：与裁判系统/实测初速对齐；
   - `air_k`：实车 17 mm 弹丸约 `0.02043`（仿真未开空气阻力时为 `0.0`）；
   - `muzzle_z_offset`：炮口相对云台旋转中心的竖直偏移（m），会与机身 `roll` 共同决定炮口在世界系的 3D 偏移。
2. **3D 弹道求交开火窗口（`check_ballistic_hit`）**：
   - `Planner` 在 `now + fly_time` 处将真实出膛 3D 弹道（叠加我方底盘速度与空气阻力）与目标所有正面装甲板直接求交，并自动扣除机身圆柱遮挡与 17 mm 弹丸半径（`8.5 mm`）：
   - `shoot_offset`：拨弹到出膛的固定发射延迟（单位 `10 ms` 步长，默认 `2` 即 `20 ms`）；
   - `fire_margin`：受击装甲板有效几何张角系数（默认 `0.70`，实车首测建议 `0.70–0.75` 保底命中率；跟随稳态后可放宽至 `0.85` 拉高开火占空比，仿真小陀螺实测 `fire_duty=73.7%+`、`sim_hit=96.2%`）；
   - `max_fire_angle`：3D 弹道相对装甲板法向的最大允许入射角（推荐 `55°`；大于 `60°` 时单目测距深度误差会被 `tan(θ)` 明显放大，容易导致边缘擦边脱靶）。
3. **实弹零偏微调**：
   - 通过 `WebViewer` / `RuntimeParams` 在线微调 `planner.yaw_offset` / `planner.pitch_offset`（`Planner` 会按机身 `roll` 自动将零偏旋转到世界系），或启用 `AimCorrector` 视觉弹道闭环自动估计偏置，确认稳定后回写 YAML。
4. **跑打（底盘速度 + `roll` 倾角）实机部署清单**：
   - **`roll` 倾角融合（已就绪）**：`io::Gimbal` 已实时从 IMU 四元数提取 `state.roll` 并喂入 `standard_mpc` 的 `SelfMotion`；**上坡/侧倾部署前**务必先确认静止平放时 `roll ≈ 0°`（排查 `R_gimbal2imu`），再将车倾斜约 `10°–15°` 核对 `roll` 符号是否符合右手系（绕枪口 `+X` 顺时针右倾为正）。
   - **底盘速度 `self_vel` 接入（待电控协议扩展）**：当前 `io::GimbalState` 已预留世界系速度 `vx / vy / vz`（默认 `0`，不破坏现有 43 B 串口帧）；后续与电控按 [`docs/gimbal_serial_protocol.md`](gimbal_serial_protocol.md) §6 接入底盘速度后，需按以下 3 步实车验收：
     1. **符号核对**：只向前开底盘，`self_vel` 应与当前云台朝向一致；横移（向左/向右）时观察 `self_vel` 与世界系方向是否一致，防止底盘系/云台系/世界系混用。
     2. **静止目标横移跑打验证**：对静止装甲板以 `1.5–2.5 m/s` 横移，`WebViewer` 观察目标 EKF 速度 `vx/vy` 是否保持接近 `0`（确认 `Target::predict(t, self_vel)` 已解耦自身运动）、且 `cmd_yaw` 自动往运动反方向给出弹道提前量。
     3. **联合小陀螺实弹复测**：在 `fire_margin: 0.75` 下实弹测试横移打小陀螺，确认命中率稳定 `> 90%` 后再逐步放宽 `fire_margin` 至 `0.85`。
   - **Orin NX 上电部署速查**（三号步兵 `standard3-orin` 分支，详见 [`scripts/README.md`](../scripts/README.md)）：
     1. 本机运行 `./scripts/deploy_orin.sh --build` 同步代码至 `yq11@192.168.55.1:~/yunqian_vision` 并远程编译，再在板端跑通 `./scripts/run_tests.sh`（4 项全绿）；
     2. 确认 `configs/standard3.yaml` 中 `air_k: 0.02043`、`fire_margin: 0.70~0.75`、`max_fire_angle: 55`、`muzzle_z_offset` 为实车实测值；
     3. 先用 `./scripts/run_standard.sh`（默认 `--no-fire --web`）核对 100 FPS / 100 Hz 帧率、`roll` 符号与跟踪曲线，再切 `./scripts/run_standard.sh --live` 进行实弹验证与 `AimCorrector` 零偏校准。

---

## 6. 常见问题快速排错表

| 现场症状 | 最可能原因 | 优先排查与处理 |
|---|---|---|
| 云台换板剧烈过冲/来回甩头 | 下位机前馈 + 传递函数 MPC 双重叠加，或 `zeta` 过低 | 设 `gimbal_yaw_wn: 0.0`（切回平滑 MPC），或降低 `Q_yaw_plant`、提高 `R_yaw_plant` |
| 静止跟得准，小陀螺永远慢半拍 | `delay_time` 偏小或 `max_yaw_acc` 过保守 | 录 CSV 看互相关 `lag`，按 `lag` 增加 `delay_time` 并适度提高 `max_yaw_acc` |
| 云台一动 EKF 就发散或 `w` 乱跳 | `camera_latency_ms` 没对齐、`R_gimbal2imu` 错、或标定不准 | 先设 `camera_latency_ms: 8.0`、检查 IMU roll/pitch；临时关 `use_uvl_ekf` 定位是否标定问题 |
| 视觉帧率卡在 50–70 FPS 或偶发 20 ms 尖峰 | CPU 推理超 8 ms、采集与推理串行、或线程抢核 | 切 TensorRT + CUDA Graph；分离采集/推理/控制线程并做核隔离；检查 CPU 是否锁在 `powersave` |
| 跟踪误差很小（`< 1°`）但实弹稳定偏一边 | 机械零偏、`muzzle_z_offset` 或 `air_k` / 弹速偏差 | 保持跟随参数不动，只调 `yaw_offset / pitch_offset` 与 `bullet_speed / air_k` |
| 云台不动，但 WebViewer 绿色“炮口角”仍在变化 | 仿真：未按 `F5` 时回退显示 `SimGimbalModel` 虚拟云台；实机：下位机 IMU 零漂 | 仿真按 `F5` 开启跟随即可切到真实炮口反馈；实机排查下位机 `yaw` 数据源零漂与 `use_euler_q` |
| 实机偶发 1–4 帧短脉冲丢检 | 曝光过长运动模糊、`min_confidence` 偏高、或 ROI/传统精修过滤 | 降低 `exposure_ms`、对比关开 `use_focus_roi` 与 `use_traditional`，并适当下调 `min_confidence` |
