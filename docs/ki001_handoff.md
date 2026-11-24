# 交接：KI-001 斜面英雄跟踪

仓库：`/home/scj/桌面/rm2027/yunqian_vision`。只改这个仓库。仿真仓库 `/home/scj/桌面/rm2027/opensource/bevy_robomaster_simulator` 只读。不要 commit / push，不要加依赖。

先读 `AGENTS.md` 和 `docs/known_issues.md` 的 KI-001。本文是给下一个 AI 的任务书，不是用户需求的替代。

## 当前进度（2026-11-17）

yaw 双解已改在进 EKF 之前处理，见 `Solver::resolve_yaw()`。状态、观测、雅可比都没变。

已复测：

- 斜面英雄静止不再失锁：`w≈0`，`r≈0.266`。
- 斜面英雄匀速转锁定后能持续：`w` 约 −7.4 rad/s，两组半径都在界内。带倾角重投影后，收敛段中位数 9.6 px。
- 平地步兵没有回归：换板顺序连续（`0,3,2,1`），距离保持在 1.25 m 附近。
- 两个半径是对的，不要扩成四块各自半径。S120/S121/S124 只要求对角两组半径、横向偏移 ≤ 50 mm、下边缘高度差 ≤ 100 mm。
- 前哨站是另一个问题，见 `docs/known_issues.md` 的 KI-002。不要在修英雄时顺手改前哨站模型。

## 要修的问题

斜面英雄转完停下，当前板贴住，另外三块预测板仍偏出车身。画面里同时出现第二块板时，`w` 从 0 跳到约 −7.5 rad/s，板号跟着跳，下一秒回到 0。侧面板的 yaw 双解被写进相邻板，车心和另一组半径被推偏；当前板下一帧被位置观测拉回。

`resolve_yaw()` 已加 0.5 rad 门限：预测离所有低代价极小值更远时，用重投影自己的最小值。这个门限修掉了静止时 `w` 积到 40 rad/s 的问题，匀速转还没复测。

平地四块小装甲的步兵已经复测过，换板和匀速旋转都对。**不要为了修英雄把步兵改坏。**

## 已经做完、不要回退的部分

`auto_aim::Target` 状态是 13 维：

```
0 x, 1 vx, 2 y, 3 vy, 4 z, 5 vz, 6 a, 7 w, 8 r, 9 l, 10 h, 11 roll, 12 pitch
```

`x[11]/x[12]` 是装甲环相对水平面的 roll/pitch，单位 rad。`h_armor_xyz()` 先在环平面内放半径和高度差，再乘 `Rx(roll) * Ry(pitch)`。roll = pitch = 0 时必须退回原来的水平环。

观测是 6 维：

```
z = [yaw, pitch, distance, armor_yaw, armor_pitch, armor_roll]
```

雅可比对同一个 `h()` 做中心差分，不要再手写一份和 `h()` 不一致的雅可比。

1 号英雄（`ArmorName::one`）是四块大装甲，不是两板平衡。只有大装甲且编号为 three/four/five 才是两板。前哨站和基地是三块板，不用倾角。

英雄初始半径用 0.25 m 这个公共先验。**不要**把仿真模型里量到的 0.21 / 0.26 / 0.28 写进代码。四块板各自的半径和高度必须在线估计。

`armor.lightbar_yaw` 已在 `solver.cpp` 的 `solve()` 里、`optimize_yaw()` 之前赋值。它是灯条中点连线的世界系 yaw，没有正反。`ypr_in_world[0]` 在 yaw 优化之后会被覆盖，匹配不要用优化后的那个值。

NIS 阈值在 `tools/extended_kalman_filter.cpp`：4 维观测保持 0.711，6 维用 12.592（χ²(6) 的 95% 上分位）。这是上分位，不是下分位。不要改回 0.711，否则正常更新会被判失败。

`armor_xyza_list()` 仍返回 4 维 `[x, y, z, yaw]`。viewer 和 MPC 依赖这个接口，不要改成 6 维。

## 已经试过、会破坏步兵，不要再做

1. 当前板预测位置还在 0.35 m 内就锁死、不换板。平地步兵换板时旧板预测仍在附近，新板观测被写进旧板，整车被拉到十几米外。
2. 把板面 yaw 从 EKF 更新里拿掉，观测 yaw 填成预测值，只靠位置推旋转。平地步兵的相位不再连续。板面 yaw 必须继续更新。
3. 一帧里按灯条 yaw 只选一块板更新。灯条连线没有正反，对面板和当前板的夹角都接近 0，会选错。现在 `tracker.cpp` 的 `update_target()` 对同名同类型的检测逐个 `solve()` + `update()`，这是步兵能换板的路径，不要改成单板更新，除非新逻辑在步兵上先验证通过。
4. 把英雄改成两板。用户明确拒绝：数字识别为 1 号就是四块大装甲。
5. 收紧 `r/l/h` 的 `P0`，或把初始半径改成 0.24。角速度会冲到 −12 rad/s。
6. 用画面最左侧的板做更新。斜面上侧面板也可能在最左侧。

## 当前失败的直接原因

斜面上 `solvePnP` 的板面 yaw 有两个极小值。车已经停住、仿真真值 `vyaw ≈ 0` 时，检测 yaw 仍会在约 −0.8 rad 和 +0.13 rad 之间跳。`Target::update()` 用这个 yaw 匹配板号，又把它写进 EKF，于是：

- `w` 被积分到数 rad/s，静止看起来在转，匀速转看起来忽快忽慢；
- 板号在 0/1/2/3 之间跳，观测被记到相邻板；
- `r` 掉到 0.05 以下，`Target::diverged()` 重置；
- 另外三块板用 `a + i * π/2` 推出，相位一错，三块全错。

灯条连线 yaw 没有这个 180° 翻转，但它是无方向的，不能单独区分对面板。位置可以区分，但“位置近就锁死”会破坏步兵换板。需要一个同时满足两边的判据，而不是二选一。

## 硬约束

- 不读共享内存里的 GroundTruth 做补偿。GT 只允许在 `tests/sim_pose_probe.cpp` 里做诊断打印。
- 不改 `configs/sim.yaml` 里已经验证过的 `pose_correction`、`R_camera2gimbal`、`t_camera2gimbal`。
- 不改仿真进程。当前仿真是 bwrap 里的 `daedalus`，共享内存在本仓库 `.sim/tmp`，绑到 `/tmp`。
- 不把某台车的四块板半径/高度写死。制作规范的公共先验可以用：相邻板 90°、板面相对竖直约 15°、单块垂直偏移 ≤ 50 mm、下边缘高度差 ≤ 100 mm。规范 PDF 在 `/home/scj/桌面/rm2027/rule/UTF-8__RoboMaster 2027 机甲大师高校系列赛机器人制作规范手册V1.0.0（20261015）.pdf`。
- P2（uv 观测）和 P3（四板各自 `r_i/z_i`）先不要做。P1 的观测表示还没把双解处理干净，同时改模型会分不清是哪边坏了。

## 怎么编译和跑

系统里没有 OpenCV / OpenVINO，用仓库内的依赖：

```bash
cmake -B build \
  -DOpenCV_DIR=/home/scj/桌面/rm2027/yunqian_vision/.deps/install/lib/cmake/opencv4 \
  -DOpenVINO_DIR=/home/scj/桌面/rm2027/yunqian_vision/.deps/l_openvino_toolkit_ubuntu22_2024.6.0.17404.4c0f47d2335_x86_64/runtime/cmake
cmake --build build --target sim_pose_probe sim_standard -j$(nproc)
cp -a build/sim_pose_probe build/sim/sim_pose_probe
cp -a build/sim_standard build/sim/sim_standard
```

运行必须进同一个 bwrap，否则读不到仿真共享内存：

```bash
export LD_LIBRARY_PATH=/home/scj/桌面/rm2027/yunqian_vision/.deps/install/lib:/home/scj/桌面/rm2027/yunqian_vision/.deps/l_openvino_toolkit_ubuntu22_2024.6.0.17404.4c0f47d2335_x86_64/runtime/lib/intel64:/home/scj/桌面/rm2027/yunqian_vision/.deps/l_openvino_toolkit_ubuntu22_2024.6.0.17404.4c0f47d2335_x86_64/runtime/3rdparty/tbb/lib
bwrap --dev-bind / / --bind /home/scj/桌面/rm2027/yunqian_vision/.sim/tmp /tmp \
  --chdir /home/scj/桌面/rm2027/yunqian_vision -- \
  build/sim/sim_pose_probe configs/sim.yaml -frames=60 -target=one
```

`sim_pose_probe` 和 `sim_standard` 抢同一块共享内存。探针之前先停掉 `build/sim/sim_standard`，不要 `pkill -f sim_standard`：当前 shell 的命令行里也有这个字符串，会把自己杀掉。用精确 PID。

网页视觉：

```bash
bwrap --dev-bind / / --bind /home/scj/桌面/rm2027/yunqian_vision/.sim/tmp /tmp \
  --chdir /home/scj/桌面/rm2027/yunqian_vision -- \
  env LD_LIBRARY_PATH=$LD_LIBRARY_PATH build/sim/sim_standard configs/sim.yaml
```

地址是 http://127.0.0.1:8090。不要另起服务，不要动 `daedalus`。

## 验收

两边都要过，不能只过一边。

斜面英雄，静止和匀速转各看一段：

- `control=true` 连续，不出现 `Target diverged!`；
- `w` 在静止时回到 0 附近，匀速转时接近常数，不在数 rad/s 之间跳；
- `last_id` 只在真实换板时变，不变成每帧跳；
- 四块预测板重投影 < 20 px，相位连续；
- `r` 保持正，且在制作规范允许的范围内。

平地步兵：

- 换板仍然发生，相位连续，不被拉到十几米外；
- 改完步兵路径后，用步兵目标再跑一轮探针或网页确认，再动英雄。

任何一次修改后，如果步兵回归失败，先回退这次修改，不要在坏掉的步兵上继续叠英雄的补丁。
