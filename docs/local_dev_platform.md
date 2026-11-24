# 本机开发平台（无 NUC）

> 状态：结论已确认，**尚未改动任何代码**。最后整理：2026-11-15。
> 关联：[AGENTS.md](../AGENTS.md) §2.3（算力平台与机器人分支）、[ROADMAP.md](../ROADMAP.md) §2 / §3（P1–P4）。
> 本文回答一个问题：**手上没有 NUC 时，如何保证仿真重建逻辑能被实现和验证。**

## 1. 结论

1. **NUC 不是本地开发的必要条件。** NUC 只提供「x86_64 + Intel iGPU + 四个外设」，个人电脑占前两项，缺的是外设。
2. **「NX 的代码不能本地跑」只成立于部署包层**，不成立于业务逻辑层。业务逻辑与推理后端无关，PC 上编得出、跑得动。
3. **仿真重建逻辑本身不跑在 NX 上。** 它在 PC 上跑，消费的是遥测；NX 只是遥测的生产者。
4. 因此要保证的不是「在 NX 上验证重建逻辑」，而是「**重建逻辑只跟数据打交道，数据生产者不跟推理后端打交道**」。
5. 没有 NUC 反而有利：它迫使先做回放/替身，而这四件替身是 P1/P3/P4 的必需品；有 NUC 会诱导走「插上相机就开跑」的捷径，那条路在 NX 上不存在。

## 2. 平台分层：为什么本地能跑

| 层 | 内容 | 本地 PC（x86_64） | 原因 |
|---|---|---|---|
| 编译产物 | aarch64 ELF | ❌ | 架构不同 |
| 推理后端 | TensorRT engine、CUDA 预处理 | ❌ | engine 绑定 GPU 架构（`sm_87`）+ TensorRT 版本 |
| 业务逻辑 | `detector` 后处理 / `solver` / `tracker` / `target`(EKF) / `aimer` / `shooter` / 决策 / 遥测 / Bevy 重建 | ✅ **同一份源码** | 只依赖 Eigen / OpenCV / 本仓库数学工具 |

只有第三行是「逻辑」，前两行是「可替换的后端与产物」。把平台相关代码锁在 `YOLOBase` 之后，NX 的不可本地运行就退化成「最后一个 replaceable 后端」。

## 3. 本机体检（2026-11-15 实测）

| 项 | 状态 | 说明 |
|---|---|---|
| OS | Ubuntu 22.04.5 (jammy) | |
| 架构 | ✅ `x86_64` | 与 NUC 同类，NUC 代码无需改一行即可编译 |
| CPU | ✅ i9-14900HX / 32 线程 | 强于 NUC 的 i7-1260P |
| Intel iGPU | ✅ `00:02.0` | NUC 跑 OpenVINO 就靠这个 |
| NVIDIA 独显 | ⚠️ 有卡，驱动未加载 | 与本仓库路径无关，可忽略 |
| 编译器 / CMake | ✅ g++ 11.4 / cmake 3.22.1 | 满足 `cmake_minimum_required(3.16.3)` |
| OpenVINO | ❌ 未安装 | `/opt/intel/openvino*` 不存在 |
| 开发依赖 | ❌ 七个全缺 | OpenCV / Eigen / fmt / spdlog / yaml-cpp / nlohmann_json / Ceres |
| `can0` / `/dev/gimbal` | ❌ 无 | 物理外设，需替身 |
| `build/` | ❌ 无 | 尚未编译过 |
| 模型资产 | ✅ 齐全（88M） | `assets/yolo11.xml`、`yolov5`、`yolo11_buff_int8`、`best2-sim.onnx` |

**缺的全是「可安装的软件」+「四个外设」，没有一项是架构问题。**

## 4. NUC 独有依赖 → 替身映射

| NUC 独有 | 代码位置 | 替身 | 对业务逻辑的影响 | 状态 |
|---|---|---|---|---|
| 海康 / 迈德威视相机 | `io/camera.hpp` 的 `CameraBase` | `VideoCamera`（`cv::VideoCapture` / 图像序列） | 无 | 待实现 |
| CAN 下位机 | `io/cboard.*` | `FakeCboard`（打印 / 写 JSONL） | 无 | 待实现 |
| 云台串口 | `io/gimbal/*` | `FakeGimbal` | 无 | 待实现 |
| IMU | `io/dm_imu/*` | 固定四元数 / 从仿真 tf 读 | 无 | 待实现 |
| Intel iGPU + OpenVINO GPU 插件 | `configs/*.yaml` 的 `device` | `device: CPU`（或装 OpenVINO 用 iGPU） | 仅延迟，**不影响结果** | 配置一行 |
| ROS2 | 顶层 `CMakeLists.txt` 条件编译 | 本机暂不编 | 无（P1 需要时再装） | 无需处理 |

现有两个抽象正好是这两条缝，不必新造：

- `io::CameraBase::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)`
  契约是「带时间戳的出帧器」，视频文件天然满足。
- `auto_aim::YOLOBase::detect / postprocess`，工厂在 `tasks/auto_aim/yolo.cpp` 按 `yolo_name` 选择。

## 5. 本机最小可跑闭环

目标只编 `tools + io + auto_aim + detector_video_test`，**不装 ROS2、不装 Ceres、不编 `auto_buff`**。

### 第 0 步：装依赖（需 sudo，沙箱内无法代跑）

```bash
sudo apt update && sudo apt install -y \
  libopencv-dev libeigen3-dev libfmt-dev libspdlog-dev \
  libyaml-cpp-dev nlohmann-json3-dev
```

Ceres 先跳过，它只服务 `auto_buff`。

### 第 1 步：OpenVINO（注意硬编码陷阱）

`CMakeLists.txt` 与 `tasks/auto_aim/CMakeLists.txt` 都是硬编码：

```cmake
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake/")
```

`set()` 会**覆盖**命令行 `-DOpenVINO_DIR=...`，所以 pip 装的 OpenVINO 指不过去。两条路：

- **方案 A（推荐，代码零改动）**：下载 OpenVINO 2024.6.0 x86_64 archive，解压到 `/opt/intel/openvino_2024.6.0`，路径刚好对上。
- **方案 B（无 sudo）**：`python3 -m venv` + `pip install openvino==2024.6.0`，再把两处 `set()` 改成
  `if(NOT OpenVINO_DIR) set(...) endif()`。仅构建脚本改动，不碰语义，并让 apt 版与 pip 版同时可用。

### 第 2 步：切设备

`configs/standard3.yaml` 的 `device: GPU` → `CPU`（走方案 B 或未加载 Intel GPU 驱动时）。

### 第 3 步：编译与验收

```bash
cmake -B build
make -C build detector_video_test auto_aim_test -j$(nproc)
```

验收：拿 `assets/` 里现成模型对一段视频跑出检测结果。跑通后即拥有**第一个不依赖任何硬件的可信闭环**。

## 6. 平台缝的硬约束与等价性验证

四条硬约束（要靠可执行检查固定，不靠口头约定）：

1. **依赖方向**：仿真 / 重建 / 遥测代码只许 include `yolo.hpp` 与 `armor.hpp`，不许 include `yolo11.hpp`、`trt_engine.hpp`、`openvino/*`。
2. **开关位置**：`#ifdef SP_WITH_TENSORRT` / `SP_WITH_CUDA_PREPROCESS` 只允许出现在后端实现与 CMake 中。一旦出现在 `solver.cpp` / `target.cpp` / `aimer.cpp` / `tracker.cpp` / `detector.cpp` 的后处理里，「PC 验证过的行为」就不再等于「板上行为」。
3. **输入契约**：业务代码只认 `(bgr_img, camera_info, gimbal_quaternion, timestamp, bullet_speed)`，不认来源。PC 用回放喂，板端用相机 / ROS 喂，代码路径相同。
4. **输出契约**：遥测 schema 先行定义（见 [ROADMAP.md](../ROADMAP.md) §4），重建端只吃 schema，schema 不含任何后端字段。

唯一真正的风险点：**ONNX → OpenVINO IR 与 ONNX → TensorRT engine 是两条编译链，预处理与后处理语义必须共享同一份 C++。**

- `YOLOBase::postprocess(double scale, ...)` 的 `scale` 是分水岭。若 `cuda_preprocess` 自己算 letterbox 的 `scale` / padding，而插值与 OV 路径不一致，反变换出的检测框会系统性偏移，且**在 PC 上永远看不到**。
- CUDA 预处理只能是**位精确的加速**，不能有自己的语义：padding 值、插值方式、BGR 顺序、归一化参数必须来自同一份配置或常量。
- 验收方式：固定一批图，OV(x86) 与 TRT(`sm_87`) 的输出框在 1px / 0.01 置信度内一致才算过。这是**唯一必须上板才能验证**的一步，但它验证的是后端等价性，不是重建逻辑。

## 7. 图像来源与模型注意事项

- `data/` 目前为空（只有 `roleplay_data`）。P1 的图像来源三个，都不需要 NUC：bevy 仿真出帧、手机 / 相机实拍、静态图。
- `assets/best2-sim.onnx` 名字像**仿真专用模型**，来源待确认。若它只在仿真图上训过，则仿真闭环里用它验证的是「逻辑」；换成实机模型验证的是「识别」。P5 之前不要混用。

## 8. 没有 NUC 也保证不了、但也不需要保证的三件事

1. 相机驱动（aarch64 的 MVS）—— 与重建逻辑无关。
2. 真实延迟 / 帧率 / 丢帧 —— P5 量化，靠板上实测。
3. TensorRT FP16/INT8 量化误差 —— 靠 §6 的等价性脚本兜住。

## 9. 待办（按顺序）

| # | 事项 | 验收标准 | 状态 |
|---|---|---|---|
| 1 | 装依赖 + 编通 `auto_aim` / `io` / `tools` | `detector_video_test` 能对视频出检测结果 | 待办 |
| 2 | 写替身 `VideoCamera` + `FakeCboard` | `standard` 在无相机 / 无 CAN 下完整跑起来 | 待办 |
| 3 | 落遥测 schema（C++ 结构体 + JSON 示例） | PC 侧能按 schema 解析并重建 | 待办 |

顺序理由：1 立刻暴露依赖问题；2 与 3 的价值要靠 1 的编译结果验证。

## 10. 未决

1. 方案 A / B 二选一（影响是否改动两处 `CMakeLists.txt`）。
2. OpenVINO 版本是否锁定 2024.6.0（NUC 与板上一致）还是允许更新版本。
3. NUC 路径是否长期保留为第二后端（当前无 NUC 硬件，但可能有使用 NUC 的机器人）。
4. `best2-sim.onnx` 的训练数据来源与适用范围。
